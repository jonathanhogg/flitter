"""
Flitter window management
"""

# pylama:ignore=C0413,E402,W0703,R0914,R0902,R0912,R0201,R1702,C901,W0223,W0231,R0915

import array
import logging
import sys

import av
import skia
import moderngl
import pyglet
if sys.platform == 'darwin':
    pyglet.options['shadow_window'] = False
import pyglet.canvas
import pyglet.window
import pyglet.gl

from . import canvas


Log = logging.getLogger(__name__)


def value_split(value, n, m):
    if m == 1:
        return value if n == 1 else list(value)
    elif n == 1:
        return tuple(value)
    return [tuple(value[i*m:(i+1)*m]) for i in range(n)]


class SceneNode:
    def __init__(self, glctx):
        self.glctx = glctx
        self.children = []
        self.width = None
        self.height = None

    @property
    def texture(self):
        raise NotImplementedError()

    @property
    def child_textures(self):
        textures = {}
        i = 0
        for child in self.children:
            if child.texture is not None:
                textures[f'texture{i}'] = child.texture
                i += 1
        return textures

    def destroy(self):
        self.release()
        self.glctx = None
        self.purge()

    def purge(self):
        while self.children:
            self.children.pop().destroy()

    async def update(self, node, **kwargs):
        references = kwargs.setdefault('references', {})
        node_id = node['id'].as_string() if 'id' in node else None
        if node_id:
            references[node_id] = self
        resized = False
        width, height = node.get('size', 2, int, (512, 512))
        if width != self.width or height != self.height:
            self.width = width
            self.height = height
            resized = True
        self.create(node, resized, **kwargs)
        await self.descend(node, **kwargs)
        self.render(node, **kwargs)

    async def descend(self, node, **kwargs):
        count = 0
        for i, child in enumerate(node.children):
            cls = {'reference': Reference, 'shader': Shader, 'canvas': Canvas, 'video': Video}[child.kind]
            if i == len(self.children):
                self.children.append(cls(self.glctx))
            elif type(self.children[i]) != cls:  # noqa
                self.children[i].destroy()
                self.children[i] = cls(self.glctx)
            await self.children[i].update(child, **kwargs)
            count += 1
        while len(self.children) > count:
            self.children.pop().destroy()

    def create(self, node, resized, **kwargs):
        pass

    def render(self, node, **kwargs):
        raise NotImplementedError()

    def release(self):
        raise NotImplementedError()


class Reference(SceneNode):
    def __init__(self, _):
        self._reference = None

    @property
    def texture(self):
        return self._reference.texture if self._reference is not None else None

    async def update(self, node, references=None, **kwargs):
        node_id = node['id'].as_string() if 'id' in node else None
        self._reference = references.get(node_id) if references is not None and node_id else None

    def destroy(self):
        self._reference = None


class ProgramNode(SceneNode):
    GL_VERSION = (4, 1)

    def __init__(self, glctx):
        super().__init__(glctx)
        self._program = None
        self._rectangle = None
        self._vertex_source = None
        self._fragment_source = None
        self._last = None

    def release(self):
        if self._last is not None:
            self._last.release()
            self._last = None
        if self._program is not None:
            self._program.release()
            self._program = None
        if self._rectangle is not None:
            self._rectangle.release()
            self._rectangle = None

    @property
    def framebuffer(self):
        raise NotImplementedError()

    @property
    def size(self):
        return self.framebuffer.size

    def create(self, node, resized, **kwargs):
        super().create(node, resized, **kwargs)
        if resized and self._last is not None:
            self._last.release()
            self._last = None

    def get_vertex_source(self, node):
        if 'vertex' in node:
            return node['vertex'].as_string()
        return """#version 410
in vec2 position;
out vec2 coord;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    coord = (position + 1.0) / 2.0;
}
"""

    def get_fragment_source(self, node):
        if 'fragment' in node:
            return node['fragment'].as_string()
        child_textures = self.child_textures
        samplers = '\n'.join(f"uniform sampler2D {name};" for name in child_textures)
        textures = '\n'.join(f"""    merge = texture({name}, coord);
    color = color * (1.0 - merge.a) + merge;""" for name in child_textures)
        return f"""#version 410
in vec2 coord;
out vec4 color;
vec4 merge;
{samplers}
void main() {{
    color = vec4(0.0, 0.0, 0.0, 0.0);
{textures}
}}
"""

    def compile(self, node):
        vertex_source = self.get_vertex_source(node)
        fragment_source = self.get_fragment_source(node)
        if vertex_source != self._vertex_source or fragment_source != self._fragment_source:
            self._vertex_source = vertex_source
            self._fragment_source = fragment_source
            if self._program is not None:
                self._program.release()
                self._program = None
            if self._rectangle is not None:
                self._rectangle.release()
                self._rectangle = None
            try:
                self._program = self.glctx.program(vertex_shader=self._vertex_source, fragment_shader=self._fragment_source)
                vertices = self.glctx.buffer(array.array('f', [-1, 1, -1, -1, 1, 1, 1, -1]))
                self._rectangle = self.glctx.vertex_array(self._program, [(vertices, '2f', 'position')])
            except Exception:
                Log.exception("Unable to compile shader:\n%s", self._fragment_source)

    def render(self, node, **kwargs):
        self.compile(node)
        if self._rectangle is not None:
            sampler_args = {'repeat_x': False, 'repeat_y': False}
            border = node.get('border', 4, float)
            repeat = node.get('repeat', 2, bool)
            if border is not None:
                sampler_args['border_color'] = tuple(border)
            elif repeat is not None:
                if repeat[0]:
                    del sampler_args['repeat_x']
                if repeat[1]:
                    del sampler_args['repeat_y']
            child_textures = self.child_textures
            self.framebuffer.use()
            samplers = []
            unit = 0
            for name in self._program:
                member = self._program[name]
                if isinstance(member, moderngl.Uniform):
                    if name == 'last':
                        if self._last is None:
                            self._last = self.glctx.texture((self.width, self.height), 4)
                        self.glctx.copy_framebuffer(self._last, self.framebuffer)
                        if sampler_args:
                            sampler = self.glctx.sampler(texture=self._last, **sampler_args)
                            sampler.use(location=unit)
                            samplers.append(sampler)
                        else:
                            self._last.use(location=unit)
                        member.value = unit
                        unit += 1
                    elif name in child_textures:
                        if sampler_args:
                            sampler = self.glctx.sampler(texture=child_textures[name], **sampler_args)
                            sampler.use(location=unit)
                            samplers.append(sampler)
                        else:
                            child_textures[name].use(location=unit)
                        member.value = unit
                        unit += 1
                    elif name == 'size':
                        member.value = self.size
                    elif name in kwargs:
                        member.value = kwargs[name]
                    elif name in node:
                        value = node.get(name, member.array_length * member.dimension, float)
                        if value is not None:
                            member.value = value_split(value, member.array_length, member.dimension)
            self.framebuffer.clear()
            self._rectangle.render(mode=moderngl.TRIANGLE_STRIP)
            for sampler in samplers:
                sampler.clear()
                sampler.release()


class Window(ProgramNode):
    class WindowWrapper(pyglet.window.Window):  # noqa
        """Disable some pyglet functionality that is broken with moderngl"""
        def on_resize(self, width, height):
            pass

        def on_draw(self):
            pass

        def on_close(self):
            pass

    def __init__(self, screen=0, fullscreen=False, vsync=False):
        super().__init__(None)
        self._closed = False
        self.window = None
        self.default_screen = screen
        self.default_fullscreen = fullscreen
        self.default_vsync = vsync

    def release(self):
        if self.window is not None:
            self.glctx.release()
            self.glctx = None
            self.window.close()
            self.window = None
        super().release()

    @property
    def texture(self):
        return None

    @property
    def framebuffer(self):
        return self.glctx.screen

    @property
    def size(self):
        return self.glctx.screen.viewport[2:]

    def create(self, node, resized, **kwargs):
        super().create(node, resized)
        if self.window is None:
            vsync = node.get('vsync', 1, bool, self.default_vsync)
            screen = node.get('screen', 1, int, self.default_screen)
            title = node['title'].as_string() if 'title' in node else "Flitter"
            fullscreen = node.get('fullscreen', 1, bool, self.default_fullscreen)
            screens = pyglet.canvas.get_display().get_screens()
            screen = screens[screen] if screen < len(screens) else screens[0]
            config = pyglet.gl.Config(major_version=self.GL_VERSION[0], minor_version=self.GL_VERSION[1], forward_compatible=True,
                                      depth_size=24, double_buffer=True, sample_buffers=1, samples=0)
            self.window = self.WindowWrapper(width=self.width, height=self.height, resizable=True, caption=title, screen=screen, vsync=vsync, config=config)
            self.window.event(self.on_resize)
            self.window.event(self.on_close)
            self.glctx = moderngl.create_context(require=self.GL_VERSION[0] * 100 + self.GL_VERSION[1] * 10)
            if fullscreen:
                self.window.set_mouse_visible(False)
                if sys.platform == 'darwin':
                    self.window._nswindow.enterFullScreenMode_(self.window._nswindow.screen())  # noqa
                else:
                    self.window.set_fullscreen(True)
        elif resized:
            self.on_resize(self.width, self.height)

    def on_resize(self, width, height):
        aspect_ratio = self.width / self.height
        width, height = self.window.get_framebuffer_size()
        if width / height > aspect_ratio:
            view_width = int(height * aspect_ratio)
            viewport = ((width - view_width) // 2, 0, view_width, height)
        else:
            view_height = int(width / aspect_ratio)
            viewport = (0, (height - view_height) // 2, width, view_height)
        self.glctx.screen.viewport = viewport
        Log.debug("Window resized to %dx%d (viewport %dx%d)", width, height, *viewport[2:])

    def on_close(self):
        self._closed = True

    def render(self, node, **kwargs):
        if self._closed:
            raise ValueError("Window closed")
        self.window.switch_to()
        super().render(node, **kwargs)
        self.window.flip()
        self.window.dispatch_events()


class Shader(ProgramNode):
    def __init__(self, glctx):
        super().__init__(glctx)
        self._framebuffer = None
        self._texture = None

    @property
    def texture(self):
        return self._texture

    @property
    def framebuffer(self):
        return self._framebuffer

    def release(self):
        if self._framebuffer is not None:
            self._framebuffer.release()
            self._framebuffer = None
        if self._texture is not None:
            self._texture.release()
            self._texture = None
        super().release()

    def create(self, node, resized, **kwargs):
        super().create(node, resized, **kwargs)
        if self._framebuffer is None or self._texture is None or resized:
            if self._framebuffer is not None:
                self._framebuffer.release()
            if self._texture is not None:
                self._texture.release()
            self._texture = self.glctx.texture((self.width, self.height), 4)
            self._framebuffer = self.glctx.framebuffer(color_attachments=(self._texture,))
            self._framebuffer.clear()


class Canvas(SceneNode):
    def __init__(self, glctx):
        super().__init__(glctx)
        self._graphics_context = skia.GrDirectContext.MakeGL()
        self._texture = None
        self._framebuffer = None
        self._surface = None
        self._canvas = None

    @property
    def texture(self):
        return self._texture

    def release(self):
        self._canvas = None
        self._surface = None
        if self._graphics_context is not None:
            self._graphics_context.abandonContext()
            self._graphics_context = None
        if self._framebuffer is not None:
            self._framebuffer.release()
            self._framebuffer = None
        if self._texture is not None:
            self._texture.release()
            self._texture = None

    def create(self, node, resized, **kwargs):
        if resized:
            if self._framebuffer is not None:
                self._framebuffer.release()
            if self._texture is not None:
                self._texture.release()
            self._texture = self.glctx.texture((self.width, self.height), 4)
            self._framebuffer = self.glctx.framebuffer(color_attachments=(self._texture,))
            backend_render_target = skia.GrBackendRenderTarget(self.width, self.height, 0, 0, skia.GrGLFramebufferInfo(self._framebuffer.glo, pyglet.gl.GL_RGBA8))
            self._surface = skia.Surface.MakeFromBackendRenderTarget(self._graphics_context, backend_render_target, skia.kBottomLeft_GrSurfaceOrigin,
                                                                     skia.kRGBA_8888_ColorType, skia.ColorSpace.MakeSRGB())
            self._canvas = self._surface.getCanvas()

    async def descend(self, node, **kwargs):
        # A canvas is a leaf node from the perspective of the OpenGL world
        pass

    def render(self, node, **kwargs):
        self._graphics_context.resetContext()
        self._framebuffer.clear()
        canvas.draw(node, self._canvas)
        self._surface.flushAndSubmit()


class Video(Shader):
    MAX_BUFFER_FRAMES = 60
    BT709 = (1, 1, 1, 0, -0.21482, 2.12798, 1.28033, -0.38059, 0)

    def __init__(self, glctx):
        super().__init__(glctx)
        self._container = None
        self._stream = None
        self._decoder = None
        self._frames = []
        self._current_pts = None
        self._plane_textures = {}
        self._scale = (1, 1)

    def release(self):
        while self._plane_textures:
            self._plane_textures.popitem()[1].release()
        self._current_pts = None
        self._stream = None
        self._decoder = None
        self._frames = []
        self._scale = (1, 1)
        if self._container is not None:
            Log.debug("Closed video %s", self._container.name)
            self._container.close()
            self._container = None
        super().release()

    @property
    def child_textures(self):
        return self._plane_textures

    def get_vertex_source(self, node):
        return """
#version 410
in vec2 position;
out vec2 coord;
void main() {
    gl_Position = vec4(position.x, -position.y, 0.0, 1.0);
    coord = (position + 1.0) / 2.0;
}
"""

    def get_fragment_source(self, node):
        return """
#version 410
in vec2 coord;
out vec4 color;
uniform sampler2D y;
uniform sampler2D u;
uniform sampler2D v;
uniform vec2 scale;
uniform mat3 color_conversion;
void main() {
    vec2 xy = coord * scale;
    vec3 yuv = vec3(texture(y, xy).r, texture(u, xy).r, texture(v, xy).r);
    yuv -= vec3(0.0627451, 0.5, 0.5);
    yuv *= 1.138393;
    color = vec4(color_conversion * yuv, 1);
}
"""

    async def update(self, node, **kwargs):
        references = kwargs.setdefault('references', {})
        node_id = node['id'].as_string() if 'id' in node else None
        if node_id:
            references[node_id] = self
        if 'filename' in node:
            filename = node['filename'].as_string()
            if self._container is not None and self._container.name != filename:
                self.release()
            if self._container is None and filename:
                try:
                    container = av.container.open(filename)
                    stream = container.streams.video[0]
                except (FileNotFoundError, av.InvalidDataError, IndexError):
                    return
                Log.info("Opened video %r", filename)
                self._container = container
                self._stream = stream
                self._stream.thread_type = 'AUTO'
                codec_context = self._stream.codec_context
                if codec_context.format.name != 'yuv420p':
                    Log.warning("Video %s has pixel format %s, which will require reformatting (slower)", container.name, codec_context.format.name)
                self.width, self.height = codec_context.width, codec_context.height
                self._texture = self.glctx.texture((self.width, self.height), 3)
                self._framebuffer = self.glctx.framebuffer(color_attachments=(self._texture,))
                self._decoder = self._container.decode(streams=(self._stream.index,))
        if self._container is not None:
            await self.read_frame(node)
            self.render(node, color_conversion=self.BT709, scale=self._scale, **kwargs)

    async def read_frame(self, node):
        position = node.get('position', 1, float, 0)
        loop = node.get('loop', 1, bool, False)
        start_position = self._stream.start_time
        if loop:
            timestamp = start_position + int(position / self._stream.time_base) % self._stream.duration
        else:
            timestamp = min(max(start_position, int(position / self._stream.time_base)), start_position + self._stream.duration)
        while True:
            while len(self._frames) > 1 and timestamp > self._frames[1].pts:
                self._frames.pop(0)
            while self._decoder is not None and (not self._frames or self._frames[-1].pts < timestamp) and len(self._frames) < self.MAX_BUFFER_FRAMES:
                try:
                    self._frames.append(next(self._decoder).reformat(format='yuv420p'))
                except StopIteration:
                    Log.debug("Reached end of video %r", self._container.name)
                    self._decoder = None
                    break
            else:
                if timestamp < self._frames[0].pts or timestamp > self._frames[-1].pts and self._decoder is not None:
                    Log.debug("Seek video %r to position %0.2f", self._container.name, timestamp * self._stream.time_base)
                    self._frames = []
                    self._container.seek(timestamp, stream=self._stream)
                    self._decoder = self._container.decode(streams=(self._stream.index,))
                    continue
            break
        frame = self._frames[0]
        if frame.pts != self._current_pts:
            for name, plane in zip('yuv', frame.planes):
                if name not in self._plane_textures:
                    if plane.line_size > plane.width:
                        self._scale = (plane.width / plane.line_size, 1)
                    self._plane_textures[name] = self.glctx.texture((plane.line_size, plane.height), 1)
                self._plane_textures[name].write(memoryview(plane))
            self._current_pts = frame.pts
