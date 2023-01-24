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
        names = list(self.child_textures.keys())
        if names:
            samplers = '\n'.join(f"uniform sampler2D {name};" for name in names)
            composite = ["    vec4 child;"] if len(names) > 1 else []
            composite.append(f"    color = texture({names.pop(0)}, coord);")
            while names:
                composite.append(f"    child = texture({names.pop(0)}, coord);")
                composite.append(f"    color = color * (1.0 - child.a) + child;")
            composite = '\n'.join(composite)
        else:
            samplers = ""
            composite = "    color = vec4(0.0);"
        return f"""#version 410
in vec2 coord;
out vec4 color;
{samplers}
void main() {{
{composite}
}}
"""

    def compile(self, node):
        vertex_source = self.get_vertex_source(node)
        fragment_source = self.get_fragment_source(node)
        if self._rectangle is None or vertex_source != self._vertex_source or fragment_source != self._fragment_source:
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
    MAX_BUFFER_FRAMES = 300

    def __init__(self, glctx):
        super().__init__(glctx)
        self._container = None
        self._stream = None
        self._decoder = None
        self._frames = []
        self._current_pts = None
        self._next_pts = None
        self._current_texture = None
        self._next_texture = None

    def release(self):
        if self._current_texture is not None:
            self._current_texture.release()
        if self._next_texture is not None:
            self._next_texture.release()
        if self._container is not None:
            Log.info("Closed video %s", self._container.name)
            self._container.close()
        self._container = None
        self._stream = None
        self._decoder = None
        self._frames = []
        self._current_pts = None
        self._next_pts = None
        self._current_texture = None
        self._next_texture = None
        super().release()

    @property
    def child_textures(self):
        return {'current_frame': self._current_texture, 'next_frame': self._next_texture}

    def get_vertex_source(self, node):
        return """#version 410
in vec2 position;
out vec2 coord;
void main() {
    gl_Position = vec4(position.x, -position.y, 0.0, 1.0);
    coord = (position + 1.0) / 2.0;
}
"""

    def get_fragment_source(self, node):
        return """#version 410
in vec2 coord;
out vec4 color;
uniform sampler2D current_frame;
uniform sampler2D next_frame;
uniform float ratio;
void main() {
    vec4 current_color = texture(current_frame, coord);
    vec4 next_color = texture(next_frame, coord);
    color = mix(current_color, next_color, ratio);
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
                if codec_context.display_aspect_ratio:
                    self.width, self.height = int(codec_context.display_aspect_ratio * codec_context.height), codec_context.height
                else:
                    self.width, self.height = codec_context.width, codec_context.height
                self._texture = self.glctx.texture((self.width, self.height), 3)
                self._framebuffer = self.glctx.framebuffer(color_attachments=(self._texture,))
                self._current_texture = self.glctx.texture((codec_context.width, codec_context.height), 3)
                self._next_texture = self.glctx.texture((codec_context.width, codec_context.height), 3)
        position = node.get('position', 1, float, 0)
        loop = node.get('loop', 1, bool, False)
        interpolate = node.get('interpolate', 1, bool, False)
        start_position = self._stream.start_time
        if loop:
            timestamp = start_position + int(position / self._stream.time_base) % self._stream.duration
        else:
            timestamp = min(max(start_position, int(position / self._stream.time_base)), start_position + self._stream.duration)
        if self._container is not None:
            await self.read_frame(node, timestamp)
            if not interpolate or self._next_pts is None:
                ratio = 0
            else:
                ratio = (timestamp - self._current_pts) / (self._next_pts - self._current_pts)
            self.render(node, ratio=ratio, **kwargs)

    async def read_frame(self, node, timestamp):
        while True:
            if len(self._frames) == 2:
                current_frame, next_frame = self._frames
                if current_frame.pts <= timestamp and (next_frame is None or timestamp < next_frame.pts):
                    break
                elif timestamp < current_frame.pts or (timestamp > next_frame.pts and next_frame.key_frame):
                    self._frames = []
            if not self._frames:
                if self._decoder is not None:
                    self._decoder.close()
                self._container.seek(timestamp, stream=self._stream)
                self._decoder = self._container.decode(streams=(self._stream.index,))
            if self._decoder is not None:
                if len(self._frames) == 2:
                    self._frames.pop(0)
                try:
                    frame = next(self._decoder)
                except StopIteration:
                    Log.debug("Reached end of video %r", self._container.name)
                    self._decoder = None
                    frame = None
                self._frames.append(frame)
        current_frame, next_frame = self._frames
        if current_frame.pts == self._next_pts:
            self._current_texture, self._next_texture = self._next_texture, self._current_texture
            self._current_pts, self._next_pts = self._next_pts, self._current_pts
        if current_frame.pts != self._current_pts:
            self._current_texture.write(memoryview(current_frame.to_rgb().planes[0]))
            self._current_pts = current_frame.pts
        if next_frame is None:
            self._next_pts = None
        elif next_frame.pts != self._next_pts:
            self._next_texture.write(memoryview(next_frame.to_rgb().planes[0]))
            self._next_pts = next_frame.pts
