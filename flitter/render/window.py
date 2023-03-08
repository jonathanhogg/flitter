"""
Flitter window management
"""

# pylama:ignore=C0413,E402,W0703,R0914,R0902,R0912,R0201,R1702,C901,W0223,W0231,R0915

import array
from collections import namedtuple
import sys
import time

from mako.template import Template
from loguru import logger
import skia
import moderngl
import pyglet
if sys.platform == 'darwin':
    pyglet.options['shadow_window'] = False
import pyglet.canvas
import pyglet.window
import pyglet.gl

from ..cache import SharedCache
from . import canvas


def value_split(value, n, m):
    if m == 1:
        return value if n == 1 else list(value)
    elif n == 1:
        return tuple(value)
    return [tuple(value[i*m:(i+1)*m]) for i in range(n)]


ColorDepth = namedtuple('ColorDepth', ('moderngl_dtype', 'gl_format', 'skia_colortype'))

COLORDEPTHS = {
    8: ColorDepth('f1', pyglet.gl.GL_RGBA8, skia.kRGBA_8888_ColorType),
    16: ColorDepth('f2', pyglet.gl.GL_RGBA16F, skia.kRGBA_F16_ColorType),
    # 32: ColorDepth('f4', pyglet.gl.GL_RGBA32F, skia.kRGBA_F32_ColorType)  -- Canvas currently fails with 32bit color
}

DEFAULT_COLORDEPTH = 8


class SceneNode:
    def __init__(self, glctx):
        self.glctx = glctx
        self.children = []
        self.width = None
        self.height = None
        self.tags = set()

    @property
    def name(self):
        return '#'.join((self.__class__.__name__.lower(), *self.tags))

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
        self.purge()
        self.release()
        self.glctx = None

    def purge(self):
        while self.children:
            self.children.pop().destroy()

    async def update(self, node, **kwargs):
        references = kwargs.setdefault('references', {})
        if node_id := node.get('id', 1, str):
            references[node_id] = self
        resized = False
        width, height = node.get('size', 2, int, (512, 512))
        if width != self.width or height != self.height:
            self.width = width
            self.height = height
            resized = True
        self.tags = node.tags
        self.create(node, resized, **kwargs)
        await self.descend(node, **kwargs)
        self.render(node, **kwargs)

    def similar_to(self, node):
        return node.tags and node.tags == self.tags

    async def descend(self, node, **kwargs):
        count = 0
        existing = self.children
        updated = []
        for child in node.children:
            cls = {'reference': Reference, 'shader': Shader, 'canvas': Canvas, 'video': Video}[child.kind]
            index = None
            for i, scene_node in enumerate(existing):
                if type(scene_node) == cls:
                    if scene_node.similar_to(child):
                        index = i
                        break
                    if index is None:
                        index = i
            if index is not None:
                scene_node = existing.pop(index)
            else:
                scene_node = cls(self.glctx)
            await scene_node.update(child, **kwargs)
            updated.append(scene_node)
        while existing:
            existing.pop().destroy()
        self.children = updated

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
        node_id = node.get('id', 1, str)
        self._reference = references.get(node_id) if references is not None and node_id else None

    def destroy(self):
        self._reference = None


class ProgramNode(SceneNode):
    GL_VERSION = (4, 1)
    DEFAULT_VERTEX_SOURCE = """#version 410
in vec2 position;
out vec2 coord;
void main() {
gl_Position = vec4(position, 0.0, 1.0);
coord = (position + 1.0) / 2.0;
}
"""
    DEFAULT_FRAGMENT_SOURCE = Template("""#version 410
in vec2 coord;
out vec4 color;
uniform float alpha = 1;
% for name in child_textures:
uniform sampler2D ${name};
% endfor
void main() {
% if child_textures:
    % for name in child_textures:
        % if loop.index == 0:
    vec4 merged = texture(${name}, coord);
        % elif loop.index == 1:
    vec4 child = texture(${name}, coord);
    merged = merged * (1.0 - child.a) + child;
        % else:
    child = texture(${name}, coord);
    merged = merged * (1.0 - child.a) + child;
        % endif
    % endfor
    color = merged * alpha;
% else:
    color = vec4(0);
% endif
}
""")

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
        if self._program is not None:
            self._program.release()
        if self._rectangle is not None:
            self._rectangle.release()
        self._program = None
        self._rectangle = None
        self._vertex_source = None
        self._fragment_source = None
        self._last = None

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
        return node.get('vertex', 1, str, self.DEFAULT_VERTEX_SOURCE)

    def get_fragment_source(self, node):
        if fragment := node.get('fragment', 1, str):
            return fragment
        return self.DEFAULT_FRAGMENT_SOURCE.render(child_textures=self.child_textures)

    def make_last(self):
        raise NotImplementedError()

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
                start = time.perf_counter()
                self._program = self.glctx.program(vertex_shader=self._vertex_source, fragment_shader=self._fragment_source)
                vertices = self.glctx.buffer(array.array('f', [-1, 1, -1, -1, 1, 1, 1, -1]))
                self._rectangle = self.glctx.vertex_array(self._program, [(vertices, '2f', 'position')])
            except Exception as exc:
                error = str(exc).strip().split('\n')
                logger.error("{} GL program compile failed: {}", self.name, error[-1])
                source = self._fragment_source if 'fragment_shader' in error else self._vertex_source
                source = '\n'.join(f'{i+1:3d}|{line}' for i, line in enumerate(source.split('\n')))
                logger.trace("Failing source:\n{}", source)
            else:
                end = time.perf_counter()
                logger.debug("{} GL program compiled in {:.1f}ms", self.name, 1000*(end-start))

    def render(self, node, **kwargs):
        self.compile(node)
        if self._rectangle is not None:
            sampler_args = {'repeat_x': False, 'repeat_y': False}
            if (border := node.get('border', 4, float)) is not None:
                sampler_args['border_color'] = tuple(border)
            elif (repeat := node.get('repeat', 2, bool)) is not None:
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
                            self._last = self.make_last()
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
                        dtype = {'f': float, 'd': float, 'i': int, 'I': int}[member.fmt[-1]]
                        if (value := node.get(name, member.array_length * member.dimension, dtype)) is not None:
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
            logger.debug("{} closed", self.name)
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
            title = node.get('title', 1, str, "flitter")
            fullscreen = node.get('fullscreen', 1, bool, self.default_fullscreen)
            screens = pyglet.canvas.get_display().get_screens()
            screen = screens[screen] if screen < len(screens) else screens[0]
            config = pyglet.gl.Config(major_version=self.GL_VERSION[0], minor_version=self.GL_VERSION[1], forward_compatible=True,
                                      double_buffer=True, sample_buffers=0)
            self.window = self.WindowWrapper(width=self.width, height=self.height, resizable=True, caption=title,
                                             screen=screen, vsync=vsync, config=config)
            self.glctx = moderngl.create_context(require=self.GL_VERSION[0] * 100 + self.GL_VERSION[1] * 10)
            self.glctx.extra = {}
            self.glctx.blend_func = moderngl.PREMULTIPLIED_ALPHA
            if fullscreen:
                self.window.set_mouse_visible(False)
                if sys.platform == 'darwin':
                    self.window._nswindow.enterFullScreenMode_(self.window._nswindow.screen())  # noqa
                else:
                    self.window.set_fullscreen(True)
            logger.debug("{} {} on {}", self.name,  "opened fullscreen" if fullscreen else "opened", screen)
            self.recalculate_viewport(True)
            logger.debug("OpenGL info: {GL_RENDERER} {GL_VERSION}", **self.glctx.info)
        else:
            self.recalculate_viewport()
        self.window.switch_to()
        self.glctx.extra['linear'] = node.get('linear', 1, bool, False)
        colordepth = node.get('colordepth', 1, int, DEFAULT_COLORDEPTH)
        if colordepth not in COLORDEPTHS:
            colordepth = DEFAULT_COLORDEPTH
        self.glctx.extra['colordepth'] = colordepth

    def recalculate_viewport(self, force=False):
        aspect_ratio = self.width / self.height
        width, height = self.window.get_framebuffer_size()
        if width / height > aspect_ratio:
            view_width = int(height * aspect_ratio)
            viewport = ((width - view_width) // 2, 0, view_width, height)
        else:
            view_height = int(width / aspect_ratio)
            viewport = (0, (height - view_height) // 2, width, view_height)
        if force or viewport != self.glctx.screen.viewport:
            width, height = self.window.width, self.window.height
            self.glctx.screen.viewport = viewport
            if viewport != (0, 0, width, height):
                logger.debug("{0} resized to {1}x{2} (viewport {5}x{6} x={3} y={4})", self.name, width, height, *viewport)
            else:
                logger.debug("{} resized to {}x{}", self.name, width, height)

    def render(self, node, **kwargs):
        if self.glctx.extra['linear']:
            self.glctx.enable_direct(pyglet.gl.GL_FRAMEBUFFER_SRGB)
        super().render(node, **kwargs)
        if self.glctx.extra['linear']:
            self.glctx.disable_direct(pyglet.gl.GL_FRAMEBUFFER_SRGB)
        self.window.flip()
        self.window.dispatch_events()

    def make_last(self):
        width, height = self.window.get_framebuffer_size()
        return self.glctx.texture((width, height), 4)


class Shader(ProgramNode):
    def __init__(self, glctx):
        super().__init__(glctx)
        self._framebuffer = None
        self._texture = None
        self._colordepth = None

    @property
    def texture(self):
        return self._texture

    @property
    def framebuffer(self):
        return self._framebuffer

    def release(self):
        self._colordepth = None
        if self._framebuffer is not None:
            self._framebuffer.release()
            self._framebuffer = None
        if self._texture is not None:
            self._texture.release()
            self._texture = None
        super().release()

    def create(self, node, resized, **kwargs):
        super().create(node, resized, **kwargs)
        colordepth = node.get('colordepth', 1, int, self.glctx.extra['colordepth'])
        if colordepth not in COLORDEPTHS:
            colordepth = self.glctx.extra['colordepth']
        if self._framebuffer is None or self._texture is None or resized or colordepth != self._colordepth:
            depth = COLORDEPTHS[colordepth]
            if self._framebuffer is not None:
                self._framebuffer.release()
            if self._texture is not None:
                self._texture.release()
            if self._last is not None:
                self._last.release()
                self._last = None
            self._texture = self.glctx.texture((self.width, self.height), 4, dtype=depth.moderngl_dtype)
            self._framebuffer = self.glctx.framebuffer(color_attachments=(self._texture,))
            self._framebuffer.clear()
            self._colordepth = colordepth

    def make_last(self):
        return self.glctx.texture((self.width, self.height), 4, dtype=COLORDEPTHS[self._colordepth].moderngl_dtype)


class Canvas(SceneNode):
    def __init__(self, glctx):
        super().__init__(glctx)
        self._graphics_context = skia.GrDirectContext.MakeGL()
        self._texture = None
        self._framebuffer = None
        self._surface = None
        self._canvas = None
        self._stats = {}
        self._total_duration = 0
        self._colordepth = None
        self._linear = None

    @property
    def texture(self):
        return self._texture

    def release(self):
        self._colordepth = None
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
        colordepth = node.get('colordepth', 1, int, self.glctx.extra['colordepth'])
        if colordepth not in COLORDEPTHS:
            colordepth = self.glctx.extra['colordepth']
        linear = self.glctx.extra['linear']
        if resized or colordepth != self._colordepth or linear != self._linear:
            if self._framebuffer is not None:
                self._framebuffer.release()
            if self._texture is not None:
                self._texture.release()
            depth = COLORDEPTHS[colordepth]
            self._texture = self.glctx.texture((self.width, self.height), 4, dtype=depth.moderngl_dtype)
            self._framebuffer = self.glctx.framebuffer(color_attachments=(self._texture,))
            backend_render_target = skia.GrBackendRenderTarget(self.width, self.height, 0, 0, skia.GrGLFramebufferInfo(self._framebuffer.glo, depth.gl_format))
            colorspace = skia.ColorSpace.MakeSRGBLinear() if linear else skia.ColorSpace.MakeSRGB()
            self._surface = skia.Surface.MakeFromBackendRenderTarget(self._graphics_context, backend_render_target, skia.kBottomLeft_GrSurfaceOrigin,
                                                                     depth.skia_colortype, colorspace)
            self._canvas = self._surface.getCanvas()
            self._colordepth = colordepth
            self._linear = linear

    async def descend(self, node, **kwargs):
        # A canvas is a leaf node from the perspective of the OpenGL world
        pass

    def purge(self):
        total_count = self._stats['canvas'][0]
        logger.info("{} render stats - {:d} x {:.1f}ms = {:.1f}s", self.name, total_count,
                    1e3*self._total_duration/total_count, self._total_duration)
        draw_duration = 0
        for duration, count, key in sorted(((duration, count, key) for (key, (count, duration)) in self._stats.items()), reverse=True):
            logger.debug("{:15s}  - {:8d}  x {:6.1f}µs = {:5.1f}s  ({:4.1f}%)",
                         key, count, 1e6*duration/count, duration, 100*duration/self._total_duration)
            draw_duration += duration
        overhead = self._total_duration - draw_duration
        logger.debug("{:15s}  - {:8d}  x {:6.1f}µs = {:5.1f}s  ({:4.1f}%)",
                     '(surface)', total_count, 1e6*overhead/total_count, overhead, 100*overhead/self._total_duration)
        self._stats = {}
        self._total_duration = 0

    def render(self, node, **kwargs):
        self._total_duration -= time.perf_counter()
        self._graphics_context.resetContext()
        self._framebuffer.clear()
        canvas.draw(node, self._canvas, stats=self._stats)
        self._surface.flushAndSubmit()
        self._total_duration += time.perf_counter()


class Video(Shader):
    def __init__(self, glctx):
        super().__init__(glctx)
        self._filename = None
        self._frame0 = None
        self._frame1 = None
        self._frame0_texture = None
        self._frame1_texture = None
        self._linear = None
        self._colordepth = None

    def release(self):
        if self._frame0_texture is not None:
            self._frame0_texture.release()
        if self._frame1_texture is not None:
            self._frame1_texture.release()
        self._frame0_texture = None
        self._frame1_texture = None
        self._frame0 = None
        self._frame1 = None
        self._linear = None
        self._colordepth = None
        super().release()

    @property
    def child_textures(self):
        return {'frame0': self._frame0_texture, 'frame1': self._frame1_texture}

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
uniform sampler2D frame0;
uniform sampler2D frame1;
uniform float ratio;
uniform float alpha;
void main() {
    vec4 frame0_color = texture(frame0, coord);
    vec4 frame1_color = texture(frame1, coord);
    color = vec4(mix(frame0_color.rgb, frame1_color.rgb, ratio) * alpha, alpha);
}
"""

    def similar_to(self, node):
        return super().similar_to(node) and node.get('filename', 1, str) == self._filename

    async def update(self, node, **kwargs):
        references = kwargs.setdefault('references', {})
        if node_id := node.get('id', 1, str):
            references[node_id] = self
        self._filename = node.get('filename', 1, str)
        position = node.get('position', 1, float, 0)
        loop = node.get('loop', 1, bool, False)
        ratio, frame0, frame1 = SharedCache[self._filename].read_video_frames(self, position, loop)
        linear = self.glctx.extra['linear']
        colordepth = node.get('colordepth', 1, int, self.glctx.extra['colordepth'])
        if colordepth not in COLORDEPTHS:
            colordepth = self.glctx.extra['colordepth']
        if self._texture is not None and (frame0 is None or (self.width, self.height) != (frame0.width, frame0.height)) \
            or linear != self._linear or colordepth != self._colordepth:
            self.release()
        if frame0 is None:
            return
        if self._texture is None:
            self.width, self.height = frame0.width, frame0.height
            depth = COLORDEPTHS[colordepth]
            self._texture = self.glctx.texture((self.width, self.height), 4, dtype=depth.moderngl_dtype)
            self._framebuffer = self.glctx.framebuffer(color_attachments=(self._texture,))
            format = pyglet.gl.GL_SRGB8 if linear else None
            self._frame0_texture = self.glctx.texture((self.width, self.height), 3, internal_format=format)
            self._frame1_texture = self.glctx.texture((self.width, self.height), 3, internal_format=format)
            self._linear = linear
            self._colordepth = colordepth
        if frame0 is self._frame1 or frame1 is self._frame0:
            self._frame0_texture, self._frame1_texture = self._frame1_texture, self._frame0_texture
            self._frame0, self._frame1 = self._frame1, self._frame0
        if frame0 is not self._frame0:
            rgb_frame = frame0.to_rgb()
            plane = rgb_frame.planes[0]
            data = memoryview(rgb_frame.to_ndarray().data) if plane.line_size > plane.width * 3 else memoryview(plane)
            self._frame0_texture.write(data)
            self._frame0 = frame0
        if frame1 is None:
            self._frame1 = None
        elif frame1 is not self._frame1:
            rgb_frame = frame1.to_rgb()
            plane = rgb_frame.planes[0]
            data = rgb_frame.to_ndarray().data if plane.line_size > plane.width * 3 else plane
            self._frame1_texture.write(memoryview(data))
            self._frame1 = frame1
        interpolate = node.get('interpolate', 1, bool, False)
        alpha = node.get('alpha', 1, float, 1)
        self.render(node, ratio=ratio if interpolate else 0, alpha=alpha, **kwargs)
