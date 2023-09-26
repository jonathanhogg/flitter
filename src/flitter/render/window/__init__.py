"""
Flitter window management
"""

import array
from collections import namedtuple
import math

import glfw
from loguru import logger
import skia
import moderngl

from ...cache import SharedCache
from . import canvas
from . import canvas3d
from ...clock import system_clock
from .glconstants import GL_RGBA8, GL_RGBA16F, GL_RGBA32F, GL_SRGB8, GL_SRGB8_ALPHA8, GL_FRAMEBUFFER_SRGB
from .glsl import TemplateLoader


def value_split(value, n, m):
    if m == 1:
        return value if n == 1 else list(value)
    elif n == 1:
        return tuple(value)
    return [tuple(value[i * m:(i + 1) * m]) for i in range(n)]


ColorFormat = namedtuple('ColorFormat', ('moderngl_dtype', 'gl_format', 'skia_colortype'))

COLOR_FORMATS = {
    8: ColorFormat('f1', GL_RGBA8, skia.kRGBA_8888_ColorType),
    16: ColorFormat('f2', GL_RGBA16F, skia.kRGBA_F16_ColorType),
    32: ColorFormat('f4', GL_RGBA32F, skia.kRGBA_F32_ColorType)  # Canvas currently fails with 32bit color
}

DEFAULT_LINEAR = False
DEFAULT_COLORBITS = 8


class SceneNode:
    def __init__(self, glctx):
        self.glctx = glctx
        self.children = []
        self.width = None
        self.height = None
        self.tags = set()
        self.hidden = False

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
            if child.texture is not None and not child.hidden:
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

    async def update(self, node, default_size=(512, 512), **kwargs):
        references = kwargs.setdefault('references', {})
        if node_id := node.get('id', 1, str):
            references[node_id] = self
        self.hidden = node.get('hidden', 1, bool, False)
        resized = False
        width, height = node.get('size', 2, int, default_size)
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
        existing = self.children
        updated = []
        for child in node.children:
            cls = {'reference': Reference, 'shader': Shader, 'canvas': Canvas, 'canvas3d': Canvas3D,
                   'record': Record, 'video': Video}.get(child.kind)
            if cls is not None:
                index = None
                for i, scene_node in enumerate(existing):
                    if type(scene_node) is cls:
                        if scene_node.similar_to(child):
                            index = i
                            break
                        if index is None:
                            index = i
                if index is not None:
                    scene_node = existing.pop(index)
                else:
                    scene_node = cls(self.glctx)
                await scene_node.update(child, default_size=(self.width, self.height), **kwargs)
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
    def __init__(self, glctx):
        super().__init__(glctx)
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
    GL_VERSION = (4, 0)
    DEFAULT_VERTEX_SOURCE = TemplateLoader.get_template('default.vert')
    DEFAULT_FRAGMENT_SOURCE = TemplateLoader.get_template('default.frag')

    def __init__(self, glctx):
        super().__init__(glctx)
        self._program = None
        self._rectangle = None
        self._vertex_source = None
        self._fragment_source = None
        self._last = None

    def release(self):
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
            self._last = None

    def get_vertex_source(self, node):
        if vertex := node.get('vertex', 1, str):
            return vertex
        return self.DEFAULT_VERTEX_SOURCE.render(child_textures=list(self.child_textures))

    def get_fragment_source(self, node):
        if fragment := node.get('fragment', 1, str):
            return fragment
        composite = node.get('composite', 1, str, node.get('blend', 1, str, 'over'))
        return self.DEFAULT_FRAGMENT_SOURCE.render(child_textures=list(self.child_textures), composite=composite)

    def make_last(self):
        raise NotImplementedError()

    def compile(self, node):
        vertex_source = self.get_vertex_source(node)
        fragment_source = self.get_fragment_source(node)
        if vertex_source != self._vertex_source or fragment_source != self._fragment_source:
            self._vertex_source = vertex_source
            self._fragment_source = fragment_source
            self._program = None
            self._rectangle = None
            try:
                start = system_clock()
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
                end = system_clock()
                logger.debug("{} GL program compiled in {:.1f}ms", self.name, 1000 * (end - start))

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
            unit = 1
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
            self.glctx.enable_direct(GL_FRAMEBUFFER_SRGB)
            self.framebuffer.clear()
            self._rectangle.render(mode=moderngl.TRIANGLE_STRIP)
            for sampler in samplers:
                sampler.clear()
            self.glctx.disable_direct(GL_FRAMEBUFFER_SRGB)


class Window(ProgramNode):
    Windows = []

    def __init__(self, screen=0, fullscreen=False, vsync=False, **kwargs):
        super().__init__(None)
        self.window = None
        self.default_screen = screen
        self.default_fullscreen = fullscreen
        self.default_vsync = vsync
        self._deferred_fullscreen = False

    def release(self):
        if self.window is not None:
            self.glctx.finish()
            self.glctx.extra.clear()
            if count := self.glctx.gc():
                logger.trace("Collected {} OpenGL objects", count)
            self.glctx.release()
            self.glctx = None
            glfw.destroy_window(self.window)
            self.window = None
            Window.Windows.remove(self)
            if not Window.Windows:
                glfw.terminate()
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
            title = node.get('title', 1, str, "flitter")
            screen = node.get('screen', 1, int, self.default_screen)
            fullscreen = node.get('fullscreen', 1, bool, self.default_fullscreen)
            resizable = node.get('resizable', 1, bool, True)
            if not Window.Windows:
                glfw.init()
            monitors = glfw.get_monitors()
            monitor = monitors[screen] if screen < len(monitors) else monitors[0]
            mx, my, mw, mh = glfw.get_monitor_workarea(monitor)
            width, height = self.width, self.height
            while width > mw * 0.95 or height > mh * 0.95:
                width = width * 2 // 3
                height = height * 2 // 3
            glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.NATIVE_CONTEXT_API)
            glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_API)
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, self.GL_VERSION[0])
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, self.GL_VERSION[1])
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
            glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE)
            glfw.window_hint(glfw.SAMPLES, 0)
            glfw.window_hint(glfw.AUTO_ICONIFY, glfw.FALSE)
            glfw.window_hint(glfw.CENTER_CURSOR, glfw.FALSE)
            glfw.window_hint(glfw.RESIZABLE, glfw.TRUE if resizable else glfw.FALSE)
            glfw.window_hint(glfw.SCALE_TO_MONITOR, glfw.TRUE)
            glfw.window_hint(glfw.SRGB_CAPABLE, glfw.TRUE)
            self.window = glfw.create_window(width, height, title, None, Window.Windows[0].window if Window.Windows else None)
            glfw.set_window_pos(self.window, mx + (mw - width) // 2, my + (mh - height) // 2)
            if fullscreen:
                mode = glfw.get_video_mode(monitor)
                glfw.set_window_monitor(self.window, monitor, 0, 0, mode.size.width, mode.size.height, mode.refresh_rate)
            Window.Windows.append(self)
            glfw.make_context_current(self.window)
            self.glctx = moderngl.create_context(self.GL_VERSION[0] * 100 + self.GL_VERSION[1] * 10)
            self.glctx.gc_mode = 'context_gc'
            self.glctx.extra = {}
            logger.debug("{} opened on {}", self.name, screen)
            self.recalculate_viewport(True)
            logger.debug("OpenGL info: {GL_RENDERER} {GL_VERSION}", **self.glctx.info)
            logger.trace("{!r}", self.glctx.info)
        else:
            glfw.make_context_current(self.window)
            self.recalculate_viewport()
        self.glctx.extra['linear'] = node.get('linear', 1, bool, DEFAULT_LINEAR)
        colorbits = node.get('colorbits', 1, int, DEFAULT_COLORBITS)
        if colorbits not in COLOR_FORMATS:
            colorbits = DEFAULT_COLORBITS
        self.glctx.extra['colorbits'] = colorbits
        self.glctx.extra['size'] = self.width, self.height

    def recalculate_viewport(self, force=False):
        aspect_ratio = self.width / self.height
        width, height = glfw.get_framebuffer_size(self.window)
        if width / height > aspect_ratio:
            view_width = int(height * aspect_ratio)
            viewport = ((width - view_width) // 2, 0, view_width, height)
        else:
            view_height = int(width / aspect_ratio)
            viewport = (0, (height - view_height) // 2, width, view_height)
        if force or viewport != self.glctx.screen.viewport:
            self.glctx.screen.viewport = viewport
            width, height = glfw.get_window_size(self.window)
            if viewport != (0, 0, width, height):
                logger.debug("{0} resized to {1}x{2} (viewport {5}x{6} x={3} y={4})", self.name, width, height, *viewport)
            else:
                logger.debug("{} resized to {}x{}", self.name, width, height)

    def render(self, node, **kwargs):
        super().render(node, **kwargs)
        vsync = node.get('vsync', 1, bool, self.default_vsync)
        glfw.swap_interval(1 if vsync else 0)
        glfw.swap_buffers(self.window)
        glfw.poll_events()
        if count := self.glctx.gc():
            logger.trace("Collected {} OpenGL objects", count)

    def make_last(self):
        width, height = self.window.get_framebuffer_size()
        return self.glctx.texture((width, height), 4)


class Shader(ProgramNode):
    def __init__(self, glctx):
        super().__init__(glctx)
        self._framebuffer = None
        self._texture = None
        self._colorbits = None

    @property
    def texture(self):
        return self._texture

    @property
    def framebuffer(self):
        return self._framebuffer

    def release(self):
        self._colorbits = None
        self._framebuffer = None
        self._texture = None
        super().release()

    def create(self, node, resized, **kwargs):
        super().create(node, resized, **kwargs)
        colorbits = node.get('colorbits', 1, int, self.glctx.extra['colorbits'])
        if colorbits not in COLOR_FORMATS:
            colorbits = self.glctx.extra['colorbits']
        if self._framebuffer is None or self._texture is None or resized or colorbits != self._colorbits:
            depth = COLOR_FORMATS[colorbits]
            self._last = None
            self._texture = self.glctx.texture((self.width, self.height), 4, dtype=depth.moderngl_dtype)
            self._framebuffer = self.glctx.framebuffer(color_attachments=(self._texture,))
            self._framebuffer.clear()
            self._colorbits = colorbits

    def make_last(self):
        return self.glctx.texture((self.width, self.height), 4, dtype=COLOR_FORMATS[self._colorbits].moderngl_dtype)


class Record(ProgramNode):
    DEFAULT_VERTEX_SOURCE = TemplateLoader.get_template('video.vert')
    DEFAULT_FRAGMENT_SOURCE = TemplateLoader.get_template('record.frag')

    def __init__(self, glctx):
        super().__init__(glctx)
        self._framebuffer = None
        self._texture = None
        self._has_alpha = None

    @property
    def texture(self):
        return self.children[0].texture if self.children else None

    @property
    def framebuffer(self):
        return self._framebuffer

    def release(self):
        self._framebuffer = None
        self._texture = None
        super().release()

    def create(self, node, resized, **kwargs):
        super().create(node, resized, **kwargs)
        has_alpha = node.get('keep_alpha', 1, bool, False)
        if self._framebuffer is None or self._texture is None or resized or has_alpha != self._has_alpha:
            self._last = None
            self._has_alpha = has_alpha
            self._texture = self.glctx.texture((self.width, self.height), 4 if self._has_alpha else 3,
                                               dtype='f1', internal_format=GL_SRGB8_ALPHA8 if self._has_alpha else GL_SRGB8)
            self._framebuffer = self.glctx.framebuffer(color_attachments=(self._texture,))
            self._framebuffer.clear()

    def render(self, node, **kwargs):
        if filename := node.get('filename', 1, str):
            super().render(node, **kwargs)
            path = SharedCache[filename]
            if path.suffix.lower() in ('.mp4', '.mov', '.m4v', '.mkv'):
                codec = node.get('codec', 1, str, 'h264')
                crf = node.get('crf', 1, int)
                bitrate = node.get('bitrate', 1, int)
                preset = node.get('preset', 1, str)
                limit = node.get('limit', 1, float)
                path.write_video_frame(self._texture, kwargs['clock'],
                                       fps=int(kwargs['fps']), realtime=kwargs['realtime'], codec=codec,
                                       crf=crf, bitrate=bitrate, preset=preset, limit=limit)
            else:
                quality = node.get('quality', 1, int)
                path.write_image(self._texture, quality=quality)


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
        self._colorbits = None
        self._linear = None
        self._colorspace = None

    @property
    def texture(self):
        return self._texture

    def release(self):
        self._colorbits = None
        self._linear = None
        self._colorspace = None
        self._canvas = None
        self._surface = None
        if self._graphics_context is not None:
            self._graphics_context.abandonContext()
            self._graphics_context = None
        self._framebuffer = None
        self._texture = None

    def create(self, node, resized, **kwargs):
        colorbits = node.get('colorbits', 1, int, self.glctx.extra['colorbits'])
        if colorbits not in COLOR_FORMATS:
            colorbits = self.glctx.extra['colorbits']
        linear = node.get('linear', 1, bool, self.glctx.extra['linear'])
        if resized or colorbits != self._colorbits or linear != self._linear:
            depth = COLOR_FORMATS[colorbits]
            internal_format = None if linear else GL_SRGB8_ALPHA8
            self._colorspace = skia.ColorSpace.MakeSRGBLinear() if linear else skia.ColorSpace.MakeSRGB()
            self._texture = self.glctx.texture((self.width, self.height), 4, dtype=depth.moderngl_dtype, internal_format=internal_format)
            self._framebuffer = self.glctx.framebuffer(color_attachments=(self._texture,))
            backend_render_target = skia.GrBackendRenderTarget(self.width, self.height, 0, 0, skia.GrGLFramebufferInfo(self._framebuffer.glo, depth.gl_format))
            self._surface = skia.Surface.MakeFromBackendRenderTarget(self._graphics_context, backend_render_target, skia.kBottomLeft_GrSurfaceOrigin,
                                                                     depth.skia_colortype, self._colorspace)
            self._canvas = self._surface.getCanvas()
            self._colorbits = colorbits
            self._linear = linear
            logger.debug("Created {:d}x{:d} canvas; skia version {}", self.width, self.height, skia.__version__)

    async def descend(self, node, **kwargs):
        # A canvas is a leaf node from the perspective of the OpenGL world
        pass

    def purge(self):
        total_count = self._stats['canvas'][0]
        logger.info("{} render stats - {:d} x {:.1f}ms = {:.1f}s", self.name, total_count,
                    1e3 * self._total_duration / total_count, self._total_duration)
        draw_duration = 0
        for duration, count, key in sorted(((duration, count, key) for (key, (count, duration)) in self._stats.items()), reverse=True):
            logger.debug("{:15s}  - {:8d}  x {:6.1f}µs = {:5.1f}s  ({:4.1f}%)",
                         key, count, 1e6 * duration / count, duration, 100 * duration / self._total_duration)
            draw_duration += duration
        overhead = self._total_duration - draw_duration
        logger.debug("{:15s}  - {:8d}  x {:6.1f}µs = {:5.1f}s  ({:4.1f}%)",
                     '(surface)', total_count, 1e6 * overhead / total_count, overhead, 100 * overhead / self._total_duration)
        self._stats = {}
        self._total_duration = 0

    def render(self, node, references=None, **kwargs):
        self._total_duration -= system_clock()
        self._graphics_context.resetContext()
        self._framebuffer.clear()
        canvas.draw(node, self._canvas, stats=self._stats, references=references, colorspace=self._colorspace)
        self._surface.flushAndSubmit()
        self._total_duration += system_clock()


class Canvas3D(SceneNode):
    def __init__(self, glctx):
        super().__init__(glctx)
        self._image_texture = None
        self._image_framebuffer = None
        self._color_renderbuffer = None
        self._depth_renderbuffer = None
        self._render_framebuffer = None
        self._colorbits = None
        self._samples = None
        self._total_duration = 0
        self._total_count = 0

    @property
    def texture(self):
        return self._image_texture

    def release(self):
        self._colorbits = None
        self._samples = None
        self._render_framebuffer = None
        self._image_texture = None
        self._image_framebuffer = None
        self._color_renderbuffer = None
        self._depth_renderbuffer = None

    def purge(self):
        logger.info("{} draw stats - {:d} x {:.1f}ms = {:.1f}s", self.name, self._total_count,
                    1e3 * self._total_duration / self._total_count, self._total_duration)
        self._total_duration = 0
        self._total_count = 0

    def create(self, node, resized, **kwargs):
        colorbits = node.get('colorbits', 1, int, self.glctx.extra['colorbits'])
        if colorbits not in COLOR_FORMATS:
            colorbits = self.glctx.extra['colorbits']
        samples = max(0, node.get('samples', 1, int, 0))
        if samples:
            samples = min(1 << int(math.log2(samples)), self.glctx.info['GL_MAX_SAMPLES'])
        if resized or colorbits != self._colorbits or samples != self._samples:
            self.release()
            format = COLOR_FORMATS[colorbits]
            self._image_texture = self.glctx.texture((self.width, self.height), 4, dtype=format.moderngl_dtype)
            self._depth_renderbuffer = self.glctx.depth_renderbuffer((self.width, self.height), samples=samples)
            if samples:
                self._color_renderbuffer = self.glctx.renderbuffer((self.width, self.height), 4, samples=samples, dtype=format.moderngl_dtype)
                self._render_framebuffer = self.glctx.framebuffer(color_attachments=(self._color_renderbuffer,), depth_attachment=self._depth_renderbuffer)
                self._image_framebuffer = self.glctx.framebuffer(self._image_texture)
                logger.debug("Created canvas3d {}x{}/{}-bit render target with {}x sampling", self.width, self.height, colorbits, samples)
            else:
                self._render_framebuffer = self.glctx.framebuffer(color_attachments=(self._image_texture,), depth_attachment=self._depth_renderbuffer)
                logger.debug("Created canvas3d {}x{}/{}-bit render target", self.width, self.height, colorbits)
            self._colorbits = colorbits
            self._samples = samples

    async def descend(self, node, **kwargs):
        # A canvas3d is a leaf node from the perspective of the OpenGL world
        pass

    def render(self, node, references=None, **kwargs):
        self._total_duration -= system_clock()
        self._render_framebuffer.use()
        fog_min = node.get('fog_min', 1, float, 0)
        fog_max = node.get('fog_max', 1, float, 0)
        if fog_max > fog_min:
            fog_color = node.get('fog_color', 3, float, (0, 0, 0))
            self._render_framebuffer.clear(*fog_color)
        else:
            self._render_framebuffer.clear()
        objects = self.glctx.extra.setdefault('canvas3d_objects', {})
        canvas3d.draw(node, (self.width, self.height), self.glctx, objects, references)
        if self._image_framebuffer is not None:
            self.glctx.copy_framebuffer(self._image_framebuffer, self._render_framebuffer)
        self._total_duration += system_clock()
        self._total_count += 1


class Video(Shader):
    DEFAULT_VERTEX_SOURCE = TemplateLoader.get_template('video.vert')
    DEFAULT_FRAGMENT_SOURCE = TemplateLoader.get_template('video.frag')

    def __init__(self, glctx):
        super().__init__(glctx)
        self._filename = None
        self._frame0 = None
        self._frame1 = None
        self._frame0_texture = None
        self._frame1_texture = None
        self._colorbits = None

    def release(self):
        self._frame0_texture = None
        self._frame1_texture = None
        self._frame0 = None
        self._frame1 = None
        self._colorbits = None
        super().release()

    @property
    def child_textures(self):
        return {'frame0': self._frame0_texture, 'frame1': self._frame1_texture}

    def similar_to(self, node):
        return super().similar_to(node) and node.get('filename', 1, str) == self._filename

    async def update(self, node, **kwargs):
        references = kwargs.setdefault('references', {})
        if node_id := node.get('id', 1, str):
            references[node_id] = self
        self.hidden = node.get('hidden', 1, bool, False)
        self._filename = node.get('filename', 1, str)
        position = node.get('position', 1, float, 0)
        loop = node.get('loop', 1, bool, False)
        threading = node.get('thread', 1, bool, False)
        if self._filename is not None:
            ratio, frame0, frame1 = SharedCache[self._filename].read_video_frames(self, position, loop, threading=threading)
        else:
            ratio, frame0, frame1 = 0, None, None
        colorbits = node.get('colorbits', 1, int, self.glctx.extra['colorbits'])
        if colorbits not in COLOR_FORMATS:
            colorbits = self.glctx.extra['colorbits']
        if self._texture is not None and (frame0 is None or (self.width, self.height) != (frame0.width, frame0.height)) \
                or colorbits != self._colorbits:
            self.release()
        if frame0 is None:
            return
        if self._texture is None:
            self.width, self.height = frame0.width, frame0.height
            depth = COLOR_FORMATS[colorbits]
            self._texture = self.glctx.texture((self.width, self.height), 4, dtype=depth.moderngl_dtype)
            self._framebuffer = self.glctx.framebuffer(color_attachments=(self._texture,))
            self._frame0_texture = self.glctx.texture((self.width, self.height), 3, internal_format=GL_SRGB8)
            self._frame1_texture = self.glctx.texture((self.width, self.height), 3, internal_format=GL_SRGB8)
            self._colorbits = colorbits
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
        self.render(node, ratio=ratio if interpolate else 0, **kwargs)


RENDERER_CLASS = Window
