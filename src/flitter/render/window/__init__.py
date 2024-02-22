"""
Flitter window management
"""

import array
from collections import namedtuple
import importlib

import glfw
from loguru import logger
import moderngl
import numpy as np

from ...clock import system_clock
from .glconstants import GL_RGBA8, GL_RGBA16F, GL_RGBA32F, GL_FRAMEBUFFER_SRGB
from .glsl import TemplateLoader
from ...model import Vector


def set_uniform_vector(uniform, vector):
    dtype = {'f': float, 'd': float, 'i': int, 'I': int}[uniform.fmt[-1]]
    n, m = uniform.array_length, uniform.dimension
    if (value := vector.match(n * m, dtype)) is not None:
        if m == 1:
            uniform.value = value if n == 1 else list(value)
        elif n == 1:
            uniform.value = tuple(value)
        else:
            uniform.value = [tuple(value[i*m:(i+1)*m]) for i in range(n)]


ColorFormat = namedtuple('ColorFormat', ('moderngl_dtype', 'gl_format'))

COLOR_FORMATS = {
    8: ColorFormat('f1', GL_RGBA8),
    16: ColorFormat('f2', GL_RGBA16F),
    32: ColorFormat('f4', GL_RGBA32F)
}

DEFAULT_LINEAR = False
DEFAULT_COLORBITS = 16
PUSHED = Vector.symbol('pushed')
RELEASED = Vector.symbol('released')
PUSHED_BEAT = Vector.symbol('pushed').concat(Vector.symbol('beat'))
RELEASED_BEAT = Vector.symbol('released').concat(Vector.symbol('beat'))


def get_scene_node_class(kind):
    global ClassCache
    if kind in ClassCache:
        return ClassCache[kind]
    try:
        module = importlib.import_module(f'.{kind}', __package__)
        cls = module.SCENE_NODE_CLASS
    except ModuleNotFoundError:
        logger.warning("No sub-module for '{}'", kind)
        cls = None
    except ImportError:
        logger.exception("Import error")
        cls = None
    except AttributeError:
        logger.warning("Sub-module '{}' does not contain a scene node class", kind)
        cls = None
    ClassCache[kind] = cls
    return cls


class SceneNode:
    def __init__(self, glctx):
        self.glctx = glctx
        self.children = []
        self.width = None
        self.height = None
        self.tags = set()
        self.hidden = False
        self._texture_data = None

    @property
    def name(self):
        return '#'.join((self.__class__.__name__.lower(), *self.tags))

    @property
    def texture(self):
        raise NotImplementedError()

    @property
    def texture_data(self):
        if self._texture_data is None and (texture := self.texture) is not None:
            dtype = {'f1': 'uint8', 'f2': 'float16', 'f4': 'float32'}[texture.dtype]
            data = np.ndarray((texture.height, texture.width, texture.components), dtype, texture.read())
            self._texture_data = data.astype('float64')
            if dtype == 'uint8':
                self._texture_data /= 255
        return self._texture_data

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

    async def update(self, engine, node, default_size=(512, 512), **kwargs):
        self._texture_data = None
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
        self.create(engine, node, resized, **kwargs)
        await self.descend(engine, node, **kwargs)
        self.render(node, **kwargs)

    def similar_to(self, node):
        return node.tags == self.tags

    async def descend(self, engine, node, **kwargs):
        existing = self.children
        updated = []
        for child in node.children:
            if self.handle_node(engine, child, **kwargs):
                continue
            cls = get_scene_node_class(child.kind)
            if cls is not None:
                for i, scene_node in enumerate(existing):
                    if type(scene_node) is cls and scene_node.similar_to(child):
                        scene_node = existing.pop(i)
                        break
                else:
                    scene_node = cls(self.glctx)
                await scene_node.update(engine, child, default_size=(self.width, self.height), **kwargs)
                updated.append(scene_node)
        while existing:
            existing.pop().destroy()
        self.children = updated

    def handle_node(self, engine, node, **kwargs):
        return False

    def create(self, engine, node, resized, **kwargs):
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

    async def update(self, engine, node, references=None, **kwargs):
        node_id = node.get('id', 1, str)
        self._reference = references.get(node_id) if references is not None and node_id else None

    def destroy(self):
        self._reference = None


class ProgramNode(SceneNode):
    GL_VERSION = (3, 3)
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

    def create(self, engine, node, resized, **kwargs):
        super().create(engine, node, resized, **kwargs)
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
                self._rectangle = self.glctx.vertex_array(self._program, [(vertices, '2f', 'position')], mode=moderngl.TRIANGLE_STRIP)
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
                        set_uniform_vector(member, node[name])
            self.glctx.enable_direct(GL_FRAMEBUFFER_SRGB)
            self.framebuffer.clear()
            self._rectangle.render()
            for sampler in samplers:
                sampler.clear()
            self.glctx.disable_direct(GL_FRAMEBUFFER_SRGB)


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

    def create(self, engine, node, resized, **kwargs):
        super().create(engine, node, resized, **kwargs)
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
            logger.debug("Created {}x{} {}-bit framebuffer for {}", self.width, self.height, self._colorbits, self.name)

    def make_last(self):
        logger.debug("Create {}x{} {}-bit last texture for {}", self.width, self.height, self._colorbits, self.name)
        return self.glctx.texture((self.width, self.height), 4, dtype=COLOR_FORMATS[self._colorbits].moderngl_dtype)


class Window(ProgramNode):
    Windows = []

    def __init__(self, screen=0, fullscreen=False, vsync=False, offscreen=False, **kwargs):
        super().__init__(None)
        self.engine = None
        self.window = None
        self.default_screen = screen
        self.default_fullscreen = fullscreen
        self.default_vsync = vsync
        self._deferred_fullscreen = False
        self._visible = not offscreen
        self._screen = None
        self._fullscreen = None
        self._resizable = None
        self._beat = None

    def release(self):
        if self.window is not None:
            self.glctx.finish()
            self.glctx.extra.clear()
            if count := self.glctx.gc():
                logger.trace("Window release collected {} OpenGL objects", count)
            self.glctx.release()
            self.glctx = None
            glfw.destroy_window(self.window)
            self.window = None
            Window.Windows.remove(self)
            if not Window.Windows:
                glfw.terminate()
            self.engine = None
            logger.debug("{} closed", self.name)
        super().release()

    def purge(self):
        self.glctx.finish()
        super().purge()
        if count := self.glctx.gc():
            logger.trace("Window purge collected {} OpenGL objects", count)

    @property
    def texture(self):
        return None

    @property
    def framebuffer(self):
        return self.glctx.screen

    @property
    def size(self):
        return self.glctx.screen.viewport[2:]

    def create(self, engine, node, resized, **kwargs):
        super().create(engine, node, resized)
        new_window = False
        screen = node.get('screen', 1, int, self.default_screen)
        fullscreen = node.get('fullscreen', 1, bool, self.default_fullscreen) if self._visible else False
        resizable = node.get('resizable', 1, bool, True) if self._visible else False
        if self.window is None:
            self.engine = engine
            title = node.get('title', 1, str, "flitter")
            if not Window.Windows:
                ok = glfw.init()
                if not ok:
                    raise RuntimeError("Unable to initialize GLFW")
            glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.NATIVE_CONTEXT_API)
            glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_API)
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, self.GL_VERSION[0])
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, self.GL_VERSION[1])
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
            glfw.window_hint(glfw.VISIBLE, glfw.TRUE if self._visible else glfw.FALSE)
            if self._visible:
                glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE)
                glfw.window_hint(glfw.SAMPLES, 0)
                glfw.window_hint(glfw.AUTO_ICONIFY, glfw.FALSE)
                glfw.window_hint(glfw.CENTER_CURSOR, glfw.FALSE)
                glfw.window_hint(glfw.SCALE_TO_MONITOR, glfw.TRUE)
            glfw.window_hint(glfw.SRGB_CAPABLE, glfw.TRUE)
            self.window = glfw.create_window(self.width, self.height, title, None, Window.Windows[0].window if Window.Windows else None)
            Window.Windows.append(self)
            new_window = True
        if resizable != self._resizable:
            glfw.set_window_attrib(self.window, glfw.RESIZABLE, glfw.TRUE if resizable else glfw.FALSE)
            self._resizable = resizable
        if self._visible and (resized or screen != self._screen or fullscreen != self._fullscreen):
            monitors = glfw.get_monitors()
            monitor = monitors[screen] if screen < len(monitors) else monitors[0]
            mx, my, mw, mh = glfw.get_monitor_workarea(monitor)
            width, height = self.width, self.height
            if not fullscreen:
                glfw.set_window_aspect_ratio(self.window, self.width, self.height)
            while width > mw * 0.95 or height > mh * 0.95:
                width = width * 2 // 3
                height = height * 2 // 3
            if self._fullscreen and not fullscreen or screen != self._screen:
                glfw.set_window_monitor(self.window, None, 0, 0, width, height, glfw.DONT_CARE)
            if new_window or (self._fullscreen and not fullscreen) or screen != self._screen:
                glfw.set_window_pos(self.window, mx + (mw - width) // 2, my + (mh - height) // 2)
            if fullscreen and (not self._fullscreen or screen != self._screen):
                mode = glfw.get_video_mode(monitor)
                glfw.set_window_monitor(self.window, monitor, 0, 0, mode.size.width, mode.size.height, mode.refresh_rate)
            if not fullscreen:
                glfw.set_window_size(self.window, width, height)
            self._screen = screen
            self._fullscreen = fullscreen
        glfw.make_context_current(self.window)
        self._keys = {}
        self._pointer_state = None
        if new_window:
            self.glctx = moderngl.create_context(self.GL_VERSION[0] * 100 + self.GL_VERSION[1] * 10)
            self.glctx.gc_mode = 'context_gc'
            self.glctx.extra = {}
            if self._visible:
                logger.debug("{} opened on screen {}", self.name, screen)
            else:
                logger.debug("{} opened off-screen", self.name)
            logger.debug("OpenGL info: {GL_RENDERER} {GL_VERSION}", **self.glctx.info)
            logger.trace("{!r}", self.glctx.info)
            if self._visible:
                glfw.set_key_callback(self.window, self.key_callback)
                glfw.set_cursor_pos_callback(self.window, self.pointer_movement_callback)
                glfw.set_mouse_button_callback(self.window, self.pointer_button_callback)
        if self._visible:
            self.recalculate_viewport(new_window)
        self.glctx.extra['linear'] = node.get('linear', 1, bool, DEFAULT_LINEAR)
        colorbits = node.get('colorbits', 1, int, DEFAULT_COLORBITS)
        if colorbits not in COLOR_FORMATS:
            colorbits = DEFAULT_COLORBITS
        self.glctx.extra['colorbits'] = colorbits
        self.glctx.extra['size'] = self.width, self.height

    def key_callback(self, window, key, scancode, action, mods):
        if key in self._keys:
            state = self._keys[key]
            if action == glfw.PRESS:
                self.engine.state[state] = 1
                self.engine.state[state.concat(PUSHED)] = 1
                self.engine.state[state.concat(RELEASED)] = 0
                self.engine.state[state.concat(PUSHED_BEAT)] = self._beat
            elif action == glfw.RELEASE:
                self.engine.state[state] = 0
                self.engine.state[state.concat(PUSHED)] = 0
                self.engine.state[state.concat(RELEASED)] = 1
                self.engine.state[state.concat(RELEASED_BEAT)] = self._beat
        elif key == glfw.KEY_LEFT:
            if action == glfw.RELEASE:
                self.engine.previous_page()
        elif key == glfw.KEY_RIGHT:
            if action == glfw.RELEASE:
                self.engine.next_page()

    def pointer_movement_callback(self, window, x, y):
        if self._pointer_state is not None and self.window is not None:
            width, height = glfw.get_window_size(self.window)
            if 0 <= x <= width and 0 <= y <= height:
                self.engine.state[self._pointer_state] = x/width, y/height
            else:
                self.engine.state[self._pointer_state] = None

    def pointer_button_callback(self, window, button, action, mods):
        if self._pointer_state is not None and self.window is not None:
            if action == glfw.PRESS:
                self.engine.state[self._pointer_state.concat(Vector(button))] = 1
            elif action == glfw.RELEASE:
                self.engine.state[self._pointer_state.concat(Vector(button))] = 0

    def handle_node(self, engine, node, **kwargs):
        if node.kind == 'key':
            if 'state' in node:
                key_name = node.get('name', 1, str)
                if key_name:
                    key_constant = 'KEY_' + key_name.upper()
                    if hasattr(glfw, key_constant):
                        key = getattr(glfw, key_constant)
                        state = node['state']
                        self._keys[key] = state
                        if state not in engine.state:
                            if self.window is not None and glfw.get_key(self.window, key) == glfw.PRESS:
                                engine.state[state] = 1
                                engine.state[state.concat(PUSHED)] = 1
                                engine.state[state.concat(RELEASED)] = 0
                            else:
                                engine.state[state] = 0
                                engine.state[state.concat(PUSHED)] = 0
                                engine.state[state.concat(RELEASED)] = 1
            return True
        elif node.kind == 'pointer':
            if 'state' in node:
                self._pointer_state = node['state']
            return True
        return super().handle_node(engine, node, **kwargs)

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

    def render(self, node, window_gamma=1, beat=None, **kwargs):
        gamma = node.get('gamma', 1, float, window_gamma)
        super().render(node, gamma=gamma, **kwargs)
        if self._visible:
            vsync = node.get('vsync', 1, bool, self.default_vsync)
            glfw.swap_interval(1 if vsync else 0)
            glfw.swap_buffers(self.window)
        self._beat = beat
        glfw.poll_events()
        if count := self.glctx.gc():
            logger.trace("Collected {} OpenGL objects", count)

    def make_last(self):
        width, height = self.window.get_framebuffer_size()
        return self.glctx.texture((width, height), 4)


RENDERER_CLASS = Window


ClassCache = {
    'reference': Reference,
    'shader': Shader,
}
