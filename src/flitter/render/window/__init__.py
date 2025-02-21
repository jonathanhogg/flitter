"""
Flitter window management
"""

import array
import asyncio
import ctypes
import os
import sys

import glfw
from loguru import logger
from mako.template import Template
import moderngl

from ...clock import system_clock
from .glsl import TemplateLoader
from .target import RenderTarget, COLOR_FORMATS
from ...model import Vector, true, false
from ...plugins import get_plugin


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
        return True
    elif n > 1 and m > 1 and (value := vector.match(m, dtype)) is not None:
        uniform.value = [tuple(value) for i in range(n)]
        return True
    return False


RESERVED_NAMES = {'fragment', 'vertex', 'loop', 'context', 'UNDEFINED'}
DEFAULT_LINEAR = False
DEFAULT_COLORBITS = 16
PUSHED = Vector.symbol('pushed')
RELEASED = Vector.symbol('released')
PUSHED_BEAT = Vector.symbol('pushed').concat(Vector.symbol('beat'))
RELEASED_BEAT = Vector.symbol('released').concat(Vector.symbol('beat'))

GLFUNCTYPE = ctypes.WINFUNCTYPE if sys.platform == 'win32' else ctypes.CFUNCTYPE


class WindowNode:
    def __init__(self, glctx):
        self.glctx = glctx
        self.children = []
        self.node_id = None
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
    def array(self):
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

    async def destroy(self):
        await self.purge()
        result = self.release()
        if asyncio.coroutines.iscoroutine(result):
            await result
        self.glctx = None

    async def purge(self):
        while self.children:
            await self.children.pop().destroy()

    def release(self):
        pass

    async def update(self, engine, node, references, *, default_size=(512, 512), **kwargs):
        self.node_id = node.get('id', 1, str)
        if self.node_id is not None:
            references[self.node_id] = self
        self.hidden = node.get('hidden', 1, bool, False)
        resized = False
        width, height = node.get('size', 2, int, default_size)
        if width != self.width or height != self.height:
            self.width = width
            self.height = height
            resized = True
        self.tags = node.tags
        result = self.create(engine, node, resized, default_size=(self.width, self.height), **kwargs)
        if asyncio.coroutines.iscoroutine(result):
            await result
        await self.descend(engine, node, references, default_size=(self.width, self.height), **kwargs)
        result = self.render(node, references, **kwargs)
        if asyncio.coroutines.iscoroutine(result):
            await result
        for child in self.children:
            result = child.parent_finished()
            if asyncio.coroutines.iscoroutine(result):
                await result

    def similar_to(self, node):
        return node.tags == self.tags and node.get('id', 1, str) == self.node_id

    def create(self, engine, node, resized, **kwargs):
        pass

    async def descend(self, engine, node, references, **kwargs):
        existing = self.children
        updated = []
        for child in node.children:
            result = self.handle_node(engine, child, **kwargs)
            if asyncio.coroutines.iscoroutine(result):
                result = await result
            if result:
                continue
            cls = get_plugin('flitter.render.window', child.kind)
            if cls is not None:
                for i, scene_node in enumerate(existing):
                    if type(scene_node) is cls and scene_node.similar_to(child):
                        scene_node = existing.pop(i)
                        await scene_node.update(engine, child, references, **kwargs)
                        break
                else:
                    scene_node = cls(self.glctx)
                    await scene_node.update(engine, child, references, **kwargs)
                    logger.trace("New window node: {}", scene_node.name)
                updated.append(scene_node)
        while existing:
            scene_node = existing.pop()
            logger.trace("Destroy window node: {}", scene_node.name)
            await scene_node.destroy()
        self.children = updated

    def render(self, node, references, **kwargs):
        raise NotImplementedError()

    def parent_finished(self):
        pass

    def handle_node(self, engine, node, **kwargs):
        return False

    def __repr__(self):
        return f"<{self.name}{' id='+self.node_id if self.node_id else ''}>"


class Reference(WindowNode):
    def __init__(self, glctx):
        super().__init__(glctx)
        self._reference = None

    def release(self):
        self._reference = None

    @property
    def texture(self):
        return self._reference.texture if self._reference is not None else None

    @property
    def array(self):
        return self._reference.array if self._reference is not None else None

    async def update(self, engine, node, references, **kwargs):
        self.node_id = node.get('id', 1, str)
        self._reference = references.get(self.node_id) if self.node_id else None


class ProgramNode(WindowNode):
    DEFAULT_VERTEX_SOURCE = TemplateLoader.get_template('default.vert')
    DEFAULT_FRAGMENT_SOURCE = TemplateLoader.get_template('default.frag')

    def __init__(self, glctx):
        super().__init__(glctx)
        self._program = None
        self._rectangle = None
        self._vertex_source = None
        self._fragment_source = None
        self._target = None
        self._retain_target = False

    def release(self):
        if self._target is not None:
            self._target.release()
        self._target = None
        self._program = None
        self._rectangle = None
        self._vertex_source = None
        self._fragment_source = None

    def parent_finished(self):
        if not self._retain_target and self._target is not None:
            self._target.release()
            self._target = None

    @property
    def size(self):
        return self.width, self.height

    @property
    def texture(self):
        return self._target.texture if self._target is not None else None

    @property
    def array(self):
        return self._target.array if self._target is not None else None

    def get_vertex_source(self, node, **defaults):
        vertex = Template(node.get('vertex', 1, str), lookup=TemplateLoader) if 'vertex' in node else self.DEFAULT_VERTEX_SOURCE
        names = dict(defaults)
        for name, value in node.items():
            if name not in RESERVED_NAMES:
                names[name] = value
        names['child_textures'] = list(self.child_textures)
        names['HEADER'] = self.glctx.extra['HEADER']
        return vertex.render(**names).strip()

    def get_fragment_source(self, node, **defaults):
        fragment = Template(node.get('fragment', 1, str), lookup=TemplateLoader) if 'fragment' in node else self.DEFAULT_FRAGMENT_SOURCE
        names = dict(defaults)
        for name, value in node.items():
            if name not in RESERVED_NAMES:
                names[name] = value
        names['child_textures'] = list(self.child_textures)
        names['HEADER'] = self.glctx.extra['HEADER']
        return fragment.render(**names).strip()

    def compile(self, node, **kwargs):
        vertex_source = self.get_vertex_source(node, **kwargs)
        fragment_source = self.get_fragment_source(node, **kwargs)
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

    def render(self, node, references, *, srgb=False, colorbits=None, composite='over', passes=1, downsample=2, downsample_passes=(),
               border=None, repeat=None, **kwargs):
        composite = node.get('composite', 1, str, node.get('blend', 1, str, composite))
        passes = max(1, node.get('passes', 1, int, passes))
        downsample = max(1, node.get('downsample', 1, int, downsample))
        downsample_passes = set(node.get('downsample_passes', 0, int, downsample_passes))
        colorbits = node.get('colorbits', 1, int, self.glctx.extra['colorbits'] if colorbits is None else colorbits)
        self.compile(node, passes=passes, composite=composite, **kwargs)
        if self._rectangle is None:
            return
        border = node.get('border', 4, float, border)
        repeat = node.get('repeat', 2, bool, repeat)
        if border is not None:
            sampler_args = {'border_color': tuple(border)}
        elif repeat is not None:
            sampler_args = {'repeat_x': repeat[0], 'repeat_y': repeat[1]}
        else:
            sampler_args = {'border_color': (0, 0, 0, 0)}
        if node.get('nearest', 1, bool, False):
            sampler_args['filter'] = moderngl.NEAREST, moderngl.NEAREST
        child_textures = self.child_textures
        samplers = []
        self.glctx.extra['zero'].use(0)
        unit = 1
        pass_member = None
        last_member = None
        first_member = None
        size_member = None
        downsample_member = None
        self._retain_target = self.node_id is not None
        for name in self._program:
            member = self._program[name]
            if isinstance(member, moderngl.Uniform):
                if name == 'last':
                    self._retain_target = True
                    last_member = member
                    last_member.value = 0
                elif name == 'first':
                    first_member = member
                    first_member.value = 0
                elif name == 'pass':
                    pass_member = member
                elif name == 'size':
                    size_member = member
                elif name == 'downsample':
                    downsample_member = member
                elif name in child_textures:
                    texture = child_textures[name]
                    if texture is not None:
                        sampler = self.glctx.sampler(texture=child_textures[name], **sampler_args)
                        sampler.use(location=unit)
                        samplers.append(sampler)
                        member.value = unit
                        unit += 1
                    else:
                        member.value = 0
                elif name in node and set_uniform_vector(member, node[name]):
                    pass
                elif name in kwargs and set_uniform_vector(member, Vector.coerce(kwargs[name])):
                    pass
                elif name in 'alpha':
                    set_uniform_vector(member, true)
                else:
                    logger.trace("Unbound uniform: {}", name)
                    set_uniform_vector(member, false)
        self.glctx.enable(moderngl.BLEND)
        self.glctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        last = self._target
        last_sampler = None
        first_sampler = None
        first = None
        for pass_number in range(passes):
            if pass_member is not None:
                pass_member.value = pass_number
            if last_member is not None and last is not None:
                last_sampler = self.glctx.sampler(texture=last.texture, **sampler_args)
                last_sampler.use(location=unit)
                last_member.value = unit
            if first_member is not None and pass_number == 1:
                first_sampler = self.glctx.sampler(texture=first.texture, **sampler_args)
                first_sampler.use(location=unit+1)
                first_member.value = unit+1
            if pass_number in downsample_passes:
                target = RenderTarget.get(self.glctx, self.width//downsample, self.height//downsample, colorbits, srgb=srgb)
                if downsample_member is not None:
                    downsample_member.value = downsample
            else:
                target = RenderTarget.get(self.glctx, self.width, self.height, colorbits, srgb=srgb)
                if downsample_member is not None:
                    downsample_member.value = 1
            if size_member is not None:
                size_member.value = target.size
            with target:
                target.clear()
                self._rectangle.render()
            if first_member is not None and pass_number == 0:
                first = target
            if last is not None and last is not first:
                last.release()
            last = target
            if last_sampler is not None:
                last_sampler.clear()
        if first_sampler is not None:
            first_sampler.clear()
        self._target = last
        if first is not None and first is not last:
            first.release()
        for sampler in samplers:
            sampler.clear()
        self.glctx.disable(moderngl.BLEND)


class GLFWLoader:
    FUNCTIONS = {}
    ALLOW_MISSING = {'glPrimitiveRestartIndex'}

    @staticmethod
    def load_opengl_function(name):
        address = glfw.get_proc_address(name)
        if address is not None:
            return address
        if name in GLFWLoader.ALLOW_MISSING:
            return 0
        if (function := getattr(GLFWLoader, 'shim_' + name, None)) is not None:
            logger.debug("Use shim for missing GL function: {}", name)
            return ctypes.cast(function, ctypes.c_void_p).value
        if (function := GLFWLoader.FUNCTIONS.get(name)) is not None:
            return ctypes.cast(function, ctypes.c_void_p).value
        logger.trace("Request for missing GL function: {}", name)

        def missing(name):
            def function():
                logger.error("Call to missing GL function: {}", name)
                os._exit(1)
            return function

        function = GLFUNCTYPE(None)(missing(name))
        GLFWLoader.FUNCTIONS[name] = function
        return ctypes.cast(function, ctypes.c_void_p).value

    @staticmethod
    def get_function(name, *signature):
        function = GLFWLoader.FUNCTIONS.get(name)
        if function is None:
            function = ctypes.cast(glfw.get_proc_address(name), GLFUNCTYPE(*signature))
            GLFWLoader.FUNCTIONS[name] = function
        return function

    @GLFUNCTYPE(None, ctypes.c_uint, ctypes.c_uint)
    @staticmethod
    def shim_glClampColor(target, clamp):
        pass

    @GLFUNCTYPE(None, ctypes.c_double)
    @staticmethod
    def shim_glClearDepth(depth):
        glClearDepthf = GLFWLoader.get_function('glClearDepthf', None, ctypes.c_float)
        glClearDepthf(depth)

    @GLFUNCTYPE(None, ctypes.c_uint)
    @staticmethod
    def shim_glDrawBuffer(buf):
        glDrawBuffers = GLFWLoader.get_function('glDrawBuffers', None, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint))
        glDrawBuffers(1, ctypes.byref(ctypes.c_uint(buf)))

    @staticmethod
    def release():
        pass


class Window(ProgramNode):
    WINDOW_FRAGMENT_SOURCE = TemplateLoader.get_template('window.frag')
    Windows = None

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
        self._cursor = None
        self._resizable = None
        self._title = None
        self._beat = None

    def release(self):
        super().release()
        if self.glctx is not None:
            self.glctx.finish()
            RenderTarget.empty_pool(self.glctx)
            self.glctx.extra.clear()
            if count := self.glctx.gc():
                logger.trace("Window release collected {} OpenGL objects", count)
            self.glctx.release()
            self.glctx = None
        if self.window is not None:
            glfw.destroy_window(self.window)
            Window.Windows.remove(self)
            logger.debug("{} closed", self.name)
            self.window = None
        self.engine = None

    def create(self, engine, node, resized, *, opengl_es=False, **kwargs):
        super().create(engine, node, resized, opengl_es=opengl_es, **kwargs)
        self._keys = {}
        self._pointer_state = None
        new_window = False
        screen = node.get('screen', 1, int, self.default_screen)
        fullscreen = node.get('fullscreen', 1, bool, self.default_fullscreen) if self._visible else False
        resizable = node.get('resizable', 1, bool, True) if self._visible else False
        cursor = node.get('cursor', 1, bool, not fullscreen)
        title = node.get('title', 1, str, "Flitter")
        if self.window is None:
            self.engine = engine
            if Window.Windows is None:
                ok = glfw.init()
                if not ok:
                    raise RuntimeError("Unable to initialize GLFW")
                logger.debug("GLFW version: {}", glfw.get_version_string().decode('utf8'))
                Window.Windows = []
            glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API if opengl_es else glfw.NATIVE_CONTEXT_API)
            glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_ES_API if opengl_es else glfw.OPENGL_API)
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 0 if opengl_es else 3)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_ANY_PROFILE if opengl_es else glfw.OPENGL_CORE_PROFILE)
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
            glfw.window_hint(glfw.SRGB_CAPABLE, glfw.FALSE)
            glfw.window_hint(glfw.SAMPLES, 0)
            glfw.window_hint(glfw.VISIBLE, glfw.TRUE if self._visible else glfw.FALSE)
            if self._visible:
                glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE)
                glfw.window_hint(glfw.AUTO_ICONIFY, glfw.FALSE)
                glfw.window_hint(glfw.CENTER_CURSOR, glfw.FALSE)
                glfw.window_hint(glfw.SCALE_TO_MONITOR, glfw.TRUE)
            self.window = glfw.create_window(self.width, self.height, title, None, Window.Windows[0].window if Window.Windows else None)
            glfw.set_key_callback(self.window, self.key_callback)
            glfw.set_cursor_pos_callback(self.window, self.pointer_movement_callback)
            glfw.set_mouse_button_callback(self.window, self.pointer_button_callback)
            self._title = title
            Window.Windows.append(self)
            new_window = True
            if self._visible and hasattr(glfw, 'get_cocoa_window'):
                try:
                    import objc
                    import Cocoa
                    nswindow = objc.objc_object(c_void_p=ctypes.c_void_p(glfw.get_cocoa_window(self.window)))
                    nswindow.setColorSpace_(Cocoa.NSColorSpace.sRGBColorSpace())
                    logger.debug("Set macOS window to sRGB colorspace")
                except ImportError:
                    pass
                except Exception:
                    logger.exception("Failed to set macOS window to sRGB colorspace")
        if self._visible and resizable != self._resizable:
            glfw.set_window_attrib(self.window, glfw.RESIZABLE, glfw.TRUE if resizable else glfw.FALSE)
            self._resizable = resizable
        if cursor != self._cursor:
            glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_NORMAL if cursor else glfw.CURSOR_HIDDEN)
            self._cursor = cursor
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
        if title != self._title:
            glfw.set_window_title(self.window, title)
            self._title = title
        glfw.make_context_current(self.window)
        if new_window:
            if self._visible:
                logger.debug("{} opened on screen {}", self.name, screen)
            else:
                logger.debug("{} opened off-screen", self.name)
            self.glctx = moderngl.create_context(require=300 if opengl_es else 330, context=GLFWLoader)
            self.glctx.gc_mode = 'context_gc'
            logger.debug("OpenGL info: {GL_RENDERER} {GL_VERSION}", **self.glctx.info)
            logger.trace("{!r}", self.glctx.info)
            zero = self.glctx.texture((1, 1), 4, dtype='f1')
            zero.write(bytes([0, 0, 0, 0]))
            if opengl_es:
                header = "#version 300 es\nprecision highp float;\nprecision highp int;\n"
            else:
                header = "#version 330\n"
            self.glctx.extra = {'zero': zero, 'HEADER': header}
            vertex_source = self.DEFAULT_VERTEX_SOURCE.render(HEADER=header)
            fragment_source = self.WINDOW_FRAGMENT_SOURCE.render(HEADER=header)
            self._window_program = self.glctx.program(vertex_shader=vertex_source, fragment_shader=fragment_source)
            vertices = self.glctx.buffer(array.array('f', [-1, 1, -1, -1, 1, 1, 1, -1]))
            self._window_rectangle = self.glctx.vertex_array(self._window_program, [(vertices, '2f', 'position')], mode=moderngl.TRIANGLE_STRIP)
        if self._visible:
            self.recalculate_viewport(new_window)
        colorbits = node.get('colorbits', 1, int, DEFAULT_COLORBITS)
        if colorbits not in COLOR_FORMATS:
            colorbits = DEFAULT_COLORBITS
        self.glctx.extra['colorbits'] = colorbits

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

    def render(self, node, references, beat=None, **kwargs):
        if self._visible or self.node_id is not None:
            super().render(node, references, **kwargs)
        if self._visible:
            self.glctx.screen.use()
            self.glctx.screen.clear(0, 0, 0, 1)
            self._target.texture.use(1)
            self._window_program['target'] = 1
            self._window_rectangle.render()
            vsync = node.get('vsync', 1, bool, self.default_vsync)
            glfw.swap_interval(1 if vsync else 0)
            glfw.swap_buffers(self.window)
        else:
            self.glctx.finish()
        self._beat = beat
        glfw.poll_events()
        RenderTarget.empty_pool(self.glctx, 15)
        self.glctx.gc()

    def make_secondary_texture(self):
        width, height = self.window.get_framebuffer_size()
        return self.glctx.texture((width, height), 4)


class Offscreen(Window):
    def __init__(self, offscreen=False, **kwargs):
        super().__init__(offscreen=True, **kwargs)
