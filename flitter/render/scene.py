"""
Flitter window management
"""

# pylama:ignore=C0413,E402,W0703,R0914,R0902

import array
import logging
import time

import cairo
import moderngl
import numpy as np

import pyglet
pyglet.options['shadow_window'] = False

import pyglet.canvas
import pyglet.window
import pyglet.gl

from . import canvas


Log = logging.getLogger(__name__)


class SceneNode:
    def __init__(self, glctx):
        self.glctx = glctx
        self.children = []

    @property
    def texture(self):
        raise NotImplementedError()

    def destroy(self):
        self.release()
        self.glctx = None
        while self.children:
            self.children.pop().destroy()

    def update(self, node):
        self.create(node)
        self.descend(node)
        self.render(node)

    def descend(self, node):
        count = 0
        for i, child in enumerate(node.children):
            cls = {'shader': Shader, 'canvas': Canvas}[child.kind]
            if i == len(self.children):
                self.children.append(cls(self.glctx))
            elif not isinstance(self.children[i], cls):
                self.children[i].destroy()
                self.children[i] = cls(self.glctx)
            self.children[i].update(child)
            count += 1
        while len(self.children) > count:
            self.children.pop().destroy()

    def create(self, node):
        raise NotImplementedError()

    def render(self, node):
        raise NotImplementedError()

    def release(self):
        raise NotImplementedError()


class Window(SceneNode):
    GL_VERSION = (4, 1)
    VERTEX_SHADER = """
    #version 410
    in vec2 position;
    out vec2 coord;
    void main() {
        gl_Position = vec4(position, 0.0, 1.0);
        coord = (position + 1.0) / 2.0;
    }
    """
    FRAGMENT_SHADER = """
    #version 410
    uniform sampler2D image;
    in vec2 coord;
    out vec4 color;
    void main() {
        color = texture(image, coord);
    }
    """

    class WindowWrapper(pyglet.window.Window):  # noqa
        """Disable some pyglet functionality that is broken on moderngl"""
        def on_resize(self, width, height):
            pass

        def on_draw(self):
            pass

    def __init__(self):
        super().__init__(None)
        self.window = None
        self.program = None
        self.fullscreen = None
        self.width = None
        self.height = None
        self.screen_rectangle = None

    @property
    def texture(self):
        return None

    def create(self, node):
        if self.window is None:
            vsync = node.get('vsync', 1, bool, True)
            self.fullscreen = node.get('fullscreen', 1, bool, False)
            screen = node.get('screen', 1, int, 0)
            title = node.get('title', 1, str, "Flitter")
            screens = pyglet.canvas.get_display().get_screens()
            screen = screens[screen] if screen < len(screens) else screens[0]
            config = pyglet.gl.Config(major_version=self.GL_VERSION[0], minor_version=self.GL_VERSION[1], forward_compatible=True,
                                      depth_size=24, double_buffer=True, sample_buffers=1, samples=0)
            self.width, self.height = node.get('size', 2, int, (1920, 1080))
            if self.fullscreen:
                self.window = self.WindowWrapper(fullscreen=True, caption=title, screen=screen, vsync=vsync, config=config)
            else:
                self.window = self.WindowWrapper(width=self.width, height=self.height, resizable=False, caption=title, screen=screen, vsync=vsync, config=config)
            self.glctx = moderngl.create_context(require=self.GL_VERSION[0] * 100 + self.GL_VERSION[1])
            self.program = self.glctx.program(vertex_shader=self.VERTEX_SHADER, fragment_shader=self.FRAGMENT_SHADER)
            vertices = self.glctx.buffer(array.array('f', [-1, 1, -1, -1, 1, 1, 1, -1]))
            self.screen_rectangle = self.glctx.vertex_array(self.program, [(vertices, '2f', 'position')])
            print(self.glctx.screen.width, self.glctx.screen.height)
        else:
            fullscreen = node.get('fullscreen', 1, bool, False)
            width, height = node.get('size', 2, int, (1920, 1080))
            if width != self.width or height != self.height:
                if not self.fullscreen:
                    self.window.set_size(self.width, self.height)
                self.width = width
                self.height = height
            if fullscreen != self.fullscreen:
                self.window.set_fullscreen(fullscreen)
                if self.fullscreen:
                    self.window.set_size(self.width, self.height)
                self.fullscreen = fullscreen

    def render(self, node):
        self.glctx.screen.use()
        if self.fullscreen:
            aspect_ratio = self.width / self.height
            width, height = self.glctx.screen.width, self.glctx.screen.height
            if width / height > aspect_ratio:
                view_width = int(height * aspect_ratio)
                self.glctx.screen.viewport = ((width - view_width) // 2, 0, view_width, height)
            else:
                view_height = int(width / aspect_ratio)
                self.glctx.screen.viewport = (0, (height - view_height) // 2, width, view_height)
        self.window.clear()
        for child in self.children:
            child.texture.use(location=0)
            self.screen_rectangle.render(mode=moderngl.TRIANGLE_STRIP)
        self.window.flip()
        self.window.dispatch_events()

    def release(self):
        self.screen_rectangle.release()
        self.program.release()
        self.window.close()


class Shader(SceneNode):
    VERTEX_PREAMBLE = """#version 410
in vec2 position;
out vec2 coord;

"""
    DEFAULT_VERTEX_SHADER = """
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    coord = (position + 1.0) / 2.0;
}
"""
    FRAGMENT_PREAMBLE = """#version 410
precision highp float;
in vec2 coord;
out vec4 color;

"""
    DEFAULT_FRAGMENT_SHADER = """
void main() {
    color = vec4(1.0, 0.0, 0.0, 1.0);
}
"""

    def __init__(self, glctx):
        super().__init__(glctx)
        self.width = None
        self.height = None
        self.vertex_source = None
        self.fragment_source = None
        self._framebuffer = None
        self._texture = None
        self._program = None
        self._rectangle = None
        self._last = None
        self._timestamp = None

    @property
    def texture(self):
        return self._texture

    def release(self):
        if self._renderbuffer is not None:
            self._framebuffer.release()
            self._framebuffer = None
            self._texture.release()
            self._texture = None
            self._program.release()
        if self._rectangle is not None:
            self._program = None
            self._rectangle.release()
            self._rectangle = None
        if self._last is not None:
            self._last.release()
            self._last = None

    def create(self, node):
        width, height = node.get('size', 2, int, (self.width, self.height))
        if self._texture is None or width != self.width or height != self.height:
            self.width = width
            self.height = height
            if self._texture is not None:
                self._framebuffer.release()
                self._texture.release()
            if self._last is not None:
                self._last.release()
                self._last = None
            self._texture = self.glctx.texture((self.width, self.height), 4)
            self._framebuffer = self.glctx.framebuffer(color_attachments=(self._texture,))
            self._framebuffer.clear()
            self._timestamp = None

    def compile(self, vertex_source, fragment_source):
        self.vertex_source = vertex_source
        self.fragment_source = fragment_source
        if self._program is not None:
            self._program.release()
            self._program = None
        if self._rectangle is not None:
            self._rectangle.release()
            self._rectangle = None
        try:
            self._program = self.glctx.program(vertex_shader=self.vertex_source, fragment_shader=self.fragment_source)
            vertices = self.glctx.buffer(array.array('f', [-1, 1, -1, -1, 1, 1, 1, -1]))
            self._rectangle = self.glctx.vertex_array(self._program, [(vertices, '2f', 'position')])
        except Exception:
            Log.exception("Unable to compile shader")
            print(self.fragment_source)

    def render(self, node):
        now = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
        uniforms = {'delta': 0.0 if self._timestamp is None else now - self._timestamp}
        self._timestamp = now
        uniform_declarations = 'uniform float delta;\n'
        last = node.get('last', 1, bool, False)
        if last:
            uniform_declarations += 'uniform sampler2D last;\n'
        for i, _ in enumerate(self.children):
            uniform_declarations += f'uniform sampler2D texture{i};\n'
        for name in node:
            if name not in {'vertex', 'fragment', 'repeat', 'last'} and not name.startswith('_'):
                vector = node[name]
                size = len(vector)
                if 1 <= size <= 4 and vector.isinstance((float, int)):
                    uniforms[name] = vector[0] if size == 1 else tuple(vector)
                    uniform_type = 'float' if size == 1 else f'vec{size}'
                    uniform_declarations += f'uniform {uniform_type} {name};\n'
        vertex_source = self.VERTEX_PREAMBLE + node.get('vertex', 1, str, self.DEFAULT_VERTEX_SHADER)
        fragment_source = self.FRAGMENT_PREAMBLE + uniform_declarations + node.get('fragment', 1, str, self.DEFAULT_FRAGMENT_SHADER)
        if vertex_source != self.vertex_source or fragment_source != self.fragment_source:
            self.compile(vertex_source, fragment_source)
        if self._rectangle is not None:
            textures = {}
            if last:
                if self._last is None:
                    self._last = self.glctx.texture((self.width, self.height), 4)
                self.glctx.copy_framebuffer(self._last, self._framebuffer)
                textures['last'] = self._last
            self._framebuffer.use()
            self._framebuffer.clear()
            for i, child in enumerate(self.children):
                textures[f'texture{i}'] = child.texture
            for name, value in uniforms.items():
                if name in self._program:
                    self._program[name] = value
            repeat = node.get('repeat', 1, bool, False)
            samplers = []
            unit = 0
            for name, texture in textures.items():
                if name in self._program:
                    sampler = self.glctx.sampler(repeat_x=repeat, repeat_y=repeat, texture=texture)
                    sampler.use(location=unit)
                    self._program[name] = unit
                    unit += 1
                    samplers.append(sampler)
            self._rectangle.render(mode=moderngl.TRIANGLE_STRIP)
            for sampler in samplers:
                sampler.clear()
                sampler.release()


class Canvas(SceneNode):
    def __init__(self, glctx):
        super().__init__(glctx)
        self._surface = None
        self._array = None
        self._texture = None
        self.width = None
        self.height = None

    @property
    def texture(self):
        return self._texture

    def release(self):
        if self._texture is not None:
            self._texture.release()
            self._texture = None
            self._array = None
            self._surface.finish()
            self._surface = None

    def create(self, node):
        width, height = node.get('size', 2, int, (self.width, self.height))
        if self._texture is None or width != self.width or height != self.height:
            self.width = width
            self.height = height
            if self._surface is not None:
                self._array = None
                self._surface.finish()
            self._surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.width, self.height)
            self._array = np.ndarray(buffer=self._surface.get_data(), shape=(self.height, self.width), dtype='uint32')
            if self._texture is not None:
                self._texture.release()
            self._texture = self.glctx.texture((self.width, self.height), 4)
            self._texture.swizzle = 'BGRA'
        else:
            self._array[:, :] = 0

    def descend(self, node):
        # A canvas is a leaf node from the perspective of the OpenGL world
        pass

    def render(self, node):
        if self._texture is not None:
            ctx = cairo.Context(self._surface)
            # OpenGL and Cairo worlds are upside-down vs each other
            ctx.translate(0, self.height)
            ctx.scale(1, -1)
            canvas.draw(node, ctx)
            self.texture.write(self._surface.get_data())
