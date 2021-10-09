"""
Flitter window management
"""

# pylama:ignore=C0413,E402,W0703,R0914,R0902

import array
import logging

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
        self.screen_rectangle = None

    @property
    def texture(self):
        return None

    def create(self, node):
        if self.window is None:
            vsync = node.get('vsync', 1, bool, True)
            fullscreen = node.get('fullscreen', 1, bool, False)
            screen = node.get('screen', 1, int, 0)
            title = node.get('title', 1, str, "Flitter")
            screens = pyglet.canvas.get_display().get_screens()
            screen = screens[screen] if screen < len(screens) else screens[0]
            config = pyglet.gl.Config(major_version=self.GL_VERSION[0], minor_version=self.GL_VERSION[1], forward_compatible=True,
                                      depth_size=24, double_buffer=True, sample_buffers=1, samples=0)
            if fullscreen:
                self.window = self.WindowWrapper(fullscreen=True, caption=title, screen=screen, vsync=vsync, config=config)
            else:
                width, height = node.get('size', 2, int, (1920, 1080))
                self.window = self.WindowWrapper(width=width, height=height, resizable=False, caption=title, screen=screen, vsync=vsync, config=config)
            self.glctx = moderngl.create_context(require=self.GL_VERSION[0] * 100 + self.GL_VERSION[1])
            program = self.glctx.program(vertex_shader=self.VERTEX_SHADER, fragment_shader=self.FRAGMENT_SHADER)
            vertices = self.glctx.buffer(array.array('f', [-1, 1, -1, -1, 1, 1, 1, -1]))
            self.screen_rectangle = self.glctx.vertex_array(program, [(vertices, '2f', 'position')])

    def render(self, node):
        self.glctx.screen.use()
        self.window.clear()
        for child in self.children:
            child.texture.use(location=0)
            self.screen_rectangle.render(mode=moderngl.TRIANGLE_STRIP)
        self.window.flip()
        self.window.dispatch_events()

    def release(self):
        self.window.close()


class Shader(SceneNode):
    VERTEX_PREAMBLE = """
    #version 410
    in vec2 position;
    out vec2 coord;
    """
    DEFAULT_VERTEX_SHADER = """
    void main() {
        gl_Position = vec4(position, 0.0, 1.0);
        coord = (position + 1.0) / 2.0;
    }
    """
    FRAGMENT_PREAMBLE = """
    #version 410
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

    def create(self, node):
        width, height = node.get('size', 2, int, (self.width, self.height))
        if self._texture is None or width != self.width or height != self.height:
            self.width = width
            self.height = height
            if self._texture is not None:
                self._framebuffer.release()
                self._texture.release()
            self._texture = self.glctx.texture((self.width, self.height), 4)
            self._framebuffer = self.glctx.framebuffer(color_attachments=(self._texture,))

    def render(self, node):
        uniforms = {}
        uniform_declarations = ''
        for i, _ in enumerate(self.children):
            uniform_declarations += f'uniform sampler2D texture{i};\n'
        for name in node:
            if name not in {'vertex', 'fragment', 'repeat'} and not name.startswith('_'):
                vector = node[name]
                size = len(vector)
                if 1 <= size <= 4 and vector.isinstance((float, int)):
                    uniforms[name] = vector[0] if size == 1 else tuple(vector)
                    uniform_type = 'float' if size == 1 else f'vec{size}'
                    uniform_declarations += f'uniform {uniform_type} {name};\n'
        vertex_source = self.VERTEX_PREAMBLE + node.get('vertex', 1, str, self.DEFAULT_VERTEX_SHADER)
        fragment_source = self.FRAGMENT_PREAMBLE + uniform_declarations + node.get('fragment', 1, str, self.DEFAULT_FRAGMENT_SHADER)
        if vertex_source != self.vertex_source or fragment_source != self.fragment_source:
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
        self._framebuffer.use()
        self._framebuffer.clear()
        if self._rectangle is not None:
            for name, value in uniforms.items():
                self._program[name] = value
            repeat = node.get('repeat', 1, bool, False)
            samplers = []
            for i, child in enumerate(self.children):
                sampler = self.glctx.sampler(repeat_x=repeat, repeat_y=repeat, texture=child.texture)
                sampler.use(location=i)
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
