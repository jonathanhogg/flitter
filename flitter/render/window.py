"""
Flitter window management
"""

import array

import moderngl
import pyglet.canvas
import pyglet.window
from pyglet import gl

from . import canvas


GL_VERSION = (4, 1)

VERTEX_SHADER = """
#version %s0
in vec2 position;
out vec2 coord;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    coord = (position + 1.0) / 2.0;
}
""" % ''.join(str(v) for v in GL_VERSION)

FRAGMENT_SHADER = """
#version %s0
uniform sampler2D image;
in vec2 coord;
out vec4 color;
void main() {
    color = texture(image, coord);
}
""" % ''.join(str(v) for v in GL_VERSION)


class WindowWrapper(pyglet.window.Window):
    """Disable some pyglet functionality that is broken on moderngl"""
    def on_resize(self, width, height):
        pass

    def on_draw(self):
        pass


class Window:
    def __init__(self):
        self.window = None
        self.children = []

    def create_window(self, node):
        vsync = node.get('vsync', 1, bool, True)
        fullscreen = node.get('fullscreen', 1, bool, False)
        screen = node.get('screen', 1, int, 0)
        screens = pyglet.canvas.get_display().get_screens()
        screen = screens[screen] if screen < len(screens) else screens[0]
        config = gl.Config(major_version=GL_VERSION[0], minor_version=GL_VERSION[1], forward_compatible=True,
                           depth_size=24, double_buffer=True, sample_buffers=1, samples=0)
        if fullscreen:
            self.window = WindowWrapper(fullscreen=True, screen=screen, vsync=vsync, config=config)
        else:
            width, height = node.get('size', 2, int, (1920, 1080))
            self.window = WindowWrapper(width=width, height=height, resizable=False, screen=screen, vsync=vsync, config=config)
        self.glctx = moderngl.create_context(require=GL_VERSION[0] * 100 + GL_VERSION[1])
        self.prog = self.glctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        vertices = self.glctx.buffer(array.array('f', [-1, 1, -1, -1, 1, 1, 1, -1]))
        self.screen_rectangle = self.glctx.vertex_array(self.prog, [(vertices, '2f', 'position')])

    def render(self, node):
        assert node.kind == 'window'
        if self.window is None:
            self.create_window(node)
        for i, child in enumerate(node.children):
            if i == len(self.children):
                self.children.append(None)
            match child.kind:
                case "canvas":
                    if not isinstance(self.children[i], canvas.Canvas):
                        self.children[i] = canvas.Canvas(self.glctx)
                    self.children[i].render(child)
        while len(self.children) > i+1:
            self.children.pop().destroy()

        self.glctx.screen.use()
        self.window.clear()
        width, height = self.window.get_size()
        for child in self.children:
            match child:
                case canvas.Canvas():
                    child.texture.use(location=0)
                    self.screen_rectangle.render(mode=moderngl.TRIANGLE_STRIP)
        self.window.flip()
        self.window.dispatch_events()
