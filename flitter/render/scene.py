"""
Flitter window management
"""

# pylama:ignore=C0413,E402,W0703,R0914,R0902,R0912,R0201,R1702,C901,W0223,W0231,R0915

import array
import logging
import sys

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


class SceneNode:
    def __init__(self, glctx):
        self.glctx = glctx
        self.children = []
        self.width = None
        self.height = None

    @property
    def texture(self):
        raise NotImplementedError()

    def destroy(self):
        self.release()
        self.glctx = None
        self.purge()

    def purge(self):
        while self.children:
            self.children.pop().destroy()

    def update(self, node, **kwargs):
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
        self.descend(node, **kwargs)
        self.render(node, **kwargs)

    def descend(self, node, **kwargs):
        count = 0
        for i, child in enumerate(node.children):
            cls = SCENE_CLASSES[child.kind]
            if i == len(self.children):
                self.children.append(cls(self.glctx))
            elif type(self.children[i]) != cls:  # noqa
                self.children[i].destroy()
                self.children[i] = cls(self.glctx)
            self.children[i].update(child, **kwargs)
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

    def update(self, node, references=None, **kwargs):
        node_id = node['id'].as_string() if 'id' in node else None
        self._reference = references.get(node_id) if references is not None and node_id else None

    def destroy(self):
        self._reference = None


class ProgramNode(SceneNode):
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

    def create(self, node, resized, **kwargs):
        super().create(node, resized, **kwargs)
        if resized and self._last is not None:
            self._last.release()
            self._last = None

    def get_vertex_source(self, _):
        return """#version 410
in vec2 position;
out vec2 coord;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    coord = (position + 1.0) / 2.0;
}
"""

    def get_fragment_source(self, _):
        children = [child for child in self.children if child.texture is not None]
        samplers = '\n'.join(f"uniform sampler2D texture{i};" for i in range(len(children)))
        textures = '\n'.join(f"""    merge = texture(texture{i}, coord);
    color = color * (1.0 - merge.a) + merge;""" for i in range(len(children)))
        return f"""#version 410
precision highp float;
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
            children = [child for child in self.children if child.texture is not None]
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
                    elif name.startswith('texture'):
                        index = int(name[7:])
                        if index < len(children):
                            child = children[index]
                            if sampler_args:
                                sampler = self.glctx.sampler(texture=child.texture, **sampler_args)
                                sampler.use(location=unit)
                                samplers.append(sampler)
                            else:
                                child.texture.use(location=unit)
                            member.value = unit
                            unit += 1
                    elif name in kwargs:
                        member.value = kwargs[name]
                    elif name in node:
                        value = node.get(name, member.dimension, float)
                        if value is not None:
                            member.value = value if member.dimension == 1 else tuple(value)
            self.framebuffer.clear()
            self._rectangle.render(mode=moderngl.TRIANGLE_STRIP)
            for sampler in samplers:
                sampler.clear()
                sampler.release()


class Window(ProgramNode):
    GL_VERSION = (4, 1)

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
        else:
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

    def on_close(self):
        self._closed = True

    def render(self, node, **kwargs):
        if self._closed:
            raise ValueError("Window closed")
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

    def get_vertex_source(self, node):
        return node['vertex'].as_string() if 'vertex' in node else super().get_vertex_source(node)

    def get_fragment_source(self, node):
        return node['fragment'].as_string() if 'fragment' in node else super().get_fragment_source(node)


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

    def descend(self, node, **kwargs):
        # A canvas is a leaf node from the perspective of the OpenGL world
        pass

    def render(self, node, **kwargs):
        self._graphics_context.resetContext()
        self._framebuffer.clear()
        canvas.draw(node, self._canvas)
        self._surface.flushAndSubmit()


SCENE_CLASSES = {'reference': Reference, 'shader': Shader, 'canvas': Canvas}
