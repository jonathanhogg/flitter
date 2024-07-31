"""
Flitter built-in filter shaders
"""

from loguru import logger

from . import ProgramNode, COLOR_FORMATS
from .glsl import TemplateLoader
from ...model import null


class Shader(ProgramNode):
    def __init__(self, glctx):
        super().__init__(glctx)
        self._colorbits = None
        self._framebuffer = None
        self._texture = None

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
            self._colorbits = colorbits
            logger.debug("Create {}x{} {}-bit framebuffer for {}", self.width, self.height, self._colorbits, self.name)
            self._texture = self.glctx.texture((self.width, self.height), 4, dtype=COLOR_FORMATS[colorbits].moderngl_dtype)
            self._framebuffer = self.glctx.framebuffer(color_attachments=(self._texture,))
            self._first = None
            self._last = None

    def make_secondary_texture(self):
        logger.debug("Create {}x{} {}-bit secondary texture for {}", self.width, self.height, self._colorbits, self.name)
        return self.glctx.texture((self.width, self.height), 4, dtype=COLOR_FORMATS[self._colorbits].moderngl_dtype)


class Transform(Shader):
    DEFAULT_FRAGMENT_SOURCE = TemplateLoader.get_template('transform.frag')

    def render(self, node, **kwargs):
        super().render(node, scale=1, translate=0, rotate=0, **kwargs)


class Vignette(Shader):
    DEFAULT_FRAGMENT_SOURCE = TemplateLoader.get_template('vignette.frag')
    EASE_FUNCTIONS = {'linear', 'quad', 'cubic'}

    def render(self, node, **kwargs):
        ease = node.get('fade', 1, str)
        if ease not in self.EASE_FUNCTIONS:
            ease = 'linear'
        super().render(node, inset=0.25, ease=ease, **kwargs)


class Adjust(Shader):
    DEFAULT_FRAGMENT_SOURCE = TemplateLoader.get_template('adjust.frag')

    def render(self, node, **kwargs):
        super().render(node, exposure=0, contrast=1, brightness=0, **kwargs)


class Blur(Shader):
    DEFAULT_FRAGMENT_SOURCE = TemplateLoader.get_template('blur.frag')

    def render(self, node, **kwargs):
        child_textures = self.child_textures
        if len(child_textures) > 1:
            passes = 3
        elif child_textures:
            passes = 2
        else:
            passes = 1
        super().render(node, passes=passes, radius=0, sigma=0.3, repeat=(False, False), **kwargs)


class Bloom(Shader):
    DEFAULT_FRAGMENT_SOURCE = TemplateLoader.get_template('bloom.frag')

    def render(self, node, **kwargs):
        child_textures = self.child_textures
        if len(child_textures) > 1:
            passes = 5
        elif child_textures:
            passes = 4
        else:
            passes = 1
        super().render(node, passes=passes, radius=0, sigma=0.3, exposure=-1, contrast=1, brightness=0, repeat=(False, False), **kwargs)


class Edges(Shader):
    DEFAULT_FRAGMENT_SOURCE = TemplateLoader.get_template('edges.frag')

    def render(self, node, **kwargs):
        child_textures = self.child_textures
        if len(child_textures) > 1:
            passes = 4
        elif child_textures:
            passes = 3
        else:
            passes = 1
        super().render(node, passes=passes, radius=0, sigma=0.3, repeat=(False, False), **kwargs)


class Feedback(Shader):
    DEFAULT_FRAGMENT_SOURCE = TemplateLoader.get_template('feedback.frag')

    def render(self, node, **kwargs):
        super().render(node, mixer=0, timebase=1, glow=0, translate=0, scale=1, rotate=0, repeat=(False, False), **kwargs)


class Noise(Shader):
    DEFAULT_FRAGMENT_SOURCE = TemplateLoader.get_template('noise.frag')

    def render(self, node, **kwargs):
        seed_hash = hash(node['seed'] if 'seed' in node else null) / (1 << 48)
        super().render(node, seed_hash=seed_hash, components=1, octaves=1, roughness=0.5, origin=0, z=0, scale=1, tscale=1, **kwargs)
