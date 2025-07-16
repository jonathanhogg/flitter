"""
Flitter built-in filter shaders
"""

from . import ProgramNode
from .glsl import TemplateLoader
from ...model import null, Matrix33


class Shader(ProgramNode):
    pass


class Transform(Shader):
    DEFAULT_FRAGMENT_SOURCE = TemplateLoader.get_template('transform.frag')

    def render(self, node, references, **kwargs):
        transform_matrix = Matrix33.identity()
        for name, value in node.items():
            if name == 'scale' and (matrix := Matrix33.scale(value)) is not None:
                transform_matrix @= matrix
            elif name == 'translate' and (matrix := Matrix33.translate(value)) is not None:
                transform_matrix @= matrix
            elif name == 'rotate' and (matrix := Matrix33.rotate(value)) is not None:
                transform_matrix @= matrix
            elif name == 'keystone':
                key_x, key_y = value.match(2, float, (0, 0))
                matrix = Matrix33([1, 0, key_x/self.width, 0, 1, key_y/self.height, 0, 0, 1])
                transform_matrix @= matrix
        super().render(node, references, transform_matrix=transform_matrix.inverse(), **kwargs)


class Vignette(Shader):
    DEFAULT_FRAGMENT_SOURCE = TemplateLoader.get_template('vignette.frag')
    EASE_FUNCTIONS = {'linear', 'quad', 'cubic'}

    def render(self, node, references, **kwargs):
        ease = node.get('fade', 1, str)
        if ease not in self.EASE_FUNCTIONS:
            ease = 'linear'
        super().render(node, references, inset=0.25, ease=ease, **kwargs)


class Adjust(Shader):
    DEFAULT_FRAGMENT_SOURCE = TemplateLoader.get_template('adjust.frag')
    TONEMAP_FUNCTIONS = {'reinhard', 'aces', 'agx', 'agx_punchy'}

    def render(self, node, references, **kwargs):
        tonemap = node.get('tonemap', 1, str)
        if tonemap not in self.TONEMAP_FUNCTIONS:
            tonemap = None
        super().render(node, references, exposure=0, contrast=1, brightness=0, shadows=0, highlights=0,
                       hue=0, saturation=1, color_matrix=(1, 0, 0, 0, 1, 0, 0, 0, 1),
                       gamma=1, tonemap_function=tonemap, whitepoint=0, **kwargs)


class Blur(Shader):
    DEFAULT_FRAGMENT_SOURCE = TemplateLoader.get_template('blur.frag')

    def render(self, node, references, **kwargs):
        child_textures = self.child_textures
        if len(child_textures) > 1:
            passes = 3
        elif child_textures:
            passes = 2
        else:
            passes = 1
        super().render(node, references, passes=passes, radius=0, sigma=0.3, repeat=(False, False), **kwargs)


class Bloom(Shader):
    DEFAULT_FRAGMENT_SOURCE = TemplateLoader.get_template('bloom.frag')

    def render(self, node, references, **kwargs):
        child_textures = self.child_textures
        if len(child_textures) > 1:
            passes = 5
            downsample_passes = {1, 2, 3}
        elif child_textures:
            passes = 4
            downsample_passes = {0, 1, 2}
        else:
            passes = 1
            downsample_passes = ()
        super().render(node, references, passes=passes, downsample_passes=downsample_passes,
                       radius=0, sigma=0.3, exposure=-1, contrast=1, brightness=0, shadows=0, highlights=0,
                       repeat=(False, False), **kwargs)


class Edges(Shader):
    DEFAULT_FRAGMENT_SOURCE = TemplateLoader.get_template('edges.frag')

    def render(self, node, references, **kwargs):
        child_textures = self.child_textures
        if len(child_textures) > 1:
            passes = 4
            downsample_passes = {1, 2}
        elif child_textures:
            passes = 3
            downsample_passes = {0, 1}
        else:
            passes = 1
            downsample_passes = ()
        super().render(node, references, passes=passes, downsample_passes=downsample_passes,
                       radius=0, sigma=0.3, mixer=0.0, repeat=(False, False), **kwargs)


class Feedback(Shader):
    DEFAULT_FRAGMENT_SOURCE = TemplateLoader.get_template('feedback.frag')

    def render(self, node, references, delta=0, **kwargs):
        if self._target is None:
            delta = 0
        super().render(node, references, delta=delta, mixer=0, timebase=1, glow=0, translate=0, scale=1, rotate=0, repeat=(False, False), **kwargs)


class Flare(Shader):
    DEFAULT_FRAGMENT_SOURCE = TemplateLoader.get_template('flare.frag')

    def render(self, node, references, **kwargs):
        child_textures = self.child_textures
        if len(child_textures) > 1:
            passes = 5
            downsample_passes = {1, 2, 3}
        elif child_textures:
            passes = 4
            downsample_passes = {0, 1, 2}
        else:
            passes = 1
            downsample_passes = ()
        super().render(node, references, passes=passes, downsample_passes=downsample_passes,
                       upright_length=1/4, diagonal_length=1/8, ghosts=6, threshold=1, attenuation=2, aberration=1,
                       halo_radius=1/16, halo_attenuation=0, **kwargs)


class Noise(Shader):
    DEFAULT_FRAGMENT_SOURCE = TemplateLoader.get_template('noise.frag')

    def render(self, node, references, **kwargs):
        seed_hash = hash(node['seed'] if 'seed' in node else null) / (1 << 48)
        default_values = node.get('default', 4, float, (1, 1, 1, 1))
        super().render(node, references, seed_hash=seed_hash, components=1, octaves=1, roughness=0.5, origin=0, z=0,
                       scale=1, tscale=1, multiplier=0.5, offset=0.5, default_values=default_values, **kwargs)
