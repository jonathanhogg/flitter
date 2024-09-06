"""
Flitter built-in filter shaders
"""

from . import ProgramNode
from .glsl import TemplateLoader
from ...model import null


class Transform(ProgramNode):
    DEFAULT_FRAGMENT_SOURCE = TemplateLoader.get_template('transform.frag')

    def render(self, node, references, **kwargs):
        super().render(node, references, scale=1, translate=0, rotate=0, **kwargs)


class Vignette(ProgramNode):
    DEFAULT_FRAGMENT_SOURCE = TemplateLoader.get_template('vignette.frag')
    EASE_FUNCTIONS = {'linear', 'quad', 'cubic'}

    def render(self, node, references, **kwargs):
        ease = node.get('fade', 1, str)
        if ease not in self.EASE_FUNCTIONS:
            ease = 'linear'
        super().render(node, references, inset=0.25, ease=ease, **kwargs)


class Adjust(ProgramNode):
    DEFAULT_FRAGMENT_SOURCE = TemplateLoader.get_template('adjust.frag')
    TONEMAP_FUNCTIONS = {'reinhard', 'aces'}

    def render(self, node, references, **kwargs):
        tonemap = node.get('tonemap', 1, str)
        if tonemap not in self.TONEMAP_FUNCTIONS:
            tonemap = None
        super().render(node, references, exposure=0, contrast=1, brightness=0, shadows=0, highlights=0,
                       color_matrix=(1, 0, 0, 0, 1, 0, 0, 0, 1),
                       gamma=1, tonemap_function=tonemap, **kwargs)


class Blur(ProgramNode):
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


class Bloom(ProgramNode):
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


class Edges(ProgramNode):
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
                       radius=0, sigma=0.3, repeat=(False, False), **kwargs)


class Feedback(ProgramNode):
    DEFAULT_FRAGMENT_SOURCE = TemplateLoader.get_template('feedback.frag')

    def render(self, node, references, **kwargs):
        super().render(node, references, mixer=0, timebase=1, glow=0, translate=0, scale=1, rotate=0, repeat=(False, False), **kwargs)


class Flare(ProgramNode):
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
                       upright_length=0.25, diagonal_length=0.125, ghosts=6, threshold=1, attenuation=2, aberration=1, **kwargs)


class Noise(ProgramNode):
    DEFAULT_FRAGMENT_SOURCE = TemplateLoader.get_template('noise.frag')

    def render(self, node, references, **kwargs):
        seed_hash = hash(node['seed'] if 'seed' in node else null) / (1 << 48)
        super().render(node, references, seed_hash=seed_hash, components=1, octaves=1, roughness=0.5, origin=0, z=0, scale=1, tscale=1, **kwargs)
