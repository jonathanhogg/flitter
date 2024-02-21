"""
Flitter offscreen OpenGL rendering

This is actually all managed within the `window` package, but we need a
module at the top-level of `flitter.render` to ensure that the auto-import
and renderer searching logic works.
"""

from .window import Window


def Offscreen(**kwargs):
    kwargs['offscreen'] = True
    return Window(**kwargs)


RENDERER_CLASS = Offscreen
