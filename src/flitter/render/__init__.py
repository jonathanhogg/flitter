"""
Flitter generic rendering
"""

import importlib


def get_renderer(kind):
    try:
        module = importlib.import_module(f'flitter.render.{kind}')
    except ImportError:
        return None
    return module.RENDERER_CLASS
