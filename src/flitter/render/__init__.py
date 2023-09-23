"""
Flitter generic rendering
"""

import importlib

from loguru import logger


_class_cache = {}


class Renderer:
    """
    Renderers should implement this API
    """
    def __init__(self, **kwargs):
        """
        `kwargs` will contain the engine global definitions. You will likely
        want to create most of the renderer state on first call of `update()`
        when you have access to the top-level node attributes.
        """
        pass

    def purge(self):
        """
        `purge()`` should wipe any local state ready for a complete redefinition.
        """
        raise NotImplementedError()

    def destroy(self):
        """
        `destroy()`` should tear down the renderer itself.
        """
        raise NotImplementedError()

    async def update(self, engine, node, **kwargs):
        """
        `update()`` is called every frame with the node to render and the engine
        global definitions (including `clock` and `beat`).
        """
        raise NotImplementedError()


def get_renderer(kind):
    if kind in _class_cache:
        return _class_cache[kind]
    try:
        module = importlib.import_module(f'.{kind}', __package__)
        cls = module.RENDERER_CLASS
    except ModuleNotFoundError:
        cls = None
    except ImportError:
        logger.exception("Import error")
        cls = None
    _class_cache[kind] = cls
    return cls
