"""
Flitter generic interaction
"""

import importlib

from loguru import logger


_class_cache = {}


def get_interactor(kind):
    if kind in _class_cache:
        return _class_cache[kind]
    try:
        module = importlib.import_module(f'.{kind}', __package__)
        cls = module.INTERACTOR_CLASS
    except ModuleNotFoundError:
        cls = None
    except ImportError:
        logger.exception("Import error")
        cls = None
    _class_cache[kind] = cls
    return cls
