"""
Flitter generic interaction
"""

import importlib


_class_cache = {}


def get_interactor(kind):
    if kind in _class_cache:
        return _class_cache[kind]
    try:
        module = importlib.import_module(f'.{kind}', __package__)
        cls = module.INTERACTOR_CLASS
    except ImportError:
        cls = None
    _class_cache[kind] = cls
    return cls
