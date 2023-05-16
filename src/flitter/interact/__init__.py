"""
Flitter generic interaction
"""

import importlib


def get_interactor(kind):
    try:
        module = importlib.import_module(f'flitter.interact.{kind}')
    except ImportError:
        return None
    return module.INTERACTOR_CLASS
