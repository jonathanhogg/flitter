# cython: language_level=3

"""
Flitter counters
"""

from loguru import logger

from .. import name_patch
from ..model cimport Node, Vector, StateDict, null_, false_


logger = name_patch(logger, __name__)

cdef Vector TIME = Vector._symbol('time')


class Counter:
    def __init__(self, **kwargs):
        pass

    def destroy(self):
        pass

    def purge(self):
        pass

    async def update(self, engine, Node node, double clock, **kwargs):
        cdef StateDict state = engine.state
        cdef Vector current_key = <Vector>node._attributes.get('state') if node._attributes else None
        if current_key is None or current_key.length == 0:
            return
        cdef Vector last_time_key = current_key.concat(TIME)
        cdef Vector time = node.get_fvec('time', 0, Vector(clock))
        cdef Vector rate = node.get_fvec('rate', 0, false_)
        cdef Vector initial = node.get_fvec('initial', 0, false_)
        cdef Vector minimum = node.get_fvec('minimum', 0, null_)
        cdef Vector maximum = node.get_fvec('maximum', 0, null_)
        cdef Vector current = state.get_item(current_key)
        cdef Vector last_time = state.get_item(last_time_key)
        if current.length == 0:
            logger.debug("New counter {}", repr(current_key))
            current = initial
        else:
            current = time.sub(last_time).mul(rate).add(current)
        state.set_item(current_key, current.clamp(minimum, maximum))
        state.set_item(last_time_key, time)


RENDERER_CLASS = Counter
