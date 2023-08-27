# cython: language_level=3, profile=False

from ..model cimport Vector, Node


cdef class Program:
    cdef readonly list instructions
    cdef bint linked

    cdef void link(self)


cdef class StateDict:
    cdef set _changed_keys
    cdef dict _state

    cdef Vector get_item(self, Vector key)
    cdef void set_item(self, Vector key, Vector value)
    cdef bint contains(self, Vector key)


cdef class Context:
    cdef readonly dict variables
    cdef readonly set unbound
    cdef readonly dict pragmas
    cdef readonly StateDict state
    cdef readonly Node graph
    cdef readonly str path
    cdef readonly Context parent
    cdef readonly set errors
    cdef readonly set logs
