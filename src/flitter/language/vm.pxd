# cython: language_level=3, profile=False

from cpython cimport PyObject

from ..model cimport Vector
from .context cimport Context


cdef dict static_builtins
cdef dict dynamic_builtins
cdef dict builtins


cdef class VectorStack:
    cdef PyObject** vectors
    cdef int top
    cdef readonly int size

    cpdef VectorStack copy(self)
    cpdef void drop(self, int count=?)
    cpdef void push(self, Vector vector)
    cpdef Vector pop(self)
    cpdef tuple pop_tuple(self, int count)
    cpdef list pop_list(self, int count)
    cpdef dict pop_dict(self, tuple keys)
    cpdef Vector pop_composed(self, int count)
    cpdef Vector peek(self)
    cpdef Vector peek_at(self, int offset)
    cpdef void poke(self, Vector vector)
    cpdef void poke_at(self, int offset, Vector vector)


cdef class Program:
    cdef readonly list instructions
    cdef bint linked
    cdef readonly object path
    cdef readonly object top
    cdef readonly VectorStack stack
    cdef readonly VectorStack lvars

    cpdef void link(self)
    cpdef optimize(self)
    cdef void _execute(self, Context context, VectorStack stack, VectorStack lvars, bint record_stats)
