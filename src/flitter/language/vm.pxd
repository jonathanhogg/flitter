# cython: language_level=3, profile=False

from ..model cimport Vector, Node


cdef dict static_builtins
cdef dict dynamic_builtins
cdef dict builtins


cdef class Program:
    cdef readonly list instructions
    cdef bint linked
    cdef readonly dict stats
    cdef readonly str path
    cdef readonly object top

    cdef dict _import(self, Context context, str filename)
    cpdef void link(self)
    cpdef list execute(self, Context context, dict additional_scope=?, bint record_stats=?)


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
