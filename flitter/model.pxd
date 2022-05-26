# cython: language_level=3, profile=True


cdef class VectorLike:
    cpdef VectorLike copynodes(self)
    cpdef Vector slice(self, Vector index)
    cpdef bint istrue(self)


cdef class Vector(VectorLike):
    cdef list values

    cdef unsigned long long _hash(self, bint floor_floats)
    cpdef bint istrue(self)
    cpdef VectorLike copynodes(self)
    cpdef Vector withlen(self, int n, bint force_copy=?)
    cpdef Vector neg(self)
    cpdef Vector pos(self)
    cpdef Vector not_(self)
    cpdef Vector add(self, Vector other)
    cpdef Vector sub(self, Vector other)
    cpdef Vector mul(self, Vector other)
    cpdef Vector truediv(self, Vector other)
    cpdef Vector floordiv(self, Vector other)
    cpdef Vector mod(self, Vector other)
    cpdef Vector pow(self, Vector other)
    cpdef int compare(self, Vector other)
    cpdef Vector slice(self, Vector index)


cdef Vector null_
cdef Vector true_
cdef Vector false_


cdef class Query:
    cdef str kind
    cdef frozenset tags
    cdef bint strict
    cdef bint stop
    cdef Query subquery, altquery


cdef class Node:
    cdef object __weakref__
    cdef readonly str kind
    cdef readonly frozenset tags
    cdef dict attributes
    cdef object _parent
    cdef Node next_sibling, first_child, last_child

    cpdef Node copy(self)
    cpdef void append(self, Node node)
    cpdef void insert(self, Node node)
    cpdef void remove(self, Node node)
    cdef bint _select(self, Query query, list nodes, bint first)
    cdef bint _equals(self, Node other)


cdef class Context:
    cdef dict variables
    cdef readonly dict pragmas
    cdef readonly dict state
    cdef readonly Node graph
    cdef list _stack

    cdef void setitem(self, str name, value)
    cpdef merge_under(self, Node node)
