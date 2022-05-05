# cython: language_level=3, profile=True


cdef class Vector:
    cdef list values

    cdef unsigned long long _hash(self, bint floor_floats)
    cpdef bint istrue(self)
    cpdef Vector withlen(self, int n, bint force_copy=?)
    cpdef Vector add(self, Vector other)
    cpdef sub(self, Vector other)
    cpdef mul(self, Vector other)
    cpdef truediv(self, Vector other)
    cpdef floordiv(self, Vector other)
    cpdef mod(self, Vector other)
    cpdef pow(self, Vector other)
    cpdef int compare(self, Vector other)


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
    cdef readonly str kind
    cdef readonly frozenset tags
    cdef dict attributes
    cdef readonly Node parent
    cdef Node next_sibling, first_child, last_child

    cpdef void dissolve(self)
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
