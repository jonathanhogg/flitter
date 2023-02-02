# cython: language_level=3, profile=True


cdef class VectorLike:
    cpdef VectorLike copynodes(self)
    cdef Vector slice(self, Vector index)
    cdef bint as_bool(self)


cdef class Vector(VectorLike):
    cdef int length
    cdef list objects
    cdef double* numbers

    @staticmethod
    cdef Vector _coerce(object other)
    @staticmethod
    cdef VectorLike _compose(list args)

    cdef int allocate_numbers(self, int n) except -1
    cdef void deallocate_numbers(self)
    cdef bint fill_range(self, startv, stopv, stepv) except False
    cpdef bint isinstance(self, t)
    cdef bint as_bool(self)
    cdef double as_float(self)
    cdef str as_string(self)
    cdef unsigned long long hash(self, bint floor_floats)
    cpdef object match(self, int n=?, type t=?, default=?)
    cpdef VectorLike copynodes(self)
    cdef Vector neg(self)
    cdef Vector pos(self)
    cdef Vector abs(self)
    cdef Vector add(self, Vector other)
    cdef Vector sub(self, Vector other)
    cdef Vector mul(self, Vector other)
    cdef Vector truediv(self, Vector other)
    cdef Vector floordiv(self, Vector other)
    cdef Vector mod(self, Vector other)
    cdef Vector pow(self, Vector other)
    cdef Vector eq(self, Vector other)
    cdef Vector ne(self, Vector other)
    cdef Vector gt(self, Vector other)
    cdef Vector ge(self, Vector other)
    cdef Vector lt(self, Vector other)
    cdef Vector le(self, Vector other)
    cdef int compare(self, Vector other) except -2
    cdef Vector slice(self, Vector index)


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
    cdef object weak_self
    cdef readonly str kind
    cdef set _tags
    cdef dict _attributes
    cdef object _parent
    cdef Node next_sibling, first_child, last_child

    cpdef Node copy(self)
    cpdef void add_tag(self, str tag)
    cpdef void remove_tag(self, str tag)
    cpdef void append(self, Node node)
    cpdef void insert(self, Node node)
    cpdef void remove(self, Node node)
    cdef bint _select(self, Query query, list nodes, bint first)
    cdef bint _equals(self, Node other)
    cpdef object get(self, str name, int n=?, type t=?, object default=?)
    cpdef void pprint(self, int indent=?)


cdef class Context:
    cdef readonly dict variables
    cdef readonly dict pragmas
    cdef readonly dict state
    cdef readonly Node graph
    cdef list _stack
