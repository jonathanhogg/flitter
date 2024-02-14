# cython: language_level=3, profile=False

from ..model cimport Vector


cdef class uniform(Vector):
    cdef double _item(self, unsigned long long i) noexcept nogil
    cpdef Vector slice(self, Vector index)
    cpdef bint as_bool(self)


cdef class beta(uniform):
    pass


cdef class normal(uniform):
    cdef bint cached
    cdef unsigned long long i
    cdef double R
    cdef double th


cpdef shuffle(uniform source, Vector xs)
