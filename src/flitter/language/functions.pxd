# cython: language_level=3, profile=False

from ..model cimport Vector


cdef class Uniform(Vector):
    cdef double _item(self, unsigned long long i) noexcept nogil
    cpdef Vector slice(self, Vector index)
    cpdef bint as_bool(self)


cdef class Beta(Uniform):
    pass


cdef class Normal(Uniform):
    cdef bint cached
    cdef unsigned long long i
    cdef double R
    cdef double th


cpdef shuffle(Uniform source, Vector xs)
