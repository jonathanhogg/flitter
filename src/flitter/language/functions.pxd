# cython: language_level=3, profile=False

from ..model cimport Vector


cdef class Uniform(Vector):
    cdef unsigned long long seed

    cdef double _item(self, unsigned long long i)


cpdef shuffle(Uniform source, Vector xs)
