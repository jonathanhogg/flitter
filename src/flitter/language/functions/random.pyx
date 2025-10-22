
import cython

from libc.math cimport floor, sqrt, log
from libc.stdint cimport int64_t, uint64_t
from cpython.object cimport PyObject
from cpython.tuple cimport PyTuple_GET_ITEM, PyTuple_SET_ITEM

from ...model cimport true_, false_, null_, cost, sint


cdef class uniform(Vector):
    def __init__(self, value=None):
        super().__init__(value)
        self._hash = self.hash(True)
        self.deallocate_numbers()
        self.length = 0
        self.objects = None

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return self.eq(Vector._coerce(other)).numbers[0] != 0

    cdef Vector eq(self, Vector other):
        if type(other) is self.__class__ and other._hash == self._hash:
            return true_
        return false_

    cpdef Vector item(self, int64_t i):
        cdef Vector value = Vector.__new__(Vector)
        value.allocate_numbers(1)
        value.numbers[0] = self._item(i)
        return value

    @cython.cdivision(True)
    cdef double _item(self, uint64_t i) noexcept nogil:
        # Compute a 64bit PRN using the 5-round Squares algorithm [https://arxiv.org/abs/2004.06278]
        cdef uint64_t t, x, y, z
        x = y = i * <uint64_t>self._hash
        z = y + <uint64_t>self._hash
        x = x*x + y
        x = (x >> 32) | (x << 32)
        x = x*x + z
        x = (x >> 32) | (x << 32)
        x = x*x + y
        x = (x >> 32) | (x << 32)
        x = x*x + z
        t = x
        x = (x >> 32) | (x << 32)
        x = x*x + y
        t ^= x >> 32
        # This double will retain *at least* 53 bits of the 64bit PRN
        return <double>t / <double>(1<<64)

    cpdef Vector slice(self, Vector index):
        if index.numbers == NULL:
            return null_
        cdef Vector result = Vector.__new__(Vector)
        cdef int64_t i
        cdef uint64_t j
        for i in range(result.allocate_numbers(index.length)):
            j = <uint64_t>(<long long>floor(index.numbers[i]))
            result.numbers[i] = self._item(j)
        return result

    cpdef bint as_bool(self):
        return True

    def __repr__(self):
        return f"{self.__class__.__name__}({self._hash!r})"


cdef class beta(uniform):
    cdef double _item(self, uint64_t i) noexcept nogil:
        i <<= 2
        cdef double u1 = uniform._item(self, i)
        cdef double u2 = uniform._item(self, i + 1)
        cdef double u3 = uniform._item(self, i + 2)
        if u1 <= u2 and u1 <= u3:
            return min(u2, u3)
        if u2 <= u1 and u2 <= u3:
            return min(u1, u3)
        return min(u1, u2)


cdef class normal(uniform):
    @cython.cdivision(True)
    cdef double _item(self, uint64_t i) noexcept nogil:
        # Use the Box-Muller transform to approximate the normal distribution
        # [https://en.wikipedia.org/wiki/Box–Muller_transform]
        cdef double u1, u2
        cdef bint odd = i & 1
        if odd:
            i ^= 1
        if not self.cached or i != self.i:
            u1 = uniform._item(self, i)
            u2 = uniform._item(self, i + 1)
            if u1 == 0:
                u1, u2 = u2, u1
            self.R = sqrt(-2 * log(u1))
            self.th = u2
            self.i = i
            self.cached = True
        if odd:
            return self.R * sint(self.th)
        return self.R * cost(self.th)


cpdef shuffle(uniform source, Vector xs):
    if xs.length == 0:
        return null_
    cdef int64_t i, j, n = xs.length
    cdef PyObject* a
    cdef PyObject* b
    xs = xs.copy()
    cdef tuple objects=xs.objects
    if objects is None:
        for i in range(n - 1):
            j = <int>floor(source.item(i) * n) + i
            n -= 1
            xs.numbers[i], xs.numbers[j] = xs.numbers[j], xs.numbers[i]
    else:
        for i in range(n - 1):
            j = <int>floor(source.item(i) * n) + i
            n -= 1
            a = PyTuple_GET_ITEM(objects, i)
            b = PyTuple_GET_ITEM(objects, j)
            PyTuple_SET_ITEM(objects, i, <object>b)
            PyTuple_SET_ITEM(objects, j, <object>a)
    return xs
