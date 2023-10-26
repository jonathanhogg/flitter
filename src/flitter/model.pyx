# cython: language_level=3, profile=False, boundscheck=False, wraparound=False

import re

import cython
from cython cimport view
import numpy as np

from libc.math cimport isnan, floor, ceil, abs, sqrt, sin, cos, tan, isnan
from cpython cimport PyObject, Py_INCREF
from cpython.bool cimport PyBool_FromLong
from cpython.dict cimport PyDict_GetItem
from cpython.float cimport PyFloat_AS_DOUBLE, PyFloat_FromDouble
from cpython.list cimport PyList_New, PyList_GET_ITEM, PyList_SET_ITEM
from cpython.long cimport (PyLong_FromLongLong, PyLong_FromUnsignedLongLong, PyLong_FromDouble,
                           PyLong_AsLongLong, PyLong_AsUnsignedLongLong, PyLong_AsDouble)
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cpython.object cimport PyObject_RichCompareBool, Py_NE, Py_EQ, Py_LT
from cpython.unicode cimport PyUnicode_DATA, PyUnicode_GET_LENGTH, PyUnicode_KIND, PyUnicode_READ
from cpython.weakref cimport PyWeakref_NewRef, PyWeakref_GetObject


cdef double Pi = 3.141592653589793
cdef double Tau = 6.283185307179586
cdef double NaN = float("nan")
cdef frozenset EmptySet = frozenset()
cdef tuple AstypeArgs = (np.float64, 'K', 'unsafe', True, False)
cdef type ndarray = np.ndarray


cdef union double_long:
    double f
    unsigned long long l


# SplitMix64 algorithm [http://xoshiro.di.unimi.it/splitmix64.c]
#
cdef unsigned long long HASH_START = 0xe220a8397b1dcdaf

cdef inline unsigned long long HASH_UPDATE(unsigned long long _hash, unsigned long long y) noexcept:
    _hash ^= y
    _hash += <unsigned long long>(0x9e3779b97f4a7c15)
    _hash ^= _hash >> 30
    _hash *= <unsigned long long>(0xbf58476d1ce4e5b9)
    _hash ^= _hash >> 27
    _hash *= <unsigned long long>(0x94d049bb133111eb)
    _hash ^= _hash >> 31
    return _hash


# FNV-1a hash algorithm [https://en.wikipedia.org/wiki/Fowler–Noll–Vo_hash_function#FNV-1a_hash]
cdef inline unsigned long long HASH_STRING(str value):
    cdef void* data = PyUnicode_DATA(value)
    cdef unsigned int i, n=PyUnicode_GET_LENGTH(value), kind=PyUnicode_KIND(value)
    cdef Py_UCS4 c
    cdef unsigned long long y = <unsigned long long>(0xcbf29ce484222325)
    for i in range(n):
        c = PyUnicode_READ(kind, data, i)
        y = (y ^ <unsigned long long>c) * <unsigned long long>(0x100000001b3)
    return y


cdef inline int vector_compare(Vector left, Vector right) except -2:
    if left is right:
        return 0
    cdef int i, n = left.length, m = right.length
    if n == 0 and m == 0:
        return 0
    cdef double x, y
    cdef list leftobj = left.objects, rightobj = right.objects
    cdef PyObject* a
    cdef PyObject* b
    if left.numbers != NULL and right.numbers != NULL:
        for i in range(min(n, m)):
            x, y = left.numbers[i], right.numbers[i]
            if x == y:
                continue
            if x < y:
                return -1
            return 1
    elif leftobj is not None and rightobj is not None:
        for i in range(min(n, m)):
            a = PyList_GET_ITEM(leftobj, i)
            b = PyList_GET_ITEM(rightobj, i)
            if PyObject_RichCompareBool(<object>a, <object>b, Py_EQ):
                continue
            if PyObject_RichCompareBool(<object>a, <object>b, Py_LT):
                return -1
            return 1
    elif n == 0:
        return -1
    elif m == 0:
        return 1
    else:
        raise TypeError("Not comparable vectors")
    if n == m:
        return 0
    if n < m:
        return -1
    return 1


cdef int NumbersCacheSize = 0
cdef void** NumbersCache = NULL

cpdef void initialize_numbers_cache(int max_size):
    global NumbersCache, NumbersCacheSize
    cdef int i, n = max_size >> 4
    if max_size & 0xf == 0:
        n -= 1
    cdef void* ptr
    cdef void* next
    if NumbersCacheSize:
        empty_numbers_cache()
        PyMem_Free(NumbersCache)
        NumbersCache = NULL
        NumbersCacheSize = 0
    if n > 0:
        NumbersCache = <void**>PyMem_Malloc(sizeof(void*) * n)
        for i in range(n):
            NumbersCache[i] = NULL
        NumbersCacheSize = n

cpdef void empty_numbers_cache():
    cdef int i
    for i in range(NumbersCacheSize):
        ptr = NumbersCache[i]
        while ptr != NULL:
            next = (<void**>ptr)[0]
            PyMem_Free(ptr)
            ptr = next
        NumbersCache[i] = NULL

cpdef dict numbers_cache_counts():
    cdef dict sizes = {}
    cdef int i, n
    cdef void* ptr
    for i in range(NumbersCacheSize):
        n = 0
        ptr = NumbersCache[i]
        while ptr != NULL:
            ptr = (<void**>ptr)[0]
            n += 1
        if n:
            sizes[16*(i+2)] = n
    return sizes

cdef inline double* malloc_numbers(int n) except NULL:
    global NumbersCache, NumbersCacheSize
    cdef double* numbers
    cdef int i = (n >> 4) - 1
    if n & 0xf == 0:
        i -= 1
    if i < NumbersCacheSize and NumbersCache[i] != NULL:
        numbers = <double*>NumbersCache[i]
        NumbersCache[i] = (<void**>NumbersCache[i])[0]
    else:
        numbers = <double*>PyMem_Malloc((i+2) * 16 * sizeof(double))
        if not numbers:
            raise MemoryError()
    return numbers

cdef inline void free_numbers(int n, double* numbers) noexcept:
    global NumbersCache, NumbersCacheSize
    cdef void* ptr
    cdef int i = (n >> 4) - 1
    if n & 0xf == 0:
        i -= 1
    if i < NumbersCacheSize:
        ptr = NumbersCache[i]
        (<void**>numbers)[0] = ptr
        NumbersCache[i] = <void*>numbers
    else:
        PyMem_Free(numbers)

initialize_numbers_cache(8192)


cdef dict InternedVectors = {}


@cython.freelist(1024)
cdef class Vector:
    @staticmethod
    def coerce(other):
        return Vector._coerce(other)

    @staticmethod
    cdef Vector _coerce(object other):
        if isinstance(other, Vector):
            return other
        if other is None or (isinstance(other, (list, tuple, set, dict)) and len(other) == 0):
            return null_
        if isinstance(other, (float, int)):
            if other == 0:
                return false_
            if other == 1:
                return true_
            if other == -1:
                return minusone_
        return Vector.__new__(Vector, other)

    @staticmethod
    def copy(other):
        cdef Vector vector = Vector._coerce(other)
        if vector is other:
            return Vector._copy(vector)
        return vector

    @staticmethod
    cdef Vector _copy(Vector other):
        cdef Vector result = Vector.__new__(Vector)
        cdef int i, n=other.length
        if other.numbers != NULL:
            result.allocate_numbers(n)
            for i in range(n):
                result.numbers[i] = other.numbers[i]
        elif other.objects is not None:
            result.objects = list(other.objects)
            result.length = n
        return result

    @staticmethod
    def compose(vectors):
        vectors = [Vector._coerce(v) for v in  vectors]
        return Vector._compose(vectors)

    @staticmethod
    cdef Vector _compose(list vectors):
        cdef int m = len(vectors)
        if m == 1:
            return <Vector>PyList_GET_ITEM(vectors, 0)
        if m == 0:
            return null_
        cdef int i, j, k, n = 0
        cdef bint numeric = True
        cdef Vector v, result = Vector.__new__(Vector)
        for i in range(m):
            v = <Vector>PyList_GET_ITEM(vectors, i)
            if v.objects is not None:
                numeric = False
            n += v.length
        if numeric:
            result.allocate_numbers(n)
            j = 0
            for i in range(m):
                v = <Vector>PyList_GET_ITEM(vectors, i)
                for k in range(v.length):
                    result.numbers[j] = v.numbers[k]
                    j += 1
            return result
        cdef list src, dest = PyList_New(n)
        cdef object obj
        cdef PyObject* objptr
        j = 0
        for i in range(m):
            v = <Vector>PyList_GET_ITEM(vectors, i)
            src = v.objects
            if src is None:
                for k in range(v.length):
                    obj = v.numbers[k]
                    Py_INCREF(obj)
                    PyList_SET_ITEM(dest, j, obj)
                    j += 1
            else:
                for k in range(v.length):
                    objptr = PyList_GET_ITEM(src, k)
                    Py_INCREF(<object>objptr)
                    PyList_SET_ITEM(dest, j, <object>objptr)
                    j += 1
        result.objects = dest
        result.length = n
        return result

    @staticmethod
    def range(*args):
        cdef Vector result = Vector.__new__(Vector)
        if len(args) == 1:
            result.fill_range(null_, Vector._coerce(args[0]), null_)
        elif len(args) == 2:
            result.fill_range(Vector._coerce(args[0]), Vector._coerce(args[1]), null_)
        elif len(args) == 3:
            result.fill_range(Vector._coerce(args[0]), Vector._coerce(args[1]), Vector._coerce(args[2]))
        else:
            raise TypeError("range takes 1-3 arguments")
        return result

    def __cinit__(self, value=None):
        if value is None:
            return
        cdef int i, n
        cdef const double[:] arr
        if type(value) is ndarray:
            arr = value.astype(*AstypeArgs)
            for i in range(self.allocate_numbers(arr.shape[0])):
                self.numbers[i] = arr[i]
        elif isinstance(value, (list, tuple, set, dict, Vector)):
            n = len(value)
            if n:
                self.allocate_numbers(n)
                try:
                    for i, v in enumerate(value):
                        self.numbers[i] = v
                except TypeError:
                    self.deallocate_numbers()
                    self.objects = list(value)
        elif isinstance(value, (float, int)):
            self.allocate_numbers(1)
            self.numbers[0] = value
        elif isinstance(value, (range, slice)):
            self.fill_range(Vector._coerce(value.start), Vector._coerce(value.stop), Vector._coerce(value.step))
        else:
            self.objects = [value]
            self.length = 1

    @property
    def numeric(self):
        return self.numbers != NULL

    @property
    def non_numeric(self):
        return self.objects is not None

    cdef int allocate_numbers(self, int n) except -1:
        if n > 16:
            self.numbers = malloc_numbers(n)
        elif n:
            self.numbers = self._numbers
        self.length = n
        return n

    cdef void deallocate_numbers(self) noexcept:
        if self.numbers != NULL and self.numbers != self._numbers:
            free_numbers(self.length, self.numbers)
        self.numbers = NULL

    def __reduce__(self):
        if self.objects is not None:
            return Vector, (self.objects,)
        cdef list values = PyList_New(self.length)
        cdef int i
        for i in range(self.length):
            value = PyFloat_FromDouble(self.numbers[i])
            Py_INCREF(value)
            PyList_SET_ITEM(values, i, value)
        return Vector, (values,)

    cpdef Vector intern(self):
        return <Vector>InternedVectors.setdefault(self, self)

    @cython.cdivision(True)
    cdef void fill_range(self, Vector startv, Vector stopv, Vector stepv):
        assert self.length == 0
        cdef double start, stop, step
        if startv.length == 0:
            start = 0
        elif startv.numbers != NULL and startv.length == 1:
            start = startv.numbers[0]
        else:
            return
        if stopv.numbers == NULL or stopv.length != 1:
            return
        stop = stopv.numbers[0]
        if stepv.length == 0:
            step = 1
        elif stepv.numbers != NULL and stepv.length == 1:
            step = stepv.numbers[0]
            if step == 0:
                return
        else:
            return
        cdef int i, n = <int>ceil((stop - start) / step)
        if n > 0:
            for i in range(self.allocate_numbers(n)):
                self.numbers[i] = start + step * i
        return

    def __dealloc__(self):
        if self.numbers != NULL and self.numbers != self._numbers:
            free_numbers(self.length, self.numbers)
        self.numbers = NULL
        self.length = 0

    def __len__(self):
        return self.length

    cpdef bint isinstance(self, t) noexcept:
        if not self.length:
            return False
        if self.objects is not None:
            for value in self.objects:
                if not isinstance(value, t):
                    return False
            return True
        else:
            return issubclass(float, t)

    def __bool__(self):
        return self.as_bool()

    cdef bint as_bool(self):
        cdef PyObject* objptr
        cdef int i
        cdef list objects
        if self.numbers != NULL:
            for i in range(self.length):
                if self.numbers[i] != 0.:
                    return True
        elif (objects := self.objects) is not None:
            for i in range(self.length):
                objptr = PyList_GET_ITEM(objects, i)
                if type(<object>objptr) is float:
                    if PyFloat_AS_DOUBLE(<object>objptr) != 0.:
                        return True
                elif type(<object>objptr) is str:
                    if PyUnicode_GET_LENGTH(<object>objptr) != 0:
                        return True
                elif type(<object>objptr) is int or type(<object>objptr) is bool:
                    if PyLong_AsLongLong(<object>objptr) != <long long>0:
                        return True
                else:
                    return True
        return False

    def __float__(self):
        return self.as_double()

    cdef double as_double(self) noexcept:
        if self.length == 1 and self.objects is None:
            return self.numbers[0]
        return NaN

    def __str__(self):
        return self.as_string()

    cdef str as_string(self):
        cdef str text = ""
        cdef PyObject* objptr
        cdef int i, n = self.length
        if self.objects is not None:
            if n == 1:
                objptr = PyList_GET_ITEM(self.objects, 0)
                if type(<object>objptr) is str:
                    return <str>objptr
            for i in range(n):
                objptr = PyList_GET_ITEM(self.objects, i)
                if type(<object>objptr) is str:
                    text += <str>objptr
                elif isinstance(<object>objptr, (float, int)):
                    text += f"{<object>objptr:.9g}"
        elif n:
            for i in range(n):
                text += f"{self.numbers[i]:.9g}"
        return text

    def __iter__(self):
        cdef int i
        if self.length:
            if self.objects:
                yield from self.objects
            else:
                for i in range(self.length):
                    yield self.numbers[i]

    def __hash__(self):
        return self.hash(False)

    cdef unsigned long long hash(self, bint floor_floats):
        if not floor_floats and self._hash:
            return self._hash
        cdef unsigned long long y, _hash = HASH_START
        cdef list objects
        cdef unsigned int i
        if self.length == 0:
            pass
        elif (objects := self.objects) is not None:
            for i in range(self.length):
                value = PyList_GET_ITEM(objects, i)
                if type(<object>value) is str:
                    y = HASH_STRING(<str>value)
                elif type(<object>value) is float:
                    if floor_floats:
                        y = <unsigned long long>(<long long>floor(PyFloat_AS_DOUBLE(<object>value)))
                    else:
                        y = double_long(f=PyFloat_AS_DOUBLE(<object>value)).l
                elif type(<object>value) is int:
                    if floor_floats:
                        y = <unsigned long long>(PyLong_AsLongLong(<object>value))
                    else:
                        y = double_long(f=PyLong_AsDouble(<object>value)).l
                else:
                    y = hash(<object>value)
                _hash = HASH_UPDATE(_hash, y)
        else:
            for i in range(self.length):
                if floor_floats:
                    y = <unsigned long long>(<long long>floor(self.numbers[i]))
                else:
                    y = double_long(f=self.numbers[i]).l
                _hash = HASH_UPDATE(_hash, y)
        if not floor_floats:
            self._hash = _hash
        return _hash

    cpdef object match(self, int n=0, type t=None, default=None):
        cdef int i, m = self.length
        cdef list values
        cdef double f
        cdef object obj
        if self.objects is None:
            if t is float:
                t = None
            if t is None or t is int or t is bool:
                if n == 0 or n == m:
                    if n == 1:
                        f = self.numbers[0]
                        if t is int:
                            return <long long>floor(f)
                        elif t is bool:
                            return f != 0
                        return f
                    else:
                        values = PyList_New(m)
                        for i in range(m):
                            f = self.numbers[i]
                            if t is int:
                                obj = PyLong_FromDouble(floor(f))
                            elif t is bool:
                                obj = PyBool_FromLong(f != 0)
                            else:
                                obj = PyFloat_FromDouble(f)
                            Py_INCREF(obj)
                            PyList_SET_ITEM(values, i, obj)
                        return values
                elif m == 1:
                    values = PyList_New(n)
                    f = self.numbers[0]
                    if t is int:
                        obj = PyLong_FromDouble(floor(f))
                    elif t is bool:
                        obj = PyBool_FromLong(f != 0)
                    else:
                        obj = PyFloat_FromDouble(f)
                    for i in range(n):
                        Py_INCREF(obj)
                        PyList_SET_ITEM(values, i, obj)
                    return values
            return default
        if n == 0 and t is None:
            return self.objects
        try:
            if m == 1:
                obj = <object>PyList_GET_ITEM(self.objects, 0)
                if t is not None:
                    obj = t(obj)
                if n == 1:
                    return obj
                if n == 0:
                    return [obj]
                values = PyList_New(n)
                for i in range(n):
                    Py_INCREF(obj)
                    PyList_SET_ITEM(values, i, obj)
                return values
            elif m == n or n == 0:
                values = PyList_New(m)
                for i in range(m):
                    obj = <object>PyList_GET_ITEM(self.objects, i)
                    if t is not None:
                        obj = t(obj)
                    Py_INCREF(obj)
                    PyList_SET_ITEM(values, i, obj)
                return values
        except ValueError:
            pass
        return default

    cpdef Vector copynodes(self, bint parented=False):
        cdef list src=self.objects
        if src is None:
            return self
        cdef int i, j, n=self.length
        cdef list dest=None
        cdef PyObject* current
        cdef PyObject* earlier
        for i in range(n):
            current = PyList_GET_ITEM(src, i)
            if type(<object>current) is Node and (not parented or (<Node>current)._parent is not None):
                if dest is None:
                    dest = PyList_New(n)
                    for j in range(i):
                        earlier = PyList_GET_ITEM(src, j)
                        Py_INCREF(<object>earlier)
                        PyList_SET_ITEM(dest, j, <object>earlier)
                value = (<Node>current).copy()
                Py_INCREF(value)
                PyList_SET_ITEM(dest, i, value)
            elif dest is not None:
                Py_INCREF(<object>current)
                PyList_SET_ITEM(dest, i, <object>current)
        if dest is None:
            return self
        cdef Vector result
        result = Vector.__new__(Vector)
        result.objects = dest
        result.length = n
        return result

    def __repr__(self):
        return self.repr()

    cdef str repr(self):
        cdef int i, n = self.length
        cdef str s
        cdef Py_UNICODE c
        if n == 0:
            return "null"
        cdef list parts = []
        if self.numbers != NULL:
            for i in range(n):
                parts.append(f"{self.numbers[i]:.9g}")
        else:
            for obj in self.objects:
                if isinstance(obj, (float, int)):
                    parts.append(f"{obj:.9g}")
                elif isinstance(obj, str):
                    s = obj
                    for i, c in enumerate(s):
                        if c == ord('_') or (c >= ord('a') and c <= ord('z')) or (c >= ord('A') and c <= ord('Z')) or \
                           (i > 0 and c >= ord('0') and c <= ord('9')):
                            pass
                        else:
                            parts.append(repr(s))
                            break
                    else:
                        parts.append(":" + s)
                else:
                    parts.append("(" + repr(obj) + ")")
        return ";".join(parts)

    def __neg__(self):
        return self.neg()

    cdef Vector neg(self):
        cdef int i, n = self.length
        cdef Vector result = Vector.__new__(Vector)
        if self.numbers != NULL:
            for i in range(result.allocate_numbers(n)):
                result.numbers[i] = -self.numbers[i]
        return result

    def __pos__(self):
        return self.pos()

    cdef Vector pos(self) noexcept:
        if self.objects is None:
            return self
        return null_

    def __abs__(self):
        return self.abs()

    cdef Vector abs(self):
        cdef int i, n = self.length
        cdef Vector result = Vector.__new__(Vector)
        if self.numbers != NULL:
            for i in range(result.allocate_numbers(n)):
                result.numbers[i] = abs(self.numbers[i])
        return result

    def __add__(self, other):
        return self.add(Vector._coerce(other))

    def __radd__(self, other):
        return Vector._coerce(other).add(self)

    @cython.cdivision(True)
    cdef Vector add(self, Vector other):
        cdef int i, n = self.length, m = other.length
        cdef Vector result = Vector.__new__(Vector)
        if self.numbers != NULL and other.numbers != NULL:
            for i in range(result.allocate_numbers(max(n, m))):
                result.numbers[i] = self.numbers[i % n] + other.numbers[i % m]
        return result

    @cython.cdivision(True)
    cpdef Vector mul_add(self, Vector left, Vector right):
        cdef int i, n = self.length, m = left.length, o = right.length
        cdef Vector result = Vector.__new__(Vector)
        if self.numbers != NULL and left.numbers != NULL and right.numbers != NULL:
            for i in range(result.allocate_numbers(max(n, m, o))):
                result.numbers[i] = self.numbers[i % n] + left.numbers[i % m] * right.numbers[i % o]
        return result

    def __sub__(self, other):
        return self.sub(Vector._coerce(other))

    def __rsub__(self, other):
        return Vector._coerce(other).sub(self)

    @cython.cdivision(True)
    cdef Vector sub(self, Vector other):
        cdef int i, n = self.length, m = other.length
        cdef Vector result = Vector.__new__(Vector)
        if self.numbers != NULL and other.numbers != NULL:
            for i in range(result.allocate_numbers(max(n, m))):
                result.numbers[i] = self.numbers[i % n] - other.numbers[i % m]
        return result

    def __mul__(self, other):
        return self.mul(Vector._coerce(other))

    def __rmul__(self, other):
        return Vector._coerce(other).mul(self)

    @cython.cdivision(True)
    cdef Vector mul(self, Vector other):
        cdef int i, n = self.length, m = other.length
        cdef Vector result = Vector.__new__(Vector)
        if self.numbers != NULL and other.numbers != NULL:
            for i in range(result.allocate_numbers(max(n, m))):
                result.numbers[i] = self.numbers[i % n] * other.numbers[i % m]
        return result

    def __truediv__(self, other):
        return self.truediv(Vector._coerce(other))

    def __rtruediv__(self, other):
        return Vector._coerce(other).truediv(self)

    @cython.cdivision(True)
    cdef Vector truediv(self, Vector other):
        cdef int i, n = self.length, m = other.length
        cdef Vector result = Vector.__new__(Vector)
        if self.numbers != NULL and other.numbers != NULL:
            for i in range(result.allocate_numbers(max(n, m))):
                result.numbers[i] = self.numbers[i % n] / other.numbers[i % m]
        return result

    def __floordiv__(Vector self, other):
        return self.floordiv(Vector._coerce(other))

    def __rfloordiv__(self, other):
        return Vector._coerce(other).floordiv(self)

    @cython.cdivision(True)
    cdef Vector floordiv(self, Vector other):
        cdef int i, n = self.length, m = other.length
        cdef Vector result = Vector.__new__(Vector)
        if self.numbers != NULL and other.numbers != NULL:
            for i in range(result.allocate_numbers(max(n, m))):
                result.numbers[i] = floor(self.numbers[i % n] / other.numbers[i % m])
        return result

    def __mod__(self, other):
        return self.mod(Vector._coerce(other))

    def __rmod__(self, other):
        return Vector._coerce(other).mod(self)

    @cython.cdivision(True)
    cdef Vector mod(self, Vector other):
        cdef int i, n = self.length, m = other.length
        cdef Vector result = Vector.__new__(Vector)
        cdef double x, y
        if self.numbers != NULL and other.numbers != NULL:
            for i in range(result.allocate_numbers(max(n, m))):
                x, y = self.numbers[i % n], other.numbers[i % m]
                result.numbers[i] = x - floor(x / y) * y
        return result

    def __pow__(self, other, modulo):
        cdef Vector v = self.pow(Vector._coerce(other))
        if modulo is not None:
            v = v.mod(Vector._coerce(modulo))
        return v

    def __rpow__(self, other, modulo):
        cdef Vector v = Vector._coerce(other).pow(self)
        if modulo is not None:
            v = v.mod(Vector._coerce(modulo))
        return v

    @cython.cdivision(True)
    cdef Vector pow(self, Vector other):
        cdef int i, n = self.length, m = other.length
        cdef Vector result = Vector.__new__(Vector)
        if self.numbers != NULL and other.numbers != NULL:
            for i in range(result.allocate_numbers(max(n, m))):
                result.numbers[i] = self.numbers[i % n] ** other.numbers[i % m]
        return result

    def __eq__(self, other):
        return self.eq(Vector._coerce(other)) is true_

    cdef Vector eq(self, Vector other):
        if self is other:
            return true_
        cdef int i, n = self.length, m = other.length
        cdef list left = self.objects, right = other.objects
        if n != m or (left is None) != (right is None):
            return false_
        if left is None:
            for i in range(n):
                if self.numbers[i] != other.numbers[i]:
                    return false_
        else:
            for i in range(n):
                if PyObject_RichCompareBool(<object>PyList_GET_ITEM(left, i), <object>PyList_GET_ITEM(right, i), Py_NE):
                    return false_
        return true_

    def __ne__(self, other):
        return self.ne(Vector._coerce(other)) is true_

    cdef Vector ne(self, Vector other):
        if self is other:
            return false_
        cdef int i, n = self.length, m = other.length
        cdef list left = self.objects, right = other.objects
        if n != m or (left is None) != (right is None):
            return true_
        if left is None:
            for i in range(n):
                if self.numbers[i] != other.numbers[i]:
                    return true_
        else:
            for i in range(n):
                if PyObject_RichCompareBool(<object>PyList_GET_ITEM(left, i), <object>PyList_GET_ITEM(right, i), Py_NE):
                    return true_
        return false_

    cdef int compare(self, Vector other) except -2:
        return vector_compare(self, other)

    def __gt__(self, other):
        return self.gt(Vector._coerce(other)) is true_

    cdef Vector gt(self, Vector other):
        return true_ if vector_compare(self, other) == 1 else false_

    def __ge__(self, other):
        return self.ge(Vector._coerce(other)) is true_

    cdef Vector ge(self, Vector other):
        return true_ if vector_compare(self, other) != -1 else false_

    def __lt__(self, other):
        return self.lt(Vector._coerce(other)) is true_

    cdef Vector lt(self, Vector other):
        return true_ if vector_compare(self, other) == -1 else false_

    def __le__(self, other):
        return self.le(Vector._coerce(other)) is true_

    cdef Vector le(self, Vector other):
        return true_ if vector_compare(self, other) != 1 else false_

    def __getitem__(self, index):
        cdef Vector result = self.slice(Vector._coerce(index))
        if result.length == 1:
            return result.objects[0] if result.objects is not None else result.numbers[0]
        return result

    cdef Vector slice(self, Vector index):
        if index.numbers == NULL:
            return null_
        cdef int i, j, m = 0, n = self.length, o = index.length
        cdef Vector result = Vector.__new__(Vector)
        cdef list src = self.objects, dest
        if src is not None:
            result.objects = dest = []
            for i in range(o):
                j = <int>floor(index.numbers[i])
                if j >= 0 and j < n:
                    dest.append(<object>PyList_GET_ITEM(src, j))
                    m += 1
            result.length = m
        else:
            result.allocate_numbers(o)
            for i in range(o):
                j = <int>floor(index.numbers[i])
                if j >= 0 and j < n:
                    result.numbers[m] = self.numbers[j]
                    m += 1
            if m == 0:
                result.deallocate_numbers()
            result.length = m
        return result

    cdef Vector item(self, int i):
        cdef int n = self.length
        if i < 0 or i >= n:
            return null_
        cdef Vector result = Vector.__new__(Vector)
        cdef list objects = self.objects
        cdef PyObject* objptr
        if objects is not None:
            objptr = PyList_GET_ITEM(objects, i)
            if type(<object>objptr) is float:
                result.allocate_numbers(1)
                result.numbers[0] = PyFloat_AS_DOUBLE(<object>objptr)
            elif type(<object>objptr) is int:
                result.allocate_numbers(1)
                result.numbers[0] = PyLong_AsDouble(<object>objptr)
            else:
                result.objects = [<object>objptr]
                result.length = 1
        else:
            result.allocate_numbers(1)
            result.numbers[0] = self.numbers[i]
        return result

    @cython.cdivision(True)
    cpdef double squared_sum(self) noexcept:
        cdef int i, n = self.length
        if self.numbers == NULL:
            return NaN
        cdef double x, y = 0
        for i in range(n):
            x = self.numbers[i]
            y += x * x
        return y

    @cython.cdivision(True)
    cpdef Vector normalize(self):
        cdef int i, n = self.length
        if self.numbers == NULL:
            return null_
        cdef double x, y = 0
        for i in range(n):
            x = self.numbers[i]
            y += x * x
        if y == 0:
            return null_
        y = sqrt(y)
        cdef Vector ys = Vector.__new__(Vector)
        ys.allocate_numbers(n)
        for i in range(n):
            ys.numbers[i] = self.numbers[i] / y
        return ys

    @cython.cdivision(True)
    cpdef Vector dot(self, Vector other):
        cdef int i, n = self.length, m = other.length
        cdef Vector result = Vector.__new__(Vector)
        cdef double sum = 0
        if self.numbers != NULL and other.numbers != NULL:
            for i in range(result.allocate_numbers(max(n, m))):
                sum += self.numbers[i % n] * other.numbers[i % m]
            result.allocate_numbers(1)
            result.numbers[0] = sum
        return result

    cpdef Vector cross(self, Vector other):
        if self.numbers == NULL or self.length != 3 or other.numbers == NULL or other.length != 3:
            return null_
        cdef double* self_numbers = self.numbers
        cdef double* other_numbers = other.numbers
        cdef Vector result = Vector.__new__(Vector)
        result.allocate_numbers(3)
        cdef double* result_numbers = result.numbers
        result_numbers[0] = self_numbers[1]*other_numbers[2] - self_numbers[2]*other_numbers[1]
        result_numbers[1] = self_numbers[2]*other_numbers[0] - self_numbers[0]*other_numbers[2]
        result_numbers[2] = self_numbers[0]*other_numbers[1] - self_numbers[1]*other_numbers[0]
        return result

    cpdef Vector concat(self, Vector other):
        cdef int i, n = self.length, m = other.length
        if m == 0:
            return self
        if n == 0:
            return other
        cdef Vector result = Vector.__new__(Vector)
        cdef list left = self.objects, right = other.objects, dest
        cdef PyObject* objptr
        cdef object obj
        if self.numbers != NULL and other.numbers != NULL:
            result.allocate_numbers(n + m)
            for i in range(n):
                result.numbers[i] = self.numbers[i]
            for i in range(m):
                result.numbers[n + i] = other.numbers[i]
        else:
            dest = PyList_New(n + m)
            if left is None:
                for i in range(n):
                    obj = self.numbers[i]
                    Py_INCREF(obj)
                    PyList_SET_ITEM(dest, i, obj)
            else:
                for i in range(n):
                    objptr = PyList_GET_ITEM(left, i)
                    Py_INCREF(<object>objptr)
                    PyList_SET_ITEM(dest, i, <object>objptr)
            if right is None:
                for i in range(m):
                    obj = other.numbers[i]
                    Py_INCREF(obj)
                    PyList_SET_ITEM(dest, n+i, obj)
            else:
                for i in range(m):
                    objptr = PyList_GET_ITEM(right, i)
                    Py_INCREF(<object>objptr)
                    PyList_SET_ITEM(dest, n+i, <object>objptr)
            result.objects = dest
            result.length = n + m
        return result


cdef Vector null_ = Vector()
cdef Vector true_ = Vector(1)
cdef Vector false_ = Vector(0)
cdef Vector minusone_ = Vector(-1)

null = null_
true = true_
false = false_


cdef class Matrix33(Vector):
    @staticmethod
    cdef Matrix33 _translate(Vector v):
        cdef Matrix33 result
        cdef double* numbers
        if v is not None and v.numbers is not NULL and v.length < 3:
            result = Matrix33.__new__(Matrix33)
            numbers = result.numbers
            if v.length == 1:
                numbers[6] = v.numbers[0]
                numbers[7] = v.numbers[0]
            else:
                numbers[6] = v.numbers[0]
                numbers[7] = v.numbers[1]
            return result
        return None

    @staticmethod
    def translate(v):
        return Matrix33._translate(Vector._coerce(v))

    @staticmethod
    cdef Matrix33 _scale(Vector v):
        cdef Matrix33 result
        cdef double* numbers
        if v is not None and v.numbers is not NULL and v.length < 3:
            result = Matrix33.__new__(Matrix33)
            numbers = result.numbers
            if v.length == 1:
                numbers[0] = v.numbers[0]
                numbers[4] = v.numbers[0]
            elif v.length == 2:
                numbers[0] = v.numbers[0]
                numbers[4] = v.numbers[1]
            return result
        return None

    @staticmethod
    def scale(v):
        return Matrix33._scale(Vector._coerce(v))

    @staticmethod
    cdef Matrix33 _rotate(double turns):
        if isnan(turns):
            return None
        cdef double theta = turns*Tau, cth = cos(theta), sth = sin(theta)
        cdef Matrix33 result = Matrix33.__new__(Matrix33)
        cdef double* numbers = result.numbers
        numbers[0] = cth
        numbers[1] = sth
        numbers[3] = -sth
        numbers[4] = cth
        return result

    @staticmethod
    def rotate(turns):
        return Matrix33._rotate(float(turns))

    @cython.cdivision(True)
    def __cinit__(self, obj=None):
        if self.objects is not None:
            raise ValueError("Argument must be a float or a sequence of 9 floats")
        if self.length == 9:
            return
        cdef double k
        if self.length == 0:
            k = 1
            self.numbers = self._numbers
        elif self.length == 1:
            k = self.numbers[0]
        else:
            raise ValueError("Argument must be a float or a sequence of 9 floats")
        cdef int i
        for i in range(9):
            self.numbers[i] = k if i % 4 == 0 else 0
        self.length = 9

    cdef Matrix33 mmul(self, Matrix33 b):
        cdef Matrix33 result = Matrix33.__new__(Matrix33)
        cdef double* numbers = result.numbers
        cdef double* a_numbers = self.numbers
        cdef double* b_numbers = b.numbers
        cdef int i, j
        for i in range(0, 9, 3):
            for j in range(3):
                numbers[i+j] = a_numbers[j]*b_numbers[i] + a_numbers[j+3]*b_numbers[i+1] + a_numbers[j+6]*b_numbers[i+2]
        return result

    cdef Vector vmul(self, Vector b):
        if b.numbers is NULL or b.length not in (2, 3):
            return None
        cdef Vector result = Vector.__new__(Vector)
        cdef double* a_numbers = self.numbers
        cdef double* b_numbers = b.numbers
        if b.length == 2:
            result.allocate_numbers(2)
            result.numbers[0] = a_numbers[0]*b_numbers[0] + a_numbers[3]*b_numbers[1] + a_numbers[6]
            result.numbers[1] = a_numbers[1]*b_numbers[0] + a_numbers[4]*b_numbers[1] + a_numbers[7]
        else:
            result.allocate_numbers(3)
            result.numbers[0] = a_numbers[0]*b_numbers[0] + a_numbers[3]*b_numbers[1] + a_numbers[6]*b_numbers[2]
            result.numbers[1] = a_numbers[1]*b_numbers[0] + a_numbers[4]*b_numbers[1] + a_numbers[7]*b_numbers[2]
            result.numbers[2] = a_numbers[2]*b_numbers[0] + a_numbers[5]*b_numbers[1] + a_numbers[8]*b_numbers[2]
        return result

    def __matmul__(self, other):
        if isinstance(other, Matrix33):
            return self.mmul(<Matrix33>other)
        return self.vmul(Vector._coerce(other))

    @cython.cdivision(True)
    cpdef Matrix33 inverse(self):
        cdef double* numbers = self.numbers
        cdef double s0 = numbers[0]*numbers[4]*numbers[8];
        cdef double s1 = numbers[7]*numbers[5];
        cdef double s2 = numbers[1]*numbers[3]*numbers[8];
        cdef double s3 = numbers[5]*numbers[6];
        cdef double s4 = numbers[2]*numbers[3]*numbers[7];
        cdef double s5 = numbers[4]*numbers[6];
        cdef double invdet = 1 / (s0 - s1 - s2 - s3 + s4 - s5);
        cdef Matrix33 result = Matrix33.__new__(Matrix33)
        cdef double* result_numbers = result.numbers
        result_numbers[0] = (numbers[4]*numbers[8] - numbers[7]*numbers[5]) * invdet;
        result_numbers[1] = (numbers[2]*numbers[7] - numbers[1]*numbers[8]) * invdet;
        result_numbers[2] = (numbers[1]*numbers[5] - numbers[2]*numbers[4]) * invdet;
        result_numbers[3] = (numbers[5]*numbers[6] - numbers[3]*numbers[8]) * invdet;
        result_numbers[4] = (numbers[0]*numbers[8] - numbers[2]*numbers[6]) * invdet;
        result_numbers[5] = (numbers[3]*numbers[2] - numbers[0]*numbers[5]) * invdet;
        result_numbers[6] = (numbers[3]*numbers[7] - numbers[6]*numbers[4]) * invdet;
        result_numbers[7] = (numbers[6]*numbers[1] - numbers[0]*numbers[7]) * invdet;
        result_numbers[8] = (numbers[0]*numbers[4] - numbers[3]*numbers[1]) * invdet;
        return result

    cpdef Matrix33 transpose(self):
        cdef double* numbers = self.numbers
        cdef Matrix33 result = Matrix33.__new__(Matrix33)
        cdef double* result_numbers = result.numbers
        cdef int i, j
        for i in range(3):
            for j in range(3):
                result_numbers[i*3+j] = numbers[j*3+i]
        return result

    def __repr__(self):
        cdef list rows = []
        cdef double* numbers = self.numbers
        cdef int i
        for i in range(3):
            rows.append(f"| {numbers[i]:7.3f} {numbers[i+3]:7.3f} {numbers[i+6]:7.3f} |")
        return '\n'.join(rows)


cdef class Matrix44(Vector):
    @cython.cdivision(True)
    @staticmethod
    cdef Matrix44 _project(double aspect_ratio, double fov, double near, double far):
        cdef Matrix44 result = Matrix44.__new__(Matrix44)
        cdef double* numbers = result.numbers
        cdef double gradient = tan(fov*Pi)
        numbers[0] = 1 / gradient
        numbers[5] = aspect_ratio / gradient
        numbers[10] = -(far+near) / (far-near)
        numbers[11] = -1
        numbers[14] = -2*far*near / (far-near)
        numbers[15] = 0
        return result

    @staticmethod
    def project(aspect_ratio, fov, near, far):
        return Matrix44._project(aspect_ratio, fov, near, far)

    @cython.cdivision(True)
    @staticmethod
    cdef Matrix44 _ortho(double aspect_ratio, double width, double near, double far):
        cdef Matrix44 result = Matrix44.__new__(Matrix44)
        cdef double* numbers = result.numbers
        numbers[0] = 2 / width
        numbers[5] = 2 * aspect_ratio / width
        numbers[10] = -2 / (far-near)
        numbers[14] = -(far+near) / (far-near)
        numbers[15] = 1
        return result

    @staticmethod
    def ortho(aspect_ratio, width, near, far):
        return Matrix44._ortho(aspect_ratio, width, near, far)

    @staticmethod
    cdef Matrix44 _look(Vector from_position, Vector to_position, Vector up_direction):
        cdef Vector z = from_position.sub(to_position).normalize()
        cdef Vector y = up_direction.sub(z.mul(up_direction.dot(z))).normalize()
        cdef Vector x = y.cross(z)
        cdef Matrix44 translation = Matrix44._translate(from_position.neg())
        cdef Matrix44 result = None
        cdef double* numbers
        if translation is not None and x.length == 3 and y.length == 3 and z.length == 3:
            result = Matrix44.__new__(Matrix44)
            numbers = result.numbers
            numbers[0] = x.numbers[0]
            numbers[1] = y.numbers[0]
            numbers[2] = z.numbers[0]
            numbers[4] = x.numbers[1]
            numbers[5] = y.numbers[1]
            numbers[6] = z.numbers[1]
            numbers[8] = x.numbers[2]
            numbers[9] = y.numbers[2]
            numbers[10] = z.numbers[2]
            result = result.mmul(translation)
        return result

    @staticmethod
    def look(from_position, to_position, up_direction):
        return Matrix44._look(Vector._coerce(from_position), Vector._coerce(to_position), Vector._coerce(up_direction))

    @staticmethod
    cdef Matrix44 _translate(Vector v):
        cdef Matrix44 result
        cdef double* numbers
        if v is not None and v.numbers is not NULL and v.length in (1, 3):
            result = Matrix44.__new__(Matrix44)
            numbers = result.numbers
            if v.length == 1:
                numbers[12] = v.numbers[0]
                numbers[13] = v.numbers[0]
                numbers[14] = v.numbers[0]
            elif v.length == 3:
                numbers[12] = v.numbers[0]
                numbers[13] = v.numbers[1]
                numbers[14] = v.numbers[2]
            return result
        return None

    @staticmethod
    def translate(v):
        return Matrix44._translate(Vector._coerce(v))

    @staticmethod
    cdef Matrix44 _scale(Vector v):
        cdef Matrix44 result
        cdef double* numbers
        if v is not None and v.numbers is not NULL and v.length in (1, 3):
            result = Matrix44.__new__(Matrix44)
            numbers = result.numbers
            if v.length == 1:
                numbers[0] = v.numbers[0]
                numbers[5] = v.numbers[0]
                numbers[10] = v.numbers[0]
            elif v.length == 3:
                numbers[0] = v.numbers[0]
                numbers[5] = v.numbers[1]
                numbers[10] = v.numbers[2]
            return result
        return None

    @staticmethod
    def scale(v):
        return Matrix44._scale(Vector._coerce(v))

    @staticmethod
    cdef Matrix44 _rotate_x(double turns):
        if isnan(turns):
            return None
        cdef double theta = turns*Tau, cth = cos(theta), sth = sin(theta)
        cdef Matrix44 result = Matrix44.__new__(Matrix44)
        cdef double* numbers = result.numbers
        numbers[5] = cth
        numbers[6] = sth
        numbers[9] = -sth
        numbers[10] = cth
        return result

    @staticmethod
    def rotate_x(turns):
        return Matrix44._rotate_x(float(turns))

    @staticmethod
    cdef Matrix44 _rotate_y(double turns):
        if isnan(turns):
            return None
        cdef double theta = turns*Tau, cth = cos(theta), sth = sin(theta)
        cdef Matrix44 result = Matrix44.__new__(Matrix44)
        cdef double* numbers = result.numbers
        numbers[0] = cth
        numbers[2] = -sth
        numbers[8] = sth
        numbers[10] = cth
        return result

    @staticmethod
    def rotate_y(turns):
        return Matrix44._rotate_y(float(turns))

    @staticmethod
    cdef Matrix44 _rotate_z(double turns):
        if isnan(turns):
            return None
        cdef double theta = turns*Tau, cth = cos(theta), sth = sin(theta)
        cdef Matrix44 result = Matrix44.__new__(Matrix44)
        cdef double* numbers = result.numbers
        numbers[0] = cth
        numbers[1] = sth
        numbers[4] = -sth
        numbers[5] = cth
        return result

    @staticmethod
    def rotate_z(turns):
        return Matrix44._rotate_z(float(turns))

    @staticmethod
    cdef Matrix44 _rotate(Vector v):
        cdef Matrix44 matrix, result = None
        if v is not None and v.numbers is not NULL and v.length in (1, 3):
            if v.length == 1:
                if v.numbers[0] and not isnan(v.numbers[0]):
                    result = Matrix44._rotate_z(v.numbers[0])
                    result = Matrix44._rotate_y(v.numbers[0]).mmul(result)
                    result = Matrix44._rotate_x(v.numbers[0]).mmul(result)
            else:
                if v.numbers[2] and not isnan(v.numbers[2]):
                    result = Matrix44._rotate_z(v.numbers[2])
                if v.numbers[1] and not isnan(v.numbers[1]):
                    matrix = Matrix44._rotate_y(v.numbers[1])
                    result = matrix if result is None else matrix.mmul(result)
                if v.numbers[0] and not isnan(v.numbers[0]):
                    matrix = Matrix44._rotate_x(v.numbers[0])
                    result = matrix if result is None else matrix.mmul(result)
        if result is None:
            result = Matrix44.__new__(Matrix44)
        return result

    @staticmethod
    def rotate(v):
        return Matrix44._rotate(Vector._coerce(v))

    @cython.cdivision(True)
    def __cinit__(self, obj=None):
        if self.objects is not None:
            raise ValueError("Argument must be a float or a sequence of 16 floats")
        if self.length == 16:
            return
        cdef double k
        if self.length == 0:
            k = 1
            self.numbers = self._numbers
        elif self.length == 1:
            k = self.numbers[0]
        else:
            raise ValueError("Argument must be a float or a sequence of 16 floats")
        cdef int i
        for i in range(16):
            self.numbers[i] = k if i % 5 == 0 else 0
        self.length = 16

    cdef Matrix44 mmul(self, Matrix44 b):
        cdef Matrix44 result = Matrix44.__new__(Matrix44)
        cdef double* numbers = result.numbers
        cdef double* a_numbers = self.numbers
        cdef double* b_numbers = b.numbers
        cdef int i, j
        for i in range(0, 16, 4):
            for j in range(4):
                numbers[i+j] = a_numbers[j]*b_numbers[i] + \
                               a_numbers[j+4]*b_numbers[i+1] + \
                               a_numbers[j+8]*b_numbers[i+2] + \
                               a_numbers[j+12]*b_numbers[i+3]
        return result

    cdef Vector vmul(self, Vector b):
        if b.numbers is NULL or b.length not in (3, 4):
            return None
        cdef Vector result = Vector.__new__(Vector)
        cdef double* numbers
        cdef double* a_numbers = self.numbers
        cdef double* b_numbers = b.numbers
        cdef int j
        if b.length == 3:
            result.allocate_numbers(3)
            numbers = result.numbers
            for j in range(3):
                numbers[j] = a_numbers[j]*b_numbers[0] + \
                             a_numbers[j+4]*b_numbers[1] + \
                             a_numbers[j+8]*b_numbers[2] + \
                             a_numbers[j+12]
        else:
            result.allocate_numbers(4)
            numbers = result.numbers
            for j in range(4):
                numbers[j] = a_numbers[j]*b_numbers[0] + \
                             a_numbers[j+4]*b_numbers[1] + \
                             a_numbers[j+8]*b_numbers[2] + \
                             a_numbers[j+12]*b_numbers[3]
        return result

    def __matmul__(self, other):
        if isinstance(other, Matrix44):
            return self.mmul(<Matrix44>other)
        return self.vmul(Vector._coerce(other))

    @cython.cdivision(True)
    cpdef Matrix44 inverse(self):
        cdef double* numbers = self.numbers
        cdef double s0 = numbers[0]*numbers[5] - numbers[4]*numbers[1]
        cdef double s1 = numbers[0]*numbers[6] - numbers[4]*numbers[2]
        cdef double s2 = numbers[0]*numbers[7] - numbers[4]*numbers[3]
        cdef double s3 = numbers[1]*numbers[6] - numbers[5]*numbers[2]
        cdef double s4 = numbers[1]*numbers[7] - numbers[5]*numbers[3]
        cdef double s5 = numbers[2]*numbers[7] - numbers[6]*numbers[3]
        cdef double c5 = numbers[10]*numbers[15] - numbers[14]*numbers[11]
        cdef double c4 = numbers[9]*numbers[15] - numbers[13]*numbers[11]
        cdef double c3 = numbers[9]*numbers[14] - numbers[13]*numbers[10]
        cdef double c2 = numbers[8]*numbers[15] - numbers[12]*numbers[11]
        cdef double c1 = numbers[8]*numbers[14] - numbers[12]*numbers[10]
        cdef double c0 = numbers[8]*numbers[13] - numbers[12]*numbers[9]
        cdef double invdet = 1 / (s0*c5 - s1*c4 + s2*c3 + s3*c2 - s4*c1 + s5*c0)
        cdef Matrix44 result = Matrix44.__new__(Matrix44)
        cdef double* result_numbers = result.numbers
        result_numbers[0] = (numbers[5]*c5 - numbers[6]*c4 + numbers[7]*c3) * invdet
        result_numbers[1] = (-numbers[1]*c5 + numbers[2]*c4 - numbers[3]*c3) * invdet
        result_numbers[2] = (numbers[13]*s5 - numbers[14]*s4 + numbers[15]*s3) * invdet
        result_numbers[3] = (-numbers[9]*s5 + numbers[10]*s4 - numbers[11]*s3) * invdet
        result_numbers[4] = (-numbers[4]*c5 + numbers[6]*c2 - numbers[7]*c1) * invdet
        result_numbers[5] = (numbers[0]*c5 - numbers[2]*c2 + numbers[3]*c1) * invdet
        result_numbers[6] = (-numbers[12]*s5 + numbers[14]*s2 - numbers[15]*s1) * invdet
        result_numbers[7] = (numbers[8]*s5 - numbers[10]*s2 + numbers[11]*s1) * invdet
        result_numbers[8] = (numbers[4]*c4 - numbers[5]*c2 + numbers[7]*c0) * invdet
        result_numbers[9] = (-numbers[0]*c4 + numbers[1]*c2 - numbers[3]*c0) * invdet
        result_numbers[10] = (numbers[12]*s4 - numbers[13]*s2 + numbers[15]*s0) * invdet
        result_numbers[11] = (-numbers[8]*s4 + numbers[9]*s2 - numbers[11]*s0) * invdet
        result_numbers[12] = (-numbers[4]*c3 + numbers[5]*c1 - numbers[6]*c0) * invdet
        result_numbers[13] = (numbers[0]*c3 - numbers[1]*c1 + numbers[2]*c0) * invdet
        result_numbers[14] = (-numbers[12]*s3 + numbers[13]*s1 - numbers[14]*s0) * invdet
        result_numbers[15] = (numbers[8]*s3 - numbers[9]*s1 + numbers[10]*s0) * invdet
        return result

    cpdef Matrix44 transpose(self):
        cdef double* numbers = self.numbers
        cdef Matrix44 result = Matrix44.__new__(Matrix44)
        cdef double* result_numbers = result.numbers
        cdef int i, j
        for i in range(4):
            for j in range(4):
                result_numbers[i*4+j] = numbers[j*4+i]
        return result

    cpdef Matrix33 matrix33(self):
        cdef double* numbers = self.numbers
        cdef Matrix33 result = Matrix33.__new__(Matrix33)
        cdef double* result_numbers = result.numbers
        cdef int i, j
        for i in range(3):
            for j in range(3):
                result_numbers[3*i+j] = numbers[4*i+j]
        return result

    def __repr__(self):
        cdef list rows = []
        cdef double* numbers = self.numbers
        cdef int i
        for i in range(4):
            rows.append(f"| {numbers[i]:7.3f} {numbers[i+4]:7.3f} {numbers[i+8]:7.3f} {numbers[i+12]:7.3f} |")
        return '\n'.join(rows)


NAME_REGEX = re.compile(r'[_a-z][_a-z0-9]*', re.IGNORECASE)


@cython.final
cdef class Query:
    def __cinit__(self, str query):
        cdef int i = 0, j, n = len(query)
        while n > 0 and query[n-1] in ' \t\r\n':
            n -= 1
        self.first = False
        if n > 0 and query[n-1] == '!':
            n -= 1
            self.first = True
        while i < n:
            if not self.strict and query[i] == '>':
                self.strict = True
            elif query[i] not in ' \t\r\n':
                break
            i += 1
        if i == n or query[i] in ('.', '|', '>'):
            raise ValueError("Query contains empty element")
        cdef list tag_list = []
        cdef str name
        for j in range(i, n+1):
            if j == n or query[j] in '#.|> ':
                if j > i:
                    if query[i] == '#':
                        if i+1 < j:
                            name = query[i+1:j]
                            if NAME_REGEX.fullmatch(name) is None:
                                raise ValueError(f"Bad tag in query: {name}")
                            tag_list.append(name)
                    elif query[i] != '*':
                        name = query[i:j]
                        if NAME_REGEX.fullmatch(name) is None:
                            raise ValueError(f"Bad kind in query: {name}")
                        self.kind = name
                    i = j
                if j < n and query[j] in '.|> \t\r\n':
                    break
        if tag_list:
            self.tags = frozenset(tag_list)
        if i < n and query[i] == '.':
            self.stop = True
            i += 1
        if i < n and query[i] == '|':
            j = i + 1
            while j < n and query[j] not in '> \t\r\n':
                j += 1
            self.altquery = Query.__new__(Query, query[i+1:j])
            i = j
        if i < n:
            query = query[i:n]
            self.subquery = Query.__new__(Query, query)

    def __str__(self):
        cdef str text = ''
        if self.strict:
            text += '> '
        if self.kind is None and not self.tags:
            text += '*'
        else:
            if self.kind is not None:
                text += self.kind
            if self.tags:
                text += '#' + '#'.join(sorted(self.tags))
        if self.stop:
            text += '.'
        if self.altquery is not None:
            text += '|' + str(self.altquery)
        if self.subquery is not None:
            text += ' ' + str(self.subquery)
        if self.first:
            text += '!'
        return text

    def __repr__(self):
        return f"Query({str(self)!r})"


@cython.final
cdef class Node:
    def __init__(self, str kind, set tags=None, dict attributes=None):
        self.kind = kind
        self._tags = None if tags is None else tags.copy()
        self._attributes = {} if attributes is None else attributes.copy()
        self._attributes_shared = False

    def __hash__(self):
        return self.hash()

    cdef unsigned long long hash(self):
        cdef unsigned long long _hash = HASH_START
        _hash = HASH_UPDATE(_hash, HASH_STRING(self.kind))
        if self._tags is not None:
            for tag in sorted(self._tags):
                _hash = HASH_UPDATE(_hash, HASH_STRING(<str>tag))
        cdef list keys
        cdef Vector value
        if self._attributes:
            keys = sorted(self._attributes.keys())
            for key in keys:
                value = (<Vector>self._attributes[key]).intern()
                self._attributes[key] = value
                _hash = HASH_UPDATE(_hash, HASH_STRING(<str>key))
                _hash = HASH_UPDATE(_hash, value.hash(False))
        cdef Node child = self.first_child
        while child is not None:
            _hash = HASH_UPDATE(_hash, child.hash())
            child = child.next_sibling
        return _hash

    def __setstate__(self, set tags):
        self._tags = tags

    def __reduce__(self):
        return Node, (self.kind,), self._tags, self.children, self.__attributes()

    @property
    def children(self):
        cdef Node node = self.first_child
        while node is not None:
            yield node
            node = node.next_sibling

    def __attributes(self):
        cdef str key
        cdef Vector value
        cdef list values
        cdef int i
        for key, value in self._attributes.items():
            if value.objects is not None:
                yield key, value.objects
            else:
                values = []
                for i in range(value.length):
                    values.append(value.numbers[i])
                yield key, values

    @property
    def tags(self):
        if self._tags is not None:
            return frozenset(self._tags)
        else:
            return EmptySet

    @property
    def parent(self):
        return <Node>PyWeakref_GetObject(self._parent) if self._parent is not None else None

    cpdef Node copy(self):
        cdef Node self_copy = Node.__new__(Node)
        self_copy.kind = self.kind
        if self._tags is not None:
            self_copy._tags = set(self._tags)
        self_copy._attributes = self._attributes
        self_copy._attributes_shared = True
        self._attributes_shared = True
        if self.first_child is None:
            return self_copy
        cdef Node child_copy, child = self.first_child
        cdef object self_weakref = PyWeakref_NewRef(self_copy, None)
        child_copy = child.copy()
        child_copy._parent = self_weakref
        self_copy.first_child = child_copy
        while child.next_sibling is not None:
            child = child.next_sibling
            child_copy.next_sibling = child.copy()
            child_copy = child_copy.next_sibling
            child_copy._parent = self_weakref
        self_copy.last_child = child_copy
        return self_copy

    cpdef void add_tag(self, str tag):
        if self._tags is None:
            self._tags = set()
        self._tags.add(tag)

    cpdef void remove_tag(self, str tag):
        if self._tags is not None:
            self._tags.discard(tag)
            if not self._tags:
                self._tags = None

    cpdef void append(self, Node node):
        cdef Node parent = <Node>PyWeakref_GetObject(node._parent) if node._parent is not None else None
        if parent is not None:
            parent.remove(node)
        node._parent = PyWeakref_NewRef(self, None)
        if self.last_child is not None:
            self.last_child.next_sibling = node
            self.last_child = node
        else:
            self.first_child = self.last_child = node

    cdef void append_vector(self, Vector nodes, bint copy):
        cdef list objects = nodes.objects
        if objects is None:
            return
        weak_self = PyWeakref_NewRef(self, None)
        cdef Node node, last=self.last_child
        cdef PyObject* obj
        cdef int i
        for i in range(nodes.length):
            obj = PyList_GET_ITEM(objects, i)
            if type(<object>obj) is not Node:
                continue
            if copy:
                node = (<Node>obj).copy()
            else:
                node = <Node>obj
                if node._parent is not None and (obj := PyWeakref_GetObject(node._parent)) != NULL:
                    (<Node>obj).remove(node)
            node._parent = weak_self
            if last is not None:
                last.next_sibling = node
            else:
                self.first_child = node
            last = node
        self.last_child = last

    cpdef void insert(self, Node node):
        cdef Node parent = <Node>PyWeakref_GetObject(node._parent) if node._parent is not None else None
        if parent is not None:
            parent.remove(node)
        node._parent = PyWeakref_NewRef(self, None)
        node.next_sibling = self.first_child
        self.first_child = node
        if self.last_child is None:
            self.last_child = self.first_child

    def extend(self, nodes):
        cdef Node node
        for node in nodes:
            self.append(node)

    def prepend(self, nodes):
        cdef Node node
        for node in reversed(nodes):
            self.insert(node)

    cpdef void remove(self, Node node):
        cdef Node parent = <Node>PyWeakref_GetObject(node._parent) if node._parent is not None else None
        if parent is not self:
            raise ValueError("Not a child of this node")
        cdef Node child = self.first_child, previous = None
        while child is not node:
            previous = child
            child = child.next_sibling
        if previous is None:
            self.first_child = node.next_sibling
        else:
            previous.next_sibling = node.next_sibling
        if node is self.last_child:
            self.last_child = previous
        node._parent = None
        node.next_sibling = None

    def delete(self):
        cdef Node parent = <Node>PyWeakref_GetObject(self._parent) if self._parent is not None else None
        if parent is None:
            raise TypeError("No parent")
        parent.remove(self)

    def select(self, qstring):
        cdef list nodes = []
        cdef Query query = qstring if isinstance(qstring, Query) else Query.__new__(Query, qstring)
        self._select(query, nodes, query.first)
        return nodes

    def select_below(self, qstring):
        cdef list nodes = []
        cdef Query query = qstring if isinstance(qstring, Query) else Query.__new__(Query, qstring)
        cdef Node node = self.first_child
        while node is not None:
            if node._select(query, nodes, query.first):
                break
            node = node.next_sibling
        return nodes

    cdef bint _select(self, Query query, list nodes, bint first):
        cdef frozenset tags
        cdef Query altquery = query
        cdef bint descend = not query.strict
        cdef Node node
        while altquery is not None:
            tags = altquery.tags
            if (altquery.kind is None or altquery.kind == self.kind) and \
               (tags is None or (self._tags and tags.issubset(self._tags))):
                if altquery.stop:
                    descend = False
                if query.subquery is not None:
                    node = self.first_child
                    while node is not None:
                        if node._select(query.subquery, nodes, first):
                            return True
                        node = node.next_sibling
                else:
                    nodes.append(self)
                    if first:
                        return True
                break
            altquery = altquery.altquery
        if descend:
            node = self.first_child
            while node is not None:
                if node._select(query, nodes, first):
                    return True
                node = node.next_sibling
        return False

    cdef bint _equals(self, Node other):
        if self.kind != other.kind:
            return False
        if self._tags != other._tags:
            return False
        if self._attributes != other._attributes:
            return False
        cdef Node child1 = self.first_child, child2 = other.first_child
        while child1 is not None and child2 is not None:
            if not child1._equals(child2):
                return False
            child1 = child1.next_sibling
            child2 = child2.next_sibling
        return child1 is None and child2 is None

    def __eq__(self, Node other not None):
        return self._equals(other)

    def __len__(self):
        return len(self._attributes)

    def __contains__(self, str name):
        return name in self._attributes

    def __getitem__(self, str name):
        return self._attributes[name]

    def __setitem__(self, str name, value):
        if self._attributes_shared:
            self._attributes = dict(self._attributes)
            self._attributes_shared = False
        cdef Vector vector = Vector._coerce(value)
        if vector.length:
            self._attributes[name] = vector
        elif name in self._attributes:
            del self._attributes[name]

    def __delitem__(self, str name):
        if self._attributes_shared:
            self._attributes = dict(self._attributes)
            self._attributes_shared = True
        del self._attributes[name]

    def keys(self):
        return self._attributes.keys()

    def values(self):
        return self._attributes.values()

    def items(self):
        return self._attributes.items()

    cpdef object get(self, str name, int n=0, type t=None, object default=None):
        cdef PyObject* objptr = PyDict_GetItem(self._attributes, name)
        if objptr == NULL:
            return default
        cdef Vector value = <Vector>objptr
        if n == 1:
            if t is bool:
                return value.as_bool()
            if t is str:
                return value.as_string()
        return value.match(n, t, default)

    cdef Vector get_fvec(self, str name, int n, Vector default):
        cdef PyObject* objptr = PyDict_GetItem(self._attributes, name)
        if objptr == NULL:
            return default
        cdef Vector result, value = <Vector>objptr
        cdef int m, i
        if value.numbers != NULL:
            m = value.length
            if m == 1 and n > 1:
                result = Vector.__new__(Vector)
                for i in range(result.allocate_numbers(n)):
                    result.numbers[i] = value.numbers[0]
                return result
            elif m == n:
                return value
        return default

    cdef double get_float(self, str name, double default):
        cdef PyObject* objptr = PyDict_GetItem(self._attributes, name)
        if objptr == NULL:
            return default
        cdef Vector value = <Vector>objptr
        if value.numbers != NULL and value.length == 1:
            return value.numbers[0]
        return default

    cdef int get_int(self, str name, int default):
        cdef PyObject* objptr = PyDict_GetItem(self._attributes, name)
        if objptr == NULL:
            return default
        cdef Vector value = <Vector>objptr
        if value.numbers != NULL and value.length == 1:
            return <int>value.numbers[0]
        return default

    cdef bint get_bool(self, str name, bint default):
        cdef PyObject* objptr = PyDict_GetItem(self._attributes, name)
        if objptr == NULL:
            return default
        return (<Vector>objptr).as_bool()

    cdef str get_str(self, str name, str default):
        cdef PyObject* objptr = PyDict_GetItem(self._attributes, name)
        if objptr == NULL:
            return default
        return (<Vector>objptr).as_string()

    def __iter__(self):
        return iter(self._attributes)

    cdef str repr(self):
        cdef int i, indent = 0
        cdef str tag, key
        cdef Vector value
        cdef list parts, lines = []
        cdef Node node = self
        while node is not None:
            parts = []
            for i in range(indent):
                parts.append("")
            parts.append("!" + node.kind)
            if node._tags:
                for tag in sorted(node._tags):
                    parts.append("#" + tag)
            for key, value in node._attributes.items():
                parts.append(key + "=" + value.repr())
            lines.append(" ".join(parts))
            if node.first_child is not None:
                node = node.first_child
                indent += 1
            elif node is self:
                break
            elif node.next_sibling is not None:
                node = node.next_sibling
            else:
                while node is not self and node.next_sibling is None:
                    indent -= 1
                    node = <Node>PyWeakref_GetObject(node._parent)
                if node is self:
                    break
                node = node.next_sibling
        return '\n'.join(lines)

    def __repr__(self):
        return self.repr()
