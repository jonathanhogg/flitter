# cython: boundscheck=False, wraparound=False

import cython
import numpy as np

from libc.math cimport isnan, floor as c_floor, ceil as c_ceil, abs as c_abs, round as c_round, sqrt, sin, cos, acos, isnan
from cpython.object cimport PyObject
from cpython.ref cimport Py_INCREF
from cpython.bool cimport PyBool_FromLong
from cpython.dict cimport PyDict_GetItem, PyDict_SetItem, PyDict_DelItem, PyDict_Copy
from cpython.float cimport PyFloat_AS_DOUBLE, PyFloat_FromDouble
from cpython.list cimport PyList_New, PyList_GET_ITEM, PyList_SET_ITEM
from cpython.long cimport (PyLong_FromDouble, PyLong_AsLongLong, PyLong_AsDouble)
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cpython.object cimport PyObject_RichCompareBool, Py_NE, Py_EQ, Py_LT
from cpython.set cimport PySet_Add
from cpython.tuple cimport PyTuple_New, PyTuple_GET_SIZE, PyTuple_GET_ITEM, PyTuple_SET_ITEM, PyTuple_Pack
from cpython.unicode cimport PyUnicode_DATA, PyUnicode_GET_LENGTH, PyUnicode_KIND, PyUnicode_READ


cdef double Pi = 3.141592653589793115997963468544185161590576171875
cdef double Tau = 6.283185307179586231995926937088370323181152343750
cdef double NaN = float("nan")
cdef uint64_t SymbolPrefix = <uint64_t>(0xffe0_0000_0000_0000)
cdef frozenset EmptySet = frozenset()
cdef tuple AstypeArgs = (np.float64, 'K', 'unsafe', True, False)
cdef type ndarray = np.ndarray

cdef dict SymbolTable = {}
cdef dict ReverseSymbolTable = {}


cdef union double_long:
    double f
    uint64_t l


# SplitMix64 algorithm [http://xoshiro.di.unimi.it/splitmix64.c]
#
cdef uint64_t HASH_START = 0xe220a8397b1dcdaf

cdef inline uint64_t HASH_UPDATE(uint64_t _hash, uint64_t y) noexcept:
    _hash ^= y
    _hash += <uint64_t>(0x9e3779b97f4a7c15)
    _hash ^= _hash >> 30
    _hash *= <uint64_t>(0xbf58476d1ce4e5b9)
    _hash ^= _hash >> 27
    _hash *= <uint64_t>(0x94d049bb133111eb)
    _hash ^= _hash >> 31
    return _hash


# FNV-1a hash algorithm [https://en.wikipedia.org/wiki/Fowler–Noll–Vo_hash_function#FNV-1a_hash]
cdef inline uint64_t HASH_STRING(str value):
    cdef void* data = PyUnicode_DATA(value)
    cdef uint64_t i, n=PyUnicode_GET_LENGTH(value), kind=PyUnicode_KIND(value)
    cdef Py_UCS4 c
    cdef uint64_t y = <uint64_t>(0xcbf29ce484222325)
    for i in range(n):
        c = PyUnicode_READ(kind, data, i)
        y = (y ^ <uint64_t>c) * <uint64_t>(0x100000001b3)
    return y


cdef inline int64_t vector_compare(Vector left, Vector right) noexcept:
    if left is right:
        return 0
    cdef int64_t i, n = left.length, m = right.length
    if n == 0 and m == 0:
        return 0
    cdef double x, y
    cdef tuple leftobj = left.objects, rightobj = right.objects
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
            a = PyTuple_GET_ITEM(leftobj, i)
            b = PyTuple_GET_ITEM(rightobj, i)
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
        return -2
    if n == m:
        return 0
    if n < m:
        return -1
    return 1


cdef int64_t NumbersCacheSize = 0
cdef void** NumbersCache = NULL

cpdef void initialize_numbers_cache(int max_size):
    global NumbersCache, NumbersCacheSize
    cdef int64_t i, n = max_size >> 4
    if max_size & 0xf == 0:
        n -= 1
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
    cdef int64_t i
    for i in range(NumbersCacheSize):
        ptr = NumbersCache[i]
        while ptr != NULL:
            next = (<void**>ptr)[0]
            PyMem_Free(ptr)
            ptr = next
        NumbersCache[i] = NULL

cpdef dict numbers_cache_counts():
    cdef dict sizes = {}
    cdef int64_t i, n
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
    cdef int64_t i = (n >> 4) - 1
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
    cdef int64_t i = (n >> 4) - 1
    if n & 0xf == 0:
        i -= 1
    if i < NumbersCacheSize:
        ptr = NumbersCache[i]
        (<void**>numbers)[0] = ptr
        NumbersCache[i] = <void*>numbers
    else:
        PyMem_Free(numbers)

initialize_numbers_cache(16384)


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
        if other is None or (isinstance(other, (list, tuple, bytes, set, dict)) and len(other) == 0):
            return null_
        if isinstance(other, (float, int)):
            if other == 0:
                return false_
            if other == 1:
                return true_
            if other == -1:
                return minusone_
        return Vector(other)

    @staticmethod
    def compose(vectors):
        vectors = [Vector._coerce(v) for v in  vectors]
        return Vector._compose(vectors)

    @staticmethod
    cdef Vector _compose(list vectors):
        cdef int64_t m = len(vectors)
        if m == 1:
            return <Vector>PyList_GET_ITEM(vectors, 0)
        if m == 0:
            return null_
        cdef int64_t i, j, k, n = 0
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
        cdef tuple src, dest = PyTuple_New(n)
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
                    PyTuple_SET_ITEM(dest, j, obj)
                    j += 1
            else:
                for k in range(v.length):
                    objptr = PyTuple_GET_ITEM(src, k)
                    Py_INCREF(<object>objptr)
                    PyTuple_SET_ITEM(dest, j, <object>objptr)
                    j += 1
        result.objects = dest
        result.length = n
        return result

    @staticmethod
    def symbol(str symbol):
        return Vector._symbol(symbol)

    @staticmethod
    def with_symbols(value):
        assert isinstance(value, (list, tuple, set, dict))
        cdef Vector vector = Vector.__new__(Vector)
        n = len(value)
        if n:
            vector.allocate_numbers(n)
            try:
                for i, v in enumerate(value):
                    if isinstance(v, str) and v in ReverseSymbolTable:
                        v = ReverseSymbolTable[v]
                    vector.numbers[i] = v
            except TypeError:
                vector.deallocate_numbers()
                vector.objects = tuple(value)
        return vector

    @staticmethod
    cdef Vector _symbol(str symbol):
        # Symbols are the top 52 bits of the FNV-1a string hash multiplied by -0x1p1023
        cdef uint64_t code = HASH_STRING(symbol)
        cdef double number = double_long(l=SymbolPrefix | (code >> 12)).f
        assert number not in SymbolTable or SymbolTable[number] == symbol, "Symbol table hash collision"
        SymbolTable[number] = symbol
        ReverseSymbolTable[symbol] = number
        cdef Vector result = Vector.__new__(Vector)
        result.numbers = result._numbers
        result.length = 1
        result.numbers[0] = number
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

    def __init__(self, value=None):
        if value is None:
            return
        cdef int64_t i, n
        cdef const double[:] arr
        if type(value) is ndarray:
            arr = value.astype(*AstypeArgs)
            for i in range(self.allocate_numbers(arr.shape[0])):
                self.numbers[i] = arr[i]
        elif isinstance(value, (list, tuple, bytes, set, dict, Vector)):
            n = len(value)
            if n:
                self.allocate_numbers(n)
                try:
                    for i, v in enumerate(value):
                        self.numbers[i] = v
                except TypeError:
                    self.deallocate_numbers()
                    self.objects = tuple(value)
        elif isinstance(value, (float, int)):
            self.numbers = self._numbers
            self.length = 1
            self.numbers[0] = value
        elif isinstance(value, (range, slice)):
            self.fill_range(Vector._coerce(value.start), Vector._coerce(value.stop), Vector._coerce(value.step))
        else:
            self.objects = (value,)
            self.length = 1

    @property
    def numeric(self):
        return self.numbers != NULL

    @property
    def non_numeric(self):
        return self.objects is not None

    cdef int64_t allocate_numbers(self, int64_t n) except -1:
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
        cdef int64_t i
        for i in range(self.length):
            value = PyFloat_FromDouble(self.numbers[i])
            Py_INCREF(value)
            PyList_SET_ITEM(values, i, value)
        return Vector, (values,)

    cpdef Vector copy(self):
        cdef int64_t i, n=self.length
        if n == 0:
            return null_
        cdef Vector result = Vector.__new__(Vector)
        if self.numbers != NULL:
            result.allocate_numbers(n)
            for i in range(n):
                result.numbers[i] = self.numbers[i]
            return result
        cdef tuple objects=self.objects
        cdef tuple copy_objects = PyTuple_New(n)
        cdef PyObject* objptr
        for i in range(n):
            objptr = PyTuple_GET_ITEM(objects, i)
            Py_INCREF(<object>objptr)
            PyTuple_SET_ITEM(copy_objects, i, <object>objptr)
        result.objects = copy_objects
        result.length = n
        return result

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
        cdef int64_t i, n = <int64_t>c_ceil((stop - start) / step)
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
        cdef int64_t i, n=self.length
        if n == 0:
            return False
        cdef PyObject* objptr
        cdef tuple objects = self.objects
        if objects is not None:
            for i in range(n):
                objptr = PyTuple_GET_ITEM(objects, i)
                if not isinstance(<object>objptr, t):
                    return False
            return True
        else:
            return issubclass(float, t)

    def __bool__(self):
        return self.as_bool()

    cdef bint as_bool(self):
        cdef PyObject* objptr
        cdef int64_t i
        cdef tuple objects
        if self.numbers != NULL:
            for i in range(self.length):
                if self.numbers[i] != 0.:
                    return True
        elif (objects := self.objects) is not None:
            for i in range(self.length):
                objptr = PyTuple_GET_ITEM(objects, i)
                if type(<object>objptr) is float:
                    if PyFloat_AS_DOUBLE(<object>objptr) != 0.:
                        return True
                elif type(<object>objptr) is str:
                    if PyUnicode_GET_LENGTH(<object>objptr) != 0:
                        return True
                elif type(<object>objptr) is int or type(<object>objptr) is bool:
                    if PyLong_AsLongLong(<object>objptr) != <int64_t>0:
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

    def __int__(self):
        return self.as_integer()

    cdef int64_t as_integer(self) noexcept:
        cdef double f
        if self.length == 1 and self.objects is None and -(1 << 63) <= (f := self.numbers[0]) < (1 << 63):
            return <int64_t>c_floor(f)
        return 0

    def __str__(self):
        return self.as_string()

    cdef str as_string(self):
        cdef PyObject* objptr
        cdef int64_t i, n = self.length
        if self.numbers != NULL and n == 1 and (objptr := PyDict_GetItem(SymbolTable, self.numbers[0])) != NULL:
            return <str>objptr
        cdef str text = ""
        if self.objects is not None:
            if n == 1:
                objptr = PyTuple_GET_ITEM(self.objects, 0)
                if type(<object>objptr) is str:
                    return <str>objptr
            for i in range(n):
                objptr = PyTuple_GET_ITEM(self.objects, i)
                if type(<object>objptr) is str:
                    text += <str>objptr
                elif isinstance(<object>objptr, (float, int)):
                    number= <object>objptr
                    text += SymbolTable.get(number, f'{number:.9g}')
                elif isinstance(<object>objptr, Node):
                    text += (<Node>objptr).kind
                elif callable(<object>objptr) and hasattr(<object>objptr, '__name__'):
                    text += (<object>objptr).__name__
        elif n:
            for i in range(n):
                if self.numbers[i] == 0:
                    text += "0"
                else:
                    text += SymbolTable.get(self.numbers[i], f'{self.numbers[i]:.9g}')
        return text

    def __iter__(self):
        cdef int64_t i
        if self.length:
            if self.objects:
                yield from self.objects
            else:
                for i in range(self.length):
                    yield self.numbers[i]

    def __hash__(self):
        return self.hash(False)

    cpdef int64_t hash(self, bint floor_floats):
        if not floor_floats and self._hash:
            return self._hash
        cdef uint64_t y, _hash = HASH_START
        cdef tuple objects
        cdef int64_t i
        if self.length == 0:
            pass
        elif (objects := self.objects) is not None:
            for i in range(self.length):
                value = PyTuple_GET_ITEM(objects, i)
                if type(<object>value) is str:
                    y = HASH_STRING(<str>value)
                elif type(<object>value) is float:
                    if floor_floats:
                        y = double_long(f=c_floor(PyFloat_AS_DOUBLE(<object>value))).l
                    else:
                        y = double_long(f=PyFloat_AS_DOUBLE(<object>value)).l
                elif type(<object>value) is int:
                    if floor_floats:
                        y = double_long(f=c_floor(PyLong_AsDouble(<object>value))).l
                    else:
                        y = double_long(f=PyLong_AsDouble(<object>value)).l
                else:
                    y = hash(<object>value)
                _hash = HASH_UPDATE(_hash, y)
        else:
            for i in range(self.length):
                if floor_floats:
                    y = double_long(f=c_floor(self.numbers[i])).l
                else:
                    y = double_long(f=self.numbers[i]).l
                _hash = HASH_UPDATE(_hash, y)
        if not floor_floats:
            self._hash = <int64_t>_hash
        return <int64_t>_hash

    cpdef object match(self, int64_t n=0, type t=None, default=None):
        cdef int64_t i, m = self.length
        cdef list values
        cdef double f
        cdef object obj
        cdef PyObject* objptr
        if self.objects is None:
            if t is None or t is float or t is int or t is bool or t is str:
                if n == 0 or n == m:
                    if n == 1:
                        f = self.numbers[0]
                        if t is str or (t is None and f in SymbolTable):
                            return SymbolTable.get(f, default)
                        if t is int:
                            return PyLong_FromDouble(c_floor(f)) if f not in SymbolTable else default
                        if t is bool:
                            return PyBool_FromLong(f != 0)
                        return f
                    else:
                        values = PyList_New(m)
                        for i in range(m):
                            f = self.numbers[i]
                            objptr = PyDict_GetItem(SymbolTable, PyFloat_FromDouble(f))
                            if t is str or (t is None and objptr != NULL):
                                if objptr == NULL:
                                    return default
                                obj = <object>objptr
                            elif t is int:
                                if f in SymbolTable:
                                    return default
                                obj = PyLong_FromDouble(c_floor(f))
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
                    objptr = PyDict_GetItem(SymbolTable, PyFloat_FromDouble(f))
                    if t is str or (t is None and objptr != NULL):
                        if objptr == NULL:
                            return default
                        obj = <object>objptr
                    elif t is int:
                        if f in SymbolTable:
                            return default
                        obj = PyLong_FromDouble(c_floor(f))
                    elif t is bool:
                        obj = PyBool_FromLong(f != 0)
                    else:
                        obj = PyFloat_FromDouble(f)
                    for i in range(n):
                        Py_INCREF(obj)
                        PyList_SET_ITEM(values, i, obj)
                    return values
            return default
        try:
            if m == 1:
                obj = <object>PyTuple_GET_ITEM(self.objects, 0)
                if (t is str or t is None) and type(obj) is float and (symbol := SymbolTable.get(obj)) is not None:
                    obj = symbol
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
                    obj = <object>PyTuple_GET_ITEM(self.objects, i)
                    if (t is str or t is None) and type(obj) is float and (symbol := SymbolTable.get(obj)) is not None:
                        obj = symbol
                    if t is not None:
                        obj = t(obj)
                    Py_INCREF(obj)
                    PyList_SET_ITEM(values, i, obj)
                return values
        except ValueError:
            pass
        return default

    def __repr__(self):
        return self.repr()

    cdef str repr(self):
        cdef int64_t i, n = self.length
        if n == 0:
            return "null"
        cdef list parts = []
        cdef str symbol
        if self.numbers != NULL:
            for i in range(n):
                if self.numbers[i] == 0:
                    parts.append("0")
                else:
                    symbol = SymbolTable.get(self.numbers[i])
                    parts.append(':' + symbol if symbol is not None else f'{self.numbers[i]:.9g}')
        else:
            for obj in self.objects:
                if isinstance(obj, (float, int)):
                    symbol = SymbolTable.get(obj)
                    parts.append(':' + symbol if symbol is not None else f'{obj:.9g}')
                elif isinstance(obj, str):
                    parts.append(repr(<str>obj))
                else:
                    parts.append("(" + repr(obj) + ")")
        return ";".join(parts)

    def __neg__(self):
        return self.neg()

    cdef Vector neg(self):
        cdef int64_t i, n = self.length
        cdef Vector result = Vector.__new__(Vector)
        if self.numbers != NULL:
            for i in range(result.allocate_numbers(n)):
                result.numbers[i] = -self.numbers[i]
        return result

    def __pos__(self):
        return self.pos()

    cdef Vector pos(self):
        if self.objects is None:
            return self
        return null_

    def __abs__(self):
        return self.abs()

    cdef Vector abs(self):
        cdef int64_t i, n = self.length
        cdef Vector result = Vector.__new__(Vector)
        if self.numbers != NULL:
            for i in range(result.allocate_numbers(n)):
                result.numbers[i] = c_abs(self.numbers[i])
        return result

    def __ceil__(self):
        return self.ceil()

    cdef Vector ceil(self):
        cdef int64_t i, n = self.length
        cdef Vector result = Vector.__new__(Vector)
        if self.numbers != NULL:
            for i in range(result.allocate_numbers(n)):
                result.numbers[i] = c_ceil(self.numbers[i])
        return result

    def __floor__(self):
        return self.floor()

    cdef Vector floor(self):
        cdef int64_t i, n = self.length
        cdef Vector result = Vector.__new__(Vector)
        if self.numbers != NULL:
            for i in range(result.allocate_numbers(n)):
                result.numbers[i] = c_floor(self.numbers[i])
        return result

    cpdef Vector fract(self):
        cdef int64_t i, n = self.length
        cdef Vector result = Vector.__new__(Vector)
        if self.numbers != NULL:
            for i in range(result.allocate_numbers(n)):
                result.numbers[i] = self.numbers[i] - c_floor(self.numbers[i])
        return result

    cdef Vector round(self, int64_t ndigits):
        if self.numbers == NULL:
            return null_
        cdef int64_t i
        cdef Vector result = Vector.__new__(Vector)
        cdef double exponent = 10 ** ndigits
        for i in range(result.allocate_numbers(self.length)):
            result.numbers[i] = c_round(self.numbers[i] * exponent) / exponent
        return result

    def __round__(self, ndigits=None):
        return self.round(ndigits=ndigits if ndigits is not None else 0)

    def __add__(self, other):
        return self.add(Vector._coerce(other))

    def __radd__(self, other):
        return Vector._coerce(other).add(self)

    @cython.cdivision(True)
    cdef Vector add(self, Vector other):
        cdef int64_t i, n = self.length, m = other.length
        cdef Vector result = Vector.__new__(Vector)
        if self.numbers != NULL and other.numbers != NULL:
            for i in range(result.allocate_numbers(max(n, m))):
                result.numbers[i] = self.numbers[i % n] + other.numbers[i % m]
        return result

    @cython.cdivision(True)
    cpdef Vector mul_add(self, Vector left, Vector right):
        cdef int64_t i, n = self.length, m = left.length, o = right.length
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
        cdef int64_t i, n = self.length, m = other.length
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
        cdef int64_t i, n = self.length, m = other.length
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
        cdef int64_t i, n = self.length, m = other.length
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
        cdef int64_t i, n = self.length, m = other.length
        cdef Vector result = Vector.__new__(Vector)
        if self.numbers != NULL and other.numbers != NULL:
            for i in range(result.allocate_numbers(max(n, m))):
                result.numbers[i] = c_floor(self.numbers[i % n] / other.numbers[i % m])
        return result

    def __mod__(self, other):
        return self.mod(Vector._coerce(other))

    def __rmod__(self, other):
        return Vector._coerce(other).mod(self)

    @cython.cdivision(True)
    cdef Vector mod(self, Vector other):
        cdef int64_t i, n = self.length, m = other.length
        cdef Vector result = Vector.__new__(Vector)
        cdef double x, y
        if self.numbers != NULL and other.numbers != NULL:
            for i in range(result.allocate_numbers(max(n, m))):
                x, y = self.numbers[i % n], other.numbers[i % m]
                result.numbers[i] = x - c_floor(x / y) * y
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
        cdef int64_t i, n = self.length, m = other.length
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
        cdef int64_t i, n = self.length, m = other.length
        cdef tuple left = self.objects, right = other.objects
        if n != m or (left is None) != (right is None):
            return false_
        if left is None:
            for i in range(n):
                if self.numbers[i] != other.numbers[i]:
                    return false_
        else:
            for i in range(n):
                if PyObject_RichCompareBool(<object>PyTuple_GET_ITEM(left, i), <object>PyTuple_GET_ITEM(right, i), Py_NE):
                    return false_
        return true_

    def __ne__(self, other):
        return self.ne(Vector._coerce(other)) is true_

    cdef Vector ne(self, Vector other):
        if self is other:
            return false_
        cdef int64_t i, n = self.length, m = other.length
        cdef tuple left = self.objects, right = other.objects
        if n != m or (left is None) != (right is None):
            return true_
        if left is None:
            for i in range(n):
                if self.numbers[i] != other.numbers[i]:
                    return true_
        else:
            for i in range(n):
                if PyObject_RichCompareBool(<object>PyTuple_GET_ITEM(left, i), <object>PyTuple_GET_ITEM(right, i), Py_NE):
                    return true_
        return false_

    cpdef int64_t compare(self, Vector other) noexcept:
        return vector_compare(self, other)

    def __gt__(self, other):
        cdef Vector c = self.gt(Vector._coerce(other))
        if c is null:
            raise TypeError("Not comparable vectors")
        return c is true

    cdef Vector gt(self, Vector other):
        cdef int64_t c = vector_compare(self, other)
        return null_ if c == -2 else true_ if c == 1 else false_

    def __ge__(self, other):
        cdef Vector c = self.ge(Vector._coerce(other))
        if c is null:
            raise TypeError("Not comparable vectors")
        return c is true

    cdef Vector ge(self, Vector other):
        cdef int64_t c = vector_compare(self, other)
        return null_ if c == -2 else true_ if c != -1 else false_

    def __lt__(self, other):
        cdef Vector c = self.lt(Vector._coerce(other))
        if c is null:
            raise TypeError("Not comparable vectors")
        return c is true

    cdef Vector lt(self, Vector other):
        cdef int64_t c = vector_compare(self, other)
        return null_ if c == -2 else true_ if c == -1 else false_

    def __le__(self, other):
        cdef Vector c = self.le(Vector._coerce(other))
        if c is null:
            raise TypeError("Not comparable vectors")
        return c is true

    cdef Vector le(self, Vector other):
        cdef int64_t c = vector_compare(self, other)
        return null_ if c == -2 else true_ if c != 1 else false_

    def __getitem__(self, index):
        cdef Vector result = self.slice(Vector._coerce(index))
        if result.length == 1:
            return result.objects[0] if result.objects is not None else result.numbers[0]
        return result

    cpdef Vector slice(self, Vector index):
        cdef int64_t i, j, m = index.length, n = self.length
        if index.numbers == NULL or n == 0:
            return null_
        cdef Vector result = Vector.__new__(Vector)
        cdef tuple src = self.objects, dest
        cdef PyObject* objptr
        if src is not None:
            result.objects = dest = PyTuple_New(m)
            for i in range(m):
                j = (<int>c_floor(index.numbers[i])) % n
                objptr = PyTuple_GET_ITEM(src, j)
                Py_INCREF(<object>objptr)
                PyTuple_SET_ITEM(dest, i, <object>objptr)
            result.length = m
        elif m:
            result.allocate_numbers(m)
            for i in range(m):
                j = (<int>c_floor(index.numbers[i])) % n
                result.numbers[i] = self.numbers[j]
        return result

    cpdef Vector item(self, int64_t i):
        cdef int64_t n = self.length
        if n == 0:
            return null_
        cdef Vector result = Vector.__new__(Vector)
        result.length = 1
        cdef tuple objects = self.objects
        cdef PyObject* objptr
        if objects is not None:
            objptr = PyTuple_GET_ITEM(objects, i % n)
            if type(<object>objptr) is float:
                result.numbers = result._numbers
                result.numbers[0] = PyFloat_AS_DOUBLE(<object>objptr)
            elif type(<object>objptr) is int:
                result.numbers = result._numbers
                result.numbers[0] = PyLong_AsDouble(<object>objptr)
            else:
                result.objects = PyTuple_Pack(1, objptr)
                result.length = 1
        else:
            result.numbers = result._numbers
            result.numbers[0] = self.numbers[i % n]
        return result

    @cython.cdivision(True)
    cpdef double squared_sum(self) noexcept:
        cdef int64_t i, n = self.length
        if self.numbers == NULL:
            return NaN
        cdef double x, y = 0
        for i in range(n):
            x = self.numbers[i]
            y += x * x
        return y

    @cython.cdivision(True)
    cpdef Vector normalize(self):
        cdef int64_t i, n = self.length
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
        cdef int64_t i, n = self.length, m = other.length
        cdef Vector result = Vector.__new__(Vector)
        cdef double sum = 0
        if self.numbers != NULL and other.numbers != NULL:
            for i in range(result.allocate_numbers(max(n, m))):
                sum += self.numbers[i % n] * other.numbers[i % m]
            result.numbers = result._numbers
            result.length = 1
            result.numbers[0] = sum
        return result

    cpdef Vector cross(self, Vector other):
        if self.numbers == NULL or self.length != 3 or other.numbers == NULL or other.length != 3:
            return null_
        cdef double* self_numbers = self.numbers
        cdef double* other_numbers = other.numbers
        cdef Vector result = Vector.__new__(Vector)
        result.numbers = result._numbers
        result.length = 3
        cdef double* result_numbers = result.numbers
        result_numbers[0] = self_numbers[1]*other_numbers[2] - self_numbers[2]*other_numbers[1]
        result_numbers[1] = self_numbers[2]*other_numbers[0] - self_numbers[0]*other_numbers[2]
        result_numbers[2] = self_numbers[0]*other_numbers[1] - self_numbers[1]*other_numbers[0]
        return result

    cpdef Vector clamp(self, Vector minimum, Vector maximum):
        if self.numbers == NULL or minimum is None or maximum is None:
            return self
        cdef int64_t i, n=max(self.length, minimum.length, maximum.length)
        cdef Vector result = Vector.__new__(Vector)
        cdef double d
        for i in range(result.allocate_numbers(n)):
            d = self.numbers[i % self.length]
            if minimum.numbers != NULL:
                d = max(minimum.numbers[i % minimum.length], d)
            if maximum.numbers != NULL:
                d = min(d, maximum.numbers[i % maximum.length])
            result.numbers[i] = d
        return result

    cpdef Vector concat(self, Vector other):
        cdef int64_t i, n = self.length, m = other.length
        if m == 0:
            return self
        if n == 0:
            return other
        cdef Vector result = Vector.__new__(Vector)
        cdef tuple left = self.objects, right = other.objects, dest
        cdef PyObject* objptr
        cdef object obj
        if self.numbers != NULL and other.numbers != NULL:
            result.allocate_numbers(n + m)
            for i in range(n):
                result.numbers[i] = self.numbers[i]
            for i in range(m):
                result.numbers[n + i] = other.numbers[i]
        else:
            dest = PyTuple_New(n + m)
            if left is None:
                for i in range(n):
                    obj = self.numbers[i]
                    Py_INCREF(obj)
                    PyTuple_SET_ITEM(dest, i, obj)
            else:
                for i in range(n):
                    objptr = PyTuple_GET_ITEM(left, i)
                    Py_INCREF(<object>objptr)
                    PyTuple_SET_ITEM(dest, i, <object>objptr)
            if right is None:
                for i in range(m):
                    obj = other.numbers[i]
                    Py_INCREF(obj)
                    PyTuple_SET_ITEM(dest, n+i, obj)
            else:
                for i in range(m):
                    objptr = PyTuple_GET_ITEM(right, i)
                    Py_INCREF(<object>objptr)
                    PyTuple_SET_ITEM(dest, n+i, <object>objptr)
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
    cdef Matrix33 _identity():
        cdef Matrix33 result = Matrix33.__new__(Matrix33)
        cdef double* numbers = result._numbers
        cdef int64_t i
        for i in range(9):
            numbers[i] = 1 if i % 4 == 0 else 0
        result.numbers = numbers
        result.length = 9
        return result

    @staticmethod
    def identity():
        return Matrix33._identity()

    @staticmethod
    cdef Matrix33 _translate(Vector v):
        cdef Matrix33 result
        cdef double* numbers
        if v is not None and v.numbers is not NULL and v.length < 3:
            result = Matrix33.__new__(Matrix33)
            numbers = result._numbers
            numbers[0] = numbers[4] = numbers[8] = 1
            numbers[1] = numbers[2] = numbers[3] = numbers[5] = 0
            if v.length == 1:
                numbers[6] = v.numbers[0]
                numbers[7] = v.numbers[0]
            else:
                numbers[6] = v.numbers[0]
                numbers[7] = v.numbers[1]
            result.numbers = numbers
            result.length = 9
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
            numbers = result._numbers
            numbers[8] = 1
            numbers[1] = numbers[2] = numbers[3] = numbers[5] = numbers[6] = numbers[7] = 0
            if v.length == 1:
                numbers[0] = v.numbers[0]
                numbers[4] = v.numbers[0]
            elif v.length == 2:
                numbers[0] = v.numbers[0]
                numbers[4] = v.numbers[1]
            result.numbers = numbers
            result.length = 9
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
        cdef double* numbers = result._numbers
        numbers[0] = cth
        numbers[1] = sth
        numbers[2] = 0
        numbers[3] = -sth
        numbers[4] = cth
        numbers[5] = numbers[6] = numbers[7] = 0
        numbers[8] = 1
        result.numbers = numbers
        result.length = 9
        return result

    @staticmethod
    def rotate(turns):
        return Matrix33._rotate(float(turns))

    @cython.cdivision(True)
    def __init__(self, value=None):
        cdef int64_t i
        cdef double k
        cdef const double[:] arr
        if type(value) is ndarray:
            arr = value.astype(*AstypeArgs)
            if arr.shape[0] != 9:
                raise ValueError("Argument must be a float or a sequence of 9 floats")
            self.numbers = self._numbers
            for i in range(9):
                self.numbers[i] = arr[i]
            self.length = 9
        elif isinstance(value, (list, tuple, bytes, set, dict, Vector)):
            if len(value) != 9:
                raise ValueError("Argument must be a float or a sequence of 9 floats")
            try:
                self.numbers = self._numbers
                for i, v in enumerate(value):
                    self.numbers[i] = v
                self.length = 9
            except TypeError:
                raise ValueError("Argument must be a float or a sequence of 9 floats")
        elif value is None or isinstance(value, (float, int)):
            k = 1 if value is None else value
            self.numbers = self._numbers
            for i in range(9):
                self.numbers[i] = k if i % 4 == 0 else 0
            self.length = 9
        elif isinstance(value, (range, slice)):
            self.fill_range(Vector._coerce(value.start), Vector._coerce(value.stop), Vector._coerce(value.step))
            if self.length != 9:
                raise ValueError("Argument must be a float or a sequence of 9 floats")
        else:
            raise ValueError("Argument must be a float or a sequence of 9 floats")

    cpdef Matrix33 copy(self):
        cdef Matrix33 result = Matrix33.__new__(Matrix33)
        cdef double* self_numbers=self.numbers
        cdef double* numbers=result._numbers
        cdef int64_t i
        for i in range(9):
            numbers[i] = self_numbers[i]
        result.numbers = numbers
        result.length = 9
        return result

    cdef Matrix33 mmul(self, Matrix33 b):
        cdef Matrix33 result = Matrix33.__new__(Matrix33)
        cdef double* numbers = result._numbers
        cdef double* a_numbers = self.numbers
        cdef double* b_numbers = b.numbers
        cdef int64_t i, j
        for i in range(0, 9, 3):
            for j in range(3):
                numbers[i+j] = a_numbers[j]*b_numbers[i] + a_numbers[j+3]*b_numbers[i+1] + a_numbers[j+6]*b_numbers[i+2]
        result.numbers = numbers
        result.length = 9
        return result

    cdef Vector vmul(self, Vector b):
        if b.numbers is NULL or b.length not in (2, 3):
            return None
        cdef Vector result = Vector.__new__(Vector)
        cdef double* a_numbers = self.numbers
        cdef double* b_numbers = b.numbers
        if b.length == 2:
            result.numbers = result._numbers
            result.length = 2
            result.numbers[0] = a_numbers[0]*b_numbers[0] + a_numbers[3]*b_numbers[1] + a_numbers[6]
            result.numbers[1] = a_numbers[1]*b_numbers[0] + a_numbers[4]*b_numbers[1] + a_numbers[7]
        else:
            result.numbers = result._numbers
            result.length = 3
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
        cdef double s0 = numbers[0]*numbers[4]*numbers[8]
        cdef double s1 = numbers[7]*numbers[5]
        cdef double s2 = numbers[1]*numbers[3]*numbers[8]
        cdef double s3 = numbers[5]*numbers[6]
        cdef double s4 = numbers[2]*numbers[3]*numbers[7]
        cdef double s5 = numbers[4]*numbers[6]
        cdef double invdet = 1 / (s0 - s1 - s2 - s3 + s4 - s5)
        cdef Matrix33 result = Matrix33.__new__(Matrix33)
        cdef double* result_numbers = result._numbers
        result_numbers[0] = (numbers[4]*numbers[8] - numbers[7]*numbers[5]) * invdet
        result_numbers[1] = (numbers[2]*numbers[7] - numbers[1]*numbers[8]) * invdet
        result_numbers[2] = (numbers[1]*numbers[5] - numbers[2]*numbers[4]) * invdet
        result_numbers[3] = (numbers[5]*numbers[6] - numbers[3]*numbers[8]) * invdet
        result_numbers[4] = (numbers[0]*numbers[8] - numbers[2]*numbers[6]) * invdet
        result_numbers[5] = (numbers[3]*numbers[2] - numbers[0]*numbers[5]) * invdet
        result_numbers[6] = (numbers[3]*numbers[7] - numbers[6]*numbers[4]) * invdet
        result_numbers[7] = (numbers[6]*numbers[1] - numbers[0]*numbers[7]) * invdet
        result_numbers[8] = (numbers[0]*numbers[4] - numbers[3]*numbers[1]) * invdet
        result.numbers = result_numbers
        result.length = 9
        return result

    cpdef Matrix33 transpose(self):
        cdef double* numbers = self.numbers
        cdef Matrix33 result = Matrix33.__new__(Matrix33)
        cdef double* result_numbers = result._numbers
        cdef int64_t i, j
        for i in range(3):
            for j in range(3):
                result_numbers[i*3+j] = numbers[j*3+i]
        result.numbers = result_numbers
        result.length = 9
        return result

    cpdef Matrix44 matrix44(self):
        cdef double* numbers = self.numbers
        cdef Matrix44 result = Matrix44.__new__(Matrix44)
        cdef double* result_numbers = result._numbers
        cdef int64_t i, j
        for i in range(4):
            for j in range(4):
                result_numbers[4*i+j] = 0 if i == 3 or j == 3 else numbers[3*i+j]
        result_numbers[15] = 1
        result.numbers = result_numbers
        result.length = 16
        return result

    def __repr__(self):
        cdef list rows = []
        cdef double* numbers = self.numbers
        cdef int64_t i
        for i in range(3):
            rows.append(f"| {numbers[i]:7.3f} {numbers[i+3]:7.3f} {numbers[i+6]:7.3f} |")
        return '\n'.join(rows)


cdef class Matrix44(Vector):
    @staticmethod
    cdef Matrix44 _identity():
        cdef Matrix44 result = Matrix44.__new__(Matrix44)
        cdef double* numbers = result._numbers
        cdef int64_t i
        for i in range(16):
            numbers[i] = 1 if i % 5 == 0 else 0
        result.numbers = numbers
        result.length = 16
        return result

    @staticmethod
    def identity():
        return Matrix44._identity()

    @cython.cdivision(True)
    @staticmethod
    cdef Matrix44 _project(double xgradient, double ygradient, double near, double far):
        cdef Matrix44 result = Matrix44.__new__(Matrix44)
        cdef double* numbers = result._numbers
        numbers[0] = 1 / xgradient
        numbers[1] = numbers[2] = numbers[3] = numbers[4] = 0
        numbers[5] = 1 / ygradient
        numbers[6] = numbers[7] = numbers[8] = numbers[9] = 0
        numbers[10] = -(far+near) / (far-near)
        numbers[11] = -1
        numbers[12] = numbers[13] = 0
        numbers[14] = -2*far*near / (far-near)
        numbers[15] = 0
        result.numbers = numbers
        result.length = 16
        return result

    @staticmethod
    def project(xgradient, ygradient, near, far):
        return Matrix44._project(xgradient, ygradient, near, far)

    @cython.cdivision(True)
    @staticmethod
    cdef Matrix44 _ortho(double aspect_ratio, double width, double near, double far):
        cdef Matrix44 result = Matrix44.__new__(Matrix44)
        cdef double* numbers = result._numbers
        numbers[0] = 2 / width
        numbers[1] = numbers[2] = numbers[3] = numbers[4] = 0
        numbers[5] = 2 * aspect_ratio / width
        numbers[6] = numbers[7] = numbers[8] = numbers[9] = 0
        numbers[10] = -2 / (far-near)
        numbers[11] = numbers[12] = numbers[13] = 0
        numbers[14] = -(far+near) / (far-near)
        numbers[15] = 1
        result.numbers = numbers
        result.length = 16
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
            numbers = result._numbers
            numbers[0] = x.numbers[0]
            numbers[1] = y.numbers[0]
            numbers[2] = z.numbers[0]
            numbers[3] = 0
            numbers[4] = x.numbers[1]
            numbers[5] = y.numbers[1]
            numbers[6] = z.numbers[1]
            numbers[7] = 0
            numbers[8] = x.numbers[2]
            numbers[9] = y.numbers[2]
            numbers[10] = z.numbers[2]
            numbers[11] = numbers[12] = numbers[13] = numbers[14] = 0
            numbers[15] = 1
            result.numbers = numbers
            result.length = 16
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
            numbers = result._numbers
            numbers[0] = numbers[5] = numbers[10] = numbers[15] = 1
            numbers[1] = numbers[2] = numbers[3] = numbers[4] = 0
            numbers[6] = numbers[7] = numbers[8] = numbers[9] = 0
            numbers[11] = 0
            numbers[11] = 0
            if v.length == 1:
                numbers[12] = v.numbers[0]
                numbers[13] = v.numbers[0]
                numbers[14] = v.numbers[0]
            elif v.length == 3:
                numbers[12] = v.numbers[0]
                numbers[13] = v.numbers[1]
                numbers[14] = v.numbers[2]
            result.numbers = numbers
            result.length = 16
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
            numbers = result._numbers
            numbers[1] = numbers[2] = numbers[3] = numbers[4] = 0
            numbers[6] = numbers[7] = numbers[8] = numbers[9]= 0
            numbers[11] = numbers[12] = numbers[13] = numbers[14] = 0
            numbers[15] = 1
            if v.length == 1:
                numbers[0] = v.numbers[0]
                numbers[5] = v.numbers[0]
                numbers[10] = v.numbers[0]
            elif v.length == 3:
                numbers[0] = v.numbers[0]
                numbers[5] = v.numbers[1]
                numbers[10] = v.numbers[2]
            result.numbers = numbers
            result.length = 16
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
        cdef double* numbers = result._numbers
        numbers[0] = 1
        numbers[1] = numbers[2] = numbers[3] = numbers[4] = 0
        numbers[5] = cth
        numbers[6] = sth
        numbers[7] = numbers[8] = 0
        numbers[9] = -sth
        numbers[10] = cth
        numbers[11] = numbers[12] = numbers[13] = numbers[14] = 0
        numbers[15] = 1
        result.numbers = numbers
        result.length = 16
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
        cdef double* numbers = result._numbers
        numbers[0] = cth
        numbers[1] = 0
        numbers[2] = -sth
        numbers[3] = numbers[4] = 0
        numbers[5] = 1
        numbers[6] = numbers[7] = 0
        numbers[8] = sth
        numbers[9] = 0
        numbers[10] = cth
        numbers[11] = numbers[12] = numbers[13] = numbers[14] = 0
        numbers[15] = 1
        result.numbers = numbers
        result.length = 16
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
        cdef double* numbers = result._numbers
        numbers[0] = cth
        numbers[1] = sth
        numbers[2] = numbers[3] = 0
        numbers[4] = -sth
        numbers[5] = cth
        numbers[6] = numbers[7] = numbers[8] = numbers[9] = 0
        numbers[10] = 1
        numbers[11] = numbers[12] = numbers[13] = numbers[14] = 0
        numbers[15] = 1
        result.numbers = numbers
        result.length = 16
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
            result = Matrix44._identity()
        return result

    @staticmethod
    def rotate(v):
        return Matrix44._rotate(Vector._coerce(v))

    @staticmethod
    cdef Matrix44 _shear_x(Vector v):
        cdef Matrix44 result = None
        if v is not None and v.numbers is not NULL and v.length in (1, 2):
            result = Matrix44.__new__(Matrix44)
            numbers = result._numbers
            numbers[0] = 1
            numbers[1] = numbers[2] = numbers[3] = 0
            numbers[4] = v.numbers[0]
            numbers[5] = 1
            numbers[6] = numbers[7] = 0
            numbers[8] = v.numbers[1] if v.length == 2 else v.numbers[0]
            numbers[9] = 0
            numbers[10] = 1
            numbers[11] = numbers[12] = numbers[13] = numbers[14] = 0
            numbers[15] = 1
            result.numbers = numbers
            result.length = 16
            return result
        return result

    @staticmethod
    def shear_x(v):
        return Matrix44._shear_x(Vector._coerce(v))

    @staticmethod
    cdef Matrix44 _shear_y(Vector v):
        cdef Matrix44 result = None
        if v is not None and v.numbers is not NULL and v.length in (1, 2):
            result = Matrix44.__new__(Matrix44)
            numbers = result._numbers
            numbers[0] = 1
            numbers[1] = v.numbers[0]
            numbers[2] = numbers[3] = numbers[4] = 0
            numbers[5] = 1
            numbers[6] = numbers[7] = numbers[8] = 0
            numbers[9] = v.numbers[1] if v.length == 2 else v.numbers[0]
            numbers[10] = 1
            numbers[11] = numbers[12] = numbers[13] = numbers[14] = 0
            numbers[15] = 1
            result.numbers = numbers
            result.length = 16
            return result
        return result

    @staticmethod
    def shear_y(v):
        return Matrix44._shear_y(Vector._coerce(v))

    @staticmethod
    cdef Matrix44 _shear_z(Vector v):
        cdef Matrix44 result = None
        if v is not None and v.numbers is not NULL and v.length in (1, 2):
            result = Matrix44.__new__(Matrix44)
            numbers = result._numbers
            numbers[0] = 1
            numbers[1] = 0
            numbers[2] = v.numbers[0]
            numbers[3] = numbers[4] = 0
            numbers[5] = 1
            numbers[6] = v.numbers[1] if v.length == 2 else v.numbers[0]
            numbers[7] = numbers[8] = numbers[9] = 0
            numbers[10] = 1
            numbers[11] = numbers[12] = numbers[13] = numbers[14] = 0
            numbers[15] = 1
            result.numbers = numbers
            result.length = 16
            return result
        return result

    @staticmethod
    def shear_z(v):
        return Matrix44._shear_z(Vector._coerce(v))

    @cython.cdivision(True)
    def __init__(self, value=None):
        cdef int64_t i
        cdef double k
        cdef const double[:] arr
        if type(value) is ndarray:
            arr = value.astype(*AstypeArgs)
            if arr.shape[0] != 16:
                raise ValueError("Argument must be a float or a sequence of 16 floats")
            self.numbers = self._numbers
            for i in range(16):
                self.numbers[i] = arr[i]
            self.length = 16
        elif isinstance(value, (list, tuple, bytes, set, dict, Vector)):
            if len(value) != 16:
                raise ValueError("Argument must be a float or a sequence of 16 floats")
            try:
                self.numbers = self._numbers
                for i, v in enumerate(value):
                    self.numbers[i] = v
                self.length = 16
            except TypeError:
                raise ValueError("Argument must be a float or a sequence of 16 floats")
        elif value is None or isinstance(value, (float, int)):
            k = 1 if value is None else value
            self.numbers = self._numbers
            for i in range(16):
                self.numbers[i] = k if i % 5 == 0 else 0
            self.length = 16
        elif isinstance(value, (range, slice)):
            self.length = 0
            self.fill_range(Vector._coerce(value.start), Vector._coerce(value.stop), Vector._coerce(value.step))
            if self.length != 16:
                raise ValueError("Argument must be a float or a sequence of 16 floats")
        else:
            raise ValueError("Argument must be a float or a sequence of 16 floats")

    cpdef Matrix44 copy(self):
        cdef Matrix44 result = Matrix44.__new__(Matrix44)
        cdef double* self_numbers=self.numbers
        cdef double* numbers=result._numbers
        cdef int64_t i
        for i in range(16):
            numbers[i] = self_numbers[i]
        result.numbers = numbers
        result.length = 16
        return result

    cdef Matrix44 mmul(self, Matrix44 b):
        cdef Matrix44 result = Matrix44.__new__(Matrix44)
        cdef double* numbers = result._numbers
        cdef double* a_numbers = self.numbers
        cdef double* b_numbers = b.numbers
        numbers[0] = a_numbers[0]*b_numbers[0] + a_numbers[4]*b_numbers[1] + a_numbers[8]*b_numbers[2] + a_numbers[12]*b_numbers[3]
        numbers[1] = a_numbers[1]*b_numbers[0] + a_numbers[5]*b_numbers[1] + a_numbers[9]*b_numbers[2] + a_numbers[13]*b_numbers[3]
        numbers[2] = a_numbers[2]*b_numbers[0] + a_numbers[6]*b_numbers[1] + a_numbers[10]*b_numbers[2] + a_numbers[14]*b_numbers[3]
        numbers[3] = a_numbers[3]*b_numbers[0] + a_numbers[7]*b_numbers[1] + a_numbers[11]*b_numbers[2] + a_numbers[15]*b_numbers[3]
        numbers[4] = a_numbers[0]*b_numbers[4] + a_numbers[4]*b_numbers[5] + a_numbers[8]*b_numbers[6] + a_numbers[12]*b_numbers[7]
        numbers[5] = a_numbers[1]*b_numbers[4] + a_numbers[5]*b_numbers[5] + a_numbers[9]*b_numbers[6] + a_numbers[13]*b_numbers[7]
        numbers[6] = a_numbers[2]*b_numbers[4] + a_numbers[6]*b_numbers[5] + a_numbers[10]*b_numbers[6] + a_numbers[14]*b_numbers[7]
        numbers[7] = a_numbers[3]*b_numbers[4] + a_numbers[7]*b_numbers[5] + a_numbers[11]*b_numbers[6] + a_numbers[15]*b_numbers[7]
        numbers[8] = a_numbers[0]*b_numbers[8] + a_numbers[4]*b_numbers[9] + a_numbers[8]*b_numbers[10] + a_numbers[12]*b_numbers[11]
        numbers[9] = a_numbers[1]*b_numbers[8] + a_numbers[5]*b_numbers[9] + a_numbers[9]*b_numbers[10] + a_numbers[13]*b_numbers[11]
        numbers[10] = a_numbers[2]*b_numbers[8] + a_numbers[6]*b_numbers[9] + a_numbers[10]*b_numbers[10] + a_numbers[14]*b_numbers[11]
        numbers[11] = a_numbers[3]*b_numbers[8] + a_numbers[7]*b_numbers[9] + a_numbers[11]*b_numbers[10] + a_numbers[15]*b_numbers[11]
        numbers[12] = a_numbers[0]*b_numbers[12] + a_numbers[4]*b_numbers[13] + a_numbers[8]*b_numbers[14] + a_numbers[12]*b_numbers[15]
        numbers[13] = a_numbers[1]*b_numbers[12] + a_numbers[5]*b_numbers[13] + a_numbers[9]*b_numbers[14] + a_numbers[13]*b_numbers[15]
        numbers[14] = a_numbers[2]*b_numbers[12] + a_numbers[6]*b_numbers[13] + a_numbers[10]*b_numbers[14] + a_numbers[14]*b_numbers[15]
        numbers[15] = a_numbers[3]*b_numbers[12] + a_numbers[7]*b_numbers[13] + a_numbers[11]*b_numbers[14] + a_numbers[15]*b_numbers[15]
        result.numbers = numbers
        result.length = 16
        return result

    cdef Matrix44 immul(self, Matrix44 b):
        if b is None:
            return self
        cdef double* numbers = self.numbers
        cdef double* b_numbers = b.numbers
        cdef double a0=numbers[0], a1=numbers[1], a2=numbers[2], a3=numbers[3]
        cdef double a4=numbers[4], a5=numbers[5], a6=numbers[6], a7=numbers[7]
        cdef double a8=numbers[8], a9=numbers[9], a10=numbers[10], a11=numbers[11]
        cdef double a12=numbers[12], a13=numbers[13], a14=numbers[14], a15=numbers[15]
        numbers[0] = a0*b_numbers[0] + a4*b_numbers[1] + a8*b_numbers[2] + a12*b_numbers[3]
        numbers[1] = a1*b_numbers[0] + a5*b_numbers[1] + a9*b_numbers[2] + a13*b_numbers[3]
        numbers[2] = a2*b_numbers[0] + a6*b_numbers[1] + a10*b_numbers[2] + a14*b_numbers[3]
        numbers[3] = a3*b_numbers[0] + a7*b_numbers[1] + a11*b_numbers[2] + a15*b_numbers[3]
        numbers[4] = a0*b_numbers[4] + a4*b_numbers[5] + a8*b_numbers[6] + a12*b_numbers[7]
        numbers[5] = a1*b_numbers[4] + a5*b_numbers[5] + a9*b_numbers[6] + a13*b_numbers[7]
        numbers[6] = a2*b_numbers[4] + a6*b_numbers[5] + a10*b_numbers[6] + a14*b_numbers[7]
        numbers[7] = a3*b_numbers[4] + a7*b_numbers[5] + a11*b_numbers[6] + a15*b_numbers[7]
        numbers[8] = a0*b_numbers[8] + a4*b_numbers[9] + a8*b_numbers[10] + a12*b_numbers[11]
        numbers[9] = a1*b_numbers[8] + a5*b_numbers[9] + a9*b_numbers[10] + a13*b_numbers[11]
        numbers[10] = a2*b_numbers[8] + a6*b_numbers[9] + a10*b_numbers[10] + a14*b_numbers[11]
        numbers[11] = a3*b_numbers[8] + a7*b_numbers[9] + a11*b_numbers[10] + a15*b_numbers[11]
        numbers[12] = a0*b_numbers[12] + a4*b_numbers[13] + a8*b_numbers[14] + a12*b_numbers[15]
        numbers[13] = a1*b_numbers[12] + a5*b_numbers[13] + a9*b_numbers[14] + a13*b_numbers[15]
        numbers[14] = a2*b_numbers[12] + a6*b_numbers[13] + a10*b_numbers[14] + a14*b_numbers[15]
        numbers[15] = a3*b_numbers[12] + a7*b_numbers[13] + a11*b_numbers[14] + a15*b_numbers[15]
        return self

    cdef Vector vmul(self, Vector b):
        if b.numbers is NULL or b.length not in (3, 4):
            return None
        cdef Vector result = Vector.__new__(Vector)
        cdef double* numbers = result._numbers
        cdef double* a_numbers = self.numbers
        cdef double* b_numbers = b.numbers
        cdef double x=b_numbers[0], y=b_numbers[1], z=b_numbers[2], w
        if b.length == 3:
            numbers[0] = a_numbers[0]*x + a_numbers[4]*y + a_numbers[8]*z + a_numbers[12]
            numbers[1] = a_numbers[1]*x + a_numbers[5]*y + a_numbers[9]*z + a_numbers[13]
            numbers[2] = a_numbers[2]*x + a_numbers[6]*y + a_numbers[10]*z + a_numbers[14]
            result.length = 3
        else:
            numbers = result.numbers = result._numbers
            w = b_numbers[3]
            numbers[0] = a_numbers[0]*x + a_numbers[4]*y + a_numbers[8]*z + a_numbers[12]*w
            numbers[1] = a_numbers[1]*x + a_numbers[5]*y + a_numbers[9]*z + a_numbers[13]*w
            numbers[2] = a_numbers[2]*x + a_numbers[6]*y + a_numbers[10]*z + a_numbers[14]*w
            numbers[3] = a_numbers[3]*x + a_numbers[7]*y + a_numbers[11]*z + a_numbers[15]*w
            result.length = 4
        result.numbers = numbers
        return result

    def __matmul__(self, other):
        if isinstance(other, Matrix44):
            return self.mmul(<Matrix44>other)
        return self.vmul(Vector._coerce(other))

    def __imatmul__(self, other):
        if isinstance(other, Matrix44):
            return self.immul(<Matrix44>other)
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
        cdef double* result_numbers = result._numbers
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
        result.numbers = result_numbers
        result.length = 16
        return result

    cpdef Matrix44 transpose(self):
        cdef double* numbers = self.numbers
        cdef Matrix44 result = Matrix44.__new__(Matrix44)
        cdef double* result_numbers = result._numbers
        cdef int64_t i, j
        for i in range(4):
            for j in range(4):
                result_numbers[i*4+j] = numbers[j*4+i]
        result.numbers = result_numbers
        result.length = 16
        return result

    cpdef Matrix33 matrix33(self):
        cdef double* numbers = self.numbers
        cdef Matrix33 result = Matrix33.__new__(Matrix33)
        cdef double* result_numbers = result._numbers
        cdef int64_t i, j
        for i in range(3):
            for j in range(3):
                result_numbers[3*i+j] = numbers[4*i+j]
        result.numbers = result_numbers
        result.length = 9
        return result

    @cython.cdivision(True)
    cpdef Matrix33 inverse_transpose_matrix33(self):
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
        cdef Matrix33 result = Matrix33.__new__(Matrix33)
        cdef double* result_numbers = result._numbers
        result_numbers[0] = (numbers[5]*c5 - numbers[6]*c4 + numbers[7]*c3) * invdet
        result_numbers[3] = (-numbers[1]*c5 + numbers[2]*c4 - numbers[3]*c3) * invdet
        result_numbers[6] = (numbers[13]*s5 - numbers[14]*s4 + numbers[15]*s3) * invdet
        result_numbers[1] = (-numbers[4]*c5 + numbers[6]*c2 - numbers[7]*c1) * invdet
        result_numbers[4] = (numbers[0]*c5 - numbers[2]*c2 + numbers[3]*c1) * invdet
        result_numbers[7] = (-numbers[12]*s5 + numbers[14]*s2 - numbers[15]*s1) * invdet
        result_numbers[2] = (numbers[4]*c4 - numbers[5]*c2 + numbers[7]*c0) * invdet
        result_numbers[5] = (-numbers[0]*c4 + numbers[1]*c2 - numbers[3]*c0) * invdet
        result_numbers[8] = (numbers[12]*s4 - numbers[13]*s2 + numbers[15]*s0) * invdet
        result.numbers = result_numbers
        result.length = 9
        return result

    def __repr__(self):
        cdef list rows = []
        cdef double* numbers = self.numbers
        cdef int64_t i
        for i in range(4):
            rows.append(f"| {numbers[i]:7.3f} {numbers[i+4]:7.3f} {numbers[i+8]:7.3f} {numbers[i+12]:7.3f} |")
        return '\n'.join(rows)


cdef class Quaternion(Vector):
    @staticmethod
    cdef Quaternion _coerce(other):
        if isinstance(other, Quaternion):
            return <Quaternion>other
        return Quaternion(other)

    @staticmethod
    def coerce(other):
        return Quaternion._coerce(other)

    @staticmethod
    @cython.cdivision(True)
    cdef Quaternion _euler(Vector axis, double rotation):
        if axis is None or axis.numbers == NULL or axis.length != 3:
            raise ValueError("Axis must be a numeric 3-vector")
        cdef double x=axis.numbers[0], y=axis.numbers[1], z=axis.numbers[2]
        cdef double half_theta = rotation * Pi
        cdef double k = sin(half_theta) / sqrt(x*x + y*y + z*z)
        cdef Quaternion result = Quaternion.__new__(Quaternion)
        cdef double* numbers = result._numbers
        numbers[0] = cos(half_theta)
        numbers[1] = k * x
        numbers[2] = k * y
        numbers[3] = k * z
        result.numbers = numbers
        result.length = 4
        return result

    @staticmethod
    def euler(axis, rotation):
        return Quaternion._euler(Vector._coerce(axis), float(rotation))

    @staticmethod
    @cython.cdivision(True)
    cdef Quaternion _between(Vector a, Vector b):
        if a is None or a.numbers == NULL or a.length != 3:
            raise ValueError("First argument must be a numeric 3-vector")
        if b is None or b.numbers == NULL or b.length != 3:
            raise ValueError("Second argument must be a numeric 3-vector")
        cdef double ax=a.numbers[0], ay=a.numbers[1], az=a.numbers[2]
        cdef double s = 1 / sqrt(ax*ax + ay*ay + az*az)
        ax *= s
        ay *= s
        az *= s
        cdef double bx=b.numbers[0], by=b.numbers[1], bz=b.numbers[2]
        s = 1 / sqrt(bx*bx + by*by + bz*bz)
        bx *= s
        by *= s
        bz *= s
        cdef double cth = ax*bx + ay*by + az*bz
        cdef double x, y, z, k
        cdef Quaternion result = Quaternion.__new__(Quaternion)
        cdef double* numbers = result._numbers
        s = sqrt((1 + cth) * 2)
        if s != 0:
            x = ay*bz - az*by
            y = az*bx - ax*bz
            z = ax*by - ay*bx
            k = 1 / s
            numbers[0] = s / 2
            numbers[1] = x * k
            numbers[2] = y * k
            numbers[3] = z * k
        else:
            if ax != 0 or ay != 0:
                x = -ay
                y = ax
                z = 0
            else:
                x = 0
                y = -az
                z = ay
            s = 1 / sqrt(x*x + y*y + z*z)
            numbers[0] = 0
            numbers[1] = x * s
            numbers[2] = y * s
            numbers[3] = z * s
        result.numbers = numbers
        result.length = 4
        return result

    @staticmethod
    def between(a, b):
        return Quaternion._between(Vector._coerce(a), Vector._coerce(b))

    def __init__(self, value=None):
        cdef int64_t i
        cdef double k
        cdef const double[:] arr
        if type(value) is ndarray:
            arr = value.astype(*AstypeArgs)
            if arr.shape[0] != 4:
                raise ValueError("Argument must be a float or a sequence of 4 floats")
            self.numbers = self._numbers
            for i in range(4):
                self.numbers[i] = arr[i]
            self.length = 4
        elif isinstance(value, (list, tuple, bytes, set, dict, Vector)):
            if len(value) != 4:
                raise ValueError("Argument must be a float or a sequence of 4 floats")
            try:
                self.numbers = self._numbers
                for i, v in enumerate(value):
                    self.numbers[i] = v
                self.length = 4
            except TypeError:
                raise ValueError("Argument must be a float or a sequence of 4 floats")
        elif value is None or isinstance(value, (float, int)):
            k = 1 if value is None else value
            self.numbers = self._numbers
            for i in range(4):
                self.numbers[i] = 0 if i else k
            self.length = 4
        elif isinstance(value, (range, slice)):
            self.length = 0
            self.fill_range(Vector._coerce(value.start), Vector._coerce(value.stop), Vector._coerce(value.step))
            if self.length != 4:
                raise ValueError("Argument must be a float or a sequence of 4 floats")
        else:
            raise ValueError("Argument must be a float or a sequence of 4 floats")

    cpdef Quaternion copy(self):
        cdef double* self_numbers = self.numbers
        cdef Quaternion result = Quaternion.__new__(Quaternion)
        cdef double* numbers = result._numbers
        numbers[0] = self_numbers[0]
        numbers[1] = self_numbers[1]
        numbers[2] = self_numbers[2]
        numbers[3] = self_numbers[3]
        result.numbers = numbers
        result.length = 4
        return result

    cdef Quaternion mmul(self, Quaternion b):
        cdef double* a_numbers = self.numbers
        cdef double* b_numbers = b.numbers
        cdef double aw=a_numbers[0], ax=a_numbers[1], ay=a_numbers[2], az=a_numbers[3]
        cdef double bw=b_numbers[0], bx=b_numbers[1], by=b_numbers[2], bz=b_numbers[3]
        cdef Quaternion result = Quaternion.__new__(Quaternion)
        cdef double* numbers = result._numbers
        numbers[0] = aw*bw - ax*bx - ay*by - az*bz
        numbers[1] = aw*bx + ax*bw + ay*bz - az*by
        numbers[2] = aw*by + ay*bw + az*bx - ax*bz
        numbers[3] = aw*bz + az*bw + ax*by - ay*bx
        result.numbers = numbers
        result.length = 4
        return result

    @cython.cdivision(True)
    cpdef Vector conjugate(self, Vector v):
        if v is None or v.numbers == NULL or v.length != 3:
            raise ValueError("Can only conjugate a numeric 3-vector")
        cdef double x=v.numbers[0], y=v.numbers[1], z=v.numbers[2]
        cdef double* a_numbers = self.numbers
        cdef double aw=a_numbers[0], ax=a_numbers[1], ay=a_numbers[2], az=a_numbers[3]
        cdef double bw = -ax*x - ay*y - az*z
        cdef double bx = aw*x + ay*z - az*y
        cdef double by = aw*y + az*x - ax*z
        cdef double bz = aw*z + ax*y - ay*x
        cdef double s = 1 / (ax*ax + ay*ay + az*az + aw*aw)
        aw *= s
        ax *= -s
        ay *= -s
        az *= -s
        cdef Vector result = Vector.__new__(Vector)
        cdef double* numbers=result._numbers
        numbers[0] = bw*ax + bx*aw + by*az - bz*ay
        numbers[1] = bw*ay + by*aw + bz*ax - bx*az
        numbers[2] = bw*az + bz*aw + bx*ay - by*ax
        result.numbers = numbers
        result.length = 3
        return result

    def __matmul__(self, other):
        if isinstance(other, Quaternion):
            return self.mmul(<Quaternion>other)
        return self.conjugate(Vector._coerce(other))

    @cython.cdivision(True)
    cpdef Quaternion normalize(self):
        cdef double ax=self.numbers[1], ay=self.numbers[2], az=self.numbers[3], aw=self.numbers[0]
        cdef double k = sqrt(ax*ax + ay*ay + az*az + aw*aw)
        cdef Quaternion result = Quaternion.__new__(Quaternion)
        cdef double* numbers = result._numbers
        numbers[0] = aw / k
        numbers[1] = ax / k
        numbers[2] = ay / k
        numbers[3] = az / k
        result.numbers = numbers
        result.length = 4
        return result

    @cython.cdivision(True)
    cpdef Quaternion inverse(self):
        cdef double aw=self.numbers[0], ax=self.numbers[1], ay=self.numbers[2], az=self.numbers[3]
        cdef double s = 1 / (ax*ax + ay*ay + az*az + aw*aw)
        cdef Quaternion result = Quaternion.__new__(Quaternion)
        cdef double* numbers=result._numbers
        numbers[0] = aw * s
        numbers[1] = -ax * s
        numbers[2] = -ay * s
        numbers[3] = -az * s
        result.numbers = numbers
        result.length = 4
        return result

    @cython.cdivision(True)
    cpdef Quaternion exponent(self, double t):
        cdef double th=acos(self.numbers[0]), sth=sin(th)
        cdef Quaternion result = Quaternion.__new__(Quaternion)
        cdef double* numbers = result._numbers
        th *= t
        sth = sin(th) / sth if sth != 0 else 0
        numbers[1] = self.numbers[1] * sth
        numbers[2] = self.numbers[2] * sth
        numbers[3] = self.numbers[3] * sth
        numbers[0] = cos(th)
        result.numbers = numbers
        result.length = 4
        return result

    cpdef Quaternion slerp(self, Quaternion other, double t):
        return other.mmul(self.inverse()).exponent(t).mmul(self)

    @cython.cdivision(True)
    cpdef Matrix44 matrix44(self):
        cdef double* numbers = self.numbers
        cdef double w=numbers[0], x=numbers[1], y=numbers[2], z=numbers[3]
        cdef double xx=x*x, xy=x*y, xz=x*z, xw=x*w, yy=y*y, yz=y*z, yw=y*w, zz=z*z, zw=z*w, ww=w*w
        cdef double k = 2 / sqrt(ww + xx + yy + zz)
        cdef Matrix44 result = Matrix44.__new__(Matrix44)
        cdef double* result_numbers = result._numbers
        result_numbers[0] = 1 - k*(yy + zz)
        result_numbers[1] = k*(xy + zw)
        result_numbers[2] = k*(xz - yw)
        result_numbers[3] = 0
        result_numbers[4] = k*(xy - zw)
        result_numbers[5] = 1 - k*(xx + zz)
        result_numbers[6] = k*(yz + xw)
        result_numbers[7] = 0
        result_numbers[8] = k*(xz + yw)
        result_numbers[9] = k*(yz - xw)
        result_numbers[10] = 1 - k*(xx + yy)
        result_numbers[11] = 0
        result_numbers[12] = 0
        result_numbers[13] = 0
        result_numbers[14] = 0
        result_numbers[15] = 1
        result.numbers = result_numbers
        result.length = 16
        return result

    def __repr__(self):
        cdef double* numbers = self.numbers
        cdef double w=numbers[0], x=numbers[1], y=numbers[2], z=numbers[3]
        cdef str text = "nan" if isnan(w) else f"{w:g}" if abs(w) >= 1e-9 else "0"
        text += "﹢nan𝒊" if isnan(x) else f"﹣{-x:g}𝒊" if x <= -1e-9 else f"﹢{x:g}𝒊" if x >= 1e-9 else "﹢0𝒊"
        text += "﹢nan𝒋" if isnan(y) else f"﹣{-y:g}𝒋" if y <= -1e-9 else f"﹢{y:g}𝒋" if y >= 1e-9 else "﹢0𝒋"
        text += "﹢nan𝒌" if isnan(z) else f"﹣{-z:g}𝒌" if z <= -1e-9 else f"﹢{z:g}𝒌" if z >= 1e-9 else "﹢0𝒌"
        return text


@cython.final
cdef class Node:
    def __init__(self, str kind, set tags=None, dict attributes=None, tuple children=()):
        self.kind = kind
        self._tags = None if tags is None else tags.copy()
        self._attributes = None if attributes is None else attributes.copy()
        self._children = children

    def __hash__(self):
        return self.hash()

    cdef int64_t hash(self):
        cdef uint64_t _hash = HASH_START
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
        cdef object child
        if self._children is not ():
            for child in self._children:
                _hash = HASH_UPDATE(_hash, (<Node>child).hash())
        return <int64_t>_hash

    @property
    def children(self):
        cdef object child
        for child in self._children:
            yield child

    @property
    def tags(self):
        if self._tags is not None:
            return frozenset(self._tags)
        else:
            return EmptySet

    cpdef Node copy(self):
        cdef Node dst = Node.__new__(Node)
        dst.kind = self.kind
        if self._tags is not None:
            dst._tags = set(self._tags)
        if self._attributes is not None:
            dst._attributes = self._attributes
            dst._attributes_shared = True
            self._attributes_shared = True
        dst._children = self._children
        return dst

    cpdef void add_tag(self, str tag):
        cdef set tags = self._tags
        if tags is None:
            tags = set()
            self._tags = tags
        PySet_Add(tags, tag)

    cpdef void set_attribute(self, str name, Vector value):
        cdef dict attributes = self._attributes
        if attributes is None:
            self._attributes = attributes = {}
        elif self._attributes_shared:
            self._attributes = attributes = PyDict_Copy(self._attributes)
            self._attributes_shared = False
        if value.length:
            PyDict_SetItem(attributes, name, value)
        elif PyDict_GetItem(attributes, name) != NULL:
            PyDict_DelItem(attributes, name)

    cpdef void append(self, Node node):
        cdef tuple children=self._children
        cdef uint64_t i, m=PyTuple_GET_SIZE(children)
        cdef tuple new_children=PyTuple_New(m + 1)
        for i in range(m):
            nodeptr = PyTuple_GET_ITEM(children, i)
            Py_INCREF(<object>nodeptr)
            PyTuple_SET_ITEM(new_children, i, <object>nodeptr)
        Py_INCREF(node)
        PyTuple_SET_ITEM(new_children, m, node)
        self._children = new_children

    cpdef void append_vector(self, Vector nodes):
        cdef tuple children=self._children, objects=nodes.objects
        cdef uint64_t i, m=PyTuple_GET_SIZE(children), n=nodes.length, o=m
        if objects is None:
            return
        for i in range(n):
            if type(<object>PyTuple_GET_ITEM(objects, i)) is Node:
                o += 1
        cdef tuple new_children=PyTuple_New(o)
        cdef PyObject* nodeptr
        for i in range(m):
            nodeptr = PyTuple_GET_ITEM(children, i)
            Py_INCREF(<object>nodeptr)
            PyTuple_SET_ITEM(new_children, i, <object>nodeptr)
        for i in range(n):
            if type(<object>(nodeptr := PyTuple_GET_ITEM(objects, i))) is Node:
                Py_INCREF(<object>nodeptr)
                PyTuple_SET_ITEM(new_children, m, <object>nodeptr)
                m += 1
        self._children = new_children

    cdef bint _equals(self, Node other):
        if self.kind != other.kind:
            return False
        if self._tags != other._tags:
            return False
        if self._attributes != other._attributes:
            return False
        if self._children != other._children:
            return False
        return True

    def __eq__(self, Node other not None):
        return self._equals(other)

    def __len__(self):
        return len(self._attributes) if self._attributes else 0

    def __contains__(self, str name):
        return name in self._attributes if self._attributes else False

    def __getitem__(self, str name):
        return self._attributes[name]

    def keys(self):
        return self._attributes.keys() if self._attributes else ()

    def values(self):
        return self._attributes.values() if self._attributes else ()

    def items(self):
        return self._attributes.items() if self._attributes else ()

    cpdef object get(self, str name, int n=0, type t=None, object default=None):
        if self._attributes is None:
            return default
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
        if self._attributes is None:
            return default
        cdef PyObject* objptr = PyDict_GetItem(self._attributes, name)
        if objptr == NULL:
            return default
        cdef Vector result, value = <Vector>objptr
        cdef int64_t m, i
        if value.numbers != NULL:
            m = value.length
            if m == 1 and n > 1:
                result = Vector.__new__(Vector)
                for i in range(result.allocate_numbers(n)):
                    result.numbers[i] = value.numbers[0]
                return result
            elif m == n or n == 0:
                return value
        return default

    cdef double get_float(self, str name, double default):
        if self._attributes is None:
            return default
        cdef PyObject* objptr = PyDict_GetItem(self._attributes, name)
        if objptr == NULL:
            return default
        cdef Vector value = <Vector>objptr
        if value.numbers != NULL and value.length == 1:
            return value.numbers[0]
        return default

    cdef int64_t get_int(self, str name, int64_t default):
        if self._attributes is None:
            return default
        cdef PyObject* objptr = PyDict_GetItem(self._attributes, name)
        if objptr == NULL:
            return default
        cdef Vector value = <Vector>objptr
        if value.numbers != NULL and value.length == 1:
            return <int64_t>c_floor(value.numbers[0])
        return default

    cdef bint get_bool(self, str name, bint default):
        if self._attributes is None:
            return default
        cdef PyObject* objptr = PyDict_GetItem(self._attributes, name)
        if objptr == NULL:
            return default
        return (<Vector>objptr).as_bool()

    cdef str get_str(self, str name, str default):
        if self._attributes is None:
            return default
        cdef PyObject* objptr = PyDict_GetItem(self._attributes, name)
        if objptr == NULL:
            return default
        return (<Vector>objptr).as_string()

    def __iter__(self):
        return iter(self._attributes)

    cdef void repr(self, list lines, int indent):
        cdef int64_t i
        cdef str tag, key
        cdef Vector value
        cdef list parts
        parts = []
        for i in range(indent):
            parts.append("")
        parts.append("!" + self.kind)
        if self._tags:
            for tag in sorted(self._tags):
                parts.append("#" + tag)
        if self._attributes is not None:
            for key, value in self._attributes.items():
                parts.append(key + "=" + value.repr())
        lines.append(" ".join(parts))
        cdef Node child
        for child in self._children:
            child.repr(lines, indent+1)
        return

    def __repr__(self):
        cdef list lines = []
        self.repr(lines, 0)
        return '\n'.join(lines)


cdef class StateDict:
    def __cinit__(self, state=None):
        cdef Vector value_vector
        self._state = {}
        self._changed_keys = set()
        if state is not None:
            for key, value in state.items():
                value_vector = Vector._coerce(value)
                if value_vector.length:
                    self._state[Vector._coerce(key)] = value_vector

    def __reduce__(self):
        return StateDict, (self._state,)

    @property
    def changed(self):
        return bool(self._changed_keys)

    @property
    def changed_keys(self):
        return frozenset(self._changed_keys)

    def clear_changed(self):
        self._changed_keys = set()

    def with_keys(self, keys):
        cdef StateDict state = StateDict.__new__(StateDict)
        for key in keys:
            if key in self._state:
                state._state[key] = self._state[key]
        return state

    cdef Vector get_item(self, Vector key):
        return <Vector>self._state.get(key, null_)

    cdef void set_item(self, Vector key, Vector value):
        cdef Vector current = <Vector>self._state.get(key, null_)
        if value.length:
            if value.ne(current):
                self._state[key] = value
                self._changed_keys.add(key)
        elif current.length:
            del self._state[key]
            self._changed_keys.add(key)

    cdef bint contains(self, Vector key):
        return key in self._state

    def __getitem__(self, key):
        return self.get_item(Vector._coerce(key))

    def __setitem__(self, key, value):
        self.set_item(Vector._coerce(key), Vector._coerce(value))

    def __contains__(self, key):
        return self.contains(Vector._coerce(key))

    def __delitem__(self, key):
        cdef Vector key_vector = Vector._coerce(key)
        if key_vector in self._state:
            del self._state[key_vector]
            self._changed_keys.add(key_vector)

    def __len__(self):
        return len(self._state)

    def __iter__(self):
        return iter(self._state)

    def clear(self):
        self._changed_keys = set(self._state)
        self._state = {}

    def items(self):
        return self._state.items()

    def keys(self):
        return self._state.keys()

    def values(self):
        return self._state.values()

    def __repr__(self):
        return f"StateDict({self._state!r})"


cdef class DummyStateDict(StateDict):
    cdef Vector get_item(self, Vector key):
        return null_

    cdef void set_item(self, Vector key, Vector value):
        pass

    cdef bint contains(self, Vector key):
        return True


cdef class Context:
    def __init__(self, dict names=None, StateDict state=None, Node root=None,
                 object path=None, Context parent=None, dict references=None, dict modules=None,
                 dict exports=None, set errors=None, set logs=None, stack=None, lnames=None):
        self.names = names if names is not None else {}
        self.state = state
        self.root = root if root is not None else Node('root')
        self.path = path
        self.parent = parent
        self.modules = modules if modules is not None else {}
        self.exports = exports if exports is not None else {}
        self.errors = errors if errors is not None else set()
        self.logs = logs if logs is not None else set()
        self.references = references
        self.stack = stack
        self.lnames = lnames
