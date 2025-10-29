
import cython

from libc.stdint cimport int64_t, uint64_t
from cpython.unicode cimport PyUnicode_DATA, PyUnicode_GET_LENGTH, PyUnicode_KIND, PyUnicode_READ


cdef double sint(double t) noexcept nogil

cdef inline double cost(double t) noexcept nogil:
    return sint(t + 0.25)


# SplitMix64 algorithm [http://xoshiro.di.unimi.it/splitmix64.c]
#
cdef uint64_t HASH_START

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
cdef inline uint64_t HASH_STRING(str value) noexcept:
    cdef void* data = PyUnicode_DATA(value)
    cdef uint64_t i, n=PyUnicode_GET_LENGTH(value), kind=PyUnicode_KIND(value)
    cdef Py_UCS4 c
    cdef uint64_t y = <uint64_t>(0xcbf29ce484222325)
    for i in range(n):
        c = PyUnicode_READ(kind, data, i)
        y = (y ^ <uint64_t>c) * <uint64_t>(0x100000001b3)
    return y


cdef union double_long:
    double f
    uint64_t l


cdef class Vector:
    cdef int64_t length
    cdef tuple objects
    cdef double* numbers
    cdef double[16] _numbers
    cdef uint64_t _hash

    @staticmethod
    cdef Vector _coerce(object other)

    @staticmethod
    cdef Vector _compose(list vectors)

    @staticmethod
    cdef Vector _symbol(str symbol)

    cdef int64_t allocate_numbers(self, int64_t n) except -1
    cdef void deallocate_numbers(self) noexcept
    cpdef Vector copy(self)
    cpdef Vector intern(self)
    cdef void fill_range(self, Vector startv, Vector stopv, Vector stepv)
    cpdef bint isinstance(self, t) noexcept
    cpdef bint is_finite(self) noexcept
    cdef bint as_bool(self)
    cdef double as_double(self) noexcept
    cdef int64_t as_integer(self) noexcept
    cdef str as_string(self)
    cpdef uint64_t hash(self, bint floor_floats)
    cpdef object match(self, int64_t n=?, type t=?, default=?)
    cdef str repr(self)
    cdef Vector neg(self)
    cdef Vector pos(self)
    cdef Vector abs(self)
    cdef Vector ceil(self)
    cdef Vector floor(self)
    cpdef Vector fract(self)
    cdef Vector round(self, int64_t ndigits)
    cpdef Vector contains(self, Vector other)
    cdef Vector add(self, Vector other)
    cpdef Vector mul_add(self, Vector left, Vector right)
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
    cpdef int64_t compare(self, Vector other) noexcept
    cpdef Vector slice(self, Vector index)
    cpdef Vector item(self, int64_t i)
    cpdef double squared_sum(self) noexcept
    cpdef Vector normalize(self)
    cpdef Vector dot(self, Vector other)
    cpdef Vector cross(self, Vector other)
    cpdef Vector clamp(self, Vector minimum, Vector maximum)
    cpdef double minimum(self) noexcept
    cpdef double maximum(self) noexcept
    cpdef Vector concat(self, Vector other)


cdef Vector null_
cdef Vector true_
cdef Vector false_
cdef Vector minusone_
cdef Vector inf_
cdef Vector nan_


cdef class Matrix33(Vector):
    @staticmethod
    cdef Matrix33 _identity()

    @staticmethod
    cdef Matrix33 _translate(Vector v)

    @staticmethod
    cdef Matrix33 _scale(Vector v)

    @staticmethod
    cdef Matrix33 _rotate(double turns)

    cpdef Matrix33 copy(self)
    cdef Matrix33 mmul(self, Matrix33 b)
    cdef Vector vmul(self, Vector b)
    cpdef double det(self)
    cpdef Matrix33 inverse(self)
    cpdef Matrix33 cofactor(self)
    cpdef Matrix33 transpose(self)
    cpdef Matrix44 matrix44(self)


cdef class Matrix44(Vector):
    @staticmethod
    cdef Matrix44 _identity()

    @staticmethod
    cdef Matrix44 _project(double xgradient, double ygradient, double near, double far)

    @staticmethod
    cdef Matrix44 _ortho(double aspect_ratio, double width, double near, double far)

    @staticmethod
    cdef Matrix44 _look(Vector from_position, Vector to_position, Vector up_direction)

    @staticmethod
    cdef Matrix44 _translate(Vector v)

    @staticmethod
    cdef Matrix44 _scale(Vector v)

    @staticmethod
    cdef Matrix44 _rotate_x(double turns)

    @staticmethod
    cdef Matrix44 _rotate_y(double turns)

    @staticmethod
    cdef Matrix44 _rotate_z(double turns)

    @staticmethod
    cdef Matrix44 _rotate(Vector v)

    @staticmethod
    cdef Matrix44 _shear_x(Vector v)

    @staticmethod
    cdef Matrix44 _shear_y(Vector v)

    @staticmethod
    cdef Matrix44 _shear_z(Vector v)

    cpdef Matrix44 copy(self)
    cdef Matrix44 mmul(self, Matrix44 b)
    cdef Matrix44 immul(self, Matrix44 b)
    cdef Vector vmul(self, Vector b)
    cpdef Matrix44 inverse(self)
    cpdef Matrix44 transpose(self)
    cpdef Matrix33 inverse_transpose_matrix33(self)
    cpdef Matrix33 matrix33_cofactor(self)
    cpdef Matrix33 matrix33(self)


cdef class Quaternion(Vector):
    @staticmethod
    cdef Quaternion _coerce(other)

    @staticmethod
    cdef Quaternion _euler(Vector axis, double rotation)

    @staticmethod
    cdef Quaternion _between(Vector a, Vector b)

    cpdef Quaternion copy(self)
    cdef Quaternion mmul(self, Quaternion b)
    cpdef Vector conjugate(self, Vector v)
    cpdef Quaternion normalize(self)
    cpdef Quaternion inverse(self)
    cpdef Quaternion exponent(self, double t)
    cpdef Quaternion slerp(self, Quaternion other, double t)
    cpdef Matrix44 matrix44(self)


cdef class Node:
    cdef readonly str kind
    cdef set _tags
    cdef dict _attributes
    cdef bint _attributes_shared
    cdef tuple _children

    cdef uint64_t hash(self)
    cpdef Node copy(self)
    cpdef void add_tag(self, str tag)
    cpdef void set_attribute(self, str name, Vector value)
    cpdef void append(self, Node node)
    cpdef void append_vector(self, Vector nodes)
    cdef bint _equals(self, Node other)
    cpdef object get(self, str name, int n=?, type t=?, object default=?)
    cdef Vector get_vec(self, str name, Vector default)
    cdef Vector get_fvec(self, str name, int n, Vector default)
    cdef double get_float(self, str name, double default)
    cdef int64_t get_int(self, str name, int64_t default)
    cdef bint get_bool(self, str name, bint default)
    cdef str get_str(self, str name, str default)
    cdef void repr(self, list lines, int indent)


cdef class StateDict:
    cdef set _changed_keys
    cdef dict _state

    cdef Vector get_item(self, Vector key)
    cdef void set_item(self, Vector key, Vector value)
    cdef bint contains(self, Vector key)


cdef class Context:
    cdef readonly dict names
    cdef readonly StateDict state
    cdef readonly Node root
    cdef readonly object path
    cdef readonly Context parent
    cdef readonly dict exports
    cdef readonly set errors
    cdef readonly set logs
    cdef readonly dict references
    cdef readonly object stack
    cdef readonly object lnames
    cdef readonly set dependencies
    cdef readonly int64_t call_depth
    cdef readonly bint is_include
    cdef readonly set stables
    cdef readonly dict stable_cache
    cdef readonly bint stables_changed
