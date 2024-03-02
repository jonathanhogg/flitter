# cython: language_level=3, profile=True

import cython

from libc.stdint cimport int64_t, uint64_t


cdef class Vector:
    cdef int64_t length
    cdef tuple objects
    cdef double* numbers
    cdef double[16] _numbers
    cdef uint64_t _hash

    @staticmethod
    cdef Vector _coerce(object other)

    @staticmethod
    cdef Vector _copy(Vector other)

    @staticmethod
    cdef Vector _compose(list vectors)

    @staticmethod
    cdef Vector _symbol(str symbol)

    cdef int64_t allocate_numbers(self, int64_t n) except -1
    cdef void deallocate_numbers(self) noexcept
    cpdef Vector intern(self)
    cdef void fill_range(self, Vector startv, Vector stopv, Vector stepv)
    cpdef bint isinstance(self, t) noexcept
    cdef bint as_bool(self)
    cdef double as_double(self) noexcept
    cdef str as_string(self)
    cdef uint64_t hash(self, bint floor_floats)
    cpdef object match(self, int64_t n=?, type t=?, default=?)
    cdef str repr(self)
    cdef Vector neg(self)
    cdef Vector pos(self) noexcept
    cdef Vector abs(self)
    cdef Vector ceil(self)
    cdef Vector floor(self)
    cpdef Vector fract(self)
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
    cdef int64_t compare(self, Vector other) except -2
    cdef Vector slice(self, Vector index)
    cdef Vector item(self, int64_t i)
    cpdef double squared_sum(self) noexcept
    cpdef Vector normalize(self)
    cpdef Vector dot(self, Vector other)
    cpdef Vector cross(self, Vector other)
    cpdef Vector clamp(self, Vector minimum, Vector maximum)
    cpdef Vector concat(self, Vector other)


cdef Vector null_
cdef Vector true_
cdef Vector false_
cdef Vector minusone_


cdef class Matrix33(Vector):
    @staticmethod
    cdef Matrix33 _translate(Vector v)

    @staticmethod
    cdef Matrix33 _scale(Vector v)

    @staticmethod
    cdef Matrix33 _rotate(double turns)

    cdef Matrix33 mmul(self, Matrix33 b)
    cdef Vector vmul(self, Vector b)
    cpdef Matrix33 inverse(self)
    cpdef Matrix33 transpose(self)


cdef class Matrix44(Vector):
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

    cdef Matrix44 mmul(self, Matrix44 b)
    cdef Vector vmul(self, Vector b)
    cpdef Matrix44 inverse(self)
    cpdef Matrix44 transpose(self)
    cpdef Matrix33 inverse_transpose_matrix33(self)
    cpdef Matrix33 matrix33(self)


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
    cdef bint _equals(self, Node other)
    cpdef object get(self, str name, int n=?, type t=?, object default=?)
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
    cdef readonly set captures
    cdef readonly dict pragmas
    cdef readonly StateDict state
    cdef readonly Node root
    cdef readonly object path
    cdef readonly Context parent
    cdef readonly set errors
    cdef readonly set logs
    cdef readonly dict references
    cdef readonly object stack
    cdef readonly object lnames
