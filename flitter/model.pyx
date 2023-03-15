# cython: language_level=3, profile=False

from weakref import ref as weak

import cython
from cython cimport view

from libc.math cimport isnan, floor, ceil, abs, sqrt, sin, cos, tan, isnan
from cpython.mem cimport PyMem_Malloc, PyMem_Free


cdef double Pi = 3.141592653589793
cdef double Tau = 6.283185307179586
cdef double NaN = float("nan")


cdef union double_long:
    double f
    unsigned long long l


@cython.freelist(1000)
cdef class VectorLike:
    cpdef VectorLike copynodes(self):
        raise NotImplementedError()

    cdef Vector slice(self, Vector index):
        raise NotImplementedError()

    cdef bint as_bool(self):
        raise NotImplementedError()


cdef class Vector:
    @staticmethod
    def coerce(other):
        return Vector._coerce(other)

    @staticmethod
    cdef Vector _coerce(object other):
        if isinstance(other, Vector):
            return other
        if other is None or (isinstance(other, (list, tuple)) and len(other) == 0):
            return null_
        if isinstance(other, (int, float, bool)):
            if other == 0:
                return false_
            if other == 1:
                return true_
            if other == -1:
                return minusone_
        return Vector.__new__(Vector, other)

    @staticmethod
    def compose(args):
        return Vector._compose(args if isinstance(list, args) else list(args))

    @staticmethod
    cdef VectorLike _compose(list args):
        if len(args) == 1:
            return args[0]
        if len(args) == 0:
            return null_
        cdef list objects
        cdef Vector v
        cdef int i, j, n = 0
        cdef Vector result = Vector.__new__(Vector)
        for v in args:
            if v.objects is not None:
                break
            n += v.length
        else:
            if n == 0:
                return null_
            result.allocate_numbers(n)
            i = 0
            for v in args:
                for j in range(v.length):
                    result.numbers[i] = v.numbers[j]
                    i += 1
            return result
        objects = []
        for v in args:
            if v.objects is None:
                for j in range(v.length):
                    objects.append(v.numbers[j])
            else:
                objects.extend(v.objects)
        result.objects = objects
        result.length = len(objects)
        return result

    @staticmethod
    def range(*args):
        cdef Vector result = Vector.__new__(Vector)
        if len(args) == 1:
            result.fill_range(None, args[0], None)
        elif len(args) == 2:
            result.fill_range(args[0], args[1], None)
        elif len(args) == 3:
            result.fill_range(args[0], args[1], args[2])
        else:
            raise TypeError("range takes 1-3 arguments")
        return result

    def __cinit__(self, value=None):
        self.length = 0
        self.objects = None
        self.numbers = NULL
        if value is None:
            return
        cdef int i, n
        if isinstance(value, (range, slice)):
            self.fill_range(value.start, value.stop, value.step)
        elif isinstance(value, (int, float, bool)):
            self.allocate_numbers(1)
            self.numbers[0] = value
        elif isinstance(value, (str, Node)) or not hasattr(value, '__len__'):
            self.objects = [value]
            self.length = 1
        else:
            n = len(value)
            if n:
                try:
                    for i in range(self.allocate_numbers(n)):
                        self.numbers[i] = value[i]
                except TypeError:
                    self.deallocate_numbers()
                    self.objects = list(value)

    cdef int allocate_numbers(self, int n) except -1:
        assert n > 0
        self.numbers = <double*>PyMem_Malloc(n * sizeof(double))
        if not self.numbers:
            raise MemoryError()
        self.length = n
        return n

    cdef void deallocate_numbers(self):
        if self.numbers:
            PyMem_Free(self.numbers)
            self.numbers = NULL

    @property
    def numeric(self):
        return self.numbers != NULL

    @cython.cdivision(True)
    cdef bint fill_range(self, startv, stopv, stepv) except False:
        assert self.length == 0
        cdef double start = startv if startv is not None else NaN
        cdef double stop = stopv if stopv is not None else NaN
        cdef double step = stepv if stepv is not None else NaN
        if isnan(start):
            start = 0
        if isnan(stop):
            return True
        if isnan(step):
            step = 1
        elif step == 0:
            return True
        cdef int i, n = <int>ceil((stop - start) / step)
        if n > 0:
            for i in range(self.allocate_numbers(n)):
                self.numbers[i] = start + step * i
        return True

    def __dealloc__(self):
        self.deallocate_numbers()

    def __len__(self):
        return self.length

    cpdef bint isinstance(self, t):
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
        cdef int i
        if self.objects is not None:
            for value in self.objects:
                if isinstance(value, int):
                    if <int>value != 0:
                        return True
                elif isinstance(value, float):
                    if <double>value != 0.:
                        return True
                elif isinstance(value, str):
                    if <str>value != "":
                        return True
                else:
                    return True
        else:
            for i in range(self.length):
                if self.numbers[i] != 0.:
                    return True
        return False

    def __float__(self):
        return self.as_double()

    cdef double as_double(self):
        if self.length == 1 and self.objects is None:
            return self.numbers[0]
        return NaN

    def __str__(self):
        return self.as_string()

    cdef str as_string(self):
        cdef str text = ""
        cdef int i, n = self.length
        if n:
            if self.objects is not None:
                for value in self.objects:
                    if isinstance(value, str):
                        text += <str>value
                    elif isinstance(value, (float, int, bool)):
                        text += f"{value:g}"
            else:
                for i in range(n):
                    text += f"{self.numbers[i]:g}"
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
        # Compute a hash value using the SplitMix64 algorithm [http://xoshiro.di.unimi.it/splitmix64.c]
        cdef unsigned long long y, hash = 0xe220a8397b1dcdaf
        cdef double_long fl
        cdef int i
        cdef Py_UNICODE c
        if self.length:
            for i in range(self.length):
                if self.objects is not None:
                    value = self.objects[i]
                    if isinstance(value, str):
                        # FNV-1a hash algorithm [https://en.wikipedia.org/wiki/Fowler–Noll–Vo_hash_function#FNV-1a_hash]
                        y = <unsigned long long>(0xcbf29ce484222325)
                        for c in <str>value:
                            y = (y ^ ord(c)) * <unsigned long long>(0x100000001b3)
                    elif isinstance(value, int):
                        y = <unsigned long long>(<long long>value)
                    elif isinstance(value, float):
                        if floor_floats:
                            y = <unsigned long long>(<long long>floor(value))
                        else:
                            fl.f = value
                            y = fl.l
                    else:
                        raise TypeError("Unhashable value")
                elif floor_floats:
                    y = <unsigned long long>(<long long>floor(self.numbers[i]))
                else:
                    fl.f = self.numbers[i]
                    y = fl.l
                hash ^= y
                hash += <unsigned long long>(0x9e3779b97f4a7c15)
                hash ^= hash >> 30
                hash *= <unsigned long long>(0xbf58476d1ce4e5b9)
                hash ^= hash >> 27
                hash *= <unsigned long long>(0x94d049bb133111eb)
                hash ^= hash >> 31
        return hash

    cpdef object match(self, int n=0, type t=None, default=None):
        cdef int i, m = self.length
        cdef list values
        cdef double f
        if self.objects is None:
            if t is float:
                t = None
            if t is None or t is int or t is bool:
                if n == 0 or n == m:
                    if n == 1:
                        f = self.numbers[0]
                        return f if t is None else t(f)
                    else:
                        values = []
                        for i in range(m):
                            f = self.numbers[i]
                            values.append(f if t is None else t(f))
                        return values
                elif m == 1:
                    values = []
                    f = self.numbers[0]
                    obj = f if t is None else t(f)
                    for i in range(n):
                        values.append(obj)
                    return values
            return default
        if n == 0 and t is None:
            return self.objects
        try:
            if m == 1:
                value = self.objects[0]
                if t is not None:
                    value = t(value)
                if n == 1:
                    return value
                if n == 0:
                    return [value]
                return [value] * n
            elif m == n or n == 0:
                values = []
                for value in self.objects:
                    if t is not None:
                        value = t(value)
                    values.append(value)
                return values
        except ValueError:
            pass
        return default

    cpdef VectorLike copynodes(self):
        cdef Vector result = self
        cdef Node node
        cdef int i
        if self.objects is not None:
            for i, value in enumerate(self.objects):
                if isinstance(value, Node):
                    node = value
                    if node._parent is None:
                        if result is self:
                            result = Vector.__new__(Vector)
                            result.objects = list(self.objects)
                            result.length = self.length
                        result.objects[i] = node.copy()
        return result

    def __repr__(self):
        cdef int i, n = self.length
        cdef str s
        cdef double f
        cdef Py_UNICODE c
        if n == 0:
            return "null"
        if self.numbers != NULL:
            return ";".join(f"{self.numbers[i]:g}" for i in range(n))
        cdef list parts = []
        for obj in self.objects:
            if isinstance(obj, float):
                f = obj
                parts.append(f"{f:g}")
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
                    parts.append(f":{s}")
            else:
                parts.append(repr(obj))
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

    cdef Vector pos(self):
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

    def __truediv__(Vector self, other):
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
        return self.eq(Vector._coerce(other)).numbers[0] != 0

    cdef Vector eq(self, Vector other):
        if self is other:
            return true_
        cdef int i, n = self.length, m = other.length
        if n != m or (self.objects is None) != (other.objects is None):
            return false_
        if self.objects is None:
            for i in range(n):
                if self.numbers[i] != other.numbers[i]:
                    return false_
        else:
            for i in range(n):
                if self.objects[i] != other.objects[i]:
                    return false_
        return true_

    def __ne__(self, other):
        return self.ne(Vector._coerce(other)).numbers[0] != 0

    cdef Vector ne(self, Vector other):
        if self is other:
            return false_
        cdef int i, n = self.length, m = other.length
        if n != m or (self.objects is None) != (other.objects is None):
            return true_
        if self.objects is None:
            for i in range(n):
                if self.numbers[i] != other.numbers[i]:
                    return true_
        else:
            for i in range(n):
                if self.objects[i] != other.objects[i]:
                    return true_
        return false_

    def __gt__(self, other):
        return self.gt(Vector._coerce(other)).numbers[0] != 0

    cdef Vector gt(self, Vector other):
        return true_ if self.compare(other) == 1 else false_

    def __ge__(self, other):
        return self.ge(Vector._coerce(other)).numbers[0] != 0

    cdef Vector ge(self, Vector other):
        return true_ if self.compare(other) != -1 else false_

    def __lt__(self, other):
        return self.lt(Vector._coerce(other)).numbers[0] != 0

    cdef Vector lt(self, Vector other):
        return true_ if self.compare(other) == -1 else false_

    def __le__(self, other):
        return self.le(Vector._coerce(other)).numbers[0] != 0

    cdef Vector le(self, Vector other):
        return true_ if self.compare(other) != 1 else false_

    cdef int compare(self, Vector other) except -2:
        if self is other:
            return 0
        cdef int i, n = self.length, m = other.length
        cdef double x, y
        if self.objects is None and other.objects is None:
            for i in range(min(n, m)):
                x, y = self.numbers[i], other.numbers[i]
                if x < y:
                    return -1
                if x > y:
                    return 1
        else:
            for i in range(min(n, m)):
                a = self.objects[i] if self.objects is not None else self.numbers[i]
                b = other.objects[i] if other.objects is not None else other.numbers[i]
                if a < b:
                    return -1
                if a > b:
                    return 1
        if n < m:
            return -1
        if n > m:
            return 1
        return 0

    def __getitem__(self, index):
        cdef Vector result = self.slice(Vector._coerce(index))
        if result.length == 1:
            return result.objects[0] if result.objects is not None else result.numbers[0]
        return result

    cdef Vector slice(self, Vector index):
        cdef int i, j, m = 0, n = self.length
        if index.numbers == NULL:
            return null_
        cdef list values
        cdef Vector result = Vector.__new__(Vector)
        if self.objects is not None:
            result.objects = []
            for i in range(index.length):
                j = <int>floor(index.numbers[i])
                if j >= 0 and j < n:
                    result.objects.append(self.objects[j])
            result.length = len(result.objects)
        else:
            result.allocate_numbers(index.length)
            for i in range(index.length):
                j = <int>floor(index.numbers[i])
                if j >= 0 and j < n:
                    result.numbers[m] = self.numbers[j]
                    m += 1
            result.length = m
            if m == 0:
                result.deallocate_numbers()
        return result

    cdef Vector item(self, int i):
        cdef int n = self.length
        if i < 0 or i >= n:
            return null_
        cdef Vector result = Vector.__new__(Vector)
        if self.objects is not None:
            value = self.objects[i]
            if isinstance(value, (int, float)):
                result.allocate_numbers(1)
                result.numbers[0] = value
            else:
                result.objects = [self.objects[i]]
                result.length == 1
        else:
            result.allocate_numbers(1)
            result.numbers[0] = self.numbers[i]
        return result

    @cython.cdivision(True)
    cdef Vector normalize(self):
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
    cdef double dot(self, Vector other):
        cdef int i, n = self.length, m = other.length
        cdef Vector result = Vector.__new__(Vector)
        cdef double sum = 0
        if self.numbers != NULL and other.numbers != NULL:
            for i in range(result.allocate_numbers(max(n, m))):
                sum += self.numbers[i % n] * other.numbers[i % m]
            result.allocate_numbers(1)
            result.numbers[0] = sum
        return result

    cdef Vector cross(self, Vector other):
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


cdef Vector null_ = Vector()
cdef Vector true_ = Vector(1)
cdef Vector false_ = Vector(0)
cdef Vector minusone_ = Vector(-1)

null = null_
true = true_
false = false_


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

    @staticmethod
    cdef Matrix44 _look(Vector from_position, Vector to_position, Vector up_direction):
        cdef Vector z = from_position.sub(to_position).normalize()
        cdef Vector x = up_direction.normalize().cross(z)
        cdef Vector y = z.cross(x)
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
        if v.numbers is not NULL and v.length in (1, 3):
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
        if v.numbers is not NULL and v.length in (1, 3):
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
        cdef Matrix44 result = None
        if v.numbers is not NULL and v.length in (1, 3):
            if v.length == 1:
                result = Matrix44._rotate_z(v.numbers[0])
                result = Matrix44._rotate_y(v.numbers[0]).mmul(result)
                result = Matrix44._rotate_x(v.numbers[0]).mmul(result)
            else:
                result = Matrix44._rotate_z(v.numbers[2])
                result = Matrix44._rotate_y(v.numbers[1]).mmul(result)
                result = Matrix44._rotate_x(v.numbers[0]).mmul(result)
        return result

    @staticmethod
    def rotate(v):
        return Matrix44._rotate(Vector._coerce(v))

    @cython.cdivision(True)
    def __cinit__(self, obj=None):
        if self.objects is not None or self.length not in (0, 1, 16):
            raise ValueError("Argument must be a float or a sequence of 16 floats")
        if self.length == 16:
            return
        cdef double k = 1
        if self.length == 1:
            k = self.numbers[0]
            self.deallocate_numbers()
        self.allocate_numbers(16)
        cdef int i
        for i in range(16):
            self.numbers[i] = k if i % 5 == 0 else 0

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

    def __matmul__(Matrix44 self, other):
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

    cpdef Vector matrix33(self):
        cdef double* numbers = self.numbers
        cdef Vector result = Vector.__new__(Vector)
        result.allocate_numbers(9)
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


@cython.final
@cython.freelist(100)
cdef class Query:
    def __cinit__(self, str query=None, *, kind=None, tags=None):
        if query is None:
            self.kind = kind
            if tags is None:
                pass
            elif isinstance(tags, str):
                self.tags = frozenset(tags.split())
            else:
                self.tags = frozenset(tags)
            return
        cdef int i = 0, j, n = len(query)
        while n > 0 and query[n-1] in ' \t\r\n':
            n -= 1
        while i < n:
            if not self.strict and query[i] == '>':
                self.strict = True
            elif query[i] not in ' \t\r\n':
                break
            i += 1
        if i == n or query[i] in ('.', '|', '>'):
            raise ValueError("Bad query; contains empty element")
        cdef list tag_list = []
        for j in range(i, n+1):
            if j == n or query[j] in '#.|> ':
                if j > i:
                    if query[i] == '#':
                        if i+1 < j:
                            tag_list.append(query[i+1:j])
                    elif query[i] != '*':
                        self.kind = query[i:j]
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
            self.subquery = Query.__new__(Query, query[i:n])

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
                text += '#' + '#'.join(self.tags)
        if self.stop:
            text += '.'
        if self.altquery is not None:
            text += '|' + str(self.altquery)
        if self.subquery is not None:
            text += ' ' + str(self.subquery)
        return text

    def __repr__(self):
        return f"Query({str(self)!r})"


@cython.final
@cython.freelist(1000)
cdef class Node:
    def __cinit__(self, str kind, set tags=None, dict attributes=None):
        self.kind = kind
        self._tags = set() if tags is None else tags.copy()
        self._attributes = {} if attributes is None else attributes.copy()

    def __setstate__(self, set tags):
        if tags is not None:
            self._tags = tags

    def __reduce__(self):
        return Node, (self.kind,), self._tags or None, self.children, self.attributes

    @property
    def children(self):
        cdef Node node = self.first_child
        while node is not None:
            yield node
            node = node.next_sibling

    @property
    def attributes(self):
        cdef str key
        cdef Vector value
        for key, value in self._attributes.items():
            yield key, list(value)

    @property
    def tags(self):
        return frozenset(self._tags)

    @property
    def parent(self):
        return self._parent() if self._parent is not None else None

    cpdef Node copy(self):
        cdef Node node = Node.__new__(Node, self.kind, self._tags, self._attributes)
        cdef Node copy, child = self.first_child
        if child is not None:
            parent = node.weak_self = weak(node)
            copy = child.copy()
            copy._parent = parent
            node.first_child = node.last_child = copy
            child = child.next_sibling
            while child is not None:
                copy = child.copy()
                copy._parent = parent
                node.last_child.next_sibling = copy
                node.last_child = copy
                child = child.next_sibling
        return node

    cpdef void add_tag(self, str tag):
        self._tags.add(tag)

    cpdef void remove_tag(self, str tag):
        self._tags.discard(tag)

    cpdef void append(self, Node node):
        cdef Node parent = node._parent() if node._parent is not None else None
        if parent is not None:
            parent.remove(node)
        if self.weak_self is None:
            self.weak_self = weak(self)
        node._parent = self.weak_self
        if self.last_child is not None:
            self.last_child.next_sibling = node
            self.last_child = node
        else:
            self.first_child = self.last_child = node

    cpdef void insert(self, Node node):
        cdef Node parent = node._parent() if node._parent is not None else None
        if parent is not None:
            parent.remove(node)
        if self.weak_self is None:
            self.weak_self = weak(self)
        node._parent = self.weak_self
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
        cdef Node parent = node._parent() if node._parent is not None else None
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
        cdef Node parent = self._parent() if self._parent is not None else None
        if parent is None:
            raise TypeError("No parent")
        parent.remove(self)

    def select(self, query):
        cdef list nodes = []
        if not isinstance(query, Query):
            query = Query.__new__(Query, query)
        self._select(query, nodes, False)
        return nodes

    def select_below(self, query):
        cdef list nodes = []
        if not isinstance(query, Query):
            query = Query.__new__(Query, query)
        cdef Node node = self.first_child
        while node is not None:
            node._select(query, nodes, False)
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
               (tags is None or tags.issubset(self._tags)):
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
        self._attributes[name] = Vector.coerce(value)

    def keys(self):
        return self._attributes.keys()

    def values(self):
        return self._attributes.values()

    def items(self):
        return self._attributes.items()

    cpdef object get(self, str name, int n=0, type t=None, object default=None):
        cdef Vector value = self._attributes.get(name)
        if value is None:
            return default
        if n == 1:
            if t is bool:
                return value.as_bool()
            if t is str:
                return value.as_string()
        return value.match(n, t, default)

    cpdef void pprint(self, int indent=0):
        cdef str tag, key
        cdef Vector value
        cdef list text = [f"!{self.kind}"]
        for tag in self._tags:
            text.append(f"#{tag}")
        for key, value in self._attributes.items():
            text.append(f"{key}={value!r}")
        print(" " * indent + " ".join(text))
        cdef Node node = self.first_child
        while node is not None:
            node.pprint(indent+4)
            node = node.next_sibling

    def __iter__(self):
        return iter(self._attributes)

    def __repr__(self):
        return f"Node({self.kind!r})"


cdef class Context:
    def __cinit__(self, dict variables=None, dict state=None, Node graph=None, dict pragmas=None):
        self.variables = variables if variables is not None else {}
        self.state = state if state is not None else {}
        self.graph = graph if graph is not None else Node.__new__(Node, 'root')
        self.pragmas = pragmas if pragmas is not None else {}
        self._stack = []

    def __enter__(self):
        self._stack.append(self.variables)
        self.variables = self.variables.copy()
        return self

    def __exit__(self, *args):
        self.variables = self._stack.pop()

    def merge_under(self, Node node):
        for attr, value in node.attributes.items():
            if attr not in self.variables:
                self.variables[attr] = value

    def pragma(self, str name, value):
        self.pragmas[name] = value
