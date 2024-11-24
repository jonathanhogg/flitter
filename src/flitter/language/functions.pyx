
"""
Flitter language functions
"""


import cython

from libc.math cimport floor, sin, cos, tan, asin, acos, sqrt, exp, atan2, log, log2, log10
from libc.stdint cimport int64_t, uint64_t
from cpython.object cimport PyObject
from cpython.ref cimport Py_INCREF
from cpython.tuple cimport PyTuple_New, PyTuple_GET_ITEM, PyTuple_SET_ITEM

from ..cache import SharedCache
from ..model cimport Vector, Matrix33, Matrix44, Quaternion, Context, null_, true_, false_


cdef double Pi = 3.141592653589793115997963468544185161590576171875
cdef double Tau = 6.283185307179586231995926937088370323181152343750


def context_func(func):
    func.context_func = True
    return func


@context_func
def debug(Context context, Vector value):
    context.logs.add(value.repr())
    return value


@context_func
@cython.boundscheck(False)
def sample(Context context, Vector texture_id, Vector coord, Vector default=null_):
    cdef const float[:, :, :] data
    if coord.numbers == NULL \
            or (scene_node := context.references.get(texture_id.as_string())) is None \
            or not hasattr(scene_node, 'array') or (data := scene_node.array) is None:
        return default
    cdef int64_t x, y, height=data.shape[0], width=data.shape[1]
    cdef const float[:] color
    cdef int64_t i, j, n=coord.length // 2
    cdef Vector result = Vector.__new__(Vector)
    result.allocate_numbers(n * 4)
    for i in range(n):
        j = i * 2
        x = int(coord.numbers[j] * width)
        y = int(coord.numbers[j+1] * height)
        j = i * 4
        if x < 0 or x >= width or y < 0 or y >= height:
            if n == 1:
                return default
            if default.numbers != NULL and default.length == 4:
                result.numbers[j] = default.numbers[0]
                result.numbers[j+1] = default.numbers[1]
                result.numbers[j+2] = default.numbers[2]
                result.numbers[j+3] = default.numbers[3]
            else:
                result.numbers[j] = 0
                result.numbers[j+1] = 0
                result.numbers[j+2] = 0
                result.numbers[j+3] = 0
        else:
            color = data[y, x]
            result.numbers[j] = color[0]
            result.numbers[j+1] = color[1]
            result.numbers[j+2] = color[2]
            result.numbers[j+3] = color[3]
    return result


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
            self.th = Tau * u2
            self.i = i
            self.cached = True
        if odd:
            return self.R * sin(self.th)
        return self.R * cos(self.th)


@context_func
def glob(Context context, Vector pattern):
    return Vector._coerce(sorted(str(p.resolve()) for p in context.path.parent.glob(pattern.as_string())))


@context_func
def read_text(Context context, Vector filename):
    cdef str path = filename.as_string()
    if path:
        return Vector._coerce(SharedCache.get_with_root(path, context.path).read_text(encoding='utf8'))
    return null_


@context_func
def read_bytes(Context context, Vector filename):
    cdef str path = filename.as_string()
    if path:
        return Vector._coerce(SharedCache.get_with_root(path, context.path).read_bytes())
    return null_


@context_func
def read_csv(Context context, Vector filename, Vector row_number):
    cdef str path = str(filename)
    row = row_number.match(1, int)
    if filename and row is not None:
        return SharedCache.get_with_root(path, context.path).read_csv_vector(row)
    return null_


def ordv(Vector text):
    cdef str string = text.as_string()
    return Vector._coerce([ord(ch) for ch in string])


def chrv(Vector ordinals):
    if ordinals.numbers == NULL:
        return null_
    cdef str text = ""
    cdef int64_t i
    for i in range(ordinals.length):
        text += chr(<int>ordinals.numbers[i])
    cdef Vector result = Vector.__new__(Vector)
    result.objects = (text,)
    result.length = 1
    return result


def split(Vector text):
    if text.length == 0:
        return null_
    return Vector._coerce(text.as_string().rstrip('\n').split('\n'))


def lenv(Vector xs not None):
    cdef Vector ys = Vector.__new__(Vector)
    ys.allocate_numbers(1)
    ys.numbers[0] = xs.length
    return ys


def sinv(Vector theta not None):
    if theta.numbers == NULL:
        return null_
    cdef Vector ys = Vector.__new__(Vector)
    cdef int64_t i, n = theta.length
    for i in range(ys.allocate_numbers(n)):
        ys.numbers[i] = sin(theta.numbers[i] * Tau)
    return ys


def cosv(Vector theta not None):
    if theta.numbers == NULL:
        return null_
    cdef Vector ys = Vector.__new__(Vector)
    cdef int64_t i, n = theta.length
    for i in range(ys.allocate_numbers(n)):
        ys.numbers[i] = cos(theta.numbers[i] * Tau)
    return ys


def tanv(Vector theta not None):
    if theta.numbers == NULL:
        return null_
    cdef Vector ys = Vector.__new__(Vector)
    cdef int64_t i, n = theta.length
    for i in range(ys.allocate_numbers(n)):
        ys.numbers[i] = tan(theta.numbers[i] * Tau)
    return ys


def asinv(Vector ys not None):
    if ys.numbers == NULL:
        return null_
    cdef Vector theta = Vector.__new__(Vector)
    cdef int64_t i, n = ys.length
    for i in range(theta.allocate_numbers(n)):
        theta.numbers[i] = asin(ys.numbers[i]) / Tau
    return theta


def acosv(Vector ys not None):
    if ys.numbers == NULL:
        return null_
    cdef Vector theta = Vector.__new__(Vector)
    cdef int64_t i, n = ys.length
    for i in range(theta.allocate_numbers(n)):
        theta.numbers[i] = acos(ys.numbers[i]) / Tau
    return theta


def polar(Vector theta not None):
    if theta.numbers == NULL:
        return null_
    cdef Vector ys = Vector.__new__(Vector)
    cdef int64_t i, n = theta.length
    ys.allocate_numbers(n*2)
    for i in range(0, n):
        ys.numbers[i*2] = cos(theta.numbers[i] * Tau)
        ys.numbers[i*2+1] = sin(theta.numbers[i] * Tau)
    return ys


def angle(Vector xy not None, Vector ys=None):
    if xy.numbers == NULL or (ys is not None and ys.numbers == NULL):
        return null_
    cdef int64_t i, n = xy.length // 2 if ys is None else max(xy.length, ys.length)
    cdef double x, y
    cdef Vector theta = Vector.__new__(Vector)
    theta.allocate_numbers(n)
    if ys is None:
        for i in range(n):
            x = xy.numbers[i*2]
            y = xy.numbers[i*2+1]
            theta.numbers[i] = atan2(y, x) / Tau
    else:
        for i in range(n):
            x = xy.numbers[i % xy.length]
            y = ys.numbers[i % ys.length]
            theta.numbers[i] = atan2(y, x) / Tau
    return theta


def length(Vector xy not None, Vector ys=None):
    if xy.numbers == NULL or (ys is not None and ys.numbers == NULL):
        return null_
    cdef int64_t i, n = xy.length // 2 if ys is None else max(xy.length, ys.length)
    cdef double x, y
    cdef Vector d = Vector.__new__(Vector)
    d.allocate_numbers(n)
    if ys is None:
        for i in range(n):
            x = xy.numbers[i*2]
            y = xy.numbers[i*2+1]
            d.numbers[i] = sqrt(x*x + y*y)
    else:
        for i in range(n):
            x = xy.numbers[i % xy.length]
            y = ys.numbers[i % ys.length]
            d.numbers[i] = sqrt(x*x + y*y)
    return d


def sort(Vector xs not None):
    return Vector(sorted(xs))


def absv(Vector xs not None):
    if xs.numbers == NULL:
        return null_
    cdef Vector ys = Vector.__new__(Vector)
    for i in range(ys.allocate_numbers(xs.length)):
        ys.numbers[i] = abs(xs.numbers[i])
    return ys


def expv(Vector xs not None):
    if xs.numbers == NULL:
        return null_
    cdef Vector ys = Vector.__new__(Vector)
    for i in range(ys.allocate_numbers(xs.length)):
        ys.numbers[i] = exp(xs.numbers[i])
    return ys


def sqrtv(Vector xs not None):
    if xs.numbers == NULL:
        return null_
    cdef Vector ys = Vector.__new__(Vector)
    for i in range(ys.allocate_numbers(xs.length)):
        ys.numbers[i] = sqrt(xs.numbers[i])
    return ys


def logv(Vector xs not None):
    if xs.numbers == NULL:
        return null_
    cdef Vector ys = Vector.__new__(Vector)
    cdef int64_t i, n = xs.length
    for i in range(ys.allocate_numbers(n)):
        ys.numbers[i] = log(xs.numbers[i])
    return ys


def log2v(Vector xs not None):
    if xs.numbers == NULL:
        return null_
    cdef Vector ys = Vector.__new__(Vector)
    cdef int64_t i, n = xs.length
    for i in range(ys.allocate_numbers(n)):
        ys.numbers[i] = log2(xs.numbers[i])
    return ys


def log10v(Vector xs not None):
    if xs.numbers == NULL:
        return null_
    cdef Vector ys = Vector.__new__(Vector)
    cdef int64_t i, n = xs.length
    for i in range(ys.allocate_numbers(n)):
        ys.numbers[i] = log10(xs.numbers[i])
    return ys


def sine(Vector xs not None):
    if xs.numbers == NULL:
        return null_
    cdef Vector ys = Vector.__new__(Vector)
    for i in range(ys.allocate_numbers(xs.length)):
        ys.numbers[i] = (1 - cos(Tau * xs.numbers[i])) / 2
    return ys


def bounce(Vector xs not None):
    if xs.numbers == NULL:
        return null_
    cdef Vector ys = Vector.__new__(Vector)
    cdef double x
    for i in range(ys.allocate_numbers(xs.length)):
        x = xs.numbers[i]
        ys.numbers[i] = sin(Pi * (x-floor(x)))
    return ys


def impulse(Vector xs not None, Vector cs=None):
    if xs.numbers == NULL or (cs is not None and cs.numbers == NULL):
        return null_
    cdef Vector ys = Vector.__new__(Vector)
    cdef int64_t n=cs.length
    cdef double x, c, y, yy
    for i in range(ys.allocate_numbers(xs.length)):
        x = xs.numbers[i]
        x -= floor(x)
        c = cs.numbers[i % n] if cs is not None else 0.25
        if x < c:
            y = 1 - x/c
            yy = y * y
            y = 1 - yy*y
        else:
            y = 1 - (x-c)/(1-c)
            yy = y * y
            y = yy*y
        ys.numbers[i] = y
    return ys


def sharkfin(Vector xs not None):
    if xs.numbers == NULL:
        return null_
    cdef Vector ys = Vector.__new__(Vector)
    cdef double x, y
    for i in range(ys.allocate_numbers(xs.length)):
        x = xs.numbers[i]
        x -= floor(x)
        y = sin(Pi * x) if x < 0.5 else 1 - sin(Pi * (x - 0.5))
        ys.numbers[i] = y
    return ys


def sawtooth(Vector xs not None):
    if xs.numbers == NULL:
        return null_
    cdef Vector ys = Vector.__new__(Vector)
    cdef double x
    for i in range(ys.allocate_numbers(xs.length)):
        x = xs.numbers[i]
        ys.numbers[i] = x - floor(x)
    return ys


def triangle(Vector xs not None):
    if xs.numbers == NULL:
        return null_
    cdef Vector ys = Vector.__new__(Vector)
    cdef double x, y
    for i in range(ys.allocate_numbers(xs.length)):
        x = xs.numbers[i]
        x -= floor(x)
        y = 1 - abs(x - 0.5) * 2
        ys.numbers[i] = y
    return ys


def square(Vector xs not None):
    if xs.numbers == NULL:
        return null_
    cdef Vector ys = Vector.__new__(Vector)
    cdef double x, y
    for i in range(ys.allocate_numbers(xs.length)):
        x = xs.numbers[i]
        x -= floor(x)
        y = 0 if x < 0.5 else 1
        ys.numbers[i] = y
    return ys


def linear(Vector xs not None):
    if xs.numbers == NULL:
        return null_
    cdef Vector ys = Vector.__new__(Vector)
    cdef double x
    for i in range(ys.allocate_numbers(xs.length)):
        x = xs.numbers[i]
        if x < 0:
            x = 0
        elif x > 1:
            x = 1
        ys.numbers[i] = x
    return ys


def quad(Vector xs not None):
    if xs.numbers == NULL:
        return null_
    cdef Vector ys = Vector.__new__(Vector)
    cdef double x, y, x_
    for i in range(ys.allocate_numbers(xs.length)):
        x = xs.numbers[i]
        if x < 0:
            y = 0
        elif x > 1:
            y = 1
        elif x < 0.5:
            x_ = x * 2
            y = x_*x_ / 2
        else:
            x_ = (1 - x) * 2
            y = 1 - x_*x_ / 2
        ys.numbers[i] = y
    return ys


def cubic(Vector xs not None):
    if xs.numbers == NULL:
        return null_
    cdef Vector ys = Vector.__new__(Vector)
    cdef double x, y, x_
    for i in range(ys.allocate_numbers(xs.length)):
        x = xs.numbers[i]
        if x < 0:
            y = 0
        elif x > 1:
            y = 1
        elif x < 0.5:
            x_ = x * 2
            y = x_*x_*x_ / 2
        else:
            x_ = (1 - x) * 2
            y = 1 - x_*x_*x_ / 2
        ys.numbers[i] = y
    return ys


def snap(Vector xs not None):
    if xs.numbers == NULL:
        return null_
    cdef Vector ys = Vector.__new__(Vector)
    cdef double x, y
    for i in range(ys.allocate_numbers(xs.length)):
        x = xs.numbers[i]
        if x < 0:
            x = 0
        elif x > 1:
            x = 1
        y = sqrt(x * 2) / 2 if x < 0.5 else 1 - sqrt((1 - x) * 2) / 2
        ys.numbers[i] = y
    return ys


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


def roundv(Vector xs not None, Vector ys=false_):
    if ys.numbers == NULL or ys.length != 1:
        return null_
    return xs.round(<int64_t>floor(ys.numbers[0]))


def ceilv(Vector xs not None):
    return xs.ceil()


def floorv(Vector xs not None):
    return xs.floor()


def fract(Vector xs not None):
    return xs.fract()


def sumv(Vector xs not None, Vector w=true_):
    cdef int64_t i, j, n = xs.length
    if n == 0 or xs.objects is not None or w.length != 1 or w.objects is not None:
        return null_
    cdef Vector ys = Vector.__new__(Vector)
    cdef int64_t m = <int>(w.numbers[0])
    if m < 1:
        return null_
    ys.allocate_numbers(m)
    for i in range(m):
        ys.numbers[i] = 0
    for i in range(0, n, m):
        for j in range(min(m, n-i)):
            ys.numbers[j] += xs.numbers[i+j]
    return ys


def accumulate(Vector xs not None, Vector zs=true_):
    cdef int64_t i, j, k, n = xs.length
    if n == 0 or xs.objects is not None or zs.length != 1 or zs.objects is not None:
        return null_
    cdef Vector ys = Vector.__new__(Vector)
    cdef int64_t m = <int>(zs.numbers[0])
    if m < 1:
        return null_
    ys.allocate_numbers(n)
    for j in range(min(n, m)):
        ys.numbers[j] = xs.numbers[j]
    for i in range(m, n, m):
        for j in range(min(m, n-i)):
            k = i + j
            ys.numbers[k] = ys.numbers[k-m] + xs.numbers[k]
    return ys


def minv(Vector xs not None, *args):
    cdef Vector ys = null_
    cdef double f
    cdef int64_t c, i, n = xs.length
    if not args:
        if n:
            if xs.objects is None:
                f = xs.numbers[0]
                for i in range(1, n):
                    if xs.numbers[i] < f:
                        f = xs.numbers[i]
                ys = Vector.__new__(Vector)
                ys.allocate_numbers(1)
                ys.numbers[0] = f
            else:
                y = xs.objects[0]
                for i in range(1, n):
                    x = xs.objects[i]
                    if x < y:
                        y = x
                ys = Vector.__new__(Vector)
                ys.objects = (y,)
                ys.length = 1
    else:
        ys = xs
        for xs in args:
            if (c := xs.compare(ys)) == -1:
                ys = xs
            elif c == -2:
                return null_
    return ys


def minindex(Vector xs not None, *args):
    cdef Vector ys
    cdef double f
    cdef int64_t c, i, j = 0, n = xs.length
    if not args:
        if n:
            if xs.objects is None:
                f = xs.numbers[0]
                for i in range(1, n):
                    if xs.numbers[i] < f:
                        f = xs.numbers[i]
                        j = i
            else:
                y = xs.objects[0]
                for i in range(1, n):
                    x = xs.objects[i]
                    if x < y:
                        y = x
                        j = i
        else:
            return null_
    else:
        ys = xs
        for i, xs in enumerate(args):
            if (c := xs.compare(ys)) == -1:
                ys = xs
                j = i + 1
            elif c == -2:
                return null_
    ys = Vector.__new__(Vector)
    ys.allocate_numbers(1)
    ys.numbers[0] = j
    return ys


def maxv(Vector xs not None, *args):
    cdef Vector ys = null_
    cdef double f
    cdef int64_t c, i, n = xs.length
    if not args:
        if n:
            if xs.objects is None:
                f = xs.numbers[0]
                for i in range(1, n):
                    if xs.numbers[i] > f:
                        f = xs.numbers[i]
                ys = Vector.__new__(Vector)
                ys.allocate_numbers(1)
                ys.numbers[0] = f
            else:
                y = xs.objects[0]
                for i in range(1, n):
                    x = xs.objects[i]
                    if x > y:
                        y = x
                ys = Vector.__new__(Vector)
                ys.objects = (y,)
                ys.length = 1
    else:
        ys = xs
        for xs in args:
            if (c := xs.compare(ys)) == 1:
                ys = xs
            elif c == -2:
                return null_
    return ys


def maxindex(Vector xs not None, *args):
    cdef Vector ys
    cdef double f
    cdef int64_t c, i, j = 0, n = xs.length
    if not args:
        if n:
            if xs.objects is None:
                f = xs.numbers[0]
                for i in range(1, n):
                    if xs.numbers[i] > f:
                        f = xs.numbers[i]
                        j = i
            else:
                y = xs.objects[0]
                for i in range(1, n):
                    x = xs.objects[i]
                    if x > y:
                        y = x
                        j = i
        else:
            return null_
    else:
        ys = xs
        for i, xs in enumerate(args):
            if (c := xs.compare(ys)) == 1:
                ys = xs
                j = i + 1
            elif c == -2:
                return null_
    ys = Vector.__new__(Vector)
    ys.allocate_numbers(1)
    ys.numbers[0] = j
    return ys


def hypot(*args):
    cdef int64_t i, n = 0
    cdef double x, y = 0
    cdef Vector xs, ys = null_
    if len(args) == 1:
        xs = args[0]
        if xs.numbers == NULL:
            return null_
        for i in range(xs.length):
            x = xs.numbers[i]
            y += x * x
        y = sqrt(y)
        ys = Vector.__new__(Vector)
        ys.allocate_numbers(1)
        ys.numbers[0] = y
    elif len(args) > 1:
        for xs in args:
            if xs.numbers == NULL:
                return null_
            if xs.length > n:
                n = xs.length
        ys = Vector.__new__(Vector)
        ys.allocate_numbers(n)
        for i in range(n):
            y = 0
            for xs in args:
                x = xs.numbers[i % xs.length]
                y += x * x
            ys.numbers[i] = sqrt(y)
    return ys


def cross(Vector xs not None, Vector ys not None):
    return xs.cross(ys)


def dot(Vector xs not None, Vector ys not None):
    return xs.dot(ys)


def normalize(Vector xs not None):
    return xs.normalize()


def quaternion(Vector axis not None, Vector angle not None):
    if axis.numbers == NULL or axis.length != 3 or angle.numbers == NULL or angle.length != 1:
        return null_
    return Quaternion._euler(axis, angle.numbers[0])


def qmul(Vector a not None, Vector b not None):
    if a.numbers == NULL or a.length != 4 or b.numbers == NULL or b.length not in (3, 4):
        return null_
    return Quaternion._coerce(a) @ (Quaternion._coerce(b) if b.length == 4 else b)


def qbetween(Vector a not None, Vector b not None):
    if a.numbers == NULL or a.length != 3 or b.numbers == NULL or b.length != 3:
        return null_
    return Quaternion._between(a, b)


def slerp(Vector t not None, Vector a not None, Vector b not None):
    if t.numbers == NULL or t.length != 1 or a.numbers == NULL or a.length != 4 or b.numbers == NULL or b.length != 4:
        return null_
    return Quaternion._coerce(a).slerp(Quaternion._coerce(b), t.numbers[0])


@cython.cdivision(True)
def mapv(Vector xs not None, Vector ys not None, Vector zs not None):
    if xs.numbers == NULL or ys.numbers == NULL or zs.numbers == NULL:
        return null_
    cdef int64_t i, m=xs.length, n=ys.length, o=zs.length
    cdef Vector ws = Vector.__new__(Vector)
    cdef double x
    for i in range(ws.allocate_numbers(max(m, n, o))):
        x = xs.numbers[i % m]
        ws.numbers[i] = (1-x)*ys.numbers[i % n] + x*zs.numbers[i % o]
    return ws


def clamp(Vector xs not None, Vector ys not None, Vector zs not None):
    if xs.numbers == NULL or ys.numbers == NULL or zs.numbers == NULL:
        return null_
    return xs.clamp(ys, zs)


@cython.cdivision(True)
def zipv(*vectors):
    cdef bint numeric = True
    cdef list vs = []
    cdef Vector v
    cdef int64_t n = 0
    for v in vectors:
        if v.length:
            vs.append(v)
            if v.length > n:
                n = v.length
            numeric = numeric and v.objects is None
    cdef int64_t i, j, p, m = len(vs)
    if m == 0:
        return null_
    if m == 1:
        return vs[0]
    cdef Vector zs = Vector.__new__(Vector)
    cdef double* zp
    cdef object obj
    if numeric:
        zs.allocate_numbers(n * m)
        for j in range(m):
            v = vs[j]
            p = v.length
            zp = zs.numbers + j
            for i in range(n):
                zp[i*m] = v.numbers[i % p]
    else:
        zs.objects = PyTuple_New(n * m)
        for i in range(n):
            for j in range(m):
                v = vs[j]
                if v.objects is None:
                    obj = v.numbers[i % v.length]
                else:
                    obj = <object>PyTuple_GET_ITEM(v.objects, i % v.length)
                Py_INCREF(obj)
                PyTuple_SET_ITEM(zs.objects, i*m + j, obj)
        zs.length = n * m
    return zs


def count(Vector xs not None, Vector ys not None):
    cdef int64_t i, j, k, n=xs.length, m=ys.length
    if n == 0:
        return null_
    cdef Vector zs = Vector.__new__(Vector)
    zs.allocate_numbers(n)
    for i in range(n):
        k = 0
        for j in range(m):
            if xs.numbers != NULL and ys.numbers != NULL:
                if ys.numbers[j] == xs.numbers[i]:
                    k += 1
            elif ys.numbers != NULL:
                if ys.numbers[j] == xs.objects[i]:
                    k += 1
            elif xs.numbers != NULL:
                if ys.objects[j] == xs.numbers[i]:
                    k += 1
            elif ys.objects[j] == xs.objects[i]:
                k += 1
        zs.numbers[i] = k
    return zs


cdef double hue_to_rgb(double m1, double m2, double h):
    h = h % 6
    if h < 1:
        return m1 + (m2 - m1) * h
    if h < 3:
        return m2
    if h < 4:
        return m1 + (m2 - m1) * (4 - h)
    return m1


cdef Vector hsl_to_rgb(double h, double s, double l):
    cdef double m2 = l * (s + 1) if l <= 0.5 else l + s - l*s
    cdef double m1 = l * 2 - m2
    h *= 6
    cdef Vector rgb = Vector.__new__(Vector)
    rgb.allocate_numbers(3)
    rgb.numbers[0] = hue_to_rgb(m1, m2, h + 2)
    rgb.numbers[1] = hue_to_rgb(m1, m2, h)
    rgb.numbers[2] = hue_to_rgb(m1, m2, h - 2)
    return rgb


def hsl(Vector color):
    if color.length != 3 or color.objects is not None:
        return null_
    cdef double hue = color.numbers[0], saturation = color.numbers[1], lightness = color.numbers[2]
    saturation = min(max(0, saturation), 1)
    lightness = min(max(0, lightness), 1)
    return hsl_to_rgb(hue, saturation, lightness)


@cython.cdivision(True)
def hsv(Vector color):
    if not color.length or color.length > 3 or color.objects is not None:
        return null_
    cdef double hue = color.numbers[0]
    cdef double saturation = color.numbers[1] if color.length > 1 else 1
    cdef double value = color.numbers[2] if color.length == 3 else 1
    saturation = min(max(0, saturation), 1)
    value = min(max(0, value), 1)
    cdef double lightness = value * (1 - saturation / 2)
    return hsl_to_rgb(hue, 0 if lightness == 0 or lightness == 1 else (value - lightness) / min(lightness, 1 - lightness), lightness)


@cython.cdivision(True)
@cython.cpow(True)
def colortemp(Vector t):
    if t.numbers == NULL:
        return null_
    cdef int64_t i, n = t.length
    cdef double T, T2, x, x2, y, X, Y, Z
    cdef Vector rgb = Vector.__new__(Vector)
    rgb.allocate_numbers(3*n)
    for i in range(n):
        T = min(max(1667, t.numbers[i]), 25000)
        T2 = T * T
        if T < 4000:
            x = -0.2661239e9/(T*T2) - 0.2343589e6/T2 + 0.8776956e3/T + 0.179910
        else:
            x = -3.0258469e9/(T*T2) + 2.1070379e6/T2 + 0.2226347e3/T + 0.240390
        x2 = x * x
        if T < 2222:
            y = -1.1063814*x*x2 - 1.34811020*x2 + 2.18555832*x - 0.20219683
        elif T < 4000:
            y = -0.9549476*x*x2 - 1.37418593*x2 + 2.09137015*x - 0.16748867
        else:
            y = +3.0817580*x*x2 - 5.87338670*x2 + 3.75112997*x - 0.37001483
        Y = (max(0, t.numbers[i]) / 6503.5)**4
        X = Y * x / y
        Z = Y * (1 - x - y) / y
        rgb.numbers[3*i] = max(0, 3.2406255*X - 1.537208*Y - 0.4986286*Z)
        rgb.numbers[3*i+1] = max(0, -0.9689307*X + 1.8757561*Y + 0.0415175*Z)
        rgb.numbers[3*i+2] = max(0, 0.0557101*X - 0.2040211*Y + 1.0569959*Z)
    return rgb


cdef Matrix33 OKLab_M1_inv = Matrix33([0.8189330101, 0.0329845436, 0.0482003018,
                                       0.3618667424, 0.9293118715, 0.2643662691,
                                       -0.1288597137, 0.0361456387, 0.6338517070]).inverse()
cdef Matrix33 OKLab_M2_inv = Matrix33([0.2104542553, 1.9779984951, 0.0259040371,
                                       0.7936177850, -2.4285922050, 0.7827717662,
                                       -0.0040720468, 0.4505937099, -0.8086757660]).inverse()
cdef Matrix33 XYZ_to_sRGB = Matrix33([3.2406, -0.9689, 0.0557,
                                      -1.5372, 1.8758, -0.2040,
                                      -0.4986, 0.0415, 1.0570])
cdef Vector Three = Vector(3)


def oklab(Vector lab):
    if lab.length != 3 or lab.objects is not None:
        return null_
    return XYZ_to_sRGB.vmul(OKLab_M1_inv.vmul(OKLab_M2_inv.vmul(lab).pow(Three)))


def oklch(Vector lch):
    if lch.length != 3 or lch.objects is not None:
        return null_
    cdef Vector lab = Vector.__new__(Vector)
    cdef double L=lch.numbers[0], C=lch.numbers[1], h=lch.numbers[2]*Tau
    lab.allocate_numbers(3)
    lab.numbers[0] = L
    lab.numbers[1] = C * cos(h)
    lab.numbers[2] = C * sin(h)
    return XYZ_to_sRGB.vmul(OKLab_M1_inv.vmul(OKLab_M2_inv.vmul(lab).pow(Three)))


def point_towards(Vector direction, Vector up):
    cdef Matrix44 matrix = Matrix44._look(Vector([0, 0, 0]), direction, up)
    if matrix is not None:
        return matrix.inverse()
    return null_


STATIC_FUNCTIONS = {
    'abs': Vector(absv),
    'accumulate': Vector(accumulate),
    'acos': Vector(acosv),
    'angle': Vector(angle),
    'asin': Vector(asinv),
    'beta': Vector(beta),
    'bounce': Vector(bounce),
    'ceil': Vector(ceilv),
    'chr': Vector(chrv),
    'clamp': Vector(clamp),
    'colortemp': Vector(colortemp),
    'cos': Vector(cosv),
    'count': Vector(count),
    'cross': Vector(cross),
    'cubic': Vector(cubic),
    'dot': Vector(dot),
    'exp': Vector(expv),
    'floor': Vector(floorv),
    'fract': Vector(fract),
    'hsl': Vector(hsl),
    'hsv': Vector(hsv),
    'hypot': Vector(hypot),
    'impulse': Vector(impulse),
    'len': Vector(lenv),
    'length': Vector(length),
    'linear': Vector(linear),
    'log': Vector(logv),
    'log10': Vector(log10v),
    'log2': Vector(log2v),
    'map': Vector(mapv),
    'max': Vector(maxv),
    'maxindex': Vector(maxindex),
    'min': Vector(minv),
    'minindex': Vector(minindex),
    'normal': Vector(normal),
    'normalize': Vector(normalize),
    'oklab': Vector(oklab),
    'oklch': Vector(oklch),
    'ord': Vector(ordv),
    'point_towards': Vector(point_towards),
    'polar': Vector(polar),
    'qbetween': Vector(qbetween),
    'qmul': Vector(qmul),
    'quad': Vector(quad),
    'quaternion': Vector(quaternion),
    'round': Vector(roundv),
    'sawtooth': Vector(sawtooth),
    'sort': Vector(sort),
    'sharkfin': Vector(sharkfin),
    'shuffle': Vector(shuffle),
    'sin': Vector(sinv),
    'sine': Vector(sine),
    'slerp': Vector(slerp),
    'snap': Vector(snap),
    'split': Vector(split),
    'sqrt': Vector(sqrtv),
    'square': Vector(square),
    'sum': Vector(sumv),
    'tan': Vector(tanv),
    'triangle': Vector(triangle),
    'uniform': Vector(uniform),
    'zip': Vector(zipv),
}

DYNAMIC_FUNCTIONS = {
    'debug': Vector(debug),
    'sample': Vector(sample),
    'glob': Vector(glob),
    'csv': Vector(read_csv),
    'read': Vector(read_text),
    'read_bytes': Vector(read_bytes),
}
