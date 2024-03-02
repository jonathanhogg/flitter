# cython: language_level=3, profile=False

"""
Flitter language functions
"""


import cython

from libc.math cimport floor, round, sin, cos, tan, asin, acos, sqrt, exp, atan2, log, log2, log10
from libc.stdint cimport int64_t, uint64_t
from cpython.object cimport PyObject
from cpython.ref cimport Py_INCREF
from cpython.tuple cimport PyTuple_New, PyTuple_GET_ITEM, PyTuple_SET_ITEM

from ..cache import SharedCache
from ..model cimport Vector, Matrix44, Context, null_, true_, false_


cdef double Pi = 3.141592653589793
cdef double Tau = 6.283185307179586


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
    cdef const double[:, :, :] data
    if coord.length != 2 or coord.numbers == NULL \
            or (scene_node := context.references.get(texture_id.as_string())) is None \
            or not hasattr(scene_node, 'texture_data') or (data := scene_node.texture_data) is None:
        return default
    cdef int64_t height=data.shape[0], width=data.shape[1], x=int(coord.numbers[0] * width), y=int(coord.numbers[1] * height)
    if x < 0 or x >= width or y < 0 or y >= height:
        return default
    cdef const double[:] color = data[y, x]
    cdef Vector result = Vector.__new__(Vector)
    result.allocate_numbers(3)
    result.numbers[0] = color[0]
    result.numbers[1] = color[1]
    result.numbers[2] = color[2]
    return result


cdef class uniform(Vector):
    def __cinit__(self, value=None):
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

    cdef Vector item(self, int64_t i):
        cdef Vector value = Vector.__new__(Vector)
        value.allocate_numbers(1)
        value.numbers[0] = self._item(i)
        return value

    @cython.cdivision(True)
    cdef double _item(self, uint64_t i) noexcept nogil:
        cdef uint64_t x, y, z
        # Compute a 32bit float PRN using the Squares algorithm [https://arxiv.org/abs/2004.06278]
        x = y = i * self._hash
        z = y + self._hash
        x = x*x + y
        x = (x >> 32) | (x << 32)
        x = x*x + z
        x = (x >> 32) | (x << 32)
        x = x*x + y
        x = (x >> 32) | (x << 32)
        x = x*x + z
        return <double>(x >> 32) / <double>(1<<32)

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
            if u1 < 1 / <double>(1<<32):
                u1, u2 = u2, u1
            self.R = sqrt(-2 * log(u1))
            self.th = Tau * u2
            self.i = i
            self.cached = True
        if odd:
            return self.R * sin(self.th)
        return self.R * cos(self.th)


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


@context_func
def read_csv(Context context, Vector filename, Vector row_number):
    cdef str path = str(filename)
    row = row_number.match(1, int)
    if filename and row is not None:
        return SharedCache.get_with_root(path, context.path).read_csv_vector(row)
    return null_


def length(Vector xs not None):
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
    cdef double x, y
    for i in range(ys.allocate_numbers(xs.length)):
        x = xs.numbers[i]
        if x < 0:
            x = 0
        elif x > 1:
            x = 1
        y = (x * 2)**2 / 2 if x < 0.5 else 1 - ((1 - x) * 2)**2 / 2
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
    xs = Vector._copy(xs)
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


def roundv(Vector xs not None):
    if xs.numbers == NULL:
        return null_
    cdef Vector ys = Vector.__new__(Vector)
    cdef double x, y
    for i in range(ys.allocate_numbers(xs.length)):
        x = xs.numbers[i]
        y = round(x)
        ys.numbers[i] = y
    return ys


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
    cdef int64_t i, n = xs.length
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
            if xs.compare(ys) == -1:
                ys = xs
    return ys


def minindex(Vector xs not None, *args):
    cdef Vector ys
    cdef double f
    cdef int64_t i, j = 0, n = xs.length
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
        for i, ys in enumerate(args):
            if ys.compare(xs) == -1:
                xs = ys
                j = i + 1
    ys = Vector.__new__(Vector)
    ys.allocate_numbers(1)
    ys.numbers[0] = j
    return ys


def maxv(Vector xs not None, *args):
    cdef Vector ys = null_
    cdef double f
    cdef int64_t i, n = xs.length
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
            if xs.compare(ys) == 1:
                ys = xs
    return ys


def maxindex(Vector xs not None, *args):
    cdef Vector ys
    cdef double f
    cdef int64_t i, j = 0, n = xs.length
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
        for i, ys in enumerate(args):
            if ys.compare(xs) == 1:
                xs = ys
                j = i + 1
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


def normalize(Vector xs not None):
    return xs.normalize()


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


@cython.cdivision(True)
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
    'exp': Vector(expv),
    'floor': Vector(floorv),
    'fract': Vector(fract),
    'hsl': Vector(hsl),
    'hsv': Vector(hsv),
    'hypot': Vector(hypot),
    'impulse': Vector(impulse),
    'len': Vector(length),
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
    'ord': Vector(ordv),
    'point_towards': Vector(point_towards),
    'polar': Vector(polar),
    'quad': Vector(quad),
    'round': Vector(roundv),
    'sawtooth': Vector(sawtooth),
    'sharkfin': Vector(sharkfin),
    'shuffle': Vector(shuffle),
    'sin': Vector(sinv),
    'sine': Vector(sine),
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
    'csv': Vector(read_csv),
    'read': Vector(read_text),
    'read_bytes': Vector(read_bytes),
}
