# cython: language_level=3, profile=False

"""
Flitter language functions
"""


import cython

from libc.math cimport isnan, isinf, floor, round, sin, cos, asin, acos, sqrt, exp, ceil, atan2, log

from ..cache import SharedCache
from ..model cimport StateDict, Vector, Matrix44, null_, true_, false_


cdef double Pi = 3.141592653589793
cdef double Tau = 6.283185307179586


def state_transformer(func):
    func.state_transformer = True
    return func


cdef class Uniform(Vector):
    cdef unsigned long long seed

    def __cinit__(self, value=None):
        self.seed = self.hash(True)
        self.deallocate_numbers()
        self.length = 0
        self.objects = None

    def __hash__(self):
        return self.seed

    def __eq__(self, other):
        return self.eq(Vector._coerce(other)).numbers[0] != 0

    cdef Vector eq(self, Vector other):
        if isinstance(other, self.__class__) and (<Uniform>other).seed == self.seed:
            return true_
        return false_

    cdef Vector item(self, int i):
        cdef Vector value = Vector.__new__(Vector)
        value.allocate_numbers(1)
        value.numbers[0] = self._item(i)
        return value

    cdef double _item(self, unsigned long long i):
        cdef unsigned long long x, y, z
        # Compute a 32bit float PRN using the Squares algorithm [https://arxiv.org/abs/2004.06278]
        x = y = i * self.seed
        z = y + self.seed
        x = x*x + y
        x = (x >> 32) | (x << 32)
        x = x*x + z
        x = (x >> 32) | (x << 32)
        x = x*x + y
        return <double>(x >> 32) / <double>(1<<32)

    cpdef Vector slice(self, Vector index):
        if index.numbers == NULL:
            return null_
        cdef Vector result = Vector.__new__(Vector)
        cdef int i
        cdef unsigned long long j
        for i in range(result.allocate_numbers(index.length)):
            j = <unsigned long long>(<long long>floor(index.numbers[i]))
            result.numbers[i] = self._item(j)
        return result

    cpdef Vector copynodes(self):
        return self

    cpdef bint as_bool(self):
        return True

    def __repr__(self):
        return f"{self.__class__.__name__}({self.seed!r})"


cdef class Beta(Uniform):
    cdef double _item(self, unsigned long long i):
        i <<= 2
        cdef double u1 = Uniform._item(self, i)
        cdef double u2 = Uniform._item(self, i + 1)
        cdef double u3 = Uniform._item(self, i + 2)
        if u1 <= u2 and u1 <= u3:
            return min(u2, u3)
        if u2 <= u1 and u2 <= u3:
            return min(u1, u3)
        return min(u1, u2)


cdef class Normal(Uniform):
    cdef double _item(self, unsigned long long i):
        # Use the Box-Muller transform to approximate the normal distribution
        # [https://en.wikipedia.org/wiki/Box–Muller_transform]
        i <<= 1
        cdef double u1 = Uniform._item(self, i)
        cdef double u2 = Uniform._item(self, i + 1)
        if u1 < 1 / (1<<32):
            u1, u2 = u2, u1
        return sqrt(-2 * log(u1)) * sin(Tau * u2)


@state_transformer
def counter(StateDict state, Vector counter_id, Vector clockv, Vector speedv=None):
    if counter_id.length == 0 or clockv.numbers == NULL or clockv.length != 1:
        return null_
    cdef double clock = clockv.numbers[0]
    cdef double speed = 0
    if speedv is None:
        speed = 1
    elif speedv.numbers != NULL and speedv.length == 1 and not isnan(speedv.numbers[0]) and not isinf(speedv.numbers[0]):
        speed = speedv.numbers[0]
    cdef double clock_start = clock if speed != 0 else 0
    cdef double current_speed = speed
    cdef Vector counter_state
    cdef bint update = True
    if counter_id in state:
        counter_state = state[counter_id]
        if counter_state.numbers != NULL and counter_state.length == 2:
            clock_start = counter_state.numbers[0]
            current_speed = counter_state.numbers[1]
            update = False
    cdef double count = (clock - clock_start) * current_speed if current_speed != 0 else clock_start
    if speed != current_speed:
        clock_start = clock - count / speed if speed != 0 else count
        current_speed = speed
        update = True
    if update:
        counter_state = Vector.__new__(Vector)
        counter_state.allocate_numbers(2)
        counter_state.numbers[0] = clock_start
        counter_state.numbers[1] = current_speed
        state[counter_id] = counter_state
    cdef Vector countv = Vector.__new__(Vector)
    countv.allocate_numbers(1)
    countv.numbers[0] = count
    return countv


def read_text(Vector filename):
    cdef str path = filename.as_string()
    if path:
        return Vector._coerce(SharedCache[path].read_text(encoding='utf8'))
    return null_


def read_csv(Vector filename, Vector row_number):
    cdef str path = str(filename)
    row = row_number.match(1, int)
    if filename and row is not None:
        return SharedCache[path].read_csv_vector(row)
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
    cdef int i, n = theta.length
    for i in range(ys.allocate_numbers(n)):
        ys.numbers[i] = sin(theta.numbers[i] * Tau)
    return ys


def cosv(Vector theta not None):
    if theta.numbers == NULL:
        return null_
    cdef Vector ys = Vector.__new__(Vector)
    cdef int i, n = theta.length
    for i in range(ys.allocate_numbers(n)):
        ys.numbers[i] = cos(theta.numbers[i] * Tau)
    return ys


def asinv(Vector ys not None):
    if ys.numbers == NULL:
        return null_
    cdef Vector theta = Vector.__new__(Vector)
    cdef int i, n = ys.length
    for i in range(theta.allocate_numbers(n)):
        theta.numbers[i] = asin(ys.numbers[i]) / Tau
    return theta


def acosv(Vector ys not None):
    if ys.numbers == NULL:
        return null_
    cdef Vector theta = Vector.__new__(Vector)
    cdef int i, n = ys.length
    for i in range(theta.allocate_numbers(n)):
        theta.numbers[i] = acos(ys.numbers[i]) / Tau
    return theta


def polar(Vector theta not None):
    if theta.numbers == NULL:
        return null_
    cdef Vector ys = Vector.__new__(Vector)
    cdef int i, n = theta.length
    ys.allocate_numbers(n*2)
    for i in range(0, n):
        ys.numbers[i*2] = cos(theta.numbers[i] * Tau)
        ys.numbers[i*2+1] = sin(theta.numbers[i] * Tau)
    return ys


def angle(Vector xy not None):
    if xy.numbers == NULL:
        return null_
    cdef int i, n = xy.length // 2
    cdef double x, y
    cdef Vector theta = Vector.__new__(Vector)
    theta.allocate_numbers(n)
    for i in range(0, n):
        x = xy.numbers[i*2]
        y = xy.numbers[i*2+1]
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


def impulse(Vector xs not None):
    if xs.numbers == NULL:
        return null_
    cdef Vector ys = Vector.__new__(Vector)
    cdef double x, y
    for i in range(ys.allocate_numbers(xs.length)):
        x = xs.numbers[i]
        x -= floor(x)
        # bounce(linear(x * 4) / 2) - quad(linear((x * 4 - 1) / 3))
        x *= 4
        if x < 1:
            y = sin(Pi*x/2)
        else:
            x -= 1
            x /= 3
            y = 1 - ((x * 2)**2 / 2 if x < 0.5 else 1 - ((1 - x) * 2)**2 / 2)
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


def shuffle(Uniform source, Vector xs not None):
    if xs.length == 0:
        return null_
    cdef int i, j, n = xs.length
    xs = Vector.__new__(Vector, xs)
    if xs.objects is None:
        for i in range(n - 1):
            j = <int>floor(source.item(i) * n) + i
            n -= 1
            xs.numbers[i], xs.numbers[j] = xs.numbers[j], xs.numbers[i]
    else:
        for i in range(n - 1):
            j = <int>floor(source.item(i) * n) + i
            n -= 1
            xs.objects[i], xs.objects[j] = xs.objects[j], xs.objects[i]
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
    if xs.numbers == NULL:
        return null_
    cdef Vector ys = Vector.__new__(Vector)
    cdef double x, y
    for i in range(ys.allocate_numbers(xs.length)):
        x = xs.numbers[i]
        y = ceil(x)
        ys.numbers[i] = y
    return ys


def floorv(Vector xs not None):
    if xs.numbers == NULL:
        return null_
    cdef Vector ys = Vector.__new__(Vector)
    cdef double x, y
    for i in range(ys.allocate_numbers(xs.length)):
        x = xs.numbers[i]
        y = floor(x)
        ys.numbers[i] = y
    return ys


def sumv(Vector xs not None):
    if xs.objects is not None:
        return null_
    cdef double y = 0
    for i in range(xs.length):
        y += xs.numbers[i]
    cdef Vector ys = Vector.__new__(Vector)
    ys.allocate_numbers(1)
    ys.numbers[0] = y
    return ys


def accumulate(Vector xs not None):
    cdef int i, n = xs.length
    if n == 0 or xs.objects is not None:
        return null_
    cdef Vector ys = Vector.__new__(Vector)
    cdef double y = 0
    for i in range(ys.allocate_numbers(n)):
        y += xs.numbers[i]
        ys.numbers[i] = y
    return ys


def minv(Vector xs not None, *args):
    cdef Vector ys = null_
    cdef double f
    cdef int i, n = xs.length
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
                ys.objects = [y]
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
    cdef int i, j = 0, n = xs.length
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
    cdef int i, n = xs.length
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
                ys.objects = [y]
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
    cdef int i, j = 0, n = xs.length
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


def hypot(Vector xs not None):
    cdef int i, n = xs.length
    if n == 0 or xs.objects is not None:
        return null_
    cdef double x, y = 0
    for i in range(n):
        x = xs.numbers[i]
        y += x * x
    y = sqrt(y)
    cdef Vector ys = Vector.__new__(Vector)
    ys.allocate_numbers(1)
    ys.numbers[0] = y
    return ys


def normalize(Vector xs not None):
    return xs.normalize()


@cython.cdivision(True)
def mapv(Vector xs not None, Vector ys not None, Vector zs not None):
    if xs.numbers == NULL or ys.numbers == NULL or zs.numbers == NULL:
        return null_
    cdef int i, m=xs.length, n=ys.length, o=zs.length
    cdef Vector ws = Vector.__new__(Vector)
    cdef double x
    for i in range(ws.allocate_numbers(max(m, n, o))):
        x = xs.numbers[i % m]
        ws.numbers[i] = (1-x)*ys.numbers[i % n] + x*zs.numbers[i % o]
    return ws


@cython.cdivision(True)
def zipv(*vectors):
    cdef bint numeric = True
    cdef list vs = []
    cdef Vector v
    cdef int n = 0
    for v in vectors:
        if v.length:
            vs.append(v)
            if v.length > n:
                n = v.length
            numeric = numeric and v.objects is None
    cdef int i, j, p, m = len(vs)
    if m == 0:
        return null_
    if m == 1:
        return vs[0]
    cdef Vector zs = Vector.__new__(Vector)
    cdef double* zp
    if numeric:
        zs.allocate_numbers(n * m)
        for j in range(m):
            v = vs[j]
            p = v.length
            zp = zs.numbers + j
            for i in range(n):
                zp[i*m] = v.numbers[i % p]
    else:
        zs.objects = list()
        zs.length = n * m
        for i in range(n):
            for j in range(m):
                v = vs[j]
                zs.objects.append(v.numbers[i % v.length] if v.objects is None else v.objects[i % v.length])
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

def hsl(Vector c):
    if c.length != 3 or c.objects is not None:
        return null_
    cdef double h = c.numbers[0], s = c.numbers[1], l = c.numbers[2]
    s = min(max(0, s), 1)
    l = min(max(0, l), 1)
    return hsl_to_rgb(h, s, l)

@cython.cdivision(True)
def hsv(Vector c):
    if c.length != 3 or c.objects is not None:
        return null_
    cdef double h = c.numbers[0], s = c.numbers[1], v = c.numbers[2]
    s = min(max(0, s), 1)
    v = min(max(0, v), 1)
    cdef double l = v * (1 - s / 2)
    return hsl_to_rgb(h, 0 if l == 0 or l == 1 else (v - l) / min(l, 1 - l), l)


def point_towards(Vector direction, Vector up):
    cdef Matrix44 matrix = Matrix44._look(Vector([0, 0, 0]), direction, up)
    if matrix is not None:
        return matrix.inverse()
    return null_


STATIC_FUNCTIONS = {
    'uniform': Vector(Uniform),
    'beta': Vector(Beta),
    'normal': Vector(Normal),
    'len': Vector(length),
    'sin': Vector(sinv),
    'cos': Vector(cosv),
    'asin': Vector(asinv),
    'acos': Vector(acosv),
    'polar': Vector(polar),
    'abs': Vector(absv),
    'exp': Vector(expv),
    'sqrt': Vector(sqrtv),
    'sine': Vector(sine),
    'bounce': Vector(bounce),
    'sharkfin': Vector(sharkfin),
    'impulse': Vector(impulse),
    'sawtooth': Vector(sawtooth),
    'triangle': Vector(triangle),
    'square': Vector(square),
    'linear': Vector(linear),
    'quad': Vector(quad),
    'snap': Vector(snap),
    'shuffle': Vector(shuffle),
    'round': Vector(roundv),
    'ceil': Vector(ceilv),
    'floor': Vector(floorv),
    'sum': Vector(sumv),
    'accumulate': Vector(accumulate),
    'min': Vector(minv),
    'minindex': Vector(minindex),
    'max': Vector(maxv),
    'maxindex': Vector(maxindex),
    'hypot': Vector(hypot),
    'angle': Vector(angle),
    'normalize': Vector(normalize),
    'map': Vector(mapv),
    'zip': Vector(zipv),
    'hsl': Vector(hsl),
    'hsv': Vector(hsv),
    'point_towards': Vector(point_towards),
}

DYNAMIC_FUNCTIONS = {
    'read': Vector(read_text),
    'csv': Vector(read_csv),
    'counter': Vector(counter),
}
