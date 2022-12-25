# cython: language_level=3, profile=True

"""
Flitter language functions
"""


import cython

from libc.math cimport isnan, floor, round, sin, cos, sqrt, exp

from ..model cimport VectorLike, Vector, null_, true_


DEF PI = 3.141592653589793
DEF TwoPI = 6.283185307179586


cdef class Uniform(VectorLike):
    cdef Vector keys
    cdef unsigned long long seed

    def __cinit__(self, Vector keys not None):
        self.keys = keys
        self.seed = keys._hash(True)

    cdef double item(self, unsigned long long i):
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
        cdef Vector result = Vector.__new__(Vector)
        cdef int j
        for i in range(len(index.values)):
            value = index.values[i]
            if isinstance(value, (float, int)):
                j = <int>floor(value)
                if j >= 0:
                    result.values.append(self.item(j))
        return result

    cpdef VectorLike copynodes(self):
        return self

    cpdef bint istrue(self):
        return True

    def __repr__(self):
        return f"{self.__class__.__name__}({self.keys!r})"


cdef class Beta(Uniform):
    cdef double item(self, unsigned long long i):
        i <<= 2
        cdef double u1 = Uniform.item(self, i)
        cdef double u2 = Uniform.item(self, i + 1)
        cdef double u3 = Uniform.item(self, i + 2)
        if u1 <= u2 and u1 <= u3:
            return min(u2, u3)
        if u2 <= u1 and u2 <= u3:
            return min(u1, u3)
        return min(u1, u2)


cdef class Normal(Uniform):
    cdef double item(self, unsigned long long i):
        i <<= 4
        cdef double u = -6
        cdef int j
        for j in range(12):
            u += Uniform.item(self, i + j)
        return u


def length(Vector xs not None):
    cdef Vector ys = Vector.__new__(Vector)
    ys.values.append(len(xs))
    return ys


def sinv(Vector theta not None):
    cdef Vector ys = Vector.__new__(Vector)
    cdef double th, y
    for i in range(len(theta.values)):
        th = theta.values[i] * TwoPI
        y = sin(th)
        ys.values.append(y)
    return ys


def cosv(Vector theta not None):
    cdef Vector ys = Vector.__new__(Vector)
    cdef double th, y
    for i in range(len(theta.values)):
        th = theta.values[i] * TwoPI
        y = cos(th)
        ys.values.append(y)
    return ys


def polar(Vector theta not None):
    cdef Vector xys = Vector.__new__(Vector)
    cdef double th
    for i in range(len(theta.values)):
        th = theta.values[i] * TwoPI
        xys.values.append(cos(th))
        xys.values.append(sin(th))
    return xys


def expv(Vector xs not None):
    cdef Vector ys = Vector.__new__(Vector)
    cdef double x, y
    for i in range(len(xs.values)):
        x = xs.values[i]
        y = exp(x)
        ys.values.append(y)
    return ys


def sine(Vector xs not None):
    cdef Vector ys = Vector.__new__(Vector)
    cdef double x, y
    for i in range(len(xs.values)):
        x = xs.values[i]
        x -= floor(x)
        y = (1 - cos(TwoPI * x)) / 2
        ys.values.append(y)
    return ys


def bounce(Vector xs not None):
    cdef Vector ys = Vector.__new__(Vector)
    cdef double x, y
    for i in range(len(xs.values)):
        x = xs.values[i]
        x -= floor(x)
        y = sin(PI * x)
        ys.values.append(y)
    return ys


def sharkfin(Vector xs not None):
    cdef Vector ys = Vector.__new__(Vector)
    cdef double x, y
    for i in range(len(xs.values)):
        x = xs.values[i]
        x -= floor(x)
        y = sin(PI * x) if x < 0.5 else 1 - sin(PI * (x - 0.5))
        ys.values.append(y)
    return ys


def sawtooth(Vector xs not None):
    cdef Vector ys = Vector.__new__(Vector)
    cdef double x
    for i in range(len(xs.values)):
        x = xs.values[i]
        x -= floor(x)
        ys.values.append(x)
    return ys


def triangle(Vector xs not None):
    cdef Vector ys = Vector.__new__(Vector)
    cdef double x, y
    for i in range(len(xs.values)):
        x = xs.values[i]
        x -= floor(x)
        y = 1 - abs(x - 0.5) * 2
        ys.values.append(y)
    return ys


def square(Vector xs not None):
    cdef Vector ys = Vector.__new__(Vector)
    cdef double x, y
    for i in range(len(xs.values)):
        x = xs.values[i]
        x -= floor(x)
        y = 0 if x < 0.5 else 1
        ys.values.append(y)
    return ys


def linear(Vector xs not None):
    cdef Vector ys = Vector.__new__(Vector)
    cdef double x
    for i in range(len(xs.values)):
        x = xs.values[i]
        if x < 0:
            x = 0
        elif x > 1:
            x = 1
        ys.values.append(x)
    return ys


def quad(Vector xs not None):
    cdef Vector ys = Vector.__new__(Vector)
    cdef double x, y
    for i in range(len(xs.values)):
        x = xs.values[i]
        if x < 0:
            x = 0
        elif x > 1:
            x = 1
        y = (x * 2)**2 / 2 if x < 0.5 else 1 - ((1 - x) * 2)**2 / 2
        ys.values.append(y)
    return ys


def shuffle(Uniform source, Vector xs not None):
    cdef int j, n = len(xs.values)
    xs = Vector.__new__(Vector, xs)
    for i in range(n - 1):
        j = <int>floor(source.item(i) * n) + i
        n -= 1
        xs.values[i], xs.values[j] = xs.values[j], xs.values[i]
    return xs


def roundv(Vector xs not None):
    cdef Vector ys = Vector.__new__(Vector)
    cdef double x, y
    for i in range(len(xs.values)):
        x = xs.values[i]
        y = round(x)
        ys.values.append(y)
    return ys


def sumv(Vector xs not None):
    cdef float y = 0;
    for x in xs.values:
        if isinstance(x, (int, float)):
            y += x
    cdef Vector ys = Vector.__new__(Vector)
    ys.values.append(y)
    return ys


def minv(Vector xs not None, *args):
    cdef Vector ys = null_
    if not args:
        y = None
        for x in xs.values:
            if y is None or x < y:
                y = x
        if y is not None:
            ys = Vector.__new__(Vector)
            ys.values.append(y)
    else:
        ys = xs
        for xs in args:
            if xs.compare(ys) == -1:
                ys = xs
    return ys


def maxv(Vector xs not None, *args):
    cdef Vector ys = null_
    if not args:
        y = None
        for x in xs.values:
            if y is None or x > y:
                y = x
        if y is not None:
            ys = Vector.__new__(Vector)
            ys.values.append(y)
    else:
        ys = xs
        for xs in args:
            if xs.compare(ys) == 1:
                ys = xs
    return ys


def hypot(Vector xs not None):
    cdef double x, s = 0.0
    for x in xs.values:
        s += x * x
    s = sqrt(s)
    cdef Vector ys = Vector.__new__(Vector)
    ys.values.append(s)
    return ys


def mapv(Vector x not None, Vector a not None, Vector b not None):
    return a.mul(true_.sub(x)).add(b.mul(x))


def zipv(*vectors):
    cdef Vector zs = Vector.__new__(Vector)
    if not vectors:
        return zs
    cdef Vector vs
    cdef int i=0
    while True:
        for vs in vectors:
            if i == len(vs.values):
                return zs
            zs.values.append(vs.values[i])
        i += 1


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
    rgb.values.append(hue_to_rgb(m1, m2, h + 2))
    rgb.values.append(hue_to_rgb(m1, m2, h))
    rgb.values.append(hue_to_rgb(m1, m2, h - 2))
    return rgb

def hsl(Vector c):
    c = c.withlen(3)
    if c is null_:
        return null_
    cdef double h = c.values[0], s = c.values[1], l = c.values[2]
    s = min(max(0, s), 1)
    l = min(max(0, l), 1)
    return hsl_to_rgb(h, s, l)

def hsv(Vector c):
    c = c.withlen(3)
    if c is null_:
        return null_
    cdef double h = c.values[0], s = c.values[1], v = c.values[2]
    s = min(max(0, s), 1)
    v = min(max(0, v), 1)
    cdef double l = v * (1 - s / 2)
    return hsl_to_rgb(h, 0 if l == 0 or l == 1 else (v - l) / min(l, 1 - l), l)


FUNCTIONS = {
    'uniform': Vector((Uniform,)),
    'beta': Vector((Beta,)),
    'normal': Vector((Normal,)),
    'len': Vector((length,)),
    'sin': Vector((sinv,)),
    'cos': Vector((cosv,)),
    'polar': Vector((polar,)),
    'exp': Vector((expv,)),
    'sine': Vector((sine,)),
    'bounce': Vector((bounce,)),
    'sharkfin': Vector((sharkfin,)),
    'sawtooth': Vector((sawtooth,)),
    'triangle': Vector((triangle,)),
    'square': Vector((square,)),
    'linear': Vector((linear,)),
    'quad': Vector((quad,)),
    'shuffle': Vector((shuffle,)),
    'round': Vector((roundv,)),
    'sum': Vector((sumv,)),
    'min': Vector((minv,)),
    'max': Vector((maxv,)),
    'hypot': Vector((hypot,)),
    'map': Vector((mapv,)),
    'zip': Vector((zipv,)),
    'hsl': Vector((hsl,)),
    'hsv': Vector((hsv,)),
}
