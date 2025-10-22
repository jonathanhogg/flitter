

import cython

from libc.stdint cimport int64_t
from libc.math cimport tan, atan2, acos, asin, sqrt, exp, log, log2, log10, floor

from ...model cimport Vector, true_, false_, null_, cost, sint


cdef double Tau = 6.283185307179586231995926937088370323181152343750


def sinv(Vector theta not None):
    if theta.numbers == NULL:
        return null_
    cdef Vector ys = Vector.__new__(Vector)
    cdef int64_t i, n = theta.length
    for i in range(ys.allocate_numbers(n)):
        ys.numbers[i] = sint(theta.numbers[i])
    return ys


def cosv(Vector theta not None):
    if theta.numbers == NULL:
        return null_
    cdef Vector ys = Vector.__new__(Vector)
    cdef int64_t i, n = theta.length
    for i in range(ys.allocate_numbers(n)):
        ys.numbers[i] = cost(theta.numbers[i])
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
        ys.numbers[i*2] = cost(theta.numbers[i])
        ys.numbers[i*2+1] = sint(theta.numbers[i])
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
    if xs.objects is not None or w.length != 1 or w.objects is not None:
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


def product(Vector xs not None, Vector w=true_):
    cdef int64_t i, j, n = xs.length
    if xs.objects is not None or w.length != 1 or w.objects is not None:
        return null_
    cdef Vector ys = Vector.__new__(Vector)
    cdef int64_t m = <int>(w.numbers[0])
    if m < 1:
        return null_
    ys.allocate_numbers(m)
    for i in range(m):
        ys.numbers[i] = 1
    for i in range(0, n, m):
        for j in range(min(m, n-i)):
            ys.numbers[j] *= xs.numbers[i+j]
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


@cython.cdivision(True)
def mean(Vector xs not None, Vector zs=true_):
    cdef int64_t i, n = xs.length
    if xs.objects is not None or zs.length != 1 or zs.objects is not None:
        return null_
    cdef int64_t m = <int>(zs.numbers[0])
    if m < 1:
        return null_
    cdef Vector ys = Vector.__new__(Vector)
    cdef Vector ds = Vector.__new__(Vector)
    ys.allocate_numbers(m)
    ds.allocate_numbers(m)
    for i in range(m):
        ys.numbers[i] = 0
        ds.numbers[i] = 0
    for i in range(n):
        ys.numbers[i % m] = ys.numbers[i % m] + xs.numbers[i]
        ds.numbers[i % m] = ds.numbers[i % m] + 1
    for i in range(m):
        if ds.numbers[i]:
            ys.numbers[i] = ys.numbers[i] / ds.numbers[i]
    return ys
