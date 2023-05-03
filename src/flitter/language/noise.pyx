# cython: language_level=3, profile=False

"""
OpenSimplex 2(s) noise generation

Adapted from https://github.com/lmas/opensimplex, which itself was adapted from
https://gist.github.com/KdotJPG/b1270127455a94ac5d19
"""

import cython
import numpy as np

from libc.math cimport floor

from .functions import Uniform, shuffle
from ..model cimport Vector, null_


cdef Vector GRADIENTS2 = Vector([5, 2, 2, 5, -5, 2, -2, 5, 5, -2, 2, -5, -5, -2, -2, -5])
cdef double STRETCH_CONSTANT2 = -0.211324865405187
cdef double SQUISH_CONSTANT2 = 0.366025403784439
cdef double NORM_CONSTANT2 = 47
cdef int MAX_PERM_CACHE_ITEMS = 100
cdef dict PermCache = {}


cdef inline double _extrapolate2(Vector perm, long xsb, long ysb, double dx, double dy) noexcept nogil:
    cdef long index = <long>(perm.numbers[(<long>(perm.numbers[xsb % 256]) + ysb) % 256]) % 15
    cdef double g1 = GRADIENTS2.numbers[index]
    cdef double g2 = GRADIENTS2.numbers[index+1]
    return g1 * dx + g2 * dy


@cython.cdivision(True)
cdef Vector noise2(Vector perm, Vector xx, Vector yy):
    cdef int i, j, count = 0
    cdef long xsb, ysb, xsv_ext, ysv_ext
    cdef double x, y, stretch_offset, xs, ys, squish_offset, xb, yb, xins, yins, in_sum, dx0, dy0, value
    cdef double dx1, dy1, attn1, dx2, dy2, attn2, zins, dx_ext, dy_ext, attn0, attn_ext
    cdef Vector result = Vector.__new__(Vector)
    result.allocate_numbers(xx.length * yy.length)
    with nogil:
        for i in range(xx.length):
            x = xx.numbers[i]
            for j in range(yy.length):
                y = yy.numbers[j]
                stretch_offset = (x + y) * STRETCH_CONSTANT2
                xs = x + stretch_offset
                ys = y + stretch_offset
                xsb = <long>floor(xs)
                ysb = <long>floor(ys)
                squish_offset = (xsb + ysb) * SQUISH_CONSTANT2
                xb = xsb + squish_offset
                yb = ysb + squish_offset
                xins = xs - xsb
                yins = ys - ysb
                in_sum = xins + yins
                dx0 = x - xb
                dy0 = y - yb
                value = 0
                dx1 = dx0 - 1 - SQUISH_CONSTANT2
                dy1 = dy0 - 0 - SQUISH_CONSTANT2
                attn1 = 2 - dx1 * dx1 - dy1 * dy1
                if attn1 > 0:
                    attn1 *= attn1
                    value += attn1 * attn1 * _extrapolate2(perm, xsb + 1, ysb + 0, dx1, dy1)
                dx2 = dx0 - 0 - SQUISH_CONSTANT2
                dy2 = dy0 - 1 - SQUISH_CONSTANT2
                attn2 = 2 - dx2 * dx2 - dy2 * dy2
                if attn2 > 0:
                    attn2 *= attn2
                    value += attn2 * attn2 * _extrapolate2(perm, xsb + 0, ysb + 1, dx2, dy2)
                if in_sum <= 1:
                    zins = 1 - in_sum
                    if zins > xins or zins > yins:
                        if xins > yins:
                            xsv_ext = xsb + 1
                            ysv_ext = ysb - 1
                            dx_ext = dx0 - 1
                            dy_ext = dy0 + 1
                        else:
                            xsv_ext = xsb - 1
                            ysv_ext = ysb + 1
                            dx_ext = dx0 + 1
                            dy_ext = dy0 - 1
                    else:
                        xsv_ext = xsb + 1
                        ysv_ext = ysb + 1
                        dx_ext = dx0 - 1 - 2 * SQUISH_CONSTANT2
                        dy_ext = dy0 - 1 - 2 * SQUISH_CONSTANT2
                else:
                    zins = 2 - in_sum
                    if zins < xins or zins < yins:
                        if xins > yins:
                            xsv_ext = xsb + 2
                            ysv_ext = ysb + 0
                            dx_ext = dx0 - 2 - 2 * SQUISH_CONSTANT2
                            dy_ext = dy0 + 0 - 2 * SQUISH_CONSTANT2
                        else:
                            xsv_ext = xsb + 0
                            ysv_ext = ysb + 2
                            dx_ext = dx0 + 0 - 2 * SQUISH_CONSTANT2
                            dy_ext = dy0 - 2 - 2 * SQUISH_CONSTANT2
                    else:
                        dx_ext = dx0
                        dy_ext = dy0
                        xsv_ext = xsb
                        ysv_ext = ysb
                    xsb += 1
                    ysb += 1
                    dx0 = dx0 - 1 - 2 * SQUISH_CONSTANT2
                    dy0 = dy0 - 1 - 2 * SQUISH_CONSTANT2
                attn0 = 2 - dx0 * dx0 - dy0 * dy0
                if attn0 > 0:
                    attn0 *= attn0
                    value += attn0 * attn0 * _extrapolate2(perm, xsb, ysb, dx0, dy0)
                attn_ext = 2 - dx_ext * dx_ext - dy_ext * dy_ext
                if attn_ext > 0:
                    attn_ext *= attn_ext
                    value += attn_ext * attn_ext * _extrapolate2(perm, xsv_ext, ysv_ext, dx_ext, dy_ext)
                result.numbers[count] = value / NORM_CONSTANT2
                count += 1
    return result


def noise(Vector seed, *args):
    cdef int n = len(args)
    cdef Vector xx, yy, zz
    cdef Vector perm = PermCache.get(seed)
    if perm is None:
        perm = shuffle(Uniform(seed), Vector.range(256))
        if len(PermCache) == MAX_PERM_CACHE_ITEMS:
            PermCache.pop(next(iter(PermCache)))
        PermCache[seed] = perm
    if n == 1:
        xx = args[0]
        if xx.numbers != NULL:
            return noise2(perm, Vector(0), xx)
    elif n == 2:
        xx = args[0]
        yy = args[1]
        if xx.numbers != NULL and yy.numbers != NULL:
            return noise2(perm, xx, yy)
    return null_


NOISE_FUNCTIONS = {
    'noise': Vector(noise),
}
