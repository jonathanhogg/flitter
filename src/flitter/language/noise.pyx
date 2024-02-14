# cython: language_level=3, profile=False

"""
OpenSimplex 2(s) noise generation

Lifted from https://github.com/lmas/opensimplex, which itself was adapted from
https://gist.github.com/KdotJPG/b1270127455a94ac5d19
"""

import cython

from libc.math cimport floor

from .functions cimport uniform, shuffle
from ..model cimport Vector, null_


cdef Vector PERM_RANGE = Vector.range(256)

cdef Vector GRADIENTS2 = Vector([5, 2, 2, 5, -5, 2, -2, 5, 5, -2, 2, -5, -5, -2, -2, -5])
cdef double STRETCH_CONSTANT2 = -0.211324865405187
cdef double SQUISH_CONSTANT2 = 0.366025403784439
cdef double NORM_CONSTANT2 = 47

cdef Vector GRADIENTS3 = Vector([
    -11, 4, 4, -4, 11, 4, -4, 4, 11,
    11, 4, 4, 4, 11, 4, 4, 4, 11,
    -11, -4, 4, -4, -11, 4, -4, -4, 11,
    11, -4, 4, 4, -11, 4, 4, -4, 11,
    -11, 4, -4, -4, 11, -4, -4, 4, -11,
    11, 4, -4, 4, 11, -4, 4, 4, -11,
    -11, -4, -4, -4, -11, -4, -4, -4, -11,
    11, -4, -4, 4, -11, -4, 4, -4, -11,
])
cdef double STRETCH_CONSTANT3 = -1.0 / 6
cdef double SQUISH_CONSTANT3 = 1.0 / 3
cdef double NORM_CONSTANT3 = 103

cdef int MAX_PERM_CACHE_ITEMS = 1000
cdef dict PermCache = {}


cdef inline double extrapolate2(Vector perm, long xsb, long ysb, double dx, double dy) noexcept nogil:
    cdef long index = <long>(perm.numbers[(<long>(perm.numbers[xsb % 256]) + ysb) % 256]) % 8 * 2
    return GRADIENTS2.numbers[index] * dx + GRADIENTS2.numbers[index+1] * dy


cdef inline double extrapolate3(Vector perm, long xsb, long ysb, long zsb, double dx, double dy, double dz) noexcept nogil:
    cdef long index = <long>(perm.numbers[(<long>(perm.numbers[(<long>(perm.numbers[xsb % 256]) + ysb) % 256]) + zsb) % 256]) % 24 * 3
    return GRADIENTS3.numbers[index] * dx + GRADIENTS3.numbers[index+1] * dy + GRADIENTS3.numbers[index+2] * dz


@cython.cdivision(True)
cdef double noise2(Vector perm, double x, double y) noexcept nogil:
    cdef long xsb, ysb, xsv_ext, ysv_ext
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
    value = 0.0
    dx1 = dx0 - 1 - SQUISH_CONSTANT2
    dy1 = dy0 - 0 - SQUISH_CONSTANT2
    attn1 = 2 - dx1 * dx1 - dy1 * dy1
    if attn1 > 0:
        attn1 *= attn1
        value += attn1 * attn1 * extrapolate2(perm, xsb + 1, ysb + 0, dx1, dy1)
    dx2 = dx0 - 0 - SQUISH_CONSTANT2
    dy2 = dy0 - 1 - SQUISH_CONSTANT2
    attn2 = 2 - dx2 * dx2 - dy2 * dy2
    if attn2 > 0:
        attn2 *= attn2
        value += attn2 * attn2 * extrapolate2(perm, xsb + 0, ysb + 1, dx2, dy2)
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
        value += attn0 * attn0 * extrapolate2(perm, xsb, ysb, dx0, dy0)
    attn_ext = 2 - dx_ext * dx_ext - dy_ext * dy_ext
    if attn_ext > 0:
        attn_ext *= attn_ext
        value += attn_ext * attn_ext * extrapolate2(perm, xsv_ext, ysv_ext, dx_ext, dy_ext)
    return value / NORM_CONSTANT2


@cython.cdivision(True)
cdef double noise3(Vector perm, double x, double y, double z) noexcept nogil:
    cdef long xsb, ysb, zsb, xsv_ext0, xsv_ext1, ysv_ext0, ysv_ext1, zsv_ext0, zsv_ext1
    stretch_offset = (x + y + z) * STRETCH_CONSTANT3
    xs = x + stretch_offset
    ys = y + stretch_offset
    zs = z + stretch_offset
    xsb = <long>floor(xs)
    ysb = <long>floor(ys)
    zsb = <long>floor(zs)
    squish_offset = (xsb + ysb + zsb) * SQUISH_CONSTANT3
    xb = xsb + squish_offset
    yb = ysb + squish_offset
    zb = zsb + squish_offset
    xins = xs - xsb
    yins = ys - ysb
    zins = zs - zsb
    in_sum = xins + yins + zins
    dx0 = x - xb
    dy0 = y - yb
    dz0 = z - zb
    value = 0.0
    if in_sum <= 1:
        a_point = 0x01
        a_score = xins
        b_point = 0x02
        b_score = yins
        if a_score >= b_score and zins > b_score:
            b_score = zins
            b_point = 0x04
        elif a_score < b_score and zins > a_score:
            a_score = zins
            a_point = 0x04
        wins = 1 - in_sum
        if wins > a_score or wins > b_score:
            c = b_point if (b_score > a_score) else a_point
            if (c & 0x01) == 0:
                xsv_ext0 = xsb - 1
                xsv_ext1 = xsb
                dx_ext0 = dx0 + 1
                dx_ext1 = dx0
            else:
                xsv_ext0 = xsv_ext1 = xsb + 1
                dx_ext0 = dx_ext1 = dx0 - 1
            if (c & 0x02) == 0:
                ysv_ext0 = ysv_ext1 = ysb
                dy_ext0 = dy_ext1 = dy0
                if (c & 0x01) == 0:
                    ysv_ext1 -= 1
                    dy_ext1 += 1
                else:
                    ysv_ext0 -= 1
                    dy_ext0 += 1
            else:
                ysv_ext0 = ysv_ext1 = ysb + 1
                dy_ext0 = dy_ext1 = dy0 - 1
            if (c & 0x04) == 0:
                zsv_ext0 = zsb
                zsv_ext1 = zsb - 1
                dz_ext0 = dz0
                dz_ext1 = dz0 + 1
            else:
                zsv_ext0 = zsv_ext1 = zsb + 1
                dz_ext0 = dz_ext1 = dz0 - 1
        else:
            c = a_point | b_point
            if (c & 0x01) == 0:
                xsv_ext0 = xsb
                xsv_ext1 = xsb - 1
                dx_ext0 = dx0 - 2 * SQUISH_CONSTANT3
                dx_ext1 = dx0 + 1 - SQUISH_CONSTANT3
            else:
                xsv_ext0 = xsv_ext1 = xsb + 1
                dx_ext0 = dx0 - 1 - 2 * SQUISH_CONSTANT3
                dx_ext1 = dx0 - 1 - SQUISH_CONSTANT3
            if (c & 0x02) == 0:
                ysv_ext0 = ysb
                ysv_ext1 = ysb - 1
                dy_ext0 = dy0 - 2 * SQUISH_CONSTANT3
                dy_ext1 = dy0 + 1 - SQUISH_CONSTANT3
            else:
                ysv_ext0 = ysv_ext1 = ysb + 1
                dy_ext0 = dy0 - 1 - 2 * SQUISH_CONSTANT3
                dy_ext1 = dy0 - 1 - SQUISH_CONSTANT3
            if (c & 0x04) == 0:
                zsv_ext0 = zsb
                zsv_ext1 = zsb - 1
                dz_ext0 = dz0 - 2 * SQUISH_CONSTANT3
                dz_ext1 = dz0 + 1 - SQUISH_CONSTANT3
            else:
                zsv_ext0 = zsv_ext1 = zsb + 1
                dz_ext0 = dz0 - 1 - 2 * SQUISH_CONSTANT3
                dz_ext1 = dz0 - 1 - SQUISH_CONSTANT3
        attn0 = 2 - dx0 * dx0 - dy0 * dy0 - dz0 * dz0
        if attn0 > 0:
            attn0 *= attn0
            value += attn0 * attn0 * extrapolate3(perm, xsb + 0, ysb + 0, zsb + 0, dx0, dy0, dz0)
        dx1 = dx0 - 1 - SQUISH_CONSTANT3
        dy1 = dy0 - 0 - SQUISH_CONSTANT3
        dz1 = dz0 - 0 - SQUISH_CONSTANT3
        attn1 = 2 - dx1 * dx1 - dy1 * dy1 - dz1 * dz1
        if attn1 > 0:
            attn1 *= attn1
            value += attn1 * attn1 * extrapolate3(perm, xsb + 1, ysb + 0, zsb + 0, dx1, dy1, dz1)
        dx2 = dx0 - 0 - SQUISH_CONSTANT3
        dy2 = dy0 - 1 - SQUISH_CONSTANT3
        dz2 = dz1
        attn2 = 2 - dx2 * dx2 - dy2 * dy2 - dz2 * dz2
        if attn2 > 0:
            attn2 *= attn2
            value += attn2 * attn2 * extrapolate3(perm, xsb + 0, ysb + 1, zsb + 0, dx2, dy2, dz2)
        dx3 = dx2
        dy3 = dy1
        dz3 = dz0 - 1 - SQUISH_CONSTANT3
        attn3 = 2 - dx3 * dx3 - dy3 * dy3 - dz3 * dz3
        if attn3 > 0:
            attn3 *= attn3
            value += attn3 * attn3 * extrapolate3(perm, xsb + 0, ysb + 0, zsb + 1, dx3, dy3, dz3)
    elif in_sum >= 2:
        a_point = 0x06
        a_score = xins
        b_point = 0x05
        b_score = yins
        if a_score <= b_score and zins < b_score:
            b_score = zins
            b_point = 0x03
        elif a_score > b_score and zins < a_score:
            a_score = zins
            a_point = 0x03
        wins = 3 - in_sum
        if wins < a_score or wins < b_score:
            c = b_point if (b_score < a_score) else a_point
            if (c & 0x01) != 0:
                xsv_ext0 = xsb + 2
                xsv_ext1 = xsb + 1
                dx_ext0 = dx0 - 2 - 3 * SQUISH_CONSTANT3
                dx_ext1 = dx0 - 1 - 3 * SQUISH_CONSTANT3
            else:
                xsv_ext0 = xsv_ext1 = xsb
                dx_ext0 = dx_ext1 = dx0 - 3 * SQUISH_CONSTANT3
            if (c & 0x02) != 0:
                ysv_ext0 = ysv_ext1 = ysb + 1
                dy_ext0 = dy_ext1 = dy0 - 1 - 3 * SQUISH_CONSTANT3
                if (c & 0x01) != 0:
                    ysv_ext1 += 1
                    dy_ext1 -= 1
                else:
                    ysv_ext0 += 1
                    dy_ext0 -= 1
            else:
                ysv_ext0 = ysv_ext1 = ysb
                dy_ext0 = dy_ext1 = dy0 - 3 * SQUISH_CONSTANT3
            if (c & 0x04) != 0:
                zsv_ext0 = zsb + 1
                zsv_ext1 = zsb + 2
                dz_ext0 = dz0 - 1 - 3 * SQUISH_CONSTANT3
                dz_ext1 = dz0 - 2 - 3 * SQUISH_CONSTANT3
            else:
                zsv_ext0 = zsv_ext1 = zsb
                dz_ext0 = dz_ext1 = dz0 - 3 * SQUISH_CONSTANT3
        else:
            c = a_point & b_point
            if (c & 0x01) != 0:
                xsv_ext0 = xsb + 1
                xsv_ext1 = xsb + 2
                dx_ext0 = dx0 - 1 - SQUISH_CONSTANT3
                dx_ext1 = dx0 - 2 - 2 * SQUISH_CONSTANT3
            else:
                xsv_ext0 = xsv_ext1 = xsb
                dx_ext0 = dx0 - SQUISH_CONSTANT3
                dx_ext1 = dx0 - 2 * SQUISH_CONSTANT3
            if (c & 0x02) != 0:
                ysv_ext0 = ysb + 1
                ysv_ext1 = ysb + 2
                dy_ext0 = dy0 - 1 - SQUISH_CONSTANT3
                dy_ext1 = dy0 - 2 - 2 * SQUISH_CONSTANT3
            else:
                ysv_ext0 = ysv_ext1 = ysb
                dy_ext0 = dy0 - SQUISH_CONSTANT3
                dy_ext1 = dy0 - 2 * SQUISH_CONSTANT3
            if (c & 0x04) != 0:
                zsv_ext0 = zsb + 1
                zsv_ext1 = zsb + 2
                dz_ext0 = dz0 - 1 - SQUISH_CONSTANT3
                dz_ext1 = dz0 - 2 - 2 * SQUISH_CONSTANT3
            else:
                zsv_ext0 = zsv_ext1 = zsb
                dz_ext0 = dz0 - SQUISH_CONSTANT3
                dz_ext1 = dz0 - 2 * SQUISH_CONSTANT3
        dx3 = dx0 - 1 - 2 * SQUISH_CONSTANT3
        dy3 = dy0 - 1 - 2 * SQUISH_CONSTANT3
        dz3 = dz0 - 0 - 2 * SQUISH_CONSTANT3
        attn3 = 2 - dx3 * dx3 - dy3 * dy3 - dz3 * dz3
        if attn3 > 0:
            attn3 *= attn3
            value += attn3 * attn3 * extrapolate3(perm, xsb + 1, ysb + 1, zsb + 0, dx3, dy3, dz3)
        dx2 = dx3
        dy2 = dy0 - 0 - 2 * SQUISH_CONSTANT3
        dz2 = dz0 - 1 - 2 * SQUISH_CONSTANT3
        attn2 = 2 - dx2 * dx2 - dy2 * dy2 - dz2 * dz2
        if attn2 > 0:
            attn2 *= attn2
            value += attn2 * attn2 * extrapolate3(perm, xsb + 1, ysb + 0, zsb + 1, dx2, dy2, dz2)
        dx1 = dx0 - 0 - 2 * SQUISH_CONSTANT3
        dy1 = dy3
        dz1 = dz2
        attn1 = 2 - dx1 * dx1 - dy1 * dy1 - dz1 * dz1
        if attn1 > 0:
            attn1 *= attn1
            value += attn1 * attn1 * extrapolate3(perm, xsb + 0, ysb + 1, zsb + 1, dx1, dy1, dz1)
        dx0 = dx0 - 1 - 3 * SQUISH_CONSTANT3
        dy0 = dy0 - 1 - 3 * SQUISH_CONSTANT3
        dz0 = dz0 - 1 - 3 * SQUISH_CONSTANT3
        attn0 = 2 - dx0 * dx0 - dy0 * dy0 - dz0 * dz0
        if attn0 > 0:
            attn0 *= attn0
            value += attn0 * attn0 * extrapolate3(perm, xsb + 1, ysb + 1, zsb + 1, dx0, dy0, dz0)
    else:
        p1 = xins + yins
        if p1 > 1:
            a_score = p1 - 1
            a_point = 0x03
            a_is_further_side = True
        else:
            a_score = 1 - p1
            a_point = 0x04
            a_is_further_side = False
        p2 = xins + zins
        if p2 > 1:
            b_score = p2 - 1
            b_point = 0x05
            b_is_further_side = True
        else:
            b_score = 1 - p2
            b_point = 0x02
            b_is_further_side = False
        p3 = yins + zins
        if p3 > 1:
            score = p3 - 1
            if a_score <= b_score and a_score < score:
                a_point = 0x06
                a_is_further_side = True
            elif a_score > b_score and b_score < score:
                b_point = 0x06
                b_is_further_side = True
        else:
            score = 1 - p3
            if a_score <= b_score and a_score < score:
                a_point = 0x01
                a_is_further_side = False
            elif a_score > b_score and b_score < score:
                b_point = 0x01
                b_is_further_side = False
        if a_is_further_side == b_is_further_side:
            if a_is_further_side:
                dx_ext0 = dx0 - 1 - 3 * SQUISH_CONSTANT3
                dy_ext0 = dy0 - 1 - 3 * SQUISH_CONSTANT3
                dz_ext0 = dz0 - 1 - 3 * SQUISH_CONSTANT3
                xsv_ext0 = xsb + 1
                ysv_ext0 = ysb + 1
                zsv_ext0 = zsb + 1
                c = a_point & b_point
                if (c & 0x01) != 0:
                    dx_ext1 = dx0 - 2 - 2 * SQUISH_CONSTANT3
                    dy_ext1 = dy0 - 2 * SQUISH_CONSTANT3
                    dz_ext1 = dz0 - 2 * SQUISH_CONSTANT3
                    xsv_ext1 = xsb + 2
                    ysv_ext1 = ysb
                    zsv_ext1 = zsb
                elif (c & 0x02) != 0:
                    dx_ext1 = dx0 - 2 * SQUISH_CONSTANT3
                    dy_ext1 = dy0 - 2 - 2 * SQUISH_CONSTANT3
                    dz_ext1 = dz0 - 2 * SQUISH_CONSTANT3
                    xsv_ext1 = xsb
                    ysv_ext1 = ysb + 2
                    zsv_ext1 = zsb
                else:
                    dx_ext1 = dx0 - 2 * SQUISH_CONSTANT3
                    dy_ext1 = dy0 - 2 * SQUISH_CONSTANT3
                    dz_ext1 = dz0 - 2 - 2 * SQUISH_CONSTANT3
                    xsv_ext1 = xsb
                    ysv_ext1 = ysb
                    zsv_ext1 = zsb + 2
            else:
                dx_ext0 = dx0
                dy_ext0 = dy0
                dz_ext0 = dz0
                xsv_ext0 = xsb
                ysv_ext0 = ysb
                zsv_ext0 = zsb
                c = a_point | b_point
                if (c & 0x01) == 0:
                    dx_ext1 = dx0 + 1 - SQUISH_CONSTANT3
                    dy_ext1 = dy0 - 1 - SQUISH_CONSTANT3
                    dz_ext1 = dz0 - 1 - SQUISH_CONSTANT3
                    xsv_ext1 = xsb - 1
                    ysv_ext1 = ysb + 1
                    zsv_ext1 = zsb + 1
                elif (c & 0x02) == 0:
                    dx_ext1 = dx0 - 1 - SQUISH_CONSTANT3
                    dy_ext1 = dy0 + 1 - SQUISH_CONSTANT3
                    dz_ext1 = dz0 - 1 - SQUISH_CONSTANT3
                    xsv_ext1 = xsb + 1
                    ysv_ext1 = ysb - 1
                    zsv_ext1 = zsb + 1
                else:
                    dx_ext1 = dx0 - 1 - SQUISH_CONSTANT3
                    dy_ext1 = dy0 - 1 - SQUISH_CONSTANT3
                    dz_ext1 = dz0 + 1 - SQUISH_CONSTANT3
                    xsv_ext1 = xsb + 1
                    ysv_ext1 = ysb + 1
                    zsv_ext1 = zsb - 1
        else:
            if a_is_further_side:
                c1 = a_point
                c2 = b_point
            else:
                c1 = b_point
                c2 = a_point
            if (c1 & 0x01) == 0:
                dx_ext0 = dx0 + 1 - SQUISH_CONSTANT3
                dy_ext0 = dy0 - 1 - SQUISH_CONSTANT3
                dz_ext0 = dz0 - 1 - SQUISH_CONSTANT3
                xsv_ext0 = xsb - 1
                ysv_ext0 = ysb + 1
                zsv_ext0 = zsb + 1
            elif (c1 & 0x02) == 0:
                dx_ext0 = dx0 - 1 - SQUISH_CONSTANT3
                dy_ext0 = dy0 + 1 - SQUISH_CONSTANT3
                dz_ext0 = dz0 - 1 - SQUISH_CONSTANT3
                xsv_ext0 = xsb + 1
                ysv_ext0 = ysb - 1
                zsv_ext0 = zsb + 1
            else:
                dx_ext0 = dx0 - 1 - SQUISH_CONSTANT3
                dy_ext0 = dy0 - 1 - SQUISH_CONSTANT3
                dz_ext0 = dz0 + 1 - SQUISH_CONSTANT3
                xsv_ext0 = xsb + 1
                ysv_ext0 = ysb + 1
                zsv_ext0 = zsb - 1
            dx_ext1 = dx0 - 2 * SQUISH_CONSTANT3
            dy_ext1 = dy0 - 2 * SQUISH_CONSTANT3
            dz_ext1 = dz0 - 2 * SQUISH_CONSTANT3
            xsv_ext1 = xsb
            ysv_ext1 = ysb
            zsv_ext1 = zsb
            if (c2 & 0x01) != 0:
                dx_ext1 -= 2
                xsv_ext1 += 2
            elif (c2 & 0x02) != 0:
                dy_ext1 -= 2
                ysv_ext1 += 2
            else:
                dz_ext1 -= 2
                zsv_ext1 += 2
        dx1 = dx0 - 1 - SQUISH_CONSTANT3
        dy1 = dy0 - 0 - SQUISH_CONSTANT3
        dz1 = dz0 - 0 - SQUISH_CONSTANT3
        attn1 = 2 - dx1 * dx1 - dy1 * dy1 - dz1 * dz1
        if attn1 > 0:
            attn1 *= attn1
            value += attn1 * attn1 * extrapolate3(perm, xsb + 1, ysb + 0, zsb + 0, dx1, dy1, dz1)
        dx2 = dx0 - 0 - SQUISH_CONSTANT3
        dy2 = dy0 - 1 - SQUISH_CONSTANT3
        dz2 = dz1
        attn2 = 2 - dx2 * dx2 - dy2 * dy2 - dz2 * dz2
        if attn2 > 0:
            attn2 *= attn2
            value += attn2 * attn2 * extrapolate3(perm, xsb + 0, ysb + 1, zsb + 0, dx2, dy2, dz2)
        dx3 = dx2
        dy3 = dy1
        dz3 = dz0 - 1 - SQUISH_CONSTANT3
        attn3 = 2 - dx3 * dx3 - dy3 * dy3 - dz3 * dz3
        if attn3 > 0:
            attn3 *= attn3
            value += attn3 * attn3 * extrapolate3(perm, xsb + 0, ysb + 0, zsb + 1, dx3, dy3, dz3)
        dx4 = dx0 - 1 - 2 * SQUISH_CONSTANT3
        dy4 = dy0 - 1 - 2 * SQUISH_CONSTANT3
        dz4 = dz0 - 0 - 2 * SQUISH_CONSTANT3
        attn4 = 2 - dx4 * dx4 - dy4 * dy4 - dz4 * dz4
        if attn4 > 0:
            attn4 *= attn4
            value += attn4 * attn4 * extrapolate3(perm, xsb + 1, ysb + 1, zsb + 0, dx4, dy4, dz4)
        dx5 = dx4
        dy5 = dy0 - 0 - 2 * SQUISH_CONSTANT3
        dz5 = dz0 - 1 - 2 * SQUISH_CONSTANT3
        attn5 = 2 - dx5 * dx5 - dy5 * dy5 - dz5 * dz5
        if attn5 > 0:
            attn5 *= attn5
            value += attn5 * attn5 * extrapolate3(perm, xsb + 1, ysb + 0, zsb + 1, dx5, dy5, dz5)
        dx6 = dx0 - 0 - 2 * SQUISH_CONSTANT3
        dy6 = dy4
        dz6 = dz5
        attn6 = 2 - dx6 * dx6 - dy6 * dy6 - dz6 * dz6
        if attn6 > 0:
            attn6 *= attn6
            value += attn6 * attn6 * extrapolate3(perm, xsb + 0, ysb + 1, zsb + 1, dx6, dy6, dz6)
    attn_ext0 = 2 - dx_ext0 * dx_ext0 - dy_ext0 * dy_ext0 - dz_ext0 * dz_ext0
    if attn_ext0 > 0:
        attn_ext0 *= attn_ext0
        value += attn_ext0 * attn_ext0 * extrapolate3(perm, xsv_ext0, ysv_ext0, zsv_ext0, dx_ext0, dy_ext0, dz_ext0)
    attn_ext1 = 2 - dx_ext1 * dx_ext1 - dy_ext1 * dy_ext1 - dz_ext1 * dz_ext1
    if attn_ext1 > 0:
        attn_ext1 *= attn_ext1
        value += attn_ext1 * attn_ext1 * extrapolate3(perm, xsv_ext1, ysv_ext1, zsv_ext1, dx_ext1, dy_ext1, dz_ext1)
    return value / NORM_CONSTANT3


cdef Vector _noise(Vector perm, list args):
    cdef int i, j, k, count = 0, n = len(args)
    cdef Vector xx, yy, zz
    cdef double x, y
    cdef Vector result = Vector.__new__(Vector)
    if n == 1:
        xx = args[0]
        if xx.numbers != NULL:
            result.allocate_numbers(xx.length)
            with nogil:
                for i in range(xx.length):
                    result.numbers[i] = noise2(perm, xx.numbers[i], 0)
    elif n == 2:
        xx = args[0]
        yy = args[1]
        if xx.numbers != NULL and yy.numbers != NULL:
            result.allocate_numbers(xx.length * yy.length)
            with nogil:
                for i in range(xx.length):
                    x = xx.numbers[i]
                    for j in range(yy.length):
                        result.numbers[count] = noise2(perm, x, yy.numbers[j])
                        count += 1
    elif n == 3:
        xx = args[0]
        yy = args[1]
        zz = args[2]
        if xx.numbers != NULL and yy.numbers != NULL and zz.numbers != NULL:
            result.allocate_numbers(xx.length * yy.length * zz.length)
            with nogil:
                for i in range(xx.length):
                    x = xx.numbers[i]
                    for j in range(yy.length):
                        y = yy.numbers[j]
                        for k in range(zz.length):
                            result.numbers[count] = noise3(perm, x, y, zz.numbers[k])
                            count += 1
    return result


cdef Vector get_perm(Vector seed, int i):
    cdef unsigned long long seed_hash = seed.hash(True) ^ <unsigned long long>i
    cdef uniform prng
    cdef Vector perm = <Vector>PermCache.get(seed_hash)
    if perm is None:
        prng = uniform.__new__(uniform)
        prng._hash = seed_hash
        perm = shuffle(prng, PERM_RANGE)
        if len(PermCache) == MAX_PERM_CACHE_ITEMS:
            PermCache.pop(next(iter(PermCache)))
        PermCache[seed_hash] = perm
    return perm


def noise(Vector seed, *args):
    return _noise(get_perm(seed, 0), list(args))


@cython.cdivision(True)
def octnoise(Vector seed, Vector octaves, Vector roughness, *args):
    if octaves.numbers == NULL or octaves.length != 1 or roughness.numbers == NULL or roughness.length != 1:
        return null_
    cdef Vector arg
    cdef list coords = []
    for arg in args:
        if arg.numbers == NULL:
            return null_
        coords.append(Vector._copy(arg))
    cdef int i, j, n = <int>octaves.numbers[0]
    cdef double weight_sum = 0, weight = 1, k = roughness.numbers[0]
    cdef Vector single, result = null_
    for i in range(n):
        if i == 0:
            result = _noise(get_perm(seed, i), coords)
        else:
            single = _noise(get_perm(seed, i), coords)
            for j in range(result.length):
                result.numbers[j] += single.numbers[j] * weight
        for arg in coords:
            for j in range(arg.length):
                arg.numbers[j] *= 2
        weight_sum += weight
        weight *= k
    for j in range(result.length):
        result.numbers[j] /= weight_sum
    return result


NOISE_FUNCTIONS = {
    'noise': Vector(noise),
    'octnoise': Vector(octnoise),
}
