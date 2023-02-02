# cython: language_level=3, profile=True

import cython
from skia import PathBuilder

from libc.math cimport acos, sqrt

from ..model cimport Vector


DEF TwoPI = 6.283185307179586


@cython.cdivision(True)
cdef double turn_angle(double x0, double y0, double x1, double y1, double x2, double y2):
    cdef double xa=x1-x0, ya=y1-y0, xb=x2-x1, yb=y2-y1
    cdef double la=sqrt(xa*xa + ya*ya), lb=sqrt(xb*xb + yb*yb)
    if la == 0 or lb == 0:
        return 0
    return acos(min(max(0, (xa*xb + ya*yb) / (la*lb)), 1)) / TwoPI


@cython.boundscheck(False)
@cython.wraparound(False)
def line_path(Vector points not None, double curve, bint close):
    cdef int i, n=points.length
    assert points.numbers != NULL
    cdef double last_mid_x, last_mid_y, last_x, last_y, x, y
    builder = PathBuilder()
    lineTo = builder.lineTo
    quadTo = builder.quadTo
    for i in range(0, n, 2):
        x, y = points.numbers[i], points.numbers[i+1]
        if i == 0:
            builder.moveTo(x, y)
        elif curve <= 0:
            lineTo(x, y)
        else:
            mid_x, mid_y = (last_x + x) / 2, (last_y + y) / 2
            if i == 2:
                lineTo(mid_x, mid_y)
            elif curve >= 0.5 or turn_angle(last_mid_x, last_mid_y, last_x, last_y, mid_x, mid_y) <= curve:
                quadTo(last_x, last_y, mid_x, mid_y)
            else:
                lineTo(last_x, last_y)
                lineTo(mid_x, mid_y)
            if i == n-2:
                lineTo(x, y)
            last_mid_x, last_mid_y = mid_x, mid_y
        last_x, last_y = x, y
    if close:
        builder.close()
    return builder.detach()
