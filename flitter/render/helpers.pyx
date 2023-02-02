# cython: language_level=3, profile=True

import cython
import skia

from libc.math cimport acos, sqrt


DEF TwoPI = 6.283185307179586


cdef double turn_angle(double x0, double y0, double x1, double y1, double x2, double y2):
    cdef double xa=x1-x0, ya=y1-y0, xb=x2-x1, yb=y2-y1
    cdef double la=sqrt(xa*xa + ya*ya), lb=sqrt(xb*xb + yb*yb)
    if la == 0 or lb == 0:
        return 0
    return acos(min(max(0, (xa*xb + ya*yb) / (la*lb)), 1)) / TwoPI


@cython.boundscheck(False)
@cython.wraparound(False)
def line_path(list points not None, double curve, bint close):
    cdef int i, n=len(points)
    cdef double last_mid_x, last_mid_y, last_x, last_y, x, y
    builder = skia.PathBuilder()
    moveTo = builder.moveTo
    lineTo = builder.lineTo
    quadTo = builder.quadTo
    for i in range(0, n, 2):
        x, y = points[i], points[i+1]
        if i == 0:
            moveTo(x, y)
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
