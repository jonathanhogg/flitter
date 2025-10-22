
from ...model cimport Vector, null_


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
