
"""
Flitter language functions
"""


import cython

from libc.stdint cimport int64_t
from libc.math cimport floor, sin, sqrt
from cpython.ref cimport Py_INCREF
from cpython.tuple cimport PyTuple_New, PyTuple_GET_ITEM, PyTuple_SET_ITEM

from ...cache import SharedCache
from ...model cimport Vector, Matrix33, Matrix44, Quaternion, Context, null_, cost


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
    cdef const float[:, :, :] data = None
    if coord.numbers != NULL \
            and (scene_node := context.references.get(texture_id.as_string())) is not None \
            and hasattr(scene_node, 'array'):
        data = scene_node.array
    if data is None:
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
        text += chr(<int>floor(ordinals.numbers[i]))
    cdef Vector result = Vector.__new__(Vector)
    result.objects = (text,)
    result.length = 1
    return result


def split(Vector text, Vector separator=Vector('\n')):
    if text.length == 0 or separator.length == 0:
        return null_
    cdef str sep = separator.as_string()
    cdef list values = text.as_string().split(sep)
    while values[-1] == '':
        values.pop()
    return Vector._coerce(values)


def lenv(Vector xs not None):
    cdef Vector ys = Vector.__new__(Vector)
    ys.allocate_numbers(1)
    ys.numbers[0] = <double>(xs.length)
    return ys


def sort(Vector xs not None):
    return Vector(sorted(xs))


def sine(Vector xs not None):
    if xs.numbers == NULL:
        return null_
    cdef Vector ys = Vector.__new__(Vector)
    for i in range(ys.allocate_numbers(xs.length)):
        ys.numbers[i] = (1 - cost(xs.numbers[i])) / 2
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
    ys.numbers[0] = <double>(j)
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
    ys.numbers[0] = <double>(j)
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
        zs.numbers[i] = <double>(k)
    return zs


def point_towards(Vector direction, Vector up):
    cdef Matrix44 matrix = Matrix44._look(Vector([0, 0, 0]), direction, up)
    if matrix is not None:
        return matrix.inverse()
    return null_


def inverse(Vector matrix):
    if matrix.numbers != NULL:
        if matrix.length == 9:
            return Matrix33(matrix).inverse()
        if matrix.length == 16:
            return Matrix44(matrix).inverse()
    return null_
