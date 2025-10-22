
import cython

from libc.stdint cimport int64_t

from ...model cimport Vector, Matrix33, false_, null_, cost, sint


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
@cython.cpow(True)
def colortemp(Vector t, Vector normalize=false_):
    if t.numbers == NULL:
        return null_
    cdef int64_t i, n = t.length
    cdef bint norm = normalize.as_bool()
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
        Y = 1 if norm else (max(0, t.numbers[i]) / 6503.5)**4
        X = Y * x / y
        Z = Y * (1 - x - y) / y
        rgb.numbers[3*i] = max(0, 3.2406255*X - 1.537208*Y - 0.4986286*Z)
        rgb.numbers[3*i+1] = max(0, -0.9689307*X + 1.8757561*Y + 0.0415175*Z)
        rgb.numbers[3*i+2] = max(0, 0.0557101*X - 0.2040211*Y + 1.0569959*Z)
    return rgb


cdef Matrix33 OKLab_M1_inv = Matrix33([0.8189330101, 0.0329845436, 0.0482003018,
                                       0.3618667424, 0.9293118715, 0.2643662691,
                                       -0.1288597137, 0.0361456387, 0.6338517070]).inverse()
cdef Matrix33 OKLab_M2_inv = Matrix33([0.2104542553, 1.9779984951, 0.0259040371,
                                       0.7936177850, -2.4285922050, 0.7827717662,
                                       -0.0040720468, 0.4505937099, -0.8086757660]).inverse()
cdef Matrix33 XYZ_to_sRGB = Matrix33([3.2406, -0.9689, 0.0557,
                                      -1.5372, 1.8758, -0.2040,
                                      -0.4986, 0.0415, 1.0570])
cdef Vector Three = Vector(3)


def oklab(Vector lab):
    if lab.length != 3 or lab.objects is not None:
        return null_
    return XYZ_to_sRGB.vmul(OKLab_M1_inv.vmul(OKLab_M2_inv.vmul(lab).pow(Three)))


def oklch(Vector lch):
    if lch.length != 3 or lch.objects is not None:
        return null_
    cdef Vector lab = Vector.__new__(Vector)
    cdef double L=lch.numbers[0], C=lch.numbers[1], h=lch.numbers[2]
    lab.allocate_numbers(3)
    lab.numbers[0] = L
    lab.numbers[1] = C * cost(h)
    lab.numbers[2] = C * sint(h)
    return XYZ_to_sRGB.vmul(OKLab_M1_inv.vmul(OKLab_M2_inv.vmul(lab).pow(Three)))
