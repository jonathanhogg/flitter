# cython: language_level=3, profile=True

"""
Flitter laser control
"""

import asyncio
import logging
import struct

import cython
from libc.math cimport sqrt, abs, round, isnan, cos, sin
import numpy as np
import usb.core

from .. cimport model


cdef double TWO_PI = 2*np.pi

Log = logging.getLogger(__name__)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:] travel(double distance, double accelleration, double start_speed, double end_speed):
    cdef double d0, t0, s, t1, tt, tm, t, tp, d
    cdef int i, n
    cdef double[:] ds
    d0 = abs(end_speed*end_speed - start_speed*start_speed) / (2*accelleration)
    t0 = abs(end_speed - start_speed) / accelleration
    s = max(start_speed, end_speed)
    t1 = (sqrt(s*s + 2*accelleration*(distance-d0) / 2) - s) / accelleration
    tt = t0 + 2*t1
    tm = t0 + t1 if end_speed > start_speed else t1
    n = max(<int>(round(tt)), 1)
    distances = np.zeros(n)
    ds = distances
    for i in range(1, n+1):
        t = i * tt / n
        tp = min(tm, t)
        d = start_speed*t + tp*tp*accelleration/2
        tp = max(0, t-tm)
        d += tp*tm*accelleration - tp*tp*accelleration/2
        ds[i-1] = d
    return distances

cdef double max_speed(double distance, double accelleration, double start_speed):
    return sqrt(2*distance*accelleration + start_speed * start_speed)


cdef class LaserDriver:
    DEFAULT_SAMPLE_RATE = 30000
    DEFAULT_ACCELLERATION = 10
    DEFAULT_EPSILON = 0.0001

    cdef list _sample_bunches
    cdef int _sample_rate
    cdef double _accelleration, _epsilon

    def __init__(self, sample_rate=None, accelleration=None, epsilon=None):
        self._sample_bunches = [np.array([(0.5, 0.5, 0, 0, 0)])]
        self._sample_rate = sample_rate if sample_rate is not None else self.DEFAULT_SAMPLE_RATE
        self._accelleration = accelleration if accelleration is not None else self.DEFAULT_ACCELLERATION
        self._epsilon = epsilon if epsilon is not None else self.DEFAULT_EPSILON

    def close(self):
        pass

    cpdef flush(self):
        raise NotImplementedError()

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef draw_path(self, points, tuple color):
        cdef int i, j, n, m
        cdef double a, s, dv, dx, dy, l, x, y
        cdef double[:] accels, distances, speeds, lengths
        cdef double[:, :] samples, coords, normals
        cdef double r, g, b
        r = min(max(0, <double>(color[0])), 1)
        g = min(max(0, <double>(color[1])), 1)
        b = min(max(0, <double>(color[2])), 1)
        a = self._accelleration / self._sample_rate
        coords = np.vstack((self._sample_bunches[len(self._sample_bunches)-1][-1:, :2], points))
        n = len(coords)
        lengths = np.zeros(n-1)
        normals = np.zeros((n-1, 2))
        accels = np.zeros(n-1)
        i = 0
        for j in range(1, n):
            dx = coords[j, 0] - coords[i, 0]
            dy = coords[j, 1] - coords[i, 1]
            l = sqrt(dx*dx + dy*dy)
            if l < self._epsilon:
                n -= 1
                continue
            lengths[i] = l
            normals[i, 0] = dx / l
            normals[i, 1] = dy / l
            accels[i] = a / (max(abs(dx), abs(dy)) / l)
            i += 1
            if i != j:
                coords[i, 0] = coords[j, 0]
                coords[i, 1] = coords[j, 1]
        if n == 1:
            return
        speeds = np.zeros(n)
        for i in range(1, n-1):
            s = max_speed(lengths[i-1], accels[i-1], speeds[i-1])
            dv = max(abs(normals[i, 0] - normals[i-1, 0]), abs(normals[i, 1] - normals[i-1, 1]))
            speeds[i] = min(s, a / 2 / dv) if dv else s
        for i in range(n-2, 0, -1):
            speeds[i] = min(speeds[i], max_speed(lengths[i], accels[i], speeds[i+1]))
        for i in range(n-1):
            distances = travel(lengths[i], accels[i], speeds[i], speeds[i+1])
            m = len(distances)
            bunch = np.zeros((m, 5))
            samples = bunch
            for j in range(m):
                x = normals[i, 0] * distances[j] + coords[i, 0]
                y = normals[i, 1] * distances[j] + coords[i, 1]
                samples[j, 0] = min(max(0, x), 1)
                samples[j, 1] = min(max(0, y), 1)
                if i and 0 <= x <= 1 and 0 <= y <= 1:
                    samples[j, 2] = r
                    samples[j, 3] = g
                    samples[j, 4] = b
            self._sample_bunches.append(bunch)


cdef class LaserCubeDriver(LaserDriver):
    DEFAULT_SAMPLE_RATE = 50000  # samples/second
    DEFAULT_ACCELLERATION = 20   # sweeps/second^2
    DEFAULT_DAC_LAG = 2e-4       # seconds

    SET_ENABLED = 0x80
    GET_ENABLED = 0x81
    SET_DAC_RATE = 0x82
    GET_DAC_RATE = 0x83
    GET_MAXIMUM_DAC_RATE = 0x84
    GET_SAMPLE_ELEMENT_COUNT = 0x85
    GET_ISO_PACKET_SAMPLE_COUNT = 0x86
    GET_DAC_MINIMUM_VALUE = 0x87
    GET_DAC_MAXIMUM_VALUE = 0x88
    GET_RINGBUFFER_COUNT = 0x89
    GET_EMPTY_RINGBUFFER_COUNT = 0x8a
    GET_MAJOR_VERSION = 0x8b
    GET_MINOR_VERSION = 0x8c
    CLEAR_RINGBUFFER = 0x8d

    cdef _device
    cdef _control_out, _control_in, _data_out
    cdef int _dac_min, _dac_range, _max_dac_rate
    cdef double _dac_lag

    @classmethod
    def first_connected(cls, vendor=0x1fc9, product=0x04d8, **kwargs):
        device = usb.core.find(idVendor=vendor, idProduct=product)
        if device is None:
            raise ValueError("No device found")
        return cls(device, **kwargs)

    def __init__(self, device, dac_lag=None, **kwargs):
        super().__init__(**kwargs)
        self._device = device
        self._device.set_configuration()
        self._device.set_interface_altsetting(1, 1)
        configuration = self._device.get_active_configuration()
        self._control_out, self._control_in = configuration[0, 0].endpoints()
        self._data_out, = configuration[1, 1].endpoints()
        self._dac_min = self._execute(self.GET_DAC_MINIMUM_VALUE, 'I')
        self._dac_range = self._execute(self.GET_DAC_MAXIMUM_VALUE, 'I') - self._dac_min
        self._max_dac_rate = self._execute(self.GET_MAXIMUM_DAC_RATE, 'I')
        self._sample_rate = min(max(0, self._sample_rate), self._max_dac_rate)
        self._dac_lag = dac_lag if dac_lag is not None else self.DEFAULT_DAC_LAG
        self._execute(self.SET_DAC_RATE, None, 'I', self._sample_rate)
        self._execute(self.SET_ENABLED, None, 'B', 1)
        major, minor = self._execute(self.GET_MAJOR_VERSION, 'I'), self._execute(self.GET_MINOR_VERSION, 'I')
        Log.info("Configured USB LaserCube '%s %s' (firmware %d.%d)", self._device.manufacturer, self._device.product, major, minor)

    def close(self):
        self._execute(self.SET_ENABLED, None, 'B', 0)
        self._device = self._control_in = self._control_out = self._data_out = None

    def _execute(self, int command, str results_format, str values_format='', *values):
        self._control_out.write(struct.pack('<B' + values_format, command, *values))
        data = self._control_in.read(64)
        assert data[0] == command
        if not results_format:
            return None
        _, _, *results = struct.unpack_from("<BB" + results_format, data)
        return results if len(results) > 1 else results[0]

    @cython.boundscheck(False)
    cpdef flush(self):
        cdef int i, n, lag = max(1, <int>(round(self._sample_rate * self._dac_lag)))
        samples_array = np.vstack(self._sample_bunches)
        cdef double[:, :] samples = samples_array
        cdef unsigned short[:, :] output
        if len(samples) > lag:
            n = len(samples) - lag
            output_array = np.zeros((n, 4), dtype='uint16')
            output = output_array
            for i in range(n):
                output[i, 0] = <int>(round(samples[i, 2] * 255)) + (<int>(round(samples[i, 3] * 255)) << 8)
                output[i, 1] = <int>(round(samples[i, 4] * 255))
                output[i, 2] = <int>(round((1 - samples[i+lag, 0]) * self._dac_range)) + self._dac_min
                output[i, 3] = <int>(round((1 - samples[i+lag, 1]) * self._dac_range)) + self._dac_min
            Log.debug("Writing %d sample frame", len(output))
            self._data_out.write(output_array.tobytes())
            self._sample_bunches = [samples_array[-lag:]]
        else:
            self._sample_bunches = [samples_array]


cdef class AffineTransform:
    cdef double v00, v01, v02
    cdef double v10, v11, v12
    cdef double v20, v21, v22

    @classmethod
    def identity(cls):
        cdef AffineTransform a = cls()
        a.v00 = a.v11 = a.v22 = 1
        return a

    @classmethod
    def translate(cls, double x, double y):
        cdef AffineTransform a = cls()
        a.v00 = a.v11 = a.v22 = 1
        a.v02 = x
        a.v12 = y
        return a

    @classmethod
    def scale(cls, double sx, double sy):
        cdef AffineTransform a = cls()
        a.v00 = sx
        a.v11 = sy
        a.v22 = 1
        return a

    @classmethod
    def rotate(cls, double th):
        cdef AffineTransform a = cls()
        cdef double sx = cos(th), sy = sin(th)
        a.v00 = sx
        a.v01 = -sy
        a.v10 = sy
        a.v11 = sx
        a.v22 = 1
        return a

    cpdef tuple t(self, double x, double y):
        return (x*self.v00 + y*self.v01 + self.v02, x*self.v10 + y*self.v11 + self.v12)

    def __matmul__(AffineTransform a, AffineTransform b):
        cdef AffineTransform c = AffineTransform()
        c.v00 = a.v00*b.v00 + a.v01*b.v10 + a.v02*b.v20
        c.v01 = a.v00*b.v01 + a.v01*b.v11 + a.v02*b.v21
        c.v02 = a.v00*b.v02 + a.v01*b.v12 + a.v02*b.v22
        c.v10 = a.v10*b.v00 + a.v11*b.v10 + a.v12*b.v20
        c.v11 = a.v10*b.v01 + a.v11*b.v11 + a.v12*b.v21
        c.v12 = a.v10*b.v02 + a.v11*b.v12 + a.v12*b.v22
        c.v20 = a.v20*b.v00 + a.v21*b.v10 + a.v22*b.v20
        c.v21 = a.v20*b.v01 + a.v21*b.v11 + a.v22*b.v21
        c.v22 = a.v20*b.v02 + a.v21*b.v12 + a.v22*b.v22
        return c


cdef class Laser:
    cdef LaserDriver driver

    def __init__(self):
        self.driver = None

    def update(self, model.Node node):
        driver = node['driver'].as_string().lower()
        cls = {'lasercube': LaserCubeDriver}.get(driver)
        if not isinstance(self.driver, cls):
            if self.driver is not None:
                self.driver.close()
            self.driver = cls.first_connected(sample_rate=node.get('sample_rate', 1, int),
                                               accelleration=node.get('accelleration', 1, float),
                                               epsilon=node.get('epsilon', 1, float))
        color = (0., 0., 0.)
        transform = AffineTransform.identity()
        paths = []
        self.collect_paths(node, paths, color, transform)
        paths.sort(key=lambda pc: pc[0][0])
        for points, color in paths:
            self.driver.draw_path(points, color)
        self.driver.flush()

    @cython.cdivision(True)
    def collect_paths(self, model.Node node not None, list paths not None, tuple color not None, AffineTransform transform not None):
        cdef double x, y, sx, sy, th
        cdef int i, n
        cdef list path, points

        c = node.get('color', 3, float)
        if c is not None:
            color = tuple(c)

        if node.kind in ('laser', 'group'):
            for key in node.keys():
                if key == 'translate':
                    translate = node.get('translate', 2, float)
                    transform = transform @ AffineTransform.translate(*translate)
                elif key == 'scale':
                    scale = node.get('scale', 2, float)
                    transform = transform @ AffineTransform.scale(*scale)
                elif key == 'rotate':
                    rotate = node.get('rotate', 1, float)
                    transform = transform @ AffineTransform.rotate(TWO_PI * rotate)
            for child in node.children:
                self.collect_paths(child, paths, color, transform)

        elif node.kind == 'line':
            points = node.get('points', 0, float)
            n = len(points)
            if n >= 2:
                path = []
                for i in range(0, n, 2):
                    path.append(transform.t(points[i], points[i+1]))
                if node.get('close', 1, bool, False):
                    path.append(path[0])
                paths.append((path, color))

        elif node.kind == 'rect':
            size = node.get('size', 2, float)
            if size is not None:
                sx, sy = size
                x, y = node.get('point', 2, float, (0, 0))
                path = [transform.t(x, y), transform.t(x+sx, y), transform.t(x+sx, y+sy), transform.t(x, y+sy), transform.t(x, y)]
                paths.append((path, color))

        elif node.kind == 'ellipse':
            radius = node.get('radius', 2, float)
            if radius is not None:
                sx, sy = radius
                x, y = node.get('point', 2, float, (0, 0))
                n = node.get('segments', 1, int, 60)
                path = []
                for i in range(n+1):
                    th = TWO_PI * i/n
                    path.append(transform.t(x + sx*cos(th), y + sy*sin(th)))
                paths.append((path, color))
