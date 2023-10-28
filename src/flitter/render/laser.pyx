# cython: language_level=3, profile=True

"""
Flitter laser control
"""

import asyncio
import struct

import cython
from loguru import logger
from libc.math cimport sqrt, abs, round, isnan, cos, sin
import numpy as np
import usb.core

from .. import name_patch
from ..model cimport Node, Vector, Matrix33, null_


logger = name_patch(logger, __name__)


cdef double Tau = 2*np.pi
cdef Vector Zero2 = Vector((0, 0))
cdef Vector Half2 = Vector((0.5, 0.5))
cdef Vector Zero3 = Vector((0, 0, 0))


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef Vector travel(double distance, double accelleration, double start_speed, double end_speed):
    cdef double d0, t0, s, t1, tt, tm, t, tp, d
    cdef int i, n
    d0 = abs(end_speed*end_speed - start_speed*start_speed) / (2*accelleration)
    t0 = abs(end_speed - start_speed) / accelleration
    s = max(start_speed, end_speed)
    t1 = (sqrt(s*s + 2*accelleration*(distance-d0) / 2) - s) / accelleration
    tt = t0 + 2*t1
    tm = t0 + t1 if end_speed > start_speed else t1
    n = max(<int>(round(tt)), 1)
    cdef Vector distances = Vector.__new__(Vector)
    distances.allocate_numbers(n)
    for i in range(1, n+1):
        t = i * tt / n
        tp = min(tm, t)
        d = start_speed*t + tp*tp*accelleration/2
        tp = max(0, t-tm)
        d += tp*tm*accelleration - tp*tp*accelleration/2
        distances.numbers[i-1] = d
    return distances

cdef inline double max_speed(double distance, double accelleration, double start_speed):
    return sqrt(2*distance*accelleration + start_speed*start_speed)

cdef double distance(tuple p0, tuple p1):
    cdef double x0, y0, x1, y1, dx, dy
    x0, y0 = p0
    x1, y1 = p1
    dx = x1 - x0
    dy = y1 - y0
    return sqrt(dx*dx + dy*dy)


cdef class LaserDriver:
    DEFAULT_SAMPLE_RATE = 30000
    DEFAULT_ACCELLERATION = 10
    DEFAULT_EPSILON = 0.0001

    cdef int _sample_rate
    cdef double _accelleration, _epsilon

    @property
    def sample_rate(self):
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, int value):
        self._sample_rate = value

    @property
    def accelleration(self):
        return self._accelleration

    @accelleration.setter
    def accelleration(self, double value):
        self._accelleration = value

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, double value):
        self._epsilon = value

    def __init__(self):
        self._sample_rate = self.DEFAULT_SAMPLE_RATE
        self._accelleration = self.DEFAULT_ACCELLERATION
        self._epsilon = self.DEFAULT_EPSILON

    async def start(self):
        raise NotImplementedError()

    async def stop(self):
        raise NotImplementedError()

    async def start_update(self, Node node):
        raise NotImplementedError()

    async def append_samples_bunch(self, Vector bunch):
        raise NotImplementedError()

    async def finish_update(self):
        raise NotImplementedError()

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef Vector draw_path(self, Vector last, Vector points, Vector color, Matrix33 transform):
        cdef int i, j, k, n, m
        cdef double a, s, dv, dx, dy, l, x, y, last_x, last_y
        cdef Vector accels, speeds, lengths, samples, normals, coords, bunch
        cdef double r, g, b
        r, g, b = color.numbers[0], color.numbers[1], color.numbers[2]
        a = self._accelleration / self._sample_rate
        coords = last.concat(points)
        n = coords.length // 2
        lengths = Vector.__new__(Vector)
        lengths.allocate_numbers(n)
        normals = Vector.__new__(Vector)
        normals.allocate_numbers(n*2)
        accels = Vector.__new__(Vector)
        accels.allocate_numbers(n)
        i = 0
        last_x, last_y = coords.numbers[0], coords.numbers[1]
        for j in range(1, n):
            x, y = coords.numbers[j*2], coords.numbers[j*2+1]
            dx, dy = x-last_x, y-last_y
            l = sqrt(dx*dx + dy*dy)
            if l < self._epsilon:
                n -= 1
                continue
            lengths.numbers[i] = l
            normals.numbers[i*2] = dx / l
            normals.numbers[i*2+1] = dy / l
            accels.numbers[i] = a / (max(abs(dx), abs(dy)) / l)
            i += 1
            if i != j:
                coords.numbers[i*2] = x
                coords.numbers[i*2+1] = y
            last_x, last_y = x, y
        if n == 1:
            return
        speeds = Vector.__new__(Vector)
        speeds.allocate_numbers(n)
        s = 0
        speeds.numbers[0] = s
        for i in range(1, n-1):
            s = max_speed(lengths.numbers[i-1], accels.numbers[i-1], s)
            dv = max(abs(normals.numbers[i*2] - normals.numbers[(i-1)*2]), abs(normals.numbers[i*2+1] - normals.numbers[(i-1)*+1]))
            if dv:
                s = min(s, a / 2 / dv)
            speeds.numbers[i] = s
        for i in range(n-2, 0, -1):
            speeds.numbers[i] = min(speeds.numbers[i], max_speed(lengths.numbers[i], accels.numbers[i], speeds.numbers[i+1]))
        cdef Vector distances
        for i in range(n-1):
            distances = travel(lengths.numbers[i], accels.numbers[i], speeds.numbers[i], speeds.numbers[i+1])
            m = distances.length
            bunch = Vector.__new__(Vector)
            bunch.allocate_numbers(m*5)
            for j in range(m):
                x = normals.numbers[i*2] * distances.numbers[j] + coords.numbers[i*2]
                y = normals.numbers[i*2+1] * distances.numbers[j] + coords.numbers[i*2+1]
                k = j * 5
                bunch.numbers[k] = min(max(0, x), 1)
                bunch.numbers[k+1] = min(max(0, y), 1)
                if i and 0 <= x <= 1 and 0 <= y <= 1:
                    bunch.numbers[k+2] = r
                    bunch.numbers[k+3] = g
                    bunch.numbers[k+4] = b
            return bunch


cdef class DummyLaserDriver(LaserDriver):
    async def start(self):
        logger.debug("Starting dummy laser driver")

    async def stop(self):
        logger.debug("Stopping dummy laser driver")

    async def start_update(self, Node node):
        pass

    async def append_samples_bunch(self, Vector bunch):
        logger.trace("Samples bunch: {}", repr(bunch))

    async def finish_update(self):
        pass


cdef class LaserCubeDriver(LaserDriver):
    DEFAULT_SAMPLE_RATE = 50000  # samples/second
    DEFAULT_ACCELLERATION = 20   # sweeps/second^2
    DEFAULT_DAC_LAG = 2e-4       # seconds

    VENDOR_ID = 0x1fc9
    PRODUCT_ID = 0x04d8
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
    cdef object _samples_queue
    cdef object _run_task
    cdef double _dac_lag

    def __init__(self, id):
        super().__init__()
        self._device = None
        self._dac_lag = self.DEFAULT_DAC_LAG
        self._samples_queue = None

    @property
    def sample_rate(self):
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        self._sample_rate = min(max(0, value), self._max_dac_rate)

    async def start(self):
        self._run_task = asyncio.create_task(self._run())

    async def stop(self):
        self._run_task.cancel()
        await self._run_task
        self._run_task = None

    async def start_update(self, Node node):
        if self._run_task.done():
            await self._run_task
        self._dac_lag = node.get('dac_lag', 1, float, self.DEFAULT_DAC_LAG)

    async def append_samples_bunch(self, Vector bunch):
        if self._samples_queue is not None:
            await self._samples_queue.put(bunch)
        else:
            logger.trace("LaserCube driver not ready, dumped {} sample bunch", bunch.length // 5)

    async def finish_update(self):
        pass

    async def _run(self):
        cdef int i, j, n, lag
        cdef Vector samples, bunch
        cdef unsigned short[:, :] output
        cdef int sample_rate, current_sample_rate
        cdef int dac_min, dac_range, max_dac_rate
        cdef bint enabled = False
        while True:
            device = usb.core.find(idVendor=self.VENDOR_ID, idProduct=self.PRODUCT_ID)
            if device is None:
                await asyncio.sleep(1)
                continue
            try:
                self._device = device
                self._device.set_configuration()
                self._device.set_interface_altsetting(1, 1)
                configuration = self._device.get_active_configuration()
                self._control_out, self._control_in = configuration[0, 0].endpoints()
                self._data_out, = configuration[1, 1].endpoints()
                self._execute(self.SET_ENABLED, None, 'B', 0)
                enabled = False
                dac_min = self._execute(self.GET_DAC_MINIMUM_VALUE, 'I')
                dac_range = self._execute(self.GET_DAC_MAXIMUM_VALUE, 'I') - self._dac_min
                max_dac_rate = self._execute(self.GET_MAXIMUM_DAC_RATE, 'I')
                major, minor = self._execute(self.GET_MAJOR_VERSION, 'I'), self._execute(self.GET_MINOR_VERSION, 'I')
                current_sample_rate = min(max(0, self.DEFAULT_SAMPLE_RATE), max_dac_rate)
                self._execute(self.SET_DAC_RATE, None, 'I', current_sample_rate)
                enabled = False
                logger.info("Configured USB LaserCube '{} {}' (firmware {}.{})", self._device.manufacturer, self._device.product, major, minor)
                samples = None
                self._samples_queue = asyncio.Queue(maxsize=10)
                while True:
                    try:
                        bunch = await asyncio.wait_for(self._samples_queue.get(), timeout=1)
                    except asyncio.TimeoutError:
                        if enabled:
                            self._execute(self.SET_ENABLED, None, 'B', 0)
                            enabled = False
                        continue
                    if samples is None:
                        samples = bunch
                    else:
                        samples = samples.concat(bunch)
                    sample_rate = min(max(0, self._sample_rate), max_dac_rate)
                    if sample_rate != current_sample_rate:
                        self._execute(self.SET_DAC_RATE, None, 'I', sample_rate)
                        current_sample_rate = sample_rate
                    if not enabled:
                        self._execute(self.SET_ENABLED, None, 'B', 1)
                        enabled = True
                    lag = max(0, <int>(round(self._sample_rate * self._dac_lag)))
                    if len(samples) <= lag:
                        continue
                    n = len(samples) - lag
                    output_array = np.zeros((n, 4), dtype='uint16')
                    output = output_array
                    lag *= 5
                    for i in range(n):
                        j = i * 5
                        output[i, 0] = <int>(round(samples.numbers[j+2] * 255)) + (<int>(round(samples.numbers[j+3] * 255)) << 8)
                        output[i, 1] = <int>(round(samples.numbers[j+4] * 255))
                        output[i, 2] = <int>(round((1 - samples.numbers[j+lag]) * self._dac_range)) + self._dac_min
                        output[i, 3] = <int>(round((1 - samples.numbers[j+1+lag]) * self._dac_range)) + self._dac_min
                    logger.trace("Writing {} sample frame", len(output))
                    await asyncio.to_thread(self._data_out.write, output_array.tobytes())
                    if lag:
                        samples = samples.items(samples.length-lag, samples.length)
                    else:
                        samples = None
            except asyncio.CancelledError:
                if enabled:
                    self._execute(self.SET_ENABLED, None, 'B', 0)
                self._device = self._control_in = self._control_out = self._data_out = None
                self._samples_queue = None
            except Exception as exc:
                logger.error("Unexpected exception in LaserCube run task: {}", exc)
                raise

    def _execute(self, int command, str results_format, str values_format='', *values):
        self._control_out.write(struct.pack('<B' + values_format, command, *values))
        data = self._control_in.read(64)
        assert data[0] == command
        if not results_format:
            return None
        _, _, *results = struct.unpack_from("<BB" + results_format, data)
        return results if len(results) > 1 else results[0]


cdef class Laser:
    cdef LaserDriver driver
    cdef Vector last_point

    def __init__(self, **kwargs):
        self.driver = None

    def purge(self):
        pass

    def destroy(self):
        if self.driver is not None:
            self.driver.close()
            self.driver = None

    async def update(self, engine, Node node, **kwargs):
        driver = node.get('driver', 1, str, '').lower()
        cls = {'lasercube': LaserCubeDriver, 'dummy': DummyLaserDriver}.get(driver)
        if cls is not None:
            id = node.get('id', 1, str)
            if not isinstance(self.driver, cls):
                if self.driver is not None:
                    await self.driver.stop()
                self.driver = cls(id)
                await self.driver.start()
                self.last_point = Half2
            sample_rate = node.get('sample_rate', 1, int, cls.DEFAULT_SAMPLE_RATE)
            accelleration = node.get('accelleration', 1, float, cls.DEFAULT_ACCELLERATION)
            epsilon = node.get('epsilon', 1, float, cls.DEFAULT_EPSILON)
            if sample_rate != self.driver.sample_rate:
                self.driver.sample_rate = sample_rate
            if accelleration != self.driver.accelleration:
                self.driver.accelleration = accelleration
            if epsilon != self.driver.epsilon:
                self.driver.epsilon = epsilon
            transform = Matrix33()
            paths = []
            self.collect_paths(node, paths, Zero3, transform)
            if paths:
                await self.driver.start_update(node)
                await self.draw_paths(paths)
                await self.driver.finish_update()
        elif self.driver is not None:
            await self.driver.stop()
            self.driver = None

    async def draw_paths(self, list paths):
        cdef int i, nearest_i
        cdef double d, nearest_d=0
        cdef Vector points, color, first
        cdef Matrix33 transform
        cdef Vector last = self.last_point
        while paths:
            for i in range(len(paths)):
                points, color, transform = <tuple>paths[i]
                first = transform.vmul(points.items(0, 2))
                d = first.sub(last).squared_sum()
                if i == 0 or d < nearest_d:
                    nearest_d = d
                    nearest_i = i
            points, color, transform = <tuple>paths.pop(nearest_i)
            bunch = self.driver.draw_path(last, points, color, transform)
            await self.driver.append_samples_bunch(bunch)
            last = transform.vmul(points.items(points.length-2, points.length))
        self.last_point = last

    @cython.cdivision(True)
    cpdef void collect_paths(self, Node node, list paths, Vector color, Matrix33 transform):
        cdef double x, y, sx, sy, th
        cdef int i, j, m, n
        cdef Vector size, point, points, path
        cdef Matrix33 matrix
        cdef bint close

        color = node.get_fvec('color', 3, color)

        if node.kind in ('laser', 'group'):
            for key in node.keys():
                if key == 'translate':
                    point = node.get_fvec('translate', 2, null_)
                    if (matrix := Matrix33._translate(point)) != None:
                        transform = transform.mmul(matrix)
                elif key == 'scale':
                    point = node.get_fvec('scale', 2, null_)
                    if (matrix := Matrix33._scale(point)) != None:
                        transform = transform.mmul(matrix)
                elif key == 'rotate':
                    th = node.get_float('rotate', 0)
                    if (matrix := Matrix33._rotate(th)) != None:
                        transform = transform.mmul(matrix)
            for child in node.children:
                self.collect_paths(child, paths, color, transform)

        elif node.kind == 'line':
            points = node.get_fvec('points', 0, null_)
            if points.length >= 2:
                m = n = points.length // 2 * 2
                if node.get_bool('close', False):
                    n += 2
                path = Vector.__new__(Vector)
                path.allocate_numbers(n)
                for i in range(0, n):
                    path.numbers[i] = points.numbers[i%m]
                paths.append((path, color, transform))

        elif node.kind == 'rect':
            size = node.get_fvec('size', 2, Zero2)
            sx, sy = size.numbers[0], size.numbers[1]
            if sx and sy:
                point = node.get_fvec('point', 2, Zero2)
                x, y = point.numbers[0], point.numbers[1]
                path = Vector.__new__(Vector)
                path.allocate_numbers(10)
                path.numbers[0], path.numbers[1] = x, y
                path.numbers[2], path.numbers[3] = x+sx, y
                path.numbers[4], path.numbers[5] = x+sx, y+sy
                path.numbers[6], path.numbers[7] = x, y+sy
                path.numbers[8], path.numbers[9] = x, y
                paths.append((path, color, transform))

        elif node.kind == 'ellipse':
            size = node.get_fvec('radius', 2, Zero2)
            sx, sy = size.numbers[0], size.numbers[1]
            if sx and sy:
                point = node.get_fvec('point', 2, Zero2)
                x, y = point.numbers[0], point.numbers[1]
                n = node.get_int('segments', 60)
                if n >= 3:
                    path = Vector.__new__(Vector)
                    path.allocate_numbers((n+1)*2)
                    for i in range(n+1):
                        th = Tau * i/n
                        j = i * 2
                        path.numbers[j] = x + sx*cos(th)
                        path.numbers[j+1] = x + sx*sin(th)
                    paths.append((path, color, transform))


RENDERER_CLASS = Laser
