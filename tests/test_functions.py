"""
Tests of the flitter language built-in functions
"""

import math
import unittest

from flitter.model import Vector, null
from flitter.language.functions import (uniform, normal, beta,
                                        length, sumv, accumulate, minv, maxv, minindex, maxindex, mapv, clamp, zipv, count,
                                        roundv, absv, expv, sqrtv, logv, log2v, log10v, ceilv, floorv, fract,
                                        cosv, acosv, sinv, asinv, tanv, hypot, normalize, polar, angle,
                                        split, ordv, chrv)
from flitter.language.noise import noise, octnoise


Tau = 2*math.pi


def all_isclose(xs, ys, rel_tol=1e-9, abs_tol=0):
    for x, y in zip(xs, ys):
        if not math.isclose(x, y, rel_tol=rel_tol, abs_tol=abs_tol):
            return False
    return True


class TestUniform(unittest.TestCase):
    FACTORY = uniform
    DISTRIBUTION = ('uniform',)
    LOWER = 0
    UPPER = 1
    P_VALUE = 0.05

    def test_creation(self):
        self.assertIsInstance(self.FACTORY(), self.FACTORY)
        self.assertEqual(hash(self.FACTORY()), hash(Vector()))
        self.assertEqual(hash(self.FACTORY(1.0)), hash(self.FACTORY(1.9)))

    def test_eq(self):
        source1 = self.FACTORY(1)
        source2 = self.FACTORY(1)
        self.assertIsNot(source1, source2)
        self.assertEqual(source1, source2)

    def test_null(self):
        self.assertEqual(self.FACTORY() + 1, null)
        self.assertEqual(1 + self.FACTORY(), null)
        self.assertEqual(self.FACTORY() * 1, null)
        self.assertEqual(self.FACTORY() / 1, null)
        self.assertEqual(self.FACTORY() % 1, null)
        self.assertEqual(self.FACTORY() // 1, null)
        self.assertEqual(self.FACTORY() ** 1, null)

    def test_indexing_and_overlap(self):
        source = self.FACTORY()
        last_xs = None
        for i in range(-100, 100):
            source_i = source[i]
            self.assertIsInstance(source_i, float)
            with self.subTest(i=i):
                xs = source[i:i+10]
                self.assertIsInstance(xs, Vector)
                self.assertEqual(len(xs), 10)
                self.assertEqual(xs[0], source_i)
                for x in xs:
                    self.assertIsInstance(x, float)
                if last_xs is not None:
                    self.assertNotEqual(last_xs, xs)
                    self.assertEqual(last_xs[1:10], xs[0:9])
                last_xs = xs

    def test_reproducability(self):
        source1 = self.FACTORY(1.0)
        source2 = self.FACTORY(1.1)
        self.assertIsNot(source1, source2)
        self.assertEqual(source1[:10_000], source2[:10_000])

    def test_distribution(self):
        from scipy.stats import kstest
        for i in range(2):
            with self.subTest(i=i):
                result = kstest(self.FACTORY(i)[:100_000], *self.DISTRIBUTION)
                self.assertGreater(result.pvalue, self.P_VALUE)

    def test_range(self):
        source = self.FACTORY()
        for i in range(10000):
            with self.subTest(i=i):
                x = source[i]
                self.assertTrue(x >= self.LOWER)
                self.assertTrue(x < self.UPPER)

    def test_apparent_entropy(self):
        from struct import pack
        from zlib import compress
        data = pack('<1000L', *(int(n * (1 << 32)) for n in self.FACTORY()[:1000]))
        compressed = compress(data, 9)
        self.assertGreater(len(compressed) / len(data), 0.99)


class TestBeta(TestUniform):
    FACTORY = beta
    DISTRIBUTION = ('beta', (2, 2))


class TestNormal(TestUniform):
    FACTORY = normal
    DISTRIBUTION = ('norm',)
    LOWER = -10
    UPPER = 10

    def test_apparent_entropy(self):
        pass


class TestNoise(unittest.TestCase):
    def test_zero_behaviour(self):
        self.assertEqual(noise(Vector.symbol('seed'), Vector(0)), Vector(0))
        self.assertEqual(noise(Vector.symbol('seed'), Vector(0), Vector(0)), Vector(0))
        n3 = abs(float(noise(Vector.symbol('seed'), Vector(0), Vector(0), Vector(0))))
        self.assertGreater(abs(n3), Vector(0))
        self.assertLess(abs(n3), Vector(1e-30))

    def test_null_behaviour(self):
        self.assertEqual(noise(null, Vector(0)), Vector(0))
        self.assertEqual(noise(Vector.symbol('seed'), null), null)
        self.assertEqual(noise(Vector.symbol('seed'), Vector(0), null), null)
        self.assertEqual(noise(Vector.symbol('seed'), Vector(0), Vector(0), null), null)
        self.assertEqual(octnoise(null, Vector(1), Vector(0.5), Vector(0)), Vector(0))
        self.assertEqual(octnoise(Vector.symbol('seed'), null, Vector(0.5), Vector(0)), null)
        self.assertEqual(octnoise(Vector.symbol('seed'), Vector(1), null, Vector(0)), null)
        self.assertEqual(octnoise(Vector.symbol('seed'), Vector(1), Vector(0.5), null), null)
        self.assertEqual(octnoise(Vector.symbol('seed'), Vector(1), Vector(0.5), Vector(0), null), null)
        self.assertEqual(octnoise(Vector.symbol('seed'), Vector(1), Vector(0.5), Vector(0), Vector(0), null), null)

    def test_noise_1(self):
        seed1 = Vector.symbol('seed1')
        seed2 = Vector.symbol('seed2')
        last_n1 = None
        for x in map(lambda x: x/49, range(1, 1001)):
            n1 = float(noise(seed1, Vector(x)))
            self.assertTrue(-1 <= n1 <= 1)
            n2 = float(noise(seed2, Vector(x)))
            self.assertTrue(-1 <= n2 <= 1)
            self.assertNotEqual(n1, n2)
            if last_n1 is not None:
                self.assertLess(abs(n1 - last_n1), 0.1)
            last_n1 = n1

    def test_noise_2(self):
        seed1 = Vector.symbol('seed1')
        seed2 = Vector.symbol('seed2')
        last_n1 = None
        y = 1/49
        for x in map(lambda x: x/49, range(1, 1001)):
            n1 = float(noise(seed1, Vector(x), Vector(y)))
            self.assertTrue(-1 <= n1 <= 1)
            n2 = float(noise(seed2, Vector(x), Vector(y)))
            self.assertTrue(-1 <= n2 <= 1)
            self.assertNotEqual(n1, n2)
            if last_n1 is not None:
                self.assertLess(abs(n1 - last_n1), 0.1)
            last_n1 = n1
        for y in map(lambda y: y/49, range(1, 1001)):
            n1 = float(noise(seed1, Vector(x), Vector(y)))
            self.assertTrue(-1 <= n1 <= 1)
            n2 = float(noise(seed2, Vector(x), Vector(y)))
            self.assertTrue(-1 <= n2 <= 1)
            self.assertNotEqual(n1, n2)
            self.assertLess(abs(n1 - last_n1), 0.1)
            last_n1 = n1

    def test_noise_3(self):
        seed1 = Vector.symbol('seed1')
        seed2 = Vector.symbol('seed2')
        last_n1 = None
        y = 1/49
        z = 1/49
        for x in map(lambda x: x/49, range(1, 1001)):
            n1 = float(noise(seed1, Vector(x), Vector(y), Vector(z)))
            self.assertTrue(-1 <= n1 <= 1)
            n2 = float(noise(seed2, Vector(x), Vector(y), Vector(z)))
            self.assertTrue(-1 <= n2 <= 1)
            self.assertNotEqual(n1, n2)
            if last_n1 is not None:
                self.assertLess(abs(n1 - last_n1), 0.1)
            last_n1 = n1
        for y in map(lambda y: y/49, range(1, 1001)):
            n1 = float(noise(seed1, Vector(x), Vector(y), Vector(z)))
            self.assertTrue(-1 <= n1 <= 1)
            n2 = float(noise(seed2, Vector(x), Vector(y), Vector(z)))
            self.assertTrue(-1 <= n2 <= 1)
            self.assertNotEqual(n1, n2)
            self.assertLess(abs(n1 - last_n1), 0.1)
            last_n1 = n1
        for z in map(lambda z: z/49, range(1, 1001)):
            n1 = float(noise(seed1, Vector(x), Vector(y), Vector(z)))
            self.assertTrue(-1 <= n1 <= 1)
            n2 = float(noise(seed2, Vector(x), Vector(y), Vector(z)))
            self.assertTrue(-1 <= n2 <= 1)
            self.assertNotEqual(n1, n2)
            self.assertLess(abs(n1 - last_n1), 0.1)
            last_n1 = n1

    def test_octnoise_3(self):
        seed1 = Vector.symbol('seed1')
        seed2 = Vector.symbol('seed2')
        last_n1 = None
        y = 1/49
        z = 1/49
        octaves = Vector(3)
        roughness = Vector(0.5)
        for x in map(lambda x: x/49, range(1, 1001)):
            n1 = float(octnoise(seed1, octaves, roughness, Vector(x), Vector(y), Vector(z)))
            self.assertTrue(-1 <= n1 <= 1)
            n2 = float(octnoise(seed2, octaves, roughness, Vector(x), Vector(y), Vector(z)))
            self.assertTrue(-1 <= n2 <= 1)
            self.assertNotEqual(n1, n2)
            if last_n1 is not None:
                self.assertLess(abs(n1 - last_n1), 0.1)
            last_n1 = n1
        for y in map(lambda y: y/49, range(1, 1001)):
            n1 = float(octnoise(seed1, octaves, roughness, Vector(x), Vector(y), Vector(z)))
            self.assertTrue(-1 <= n1 <= 1)
            n2 = float(octnoise(seed2, octaves, roughness, Vector(x), Vector(y), Vector(z)))
            self.assertTrue(-1 <= n2 <= 1)
            self.assertNotEqual(n1, n2)
            self.assertLess(abs(n1 - last_n1), 0.1)
            last_n1 = n1
        for z in map(lambda z: z/49, range(1, 1001)):
            n1 = float(octnoise(seed1, octaves, roughness, Vector(x), Vector(y), Vector(z)))
            self.assertTrue(-1 <= n1 <= 1)
            n2 = float(octnoise(seed2, octaves, roughness, Vector(x), Vector(y), Vector(z)))
            self.assertTrue(-1 <= n2 <= 1)
            self.assertNotEqual(n1, n2)
            self.assertLess(abs(n1 - last_n1), 0.1)
            last_n1 = n1

    def test_seeds(self, n=1001):
        values = sorted([float(noise(Vector(seed), Vector(50/49))) for seed in range(n)])
        self.assertLess(values[0], -0.5)
        self.assertTrue(-0.5 < values[n//4] < -0.25)
        self.assertLess(abs(values[n // 2]), 0.1)
        self.assertTrue(0.25 < values[-n//4] < 0.5)
        self.assertGreater(values[-1], 0.5)

    def test_single_range(self):
        values1 = noise(Vector.symbol('seed'), Vector.range(100))
        values2 = [float(noise(Vector.symbol('seed'), Vector(x))) for x in range(100)]
        self.assertEqual(values1, values2)

    def test_double_range(self):
        values1 = noise(Vector.symbol('seed'), Vector.range(10), Vector.range(10))
        values2 = [float(noise(Vector.symbol('seed'), Vector(x), Vector(y))) for x in range(10) for y in range(10)]
        self.assertEqual(values1, values2)

    def test_triple_range(self):
        values1 = noise(Vector.symbol('seed'), Vector.range(10), Vector.range(10), Vector.range(10))
        values2 = [float(noise(Vector.symbol('seed'), Vector(x), Vector(y), Vector(z))) for x in range(10) for y in range(10) for z in range(10)]
        self.assertEqual(values1, values2)


class TestBasicVectorFunctions(unittest.TestCase):
    def test_length(self):
        self.assertEqual(length(null), 0)
        self.assertEqual(length(Vector(1)), 1)
        self.assertEqual(length(Vector('hello')), 1)
        self.assertEqual(length(Vector(['hello', 'world'])), 2)
        self.assertEqual(length(Vector.range(1000)), 1000)

    def test_sum(self):
        xs = Vector.range(10)
        self.assertEqual(sumv(null), null)
        self.assertEqual(sumv(Vector('hello')), null)
        self.assertEqual(sumv(xs, null), null)
        self.assertEqual(sumv(xs, Vector('hello')), null)
        self.assertEqual(sumv(xs, xs), null)
        self.assertEqual(sumv(xs, Vector(0)), null)
        self.assertEqual(sumv(xs), Vector(45))
        self.assertEqual(sumv(xs, Vector(2)), Vector([20, 25]))

    def test_accumulate(self):
        xs = Vector.range(10)
        self.assertEqual(accumulate(null), null)
        self.assertEqual(accumulate(Vector('hello')), null)
        self.assertEqual(accumulate(xs, null), null)
        self.assertEqual(accumulate(xs, Vector('hello')), null)
        self.assertEqual(accumulate(xs, xs), null)
        self.assertEqual(accumulate(xs, Vector(0)), null)
        self.assertEqual(accumulate(xs), Vector([0, 1, 3, 6, 10, 15, 21, 28, 36, 45]))
        self.assertEqual(accumulate(xs, Vector(2)), Vector([0, 1, 2, 4, 6, 9, 12, 16, 20, 25]))

    def test_min(self):
        xs = Vector([10, 4, 9.5, -3, -3.01, 10.01, 3])
        self.assertEqual(minv(null), null)
        self.assertEqual(minv(xs, null), null)
        self.assertEqual(minv(xs, Vector('hello')), null)
        self.assertEqual(minv(xs), -3.01)
        self.assertEqual(minv(*[Vector(x) for x in xs]), -3.01)
        xs = Vector(['b', 'c', 'aa', 'a', 'z', 'zz', 'x', 'w'])
        self.assertEqual(minv(xs), 'a')
        self.assertEqual(minv(*[Vector(x) for x in xs]), 'a')

    def test_minindex(self):
        xs = Vector([10, 4, 9.5, -3, -3.01, 10.01, 3])
        self.assertEqual(minindex(null), null)
        self.assertEqual(minindex(xs, null), 1)
        self.assertEqual(minindex(xs, Vector('hello')), null)
        self.assertEqual(minindex(xs), 4)
        self.assertEqual(minindex(*[Vector(x) for x in xs]), 4)
        xs = Vector(['b', 'c', 'aa', 'a', 'z', 'zz', 'x', 'w'])
        self.assertEqual(minindex(xs), 3)
        self.assertEqual(minindex(*[Vector(x) for x in xs]), 3)

    def test_max(self):
        xs = Vector([10, 4, 9.5, -3, -3.01, 10.01, 3])
        self.assertEqual(maxv(null), null)
        self.assertEqual(maxv(xs, null), xs)
        self.assertEqual(maxv(xs, Vector('hello')), null)
        self.assertEqual(maxv(xs), 10.01)
        self.assertEqual(maxv(*[Vector(x) for x in xs]), 10.01)
        xs = Vector(['b', 'c', 'aa', 'a', 'z', 'zz', 'x', 'w'])
        self.assertEqual(maxv(xs), 'zz')
        self.assertEqual(maxv(*[Vector(x) for x in xs]), 'zz')

    def test_maxindex(self):
        xs = Vector([10, 4, 9.5, -3, -3.01, 10.01, 3])
        self.assertEqual(maxindex(null), null)
        self.assertEqual(maxindex(xs, null), 0)
        self.assertEqual(maxindex(xs, Vector('hello')), null)
        self.assertEqual(maxindex(xs), 5)
        self.assertEqual(maxindex(*[Vector(x) for x in xs]), 5)
        xs = Vector(['b', 'c', 'aa', 'a', 'z', 'zz', 'x', 'w'])
        self.assertEqual(maxindex(xs), 5)
        self.assertEqual(maxindex(*[Vector(x) for x in xs]), 5)

    def test_map(self):
        self.assertEqual(mapv(null, Vector(-1), Vector(1)), null)
        self.assertEqual(mapv(Vector(0.5), null, Vector(1)), null)
        self.assertEqual(mapv(Vector(0.5), Vector(-1), null), null)
        self.assertEqual(mapv(Vector(0.5), Vector(-1), Vector(1)), Vector(0))
        self.assertEqual(mapv(Vector(0.5), Vector([-1, 0]), Vector(1)), Vector([0, 0.5]))
        self.assertEqual(mapv(Vector(0.5), Vector([-1, 0]), Vector([1, 2])), Vector([0, 1]))

    def test_clamp(self):
        self.assertEqual(clamp(null, Vector(-1), Vector(1)), null)
        self.assertEqual(clamp(Vector(0.5), null, Vector(1)), null)
        self.assertEqual(clamp(Vector(0.5), Vector(-1), null), null)
        self.assertEqual(clamp(Vector(0.5), Vector(-1), Vector(1)), Vector(0.5))
        self.assertEqual(clamp(Vector(-2), Vector(-1), Vector(1)), Vector(-1))
        self.assertEqual(clamp(Vector(2), Vector(-1), Vector(1)), Vector(1))
        self.assertEqual(clamp(Vector([-2, 0.5, 2]), Vector(-1), Vector(1)), Vector([-1, 0.5, 1]))
        self.assertEqual(clamp(Vector([-2, 0.5, 2]), Vector([-1, 1]), Vector([1, 2])), Vector([-1, 1, 1]))

    def test_zip(self):
        xs = Vector.range(10)
        ys = Vector.range(0.5, 10)
        self.assertEqual(zipv(), null)
        self.assertEqual(zipv(null), null)
        self.assertEqual(zipv(xs), xs)
        self.assertEqual(zipv(xs, ys), Vector.range(0, 10, 0.5))
        self.assertEqual(zipv(*(Vector(x) for x in xs)), xs)
        self.assertEqual(zipv(Vector(['a', 'c', 'e']), Vector(['b', 'd'])), Vector(['a', 'b', 'c', 'd', 'e', 'b']))
        self.assertEqual(zipv(Vector(['a', 'c', 'e']), Vector([0, 1])), Vector(['a', 0, 'c', 1, 'e', 0]))

    def test_count(self):
        xs = Vector.range(10)
        self.assertEqual(count(null, null), null)
        self.assertEqual(count(xs, null), [0]*10)
        self.assertEqual(count(null, xs), null)
        self.assertEqual(count(Vector(0), xs), 1)
        self.assertEqual(count(Vector(10), xs), 0)
        self.assertEqual(count(xs, xs), [1]*10)
        self.assertEqual(count(xs, xs.concat(xs)), [2]*10)
        self.assertEqual(count(Vector(['a', 1.0]), xs), [0, 1])
        self.assertEqual(count(Vector(1), Vector(['a', 'b', 1, 2, 3, 1])), 2)
        self.assertEqual(count(Vector(['a', 1]), Vector(['a', 'b', 1, 2, 3, 1])), [1, 2])


class TestUnaryMathFunctions(unittest.TestCase):
    def assertMatchesUnaryFunc(self, vfunc, func, xs=None, rel_tol=1e-9, abs_tol=0):
        self.assertEqual(vfunc(null), null)
        self.assertEqual(vfunc(Vector('hello')), null)
        xs = Vector.range(-10, 10, 0.2) if xs is None else xs
        ys = [func(x) for x in xs]
        for i in range(len(ys)):
            self.assertTrue(math.isclose(vfunc(xs.item(i)), ys[i], rel_tol=rel_tol, abs_tol=abs_tol))
        self.assertTrue(all_isclose(vfunc(xs), ys))

    def test_round(self):
        self.assertMatchesUnaryFunc(roundv, round)

    def test_abs(self):
        self.assertMatchesUnaryFunc(absv, abs)

    def test_exp(self):
        self.assertMatchesUnaryFunc(expv, math.exp)

    def test_sqrt(self):
        self.assertMatchesUnaryFunc(sqrtv, math.sqrt, Vector.range(0, 10, 0.1))

    def test_log(self):
        self.assertMatchesUnaryFunc(logv, math.log, Vector.range(0.001, 10.001, 0.1))

    def test_log2(self):
        self.assertMatchesUnaryFunc(log2v, math.log2, Vector.range(0.001, 10.001, 0.1))

    def test_log10(self):
        self.assertMatchesUnaryFunc(log10v, math.log10, Vector.range(0.001, 10.001, 0.1))

    def test_ceil(self):
        self.assertMatchesUnaryFunc(ceilv, math.ceil)

    def test_floor(self):
        self.assertMatchesUnaryFunc(floorv, math.floor)

    def test_fract(self):
        self.assertMatchesUnaryFunc(fract, lambda x: x % 1)


class TestTrig(unittest.TestCase):
    def setUp(self):
        self.a = Vector([1, 2, 3, 4])
        self.b = Vector([4, 5, 6, 7])
        self.c = Vector([2])

    def test_cos(self):
        self.assertEqual(cosv(null), null)
        self.assertEqual(cosv(Vector('hello')), null)
        theta = Vector.range(0, 1, 0.01)
        values = [math.cos(th) for th in theta*Tau]
        for i in range(len(values)):
            self.assertEqual(cosv(theta.item(i)), values[i])
        self.assertEqual(cosv(theta), values)

    def test_acos(self):
        self.assertEqual(acosv(null), null)
        self.assertEqual(acosv(Vector('hello')), null)
        xs = Vector.range(-1, 1, 0.02)
        values = [math.acos(x)/Tau for x in xs]
        for i in range(len(values)):
            self.assertEqual(acosv(xs.item(i)), values[i])
        self.assertTrue(all_isclose(acosv(xs), values))

    def test_sin(self):
        self.assertEqual(sinv(null), null)
        self.assertEqual(sinv(Vector('hello')), null)
        theta = Vector.range(0, 1, 0.01)
        values = [math.sin(th) for th in theta*Tau]
        for i in range(len(values)):
            self.assertEqual(sinv(theta.item(i)), values[i])
        self.assertEqual(sinv(theta), values)

    def test_asin(self):
        self.assertEqual(asinv(null), null)
        self.assertEqual(asinv(Vector('hello')), null)
        xs = Vector.range(-1, 1, 0.02)
        values = [math.asin(x)/Tau for x in xs]
        for i in range(len(values)):
            self.assertEqual(asinv(xs.item(i)), values[i])
        self.assertTrue(all_isclose(asinv(xs), values))

    def test_tan(self):
        self.assertEqual(tanv(null), null)
        self.assertEqual(tanv(Vector('hello')), null)
        theta = Vector.range(0, 1, 0.01)
        values = [math.tan(th) for th in theta*Tau]
        for i in range(len(values)):
            self.assertEqual(tanv(theta.item(i)), values[i])
        self.assertEqual(tanv(theta), values)

    def test_hypot(self):
        self.assertEqual(hypot(), null)
        self.assertEqual(hypot(null), null)
        self.assertEqual(hypot(self.a, null), null)
        self.assertEqual(hypot(Vector([3, 4])), Vector(5))
        self.assertEqual(hypot(Vector([3, 4, 5])), Vector(math.sqrt(50)))
        self.assertTrue(all_isclose(hypot(self.a, self.b), Vector([math.sqrt(17), math.sqrt(29), math.sqrt(45), math.sqrt(65)])))
        self.assertTrue(all_isclose(hypot(self.a, self.c), Vector([math.sqrt(5), math.sqrt(8), math.sqrt(13), math.sqrt(20)])))
        self.assertTrue(all_isclose(hypot(self.a, self.b, self.c),
                                    Vector([math.sqrt(21), math.sqrt(33), math.sqrt(49), math.sqrt(69)])))

    def test_normalize(self):
        self.assertEqual(normalize(null), null)
        self.assertEqual(normalize(Vector([3, 4])), Vector([3, 4]) / Vector(5))
        self.assertEqual(normalize(Vector([3, 4, 5])), Vector([3, 4, 5]) / Vector(math.sqrt(50)))

    def test_polar(self):
        self.assertEqual(polar(null), null)
        self.assertEqual(polar(Vector('hello')), null)
        theta = Vector.range(0, 1, 0.01)
        values = [(math.cos(th), math.sin(th)) for th in theta*Tau]
        for i in range(len(values)):
            self.assertEqual(polar(theta.item(i)), values[i])
        self.assertEqual(polar(theta), Vector.compose(values))

    def test_angle(self):
        self.assertEqual(angle(null), null)
        self.assertTrue(all_isclose(angle(self.a), Vector([math.atan2(2, 1)/Tau, math.atan2(4, 3)/Tau])))
        self.assertEqual(angle(self.a, null), null)
        self.assertEqual(angle(null, self.a), null)
        self.assertTrue(all_isclose(angle(self.a, self.b),
                                    Vector([math.atan2(4, 1)/Tau, math.atan2(5, 2)/Tau, math.atan2(6, 3)/Tau, math.atan2(7, 4)/Tau])))
        self.assertTrue(all_isclose(angle(self.a, self.c),
                                    Vector([math.atan2(2, 1)/Tau, math.atan2(2, 2)/Tau, math.atan2(2, 3)/Tau, math.atan2(2, 4)/Tau])))


class TestStringFuncs(unittest.TestCase):
    def test_ord(self):
        self.assertEqual(ordv(null), null)
        self.assertEqual(ordv(Vector('A')), Vector([65]))
        self.assertEqual(ordv(Vector('AB')), Vector([65, 66]))
        self.assertEqual(ordv(Vector(['A', 'B'])), Vector([65, 66]))

    def test_chr(self):
        self.assertEqual(chrv(null), null)
        self.assertEqual(chrv(Vector('A')), null)
        self.assertEqual(chrv(Vector([65])), Vector(['A']))
        self.assertEqual(chrv(Vector([65, 66])), Vector(['AB']))

    def test_split(self):
        self.assertEqual(split(null), null)
        self.assertEqual(split(Vector(['Hello world!'])), Vector(['Hello world!']))
        self.assertEqual(split(Vector(['Hello world!\n'])), Vector(['Hello world!']))
        self.assertEqual(split(Vector(['Hello\nworld!'])), Vector(['Hello', 'world!']))
        self.assertEqual(split(Vector(['Hello\nworld!\n'])), Vector(['Hello', 'world!']))
        self.assertEqual(split(Vector(['Hello\n\nworld!\n'])), Vector(['Hello', '', 'world!']))
