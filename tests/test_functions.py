"""
Tests of the flitter language built-in functions
"""

import unittest

from flitter.model import Vector, null
from flitter.language.functions import Uniform, Normal, Beta


class TestUniform(unittest.TestCase):
    FACTORY = Uniform
    DISTRIBUTION = ('uniform',)
    LOWER = 0
    UPPER = 1

    def test_creation(self):
        self.assertIsInstance(self.FACTORY(), self.FACTORY)
        self.assertEqual(hash(self.FACTORY()), hash(Vector()))
        self.assertEqual(hash(self.FACTORY(1.0)), hash(Normal(1.9)))

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
        for i in range(10):
            with self.subTest(i=i):
                result = kstest(self.FACTORY(i)[:1_000_000], *self.DISTRIBUTION)
                self.assertGreater(result.pvalue, 0.05)

    def test_range(self):
        source = self.FACTORY()
        for i in range(1000):
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
    FACTORY = Beta
    DISTRIBUTION = ('beta', (2, 2))


class TestNormal(TestUniform):
    FACTORY = Normal
    DISTRIBUTION = ('norm',)
    LOWER = -25
    UPPER = 25

    def test_apparent_entropy(self):
        pass


#     'len': Vector(length),
#     'sin': Vector(sinv),
#     'cos': Vector(cosv),
#     'asin': Vector(asinv),
#     'acos': Vector(acosv),
#     'polar': Vector(polar),
#     'abs': Vector(absv),
#     'exp': Vector(expv),
#     'sqrt': Vector(sqrtv),
#     'sine': Vector(sine),
#     'bounce': Vector(bounce),
#     'sharkfin': Vector(sharkfin),
#     'impulse': Vector(impulse),
#     'sawtooth': Vector(sawtooth),
#     'triangle': Vector(triangle),
#     'square': Vector(square),
#     'linear': Vector(linear),
#     'quad': Vector(quad),
#     'snap': Vector(snap),
#     'shuffle': Vector(shuffle),
#     'round': Vector(roundv),
#     'ceil': Vector(ceilv),
#     'floor': Vector(floorv),
#     'sum': Vector(sumv),
#     'accumulate': Vector(accumulate),
#     'min': Vector(minv),
#     'max': Vector(maxv),
#     'hypot': Vector(hypot),
#     'angle': Vector(angle),
#     'normalize': Vector(normalize),
#     'map': Vector(mapv),
#     'zip': Vector(zipv),
#     'hsl': Vector(hsl),
#     'hsv': Vector(hsv),