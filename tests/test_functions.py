"""
Tests of the flitter language built-in functions
"""

import math
import unittest

from flitter.model import Vector, Context, StateDict, null
from flitter.language.functions import (Uniform, Normal, Beta, counter, hypot, angle, split, ordv, chrv)


Tau = 2*math.pi


def all_isclose(xs, ys, rel_tol=1e-9, abs_tol=0):
    for x, y in zip(xs, ys):
        if not math.isclose(x, y, rel_tol=rel_tol, abs_tol=abs_tol):
            return False
    return True


class TestUniform(unittest.TestCase):
    FACTORY = Uniform
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
                result = kstest(self.FACTORY(i)[:10_000_000], *self.DISTRIBUTION)
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
    FACTORY = Beta
    DISTRIBUTION = ('beta', (2, 2))


class TestNormal(TestUniform):
    FACTORY = Normal
    DISTRIBUTION = ('norm',)
    LOWER = -10
    UPPER = 10

    def test_apparent_entropy(self):
        pass


class TestCounter(unittest.TestCase):
    def setUp(self):
        self.state = StateDict()
        self.context = Context(state=self.state)
        self.counter_id = Vector('counter')

    def test_simple(self):
        clock = count = Vector(0)
        while clock < 5:
            self.assertEqual(counter(self.context, self.counter_id, clock), count)
            clock += 0.5
            count += 0.5

    def test_offset(self):
        clock = Vector(10)
        count = Vector(0)
        while clock < 5:
            self.assertEqual(counter(self.context, self.counter_id, clock), count)
            clock += 0.5
            count += 0.5

    def test_fast(self):
        clock = count = Vector(0)
        while clock < 5:
            self.assertEqual(counter(self.context, self.counter_id, clock, Vector(2)), count)
            clock += 0.5
            count += 1.0

    def test_slow(self):
        clock = count = Vector(0)
        while clock < 5:
            self.assertEqual(counter(self.context, self.counter_id, clock, Vector(0.5)), count)
            clock += 0.5
            count += 0.25

    def test_speed_up(self):
        clock = count = Vector(0)
        while clock < 5:
            self.assertEqual(counter(self.context, self.counter_id, clock, Vector(0.5)), count)
            clock += 0.5
            count += 0.25
        while clock < 10:
            self.assertEqual(counter(self.context, self.counter_id, clock, Vector(2)), count)
            clock += 0.5
            count += 1.0

    def test_slow_down(self):
        clock = count = Vector(0)
        while clock < 5:
            self.assertEqual(counter(self.context, self.counter_id, clock, Vector(2)), count)
            clock += 0.5
            count += 1.0
        while clock < 10:
            self.assertEqual(counter(self.context, self.counter_id, clock, Vector(0.5)), count)
            clock += 0.5
            count += 0.25

    def test_pause(self):
        clock = count = Vector(0)
        while clock < 5:
            self.assertEqual(counter(self.context, self.counter_id, clock), count)
            clock += 0.5
            count += 0.5
        while clock < 10:
            self.assertEqual(counter(self.context, self.counter_id, clock, Vector(0)), count)
            clock += 0.5

    def test_reverse(self):
        clock = count = Vector(0)
        while clock < 5:
            self.assertEqual(counter(self.context, self.counter_id, clock), count)
            clock += 0.5
            count += 0.5
        while clock < 10:
            self.assertEqual(counter(self.context, self.counter_id, clock, Vector(-2)), count)
            clock += 0.5
            count -= 1.0

    def test_vector_speed(self):
        clock = Vector(0)
        count = Vector([0, 0])
        while clock < 5:
            self.assertEqual(counter(self.context, self.counter_id, clock, Vector([1, 2])), count)
            clock += 0.5
            count += [0.5, 1]

    def test_vector_clock(self):
        clock = Vector([0, 0])
        count = Vector([0, 0])
        while clock < 5:
            self.assertEqual(counter(self.context, self.counter_id, clock, Vector(2)), count)
            clock += [0.5, 1]
            count += [1, 2]

    def test_vector_clock_and_speed(self):
        clock = Vector([0, 0])
        count = Vector([0, 0])
        while clock < 5:
            self.assertEqual(counter(self.context, self.counter_id, clock, Vector([1, 2])), count)
            clock += [1, 0.5]
            count += [1, 1]

    def test_state_changes(self):
        clock = Vector([0, 0])
        count = counter(self.context, self.counter_id, clock, Vector([1, 2]))
        self.assertTrue(self.counter_id in self.state.changed_keys)
        self.state.clear_changed()
        while clock < 5:
            self.assertEqual(counter(self.context, self.counter_id, clock, Vector([1, 2])), count)
            self.assertFalse(self.counter_id in self.state.changed_keys)
            clock += [1, 0.5]
            count += [1, 1]
        self.assertEqual(counter(self.context, self.counter_id, clock, Vector([1, -2])), count)
        self.assertTrue(self.counter_id in self.state.changed_keys)

    def test_read_without_update(self):
        clock = count = Vector(0)
        count = counter(self.context, self.counter_id, clock, Vector(2))
        self.assertTrue(self.counter_id in self.state.changed_keys)
        self.state.clear_changed()
        clock += 0.5
        count += 1.0
        read_count = counter(self.context, self.counter_id, clock)
        self.assertFalse(self.counter_id in self.state.changed_keys)
        self.assertEqual(count, read_count)


class TestTrig(unittest.TestCase):
    def setUp(self):
        self.a = Vector([1, 2, 3, 4])
        self.b = Vector([4, 5, 6, 7])
        self.c = Vector([2])

    def test_hypot_one_arg(self):
        self.assertEqual(hypot(), null)
        self.assertEqual(hypot(null), null)
        self.assertTrue(all_isclose(hypot(self.a), Vector(math.sqrt(30))))
        self.assertTrue(all_isclose(hypot(self.c), self.c))

    def test_hypot_multiple_args(self):
        self.assertEqual(hypot(self.a, null), null)
        self.assertTrue(all_isclose(hypot(self.a, self.b), Vector([math.sqrt(17), math.sqrt(29), math.sqrt(45), math.sqrt(65)])))
        self.assertTrue(all_isclose(hypot(self.a, self.c), Vector([math.sqrt(5), math.sqrt(8), math.sqrt(13), math.sqrt(20)])))
        self.assertTrue(all_isclose(hypot(self.a, self.b, self.c),
                                    Vector([math.sqrt(21), math.sqrt(33), math.sqrt(49), math.sqrt(69)])))

    def test_angle_one_arg(self):
        self.assertEqual(angle(null), null)
        self.assertTrue(all_isclose(angle(self.a), Vector([math.atan2(2, 1)/Tau, math.atan2(4, 3)/Tau])))

    def test_angle_multiple_args(self):
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
