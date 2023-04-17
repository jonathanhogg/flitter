"""
Tests of the model.Vector class
"""

import math
import unittest

from flitter.model import Vector, true, false, null, Node


def test_func():
    pass


class test_class:
    def __len__(self):
        return 0


def all_isclose(xs, ys):
    for x, y in zip(xs, ys):
        if not math.isclose(x, y):
            return False
    return True


class TestVector(unittest.TestCase):
    """
    Tests of the Vector class
    """

    def test_construct(self):
        for constructor in (Vector, Vector.coerce):
            for value in [None, [], (), {}, set()]:
                with self.subTest(value=value, constructor=constructor):
                    vector = constructor(value)
                    self.assertIsInstance(vector, Vector)
                    self.assertEqual(len(vector), 0)
                    self.assertFalse(vector.numeric)
                    self.assertFalse(vector.non_numeric)
            for value in [3, 3.4, "Three", True, False, -1, test_func, test_class]:
                for container in [lambda xs: xs[0], list, tuple, set, Vector, lambda xs: {x: None for x in xs}]:
                    with self.subTest(value=value, constructor=constructor, container=container):
                        values = container([value])
                        vector = constructor(values)
                        self.assertIsInstance(vector, Vector)
                        self.assertEqual(len(vector), 1)
                        self.assertEqual(vector[0], value)
                        self.assertTrue(vector.numeric == isinstance(value, (float, int)))
                        self.assertTrue(vector.non_numeric == (not vector.numeric))
            for value in [Node('foo'), Node('bar', {'baz'}, {'color': Vector(1)})]:
                for container in [lambda xs: xs[0], list, tuple, Vector]:
                    with self.subTest(value=value, constructor=constructor, container=container):
                        values = container([value])
                        vector = constructor(values)
                        self.assertIsInstance(vector, Vector)
                        self.assertEqual(len(vector), 1)
                        self.assertIs(vector[0], value)
                        self.assertFalse(vector.numeric)
                        self.assertTrue(vector.non_numeric)
            for values in [[0, 0.1, 1.5, -1e99, math.inf], {"Hello", "world!"}, [99, "red", "balloons"],
                           (test_func, test_class), {3: "three"}, {"three": 3},
                           range(10), range(-10), range(0, 10), range(0, 10, 1), range(9, -1, -1),
                           [Node('foo'), Node('bar', {'baz'}, {'color': Vector(1)})]]:
                with self.subTest(values=values, constructor=constructor):
                    vector = constructor(values)
                    self.assertIsInstance(vector, Vector)
                    self.assertEqual(len(vector), len(values))
                    all_numbers = True
                    for x, y in zip(vector, values):
                        self.assertEqual(x, y)
                        all_numbers = all_numbers and isinstance(y, (float, int))
                    self.assertEqual(vector.numeric, len(vector) > 0 and all_numbers)
                    self.assertEqual(vector.non_numeric, len(vector) > 0 and not all_numbers)
                    vector2 = Vector.coerce(vector)
                    self.assertIs(vector2, vector)

    def test_special_coercion_values(self):
        self.assertIs(Vector.coerce(None), null)
        self.assertIs(Vector.coerce([]), null)
        self.assertIs(Vector.coerce(()), null)
        self.assertIs(Vector.coerce({}), null)
        self.assertIs(Vector.coerce(set()), null)
        self.assertIs(Vector.coerce(True), true)
        self.assertIs(Vector.coerce(False), false)
        self.assertIs(Vector.coerce(1), true)
        self.assertIs(Vector.coerce(0), false)
        self.assertIs(Vector.coerce(-1), Vector.coerce(-1))

    def test_range_slice(self):
        TESTS = [
            ((None,), []),
            ((10,), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            ((9.9,), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            ((10.3,), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            ((0.5, 10.5), [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]),
            ((10.5, 0.5), []),
            ((0.5, 10.5, 1.5), [0.5, 2, 3.5, 5, 6.5, 8, 9.5]),
            ((0.5, 10.5, -1.5), []),
            ((10.5, 0.5, -1.5), [10.5, 9, 7.5, 6, 4.5, 3, 1.5]),
            ((10.5, 0.5, -10.5), [10.5]),
        ]
        for s, values in TESTS:
            with self.subTest(slice=s):
                vector = Vector(slice(*s))
                self.assertEqual(list(vector), values)
                self.assertTrue(len(values) == 0 or vector.numeric)
                vector = Vector.range(*s)
                self.assertEqual(list(vector), values)
                self.assertTrue(len(values) == 0 or vector.numeric)

    def test_isinstance(self):
        TESTS = [
            (None, []),
            (float, [0, 0.1, 1.5, -1e99, math.nan, math.inf]),
            (str, ["Hello", "world!"]),
            (Node, [Node('foo'), Node('bar', {'baz'}, {'color': Vector(1)})]),
            (None, [99, "red", "balloons"]),
        ]
        for t, values in TESTS:
            with self.subTest(type=t, values=values):
                vector = Vector(values)
                if t is not None:
                    self.assertTrue(vector.isinstance(t))
                for t2, _ in TESTS:
                    if t2 is not None and t2 is not t:
                        self.assertFalse(vector.isinstance(t2))

    def test_as_bool(self):
        self.assertFalse(null)
        self.assertFalse(false)
        self.assertFalse(Vector())
        self.assertFalse(Vector(0))
        self.assertFalse(Vector([0, 0, 0]))
        self.assertFalse(Vector([""]))
        self.assertFalse(Vector(["", "", ""]))
        self.assertFalse(Vector([0, "", 0]))
        self.assertTrue(true)
        self.assertTrue(Vector(1))
        self.assertTrue(Vector([0, 0, 1]))
        self.assertTrue(Vector(["foo"]))
        self.assertTrue(Vector(["", "bar"]))
        self.assertTrue(Vector(Node("foo")))
        self.assertTrue(Vector(test_func))
        self.assertTrue(Vector(test_class))

    def test_as_double(self):
        for value in [0, 1, 0.1, 1.5, -1e99, math.inf, True, False]:
            self.assertEqual(float(Vector(value)), value)
        self.assertTrue(math.isnan(float(Vector(math.nan))))
        self.assertTrue(math.isnan(float(null)))
        self.assertTrue(math.isnan(float(Vector([1, 2, 3]))))
        self.assertTrue(math.isnan(float(Vector("Hello world!"))))

    def test_as_string(self):
        TESTS = [
            (null, ""),
            (Vector([1, -2, 3.0, 1e99, 1.567e-9]), "1-231e+991.567e-09"),
            (Vector([1 / 3]), "0.333333333"),
            (Vector("Hello world!"), "Hello world!"),
            (Vector(["Hello ", "world!"]), "Hello world!"),
            (Vector(["testing", "testing", 1, 2.2, 3.0]), "testingtesting12.23"),
        ]
        for vector, string in TESTS:
            self.assertEqual(str(vector), string)

    def test_iter(self):
        self.assertRaises(StopIteration, next, iter(null))
        i = iter(Vector([1, 2, 3]))
        self.assertEqual(next(i), 1)
        self.assertEqual(next(i), 2)
        self.assertEqual(next(i), 3)
        self.assertRaises(StopIteration, next, i)
        i = iter(Vector(["Hello", "world!"]))
        self.assertEqual(next(i), "Hello")
        self.assertEqual(next(i), "world!")
        self.assertRaises(StopIteration, next, i)

    def test_hash(self):
        self.assertEqual(hash(null), 0xe220a8397b1dcdaf - (1 << 64))
        self.assertEqual(hash(Vector()), 0xe220a8397b1dcdaf - (1 << 64))
        self.assertEqual(hash(Vector(0)), -6411193824288604561)
        self.assertEqual(hash(Vector(0.1)), -7435092146473341298)
        self.assertEqual(hash(true), 258181728628715636)
        self.assertEqual(hash(Vector(1.0)), 258181728628715636)
        self.assertEqual(hash(Vector("Hello world!")), 2211555187250521325)
        self.assertEqual(hash(Vector(["Hello ", "world!"])), -463109597642348796)
        self.assertRaises(TypeError, hash, Vector(Node('foo')))
        self.assertRaises(TypeError, hash, Vector(test_func))
        self.assertRaises(TypeError, hash, Vector(test_class))

    def test_match(self):
        self.assertEqual(null.match(1, float), None)
        self.assertEqual(null.match(1, int), None)
        self.assertEqual(null.match(1, str), None)
        self.assertEqual(null.match(1, float, 3.5), 3.5)
        self.assertEqual(null.match(3, float, [1, 2, 3.5]), [1, 2, 3.5])
        self.assertEqual(Vector(3.5).match(0, float), [3.5])
        self.assertEqual(Vector(3.5).match(1, float), 3.5)
        self.assertEqual(Vector(3.5).match(1, int), 3)
        self.assertEqual(Vector(3.5).match(1, str), None)
        self.assertEqual(Vector(3.5).match(3, float), [3.5, 3.5, 3.5])
        self.assertEqual(Vector("Hello world!").match(1, str), "Hello world!")
        self.assertEqual(Vector("Hello world!").match(2, str), ["Hello world!", "Hello world!"])
        self.assertEqual(Vector(["Hello ", "world!"]).match(2, str), ["Hello ", "world!"])
        self.assertEqual(Vector(["Hello ", "world!"]).match(1, str), None)
        self.assertEqual(true.match(1, float), 1.0)
        self.assertIs(true.match(1, bool), True)
        self.assertEqual(true.match(2, bool), [True, True])
        self.assertEqual(Vector([0, 1]).match(2, bool), [False, True])

    def test_copynodes(self):
        self.assertIs(null.copynodes(), null)
        vector = Vector([1, 2, 3])
        self.assertIs(vector.copynodes(), vector)
        vector = Vector([1, 2, "Hello"])
        self.assertIs(vector.copynodes(), vector)
        color = Vector([1, 0, 1])
        vector = Vector(Node('foo', {'bar'}, {'color': color}))
        copy = vector.copynodes()
        self.assertIsNot(copy, vector)
        self.assertIsNot(copy[0], vector[0])
        self.assertEqual(copy[0].kind, 'foo')
        self.assertEqual(copy[0].tags, {'bar'})
        self.assertIs(copy[0]['color'], color)

    def test_repr(self):
        self.assertEqual(repr(null), "null")
        self.assertEqual(repr(true), "1")
        self.assertEqual(repr(false), "0")
        self.assertEqual(repr(Vector(1 / 3)), "0.333333333")
        self.assertEqual(repr(Vector([1, 2, 3])), "1;2;3")
        self.assertEqual(repr(Vector([1, 2.5, 3])), "1;2.5;3")
        self.assertEqual(repr(Vector("Hello world!")), "'Hello world!'")
        self.assertEqual(repr(Vector("one_two_3")), ":one_two_3")
        self.assertEqual(repr(Vector("1_two_three")), "'1_two_three'")
        self.assertEqual(repr(Vector([99, "red", "balloons"])), "99;:red;:balloons")
        node = Node('foo', {'bar'}, {'color': Vector(1)})
        self.assertEqual(repr(Vector(node)), f"({node!r})")

    def test_neg(self):
        self.assertEqual(-null, null)
        self.assertEqual(-Vector("Hello world!"), null)
        self.assertEqual(-Vector(-3), Vector(3))
        self.assertEqual(-Vector([0, 0.1, 3, -99, 1e99, math.inf]), Vector([0, -0.1, -3, 99, -1e99, -math.inf]))

    def test_pos(self):
        self.assertEqual(+null, null)
        self.assertEqual(+Vector("Hello world!"), null)
        self.assertEqual(+Vector(-3), Vector(-3))
        self.assertEqual(+Vector([0, 0.1, 3, -99, 1e99, math.inf]), Vector([0, 0.1, 3, -99, 1e99, math.inf]))

    def test_abs(self):
        self.assertEqual(abs(null), null)
        self.assertEqual(abs(Vector("Hello world!")), null)
        self.assertEqual(abs(Vector(-3)), Vector(3))
        self.assertEqual(abs(Vector([0, 0.1, 3, -99, 1e99, math.inf])), Vector([0, 0.1, 3, 99, 1e99, math.inf]))

    def test_add(self):
        x = Vector([1, 0.1, -5, 1e6, math.inf])
        self.assertEqual(x + null, null)
        self.assertEqual(null + x, null)
        self.assertEqual(x + Vector("Hello world!"), null)
        self.assertEqual(Vector("Hello world!") + x, null)
        self.assertEqual(x + Vector(1), Vector([2, 1.1, -4, 1000001, math.inf]))
        self.assertEqual(Vector(1) + x, Vector([2, 1.1, -4, 1000001, math.inf]))
        self.assertEqual(x + Vector([1, 2]), Vector([2, 2.1, -4, 1000002, math.inf]))
        self.assertEqual(Vector([1, 2]) + x, Vector([2, 2.1, -4, 1000002, math.inf]))
        self.assertEqual(x + x, Vector([2, 0.2, -10, 2e6, math.inf]))

    def test_sub(self):
        x = Vector([1, 0.1, -5, 1e6, math.inf])
        self.assertEqual(x - null, null)
        self.assertEqual(null - x, null)
        self.assertEqual(x - Vector("Hello world!"), null)
        self.assertEqual(Vector("Hello world!") - x, null)
        self.assertEqual(x - Vector(1), Vector([0, -0.9, -6, 999999, math.inf]))
        self.assertEqual(Vector(1) - x, Vector([0, 0.9, 6, -999999, -math.inf]))
        self.assertEqual(x - Vector([1, 2]), Vector([0, -1.9, -6, 999998, math.inf]))
        self.assertEqual(Vector([1, 2]) - x, Vector([0, 1.9, 6, -999998, -math.inf]))
        y = x - x
        self.assertEqual(y[:4], Vector([0, 0, 0, 0]))
        self.assertTrue(math.isnan(y[4]))

    def test_mul(self):
        x = Vector([1, 0.1, -5, 1e6, math.inf])
        self.assertEqual(x * null, null)
        self.assertEqual(null * x, null)
        self.assertEqual(x * Vector("Hello world!"), null)
        self.assertEqual(Vector("Hello world!") * x, null)
        self.assertEqual(x * Vector(1), x)
        self.assertEqual(Vector(1) * x, x)
        self.assertEqual(x * Vector([1, 2]), Vector([1, 0.2, -5, 2e6, math.inf]))
        self.assertEqual(Vector([1, 2]) * x, Vector([1, 0.2, -5, 2e6, math.inf]))
        self.assertTrue(all_isclose(x * x, Vector([1, 0.01, 25, 1e12, math.inf])))

    def test_truediv(self):
        x = Vector([1, 0.1, -5, 1e6, math.inf])
        self.assertEqual(x / null, null)
        self.assertEqual(null / x, null)
        self.assertEqual(x / Vector("Hello world!"), null)
        self.assertEqual(Vector("Hello world!") / x, null)
        self.assertEqual(x / Vector(1), x)
        self.assertEqual(Vector(1) / x, Vector([1, 10, -0.2, 1e-6, 0]))
        self.assertEqual(x / Vector([1, 2]), Vector([1, 0.05, -5, 5e5, math.inf]))
        self.assertEqual(Vector([1, 2]) / x, Vector([1, 20, -0.2, 2e-6, 0]))
        y = x / x
        self.assertEqual(y[:4], Vector([1, 1, 1, 1]))
        self.assertTrue(math.isnan(y[4]))

    def test_floordiv(self):
        x = Vector([1, 0.1, -5.3, 1e6, math.inf])
        self.assertEqual(x // null, null)
        self.assertEqual(null // x, null)
        self.assertEqual(x // Vector("Hello world!"), null)
        self.assertEqual(Vector("Hello world!") // x, null)
        self.assertEqual(x // Vector(1), Vector([1, 0, -6, 1e6, math.inf]))
        self.assertEqual(Vector(1) // x, Vector([1, 10, -1, 0, 0]))
        self.assertEqual(x // Vector([1, 2]), Vector([1, 0, -6, 5e5, math.inf]))
        self.assertEqual(Vector([1, 2]) // x, Vector([1, 20, -1, 0, 0]))
        y = x // x
        self.assertEqual(y[:4], Vector([1, 1, 1, 1]))
        self.assertTrue(math.isnan(y[4]))

    def test_mod(self):
        x = Vector([1, 0.1, -5, 1e6])
        self.assertEqual(x % null, null)
        self.assertEqual(null % x, null)
        self.assertEqual(x % Vector("Hello world!"), null)
        self.assertEqual(Vector("Hello world!") % x, null)
        self.assertEqual(x % Vector(2), Vector([1, 0.1, 1, 0]))
        self.assertEqual(Vector(2) % x, Vector([0, 0, -3, 2]))
        self.assertEqual(x % Vector([2, 3]), Vector([1, 0.1, 1, 1]))
        self.assertEqual(Vector([2, 3]) % x, Vector([0, 0, -3, 3]))
        self.assertEqual(x % x, Vector([0, 0, 0, 0]))

    def test_pow(self):
        x = Vector([1, 0.1, -5, 1e6, math.inf])
        self.assertEqual(x ** null, null)
        self.assertEqual(null ** x, null)
        self.assertEqual(x ** Vector("Hello world!"), null)
        self.assertEqual(Vector("Hello world!") ** x, null)
        self.assertTrue(all_isclose(x ** Vector(2), Vector([1, 0.01, 25, 1e12, math.inf])))
        self.assertTrue(all_isclose(Vector(2) ** x, Vector([2, 1.0717734625362931, 0.03125, math.inf, math.inf])))
        self.assertTrue(all_isclose(x ** Vector([1, 2]), Vector([1, 0.01, -5, 1e12, math.inf])))
        self.assertTrue(all_isclose(Vector([1, 2]) ** x, Vector([1, 1.0717734625362931, 1, math.inf, 1])))
        self.assertTrue(all_isclose(x * x, Vector([1, 0.01, 25, 1e12, math.inf])))

    def test_eq(self):
        self.assertTrue(null == Vector())
        self.assertTrue(true == Vector(1))
        self.assertTrue(false == Vector(0))
        self.assertTrue(Vector([1, 2, 3]) == Vector([1, 2, 3]))
        self.assertTrue(Vector(["Hello ", "world!"]) == Vector(["Hello ", "world!"]))

    def test_ne(self):
        self.assertFalse(null != Vector())
        self.assertTrue(true != Vector(0))
        self.assertTrue(false != Vector(1))
        self.assertFalse(Vector([1, 2, 3]) != Vector([1, 2, 3]))
        self.assertTrue(Vector(["Hello ", "world!"]) != Vector(["Hello world!"]))

    def test_gt(self):
        self.assertTrue(Vector(1) > null)
        self.assertTrue(true > Vector(0))
        self.assertFalse(false > Vector(1))
        self.assertTrue(Vector([1, 2, 4]) > Vector([1, 2, 3]))
        self.assertTrue(Vector([1, 3]) > Vector([1, 2, 3]))
        self.assertTrue(Vector(["Hello world!"]) > Vector(["Hello ", "world!"]))
        self.assertTrue(Vector(["Z"]) > Vector(["Hello world!"]))

    def test_ge(self):
        self.assertTrue(Vector(1) >= null)
        self.assertTrue(true >= Vector(0))
        self.assertTrue(false >= Vector(0))
        self.assertFalse(Vector([1, 2, 3]) >= Vector([1, 2, 3, 4]))
        self.assertTrue(Vector([1, 2, 3]) >= Vector([1, 2, 3]))
        self.assertTrue(Vector([1, 3]) >= Vector([1, 2, 3]))
        self.assertTrue(Vector(["Hello world!"]) >= Vector(["Hello ", "world!"]))
        self.assertTrue(Vector(["Z"]) >= Vector(["Hello world!"]))

    def test_lt(self):
        self.assertTrue(null < Vector(1))
        self.assertTrue(Vector(0) < true)
        self.assertFalse(Vector(1) < false)
        self.assertTrue(Vector([1, 2, 3]) < Vector([1, 2, 4]))
        self.assertTrue(Vector([1, 2, 3]) < Vector([1, 3]))
        self.assertTrue(Vector(["Hello ", "world!"]) < Vector(["Hello world!"]))
        self.assertTrue(Vector(["Hello world!"]) < Vector(["Z"]))

    def test_le(self):
        self.assertTrue(null <= Vector(1))
        self.assertTrue(Vector(0) <= true)
        self.assertTrue(Vector(0) <= false)
        self.assertFalse(Vector([1, 2, 3, 4]) <= Vector([1, 2, 3]))
        self.assertTrue(Vector([1, 2, 3]) <= Vector([1, 2, 3]))
        self.assertTrue(Vector([1, 2, 3]) <= Vector([1, 3]))
        self.assertTrue(Vector(["Hello ", "world!"]) <= Vector(["Hello world!"]))
        self.assertTrue(Vector(["Hello world!"]) <= Vector(["Z"]))

    def test_getitem(self):
        x = Vector.range(100)
        self.assertTrue(x[:10] == Vector.range(10))
        self.assertTrue(x[10:100] == Vector.range(10, 100))
        self.assertTrue(x[10:100:5] == Vector.range(10, 100, 5))
        self.assertTrue(x[99:9:-1] == Vector.range(99, 9, -1))
        self.assertTrue(x[99:9:-2] == Vector.range(99, 9, -2))
        self.assertTrue(x[99:9:-13 / 3] == Vector([99, 94, 90, 86, 81, 77, 73, 68, 64, 60, 55, 51, 47, 42, 38, 34, 29, 25, 21, 16, 12]))

    def test_normalize(self):
        self.assertEqual(null.normalize(), null)
        self.assertEqual(Vector(["Hello world!"]).normalize(), null)
        self.assertEqual(Vector(1).normalize(), Vector(1))
        self.assertEqual(Vector([0, 1]).normalize(), Vector([0, 1]))
        self.assertTrue(all_isclose(Vector([1, 1]).normalize(), Vector([0.707106781, 0.707106781])))
        self.assertTrue(all_isclose(Vector([1, -2, 3]).normalize(), Vector([0.267261242, -0.534522484, 0.801783726])))

    def test_dot(self):
        self.assertEqual(null.dot(null), null)
        self.assertEqual(null.dot(Vector([1, 2, 3])), null)
        self.assertEqual(Vector([1, 2, 3]).dot(null), null)
        self.assertEqual(Vector([1, 2, 3]).dot(Vector([1, 2, 3])), Vector(14))
        self.assertEqual(Vector([1, 2, 3]).dot(Vector([3, 2, 1])), Vector(10))
        self.assertEqual(Vector([1, 2, 3]).dot(Vector(3)), Vector(18))
        self.assertEqual(Vector([1, 2, 3]).dot(Vector([3, 2])), Vector(16))

    def test_cross(self):
        self.assertEqual(null.cross(null), null)
        self.assertEqual(null.cross(Vector([1, 2, 3])), null)
        self.assertEqual(Vector([1, 2, 3]).cross(null), null)
        self.assertEqual(Vector([1, 2, 3]).cross(Vector([1, 2, 3])), Vector([0, 0, 0]))
        self.assertEqual(Vector([1, 2, 3]).cross(Vector([-1, -2, -3])), Vector([0, 0, 0]))
        self.assertEqual(Vector([1, 2, 3]).cross(Vector([1, 2])), null)
        self.assertEqual(Vector([1, 2, 3]).cross(Vector([3, 2, 1])), Vector([-4, 8, -4]))