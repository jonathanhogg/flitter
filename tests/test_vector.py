"""
Tests of the model.Vector class
"""

import pickle
import math

import numpy

from flitter.model import Vector, true, false, null, Node, initialize_numbers_cache, empty_numbers_cache, numbers_cache_counts

from . import utils


FOO_SYMBOL_NUMBER = float.fromhex('-0x1.dcb27518fed9dp+1023')


def test_func():
    pass


class test_class:
    def __len__(self):
        return 0


class TestVector(utils.TestCase):
    """
    Tests of the Vector class
    """

    def test_construct(self):
        """Test constructor and coerce method"""
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

    def test_construct_from_numpy_array(self):
        self.assertEqual(Vector(numpy.arange(0, 10, 0.5, dtype='float64')), Vector.range(0, 10, 0.5))
        self.assertEqual(Vector(numpy.arange(0, 10, 0.5, dtype='float32')), Vector.range(0, 10, 0.5))

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

    def test_numbers_cache(self):
        initialize_numbers_cache(100_001)
        x = Vector.range(100_001)
        self.assertEqual(numbers_cache_counts(), {})
        del x
        self.assertEqual(numbers_cache_counts(), {100016: 1})
        x = Vector.range(100_001)
        self.assertEqual(numbers_cache_counts(), {})
        del x
        self.assertEqual(numbers_cache_counts(), {100016: 1})
        empty_numbers_cache()
        self.assertEqual(numbers_cache_counts(), {})
        x = Vector.range(16)
        del x
        self.assertEqual(numbers_cache_counts(), {})
        x = Vector.range(17)
        del x
        self.assertEqual(numbers_cache_counts(), {32: 1})

    def test_symbol(self):
        foo = Vector.symbol('foo')
        foo_n = foo[0]
        self.assertIsInstance(foo_n, float)
        self.assertTrue(foo_n < float.fromhex('-1p1023'))
        self.assertEqual(math.floor(foo_n), foo_n)
        self.assertEqual(math.floor(foo), foo)
        self.assertEqual(foo, Vector.symbol('foo'))
        self.assertNotEqual(foo, Vector.symbol('bar'))
        self.assertIs(str(foo), 'foo')

    def test_with_symbols(self):
        foo_1 = Vector.with_symbols(('foo', 1))
        self.assertIsInstance(foo_1, Vector)
        self.assertEqual(len(foo_1), 2)
        self.assertTrue(foo_1.numeric)
        self.assertEqual(foo_1[0], FOO_SYMBOL_NUMBER)
        self.assertEqual(foo_1[1], 1)
        value = ('foo', 1, lambda x: x)
        foo_2 = Vector.with_symbols(value)
        self.assertIsInstance(foo_2, Vector)
        self.assertEqual(len(foo_2), 3)
        self.assertIs(foo_2[0], value[0])
        self.assertIs(foo_2[1], value[1])
        self.assertIs(foo_2[2], value[2])

    def test_copy(self):
        for x in (Vector(), Vector(["Hello ", "world!"]), Vector.range(10), Vector(5)):
            y = x.copy()
            self.assertEqual(x, y)
            self.assertFalse(x is y)

    def test_compose(self):
        self.assertEqual(Vector.compose([]), null)
        self.assertEqual(Vector.compose([null]), null)
        self.assertEqual(Vector.compose([null, null]), null)
        self.assertEqual(Vector.compose([true, null]), true)
        self.assertEqual(Vector.compose([true, true]), Vector([1, 1]))
        self.assertEqual(Vector.compose([Vector.range(i*100, (i+1)*100) for i in range(10)]), Vector.range(1000))
        self.assertEqual(Vector.compose([Vector("Hello world!"), Vector([1, 2, 3])]), Vector(["Hello world!", 1, 2, 3]))
        foo = Vector.symbol('foo')
        bar = Vector.symbol('bar')
        self.assertEqual(Vector.compose([foo, bar]), Vector([float(foo), float(bar)]))

    def test_item(self):
        x = Vector([0.5, "hello", 123])
        self.assertIsInstance(x.item(0), Vector)
        self.assertEqual(x.item(0), 0.5)
        self.assertEqual(x.item(1), "hello")
        self.assertEqual(x.item(2), 123)
        numbers = Vector.range(10)
        for i in range(-10, 20):
            self.assertEqual(numbers.item(i), i % 10)
        objects = Vector([str(i) for i in range(10)])
        for i in range(-10, 20):
            self.assertEqual(objects.item(i), str(i % 10))
        for i in range(-10, 20):
            self.assertEqual(null.item(i), null)

    def test_indexing(self):
        numbers = Vector.range(10)
        for i in range(-10, 20):
            self.assertEqual(numbers[i], i % 10)
        objects = Vector([str(i) for i in range(10)])
        for i in range(-10, 20):
            self.assertEqual(objects[i], str(i % 10))
        for i in range(-10, 20):
            self.assertEqual(null[i], null)

    def test_slicing(self):
        numbers = Vector.range(10)
        objects = Vector([str(i) for i in range(10)])
        for i in range(-20, 10):
            for j in range(i+2, 10):
                r = Vector.range(i, j)
                self.assertEqual(numbers[r], r % 10)
                self.assertEqual(objects[r], [str(int(i) % 10) for i in r])
                self.assertEqual(null[r], null)

    def test_range(self):
        self.assertEqual(Vector.range(0), null)
        self.assertEqual(Vector.range(1), [0])
        self.assertEqual(Vector.range(1, 2), [1])
        self.assertEqual(Vector.range(0, 4, 2), [0, 2])
        self.assertEqual(Vector.range(0, 4, 0), null)
        self.assertEqual(Vector.range(0, 10, 1), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(Vector.range(10, 0, -1), [10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        self.assertEqual(Vector.range('a', 10), null)
        self.assertEqual(Vector.range(0, 'b'), null)
        self.assertEqual(Vector.range(0, 10, 'c'), null)
        with self.assertRaises(TypeError):
            Vector.range()
        with self.assertRaises(TypeError):
            Vector.range(0, 1, 2, 3)

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

    def test_pickling(self):
        TESTS = [
            null,
            true,
            false,
            Vector.range(10),
            Vector.symbol('foo'),
            Vector('Hello world!'),
            Vector([5, 1.3, 'a']),
        ]
        for v in TESTS:
            self.assertEqual(pickle.loads(pickle.dumps(v)), v)

    def test_isinstance(self):
        TESTS = [
            (None, []),
            (float, [0, 0.1, 1.5, -1e99, math.nan, math.inf]),
            (str, ["Hello", "world!"]),
            (Node, [Node('foo'), Node('bar', {'baz'}, {'color': Vector(1)})]),
            (None, [99, "red", "balloons"]),
            (float, Vector.symbol('foo')),
        ]
        for t, values in TESTS:
            with self.subTest(type=t, values=values):
                vector = Vector(values)
                if t is not None:
                    self.assertTrue(vector.isinstance(t))
                for t2, _ in TESTS:
                    if t2 is not None and t2 is not t:
                        self.assertFalse(vector.isinstance(t2))

    def test_is_finite(self):
        self.assertTrue(null.is_finite())
        self.assertFalse(Vector("Hello world!").is_finite())
        self.assertTrue(Vector(1).is_finite())
        self.assertFalse(Vector(math.inf).is_finite())
        self.assertFalse(Vector(math.nan).is_finite())
        self.assertFalse((Vector(1) / Vector(0)).is_finite())
        self.assertFalse(Vector([0, -math.inf]).is_finite())
        self.assertFalse(Vector([1, math.nan]).is_finite())

    def test_as_bool(self):
        self.assertFalse(null)
        self.assertFalse(false)
        self.assertFalse(Vector())
        self.assertFalse(Vector(0))
        self.assertFalse(Vector([0, 0, 0]))
        self.assertFalse(Vector([""]))
        self.assertFalse(Vector(["", "", ""]))
        self.assertFalse(Vector([0.0, "", 0]))
        self.assertTrue(true)
        self.assertTrue(Vector(1))
        self.assertTrue(Vector([0, 0, 1]))
        self.assertTrue(Vector(["foo"]))
        self.assertTrue(Vector(["", "bar"]))
        self.assertTrue(Vector([0, "", 1.0]))
        self.assertTrue(Vector([0.0, "", 1]))
        self.assertTrue(Vector(Node("foo")))
        self.assertTrue(Vector(test_func))
        self.assertTrue(Vector(test_class))
        self.assertTrue(Vector.symbol('foo'))

    def test_as_double(self):
        for value in [0, 1, 0.1, 1.5, -1e99, math.inf, True, False]:
            self.assertEqual(float(Vector(value)), value)
        self.assertTrue(math.isnan(float(Vector(math.nan))))
        self.assertTrue(math.isnan(float(null)))
        self.assertTrue(math.isnan(float(Vector([1, 2, 3]))))
        self.assertTrue(math.isnan(float(Vector("Hello world!"))))
        foo = Vector.symbol('foo')
        self.assertEqual(float(foo), foo[0])

    def test_as_integer(self):
        self.assertEqual(int(Vector(0)), 0)
        self.assertEqual(int(Vector(1)), 1)
        self.assertEqual(int(Vector(0.1)), 0)
        self.assertEqual(int(Vector(1.5)), 1)
        self.assertEqual(int(Vector(-1e99)), 0)
        self.assertEqual(int(Vector(math.inf)), 0)
        self.assertEqual(int(true), 1)
        self.assertEqual(int(false), 0)
        self.assertEqual(int(null), 0)
        self.assertEqual(int(Vector([1, 2, 3])), 0)
        self.assertEqual(int(Vector("Hello world!")), 0)
        self.assertEqual(int(Vector.symbol('foo')), 0)

    def test_as_string(self):
        TESTS = [
            (null, ""),
            (Vector([1, -2, 3.0, 1e99, 1.567e-9]), "1-231e+991.567e-09"),
            (Vector([1 / 3]), "0.333333333"),
            (Vector("Hello world!"), "Hello world!"),
            (Vector(["Hello ", "world!"]), "Hello world!"),
            (Vector(["testing", "testing", 0, 1, 2.2, 3.0]), "testingtesting012.23"),
            (Vector.symbol('foo'), "foo"),
            (Vector.symbol('foo').concat(Vector.symbol('bar')), "foobar"),
            (Vector.symbol('foo').concat(Vector([0, 1, 2.2, 3.0])), "foo012.23"),
            (Vector(Node('foo', {'bar'}, {'baz': Vector(2)})), "foo"),
            (Vector(self.test_as_string), "test_as_string"),
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
        self.assertEqual(hash(Vector(["foo", 1])), hash(Vector(["foo", 1.0])))
        self.assertNotEqual(hash(Vector(["foo", 1.0])), hash(Vector(["foo", 1.1])))
        self.assertEqual(hash(Vector(Node('foo', {'bar'}, {'test': Vector(5)}))), 9166098286934782834)
        self.assertEqual(hash(Vector.symbol('foo')), hash(Vector(FOO_SYMBOL_NUMBER)))
        self.assertIsNotNone(hash(Vector(test_func)))  # just check it works, value will not be stable
        self.assertIsNotNone(hash(Vector(test_class)))  # just check it works, value will not be stable

    def test_hash_floor_floats(self):
        self.assertEqual(null.hash(True), null.hash(False))
        self.assertEqual(Vector(0.1).hash(True), Vector(0).hash(True))
        self.assertEqual(Vector("Hello world!").hash(True), Vector("Hello world!").hash(False))
        self.assertEqual(Vector(["foo", 1]).hash(True), Vector(["foo", 1.0]).hash(True))
        self.assertEqual(Vector(["foo", 1.1]).hash(True), Vector(["foo", 1.0]).hash(True))

    def test_hash_uniformity(self):
        from scipy.stats import kstest
        hashes = []
        scale = 1 << 64
        for c0 in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            for c1 in 'abcdefghijklmnopqrstuvwxyz':
                for i in range(100):
                    hashes.append((hash(Vector([c0 + c1, i])) % scale) / scale)
        result = kstest(hashes, 'uniform')
        self.assertGreater(result.pvalue, 0.05)

    def test_match(self):
        self.assertEqual(null.match(1, float), None)
        self.assertEqual(null.match(1, int), None)
        self.assertEqual(null.match(1, str), None)
        self.assertEqual(null.match(1, float, 3.5), 3.5)
        self.assertEqual(null.match(3, float, [1, 2, 3.5]), [1, 2, 3.5])
        self.assertEqual(Vector(3.5).match(0, float), [3.5])
        self.assertEqual(Vector(3.5).match(1, float), 3.5)
        self.assertEqual(Vector(3.5).match(1, int), 3)
        self.assertEqual(Vector(-3.5).match(1, int), -4)
        self.assertEqual(Vector(3.5).match(1, str), None)
        self.assertEqual(Vector(3.5).match(3, float), [3.5, 3.5, 3.5])
        self.assertEqual(Vector(3.5).match(3, str), None)
        self.assertEqual(Vector(3.5).match(3, int), [3, 3, 3])
        self.assertEqual(Vector("Hello world!").match(1, str), "Hello world!")
        self.assertEqual(Vector("Hello world!").match(2, str), ["Hello world!", "Hello world!"])
        self.assertEqual(Vector(["Hello ", "world!"]).match(2, str), ["Hello ", "world!"])
        self.assertEqual(Vector(["Hello ", 3.5]).match(2, str), ["Hello ", "3.5"])
        self.assertEqual(Vector(["Hello ", "world!"]).match(1, str), None)
        self.assertEqual(Vector(["0"]).match(1, int), 0)
        self.assertEqual(Vector(["0"]).match(0, int), [0])
        self.assertEqual(true.match(1, float), 1.0)
        self.assertIs(true.match(1, bool), True)
        self.assertEqual(true.match(2, bool), [True, True])
        self.assertEqual(Vector([0, 1]).match(2, bool), [False, True])
        self.assertEqual(Vector.symbol('foo').match(), ['foo'])
        self.assertEqual(Vector.symbol('foo').match(1, float), FOO_SYMBOL_NUMBER)
        self.assertEqual(Vector.symbol('foo').match(0, str), ['foo'])
        self.assertEqual(Vector.symbol('foo').match(1, str), 'foo')
        self.assertEqual(Vector.symbol('foo').match(0, int), None)
        self.assertEqual(Vector.symbol('foo').match(1, int), None)
        self.assertEqual(Vector.symbol('foo').match(2, str), ['foo', 'foo'])
        self.assertEqual(Vector.symbol('foo').concat(Vector.symbol('bar')).match(2, str), ['foo', 'bar'])
        self.assertEqual(Vector.symbol('foo').concat(Vector(1)).match(2), ['foo', 1])
        self.assertEqual(Vector.symbol('foo').concat(Vector(1)).match(2, str), None)
        self.assertEqual(Vector.symbol('foo').concat(Vector(1)).match(2, int), None)
        self.assertEqual(Vector.symbol('foo').concat(Vector(1)).match(2, float), [FOO_SYMBOL_NUMBER, 1])
        self.assertEqual(Vector(FOO_SYMBOL_NUMBER).match(1, str), 'foo')
        self.assertEqual(Vector(FOO_SYMBOL_NUMBER).match(2, None), ['foo', 'foo'])
        self.assertEqual(Vector([FOO_SYMBOL_NUMBER, 'bar']).match(0, str), ['foo', 'bar'])
        self.assertEqual(Vector([FOO_SYMBOL_NUMBER, 'bar']).slice(Vector(0)).match(2, str), ['foo', 'foo'])

    def test_repr(self):
        self.assertEqual(repr(null), "null")
        self.assertEqual(repr(true), "1")
        self.assertEqual(repr(false), "0")
        self.assertEqual(repr(Vector(1 / 3)), "0.333333333")
        self.assertEqual(repr(Vector([1, 2, 3])), "1;2;3")
        self.assertEqual(repr(Vector([1, 2.5, 3])), "1;2.5;3")
        self.assertEqual(repr(Vector("Hello world!")), "'Hello world!'")
        self.assertEqual(repr(Vector("")), "''")
        self.assertEqual(repr(Vector([99, "red", "balloons"])), "99;'red';'balloons'")
        node = Node('foo', {'bar'}, {'color': Vector(1)})
        self.assertEqual(repr(Vector(node)), f"({node!r})")
        self.assertEqual(repr(Vector.symbol('foo')), ':foo')
        self.assertEqual(repr(Vector.symbol('foo').concat(Vector(99))), ':foo;99')

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
        self.assertEqual(+Vector.symbol('foo'), Vector.symbol('foo'))

    def test_abs(self):
        self.assertEqual(abs(null), null)
        self.assertEqual(abs(Vector("Hello world!")), null)
        self.assertEqual(abs(Vector(-3)), Vector(3))
        self.assertEqual(abs(Vector([0, 0.1, 3, -99, 1e99, math.inf])), Vector([0, 0.1, 3, 99, 1e99, math.inf]))

    def test_ceil(self):
        self.assertEqual(math.ceil(null), null)
        self.assertEqual(math.ceil(Vector("Hello world!")), null)
        self.assertEqual(math.ceil(Vector(-3.5)), Vector(-3))
        self.assertEqual(math.ceil(Vector([0, 0.1, 3.5, -99.5, 1e-99, math.inf])), Vector([0, 1, 4, -99, 1, math.inf]))

    def test_floor(self):
        self.assertEqual(math.floor(null), null)
        self.assertEqual(math.floor(Vector("Hello world!")), null)
        self.assertEqual(math.floor(Vector(-3.5)), Vector(-4))
        self.assertEqual(math.floor(Vector([0, 0.1, 3.5, -99.5, 1e-99, math.inf])), Vector([0, 0, 3, -100, 0, math.inf]))

    def test_fract(self):
        self.assertEqual(null.fract(), null)
        self.assertEqual(Vector("Hello world!").fract(), null)
        self.assertAllAlmostEqual(Vector(-3.5).fract(), Vector(0.5))
        self.assertAllAlmostEqual(Vector([0, 0.1, 3.5, -99.5, 1e-99, math.inf]).fract(), Vector([0, 0.1, 0.5, 0.5, 1e-99, math.nan]))

    def test_round(self):
        self.assertEqual(round(null), null)
        self.assertEqual(round(Vector("Hello world!")), null)
        self.assertAllAlmostEqual(round(Vector(-3.5)), Vector(-4))
        self.assertAllAlmostEqual(round(Vector([0, 0.1, 3.4, -99.5, 1e-99, math.inf])), Vector([0, 0, 3, -100, 0, math.inf]))

    def test_contains(self):
        self.assertTrue(null in null)
        self.assertFalse(Vector(4) in null)
        self.assertTrue(null in Vector(4))
        self.assertTrue(null in Vector('hello'))
        self.assertFalse(-1 in Vector.range(10))
        self.assertTrue(0 in Vector.range(10))
        self.assertTrue(9 in Vector.range(10))
        self.assertFalse(10 in Vector.range(10))
        self.assertTrue(Vector([0, 1, 2]) in Vector.range(10))
        self.assertFalse(Vector([0, 1, 3]) in Vector.range(10))
        self.assertTrue(Vector([7, 8, 9]) in Vector.range(10))
        self.assertFalse(Vector([8, 9, 10]) in Vector.range(10))
        self.assertTrue(Vector.range(10) in Vector.range(10))
        self.assertTrue(Vector("Hello") in Vector(["Hello", "world"]))
        self.assertTrue(Vector(["world"]) in Vector(["Hello", "world"]))
        self.assertTrue(Vector(["Hello", "world"]) in Vector(["Hello", "world"]))
        self.assertFalse(Vector(["Hello", "Dave"]) in Vector(["Hello", "world"]))
        self.assertTrue(Vector([1, 2]) in Vector(["Hello", 1, 2, "world"]))

    def test_add(self):
        x = Vector([1, 0.1, -5, 1e6, math.inf])
        self.assertEqual(x + null, null)
        self.assertEqual(null + x, null)
        self.assertEqual(x + Vector("Hello world!"), null)
        self.assertEqual(Vector("Hello world!") + x, null)
        self.assertAllAlmostEqual(x + Vector(1), Vector([2, 1.1, -4, 1000001, math.inf]))
        self.assertAllAlmostEqual(Vector(1) + x, Vector([2, 1.1, -4, 1000001, math.inf]))
        self.assertAllAlmostEqual(x + Vector([1, 2]), Vector([2, 2.1, -4, 1000002, math.inf]))
        self.assertAllAlmostEqual(Vector([1, 2]) + x, Vector([2, 2.1, -4, 1000002, math.inf]))
        self.assertAllAlmostEqual(x + x, Vector([2, 0.2, -10, 2e6, math.inf]))
        self.assertAllAlmostEqual(x + 0, x)
        self.assertAllAlmostEqual(0 + x, x)

    def test_mul_add(self):
        x = Vector([1, 0.1, -5, 1e6, math.inf])
        self.assertEqual(x.mul_add(Vector(1), null), null)
        self.assertEqual(x.mul_add(null, Vector(1)), null)
        self.assertEqual(null.mul_add(x, Vector(1)), null)
        self.assertEqual(x.mul_add(Vector(1), Vector("Hello world!")), null)
        self.assertEqual(Vector("Hello world!").mul_add(x, Vector(1)), null)
        self.assertAllAlmostEqual(x.mul_add(Vector(0), Vector(1)), x)
        self.assertAllAlmostEqual(x.mul_add(Vector(1), Vector(1)), x + 1)
        self.assertAllAlmostEqual(x.mul_add(x, Vector(1)), x + x)
        self.assertAllAlmostEqual(Vector(0).mul_add(x, Vector(1)), x)
        self.assertAllAlmostEqual(Vector(1).mul_add(x, Vector(1)), x + 1)
        self.assertAllAlmostEqual(Vector(0).mul_add(x, x), x * x)
        self.assertAllAlmostEqual(Vector(x).mul_add(x, x), x * x + x)

    def test_sub(self):
        x = Vector([1, 0.1, -5, 1e6, math.inf])
        self.assertEqual(x - null, null)
        self.assertEqual(null - x, null)
        self.assertEqual(x - Vector("Hello world!"), null)
        self.assertEqual(Vector("Hello world!") - x, null)
        self.assertAllAlmostEqual(x - Vector(1), Vector([0, -0.9, -6, 999999, math.inf]))
        self.assertAllAlmostEqual(Vector(1) - x, Vector([0, 0.9, 6, -999999, -math.inf]))
        self.assertAllAlmostEqual(x - Vector([1, 2]), Vector([0, -1.9, -6, 999998, math.inf]))
        self.assertAllAlmostEqual(Vector([1, 2]) - x, Vector([0, 1.9, 6, -999998, -math.inf]))
        y = x - x
        self.assertAllAlmostEqual(y[:4], Vector([0, 0, 0, 0]))
        self.assertTrue(math.isnan(y[4]))
        self.assertAllAlmostEqual(x - 0, x)
        self.assertAllAlmostEqual(0 - x, -x)

    def test_mul(self):
        x = Vector([1, 0.1, -5, 1e6, math.inf])
        self.assertEqual(x * null, null)
        self.assertEqual(null * x, null)
        self.assertEqual(x * Vector("Hello world!"), null)
        self.assertEqual(Vector("Hello world!") * x, null)
        self.assertAllAlmostEqual(x * Vector(1), x)
        self.assertAllAlmostEqual(Vector(1) * x, x)
        self.assertAllAlmostEqual(x * Vector([1, 2]), Vector([1, 0.2, -5, 2e6, math.inf]))
        self.assertAllAlmostEqual(Vector([1, 2]) * x, Vector([1, 0.2, -5, 2e6, math.inf]))
        self.assertAllAlmostEqual(x * x, Vector([1, 0.01, 25, 1e12, math.inf]))
        self.assertAllAlmostEqual(x * 0, Vector([0, 0, 0, 0, math.nan]))
        self.assertAllAlmostEqual(0 * x, Vector([0, 0, 0, 0, math.nan]))

    def test_truediv(self):
        x = Vector([1, 0.1, -5, 1e6, math.inf])
        self.assertEqual(x / null, null)
        self.assertEqual(null / x, null)
        self.assertEqual(x / Vector("Hello world!"), null)
        self.assertEqual(Vector("Hello world!") / x, null)
        self.assertAllAlmostEqual(x / Vector(1), x)
        self.assertAllAlmostEqual(Vector(1) / x, Vector([1, 10, -0.2, 1e-6, 0]))
        self.assertAllAlmostEqual(x / Vector([1, 2]), Vector([1, 0.05, -5, 5e5, math.inf]))
        self.assertAllAlmostEqual(Vector([1, 2]) / x, Vector([1, 20, -0.2, 2e-6, 0]))
        y = x / x
        self.assertAllAlmostEqual(y[:4], Vector([1, 1, 1, 1]))
        self.assertTrue(math.isnan(y[4]))
        self.assertAllAlmostEqual(x / 1, x)
        self.assertAllAlmostEqual(1 / x, Vector([1, 10, -0.2, 1e-6, 0]))

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
        self.assertAllAlmostEqual(x // 1, Vector([1, 0, -6, 1e6, math.inf]))
        self.assertAllAlmostEqual(1 // x, Vector([1, 10, -1, 0, 0]))

    def test_mod(self):
        x = Vector([1, 0.1, -5, 1e6])
        self.assertEqual(x % null, null)
        self.assertEqual(null % x, null)
        self.assertEqual(x % Vector("Hello world!"), null)
        self.assertEqual(Vector("Hello world!") % x, null)
        self.assertAllAlmostEqual(x % Vector(2), Vector([1, 0.1, 1, 0]))
        self.assertAllAlmostEqual(Vector(2) % x, Vector([0, 0, -3, 2]))
        self.assertAllAlmostEqual(x % Vector([2, 3]), Vector([1, 0.1, 1, 1]))
        self.assertAllAlmostEqual(Vector([2, 3]) % x, Vector([0, 0, -3, 3]))
        self.assertAllAlmostEqual(x % x, Vector([0, 0, 0, 0]))
        self.assertAllAlmostEqual(x % 2, Vector([1, 0.1, 1, 0]))
        self.assertAllAlmostEqual(2 % x, Vector([0, 0, -3, 2]))

    def test_pow(self):
        x = Vector([1, 0.1, -5, 1e6, math.inf])
        self.assertEqual(x ** null, null)
        self.assertEqual(null ** x, null)
        self.assertEqual(x ** Vector("Hello world!"), null)
        self.assertEqual(Vector("Hello world!") ** x, null)
        self.assertAllAlmostEqual(x ** Vector(2), Vector([1, 0.01, 25, 1e12, math.inf]))
        self.assertAllAlmostEqual(Vector(2) ** x, Vector([2, 1.0717734625362931, 0.03125, math.inf, math.inf]))
        self.assertAllAlmostEqual(x ** Vector([1, 2]), Vector([1, 0.01, -5, 1e12, math.inf]))
        self.assertAllAlmostEqual(Vector([1, 2]) ** x, Vector([1, 1.0717734625362931, 1, math.inf, 1]))
        self.assertAllAlmostEqual(x * x, Vector([1, 0.01, 25, 1e12, math.inf]))
        self.assertAllAlmostEqual(pow(x, 2, 2), Vector([1, 0.01, 1, 0, math.nan]))
        self.assertAllAlmostEqual(pow(2, x, 2), Vector([0, 1.0717734625362931, 0.03125, math.nan, math.nan]))

    def test_compare(self):
        x = Vector(1)
        y = Vector(2)
        self.assertEqual(x.compare(x), 0)
        self.assertEqual(x.compare(y), -1)
        self.assertEqual(y.compare(x), 1)
        self.assertEqual(x.compare(Vector('x')), -2)

    def test_eq(self):
        self.assertTrue(null == null)
        self.assertTrue(null == Vector())
        self.assertTrue(null == Vector())
        self.assertTrue(true == Vector(1))
        self.assertTrue(false == Vector(0))
        self.assertTrue(Vector([1, 2, 3]) == Vector([1, 2, 3]))
        self.assertFalse(Vector([1, 2]) == Vector([1, 2, 3]))
        self.assertFalse(Vector([1, 2, 4]) == Vector([1, 2, 3]))
        self.assertTrue(Vector(["Hello ", "world!"]) == Vector(["Hello ", "world!"]))
        self.assertFalse(Vector(["Hello ", "you!"]) == Vector(["Hello ", "world!"]))
        self.assertTrue(Vector.symbol('foo') == Vector(FOO_SYMBOL_NUMBER))
        self.assertTrue(Vector.symbol('foo') == Vector.symbol('foo'))
        self.assertFalse(Vector(['a', 'b']) == Vector(['a', 2]))

    def test_ne(self):
        self.assertFalse(null != null)
        self.assertFalse(null != Vector())
        self.assertTrue(true != Vector(0))
        self.assertTrue(false != Vector(1))
        self.assertFalse(Vector([1, 2, 3]) != Vector([1, 2, 3]))
        self.assertTrue(Vector([1, 2]) != Vector([1, 2, 3]))
        self.assertTrue(Vector([1, 2, 4]) != Vector([1, 2, 3]))
        self.assertTrue(Vector(["Hello ", "world!"]) != Vector(["Hello world!"]))
        self.assertTrue(Vector(["Hello ", "you!"]) != Vector(["Hello ", "world!"]))
        self.assertTrue(Vector.symbol('foo') != Vector.symbol('bar'))
        self.assertTrue(Vector(['a', 'b']) != Vector(['a', 2]))

    def test_gt(self):
        self.assertTrue(Vector(1) > null)
        self.assertTrue(true > Vector(0))
        self.assertFalse(false > Vector(1))
        self.assertTrue(Vector([1, 2, 4]) > Vector([1, 2, 3]))
        self.assertTrue(Vector([1, 3]) > Vector([1, 2, 3]))
        self.assertTrue(Vector(["Hello world!"]) > Vector(["Hello ", "world!"]))
        self.assertTrue(Vector(["Hello ", "world!"]) > Vector(["Hello ", "cruel world!"]))
        self.assertTrue(Vector(["Z"]) > Vector(["Hello world!"]))
        with self.assertRaises(TypeError):
            Vector('a') > Vector(1)
        with self.assertRaises(TypeError):
            Vector(['a', 'b']) > Vector(['a', 2])

    def test_ge(self):
        self.assertTrue(null >= null)
        self.assertTrue(null >= Vector())
        self.assertTrue(Vector(1) >= null)
        self.assertTrue(true >= Vector(0))
        self.assertTrue(false >= Vector(0))
        self.assertFalse(Vector([1, 2, 3]) >= Vector([1, 2, 3, 4]))
        self.assertTrue(Vector([1, 2, 3]) >= Vector([1, 2, 3]))
        self.assertTrue(Vector([1, 3]) >= Vector([1, 2, 3]))
        self.assertTrue(Vector(["Hello world!"]) >= Vector(["Hello ", "world!"]))
        self.assertTrue(Vector(["Z"]) >= Vector(["Hello world!"]))
        x = Vector.symbol('foo').compose(Vector([1, 2, 3]))
        self.assertTrue(x >= x)
        with self.assertRaises(TypeError):
            Vector('a') >= Vector(1)
        with self.assertRaises(TypeError):
            Vector(['a', 'b']) >= Vector(['a', 2])

    def test_lt(self):
        self.assertTrue(null < Vector(1))
        self.assertTrue(Vector(0) < true)
        self.assertFalse(Vector(1) < false)
        self.assertTrue(Vector([1, 2, 3]) < Vector([1, 2, 4]))
        self.assertTrue(Vector([1, 2, 3]) < Vector([1, 3]))
        self.assertTrue(Vector(["Hello ", "world!"]) < Vector(["Hello world!"]))
        self.assertTrue(Vector(["Hello world!"]) < Vector(["Z"]))
        with self.assertRaises(TypeError):
            Vector('a') < Vector(1)
        with self.assertRaises(TypeError):
            Vector(['a', 'b']) < Vector(['a', 2])

    def test_le(self):
        self.assertTrue(null <= null)
        self.assertTrue(null <= Vector())
        self.assertTrue(null <= Vector(1))
        self.assertTrue(Vector(0) <= true)
        self.assertTrue(Vector(0) <= false)
        self.assertFalse(Vector([1, 2, 3, 4]) <= Vector([1, 2, 3]))
        self.assertTrue(Vector([1, 2, 3]) <= Vector([1, 2, 3]))
        self.assertTrue(Vector([1, 2, 3]) <= Vector([1, 3]))
        self.assertTrue(Vector(["Hello ", "world!"]) <= Vector(["Hello world!"]))
        self.assertTrue(Vector(["Hello world!"]) <= Vector(["Z"]))
        with self.assertRaises(TypeError):
            Vector('a') <= Vector(1)
        with self.assertRaises(TypeError):
            Vector(['a', 'b']) <= Vector(['a', 2])

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
        self.assertEqual(Vector(0).normalize(), null)
        self.assertEqual(Vector(["Hello world!"]).normalize(), null)
        self.assertEqual(Vector(1).normalize(), Vector(1))
        self.assertEqual(Vector([0, 2]).normalize(), Vector([0, 1]))
        self.assertEqual(Vector([-2, 0]).normalize(), Vector([-1, 0]))
        self.assertEqual(Vector([0, 0]).normalize(), null)
        self.assertAllAlmostEqual(Vector([1, 1]).normalize(), Vector([0.707106781, 0.707106781]))
        self.assertAllAlmostEqual(Vector([1, -2, 3]).normalize(), Vector([0.267261242, -0.534522484, 0.801783726]))

    def test_dot(self):
        self.assertEqual(null.dot(null), null)
        self.assertEqual(null.dot(Vector([1, 2, 3])), null)
        self.assertEqual(Vector([1, 2, 3]).dot(null), null)
        self.assertAllAlmostEqual(Vector([1, 2, 3]).dot(Vector([1, 2, 3])), Vector(14))
        self.assertAllAlmostEqual(Vector([1, 2, 3]).dot(Vector([3, 2, 1])), Vector(10))
        self.assertAllAlmostEqual(Vector([1, 2, 3]).dot(Vector(3)), Vector(18))
        self.assertAllAlmostEqual(Vector([1, 2, 3]).dot(Vector([3, 2])), Vector(16))

    def test_cross(self):
        self.assertEqual(null.cross(null), null)
        self.assertEqual(null.cross(Vector([1, 2, 3])), null)
        self.assertEqual(Vector([1, 2, 3]).cross(null), null)
        self.assertAllAlmostEqual(Vector([1, 2, 3]).cross(Vector([1, 2, 3])), Vector([0, 0, 0]))
        self.assertAllAlmostEqual(Vector([1, 2, 3]).cross(Vector([-1, -2, -3])), Vector([0, 0, 0]))
        self.assertEqual(Vector([1, 2, 3]).cross(Vector([1, 2])), null)
        self.assertAllAlmostEqual(Vector([1, 2, 3]).cross(Vector([3, 2, 1])), Vector([-4, 8, -4]))

    def test_concat(self):
        a = Vector([1, 2, 3])
        b = Vector([4, 5, 6])
        self.assertIs(a.concat(null), a)
        self.assertIs(null.concat(b), b)
        self.assertEqual(a.concat(b), Vector([1, 2, 3, 4, 5, 6]))
        self.assertEqual(b.concat(a), Vector([4, 5, 6, 1, 2, 3]))
        self.assertTrue(a.concat(b).numeric)
        self.assertEqual(Vector(['hello']).concat(b), Vector(['hello', 4, 5, 6]))
        self.assertEqual(a.concat(Vector(['world'])), Vector([1, 2, 3, 'world']))
        self.assertEqual(Vector(['hello']).concat(Vector(['world'])), Vector(['hello', 'world']))

    def test_clamp(self):
        x = Vector.range(10)
        self.assertIs(x.clamp(None, None), x)
        self.assertIs(x.clamp(Vector(1), None), x)
        self.assertIs(x.clamp(None, Vector(5)), x)
        self.assertEqual(x.clamp(Vector(0), Vector(10)), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(x.clamp(Vector(1), Vector(8)), [1, 1, 2, 3, 4, 5, 6, 7, 8, 8])
        self.assertEqual(x.clamp(Vector(-1), Vector(8)), [0, 1, 2, 3, 4, 5, 6, 7, 8, 8])
        self.assertEqual(x.clamp(Vector(1), Vector(11)), [1, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    def test_maximum(self):
        self.assertTrue(math.isnan(null.maximum()))
        self.assertTrue(math.isnan(Vector('hello').maximum()))
        self.assertEqual(Vector.range(10).maximum(), 9)

    def test_minimum(self):
        self.assertTrue(math.isnan(null.minimum()))
        self.assertTrue(math.isnan(Vector('hello').minimum()))
        self.assertEqual(Vector.range(9, -1, -1).minimum(), 0)

    def test_squared_sum(self):
        self.assertTrue(math.isnan(null.squared_sum()))
        self.assertTrue(math.isnan(Vector('a').squared_sum()))
        self.assertEqual(Vector(0).squared_sum(), 0)
        self.assertEqual(Vector([0, 1, 2, 3]).squared_sum(), 14)
