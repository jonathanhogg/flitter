"""
Tests of the model.StateDict class
"""

import pickle
import unittest

from flitter.model import Vector, StateDict, DummyStateDict, null


class TestStateDict(unittest.TestCase):
    def test_construct(self):
        state = StateDict()
        self.assertEqual(len(state), 0)
        state = StateDict({'x': 5})
        self.assertEqual(len(state), 1)
        self.assertEqual(state['x'], Vector(5))

    def test_repr(self):
        state = StateDict({'x': [1, 2, 3], 'y': 'test', 'z': null})
        self.assertEqual(repr(state), "StateDict({'x': 1;2;3, 'y': 'test'})")

    def test_changed(self):
        state = StateDict({'x': 5})
        self.assertFalse(state.changed)
        state['x'] = 6
        self.assertTrue(state.changed)
        self.assertEqual(state.changed_keys, {Vector('x')})
        self.assertEqual(state['x'], Vector(6))
        state['y'] = 7
        self.assertTrue(state.changed)
        self.assertEqual(state.changed_keys, {Vector('x'), Vector('y')})
        self.assertEqual(state['x'], Vector(6))
        self.assertEqual(state['y'], Vector(7))
        state.clear_changed()
        self.assertFalse(state.changed)
        self.assertEqual(state.changed_keys, set())
        del state['x']
        self.assertTrue(state.changed)
        self.assertEqual(state.changed_keys, {Vector('x')})
        self.assertFalse('x' in state)

    def test_set_item(self):
        state = StateDict()
        state['x'] = 5
        self.assertTrue('x' in state)
        self.assertEqual(state.changed_keys, {Vector('x')})
        state.clear_changed()
        state['x'] = null
        self.assertFalse('x' in state)
        self.assertEqual(state.changed_keys, {Vector('x')})

    def test_get_item(self):
        state = StateDict({'x': 5})
        self.assertFalse(state.changed)
        x = state['x']
        self.assertIsInstance(x, Vector)
        self.assertEqual(x, 5)
        y = state['y']
        self.assertIsInstance(y, Vector)
        self.assertEqual(y, null)
        self.assertFalse(state.changed)

    def test_keys(self):
        state = StateDict({'x': 5})
        self.assertEqual(state.keys(), {Vector('x')})
        state['y'] = 7
        self.assertEqual(state.keys(), {Vector('x'), Vector('y')})

    def test_iter(self):
        state = StateDict({'x': 5})
        state['y'] = 7
        i = iter(state)
        self.assertEqual(next(i), Vector('x'))
        self.assertEqual(next(i), Vector('y'))
        with self.assertRaises(StopIteration):
            next(i)

    def test_values(self):
        state = StateDict({'x': 5})
        self.assertEqual(list(state.values()), [Vector(5)])
        state['y'] = 7
        self.assertEqual(list(state.values()), [Vector(5), Vector(7)])

    def test_items(self):
        state = StateDict({'x': 5})
        self.assertEqual(list(state.items()), [(Vector('x'), Vector(5))])
        state['y'] = 7
        self.assertEqual(list(state.items()), [(Vector('x'), Vector(5)), (Vector('y'), Vector(7))])

    def test_clear(self):
        state = StateDict({'x': 5, 'y': 6})
        self.assertEqual(len(state), 2)
        state.clear()
        self.assertEqual(len(state), 0)
        self.assertTrue(state.changed)
        self.assertEqual(state.changed_keys, {Vector('x'), Vector('y')})

    def test_with_keys(self):
        state = StateDict({'x': [1, 2, 3], 'y': 'test', Vector.symbol('z'): Vector.range(10)})
        state2 = state.with_keys([Vector('x'), Vector.symbol('z')])
        self.assertFalse(state.changed)
        self.assertEqual(len(state), 3)
        self.assertFalse(state2.changed)
        self.assertEqual(len(state2), 2)
        self.assertEqual(state2['x'], Vector([1, 2, 3]))
        self.assertEqual(state2[Vector.symbol('z')], Vector.range(10))

    def test_pickling(self):
        state = StateDict({'x': [1, 2, 3], 'y': 'test'})
        state[Vector.symbol('z')] = Vector.range(10)
        self.assertTrue(state.changed)
        state2 = pickle.loads(pickle.dumps(state))
        self.assertFalse(state2.changed)
        self.assertEqual(list(state.items()), list(state2.items()))


class TestDummyState(unittest.TestCase):
    def test_dummy(self):
        state = DummyStateDict()
        self.assertTrue(Vector('x') in state)
        self.assertEqual(state[Vector('x')], null)
        self.assertTrue(Vector(43) in state)
        self.assertEqual(state[Vector(43)], null)
        self.assertTrue(Vector.symbol('z') in state)
        self.assertEqual(state[Vector.symbol('z')], null)
