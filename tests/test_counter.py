"""
Tests of the flitter `!counter` object
"""

import unittest
import unittest.mock

from flitter.render.counter import Counter
from flitter.model import Node, Vector, StateDict


class TestCounter(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.counter_id = Vector.symbol('foo')
        self.time_id = self.counter_id.concat(Vector.symbol('time'))
        self.counter = Counter()
        self.state = StateDict()
        self.engine = unittest.mock.Mock()
        self.engine.state = self.state

    async def test_empty(self):
        node = Node('counter', set(), {})
        await self.counter.update(self.engine, node, clock=1)
        self.assertFalse(self.state.changed)

    async def test_basic_setup(self):
        node = Node('counter', set(), {'state': self.counter_id})
        await self.counter.update(self.engine, node, clock=1)
        self.assertEqual(self.state[self.counter_id], Vector(0))
        self.assertEqual(self.state[self.time_id], Vector(1))
        await self.counter.update(self.engine, node, clock=2)
        self.assertEqual(self.state[self.counter_id], Vector(0))
        self.assertEqual(self.state[self.time_id], Vector(2))

    async def test_basic_incrementing(self):
        node = Node('counter', set(), {'state': self.counter_id, 'rate': Vector(1)})
        await self.counter.update(self.engine, node, clock=1)
        self.assertEqual(self.state[self.counter_id], Vector(0))
        self.assertEqual(self.state[self.time_id], Vector(1))
        await self.counter.update(self.engine, node, clock=1)
        self.assertEqual(self.state[self.counter_id], Vector(0))
        self.assertEqual(self.state[self.time_id], Vector(1))
        await self.counter.update(self.engine, node, clock=2)
        self.assertEqual(self.state[self.counter_id], Vector(1))
        self.assertEqual(self.state[self.time_id], Vector(2))
        await self.counter.update(self.engine, node, clock=1)
        self.assertEqual(self.state[self.counter_id], Vector(0))
        self.assertEqual(self.state[self.time_id], Vector(1))

    async def test_basic_decrementing(self):
        node = Node('counter', set(), {'state': self.counter_id, 'rate': Vector(-1)})
        await self.counter.update(self.engine, node, clock=1)
        self.assertEqual(self.state[self.counter_id], Vector(0))
        self.assertEqual(self.state[self.time_id], Vector(1))
        await self.counter.update(self.engine, node, clock=1)
        self.assertEqual(self.state[self.counter_id], Vector(0))
        self.assertEqual(self.state[self.time_id], Vector(1))
        await self.counter.update(self.engine, node, clock=2)
        self.assertEqual(self.state[self.counter_id], Vector(-1))
        self.assertEqual(self.state[self.time_id], Vector(2))
        await self.counter.update(self.engine, node, clock=1)
        self.assertEqual(self.state[self.counter_id], Vector(0))
        self.assertEqual(self.state[self.time_id], Vector(1))

    async def test_basic_rate_change(self):
        node = Node('counter', set(), {'state': self.counter_id, 'rate': Vector(1)})
        await self.counter.update(self.engine, node, clock=1)
        self.assertEqual(self.state[self.counter_id], Vector(0))
        self.assertEqual(self.state[self.time_id], Vector(1))
        await self.counter.update(self.engine, node, clock=2)
        self.assertEqual(self.state[self.counter_id], Vector(1))
        self.assertEqual(self.state[self.time_id], Vector(2))
        node = Node('counter', set(), {'state': self.counter_id, 'rate': Vector(2)})
        await self.counter.update(self.engine, node, clock=3)
        self.assertEqual(self.state[self.counter_id], Vector(3))
        self.assertEqual(self.state[self.time_id], Vector(3))
        node = Node('counter', set(), {'state': self.counter_id, 'rate': Vector(-3)})
        await self.counter.update(self.engine, node, clock=4)
        self.assertEqual(self.state[self.counter_id], Vector(0))
        self.assertEqual(self.state[self.time_id], Vector(4))

    async def test_initial(self):
        node = Node('counter', set(), {'state': self.counter_id, 'initial': Vector(3.5)})
        await self.counter.update(self.engine, node, clock=1)
        self.assertEqual(self.state[self.counter_id], Vector(3.5))
        self.assertEqual(self.state[self.time_id], Vector(1))
        await self.counter.update(self.engine, node, clock=2)
        self.assertEqual(self.state[self.counter_id], Vector(3.5))
        self.assertEqual(self.state[self.time_id], Vector(2))

    async def test_initial_incrementing(self):
        node = Node('counter', set(), {'state': self.counter_id, 'initial': Vector(3.5), 'rate': Vector(1)})
        await self.counter.update(self.engine, node, clock=1)
        self.assertEqual(self.state[self.counter_id], Vector(3.5))
        self.assertEqual(self.state[self.time_id], Vector(1))
        await self.counter.update(self.engine, node, clock=1)
        self.assertEqual(self.state[self.counter_id], Vector(3.5))
        self.assertEqual(self.state[self.time_id], Vector(1))
        await self.counter.update(self.engine, node, clock=2)
        self.assertEqual(self.state[self.counter_id], Vector(4.5))
        self.assertEqual(self.state[self.time_id], Vector(2))
        await self.counter.update(self.engine, node, clock=1)
        self.assertEqual(self.state[self.counter_id], Vector(3.5))
        self.assertEqual(self.state[self.time_id], Vector(1))

    async def test_initial_decrementing(self):
        node = Node('counter', set(), {'state': self.counter_id, 'initial': Vector(3.5), 'rate': Vector(-1)})
        await self.counter.update(self.engine, node, clock=1)
        self.assertEqual(self.state[self.counter_id], Vector(3.5))
        self.assertEqual(self.state[self.time_id], Vector(1))
        await self.counter.update(self.engine, node, clock=1)
        self.assertEqual(self.state[self.counter_id], Vector(3.5))
        self.assertEqual(self.state[self.time_id], Vector(1))
        await self.counter.update(self.engine, node, clock=2)
        self.assertEqual(self.state[self.counter_id], Vector(2.5))
        self.assertEqual(self.state[self.time_id], Vector(2))
        await self.counter.update(self.engine, node, clock=1)
        self.assertEqual(self.state[self.counter_id], Vector(3.5))
        self.assertEqual(self.state[self.time_id], Vector(1))

    async def test_minimum(self):
        node = Node('counter', set(), {'state': self.counter_id, 'rate': Vector(-1), 'minimum': Vector(-1)})
        await self.counter.update(self.engine, node, clock=1)
        self.assertEqual(self.state[self.counter_id], Vector(0))
        self.assertEqual(self.state[self.time_id], Vector(1))
        await self.counter.update(self.engine, node, clock=2)
        self.assertEqual(self.state[self.counter_id], Vector(-1))
        self.assertEqual(self.state[self.time_id], Vector(2))
        await self.counter.update(self.engine, node, clock=3)
        self.assertEqual(self.state[self.counter_id], Vector(-1))
        self.assertEqual(self.state[self.time_id], Vector(3))
        await self.counter.update(self.engine, node, clock=2)
        self.assertEqual(self.state[self.counter_id], Vector(0))
        self.assertEqual(self.state[self.time_id], Vector(2))

    async def test_maximum(self):
        node = Node('counter', set(), {'state': self.counter_id, 'rate': Vector(1), 'maximum': Vector(1)})
        await self.counter.update(self.engine, node, clock=1)
        self.assertEqual(self.state[self.counter_id], Vector(0))
        self.assertEqual(self.state[self.time_id], Vector(1))
        await self.counter.update(self.engine, node, clock=2)
        self.assertEqual(self.state[self.counter_id], Vector(1))
        self.assertEqual(self.state[self.time_id], Vector(2))
        await self.counter.update(self.engine, node, clock=3)
        self.assertEqual(self.state[self.counter_id], Vector(1))
        self.assertEqual(self.state[self.time_id], Vector(3))
        await self.counter.update(self.engine, node, clock=2)
        self.assertEqual(self.state[self.counter_id], Vector(0))
        self.assertEqual(self.state[self.time_id], Vector(2))

    async def test_minimum_maximum(self):
        node = Node('counter', set(), {'state': self.counter_id, 'rate': Vector(1), 'minimum': Vector(-1), 'maximum': Vector(1)})
        await self.counter.update(self.engine, node, clock=1)
        self.assertEqual(self.state[self.counter_id], Vector(0))
        self.assertEqual(self.state[self.time_id], Vector(1))
        await self.counter.update(self.engine, node, clock=2)
        self.assertEqual(self.state[self.counter_id], Vector(1))
        self.assertEqual(self.state[self.time_id], Vector(2))
        await self.counter.update(self.engine, node, clock=3)
        self.assertEqual(self.state[self.counter_id], Vector(1))
        self.assertEqual(self.state[self.time_id], Vector(3))
        await self.counter.update(self.engine, node, clock=2)
        self.assertEqual(self.state[self.counter_id], Vector(0))
        self.assertEqual(self.state[self.time_id], Vector(2))
        await self.counter.update(self.engine, node, clock=1)
        self.assertEqual(self.state[self.counter_id], Vector(-1))
        self.assertEqual(self.state[self.time_id], Vector(1))
        await self.counter.update(self.engine, node, clock=0)
        self.assertEqual(self.state[self.counter_id], Vector(-1))
        self.assertEqual(self.state[self.time_id], Vector(0))
        await self.counter.update(self.engine, node, clock=1)
        self.assertEqual(self.state[self.counter_id], Vector(0))
        self.assertEqual(self.state[self.time_id], Vector(1))
