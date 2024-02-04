"""
Tests of the model.Node class
"""

import unittest

from flitter.model import Node, Vector, null

from test_vector import FOO_SYMBOL_NUMBER


class TestNode(unittest.TestCase):
    """
    Tests of the Node class
    """

    def setUp(self):
        self.node1 = Node('node1')
        self.node2 = Node('node2', {'bar', 'baz'})
        self.node3 = Node('node3', {'bar', 'baz'}, {'color': Vector(1)})
        self.node4 = Node('node4')
        self.node5 = Node('node5')

    def test_construct(self):
        self.assertEqual(self.node1.kind, 'node1')
        self.assertEqual(self.node1.tags, set())
        self.assertEqual(len(self.node1), 0)
        self.assertEqual(list(self.node1.items()), [])
        self.assertEqual(list(self.node1.children), [])
        self.assertEqual(self.node2.kind, 'node2')
        self.assertEqual(self.node2.tags, {'bar', 'baz'})
        self.assertEqual(len(self.node2), 0)
        self.assertEqual(list(self.node2.items()), [])
        self.assertEqual(list(self.node2.children), [])
        self.assertEqual(self.node3.kind, 'node3')
        self.assertEqual(self.node3.tags, {'bar', 'baz'})
        self.assertEqual(len(self.node3), 1)
        self.assertEqual(list(self.node3.items()), [('color', Vector(1))])
        self.assertEqual(list(self.node3.children), [])

    def test_children(self):
        self.node1.append(self.node2)
        self.node1.append(self.node3)
        children = self.node1.children
        self.assertIs(next(children), self.node2)
        self.assertIs(next(children), self.node3)
        self.assertRaises(StopIteration, next, children)

    def test_copy(self):
        self.node2.append(self.node3)
        self.node1.append(self.node2)
        node1c = self.node1.copy()
        self.assertIsNot(node1c, self.node1)
        self.assertEqual(node1c, self.node1)
        node2c, = node1c.children
        node3c, = node2c.children
        self.assertIs(node2c, self.node2)
        self.assertIs(node3c, self.node3)

    def test_add_tag(self):
        self.node1.add_tag('test')
        self.assertEqual(self.node1.tags, {'test'})
        self.node2.add_tag('test')
        self.assertEqual(self.node2.tags, {'bar', 'baz', 'test'})

    def test_append(self):
        self.node2.append(self.node3)
        self.node1.append(self.node2)
        self.node1.append(self.node3)
        children = list(self.node1.children)
        self.assertEqual(len(children), 2)
        self.assertIs(children[0], self.node2)
        self.assertIs(children[1], self.node3)
        children = list(self.node2.children)
        self.assertEqual(len(children), 1)
        self.assertIs(children[0], self.node3)

    def test_attributes(self):
        self.assertTrue('color' in self.node3)
        self.assertEqual(self.node3['color'], Vector(1))
        self.node3.set_attribute('shape', Vector('circle'))
        shape = self.node3['shape']
        self.assertIsInstance(shape, Vector)
        self.assertEqual(shape, Vector('circle'))
        size = Vector([1, 2, 3])
        self.node3.set_attribute('size', size)
        self.assertIs(self.node3['size'], size)
        self.assertEqual(set(self.node3), {'color', 'shape', 'size'})
        self.assertEqual(self.node3.keys(), {'color', 'shape', 'size'})
        self.assertEqual(list(self.node3.values()), [Vector(1), Vector('circle'), size])
        self.assertEqual(list(self.node3.items()), [('color', Vector(1)), ('shape', Vector('circle')), ('size', size)])
        self.node3.set_attribute('size', null)
        self.assertFalse('size' in self.node3)
        with self.assertRaises(KeyError):
            self.node3['size']

    def test_get(self):
        self.node1.set_attribute('str1', Vector('circle'))
        self.node1.set_attribute('str2', Vector(('Hello ', 'world!')))
        self.node1.set_attribute('float1', Vector(1.5))
        self.node1.set_attribute('float2', Vector((0, 0)))
        self.node1.set_attribute('float3', Vector((1, 2, 3.5)))
        self.node1.set_attribute('symbol', Vector.symbol('foo'))
        self.assertEqual(self.node1.get('missing'), None)
        self.assertEqual(self.node1.get('missing', 0, float), None)
        self.assertEqual(self.node1.get('missing', 1, str), None)
        self.assertEqual(self.node1.get('missing', default=5), 5)
        self.assertEqual(self.node1.get('str1'), ['circle'])
        self.assertEqual(self.node1.get('str1', 1), 'circle')
        self.assertEqual(self.node1.get('str1', 1), 'circle')
        self.assertEqual(self.node1.get('str2', 1), None)
        self.assertEqual(self.node1.get('str2', 1, str), 'Hello world!')
        self.assertEqual(self.node1.get('str2', 1, float), None)
        self.assertEqual(self.node1.get('str2', 2, float), None)
        self.assertEqual(self.node1.get('float1', 0, float), [1.5])
        self.assertEqual(self.node1.get('float1', 1, float), 1.5)
        self.assertEqual(self.node1.get('float1', 3, float), [1.5, 1.5, 1.5])
        self.assertEqual(self.node1.get('float3', 0, float), [1, 2, 3.5])
        self.assertEqual(self.node1.get('float3', 1, float), None)
        self.assertEqual(self.node1.get('float3', 3, float), [1, 2, 3.5])
        self.assertEqual(self.node1.get('float3', 3, int), [1, 2, 3])
        self.assertEqual(self.node1.get('str1', 1, bool), True)
        self.assertEqual(self.node1.get('str2', 1, bool), True)
        self.assertEqual(self.node1.get('str2', 2, bool), [True, True])
        self.assertEqual(self.node1.get('str2', 3, bool), None)
        self.assertEqual(self.node1.get('float3', 1, bool), True)
        self.assertEqual(self.node1.get('float2', 1, bool), False)
        self.assertEqual(self.node1.get('float2', 2, bool), [False, False])
        self.assertEqual(self.node1.get('float2', 3, bool), None)
        self.assertEqual(self.node1.get('symbol', 1, str), 'foo')
        self.assertEqual(self.node1.get('symbol', 1, float), FOO_SYMBOL_NUMBER)

    def test_repr(self):
        self.assertEqual(repr(self.node1), '!node1')
        self.assertEqual(repr(self.node2), '!node2 #bar #baz')
        self.assertEqual(repr(self.node3), '!node3 #bar #baz color=1')
        self.node3.append(self.node4)
        self.node2.append(self.node3)
        self.node1.append(self.node2)
        self.node1.append(self.node5)
        self.node1.set_attribute('foo', Vector.symbol('foo'))
        self.assertEqual(repr(self.node1), """
!node1 foo=:foo
 !node2 #bar #baz
  !node3 #bar #baz color=1
   !node4
 !node5
""".strip())
