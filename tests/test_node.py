"""
Tests of the model.Node class
"""

import unittest

from flitter.model import Node, Vector, Query


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

    def test_parent(self):
        self.node1.append(self.node2)
        self.assertIsNone(self.node1.parent)
        self.assertIs(self.node2.parent, self.node1)

    def test_copy(self):
        self.node1.append(self.node2)
        self.node2.append(self.node3)
        node = self.node1.copy()
        self.assertIsNot(node, self.node1)
        self.assertEqual(node, self.node1)

    def test_add_tag(self):
        self.node1.add_tag('test')
        self.assertEqual(self.node1.tags, {'test'})
        self.node2.add_tag('test')
        self.assertEqual(self.node2.tags, {'bar', 'baz', 'test'})

    def test_remove_tag(self):
        self.node1.remove_tag('test')
        self.assertEqual(self.node1.tags, set())
        self.node2.remove_tag('baz')
        self.assertEqual(self.node2.tags, {'bar'})

    def test_append(self):
        self.node1.append(self.node2)
        self.node2.append(self.node3)
        self.assertIs(self.node2.parent, self.node1)
        self.assertIs(self.node3.parent, self.node2)
        children = list(self.node1.children)
        self.assertEqual(len(children), 1)
        self.assertIs(children[0], self.node2)
        children = list(self.node2.children)
        self.assertEqual(len(children), 1)
        self.assertIs(children[0], self.node3)
        self.node1.append(self.node3)
        self.assertIs(self.node2.parent, self.node1)
        self.assertIs(self.node3.parent, self.node1)
        children = list(self.node1.children)
        self.assertEqual(len(children), 2)
        self.assertIs(children[0], self.node2)
        self.assertIs(children[1], self.node3)

    def test_insert(self):
        self.node1.insert(self.node2)
        self.node2.insert(self.node3)
        self.assertIs(self.node2.parent, self.node1)
        self.assertIs(self.node3.parent, self.node2)
        children = list(self.node1.children)
        self.assertEqual(len(children), 1)
        self.assertIs(children[0], self.node2)
        children = list(self.node2.children)
        self.assertEqual(len(children), 1)
        self.assertIs(children[0], self.node3)
        self.node1.insert(self.node3)
        self.assertIs(self.node2.parent, self.node1)
        self.assertIs(self.node3.parent, self.node1)
        children = list(self.node1.children)
        self.assertEqual(len(children), 2)
        self.assertIs(children[0], self.node3)
        self.assertIs(children[1], self.node2)

    def test_extend(self):
        self.node1.extend([self.node2, self.node3])
        self.assertIs(self.node2.parent, self.node1)
        self.assertIs(self.node3.parent, self.node1)
        children = list(self.node1.children)
        self.assertEqual(len(children), 2)
        self.assertIs(children[0], self.node2)
        self.assertIs(children[1], self.node3)

    def test_prepend(self):
        self.node1.prepend([self.node2, self.node3])
        self.assertIs(self.node2.parent, self.node1)
        self.assertIs(self.node3.parent, self.node1)
        children = list(self.node1.children)
        self.assertEqual(len(children), 2)
        self.assertIs(children[0], self.node2)
        self.assertIs(children[1], self.node3)
        self.node1.prepend([self.node4, self.node5])
        self.assertIs(self.node4.parent, self.node1)
        self.assertIs(self.node5.parent, self.node1)
        children = list(self.node1.children)
        self.assertEqual(len(children), 4)
        self.assertIs(children[0], self.node4)
        self.assertIs(children[1], self.node5)
        self.assertIs(children[2], self.node2)
        self.assertIs(children[3], self.node3)

    def test_remove(self):
        self.node1.extend([self.node2, self.node3, self.node4, self.node5])
        self.assertEqual(list(self.node1.children), [self.node2, self.node3, self.node4, self.node5])
        self.assertIs(self.node2.parent, self.node1)
        self.node1.remove(self.node2)
        self.assertIsNone(self.node2.parent)
        self.assertEqual(list(self.node1.children), [self.node3, self.node4, self.node5])
        self.node1.remove(self.node4)
        self.assertIsNone(self.node4.parent)
        self.assertEqual(list(self.node1.children), [self.node3, self.node5])
        self.node1.remove(self.node5)
        self.assertIsNone(self.node5.parent)
        self.assertEqual(list(self.node1.children), [self.node3])
        self.node1.remove(self.node3)
        self.assertIsNone(self.node3.parent)
        self.assertEqual(list(self.node1.children), [])
        self.node2.append(self.node3)
        self.assertRaises(ValueError, self.node1.remove, self.node2)
        self.assertRaises(ValueError, self.node1.remove, self.node3)

    def test_delete(self):
        self.node1.extend([self.node2, self.node3, self.node4, self.node5])
        self.assertEqual(list(self.node1.children), [self.node2, self.node3, self.node4, self.node5])
        self.assertIs(self.node2.parent, self.node1)
        self.node2.delete()
        self.assertIsNone(self.node2.parent)
        self.assertEqual(list(self.node1.children), [self.node3, self.node4, self.node5])
        self.node4.delete()
        self.assertIsNone(self.node4.parent)
        self.assertEqual(list(self.node1.children), [self.node3, self.node5])
        self.node5.delete()
        self.assertIsNone(self.node5.parent)
        self.assertEqual(list(self.node1.children), [self.node3])
        self.node3.delete()
        self.assertIsNone(self.node3.parent)
        self.assertEqual(list(self.node1.children), [])
        self.assertRaises(TypeError, self.node1.delete)

    def test_attributes(self):
        self.assertTrue('color' in self.node3)
        self.assertEqual(self.node3['color'], Vector(1))
        self.node3['shape'] = 'circle'
        shape = self.node3['shape']
        self.assertIsInstance(shape, Vector)
        self.assertEqual(shape, Vector('circle'))
        size = Vector([1, 2, 3])
        self.node3['size'] = size
        self.assertIs(self.node3['size'], size)
        self.assertEqual(set(self.node3), {'color', 'shape', 'size'})
        self.assertEqual(self.node3.keys(), {'color', 'shape', 'size'})
        self.assertEqual(list(self.node3.values()), [Vector(1), Vector('circle'), size])
        self.assertEqual(list(self.node3.items()), [('color', Vector(1)), ('shape', Vector('circle')), ('size', size)])
        del self.node3['size']
        self.assertFalse('size' in self.node3)
        with self.assertRaises(KeyError):
            self.node3['size']
        self.node3['shape'] = None
        self.assertFalse('shape' in self.node3)
        with self.assertRaises(KeyError):
            self.node3['shape']

    def test_get(self):
        self.node1['str1'] = 'circle'
        self.node1['str2'] = 'Hello ', 'world!'
        self.node1['float1'] = 1.5
        self.node1['float2'] = 0, 0
        self.node1['float3'] = 1, 2, 3.5
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

    def test_repr(self):
        self.assertEqual(repr(self.node1), '!node1')
        self.assertEqual(repr(self.node2), '!node2 #bar #baz')
        self.assertEqual(repr(self.node3), '!node3 #bar #baz color=1')
        self.node1.append(self.node2)
        self.node2.append(self.node3)
        self.node3.append(self.node4)
        self.node1.append(self.node5)
        self.assertEqual(repr(self.node1), """
!node1
 !node2 #bar #baz
  !node3 #bar #baz color=1
   !node4
 !node5
""".strip())


class TestQuery(unittest.TestCase):
    """
    Tests of executing queries on the Node class
    """

    def setUp(self):
        self.node1 = Node('node1')
        self.node2 = Node('node2', {'bar'})
        self.node1.append(self.node2)
        self.node3 = Node('node3', {'bar', 'baz'}, {'color': Vector(1)})
        self.node1.append(self.node3)
        self.node4 = Node('node4', {'bar'})
        self.node3.append(self.node4)
        self.node5 = Node('node5', {'baz'})
        self.node4.append(self.node5)
        self.group = Node('group')
        self.node2.append(self.group)
        for i in range(10):
            node = Node('multiple', {'grouped'})
            node['index'] = i
            node.append(Node('group'))
            self.group.append(node)

    def test_setup(self):
        """Self-testing documentation of the test node layout"""
        self.assertEqual(repr(self.node1), """
!node1
 !node2 #bar
  !group
   !multiple #grouped index=0
    !group
   !multiple #grouped index=1
    !group
   !multiple #grouped index=2
    !group
   !multiple #grouped index=3
    !group
   !multiple #grouped index=4
    !group
   !multiple #grouped index=5
    !group
   !multiple #grouped index=6
    !group
   !multiple #grouped index=7
    !group
   !multiple #grouped index=8
    !group
   !multiple #grouped index=9
    !group
 !node3 #bar #baz color=1
  !node4 #bar
   !node5 #baz
""".strip())

    def test_parse(self):
        self.assertEqual(str(Query('kind')), 'kind')
        self.assertEqual(str(Query('#tag')), '#tag')
        self.assertEqual(str(Query('#tag2#tag1')), '#tag1#tag2')
        self.assertEqual(str(Query('#tag kind')), '#tag kind')
        self.assertEqual(str(Query('#tag. kind')), '#tag. kind')
        self.assertEqual(str(Query('#tag kind!')), '#tag kind!')
        self.assertEqual(str(Query('#tag|kind1. kind2')), '#tag|kind1. kind2')
        self.assertEqual(str(Query('#tag.|kind1 kind2')), '#tag.|kind1 kind2')
        self.assertEqual(str(Query('#tag.|kind1. kind2')), '#tag.|kind1. kind2')
        self.assertEqual(str(Query('#tag>kind1|kind2')), '#tag > kind1|kind2')
        self.assertEqual(str(Query('#tag>kind1|kind2!')), '#tag > kind1|kind2!')

    def test_parse_errors(self):
        self.assertRaises(ValueError, Query, '123testing')
        self.assertRaises(ValueError, Query, 'testing 123')
        self.assertRaises(ValueError, Query, 'testing123 !')
        self.assertRaises(ValueError, Query, 'testing >')
        self.assertRaises(ValueError, Query, '#!foo')
        self.assertRaises(ValueError, Query, '#foo!|bar')

    def test_star(self):
        self.assertEqual(len(self.node1.select('*')), 26)
        self.assertEqual(len(self.node1.select_below('*')), 25)
        self.assertEqual(len(self.node2.select('*')), 22)
        self.assertEqual(len(self.group.select('*')), 21)

    def test_kind(self):
        nodes = self.node1.select('node1')
        self.assertEqual(len(nodes), 1)
        self.assertIs(nodes[0], self.node1)
        nodes = self.node1.select('multiple')
        self.assertEqual(len(nodes), 10)
        self.assertTrue(all(node.kind == 'multiple' and node['index'] == i for i, node in enumerate(nodes)))
        nodes = self.node1.select('group')
        self.assertEqual(len(nodes), 11)

    def test_tag(self):
        nodes = self.node1.select('#bar')
        self.assertEqual(len(nodes), 3)
        self.assertIs(nodes[0], self.node2)
        self.assertIs(nodes[1], self.node3)
        self.assertIs(nodes[2], self.node4)
        nodes = self.node3.select('#bar')
        self.assertEqual(len(nodes), 2)
        self.assertIs(nodes[0], self.node3)
        self.assertIs(nodes[1], self.node4)
        nodes = self.node3.select_below('#bar')
        self.assertEqual(len(nodes), 1)
        self.assertIs(nodes[0], self.node4)
        nodes = self.node1.select('#baz')
        self.assertEqual(len(nodes), 2)
        self.assertIs(nodes[0], self.node3)
        self.assertIs(nodes[1], self.node5)
        nodes = self.node1.select('#bar#baz')
        self.assertEqual(len(nodes), 1)
        self.assertIs(nodes[0], self.node3)
        nodes = self.node1.select('#baz#bar')
        self.assertEqual(len(nodes), 1)
        self.assertIs(nodes[0], self.node3)
        nodes = self.node1.select('#grouped')
        self.assertEqual(len(nodes), 10)
        self.assertTrue(all(node.kind == 'multiple' and node['index'] == i for i, node in enumerate(nodes)))

    # TODO: Figure out why this test doesn't work and fix select
    @unittest.expectedFailure
    def test_multiple_paths(self):
        nodes = self.node1.select('#bar. #baz')
        self.assertEqual(len(nodes), 2)
        self.assertIs(nodes[0], self.node3)
        self.assertIs(nodes[1], self.node5)

    def test_stop(self):
        nodes = self.node1.select('*.')
        self.assertEqual(len(nodes), 1)
        self.assertIs(nodes[0], self.node1)
        nodes = self.node1.select('group.')
        self.assertEqual(len(nodes), 1)
        self.assertIs(nodes[0], self.group)
        nodes = self.node1.select('#bar.')
        self.assertEqual(len(nodes), 2)
        self.assertIs(nodes[0], self.node2)
        self.assertIs(nodes[1], self.node3)

    def test_altquery(self):
        nodes = self.node1.select('node2|node3')
        self.assertEqual(len(nodes), 2)
        self.assertIs(nodes[0], self.node2)
        self.assertIs(nodes[1], self.node3)
        nodes = self.node1.select('#bar|node5')
        self.assertEqual(len(nodes), 4)
        self.assertIs(nodes[0], self.node2)
        self.assertIs(nodes[1], self.node3)
        self.assertIs(nodes[2], self.node4)
        self.assertIs(nodes[3], self.node5)

    def test_subquery(self):
        nodes = self.node1.select('* node3')
        self.assertEqual(len(nodes), 1)
        self.assertIs(nodes[0], self.node3)
        nodes = self.node1.select('node3 *')
        self.assertEqual(len(nodes), 2)
        self.assertIs(nodes[0], self.node4)
        self.assertIs(nodes[1], self.node5)
        nodes = self.node1.select('multiple group')
        self.assertEqual(len(nodes), 10)
        nodes = self.node1.select('group group')
        self.assertEqual(len(nodes), 10)
        nodes = self.node1.select('node3 #bar')
        self.assertEqual(len(nodes), 1)
        self.assertIs(nodes[0], self.node4)
        nodes = self.node1.select('#bar group')
        self.assertEqual(len(nodes), 11)
        nodes = self.node1.select('#bar group.')
        self.assertEqual(len(nodes), 1)

    def test_strict(self):
        nodes = self.node1.select('node3>*')
        self.assertEqual(len(nodes), 1)
        self.assertIs(nodes[0], self.node4)
        nodes = self.node1.select('node3 > *')
        self.assertEqual(len(nodes), 1)
        self.assertIs(nodes[0], self.node4)
        nodes = self.node1.select('multiple > group')
        self.assertEqual(len(nodes), 10)
        nodes = self.node1.select('group > group')
        self.assertEqual(len(nodes), 0)

    def test_first(self):
        nodes = self.node1.select('*!')
        self.assertEqual(len(nodes), 1)
        self.assertIs(nodes[0], self.node1)
        nodes = self.node1.select('group!')
        self.assertEqual(len(nodes), 1)
        self.assertIs(nodes[0], self.group)
        nodes = self.node1.select('#bar!')
        self.assertEqual(len(nodes), 1)
        self.assertIs(nodes[0], self.node2)
        nodes = self.node1.select('multiple!')
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0]['index'], 0)
        nodes = self.node1.select('group group!')
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].parent.kind, 'multiple')
        self.assertEqual(nodes[0].parent['index'], 0)
