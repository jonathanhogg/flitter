"""
Tests of the flitter language virtual machine

Note that `Import` is currently not tested.
"""

import unittest
import unittest.mock

from flitter.model import Vector, Node, Query, null, true, false
from flitter.language.vm import Program, Context, StateDict, Function


class TestBasicInstructions(unittest.TestCase):
    def setUp(self):
        self.program = Program()
        self.state = StateDict()
        self.variables = {}
        self.context = Context(state=self.state, variables=self.variables)

    def test_Add(self):
        self.program.literal(3)
        self.program.literal(4)
        self.program.add()
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [7])

    def test_Append(self):
        node = Node('foo', {'test'}, {'x': Vector(1)})
        self.program.literal(node)
        self.program.literal(Node('bar'))
        self.program.append()
        self.program.literal(Node('baz'))
        self.program.append()
        stack = self.program.execute(self.context)
        self.assertEqual(len(stack), 1)
        self.assertEqual(repr(stack[0]), "(!foo #test x=1\n !bar\n !baz)")

    def test_AppendRoot(self):
        node = Node('foo', {'test'}, {'x': Vector(1)})
        self.program.literal(node)
        self.program.append_root()
        stack = self.program.execute(self.context)
        self.assertEqual(len(stack), 0)
        self.assertEqual(next(self.context.graph.children), node)

    def test_AppendAttribute(self):
        node = Node('foo', {'test'}, {'x': Vector(1)})
        self.program.literal(node)
        self.program.literal(12)
        self.program.attribute('y')
        stack = self.program.execute(self.context)
        self.assertEqual(len(stack), 1)
        self.assertEqual(repr(stack[0]), "(!foo #test x=1 y=12)")

    def test_Compose(self):
        self.program.literal(3)
        self.program.literal(4)
        self.program.compose(2)
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [Vector([3, 4])])

    def test_Drop(self):
        self.program.literal(3)
        self.program.literal(4)
        self.program.drop()
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [3])

    def test_Dup(self):
        self.program.literal(3)
        self.program.literal(4)
        self.program.dup()
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [3, 4, 4])

    def test_Eq(self):
        self.program.literal(3)
        self.program.literal(4)
        self.program.eq()
        self.program.literal(5)
        self.program.literal(5)
        self.program.eq()
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [false, true])

    def test_FloorDiv(self):
        self.program.literal(11)
        self.program.literal(4)
        self.program.floordiv()
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [2])

    def test_Ge(self):
        self.program.literal(3)
        self.program.literal(4)
        self.program.ge()
        self.program.literal(3)
        self.program.literal(3)
        self.program.ge()
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [false, true])

    def test_Gt(self):
        self.program.literal(3)
        self.program.literal(3)
        self.program.gt()
        self.program.literal(4)
        self.program.literal(3)
        self.program.gt()
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [false, true])

    def test_IndexLiteral(self):
        self.program.literal(range(1, 11))
        self.program.slice_literal(Vector(3))
        self.assertEqual(str(self.program.instructions[-1]), 'IndexLiteral 3')
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [4])

    def test_LiteralNode(self):
        self.program.literal(Node('foo', {'test'}, {'x': Vector(1)}))
        self.assertEqual(str(self.program.instructions[-1]), 'LiteralNode (!foo #test x=1)')

    def test_Literal(self):
        self.program.literal(range(10))
        self.assertEqual(str(self.program.instructions[-1]), 'Literal 0;1;2;3;4;5;6;7;8;9')

    def test_Le(self):
        self.program.literal(3)
        self.program.literal(3)
        self.program.le()
        self.program.literal(4)
        self.program.literal(3)
        self.program.le()
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [true, false])

    def test_Let(self):
        self.program.literal((1, 2))
        self.program.let(('x',))
        self.program.literal((3, 4))
        self.program.let(('y', 'z'))
        stack = self.program.execute(self.context)
        self.assertEqual(len(stack), 0)
        self.assertEqual(self.variables, {'x': Vector([1, 2]), 'y': Vector(3), 'z': Vector(4)})

    def test_Lookup(self):
        self.state['y'] = 12
        self.program.literal('x')
        self.program.lookup()
        self.program.literal('y')
        self.program.lookup()
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [null, 12])

    def test_Lt(self):
        self.program.literal(3)
        self.program.literal(3)
        self.program.lt()
        self.program.literal(3)
        self.program.literal(4)
        self.program.lt()
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [false, true])

    def test_Mod(self):
        self.program.literal(7)
        self.program.literal(4)
        self.program.mod()
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [3])

    def test_Mul(self):
        self.program.literal(3)
        self.program.literal(4)
        self.program.mul()
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [12])

    def test_Name(self):
        self.variables['y'] = 12
        self.program.name('x')
        self.program.name('y')
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [null, 12])

    def test_Ne(self):
        self.program.literal(3)
        self.program.literal(4)
        self.program.ne()
        self.program.literal(5)
        self.program.literal(5)
        self.program.ne()
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [true, false])

    def test_Neg(self):
        self.program.literal(3)
        self.program.neg()
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [-3])

    def test_Not(self):
        self.program.literal(true)
        self.program.not_()
        self.program.literal(false)
        self.program.not_()
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [false, true])

    def test_Pos(self):
        self.program.literal(3)
        self.program.pos()
        self.program.literal('hello')
        self.program.pos()
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [3, null])

    def test_Pow(self):
        self.program.literal(3)
        self.program.literal(4)
        self.program.pow()
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [81])

    def test_Pragma(self):
        self.program.literal(3)
        self.program.pragma('x')
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [null])
        self.assertEqual(self.context.pragmas, {'x': Vector(3)})

    def test_Prepend(self):
        node = Node('foo', {'test'}, {'x': Vector(1)})
        self.program.literal(node)
        self.program.literal(Node('bar'))
        self.program.prepend()
        self.program.literal(Node('baz'))
        self.program.prepend()
        stack = self.program.execute(self.context)
        self.assertEqual(len(stack), 1)
        self.assertEqual(repr(stack[0]), "(!foo #test x=1\n !baz\n !bar)")

    def test_Range(self):
        self.program.literal(0)
        self.program.literal(10)
        self.program.literal(2)
        self.program.range()
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [(0, 2, 4, 6, 8)])

    def test_Search(self):
        node = Node('foo', {'test'}, {'x': Vector(1)})
        self.context.graph.append(node)
        self.program.search(Query('#test'))
        stack = self.program.execute(self.context)
        self.assertEqual(len(stack), 1)
        self.assertIs(stack[0][0], node)

    def test_Slice(self):
        self.program.literal(range(1, 11))
        self.program.literal((0, 2))
        self.program.slice()
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [(1, 3)])

    def test_SliceLiteral(self):
        self.program.literal(range(1, 11))
        self.program.slice_literal(Vector([0, 2]))
        self.assertEqual(str(self.program.instructions[-1]), 'SliceLiteral 0;2')
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [(1, 3)])

    def test_Sub(self):
        self.program.literal(3)
        self.program.literal(4)
        self.program.sub()
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [-1])

    def test_Tag(self):
        node = Node('foo', None, {'x': Vector(1)})
        self.program.literal(node)
        self.program.tag('test')
        stack = self.program.execute(self.context)
        self.assertEqual(len(stack), 1)
        self.assertEqual(repr(stack[0]), "(!foo #test x=1)")

    def test_TrueDiv(self):
        self.program.literal(3)
        self.program.literal(4)
        self.program.truediv()
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [0.75])

    def test_Xor(self):
        self.program.literal(false)
        self.program.literal(false)
        self.program.xor()
        self.program.literal(true)
        self.program.literal(false)
        self.program.xor()
        self.program.literal(false)
        self.program.literal(true)
        self.program.xor()
        self.program.literal(true)
        self.program.literal(true)
        self.program.xor()
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [false, true, true, false])


class TestJumps(unittest.TestCase):
    def setUp(self):
        self.program = Program()
        self.state = StateDict()
        self.variables = {}
        self.context = Context(state=self.state, variables=self.variables)

    def test_jump(self):
        LABEL = self.program.new_label()
        self.program.literal(1)
        self.program.jump(LABEL)
        self.program.literal(2)
        self.program.label(LABEL)
        self.program.literal(3)
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [1, 3])

    def test_branch_true(self):
        LABEL1 = self.program.new_label()
        LABEL2 = self.program.new_label()
        self.program.literal(1)
        self.program.literal(true)
        self.program.branch_true(LABEL1)
        self.program.literal(2)
        self.program.label(LABEL1)
        self.program.literal(3)
        self.program.literal(false)
        self.program.branch_true(LABEL2)
        self.program.literal(4)
        self.program.label(LABEL2)
        self.program.literal(5)
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [1, 3, 4, 5])

    def test_branch_false(self):
        LABEL1 = self.program.new_label()
        LABEL2 = self.program.new_label()
        self.program.literal(1)
        self.program.literal(true)
        self.program.branch_false(LABEL1)
        self.program.literal(2)
        self.program.label(LABEL1)
        self.program.literal(3)
        self.program.literal(false)
        self.program.branch_false(LABEL2)
        self.program.literal(4)
        self.program.label(LABEL2)
        self.program.literal(5)
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [1, 2, 3, 5])


class TestForLoops(unittest.TestCase):
    def setUp(self):
        self.program = Program()
        self.state = StateDict()
        self.variables = {}
        self.context = Context(state=self.state, variables=self.variables)

    def test_simple(self):
        NEXT = self.program.new_label()
        END = self.program.new_label()
        self.program.literal(Vector.range(10))
        self.program.begin_for()
        self.program.label(NEXT)
        self.program.next(('i',), END)
        self.program.name('i')
        self.program.literal(2)
        self.program.mul()
        self.program.jump(NEXT)
        self.program.label(END)
        self.program.end_for()
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [Vector.range(0, 20, 2)])

    def test_multiple_names(self):
        NEXT = self.program.new_label()
        END = self.program.new_label()
        self.program.literal(Vector.range(10))
        self.program.begin_for()
        self.program.label(NEXT)
        self.program.next(('i', 'j'), END)
        self.program.name('i')
        self.program.literal(2)
        self.program.mul()
        self.program.name('j')
        self.program.compose(2)
        self.program.jump(NEXT)
        self.program.label(END)
        self.program.end_for()
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [(0, 1, 4, 3, 8, 5, 12, 7, 16, 9)])

    def test_short(self):
        NEXT = self.program.new_label()
        END = self.program.new_label()
        self.program.literal(Vector.range(9))
        self.program.begin_for()
        self.program.label(NEXT)
        self.program.next(('i', 'j'), END)
        self.program.name('i')
        self.program.literal(2)
        self.program.mul()
        self.program.name('j')
        self.program.compose(2)
        self.program.jump(NEXT)
        self.program.label(END)
        self.program.end_for()
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [(0, 1, 4, 3, 8, 5, 12, 7, 16)])

    def test_empty(self):
        NEXT = self.program.new_label()
        END = self.program.new_label()
        self.program.literal(null)
        self.program.begin_for()
        self.program.label(NEXT)
        self.program.next(('i',), END)
        self.program.name('i')
        self.program.literal(2)
        self.program.mul()
        self.program.jump(NEXT)
        self.program.label(END)
        self.program.end_for()
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [null])

    def test_next_push(self):
        NEXT = self.program.new_label()
        END = self.program.new_label()
        self.program.literal(Vector.range(10))
        self.program.begin_for()
        self.program.label(NEXT)
        self.program.push_next(END)
        self.program.literal(2)
        self.program.mul()
        self.program.jump(NEXT)
        self.program.label(END)
        self.program.end_for()
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [Vector.range(0, 20, 2)])


class TestScopes(unittest.TestCase):
    def setUp(self):
        self.program = Program()
        self.state = StateDict()
        self.variables = {}
        self.context = Context(state=self.state, variables=self.variables)

    def test_scoped_let(self):
        self.program.begin_scope()
        self.program.literal(3)
        self.program.let(('x',))
        self.program.name('x')
        self.program.end_scope()
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [3])
        self.assertEqual(self.variables, {})

    def test_shadowed_name(self):
        self.variables['x'] = Vector(12)
        self.program.begin_scope()
        self.program.literal(3)
        self.program.let(('x',))
        self.program.name('x')
        self.program.end_scope()
        self.program.name('x')
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [3, 12])
        self.assertEqual(self.variables, {'x': 12})

    def test_nested_scopes(self):
        self.program.begin_scope()
        self.program.literal(12)
        self.program.let(('x',))
        self.program.begin_scope()
        self.program.literal(3)
        self.program.let(('x',))
        self.program.name('x')
        self.program.end_scope()
        self.program.name('x')
        self.program.end_scope()
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [3, 12])
        self.assertEqual(self.variables, {})

    def test_simple_node_scope(self):
        node = Node('foo', {'test'}, {'x': Vector(1)})
        self.program.literal(node)
        self.program.set_node_scope()
        self.program.name('x')
        self.program.clear_node_scope()
        self.program.name('x')
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [node, 1, null])

    def test_shadowed_node_scope(self):
        self.variables['x'] = Vector(12)
        node = Node('foo', {'test'}, {'x': Vector(1)})
        self.program.literal(node)
        self.program.set_node_scope()
        self.program.name('x')
        self.program.clear_node_scope()
        self.program.name('x')
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [node, 12, 12])


class TestFunc(unittest.TestCase):
    def setUp(self):
        self.program = Program()
        self.state = StateDict()
        self.variables = {}
        self.context = Context(state=self.state, variables=self.variables)
        self.func_program = Program()
        self.func_program.name('x')
        self.func_program.name('y')
        self.func_program.add()
        self.program.literal(self.func_program)
        self.program.literal(null)
        self.program.literal(1)
        self.program.func(('f', 'x', 'y'))
        self.program.let(('f',))

    def test_declare(self):
        stack = self.program.execute(self.context)
        self.assertEqual(len(stack), 0)
        function, = self.variables['f']
        self.assertTrue(isinstance(function, Function))
        self.assertEqual(function.__name__, 'f')
        self.assertEqual(function.parameters, ('x', 'y'))
        self.assertEqual(function.defaults, (null, 1))
        self.assertEqual(function.program.instructions, self.func_program.instructions)

    def test_call(self):
        self.program.literal(3)
        self.program.literal(4)
        self.program.name('f')
        self.program.call(2)
        stack = self.program.execute(self.context)
        self.assertEqual(len(stack), 1)
        self.assertEqual(stack, [7])

    def test_default_arg(self):
        self.program.literal(3)
        self.program.name('f')
        self.program.call(1)
        stack = self.program.execute(self.context)
        self.assertEqual(len(stack), 1)
        self.assertEqual(stack, [4])

    def test_default_arg2(self):
        self.program.name('f')
        self.program.call(0)
        stack = self.program.execute(self.context)
        self.assertEqual(len(stack), 1)
        self.assertEqual(stack, [null])


class TestCalls(unittest.TestCase):
    def setUp(self):
        self.program = Program()
        self.state = StateDict()
        self.test_function = unittest.mock.Mock(state_transformer=False)
        self.state_function = unittest.mock.Mock(state_transformer=True)
        self.variables = {'test': Vector(self.test_function), 'state': Vector(self.state_function)}
        self.context = Context(state=self.state, variables=self.variables)

    def test_no_args(self):
        self.test_function.return_value = Vector(12)
        self.program.name('test')
        self.program.call(0)
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [12])
        self.test_function.assert_called_once_with()

    def test_one_arg(self):
        self.test_function.return_value = Vector(12)
        self.program.literal(1)
        self.program.name('test')
        self.program.call(1)
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [12])
        self.test_function.assert_called_once_with(Vector(1))

    def test_multiple_args(self):
        self.test_function.return_value = Vector(12)
        self.program.literal(1)
        self.program.literal(2)
        self.program.name('test')
        self.program.call(2)
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [12])
        self.test_function.assert_called_once_with(Vector(1), Vector(2))

    def test_kwargs(self):
        self.test_function.return_value = Vector(12)
        self.program.literal(1)
        self.program.literal(2)
        self.program.name('test')
        self.program.call(0, ('x', 'y'))
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [12])
        self.test_function.assert_called_once_with(x=Vector(1), y=Vector(2))

    def test_args_kwargs(self):
        self.test_function.return_value = Vector(12)
        self.program.literal(1)
        self.program.literal(2)
        self.program.literal(3)
        self.program.name('test')
        self.program.call(1, ('x', 'y'))
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [12])
        self.test_function.assert_called_once_with(Vector(1), x=Vector(2), y=Vector(3))
