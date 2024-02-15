"""
Tests of the flitter language virtual machine

Note that `Import` is currently not tested.
"""

import gc
import tracemalloc
import unittest
import unittest.mock

from flitter.model import Vector, Node, Context, StateDict, null, true, false
from flitter.language.vm import Program, Function, VectorStack


class TestBasicInstructions(unittest.TestCase):
    def setUp(self):
        self.program = Program()
        self.state = StateDict()
        self.names = {}
        self.context = Context(state=self.state, names=self.names)

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
        self.program.literal(Vector([Node('baz1', None, {'a': Vector(1)}), Node('baz2', None, {'b': Vector(2)})]))
        self.program.literal(Vector(Node('baz3', None, {'c': Vector(3)})))
        self.program.append(2)
        stack = self.program.execute(self.context)
        self.assertEqual(len(stack), 1)
        self.assertEqual(repr(stack[0]), "(!foo #test x=1\n !bar\n !baz1 a=1\n !baz2 b=2\n !baz3 c=3)")

    def test_Attribute(self):
        foo = Node('foo', {'test'}, {'x': Vector(1)})
        bar = Node('bar', None, {'x': Vector(2)})
        self.names['bar'] = Vector(bar)
        self.program.literal(foo)
        self.program.literal(bar)
        self.program.local_push(1)
        self.program.literal(12)
        self.program.attribute('y')
        self.program.literal(5)
        self.program.attribute('x')
        self.program.literal(null)
        self.program.attribute('y')
        self.program.local_load(0)
        self.program.literal(7)
        self.program.attribute('y')
        stack = self.program.execute(self.context)
        self.assertEqual(len(stack), 2)
        self.assertEqual(repr(stack[0]), "(!foo #test x=5)")
        self.assertFalse(stack[0][0] is foo)
        self.assertEqual(repr(stack[1]), "(!bar x=2 y=7)")
        self.assertFalse(stack[1][0] is bar)
        self.assertEqual(foo['x'], Vector(1))

    def test_Compose(self):
        self.program.literal(3)
        self.program.literal(4)
        self.program.compose(2)
        self.program.literal(Vector(['a', 'b']))
        self.program.literal(Vector('c'))
        self.program.compose(2)
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [Vector([3, 4]), Vector(['a', 'b', 'c'])])

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

    def test_Ceil(self):
        self.program.literal(-3.5)
        self.program.ceil()
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [-3])

    def test_Floor(self):
        self.program.literal(-3.5)
        self.program.floor()
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [-4])

    def test_FloorDiv(self):
        self.program.literal(11)
        self.program.literal(4)
        self.program.floordiv()
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [2])

    def test_Fract(self):
        self.program.literal(-3.5)
        self.program.fract()
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [0.5])

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
        self.assertEqual(str(self.program.instructions[-1]), 'LiteralNode !foo #test x=1')

    def test_LiteralNodes(self):
        self.program.literal([Node('foo', {'test'}, {'x': Vector(1)}), Node('bar')])
        self.assertEqual(str(self.program.instructions[-1]), 'LiteralNodes (!foo #test x=1, !bar)')

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

    def test_StoreGlobal(self):
        self.program.literal((1, 2))
        self.program.store_global('x')
        self.program.literal(3)
        self.program.store_global('y')
        self.program.literal("Hello world!")
        self.program.store_global('z')
        stack = self.program.execute(self.context)
        self.assertEqual(len(stack), 0)
        self.assertEqual(self.names, {'x': Vector([1, 2]), 'y': Vector(3), 'z': Vector("Hello world!")})

    def test_Lookup(self):
        self.state['y'] = 12
        self.program.literal('x')
        self.program.lookup()
        self.program.literal('y')
        self.program.lookup()
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [null, 12])

    def test_LookupLiteral(self):
        self.state['y'] = 12
        self.program.lookup_literal(Vector('x'))
        self.program.lookup_literal(Vector('y'))
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
        self.assertEqual(len(stack), 0)
        self.assertEqual(self.context.pragmas, {'x': Vector(3)})

    def test_Range(self):
        self.program.literal(0)
        self.program.literal(10)
        self.program.literal(2)
        self.program.range()
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [(0, 2, 4, 6, 8)])

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
        self.names = {}
        self.context = Context(state=self.state, names=self.names)

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
        self.names = {}
        self.context = Context(state=self.state, names=self.names)

    def test_simple(self):
        NEXT = self.program.new_label()
        END = self.program.new_label()
        self.program.literal(Vector.range(10))
        self.program.begin_for()
        self.program.literal(null)
        self.program.local_push(1)
        self.program.label(NEXT)
        self.program.next(1, END)
        self.program.local_load(0)
        self.program.literal(2)
        self.program.mul()
        self.program.jump(NEXT)
        self.program.label(END)
        self.program.end_for_compose()
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [Vector.range(0, 20, 2)])

    def test_multiple_names(self):
        NEXT = self.program.new_label()
        END = self.program.new_label()
        self.program.literal(Vector.range(10))
        self.program.begin_for()
        self.program.literal(null)
        self.program.local_push(2)
        self.program.label(NEXT)
        self.program.next(2, END)
        self.program.local_load(1)
        self.program.literal(2)
        self.program.mul()
        self.program.local_load(0)
        self.program.compose(2)
        self.program.jump(NEXT)
        self.program.label(END)
        self.program.end_for_compose()
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [(0, 1, 4, 3, 8, 5, 12, 7, 16, 9)])

    def test_short(self):
        NEXT = self.program.new_label()
        END = self.program.new_label()
        self.program.literal(Vector.range(9))
        self.program.begin_for()
        self.program.literal(null)
        self.program.local_push(2)
        self.program.label(NEXT)
        self.program.next(2, END)
        self.program.local_load(1)
        self.program.literal(2)
        self.program.mul()
        self.program.local_load(0)
        self.program.compose(2)
        self.program.jump(NEXT)
        self.program.label(END)
        self.program.end_for_compose()
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [(0, 1, 4, 3, 8, 5, 12, 7, 16)])

    def test_empty(self):
        NEXT = self.program.new_label()
        END = self.program.new_label()
        self.program.literal(null)
        self.program.begin_for()
        self.program.literal(null)
        self.program.local_push(1)
        self.program.label(NEXT)
        self.program.next(1, END)
        self.program.local_load(0)
        self.program.literal(2)
        self.program.mul()
        self.program.jump(NEXT)
        self.program.label(END)
        self.program.end_for_compose()
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [null])


class TestLocalVars(unittest.TestCase):
    def setUp(self):
        self.program = Program()
        self.state = StateDict()
        self.names = {}
        self.context = Context(state=self.state, names=self.names)

    def test_simple(self):
        self.program.literal(5)
        self.program.local_push(1)
        self.program.literal(2)
        self.program.local_load(0)
        self.program.add()
        self.program.local_drop(1)
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [7])

    def test_shadow(self):
        self.program.literal(5)
        self.program.local_push(1)
        self.program.literal(2)
        self.program.local_load(0)
        self.program.add()
        self.program.local_push(1)
        self.program.local_load(0)
        self.program.dup()
        self.program.local_drop(1)
        self.program.local_load(0)
        self.program.add()
        self.program.local_drop(1)
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [7, 12])

    def test_multiple(self):
        self.program.literal([1, 2, 3])
        self.program.local_push(2)
        self.program.local_load(0)
        self.program.dup()
        self.program.local_load(1)
        self.program.add()
        self.program.local_drop(2)
        self.program.literal([1, 2])
        self.program.local_push(3)
        self.program.local_load(0)
        self.program.local_drop(3)
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [2, 3, 1])


class TestFunc(unittest.TestCase):
    def setUp(self):
        self.program = Program()
        self.state = StateDict()
        self.names = {}
        self.context = Context(state=self.state, names=self.names)
        self.program.literal(2)
        self.program.local_push(1)
        self.program.local_load(0)
        self.program.literal(null)
        self.program.literal(1)
        END = self.program.new_label()
        self.program.jump(END)
        START = self.program.new_label()
        self.program.label(START)
        self.program.local_load(1)
        self.program.local_load(0)
        self.program.add()
        self.program.local_load(3)
        self.program.add()
        LABEL1 = self.program.new_label()
        self.program.literal('test')
        self.program.lookup()
        self.program.branch_false(LABEL1)
        self.program.literal(1)
        self.program.add()
        self.program.label(LABEL1)
        self.program.exit()
        self.program.label(END)
        self.program.func(START, 'f', ('x', 'y'), 1)
        self.program.local_push(1)

    def test_declare(self):
        lnames = []
        stack = self.program.execute(self.context, lnames=lnames)
        self.assertEqual(len(stack), 0)
        self.assertEqual(len(lnames), 2)
        self.assertEqual(lnames[0], 2)
        function, = lnames[1]
        self.assertTrue(isinstance(function, Function))
        self.assertEqual(function.__name__, 'f')
        self.assertEqual(function.parameters, ('x', 'y'))
        self.assertEqual(function.defaults, (null, 1))
        self.assertIs(function.program, self.program)

    def test_call(self):
        self.program.literal(3)
        self.program.literal(4)
        self.program.local_load(0)
        self.program.call(2)
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [9])

    def test_additional_lnames(self):
        self.program.literal(-2)
        self.program.local_push(1)
        self.program.literal(3)
        self.program.literal(4)
        self.program.local_load(1)
        self.program.call(2)
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [9])

    def test_default_arg(self):
        self.program.literal(3)
        self.program.local_load(0)
        self.program.call(1)
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [6])

    def test_default_arg2(self):
        self.program.local_load(0)
        self.program.call(0)
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [null])

    def test_state(self):
        self.state['test'] = Vector(1)
        self.program.literal(3)
        self.program.literal(4)
        self.program.local_load(0)
        self.program.call(2)
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [10])


class TestRecursiveFunc(unittest.TestCase):
    def test_recursive_fib(self):
        self.program = Program()
        self.state = StateDict()
        self.names = {}
        self.context = Context(state=self.state, names=self.names)
        self.program.literal(null)
        END = self.program.new_label()
        self.program.jump(END)
        START = self.program.new_label()
        self.program.label(START)
        self.program.local_load(0)
        self.program.literal(2)
        self.program.lt()
        LABEL2 = self.program.new_label()
        self.program.branch_false(LABEL2)
        self.program.local_load(0)
        LABEL1 = self.program.new_label()
        self.program.jump(LABEL1)
        self.program.label(LABEL2)
        self.program.local_load(0)
        self.program.literal(1)
        self.program.sub()
        self.program.local_load(1)
        self.program.call(1)
        self.program.local_load(0)
        self.program.literal(2)
        self.program.sub()
        self.program.local_load(1)
        self.program.call(1)
        self.program.add()
        self.program.label(LABEL1)
        self.program.exit()
        self.program.label(END)
        self.program.func(START, 'fib', ('x',))
        self.program.local_push(1)
        self.program.literal(10)
        self.program.local_load(0)
        self.program.call(1)
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [55])


class TestCalls(unittest.TestCase):
    def setUp(self):
        self.program = Program()
        self.state = StateDict()
        self.test_function = unittest.mock.Mock()
        del self.test_function.context_func
        self.test_function.return_value = Vector(12)
        self.context_function = unittest.mock.Mock(context_func=True)
        self.context_function.return_value = Vector(12)
        self.names = {'test': Vector(self.test_function), 'context': Vector(self.context_function)}
        self.context = Context(state=self.state, names=self.names)

    def test_no_args(self):
        self.program.local_load(0)
        self.program.call(0)
        stack = self.program.execute(self.context, lnames=[self.test_function])
        self.assertEqual(stack, [12])
        self.test_function.assert_called_once_with()

    def test_one_arg(self):
        self.program.literal(1)
        self.program.local_load(0)
        self.program.call(1)
        stack = self.program.execute(self.context, lnames=[self.test_function])
        self.assertEqual(stack, [12])
        self.test_function.assert_called_once_with(Vector(1))

    def test_multiple_args(self):
        self.program.literal(1)
        self.program.literal(2)
        self.program.local_load(0)
        self.program.call(2)
        stack = self.program.execute(self.context, lnames=[self.test_function])
        self.assertEqual(stack, [12])
        self.test_function.assert_called_once_with(Vector(1), Vector(2))

    def test_kwargs(self):
        self.program.literal(1)
        self.program.literal(2)
        self.program.local_load(0)
        self.program.call(0, ('x', 'y'))
        stack = self.program.execute(self.context, lnames=[self.test_function])
        self.assertEqual(stack, [12])
        self.test_function.assert_called_once_with(x=Vector(1), y=Vector(2))

    def test_args_kwargs(self):
        self.program.literal(1)
        self.program.literal(2)
        self.program.literal(3)
        self.program.local_load(0)
        self.program.call(1, ('x', 'y'))
        stack = self.program.execute(self.context, lnames=[self.test_function])
        self.assertEqual(stack, [12])
        self.test_function.assert_called_once_with(Vector(1), x=Vector(2), y=Vector(3))

    def test_context_func(self):
        self.program.literal(1)
        self.program.literal(2)
        self.program.literal(3)
        self.program.local_load(0)
        self.program.call(1, ('x', 'y'))
        stack = self.program.execute(self.context, lnames=[self.context_function])
        self.assertEqual(stack, [12])
        self.context_function.assert_called_once_with(self.context, Vector(1), x=Vector(2), y=Vector(3))

    def test_call_fast_multiple_args(self):
        self.program.literal(1)
        self.program.literal(2)
        self.program.call_fast(self.test_function, 2)
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [12])
        self.test_function.assert_called_once_with(Vector(1), Vector(2))

    def test_call_fast_single_arg(self):
        self.program.literal(1)
        self.program.call_fast(self.test_function, 1)
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [12])
        self.test_function.assert_called_once_with(Vector(1))

    def test_call_fast_no_args(self):
        self.program.call_fast(self.test_function, 0)
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [12])
        self.test_function.assert_called_once_with()


class TestStack(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tracemalloc.start()

    @classmethod
    def tearDownClass(cls):
        tracemalloc.stop()

    def setUp(self):
        gc.collect()
        gc.disable()

    def tearDown(self):
        for obj in gc.get_objects(0):
            self.assertNotIsInstance(obj, Vector, "Memory leak")
        gc.enable()

    def test_create(self):
        stack = VectorStack(10)
        self.assertEqual(stack.size, 10)
        self.assertEqual(len(stack), 0)

    def test_basic_push_and_pop(self):
        stack = VectorStack()
        stack.push(Vector(0))
        self.assertEqual(len(stack), 1)
        stack.push(Vector(1))
        self.assertEqual(len(stack), 2)
        self.assertEqual(stack.pop(), Vector(1))
        self.assertEqual(len(stack), 1)
        self.assertEqual(stack.pop(), Vector(0))
        self.assertEqual(len(stack), 0)

    def test_drop(self):
        stack = VectorStack()
        stack.push(Vector(0))
        stack.push(Vector(1))
        stack.push(Vector(2))
        stack.push(Vector(3))
        self.assertEqual(len(stack), 4)
        stack.drop()
        self.assertEqual(len(stack), 3)
        stack.drop(2)
        self.assertEqual(len(stack), 1)
        self.assertEqual(stack.pop(), Vector(0))
        self.assertEqual(len(stack), 0)

    def test_realloc(self):
        stack = VectorStack(1)
        stack.push(Vector(0))
        self.assertEqual(stack.size, 1)
        self.assertEqual(len(stack), 1)
        stack.push(Vector(1))
        self.assertEqual(stack.size, 2)
        self.assertEqual(len(stack), 2)
        stack.push(Vector(2))
        self.assertEqual(stack.size, 4)
        self.assertEqual(len(stack), 3)
        stack.push(Vector(3))
        self.assertEqual(stack.size, 4)
        self.assertEqual(len(stack), 4)
        stack.push(Vector(4))
        self.assertEqual(stack.size, 8)
        self.assertEqual(len(stack), 5)

    def test_peek_and_poke(self):
        stack = VectorStack()
        stack.push(Vector(0))
        self.assertEqual(len(stack), 1)
        self.assertEqual(stack.peek(), Vector(0))
        stack.poke(Vector(1))
        self.assertEqual(len(stack), 1)
        self.assertEqual(stack.pop(), Vector(1))
        self.assertEqual(len(stack), 0)

    def test_peek_at_and_poke_at(self):
        stack = VectorStack()
        stack.push(Vector(0))
        stack.push(Vector(1))
        stack.push(Vector(2))
        self.assertEqual(len(stack), 3)
        self.assertEqual(stack.peek_at(0), Vector(2))
        self.assertEqual(stack.peek_at(1), Vector(1))
        self.assertEqual(stack.peek_at(2), Vector(0))
        stack.poke_at(0, Vector(3))
        stack.poke_at(1, Vector(4))
        stack.poke_at(2, Vector(5))
        self.assertEqual(len(stack), 3)
        self.assertEqual(stack.pop(), Vector(3))
        self.assertEqual(stack.pop(), Vector(4))
        self.assertEqual(stack.pop(), Vector(5))
        self.assertEqual(len(stack), 0)

    def test_pop_tuple(self):
        stack = VectorStack()
        stack.push(Vector(0))
        stack.push(Vector(1))
        stack.push(Vector(2))
        self.assertEqual(len(stack), 3)
        self.assertEqual(stack.pop_tuple(2), (Vector(1), Vector(2)))
        self.assertEqual(len(stack), 1)
        self.assertEqual(stack.pop(), Vector(0))
        self.assertEqual(len(stack), 0)

    def test_pop_list(self):
        stack = VectorStack()
        stack.push(Vector(0))
        stack.push(Vector(1))
        stack.push(Vector(2))
        self.assertEqual(len(stack), 3)
        self.assertEqual(stack.pop_list(2), [Vector(1), Vector(2)])
        self.assertEqual(len(stack), 1)
        self.assertEqual(stack.pop(), Vector(0))
        self.assertEqual(len(stack), 0)

    def test_pop_dict(self):
        stack = VectorStack()
        stack.push(Vector(0))
        stack.push(Vector(1))
        stack.push(Vector(2))
        self.assertEqual(len(stack), 3)
        self.assertEqual(stack.pop_dict(('x', 'y')), {'x': Vector(1), 'y': Vector(2)})
        self.assertEqual(len(stack), 1)
        self.assertEqual(stack.pop(), Vector(0))
        self.assertEqual(len(stack), 0)

    def test_pop_composed(self):
        stack = VectorStack()
        stack.push(Vector(0))
        stack.push(Vector(1))
        stack.push(Vector(2))
        self.assertEqual(len(stack), 3)
        self.assertEqual(stack.pop_composed(2), Vector([1, 2]))
        self.assertEqual(len(stack), 1)
        self.assertEqual(stack.pop(), Vector(0))
        self.assertEqual(len(stack), 0)
