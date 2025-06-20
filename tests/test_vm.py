"""
Tests of the flitter language virtual machine
"""

import gc
import tempfile
import tracemalloc
from pathlib import Path
import unittest
import unittest.mock

from flitter import configure_logger
from flitter.model import Vector, Node, Context, StateDict, null, true, false
from flitter.language.tree import Top, Literal
from flitter.language.vm import Program, Function, VectorStack, StackUnderflow, StackOverflow, log_vm_stats


configure_logger('ERROR')


class TestProgramMethods(unittest.TestCase):
    def test_run(self):
        program = Program(lnames=('a',))
        program.literal(Node('foo'))
        program.local_load(0)
        program.attributes(('a',))
        program.append()
        program.link()
        context = program.run(Context(state=StateDict(), names={'a': Vector(5)}))
        self.assertEqual(context.root, Node('root', children=(Node('foo', attributes={'a': Vector(5)}),)))

    def test_new_label(self):
        program = Program()
        self.assertEqual(program.new_label(), 1)
        self.assertEqual(program.new_label(), 2)
        self.assertEqual(program.new_label(), 3)

    def test_set_path(self):
        program = Program()
        self.assertIsNone(program.path)
        program.set_path(Path('a/b/c'))
        self.assertEqual(program.path, Path('a/b/c'))

    def test_set_top(self):
        program = Program()
        self.assertIsNone(program.top)
        top = Top((), Literal(null))
        program.set_top(top)
        self.assertIs(program.top, top)

    def test_set_pragma(self):
        program = Program()
        self.assertEqual(program.pragmas, {})
        program.set_pragma('tempo', Vector(60))
        self.assertEqual(program.pragmas, {'tempo': Vector(60)})

    def test_use_simplifier(self):
        program = Program()
        self.assertTrue(program.simplify)
        program.use_simplifier(False)
        self.assertFalse(program.simplify)

    @unittest.mock.patch('flitter.language.vm.logger')
    def test_log_vm_stats(self, logger_mock):
        program = Program()
        context = Context(state=StateDict(), path=Path('.'))
        program.literal(1)
        program.literal(2)
        program.add()
        program.local_load(0)
        program.call(1)
        program.call_fast(lambda x: x+1, 1)
        program.link()
        stack = program.execute(context, lnames=[lambda x: x+1], record_stats=True)
        self.assertEqual(stack, [5])
        log_vm_stats()
        self.assertEqual(logger_mock.info.mock_calls[0], unittest.mock.call('VM execution statistics:'))
        self.assertEqual(set(call.args[1] for call in logger_mock.info.mock_calls[1:]), {'(native funcs)', 'Add', 'Literal', 'LocalLoad', 'Call', 'CallFast'})


class TestBasicInstructions(unittest.TestCase):
    def setUp(self):
        self.program = Program()
        self.state = StateDict()
        self.names = {}
        self.exports = {}
        self.context = Context(state=self.state, names=self.names, exports=self.exports)

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
        self.program.literal([1, Node('foo')])
        self.program.literal([Node('bar'), 2])
        self.program.append()
        self.program.literal(Node('baz'))
        self.program.literal(['a', 'b'])
        self.program.append()
        stack = self.program.execute(self.context)
        self.assertEqual(len(stack), 3)
        self.assertEqual(repr(stack[0]), "(!foo #test x=1\n !bar\n !baz1 a=1\n !baz2 b=2\n !baz3 c=3)")
        self.assertEqual(stack[1], [1, Node('foo', children=(Node('bar'),))])
        self.assertEqual(stack[2], [Node('baz')])

    def test_Attribute(self):
        foo = Node('foo', {'test'}, {'x': Vector(1)})
        bar = Node('bar', None, {'x': Vector(2)})
        self.names['bar'] = Vector(bar)
        self.program.literal(foo)
        self.program.literal(bar)
        self.program.local_push(1)
        self.program.literal(5)
        self.program.literal(12)
        self.program.attributes(('x', 'y',))
        self.program.literal(null)
        self.program.attributes(('y',))
        self.program.local_load(0)
        self.program.literal(7)
        self.program.attributes(('y',))
        self.program.literal([0, 1, 2])
        self.program.literal(5)
        self.program.attributes(('x',))
        self.program.literal(['a', 'b', 'c'])
        self.program.literal(10)
        self.program.attributes(('x',))
        stack = self.program.execute(self.context)
        self.assertEqual(len(stack), 4)
        self.assertEqual(repr(stack[0]), "(!foo #test x=5)")
        self.assertFalse(stack[0][0] is foo)
        self.assertEqual(repr(stack[1]), "(!bar x=2 y=7)")
        self.assertFalse(stack[1][0] is bar)
        self.assertEqual(foo['x'], Vector(1))
        self.assertEqual(stack[2], [0, 1, 2])
        self.assertEqual(stack[3], ['a', 'b', 'c'])

    def test_LiteralNode_sharing(self):
        foo = Vector([Node('foo', attributes={'y': Vector(10)})])
        bar = Vector([Node('bar'), Node('baz')])
        self.program.literal(foo)
        self.program.literal(foo)
        self.program.literal(bar)
        self.program.append()
        self.program.literal(5)
        self.program.attributes(('x',))
        stack = self.program.execute(self.context)
        self.assertEqual(len(stack), 2)
        self.assertIs(stack[0][0], foo[0])
        self.assertEqual(repr(stack[1]), "(!foo y=10 x=5\n !bar\n !baz)")
        self.assertIsNot(stack[1][0], foo[0])
        children = tuple(stack[1][0].children)
        self.assertIs(children[0], bar[0])
        self.assertIs(children[1], bar[1])
        self.assertEqual(repr(foo), "(!foo y=10)")
        self.assertEqual(repr(bar), "(!bar);(!baz)")

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

    def test_Export(self):
        self.program.literal((1, 2))
        self.program.export('x')
        self.program.literal(3)
        self.program.export('y')
        self.program.literal("Hello world!")
        self.program.export('z')
        stack = self.program.execute(self.context)
        self.assertEqual(len(stack), 0)
        self.assertEqual(self.exports, {'x': Vector([1, 2]), 'y': Vector(3), 'z': Vector("Hello world!")})

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

    def test_MulAdd(self):
        self.program.literal(3)
        self.program.literal(4)
        self.program.literal(5)
        self.program.mul_add()
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [23])

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
        self.program.literal(Node('foo'))
        self.program.tag('bar')
        self.program.literal(5)
        self.program.attributes(('x',))
        self.program.tag('baz')
        self.program.literal([0, 1, 2])
        self.program.tag('test')
        self.program.literal(['a', 'b', 'c'])
        self.program.tag('test')
        stack = self.program.execute(self.context)
        self.assertEqual(len(stack), 3)
        self.assertEqual(repr(stack[0]), "(!foo #bar #baz x=5)")
        self.assertEqual(stack[1], [0, 1, 2])
        self.assertEqual(stack[2], ['a', 'b', 'c'])

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
        self.program.begin_for(1)
        self.program.label(NEXT)
        self.program.next(END)
        self.program.local_load(0)
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
        self.program.begin_for(2)
        self.program.label(NEXT)
        self.program.next(END)
        self.program.local_load(1)
        self.program.literal(2)
        self.program.mul()
        self.program.local_load(0)
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
        self.program.begin_for(2)
        self.program.label(NEXT)
        self.program.next(END)
        self.program.local_load(1)
        self.program.literal(2)
        self.program.mul()
        self.program.local_load(0)
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
        self.program.begin_for(1)
        self.program.label(NEXT)
        self.program.next(END)
        self.program.local_load(0)
        self.program.literal(2)
        self.program.mul()
        self.program.jump(NEXT)
        self.program.label(END)
        self.program.end_for()
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [null])

    def test_nested(self):
        NEXT_OUTER = self.program.new_label()
        END_OUTER = self.program.new_label()
        NEXT_INNER = self.program.new_label()
        END_INNER = self.program.new_label()
        self.program.literal(Vector.range(0, 10, 5))
        self.program.begin_for(1)
        self.program.label(NEXT_OUTER)
        self.program.next(END_OUTER)
        self.program.literal(Vector.range(5))
        self.program.begin_for(1)
        self.program.label(NEXT_INNER)
        self.program.next(END_INNER)
        self.program.local_load(1)
        self.program.local_load(0)
        self.program.add()
        self.program.jump(NEXT_INNER)
        self.program.label(END_INNER)
        self.program.end_for()
        self.program.jump(NEXT_OUTER)
        self.program.label(END_OUTER)
        self.program.end_for()
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [Vector.range(10)])


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
        self.program.truediv()
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
        self.assertEqual(stack, [2.75])

    def test_call_kwarg(self):
        self.program.literal(4)
        self.program.literal(3)
        self.program.local_load(0)
        self.program.call(0, ('y', 'x'))
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [2.75])

    def test_additional_lnames(self):
        self.program.literal(-2)
        self.program.local_push(1)
        self.program.literal(3)
        self.program.literal(4)
        self.program.local_load(1)
        self.program.call(2)
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [2.75])

    def test_default_arg(self):
        self.program.literal(3)
        self.program.local_load(0)
        self.program.call(1)
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [5])

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
        self.assertEqual(stack, [3.75])


class TestRecursiveFunc(unittest.TestCase):
    def test_recursive_fib(self):
        program = Program()
        state = StateDict()
        names = {}
        context = Context(state=state, names=names)
        program.literal(null)
        END = program.new_label()
        program.jump(END)
        START = program.new_label()
        program.label(START)
        program.local_load(0)
        program.literal(2)
        program.lt()
        LABEL2 = program.new_label()
        program.branch_false(LABEL2)
        program.local_load(0)
        LABEL1 = program.new_label()
        program.jump(LABEL1)
        program.label(LABEL2)
        program.local_load(0)
        program.literal(1)
        program.sub()
        program.local_load(1)
        program.call(1)
        program.local_load(0)
        program.literal(2)
        program.sub()
        program.local_load(1)
        program.call(1)
        program.add()
        program.label(LABEL1)
        program.exit()
        program.label(END)
        program.func(START, 'fib', ('x',))
        program.local_push(1)
        program.literal(10)
        program.local_load(0)
        program.call(1)
        stack = program.execute(context)
        self.assertEqual(stack, [55])

    def test_infinite_recursion(self):
        program = Program()
        state = StateDict()
        names = {}
        context = Context(state=state, names=names)
        program.literal(null)
        END = program.new_label()
        program.jump(END)
        START = program.new_label()
        program.label(START)
        program.local_load(0)
        program.literal(1)
        program.add()
        program.local_load(1)
        program.call(1)
        program.exit()
        program.label(END)
        program.func(START, 'f', ('x',))
        program.local_push(1)
        program.literal(0)
        program.local_load(0)
        program.call(1)
        stack = program.execute(context)
        self.assertEqual(stack, [null])
        self.assertEqual(context.errors, {"Recursion depth exceeded for func 'f'"})


class TestCalls(unittest.TestCase):
    def setUp(self):
        self.program = Program()
        self.state = StateDict()
        self.test_function = unittest.mock.Mock()
        del self.test_function.context_func
        self.test_function.__name__ = 'test'
        self.test_function.return_value = Vector(12)
        self.context_function = unittest.mock.Mock(context_func=True)
        self.context_function.return_value = Vector(13)
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
        self.assertEqual(stack, [13])
        self.context_function.assert_called_once_with(self.context, Vector(1), x=Vector(2), y=Vector(3))

    def test_not_a_func(self):
        self.program.literal(1)
        self.program.literal(2)
        self.program.literal(null)
        self.program.call(2)
        stack = self.program.execute(self.context, lnames=[self.test_function, self.context_function])
        self.assertEqual(stack, [null])

    def test_multiple_funcs(self):
        self.program.literal(1)
        self.program.literal(2)
        self.program.local_load(1)
        self.program.local_load(0)
        self.program.compose(2)
        self.program.call(2)
        stack = self.program.execute(self.context, lnames=[self.test_function, self.context_function])
        self.assertEqual(stack, [(12, 13)])
        self.test_function.assert_called_once_with(Vector(1), Vector(2))
        self.context_function.assert_called_once_with(self.context, Vector(1), Vector(2))

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

    def test_call_fast_multiple_args_exception(self):
        self.test_function.side_effect = ValueError("Test error")
        self.program.literal(1)
        self.program.literal(2)
        self.program.call_fast(self.test_function, 2)
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [null])
        self.assertEqual(self.context.errors, {'Error calling test: Test error'})

    def test_call_fast_single_arg_exception(self):
        self.test_function.side_effect = ValueError("Test error")
        self.program.literal(1)
        self.program.call_fast(self.test_function, 1)
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [null])
        self.assertEqual(self.context.errors, {'Error calling test: Test error'})


class TestImports(unittest.TestCase):
    def setUp(self):
        self.program = Program()
        self.context = Context(state=StateDict(), path=Path('.'))
        self.test_module = Path(tempfile.mktemp('.fl'))
        self.test_module.write_text("""
let x=5 y=10
""", encoding='utf8')
        self.circular_module_a = Path(tempfile.mktemp('.fl'))
        self.circular_module_b = Path(tempfile.mktemp('.fl'))
        module_a = f"""
import y from {str(self.circular_module_b)!r}
let x=5 + y
"""
        module_b = f"""
import x from {str(self.circular_module_a)!r}
let y=10 + x
"""
        self.circular_module_a.write_text(module_a, encoding='utf8')
        self.circular_module_b.write_text(module_b, encoding='utf8')

    def tearDown(self):
        if self.test_module.exists():
            self.test_module.unlink()
        if self.circular_module_a.exists():
            self.circular_module_a.unlink()
        if self.circular_module_b.exists():
            self.circular_module_b.unlink()

    def test_import_one_name(self):
        self.program.literal(str(self.test_module))
        self.program.import_(('x',))
        self.program.local_load(0)
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [5])

    def test_import_two_names(self):
        self.program.literal(str(self.test_module))
        self.program.import_(('x', 'y'))
        self.program.local_load(1)
        self.program.local_load(0)
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [5, 10])

    def test_import_twice(self):
        self.program.literal(str(self.test_module))
        self.program.import_(('x',))
        self.program.local_load(0)
        self.program.literal(str(self.test_module))
        self.program.import_(('y',))
        self.program.local_load(0)
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [5, 10])

    def test_import_bad_name(self):
        self.program.literal(str(self.test_module))
        self.program.import_(('z',))
        self.program.local_load(0)
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [null])
        self.assertEqual(self.context.errors, {f"Unable to import 'z' from '{self.test_module}'"})

    def test_import_missing_module(self):
        self.program.literal('this/is/not/a/module.fl')
        self.program.import_(('x',))
        self.program.local_load(0)
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [null])
        self.assertEqual(self.context.errors, {"Unable to import from 'this/is/not/a/module.fl'"})

    def test_circular_import(self):
        self.program.literal(str(self.circular_module_a))
        self.program.import_(('x', 'y'))
        self.program.local_load(1)
        self.program.local_load(0)
        stack = self.program.execute(self.context)
        self.assertEqual(stack, [null, null])
        self.assertEqual(self.context.errors, {f"Circular import of '{self.circular_module_a}'"})


class TestStackExhaustion(unittest.TestCase):
    def test_for_loop_push(self):
        program = Program()
        NEXT = program.new_label()
        END = program.new_label()
        program.literal(null)
        program.literal(1000)
        program.literal(null)
        program.range()
        program.begin_for(1)
        program.label(NEXT)
        program.next(END)
        program.local_load(0)
        program.jump(NEXT)
        program.label(END)
        program.end_for()
        context = Context(state=StateDict(), stack=VectorStack(max_size=1000))
        stack = program.execute(context)
        self.assertEqual(len(stack), 1)
        self.assertEqual(stack[0], Vector.range(1000))
        context = Context(state=StateDict(), stack=VectorStack(max_size=999))
        with self.assertRaises(StackOverflow):
            program.execute(context)

    def test_too_many_locals(self):
        program = Program()
        for i in range(1000):
            program.literal(i)
            program.local_push(1)
        context = Context(state=StateDict(), lnames=VectorStack(max_size=1000))
        lnames = []
        stack = program.execute(context, lnames=lnames)
        self.assertEqual(len(stack), 0)
        self.assertEqual(len(lnames), 1000)
        context = Context(state=StateDict(), lnames=VectorStack(max_size=999))
        with self.assertRaises(StackOverflow):
            program.execute(context)

    def test_run_for_loop_push(self):
        program = Program()
        NEXT = program.new_label()
        END = program.new_label()
        program.literal(null)
        program.literal(1000)
        program.literal(null)
        program.range()
        program.begin_for(1)
        program.label(NEXT)
        program.next(END)
        program.local_load(0)
        program.jump(NEXT)
        program.label(END)
        program.end_for()
        program.attributes(('x',))
        program.link()
        context = Context(state=StateDict(), root=Node('root'))
        program.run(context)
        self.assertEqual(context.errors, set())
        self.assertEqual(context.root['x'], Vector.range(1000))
        context = Context(state=StateDict(), root=Node('root'), stack=VectorStack(max_size=999))
        program.run(context)
        self.assertEqual(context.errors, {"Stack overflow in program"})
        self.assertEqual(context.root, Node('root'))


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
        self._vector_count = 0
        for obj in gc.get_objects(0):
            if isinstance(obj, Vector):
                self._vector_count += 1

    def tearDown(self):
        vector_count = 0
        for obj in gc.get_objects(0):
            if isinstance(obj, Vector):
                vector_count += 1
        self.assertEqual(vector_count, self._vector_count, "Memory leak")
        gc.enable()

    def test_create(self):
        stack = VectorStack(10)
        self.assertEqual(stack.size, 10)
        self.assertEqual(len(stack), 0)

    def test_basic_push_and_pop(self):
        stack = VectorStack(1, max_size=2)
        stack.push(Vector(0))
        self.assertEqual(len(stack), 1)
        stack.push(Vector(1))
        self.assertEqual(len(stack), 2)
        with self.assertRaises(StackOverflow):
            stack.push(Vector(2))
        self.assertEqual(stack.pop(), Vector(1))
        self.assertEqual(len(stack), 1)
        self.assertEqual(stack.pop(), Vector(0))
        self.assertEqual(len(stack), 0)
        with self.assertRaises(StackUnderflow):
            stack.pop()

    def test_copy(self):
        stack = VectorStack()
        stack.push(Vector(0))
        stack.push(Vector(1))
        copy = stack.copy()
        self.assertIsNot(stack, copy)
        self.assertEqual(len(stack), len(copy))
        self.assertEqual(stack.pop_list(len(stack)), copy.pop_list(len(copy)))

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
        with self.assertRaises(StackUnderflow):
            stack.drop()

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
        with self.assertRaises(StackUnderflow):
            stack.peek()
        with self.assertRaises(StackUnderflow):
            stack.poke(Vector(1))

    def test_peek_at_and_poke_at(self):
        stack = VectorStack()
        stack.push(Vector(0))
        stack.push(Vector(1))
        stack.push(Vector(2))
        self.assertEqual(len(stack), 3)
        self.assertEqual(stack.peek_at(0), Vector(2))
        self.assertEqual(stack.peek_at(1), Vector(1))
        self.assertEqual(stack.peek_at(2), Vector(0))
        with self.assertRaises(StackUnderflow):
            stack.peek_at(3)
        stack.poke_at(0, Vector(3))
        stack.poke_at(1, Vector(4))
        stack.poke_at(2, Vector(5))
        with self.assertRaises(StackUnderflow):
            stack.poke_at(3, Vector(6))
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
        with self.assertRaises(StackUnderflow):
            stack.pop_tuple(4)
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
        with self.assertRaises(StackUnderflow):
            stack.pop_list(4)
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
        with self.assertRaises(StackUnderflow):
            stack.pop_dict(('w', 'x', 'y', 'z'))
        self.assertEqual(stack.pop_dict(('x', 'y')), {'x': Vector(1), 'y': Vector(2)})
        self.assertEqual(len(stack), 1)
        self.assertEqual(stack.pop(), Vector(0))
        self.assertEqual(len(stack), 0)

    def test_pop_composed(self):
        stack = VectorStack()
        self.assertEqual(stack.pop_composed(0), null)
        stack.push(Vector(0))
        stack.push(Vector(1))
        stack.push(Vector(2))
        stack.push(null)
        self.assertEqual(len(stack), 4)
        with self.assertRaises(StackUnderflow):
            stack.pop_composed(5)
        self.assertEqual(stack.pop_composed(3), Vector([1, 2]))
        self.assertEqual(len(stack), 1)
        self.assertEqual(stack.pop_composed(1), Vector(0))
        self.assertEqual(len(stack), 0)
        stack.push(null)
        stack.push(null)
        self.assertEqual(stack.pop_composed(2), null)
        stack.push(Vector(1))
        stack.push(Vector('a'))
        self.assertEqual(stack.pop_composed(2), Vector([1, 'a']))
