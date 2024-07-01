"""
Tests of the compiler, which is part of the language AST
"""

import unittest

from flitter.model import Vector, Node, null
from flitter.language import functions
from flitter.language.tree import (Literal, Name, Sequence,
                                   Positive, Negative, Ceil, Floor, Fract, Power,
                                   Add, Subtract, Multiply, Divide, FloorDivide, Modulo,
                                   EqualTo, NotEqualTo, LessThan, GreaterThan, LessThanOrEqualTo, GreaterThanOrEqualTo,
                                   Not, And, Or, Xor, Range, Slice, Lookup,
                                   Tag, Attributes, Append,
                                   InlineLet, Call, For, IfElse,
                                   Pragma, Import, Let, Function, StoreGlobal, Top,
                                   Binding, PolyBinding, IfCondition)
from flitter.language.vm import Program


class CompilerTestCase(unittest.TestCase):
    def assertCompilesTo(self, expr, program, lnames=(), with_errors=None):
        program.link()
        compiled = expr.compile(initial_lnames=lnames, log_errors=False)
        self.assertEqual(str(compiled), str(program))
        self.assertEqual(compiled.compiler_errors, set() if with_errors is None else with_errors)


class TestLiteral(CompilerTestCase):
    def test_null(self):
        self.assertCompilesTo(Literal(null), Program().literal(null))

    def test_vector(self):
        self.assertCompilesTo(Literal([1, 2, 3]), Program().literal([1, 2, 3]))

    def test_builtin(self):
        self.assertCompilesTo(Literal(functions.sqrtv), Program().literal(functions.sqrtv))


class TestName(CompilerTestCase):
    def test_undefined(self):
        self.assertCompilesTo(Name('z'), Program().literal(null), with_errors={"Unbound name 'z'"})

    def test_lname(self):
        self.assertCompilesTo(Name('x'), Program().local_load(0), lnames=('x',))
        self.assertCompilesTo(Name('x'), Program().local_load(1), lnames=('x', 'y'))

    def test_static_builtin(self):
        self.assertCompilesTo(Name('null'), Program().literal(null))
        self.assertCompilesTo(Name('sqrt'), Program().literal(functions.sqrtv))

    def test_dynamic_builtin(self):
        self.assertCompilesTo(Name('debug'), Program().literal(functions.debug))


class TestSequence(CompilerTestCase):
    def test_empty(self):
        self.assertCompilesTo(Sequence(()), Program())

    def test_single(self):
        self.assertCompilesTo(Sequence((Name('x'),)), Program().local_load(0), lnames=('x',))

    def test_multiple(self):
        self.assertCompilesTo(Sequence((Name('x'), Name('y'))), Program().local_load(1).local_load(0).compose(2), lnames=('x', 'y'))


class TestUnaryExpressions(CompilerTestCase):
    def test_positive(self):
        self.assertCompilesTo(Positive(Name('x')), Program().local_load(0).pos(), lnames=('x',))

    def test_negative(self):
        self.assertCompilesTo(Negative(Name('x')), Program().local_load(0).neg(), lnames=('x',))

    def test_ceiling(self):
        self.assertCompilesTo(Ceil(Name('x')), Program().local_load(0).ceil(), lnames=('x',))

    def test_floor(self):
        self.assertCompilesTo(Floor(Name('x')), Program().local_load(0).floor(), lnames=('x',))

    def test_fract(self):
        self.assertCompilesTo(Fract(Name('x')), Program().local_load(0).fract(), lnames=('x',))

    def test_not(self):
        self.assertCompilesTo(Not(Name('x')), Program().local_load(0).not_(), lnames=('x',))


class TestBinaryExpressions(CompilerTestCase):
    def test_add(self):
        self.assertCompilesTo(Add(Name('x'), Name('y')), Program().local_load(1).local_load(0).add(), lnames=('x', 'y'))

    def test_subtract(self):
        self.assertCompilesTo(Subtract(Name('x'), Name('y')), Program().local_load(1).local_load(0).sub(), lnames=('x', 'y'))

    def test_multiply(self):
        self.assertCompilesTo(Multiply(Name('x'), Name('y')), Program().local_load(1).local_load(0).mul(), lnames=('x', 'y'))

    def test_divide(self):
        self.assertCompilesTo(Divide(Name('x'), Name('y')), Program().local_load(1).local_load(0).truediv(), lnames=('x', 'y'))

    def test_floor_divide(self):
        self.assertCompilesTo(FloorDivide(Name('x'), Name('y')), Program().local_load(1).local_load(0).floordiv(), lnames=('x', 'y'))

    def test_modulo(self):
        self.assertCompilesTo(Modulo(Name('x'), Name('y')), Program().local_load(1).local_load(0).mod(), lnames=('x', 'y'))

    def test_power(self):
        self.assertCompilesTo(Power(Name('x'), Name('y')), Program().local_load(1).local_load(0).pow(), lnames=('x', 'y'))

    def test_equal_to(self):
        self.assertCompilesTo(EqualTo(Name('x'), Name('y')), Program().local_load(1).local_load(0).eq(), lnames=('x', 'y'))

    def test_not_equal_to(self):
        self.assertCompilesTo(NotEqualTo(Name('x'), Name('y')), Program().local_load(1).local_load(0).ne(), lnames=('x', 'y'))

    def test_less_than(self):
        self.assertCompilesTo(LessThan(Name('x'), Name('y')), Program().local_load(1).local_load(0).lt(), lnames=('x', 'y'))

    def test_greater_than(self):
        self.assertCompilesTo(GreaterThan(Name('x'), Name('y')), Program().local_load(1).local_load(0).gt(), lnames=('x', 'y'))

    def test_less_than_or_equal_to(self):
        self.assertCompilesTo(LessThanOrEqualTo(Name('x'), Name('y')), Program().local_load(1).local_load(0).le(), lnames=('x', 'y'))

    def test_greater_than_or_equal_to(self):
        self.assertCompilesTo(GreaterThanOrEqualTo(Name('x'), Name('y')), Program().local_load(1).local_load(0).ge(), lnames=('x', 'y'))

    def test_xor(self):
        self.assertCompilesTo(Xor(Name('x'), Name('y')), Program().local_load(1).local_load(0).xor(), lnames=('x', 'y'))


class TestShortcutLogic(CompilerTestCase):
    def test_and(self):
        program = Program()
        END = program.new_label()
        program.local_load(1)
        program.dup()
        program.branch_false(END)
        program.drop()
        program.local_load(0)
        program.label(END)
        self.assertCompilesTo(And(Name('x'), Name('y')), program, lnames=('x', 'y'))

    def test_or(self):
        program = Program()
        END = program.new_label()
        program.local_load(1)
        program.dup()
        program.branch_true(END)
        program.drop()
        program.local_load(0)
        program.label(END)
        self.assertCompilesTo(Or(Name('x'), Name('y')), program, lnames=('x', 'y'))


class TestTernaryExpressions(CompilerTestCase):
    def test_range(self):
        self.assertCompilesTo(Range(Name('x'), Name('y'), Name('z')), Program().local_load(2).local_load(1).local_load(0).range(), lnames=('x', 'y', 'z'))


class TestSlice(CompilerTestCase):
    def test_dynamic(self):
        self.assertCompilesTo(Slice(Name('x'), Name('y')), Program().local_load(1).local_load(0).slice(), lnames=('x', 'y'))

    def test_literal(self):
        self.assertCompilesTo(Slice(Name('x'), Literal(0)), Program().local_load(0).slice_literal(Vector(0)), lnames=('x',))


class TestLookup(CompilerTestCase):
    def test_dynamic(self):
        self.assertCompilesTo(Lookup(Name('x')), Program().local_load(0).lookup(), lnames=('x',))

    def test_literal(self):
        self.assertCompilesTo(Lookup(Literal(Vector.symbol('x'))), Program().lookup_literal(Vector.symbol('x')))


class TestNodeOperations(CompilerTestCase):
    def test_tag(self):
        self.assertCompilesTo(Tag(Name('x'), 'y'), Program().local_load(0).tag('y'), lnames=('x',))

    def test_single_attribute(self):
        self.assertCompilesTo(Attributes(Name('x'), (Binding('y', Name('y')),)), Program().local_load(1).local_load(0).attribute('y'), lnames=('x', 'y'))

    def test_multiple_attributes(self):
        self.assertCompilesTo(Attributes(Name('x'), (Binding('y', Name('y')), Binding('z', Name('z')))),
                              Program().local_load(2).local_load(1).attribute('y').local_load(0).attribute('z'), lnames=('x', 'y', 'z'))

    def test_append(self):
        self.assertCompilesTo(Append(Name('x'), Name('y')), Program().local_load(1).local_load(0).append(), lnames=('x', 'y'))


class TestInlineLet(CompilerTestCase):
    def test_single(self):
        self.assertCompilesTo(InlineLet(Name('x'), (PolyBinding(('x',), Literal(5)),)),
                              Program().literal(5).local_push(1).local_load(0).local_drop(1))

    def test_multiple(self):
        self.assertCompilesTo(InlineLet(Add(Name('x'), Name('y')), (PolyBinding(('x',), Literal(5)), PolyBinding(('y',), Literal(10)))),
                              Program().literal(5).local_push(1).literal(10).local_push(1).local_load(1).local_load(0).add().local_drop(2))

    def test_multi_binding(self):
        self.assertCompilesTo(InlineLet(Add(Name('x'), Name('y')), (PolyBinding(('x', 'y'), Literal([5, 10])),)),
                              Program().literal([5, 10]).local_push(2).local_load(1).local_load(0).add().local_drop(2))


class TestCall(CompilerTestCase):
    def test_builtin_single_arg(self):
        self.assertCompilesTo(Call(Literal(functions.sqrtv), (Literal(25),)),
                              Program().literal(25).call_fast(functions.sqrtv, 1))

    def test_builtin_multiple_args(self):
        self.assertCompilesTo(Call(Literal(functions.clamp), (Name('x'), Literal(0), Literal(1))),
                              Program().local_load(0).literal(0).literal(1).call_fast(functions.clamp, 3), lnames=('x',))

    def test_builtin_keyword_arg_only(self):
        self.assertCompilesTo(Call(Literal(functions.sqrtv), (), (Binding('xs', Literal(25)),)),
                              Program().literal(25).literal(functions.sqrtv).call(0, ('xs',)))

    def test_builtin_multiple_keyword_args(self):
        self.assertCompilesTo(Call(Literal(functions.clamp), (), (Binding('xs', Name('x')), Binding('ys', Literal(0)), Binding('zs', Literal(1)))),
                              Program().local_load(0).literal(0).literal(1).literal(functions.clamp).call(0, ('xs', 'ys', 'zs')), lnames=('x',))

    def test_builtin_mixed_positional_and_keyword_args(self):
        self.assertCompilesTo(Call(Literal(functions.clamp), (Name('x'),), (Binding('ys', Literal(0)), Binding('zs', Literal(1)))),
                              Program().local_load(0).literal(0).literal(1).literal(functions.clamp).call(1, ('ys', 'zs')), lnames=('x',))

    def test_func(self):
        func = Function('func', (Binding('x', Literal(null)),), Name('x')).simplify()
        self.assertCompilesTo(Call(Literal(func), (Name('x'),)),
                              Program().local_load(0).literal(func).call(1), lnames=('x',))

    def test_multiple_builtins(self):
        self.assertCompilesTo(Call(Literal([functions.cosv, functions.sinv]), (Name('th'),)),
                              Program().local_load(0).literal([functions.cosv, functions.sinv]).call(1), lnames=('th',))


class TestFor(CompilerTestCase):
    def test_single_binding(self):
        program = Program()
        START = program.new_label()
        END = program.new_label()
        program.literal([1, 2, 3])
        program.begin_for(1)
        program.label(START)
        program.next(END)
        program.local_load(0)
        program.literal(2)
        program.mul()
        program.jump(START)
        program.label(END)
        program.end_for()
        self.assertCompilesTo(For(('x',), Literal([1, 2, 3]), Multiply(Name('x'), Literal(2))), program)

    def test_multi_binding(self):
        program = Program()
        START = program.new_label()
        END = program.new_label()
        program.literal([1, 2, 3, 4])
        program.begin_for(2)
        program.label(START)
        program.next(END)
        program.local_load(1)
        program.local_load(0)
        program.mul()
        program.jump(START)
        program.label(END)
        program.end_for()
        self.assertCompilesTo(For(('x', 'y'), Literal([1, 2, 3, 4]), Multiply(Name('x'), Name('y'))), program)


class TestIfElse(CompilerTestCase):
    def test_single_if_no_else(self):
        program = Program()
        END = program.new_label()
        program.local_load(1)
        NEXT = program.new_label()
        program.branch_false(NEXT)
        program.local_load(0)
        program.jump(END)
        program.label(NEXT)
        program.literal(null)
        program.label(END)
        self.assertCompilesTo(IfElse((IfCondition(Name('x'), Name('y')),), None), program, lnames=('x', 'y'))

    def test_single_if_else(self):
        program = Program()
        END = program.new_label()
        program.local_load(2)
        NEXT = program.new_label()
        program.branch_false(NEXT)
        program.local_load(1)
        program.jump(END)
        program.label(NEXT)
        program.local_load(0)
        program.label(END)
        self.assertCompilesTo(IfElse((IfCondition(Name('x'), Name('y')),), Name('z')), program, lnames=('x', 'y', 'z'))

    def test_multiple_if_else(self):
        program = Program()
        END = program.new_label()
        program.local_load(2)
        NEXT = program.new_label()
        program.branch_false(NEXT)
        program.local_load(1)
        program.jump(END)
        program.label(NEXT)
        program.local_load(4)
        NEXT = program.new_label()
        program.branch_false(NEXT)
        program.local_load(3)
        program.jump(END)
        program.label(NEXT)
        program.local_load(0)
        program.label(END)
        self.assertCompilesTo(IfElse((IfCondition(Name('x'), Name('y')), IfCondition(Name('a'), Name('b'))), Name('z')), program,
                              lnames=('a', 'b', 'x', 'y', 'z'))


class TestPragma(CompilerTestCase):
    def test_pragma(self):
        self.assertCompilesTo(Pragma('fps', Literal(25)), Program().literal(25).pragma('fps'))


class TestImport(CompilerTestCase):
    def test_import(self):
        self.assertCompilesTo(Import(('x', 'y'), Literal('module.fl')),
                              Program().literal('module.fl').import_(('x', 'y')))


class TestLet(CompilerTestCase):
    def test_single_let(self):
        self.assertCompilesTo(Let((PolyBinding(('x',), Literal(5)),)), Program().literal(5).local_push(1))

    def test_multiple_lets(self):
        self.assertCompilesTo(Let((PolyBinding(('x',), Literal(5)), PolyBinding(('y',), Literal(10)))),
                              Program().literal(5).local_push(1).literal(10).local_push(1))

    def test_multibind_let(self):
        self.assertCompilesTo(Let((PolyBinding(('x', 'y'), Literal([5, 10])),)),
                              Program().literal([5, 10]).local_push(2))


class TestFunction(CompilerTestCase):
    def test_single_parameter_no_captures(self):
        func = Function('func', (Binding('x', Literal(null)),), Add(Name('x'), Literal(5)))
        program = Program()
        START = program.new_label()
        END = program.new_label()
        program.literal(null)
        program.jump(END)
        program.label(START)
        program.local_load(0)
        program.literal(5)
        program.add()
        program.exit()
        program.label(END)
        program.func(START, 'func', ('x',), 0)
        program.local_push(1)
        self.assertCompilesTo(func, program)

    def test_two_parameters_no_captures(self):
        func = Function('func', (Binding('x', Literal(null)), Binding('y', Literal(5))), Add(Name('x'), Name('y')))
        program = Program()
        START = program.new_label()
        END = program.new_label()
        program.literal(null)
        program.literal(5)
        program.jump(END)
        program.label(START)
        program.local_load(1)
        program.local_load(0)
        program.add()
        program.exit()
        program.label(END)
        program.func(START, 'func', ('x', 'y'), 0)
        program.local_push(1)
        self.assertCompilesTo(func, program)

    def test_two_parameters_one_capture(self):
        func = Function('func', (Binding('x', Literal(null)), Binding('y', Literal(5))), Multiply(Add(Name('x'), Name('y')), Name('z')))
        program = Program()
        START = program.new_label()
        END = program.new_label()
        program.local_load(0)
        program.literal(null)
        program.literal(5)
        program.jump(END)
        program.label(START)
        program.local_load(1)
        program.local_load(0)
        program.add()
        program.local_load(3)
        program.mul()
        program.exit()
        program.label(END)
        program.func(START, 'func', ('x', 'y'), 1)
        program.local_push(1)
        self.assertCompilesTo(func, program, lnames=('z',))

    def test_recursive_no_captures(self):
        func = Function('func',
                        (Binding('x', Literal(null)),),
                        IfElse((IfCondition(GreaterThan(Name('x'), Literal(0)), Add(Name('x'), Call(Name('func'), (Subtract(Name('x'), Literal(1)),)))),),
                               Literal(0)))
        program = Program()
        START = program.new_label()
        END = program.new_label()
        program.literal(null)
        program.jump(END)
        program.label(START)
        LAST = program.new_label()
        program.local_load(0)
        program.literal(0)
        program.gt()
        NEXT = program.new_label()
        program.branch_false(NEXT)
        program.local_load(0)
        program.local_load(0)
        program.literal(1)
        program.sub()
        program.local_load(1)
        program.call(1)
        program.add()
        program.jump(LAST)
        program.label(NEXT)
        program.literal(0)
        program.label(LAST)
        program.exit()
        program.label(END)
        program.func(START, 'func', ('x',), 0)
        program.local_push(1)
        self.assertCompilesTo(func, program)


class TestStoreGlobal(CompilerTestCase):
    def test_store_one(self):
        self.assertCompilesTo(StoreGlobal((Binding('x', Literal(5)),)), Program().literal(5).store_global('x'))

    def test_store_multiple(self):
        self.assertCompilesTo(StoreGlobal((Binding('x', Literal(5)), Binding('y', Literal(10)))),
                              Program().literal(5).store_global('x').literal(10).store_global('y'))


class TestTop(CompilerTestCase):
    def test_empty(self):
        self.assertCompilesTo(Top(()), Program())

    def test_literal_node(self):
        self.assertCompilesTo(Top((Literal(Node('window')),)), Program().literal(Node('window')).append())

    def test_single_let(self):
        self.assertCompilesTo(Top((Let((PolyBinding(('x',), Literal(5)),)),)),
                              Program().literal(5).local_push(1).local_load(0).store_global('x').local_drop(1))

    def test_multiple_lets(self):
        self.assertCompilesTo(Top((Let((PolyBinding(('x',), Literal(5)),)), Let((PolyBinding(('y',), Literal(10)),)))),
                              Program().literal(5).local_push(1).literal(10).local_push(1)
                                       .local_load(0).store_global('y')
                                       .local_load(1).store_global('x')
                                       .local_drop(2))

    def test_import(self):
        self.assertCompilesTo(Top((Import(('x', 'y'), Literal('module.fl')),)),
                              Program().literal('module.fl').import_(('x', 'y'))
                                       .local_load(0).store_global('y')
                                       .local_load(1).store_global('x')
                                       .local_drop(2))

    def test_function(self):
        func = Function('func', (Binding('x', Literal(null)),), Add(Name('x'), Literal(5)))
        program = Program()
        START = program.new_label()
        END = program.new_label()
        program.literal(null)
        program.jump(END)
        program.label(START)
        program.local_load(0)
        program.literal(5)
        program.add()
        program.exit()
        program.label(END)
        program.func(START, 'func', ('x',), 0)
        program.local_push(1)
        program.local_load(0)
        program.store_global('func')
        program.local_drop(1)
        self.assertCompilesTo(Top((func,)), program)
