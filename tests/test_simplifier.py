"""
Tests of the simplifier, which is part of the language AST
"""

import unittest

from flitter.model import null
from flitter.language.tree import (Literal, Name, Pragma, Import, Sequence, Function,
                                   Negative, Positive, Add, Subtract, Multiply, Divide,
                                   PolyBinding, InlineLet)


# AST classes still to test:
#
# Top, Function, Lookup, LookupLiteral,
# Range, Not, Ceil, Floor, Fract, Add, Subtract, Multiply, Divide, FloorDivide, Modulo, Power,
# EqualTo, NotEqualTo, LessThan, GreaterThan, LessThanOrEqualTo, GreaterThanOrEqualTo, And, Or, Xor,
# Slice, FastSlice, Call, NodeModifier, Tag, Attributes,
# Append, Let, StoreGlobal, For, IfCondition, IfElse


class SimplifierTestCase(unittest.TestCase):
    def assertSimplifiesTo(self, x, y, state=None, dynamic=None, static=None, with_errors=None):
        errors = set()
        self.assertEqual(repr(x.simplify(state=state, dynamic=dynamic, static=static, errors=errors)), repr(y))
        self.assertEqual(errors, set() if with_errors is None else with_errors)


class TestLiteral(SimplifierTestCase):
    def test_unchanged(self):
        """Literals are unaffected by simplification."""
        self.assertSimplifiesTo(Literal([1, 2, 3]), Literal([1, 2, 3]))
        self.assertSimplifiesTo(Literal('foo'), Literal('foo'))


class TestName(SimplifierTestCase):
    def test_undefined(self):
        """Undefined names are replaced with literal nulls"""
        self.assertSimplifiesTo(Name('x'), Literal(null), with_errors={"Unbound name 'x'"})

    def test_dynamic(self):
        """Dynamic names are unchanged"""
        self.assertSimplifiesTo(Name('x'), Name('x'), dynamic={'x'})

    def test_static(self):
        """Static Vectors simplify to a Literal"""
        self.assertSimplifiesTo(Name('x'), Literal(5), static={'x': 5})

    def test_rename(self):
        """Static Names simplify to the result of simplifying that name"""
        self.assertSimplifiesTo(Name('x'), Name('y'), static={'x': Name('y')}, dynamic={'y'})

    def test_function_name(self):
        """Static Functions are left alone (inlining hack)"""
        self.assertSimplifiesTo(Name('f'), Name('f'), static={'f': Function('f', (), Literal(null))})


class TestPragma(SimplifierTestCase):
    def test_recursive(self):
        """Pragmas are left alone except for the sub-expression being simplified"""
        self.assertSimplifiesTo(Pragma('foo', Name('x')), Pragma('foo', Literal(5)), static={'x': 5})


class TestImport(SimplifierTestCase):
    def test_recursive(self):
        """Imports are left alone except for the sub-expression being simplified"""
        self.assertSimplifiesTo(Import(('x', 'y'), Name('m')), Import(('x', 'y'), Literal('module.fl')), static={'m': 'module.fl'})


class TestSequence(SimplifierTestCase):
    def test_single(self):
        """Single-item sequences simplify to the single expression"""
        self.assertSimplifiesTo(Sequence((Name('x'),)), Name('x'), dynamic={'x'})

    def test_literal_composition(self):
        """Sequential literal vectors are composed"""
        self.assertSimplifiesTo(Sequence((Name('x'), Literal([1, 2, 3]), Literal([4, 5]), Name('y'))),
                                Sequence((Name('x'), Literal([1, 2, 3, 4, 5]), Name('y'))), dynamic={'x', 'y'})

    def test_recursive(self):
        """Each item in a sequence is simplified"""
        self.assertSimplifiesTo(Sequence((Name('x'), Name('y'))), Literal([1, 2, 3, 4, 5]), static={'x': [1, 2, 3], 'y': [4, 5]})


class TestPositive(SimplifierTestCase):
    def test_numeric_literal(self):
        """Numeric literals are left alone"""
        self.assertSimplifiesTo(Positive(Literal(5)), Literal(5))

    def test_non_numeric_literal(self):
        """Non-numeric literals become nulls"""
        self.assertSimplifiesTo(Positive(Literal('foo')), Literal(null))

    def test_double_positive(self):
        """Double-positives become positive"""
        self.assertSimplifiesTo(Positive(Positive(Name('x'))), Positive(Name('x')), dynamic={'x'})

    def test_positive_negative(self):
        """Positive of a negative becomes the negative"""
        self.assertSimplifiesTo(Positive(Negative(Name('x'))), Negative(Name('x')), dynamic={'x'})

    def test_positive_binary_maths(self):
        """Positive of a binary mathematical operation becomes that operation"""
        self.assertSimplifiesTo(Positive(Add(Name('x'), Name('y'))), Add(Name('x'), Name('y')), dynamic={'x', 'y'})


class TestNegative(SimplifierTestCase):
    def test_numeric_literal(self):
        """Numeric literal gets negated"""
        self.assertSimplifiesTo(Negative(Literal(5)), Literal(-5))

    def test_non_numeric_literal(self):
        """Non-numeric literal becomes null"""
        self.assertSimplifiesTo(Negative(Literal('foo')), Literal(null))

    def test_double_negative(self):
        """Double-negative becomes positive"""
        self.assertSimplifiesTo(Negative(Negative(Name('x'))), Positive(Name('x')), dynamic={'x'})

    def test_multiplication(self):
        """Half-literal multiplication has negative pushed into literal"""
        self.assertSimplifiesTo(Negative(Multiply(Literal(5), Name('x'))), Multiply(Literal(-5), Name('x')), dynamic={'x'})
        # And the other way round:
        self.assertSimplifiesTo(Negative(Multiply(Name('x'), Literal(5))), Multiply(Name('x'), Literal(-5)), dynamic={'x'})

    def test_division(self):
        """Half-literal division has negative pushed into literal"""
        self.assertSimplifiesTo(Negative(Divide(Literal(5), Name('x'))), Divide(Literal(-5), Name('x')), dynamic={'x'})
        # The other way round the division is turned into a multiplication by the inverse of the literal:
        self.assertSimplifiesTo(Negative(Divide(Name('x'), Literal(5))), Multiply(Name('x'), Literal(-0.2)), dynamic={'x'})

    def test_addition(self):
        """Half-literal addition becomes a subtraction"""
        # Either way round - this is because of the rule for adding a negative
        self.assertSimplifiesTo(Negative(Add(Literal(5), Name('x'))), Subtract(Literal(-5), Name('x')), dynamic={'x'})
        self.assertSimplifiesTo(Negative(Add(Name('x'), Literal(5))), Subtract(Literal(-5), Name('x')), dynamic={'x'})

    def test_subtraction(self):
        """Half-literal subtraction becomes an addition"""
        self.assertSimplifiesTo(Negative(Subtract(Literal(5), Name('x'))), Add(Literal(-5), Name('x')), dynamic={'x'})
        # However, rule for adding a negative results in a subtraction again the other way round
        self.assertSimplifiesTo(Negative(Subtract(Name('x'), Literal(5))), Subtract(Literal(5), Name('x')), dynamic={'x'})


class TestInlineLet(SimplifierTestCase):
    def test_all_dynamic(self):
        """Binding to a dynamic expression is left alone"""
        self.assertSimplifiesTo(InlineLet(Add(Name('x'), Name('y')), (PolyBinding(('x',), Add(Name('y'), Literal(5))),)),
                                InlineLet(Add(Name('x'), Name('y')), (PolyBinding(('x',), Add(Name('y'), Literal(5))),)),
                                dynamic={'y'})

    def test_literal_binding(self):
        """Simple binding of a name to a literal"""
        self.assertSimplifiesTo(InlineLet(Add(Name('x'), Name('y')), (PolyBinding(('x',), Literal(5)),)),
                                Add(Literal(5), Name('y')),
                                dynamic={'y'})

    def test_rename(self):
        """Simple rename of a local"""
        self.assertSimplifiesTo(InlineLet(Add(Name('x'), Name('y')), (PolyBinding(('x',), Name('y')),)),
                                Add(Name('y'), Name('y')),
                                dynamic={'y'})

    def test_expr_shadowed_rename(self):
        """Rename of a local that is shadowed by a later binding to an expression"""
        self.assertSimplifiesTo(InlineLet(Add(Name('x'), Name('y')), (PolyBinding(('x',), Name('y')), PolyBinding(('y',), Add(Name('y'), Literal(5))))),
                                InlineLet(Add(Name('x'), Name('y')), (PolyBinding(('x',), Name('y')), PolyBinding(('y',), Add(Name('y'), Literal(5))))),
                                dynamic={'y'})

    def test_expr_shadowed_rename_subexpr(self):
        """Rename of a local that is shadowed by a binding to an expression in a sub-expression"""
        self.assertSimplifiesTo(InlineLet(Add(Literal(5), InlineLet(Add(Name('x'), Name('y')), (PolyBinding(('y',), Add(Name('y'), Literal(5))),))),
                                          (PolyBinding(('x',), Name('y')),)),
                                Add(Literal(5), InlineLet(Add(Name('x'), Name('y')), (PolyBinding(('x',), Name('y')),
                                                                                      PolyBinding(('y',), Add(Name('y'), Literal(5))),))),
                                dynamic={'y'})

    def test_literal_shadowed_rename(self):
        """Rename of a local that is shadowed by a later binding to a literal"""
        self.assertSimplifiesTo(InlineLet(Add(Name('x'), Name('y')), (PolyBinding(('x',), Name('y')), PolyBinding(('y',), Literal(5)))),
                                Add(Name('y'), Literal(5)),
                                dynamic={'y'})

    def test_rename_shadowed_rename(self):
        """Rename of a local that is shadowed by a later binding to a rename"""
        self.assertSimplifiesTo(InlineLet(Add(Name('x'), Name('y')), (PolyBinding(('x',), Name('y')), PolyBinding(('y',), Name('z')))),
                                Add(Name('y'), Name('z')),
                                dynamic={'y', 'z'})
