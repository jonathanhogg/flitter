"""
Tests of the simplifier, which is part of the language AST
"""

import unittest

from flitter.model import Vector, StateDict, null, true, false
from flitter.language import functions
from flitter.language.tree import (Binding, PolyBinding,
                                   Literal, Name, Sequence, Function,
                                   Positive, Negative, Ceil, Floor, Fract, Power,
                                   Add, Subtract, Multiply, Divide, FloorDivide, Modulo,
                                   EqualTo, NotEqualTo, LessThan, GreaterThan, LessThanOrEqualTo, GreaterThanOrEqualTo,
                                   Not, And, Or, Xor, Range, Slice, Lookup,
                                   InlineLet, Call,
                                   Pragma, Import)


# AST classes still to test:
#
# For, IfElse
# Tag, Attributes, Append
# Let, Function, StoreGlobal
# Top


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
        """Static Names simplify to the result of simplifying that name (renaming hack)"""
        self.assertSimplifiesTo(Name('x'), Name('y'), static={'x': Name('y')}, dynamic={'y'})

    def test_function_name(self):
        """Static Functions are left alone (inlining hack)"""
        self.assertSimplifiesTo(Name('f'), Name('f'), static={'f': Function('f', (), Literal(null))})

    def test_static_builtin(self):
        """Static built-ins resolve to their literal value"""
        self.assertSimplifiesTo(Name('null'), Literal(null))
        self.assertSimplifiesTo(Name('sqrt'), Literal(functions.sqrtv))

    def test_dynamic_builtin(self):
        """Dynamic built-ins are left alone"""
        self.assertSimplifiesTo(Name('debug'), Name('debug'))


class TestSequence(SimplifierTestCase):
    def test_single(self):
        """Single-item sequences simplify to the single expression"""
        self.assertSimplifiesTo(Sequence((Name('x'),)), Name('x'), dynamic={'x'})

    def test_sequence_packing(self):
        """Sequences within a Sequence are packed together"""
        self.assertSimplifiesTo(Sequence((Name('x'), Sequence((Name('y'), Sequence((Name('y'), Name('y'))))), Sequence((Name('z'),)))),
                                Sequence((Name('x'), Name('y'), Name('y'), Name('y'), Name('z'))),
                                dynamic={'x', 'y', 'z'})

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
        self.assertSimplifiesTo(Negative(Divide(Name('x'), Literal(5))), Multiply(Literal(-0.2), Name('x')), dynamic={'x'})

    def test_addition(self):
        """Half-literal addition becomes a subtraction"""
        # Either way round - this is because of the rule for adding a negative
        self.assertSimplifiesTo(Negative(Add(Literal(5), Name('x'))), Subtract(Literal(-5), Name('x')), dynamic={'x'})
        self.assertSimplifiesTo(Negative(Add(Name('x'), Literal(5))), Subtract(Literal(-5), Name('x')), dynamic={'x'})

    def test_subtraction(self):
        """Half-literal subtraction becomes an addition"""
        self.assertSimplifiesTo(Negative(Subtract(Literal(5), Name('x'))), Add(Literal(-5), Name('x')), dynamic={'x'})
        # However, rule for adding a negative results in a subtraction the other way round
        self.assertSimplifiesTo(Negative(Subtract(Name('x'), Literal(5))), Subtract(Literal(5), Name('x')), dynamic={'x'})


class TestCeil(SimplifierTestCase):
    def test_dynamic(self):
        """Dynamic ceiling is left alone"""
        self.assertSimplifiesTo(Ceil(Name('x')), Ceil(Name('x')), dynamic={'x'})

    def test_literal(self):
        """Literal ceiling is evaluated to a literal"""
        self.assertSimplifiesTo(Ceil(Literal(4.3)), Literal(5))


class TestFloor(SimplifierTestCase):
    def test_dynamic(self):
        """Dynamic floor is left alone"""
        self.assertSimplifiesTo(Floor(Name('x')), Floor(Name('x')), dynamic={'x'})

    def test_literal(self):
        """Literal floor is evaluated to a literal"""
        self.assertSimplifiesTo(Floor(Literal(4.3)), Literal(4))


class TestFract(SimplifierTestCase):
    def test_dynamic(self):
        """Dynamic floor is left alone"""
        self.assertSimplifiesTo(Fract(Name('x')), Fract(Name('x')), dynamic={'x'})

    def test_literal(self):
        """Literal fract is evaluated to a literal"""
        self.assertSimplifiesTo(Fract(Literal(4.3)), Literal(0.3))


class TestAdd(SimplifierTestCase):
    def test_dynamic(self):
        """Dynamic is left alone"""
        self.assertSimplifiesTo(Add(Name('x'), Name('y')), Add(Name('x'), Name('y')), dynamic={'x', 'y'})

    def test_recursive(self):
        """Left and right are simplified"""
        self.assertSimplifiesTo(Add(Name('x'), Name('y')), Add(Name('x'), Name('z')), dynamic={'x', 'z'}, static={'y': Name('z')})
        self.assertSimplifiesTo(Add(Name('x'), Name('y')), Add(Name('z'), Name('y')), dynamic={'y', 'z'}, static={'x': Name('z')})

    def test_literal(self):
        """Literal left and right is evaluated"""
        self.assertSimplifiesTo(Add(Literal(5), Literal(10)), Literal(15))

    def test_zero(self):
        """Adding literal zero becomes Positive"""
        self.assertSimplifiesTo(Add(Literal(0), Name('x')), Positive(Name('x')), dynamic={'x'})
        self.assertSimplifiesTo(Add(Name('x'), Literal(0)), Positive(Name('x')), dynamic={'x'})

    def test_negative(self):
        """Adding a Negative becomes a Subtract"""
        self.assertSimplifiesTo(Add(Name('x'), Negative(Name('y'))), Subtract(Name('x'), Name('y')), dynamic={'x', 'y'})
        self.assertSimplifiesTo(Add(Negative(Name('x')), Name('y')), Subtract(Name('y'), Name('x')), dynamic={'x', 'y'})


class TestSubtract(SimplifierTestCase):
    def test_dynamic(self):
        """Dynamic is left alone"""
        self.assertSimplifiesTo(Subtract(Name('x'), Name('y')), Subtract(Name('x'), Name('y')), dynamic={'x', 'y'})

    def test_recursive(self):
        """Left and right are simplified"""
        self.assertSimplifiesTo(Subtract(Name('x'), Name('y')), Subtract(Name('x'), Name('z')), dynamic={'x', 'z'}, static={'y': Name('z')})
        self.assertSimplifiesTo(Subtract(Name('x'), Name('y')), Subtract(Name('z'), Name('y')), dynamic={'y', 'z'}, static={'x': Name('z')})

    def test_literal(self):
        """Literal left and right is evaluated"""
        self.assertSimplifiesTo(Subtract(Literal(5), Literal(10)), Literal(-5))

    def test_subtract_zero(self):
        """Subtracting literal zero becomes Positive"""
        self.assertSimplifiesTo(Subtract(Name('x'), Literal(0)), Positive(Name('x')), dynamic={'x'})

    def test_subtract_from_zero(self):
        """Subtracting from literal zero becomes Negative"""
        self.assertSimplifiesTo(Subtract(Literal(0), Name('x')), Negative(Name('x')), dynamic={'x'})

    def test_negative(self):
        """Subtracting a Negative becomes an Add"""
        self.assertSimplifiesTo(Subtract(Name('x'), Negative(Name('y'))), Add(Name('x'), Name('y')), dynamic={'x', 'y'})


class TestMultiply(SimplifierTestCase):
    def test_dynamic(self):
        """Dynamic is left alone"""
        self.assertSimplifiesTo(Multiply(Name('x'), Name('y')), Multiply(Name('x'), Name('y')), dynamic={'x', 'y'})

    def test_recursive(self):
        """Left and right are simplified"""
        self.assertSimplifiesTo(Multiply(Name('x'), Name('y')), Multiply(Name('x'), Name('z')), dynamic={'x', 'z'}, static={'y': Name('z')})
        self.assertSimplifiesTo(Multiply(Name('x'), Name('y')), Multiply(Name('z'), Name('y')), dynamic={'y', 'z'}, static={'x': Name('z')})

    def test_literal(self):
        """Literal left and right is evaluated"""
        self.assertSimplifiesTo(Multiply(Literal(5), Literal(10)), Literal(50))

    def test_multiply_one(self):
        """Multiplying by literal 1 becomes Positive"""
        self.assertSimplifiesTo(Multiply(Name('x'), Literal(1)), Positive(Name('x')), dynamic={'x'})
        self.assertSimplifiesTo(Multiply(Literal(1), Name('x')), Positive(Name('x')), dynamic={'x'})

    def test_multiply_minus_one(self):
        """Multiplying by literal -1 becomes Negative"""
        self.assertSimplifiesTo(Multiply(Name('x'), Literal(-1)), Negative(Name('x')), dynamic={'x'})
        self.assertSimplifiesTo(Multiply(Literal(-1), Name('x')), Negative(Name('x')), dynamic={'x'})

    def test_add_propogation(self):
        """Multiplying a half-literal Add by a literal propogates constant"""
        self.assertSimplifiesTo(Multiply(Add(Name('x'), Literal(5)), Literal(10)), Add(Multiply(Literal(10), Name('x')), Literal(50)), dynamic={'x'})
        self.assertSimplifiesTo(Multiply(Literal(10), Add(Name('x'), Literal(5))), Add(Multiply(Literal(10), Name('x')), Literal(50)), dynamic={'x'})

    def test_subtract_propogation(self):
        """Multiplying a half-literal Subtract by a literal propogates constant"""
        self.assertSimplifiesTo(Multiply(Subtract(Literal(5), Name('x')), Literal(10)), Subtract(Literal(50), Multiply(Literal(10), Name('x'))), dynamic={'x'})
        self.assertSimplifiesTo(Multiply(Literal(10), Subtract(Literal(5), Name('x'))), Subtract(Literal(50), Multiply(Literal(10), Name('x'))), dynamic={'x'})

    def test_multiply_propogation(self):
        """Multiplying a half-literal Multiply by a literal propogates constant"""
        self.assertSimplifiesTo(Multiply(Multiply(Literal(5), Name('x')), Literal(10)), Multiply(Literal(50), Name('x')), dynamic={'x'})
        self.assertSimplifiesTo(Multiply(Literal(10), Multiply(Literal(5), Name('x'))), Multiply(Literal(50), Name('x')), dynamic={'x'})
        self.assertSimplifiesTo(Multiply(Multiply(Name('x'), Literal(5)), Literal(10)), Multiply(Literal(50), Name('x')), dynamic={'x'})
        self.assertSimplifiesTo(Multiply(Literal(10), Multiply(Name('x'), Literal(5))), Multiply(Literal(50), Name('x')), dynamic={'x'})

    def test_divide_propogation(self):
        """Multiplying a half-literal Divide by a literal propogates constant"""
        self.assertSimplifiesTo(Multiply(Divide(Literal(5), Name('x')), Literal(10)), Divide(Literal(50), Name('x')), dynamic={'x'})
        self.assertSimplifiesTo(Multiply(Literal(10), Divide(Literal(5), Name('x'))), Divide(Literal(50), Name('x')), dynamic={'x'})
        # When the Divide denominator is the literal, it is propogated into the Multiply literal
        self.assertSimplifiesTo(Multiply(Divide(Name('x'), Literal(5)), Literal(10)), Multiply(Literal(2), Name('x')), dynamic={'x'})
        self.assertSimplifiesTo(Multiply(Literal(10), Divide(Name('x'), Literal(5))), Multiply(Literal(2), Name('x')), dynamic={'x'})

    def test_negative_fold(self):
        """Multiplying a Negative by a literal folds negation into literal"""
        self.assertSimplifiesTo(Multiply(Negative(Name('x')), Literal(10)), Multiply(Literal(-10), Name('x')), dynamic={'x'})
        self.assertSimplifiesTo(Multiply(Literal(10), Negative(Name('x'))), Multiply(Literal(-10), Name('x')), dynamic={'x'})


class TestDivide(SimplifierTestCase):
    def test_dynamic(self):
        """Dynamic is left alone"""
        self.assertSimplifiesTo(Divide(Name('x'), Name('y')), Divide(Name('x'), Name('y')), dynamic={'x', 'y'})

    def test_recursive(self):
        """Left and right are simplified"""
        self.assertSimplifiesTo(Divide(Name('x'), Name('y')), Divide(Name('x'), Name('z')), dynamic={'x', 'z'}, static={'y': Name('z')})
        self.assertSimplifiesTo(Divide(Name('x'), Name('y')), Divide(Name('z'), Name('y')), dynamic={'y', 'z'}, static={'x': Name('z')})

    def test_literal(self):
        """Literal left and right is evaluated"""
        self.assertSimplifiesTo(Divide(Literal(5), Literal(10)), Literal(0.5))

    def test_divide_by_one(self):
        """Dividing by literal 1 becomes Positive"""
        self.assertSimplifiesTo(Divide(Name('x'), Literal(1)), Positive(Name('x')), dynamic={'x'})

    def test_divide_by_literal(self):
        """Dividing by literal becomes Multiply of inverse"""
        self.assertSimplifiesTo(Divide(Name('x'), Literal(10)), Multiply(Literal(0.1), Name('x')), dynamic={'x'})


class TestFloorDivide(SimplifierTestCase):
    def test_dynamic(self):
        """Dynamic is left alone"""
        self.assertSimplifiesTo(FloorDivide(Name('x'), Name('y')), FloorDivide(Name('x'), Name('y')), dynamic={'x', 'y'})

    def test_recursive(self):
        """Left and right are simplified"""
        self.assertSimplifiesTo(FloorDivide(Name('x'), Name('y')), FloorDivide(Name('x'), Name('z')), dynamic={'x', 'z'}, static={'y': Name('z')})
        self.assertSimplifiesTo(FloorDivide(Name('x'), Name('y')), FloorDivide(Name('z'), Name('y')), dynamic={'y', 'z'}, static={'x': Name('z')})

    def test_literal(self):
        """Literal left and right is evaluated"""
        self.assertSimplifiesTo(FloorDivide(Literal(5), Literal(10)), Literal(0))

    def test_divide_by_one(self):
        """Dividing by literal 1 becomes Floor"""
        self.assertSimplifiesTo(FloorDivide(Name('x'), Literal(1)), Floor(Name('x')), dynamic={'x'})


class TestModulo(SimplifierTestCase):
    def test_dynamic(self):
        """Dynamic is left alone"""
        self.assertSimplifiesTo(Modulo(Name('x'), Name('y')), Modulo(Name('x'), Name('y')), dynamic={'x', 'y'})

    def test_recursive(self):
        """Left and right are simplified"""
        self.assertSimplifiesTo(Modulo(Name('x'), Name('y')), Modulo(Name('x'), Name('z')), dynamic={'x', 'z'}, static={'y': Name('z')})
        self.assertSimplifiesTo(Modulo(Name('x'), Name('y')), Modulo(Name('z'), Name('y')), dynamic={'y', 'z'}, static={'x': Name('z')})

    def test_literal(self):
        """Literal left and right is evaluated"""
        self.assertSimplifiesTo(Modulo(Literal(5), Literal(10)), Literal(5))

    def test_modulo_one(self):
        """Modulo literal 1 becomes Fract"""
        self.assertSimplifiesTo(Modulo(Name('x'), Literal(1)), Fract(Name('x')), dynamic={'x'})


class TestPower(SimplifierTestCase):
    def test_dynamic(self):
        """Dynamic is left alone"""
        self.assertSimplifiesTo(Power(Name('x'), Name('y')), Power(Name('x'), Name('y')), dynamic={'x', 'y'})

    def test_recursive(self):
        """Left and right are simplified"""
        self.assertSimplifiesTo(Power(Name('x'), Name('y')), Power(Name('x'), Name('z')), dynamic={'x', 'z'}, static={'y': Name('z')})
        self.assertSimplifiesTo(Power(Name('x'), Name('y')), Power(Name('z'), Name('y')), dynamic={'y', 'z'}, static={'x': Name('z')})

    def test_literal(self):
        """Literal left and right is evaluated"""
        self.assertSimplifiesTo(Power(Literal(5), Literal(2)), Literal(25))

    def test_raise_to_power_of_one(self):
        """Power to literal 1 becomes Positive"""
        self.assertSimplifiesTo(Power(Name('x'), Literal(1)), Positive(Name('x')), dynamic={'x'})


class TestEqualTo(SimplifierTestCase):
    def test_dynamic(self):
        self.assertSimplifiesTo(EqualTo(Name('x'), Name('y')), EqualTo(Name('x'), Name('y')), dynamic={'x', 'y'})
        self.assertSimplifiesTo(EqualTo(Literal(5), Name('y')), EqualTo(Literal(5), Name('y')), dynamic={'y'})
        self.assertSimplifiesTo(EqualTo(Name('x'), Literal(5)), EqualTo(Name('x'), Literal(5)), dynamic={'x'})

    def test_recursive(self):
        """Left and right are simplified"""
        self.assertSimplifiesTo(EqualTo(Name('x'), Name('y')), EqualTo(Name('x'), Name('z')), dynamic={'x', 'z'}, static={'y': Name('z')})
        self.assertSimplifiesTo(EqualTo(Name('x'), Name('y')), EqualTo(Name('z'), Name('y')), dynamic={'y', 'z'}, static={'x': Name('z')})

    def test_literal(self):
        """Literal left and right is evaluated"""
        self.assertSimplifiesTo(EqualTo(Literal(5), Literal(5)), Literal(true))
        self.assertSimplifiesTo(EqualTo(Literal(5), Literal(4)), Literal(false))
        self.assertSimplifiesTo(EqualTo(Literal(4), Literal(5)), Literal(false))


class TestNotEqualTo(SimplifierTestCase):
    def test_dynamic(self):
        self.assertSimplifiesTo(NotEqualTo(Name('x'), Name('y')), NotEqualTo(Name('x'), Name('y')), dynamic={'x', 'y'})
        self.assertSimplifiesTo(NotEqualTo(Literal(5), Name('y')), NotEqualTo(Literal(5), Name('y')), dynamic={'y'})
        self.assertSimplifiesTo(NotEqualTo(Name('x'), Literal(5)), NotEqualTo(Name('x'), Literal(5)), dynamic={'x'})

    def test_recursive(self):
        """Left and right are simplified"""
        self.assertSimplifiesTo(NotEqualTo(Name('x'), Name('y')), NotEqualTo(Name('x'), Name('z')), dynamic={'x', 'z'}, static={'y': Name('z')})
        self.assertSimplifiesTo(NotEqualTo(Name('x'), Name('y')), NotEqualTo(Name('z'), Name('y')), dynamic={'y', 'z'}, static={'x': Name('z')})

    def test_literal(self):
        """Literal left and right is evaluated"""
        self.assertSimplifiesTo(NotEqualTo(Literal(5), Literal(5)), Literal(false))
        self.assertSimplifiesTo(NotEqualTo(Literal(5), Literal(4)), Literal(true))
        self.assertSimplifiesTo(NotEqualTo(Literal(4), Literal(5)), Literal(true))


class TestLessThan(SimplifierTestCase):
    def test_dynamic(self):
        self.assertSimplifiesTo(LessThan(Name('x'), Name('y')), LessThan(Name('x'), Name('y')), dynamic={'x', 'y'})
        self.assertSimplifiesTo(LessThan(Literal(5), Name('y')), LessThan(Literal(5), Name('y')), dynamic={'y'})
        self.assertSimplifiesTo(LessThan(Name('x'), Literal(5)), LessThan(Name('x'), Literal(5)), dynamic={'x'})

    def test_recursive(self):
        """Left and right are simplified"""
        self.assertSimplifiesTo(LessThan(Name('x'), Name('y')), LessThan(Name('x'), Name('z')), dynamic={'x', 'z'}, static={'y': Name('z')})
        self.assertSimplifiesTo(LessThan(Name('x'), Name('y')), LessThan(Name('z'), Name('y')), dynamic={'y', 'z'}, static={'x': Name('z')})

    def test_literal(self):
        """Literal left and right is evaluated"""
        self.assertSimplifiesTo(LessThan(Literal(5), Literal(5)), Literal(false))
        self.assertSimplifiesTo(LessThan(Literal(5), Literal(4)), Literal(false))
        self.assertSimplifiesTo(LessThan(Literal(4), Literal(5)), Literal(true))


class TestGreaterThan(SimplifierTestCase):
    def test_dynamic(self):
        self.assertSimplifiesTo(GreaterThan(Name('x'), Name('y')), GreaterThan(Name('x'), Name('y')), dynamic={'x', 'y'})
        self.assertSimplifiesTo(GreaterThan(Literal(5), Name('y')), GreaterThan(Literal(5), Name('y')), dynamic={'y'})
        self.assertSimplifiesTo(GreaterThan(Name('x'), Literal(5)), GreaterThan(Name('x'), Literal(5)), dynamic={'x'})

    def test_recursive(self):
        """Left and right are simplified"""
        self.assertSimplifiesTo(GreaterThan(Name('x'), Name('y')), GreaterThan(Name('x'), Name('z')), dynamic={'x', 'z'}, static={'y': Name('z')})
        self.assertSimplifiesTo(GreaterThan(Name('x'), Name('y')), GreaterThan(Name('z'), Name('y')), dynamic={'y', 'z'}, static={'x': Name('z')})

    def test_literal(self):
        """Literal left and right is evaluated"""
        self.assertSimplifiesTo(GreaterThan(Literal(5), Literal(5)), Literal(false))
        self.assertSimplifiesTo(GreaterThan(Literal(5), Literal(4)), Literal(true))
        self.assertSimplifiesTo(GreaterThan(Literal(4), Literal(5)), Literal(false))


class TestLessThanOrEqualTo(SimplifierTestCase):
    def test_dynamic(self):
        self.assertSimplifiesTo(LessThanOrEqualTo(Name('x'), Name('y')), LessThanOrEqualTo(Name('x'), Name('y')), dynamic={'x', 'y'})
        self.assertSimplifiesTo(LessThanOrEqualTo(Literal(5), Name('y')), LessThanOrEqualTo(Literal(5), Name('y')), dynamic={'y'})
        self.assertSimplifiesTo(LessThanOrEqualTo(Name('x'), Literal(5)), LessThanOrEqualTo(Name('x'), Literal(5)), dynamic={'x'})

    def test_recursive(self):
        """Left and right are simplified"""
        self.assertSimplifiesTo(LessThanOrEqualTo(Name('x'), Name('y')), LessThanOrEqualTo(Name('x'), Name('z')), dynamic={'x', 'z'}, static={'y': Name('z')})
        self.assertSimplifiesTo(LessThanOrEqualTo(Name('x'), Name('y')), LessThanOrEqualTo(Name('z'), Name('y')), dynamic={'y', 'z'}, static={'x': Name('z')})

    def test_literal(self):
        """Literal left and right is evaluated"""
        self.assertSimplifiesTo(LessThanOrEqualTo(Literal(5), Literal(5)), Literal(true))
        self.assertSimplifiesTo(LessThanOrEqualTo(Literal(5), Literal(4)), Literal(false))
        self.assertSimplifiesTo(LessThanOrEqualTo(Literal(4), Literal(5)), Literal(true))


class TestGreaterThanOrEqualTo(SimplifierTestCase):
    def test_dynamic(self):
        self.assertSimplifiesTo(GreaterThanOrEqualTo(Name('x'), Name('y')), GreaterThanOrEqualTo(Name('x'), Name('y')), dynamic={'x', 'y'})
        self.assertSimplifiesTo(GreaterThanOrEqualTo(Literal(5), Name('y')), GreaterThanOrEqualTo(Literal(5), Name('y')), dynamic={'y'})
        self.assertSimplifiesTo(GreaterThanOrEqualTo(Name('x'), Literal(5)), GreaterThanOrEqualTo(Name('x'), Literal(5)), dynamic={'x'})

    def test_recursive(self):
        """Left and right are simplified"""
        self.assertSimplifiesTo(GreaterThanOrEqualTo(Name('x'), Name('y')), GreaterThanOrEqualTo(Name('x'), Name('z')),
                                dynamic={'x', 'z'}, static={'y': Name('z')})
        self.assertSimplifiesTo(GreaterThanOrEqualTo(Name('x'), Name('y')), GreaterThanOrEqualTo(Name('z'), Name('y')),
                                dynamic={'y', 'z'}, static={'x': Name('z')})

    def test_literal(self):
        """Literal left and right is evaluated"""
        self.assertSimplifiesTo(GreaterThanOrEqualTo(Literal(5), Literal(5)), Literal(true))
        self.assertSimplifiesTo(GreaterThanOrEqualTo(Literal(5), Literal(4)), Literal(true))
        self.assertSimplifiesTo(GreaterThanOrEqualTo(Literal(4), Literal(5)), Literal(false))


class TestNot(SimplifierTestCase):
    def test_dynamic(self):
        """Dynamic is left alone"""
        self.assertSimplifiesTo(Not(Name('x')), Not(Name('x')), dynamic={'x'})

    def test_recursive(self):
        """Expression is simplified"""
        self.assertSimplifiesTo(Not(Name('x')), Not(Name('y')), dynamic={'y'}, static={'x': Name('y')})

    def test_literal(self):
        """Literal is evaluated"""
        self.assertSimplifiesTo(Not(Literal(false)), Literal(true))
        self.assertSimplifiesTo(Not(Literal(true)), Literal(false))


class TestAnd(SimplifierTestCase):
    def test_dynamic(self):
        """Dynamic is left alone"""
        self.assertSimplifiesTo(And(Name('x'), Name('y')), And(Name('x'), Name('y')), dynamic={'x', 'y'})

    def test_recursive(self):
        """Left and right are simplified"""
        self.assertSimplifiesTo(And(Name('x'), Name('y')), And(Name('z'), Name('y')), static={'x': Name('z')}, dynamic={'y', 'z'})
        self.assertSimplifiesTo(And(Name('x'), Name('y')), And(Name('x'), Name('z')), static={'y': Name('z')}, dynamic={'x', 'z'})

    def test_literal(self):
        """Literal is evaluated"""
        self.assertSimplifiesTo(And(Literal(true), Literal(true)), Literal(true))
        self.assertSimplifiesTo(And(Literal(true), Literal(false)), Literal(false))
        self.assertSimplifiesTo(And(Literal(false), Literal(true)), Literal(false))
        self.assertSimplifiesTo(And(Literal(false), Literal(false)), Literal(false))

    def test_true_left(self):
        """True left shortcuts to right"""
        self.assertSimplifiesTo(And(Literal(true), Name('y')), Name('y'), dynamic={'y'})

    def test_true_right(self):
        """True right ignored"""
        self.assertSimplifiesTo(And(Name('x'), Literal(true)), And(Name('x'), Literal(true)), dynamic={'x'})

    def test_false_left(self):
        """False left shortcuts to left"""
        self.assertSimplifiesTo(And(Literal(false), Name('y')), Literal(false), dynamic={'y'})

    def test_false_right(self):
        """False right ignored"""
        self.assertSimplifiesTo(And(Name('x'), Literal(false)), And(Name('x'), Literal(false)), dynamic={'x'})


class TestOr(SimplifierTestCase):
    def test_dynamic(self):
        """Dynamic is left alone"""
        self.assertSimplifiesTo(Or(Name('x'), Name('y')), Or(Name('x'), Name('y')), dynamic={'x', 'y'})

    def test_recursive(self):
        """Left and right are simplified"""
        self.assertSimplifiesTo(Or(Name('x'), Name('y')), Or(Name('z'), Name('y')), static={'x': Name('z')}, dynamic={'y', 'z'})
        self.assertSimplifiesTo(Or(Name('x'), Name('y')), Or(Name('x'), Name('z')), static={'y': Name('z')}, dynamic={'x', 'z'})

    def test_literal(self):
        """Literal is evaluated"""
        self.assertSimplifiesTo(Or(Literal(true), Literal(true)), Literal(true))
        self.assertSimplifiesTo(Or(Literal(true), Literal(false)), Literal(true))
        self.assertSimplifiesTo(Or(Literal(false), Literal(true)), Literal(true))
        self.assertSimplifiesTo(Or(Literal(false), Literal(false)), Literal(false))

    def test_true_left(self):
        """True left shortcuts to left"""
        self.assertSimplifiesTo(Or(Literal(true), Name('y')), Literal(true), dynamic={'y'})

    def test_true_right(self):
        """True right ignored"""
        self.assertSimplifiesTo(Or(Name('x'), Literal(true)), Or(Name('x'), Literal(true)), dynamic={'x'})

    def test_false_left(self):
        """False left shortcuts to right"""
        self.assertSimplifiesTo(Or(Literal(false), Name('y')), Name('y'), dynamic={'y'})

    def test_false_right(self):
        """False right ignored"""
        self.assertSimplifiesTo(Or(Name('x'), Literal(false)), Or(Name('x'), Literal(false)), dynamic={'x'})


class TestXor(SimplifierTestCase):
    def test_dynamic(self):
        """Dynamic is left alone"""
        self.assertSimplifiesTo(Xor(Name('x'), Name('y')), Xor(Name('x'), Name('y')), dynamic={'x', 'y'})

    def test_recursive(self):
        """Left and right are simplified"""
        self.assertSimplifiesTo(Xor(Name('x'), Name('y')), Xor(Name('z'), Name('y')), static={'x': Name('z')}, dynamic={'y', 'z'})
        self.assertSimplifiesTo(Xor(Name('x'), Name('y')), Xor(Name('x'), Name('z')), static={'y': Name('z')}, dynamic={'x', 'z'})

    def test_literal(self):
        """Literal is evaluated"""
        self.assertSimplifiesTo(Xor(Literal(true), Literal(true)), Literal(false))
        self.assertSimplifiesTo(Xor(Literal(true), Literal(false)), Literal(true))
        self.assertSimplifiesTo(Xor(Literal(false), Literal(true)), Literal(true))
        self.assertSimplifiesTo(Xor(Literal(false), Literal(false)), Literal(false))

    def test_true_left(self):
        """True left ignored"""
        self.assertSimplifiesTo(Xor(Literal(true), Name('y')), Xor(Literal(true), Name('y')), dynamic={'y'})

    def test_true_right(self):
        """True right ignored"""
        self.assertSimplifiesTo(Xor(Name('x'), Literal(true)), Xor(Name('x'), Literal(true)), dynamic={'x'})

    def test_false_left(self):
        """False left shortcuts to right"""
        self.assertSimplifiesTo(Xor(Literal(false), Name('y')), Name('y'), dynamic={'y'})

    def test_false_right(self):
        """False right shorcuts to left"""
        self.assertSimplifiesTo(Xor(Name('x'), Literal(false)), Name('x'), dynamic={'x'})


class TestRange(SimplifierTestCase):
    def test_dynamic(self):
        """Dynamic is left alone"""
        self.assertSimplifiesTo(Range(Name('w'), Name('x'), Name('y')), Range(Name('w'), Name('x'), Name('y')), dynamic={'w', 'x', 'y'})

    def test_recursive(self):
        """Start, stop and step are simplified"""
        self.assertSimplifiesTo(Range(Name('w'), Name('x'), Name('y')), Range(Name('z'), Name('x'), Name('y')),
                                dynamic={'x', 'y', 'z'}, static={'w': Name('z')})
        self.assertSimplifiesTo(Range(Name('w'), Name('x'), Name('y')), Range(Name('w'), Name('z'), Name('y')),
                                dynamic={'w', 'y', 'z'}, static={'x': Name('z')})
        self.assertSimplifiesTo(Range(Name('w'), Name('x'), Name('y')), Range(Name('w'), Name('x'), Name('z')),
                                dynamic={'w', 'x', 'z'}, static={'y': Name('z')})

    def test_literal(self):
        """Literal Range is evaluated"""
        self.assertSimplifiesTo(Range(Literal(0), Literal(10), Literal(2)), Literal([0, 2, 4, 6, 8]))


class TestSlice(SimplifierTestCase):
    def test_dynamic(self):
        """Dynamic is left alone"""
        self.assertSimplifiesTo(Slice(Name('x'), Name('y')), Slice(Name('x'), Name('y')), dynamic={'x', 'y'})
        self.assertSimplifiesTo(Slice(Literal(range(1, 6)), Name('y')), Slice(Literal(range(1, 6)), Name('y')), dynamic={'y'})
        self.assertSimplifiesTo(Slice(Name('x'), Literal(3)), Slice(Name('x'), Literal(3)), dynamic={'x'})

    def test_recursive(self):
        """Expression and index are simplified"""
        self.assertSimplifiesTo(Slice(Name('x'), Name('y')), Slice(Name('z'), Name('y')), static={'x': Name('z')}, dynamic={'y', 'z'})
        self.assertSimplifiesTo(Slice(Name('x'), Name('y')), Slice(Name('x'), Name('z')), static={'y': Name('z')}, dynamic={'x', 'z'})

    def test_literal(self):
        """Literal expr and key is evaluated"""
        self.assertSimplifiesTo(Slice(Literal(range(1, 6)), Literal(3)), Literal(4))


class TestLookup(SimplifierTestCase):
    def test_dynamic(self):
        """Dynamic is left alone"""
        self.assertSimplifiesTo(Lookup(Name('x')), Lookup(Name('x')), dynamic={'x'})

    def test_recursive(self):
        """Key is simplified"""
        self.assertSimplifiesTo(Lookup(Name('x')), Lookup(Name('y')), dynamic={'y'}, static={'x': Name('y')})

    def test_literal_key_no_state(self):
        """Literal key with no supplied state is left alone"""
        self.assertSimplifiesTo(Lookup(Literal(Vector.symbol('foo'))), Lookup(Literal(Vector.symbol('foo'))))

    def test_literal_key_not_in_state(self):
        """Literal key not in supplied state is left alone"""
        self.assertSimplifiesTo(Lookup(Literal(Vector.symbol('foo'))), Lookup(Literal(Vector.symbol('foo'))), state=StateDict())

    def test_literal_key_in_state(self):
        """Literal key in supplied state is simplified to literal value"""
        self.assertSimplifiesTo(Lookup(Literal(Vector.symbol('foo'))), Literal(5), state=StateDict({Vector.symbol('foo'): 5}))


class TestInlineLet(SimplifierTestCase):
    def test_dynamic(self):
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


class TestCall(SimplifierTestCase):
    def test_dynamic(self):
        """Dynamic calls left alone"""
        self.assertSimplifiesTo(Call(Name('x'), (Literal(5),), ()), Call(Name('x'), (Literal(5),), ()), dynamic={'x'})
        self.assertSimplifiesTo(Call(Literal(functions.sqrtv), (Name('y'),), ()), Call(Literal(functions.sqrtv), (Name('y'),), ()), dynamic={'y'})
        self.assertSimplifiesTo(Call(Literal(functions.sqrtv), (), (Binding('xs', Name('y')),)),
                                Call(Literal(functions.sqrtv), (), (Binding('xs', Name('y')),)), dynamic={'y'})

    def test_static(self):
        """Static calls to built-in functions are replaced with their return value"""
        self.assertSimplifiesTo(Call(Literal(functions.sqrtv), (Literal(25),), ()), Literal(5))
        self.assertSimplifiesTo(Call(Literal(functions.sqrtv), (), (Binding('xs', Literal(25)),)), Literal(5))

    def test_inlining(self):
        """Calls to names that resolve to Function objects are inlined as let expressions"""
        func = Function('func', (Binding('x', Literal(null)),), Add(Name('x'), Literal(5))).simplify()
        self.assertSimplifiesTo(Call(Name('func'), (Add(Literal(1), Name('y')),), ()),
                                InlineLet(Add(Name('x'), Literal(5)), (PolyBinding(('x',), Add(Literal(1), Name('y'))),)),
                                static={'func': func}, dynamic={'y'})


class TestPragma(SimplifierTestCase):
    def test_recursive(self):
        """Pragmas are left alone except for the sub-expression being simplified"""
        self.assertSimplifiesTo(Pragma('foo', Name('x')), Pragma('foo', Literal(5)), static={'x': 5})


class TestImport(SimplifierTestCase):
    def test_recursive(self):
        """Imports are left alone except for the sub-expression being simplified"""
        self.assertSimplifiesTo(Import(('x', 'y'), Name('m')), Import(('x', 'y'), Literal('module.fl')), static={'m': 'module.fl'})
