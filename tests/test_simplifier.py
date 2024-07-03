"""
Flitter language simplifier unit tests
"""

import unittest

from flitter.model import Vector, Node, StateDict, null, true, false
from flitter.language import functions
from flitter.language.tree import (Literal, Name, Sequence,
                                   Positive, Negative, Ceil, Floor, Fract, Power,
                                   Add, Subtract, Multiply, Divide, FloorDivide, Modulo,
                                   EqualTo, NotEqualTo, LessThan, GreaterThan, LessThanOrEqualTo, GreaterThanOrEqualTo,
                                   Not, And, Or, Xor, Range, Slice, Lookup,
                                   Tag, Attributes, Append,
                                   Let, Call, For, IfElse,
                                   Import, Function, Top, Export,
                                   Binding, PolyBinding, IfCondition)


class SimplifierTestCase(unittest.TestCase):
    def assertSimplifiesTo(self, x, y, state=None, dynamic=None, static=None, with_errors=None, with_names=None):
        x, context = x.simplify(state=state, dynamic=dynamic, static=static, return_context=True)
        if not isinstance(x, Let):
            z = x.simplify(state=state, dynamic=dynamic, static=static)
            self.assertEqual(repr(z), repr(x), msg="Simplification not complete")
        self.assertEqual(repr(x), repr(y))
        self.assertEqual(context.errors, set() if with_errors is None else with_errors)
        if with_names:
            for name, value in with_names.items():
                if isinstance(value, (Name, Function)):
                    self.assertEqual(repr(context.names.pop(name)), repr(value), msg=f"{name} differs from expected")
                else:
                    self.assertEqual(context.names.pop(name), value, msg=f"{name} differs from expected")
        if static is not None:
            for name in static:
                if with_names is None or name not in with_names:
                    self.assertEqual(context.names.pop(name), static[name], msg=f"{name} differs from original static value")
        if dynamic is not None:
            for name in dynamic:
                if with_names is None or name not in with_names:
                    self.assertEqual(context.names.pop(name), None, msg=f"{name} should be dynamic")
        self.assertEqual(len(context.names), 0, msg=f"Unexpected names: {context.names!r}")


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
    def test_empty(self):
        """Empty sequences simplify to literal null"""
        self.assertSimplifiesTo(Sequence(()), Literal(null))

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


class TestTag(SimplifierTestCase):
    def test_dynamic(self):
        """Dynamic node expression is left alone"""
        self.assertSimplifiesTo(Tag(Name('node'), 'tag'), Tag(Name('node'), 'tag'), dynamic={'node'})

    def test_literal_node(self):
        """Literal node is tagged"""
        self.assertSimplifiesTo(Tag(Literal(Node('node')), 'tag'), Literal(Node('node', {'tag'})))

    def test_literal_nodes(self):
        """Literal nodes are tagged"""
        self.assertSimplifiesTo(Tag(Literal([Node('node1'), Node('node2', {'tag1'})]), 'tag2'),
                                Literal([Node('node1', {'tag2'}), Node('node2', {'tag1', 'tag2'})]))

    def test_literal_null(self):
        """Tag of null becomes null"""
        self.assertSimplifiesTo(Tag(Literal(null), 'tag'), Literal(null))

    def test_literal_numeric(self):
        """Tag of numeric is discarded"""
        self.assertSimplifiesTo(Tag(Literal(5), 'tag'), Literal(5))

    def test_literal_mixed(self):
        """Tag of mixed literal has all nodes updated"""
        self.assertSimplifiesTo(Tag(Literal([5, "hello", Node('node')]), 'tag'), Literal([5, "hello", Node('node', {'tag'})]))


class TestAttributes(SimplifierTestCase):
    def test_dynamic(self):
        """Dynamic node or value expressions are left alone"""
        self.assertSimplifiesTo(Attributes(Name('x'), (Binding('y', Literal(5)),)), Attributes(Name('x'), (Binding('y', Literal(5)),)), dynamic={'x'})
        self.assertSimplifiesTo(Attributes(Literal(Node('node')), (Binding('y', Name('y')),)),
                                Attributes(Literal(Node('node')), (Binding('y', Name('y')),)), dynamic={'y'})

    def test_literal_node(self):
        """Literal node has attributes updated"""
        self.assertSimplifiesTo(Attributes(Literal(Node('node')), (Binding('y', Literal(5)),)),
                                Literal(Node('node', attributes={'y': Vector(5)})))

    def test_literal_nodes(self):
        """Literal nodes have attributes updated"""
        self.assertSimplifiesTo(Attributes(Literal([Node('node1'), Node('node2', attributes={'x': Vector(1)})]), (Binding('y', Literal(5)),)),
                                Literal([Node('node1', attributes={'y': Vector(5)}), Node('node2', attributes={'x': Vector(1), 'y': Vector(5)})]))

    def test_partial_application(self):
        """Literal bindings are applied until first dynamic binding"""
        self.assertSimplifiesTo(Attributes(Literal(Node('node')), (Binding('y', Literal(5)), Binding('z', Name('z')))),
                                Attributes(Literal(Node('node', attributes={'y': Vector(5)})), (Binding('z', Name('z')),)), dynamic={'z'})

    def test_combining(self):
        """Nested attributes are combined"""
        self.assertSimplifiesTo(Attributes(Attributes(Name('node'), (Binding('x', Name('x')),)), (Binding('y', Name('y')), Binding('z', Name('z')))),
                                Attributes(Name('node'), (Binding('x', Name('x')), Binding('y', Name('y')), Binding('z', Name('z')))),
                                dynamic={'node', 'x', 'y', 'z'})

    def test_literal_null(self):
        """Attribute set of null becomes null whether dynamic or not"""
        self.assertSimplifiesTo(Attributes(Literal(null), (Binding('x', Name('x')),)), Literal(null), dynamic={'x'})

    def test_literal_numeric(self):
        """Attribute set of numeric is discarded whether dynamic or not"""
        self.assertSimplifiesTo(Attributes(Literal(5), (Binding('x', Name('x')),)), Literal(5), dynamic={'x'})

    def test_literal_mixed(self):
        """Literal attribute set of mixed literal has all nodes updated"""
        self.assertSimplifiesTo(Attributes(Literal([5, "hello", Node('node')]), (Binding('x', Literal(5)),)),
                                Literal([5, "hello", Node('node', attributes={'x': Vector(5)})]))


class TestAppend(SimplifierTestCase):
    def test_dynamic(self):
        """Dynamic node expressions are left alone"""
        self.assertSimplifiesTo(Append(Name('x'), Literal(Node('y'))), Append(Name('x'), Literal(Node('y'))), dynamic={'x'})
        self.assertSimplifiesTo(Append(Literal(Node('x')), Name('y')), Append(Literal(Node('x')), Name('y')), dynamic={'y'})

    def test_literal_node(self):
        """Literal node has sub-node appended"""
        self.assertSimplifiesTo(Append(Literal(Node('x')), Literal(Node('y'))),
                                Literal(Node('x', children=(Node('y'),))))

    def test_literal_nodes(self):
        """Literal nodes have attributes updated"""
        self.assertSimplifiesTo(Append(Literal([Node('x1'), Node('x2', children=(Node('y1'),))]), Literal(Node('y'))),
                                Literal([Node('x1', children=(Node('y'),)), Node('x2', children=(Node('y1'), Node('y')))]))

    def test_push_through_attributes_to_literal(self):
        """Literal appends are pushed through intermediate attribute operation to a literal root"""
        self.assertSimplifiesTo(Append(Attributes(Literal(Node('node1')), (Binding('x', Name('x')),)), Literal(Node('node2'))),
                                Attributes(Literal(Node('node1', children=(Node('node2'),))), (Binding('x', Name('x')),)),
                                dynamic={'x'})

    def test_pull_literal_from_sequence(self):
        """A literal at the start of an appended sequence is pulled out and appended to a literal node"""
        self.assertSimplifiesTo(Append(Literal(Node('node1')), Sequence((Literal(Node('node2')), Name('x'), Name('y')))),
                                Append(Literal(Node('node1', children=(Node('node2'),))), Sequence((Name('x'), Name('y')))),
                                dynamic={'x', 'y'})

    def test_append_to_literal_null(self):
        """Append to null is discarded"""
        self.assertSimplifiesTo(Append(Literal(null), Name('x')), Literal(null), dynamic={'x'})

    def test_append_of_literal_null(self):
        """Append of null is discarded"""
        self.assertSimplifiesTo(Append(Name('x'), Literal(null)), Name('x'), dynamic={'x'})

    def test_append_to_literal_numeric(self):
        """Append to numeric is discarded"""
        self.assertSimplifiesTo(Append(Literal(5), Name('x')), Literal(5), dynamic={'x'})

    def test_append_of_literal_numeric(self):
        """Append of numeric is discarded"""
        self.assertSimplifiesTo(Append(Name('x'), Literal(5)), Name('x'), dynamic={'x'})

    def test_literal_mixed(self):
        """Mixed literal append to mixed literal has all nodes updated"""
        self.assertSimplifiesTo(Append(Literal([5, "hello", Node('node')]), Literal([Node('node2'), 7])),
                                Literal([5, "hello", Node('node', children=(Node('node2'),))]))


class TestLet(SimplifierTestCase):
    def test_dynamic(self):
        """Binding to a dynamic expression is left alone"""
        self.assertSimplifiesTo(Let((PolyBinding(('x',), Add(Name('y'), Literal(5))),), Add(Name('x'), Name('y'))),
                                Let((PolyBinding(('x',), Add(Name('y'), Literal(5))),), Add(Name('x'), Name('y'))),
                                dynamic={'y'})

    def test_let_in_literal(self):
        """A let over a literal is always just the literal"""
        self.assertSimplifiesTo(Let((PolyBinding(('x',), Add(Name('y'), Literal(5))),), Literal(10)), Literal(10), dynamic={'y'})

    def test_literal_binding(self):
        """Simple binding of a name to a literal"""
        self.assertSimplifiesTo(Let((PolyBinding(('x',), Literal(5)),), Add(Name('x'), Name('y'))),
                                Add(Literal(5), Name('y')),
                                dynamic={'y'})

    def test_literal_multi_binding(self):
        """Binding of a name sequence to a literal"""
        self.assertSimplifiesTo(Let((PolyBinding(('x', 'y'), Literal([5, 10])),), Add(Name('x'), Name('y'))), Literal(15))

    def test_literal_short_multi_binding(self):
        """Binding of a name sequence to a short literal wraps"""
        self.assertSimplifiesTo(Let((PolyBinding(('x', 'y', 'z'), Literal([5, 10])),), Name('z')), Literal(5))

    def test_function_inlining(self):
        """Binding a name to a Function expression places Function into names for inlining"""
        self.assertSimplifiesTo(Let((PolyBinding(('f',), Function('f', (Binding('x', Literal(null)),), Name('x'))),), Call(Name('f'), (Literal(5),))),
                                Literal(5))

    def test_non_literal_function_parameters(self):
        """A function with non-literal default parameter values is not inlined"""
        self.assertSimplifiesTo(Let((PolyBinding(('f',), Function('f', (Binding('x', Name('y')),), Name('x'))),), Call(Name('f'), (Literal(5),))),
                                Let((PolyBinding(('f',), Function('f', (Binding('x', Name('y')),), Name('x'), ())),), Call(Name('f'), (Literal(5),))),
                                dynamic={'y'})

    def test_function_with_captures(self):
        """A function with non-literal default parameter values is not inlined"""
        self.assertSimplifiesTo(Let((PolyBinding(('f',), Function('f', (Binding('x', Literal(null)),), Name('y'))),), Call(Name('f'), (Literal(5),))),
                                Let((PolyBinding(('f',), Function('f', (Binding('x', Literal(null)),), Name('y'), ('y',))),), Call(Name('f'), (Literal(5),))),
                                dynamic={'y'})

    def test_rename(self):
        """Simple rename of a local"""
        self.assertSimplifiesTo(Let((PolyBinding(('x',), Name('y')),), Add(Name('x'), Name('y'))),
                                Add(Name('y'), Name('y')),
                                dynamic={'y'})

    def test_expr_shadowed_rename(self):
        """Rename of a local that is shadowed by a later binding to an expression"""
        self.assertSimplifiesTo(Let((PolyBinding(('y',), Add(Name('y'), Literal(5))),), Add(Name('x'), Name('y'))),
                                Let((PolyBinding(('x',), Name('y')), PolyBinding(('y',), Add(Name('y'), Literal(5)))), Add(Name('x'), Name('y'))),
                                static={'x': Name('y')}, dynamic={'y'})

    def test_expr_shadowed_rename_subexpr(self):
        """Rename of a local that is shadowed by a binding to an expression in a sub-expression"""
        self.assertSimplifiesTo(Let((PolyBinding(('x',), Name('y')),),
                                    Add(Literal(5), Let((PolyBinding(('y',), Add(Name('y'), Literal(5))),), Add(Name('x'), Name('y'))))),
                                Add(Literal(5), Let((PolyBinding(('x',), Name('y')),
                                                     PolyBinding(('y',), Add(Name('y'), Literal(5))),), Add(Name('x'), Name('y')))),
                                dynamic={'y'})

    def test_literal_shadowed_rename(self):
        """Rename of a local that is shadowed by a later binding to a literal"""
        self.assertSimplifiesTo(Let((PolyBinding(('x',), Name('y')), PolyBinding(('y',), Literal(5))), Add(Name('x'), Name('y'))),
                                Add(Name('y'), Literal(5)),
                                dynamic={'y'})

    def test_rename_shadowed_rename(self):
        """Rename of a local that is shadowed by a later binding to a rename"""
        self.assertSimplifiesTo(Let((PolyBinding(('x',), Name('y')), PolyBinding(('y',), Name('z'))), Add(Name('x'), Name('y'))),
                                Add(Name('y'), Name('z')),
                                dynamic={'y', 'z'})


class TestCall(SimplifierTestCase):
    def test_dynamic(self):
        """Dynamic calls left alone"""
        self.assertSimplifiesTo(Call(Name('x'), (Literal(5),)), Call(Name('x'), (Literal(5),)), dynamic={'x'})
        self.assertSimplifiesTo(Call(Literal(functions.sqrtv), (Name('y'),)), Call(Literal(functions.sqrtv), (Name('y'),)), dynamic={'y'})
        self.assertSimplifiesTo(Call(Literal(functions.sqrtv), (), (Binding('xs', Name('y')),)),
                                Call(Literal(functions.sqrtv), (), (Binding('xs', Name('y')),)), dynamic={'y'})

    def test_static(self):
        """Static calls to built-in functions are replaced with their return value"""
        self.assertSimplifiesTo(Call(Literal(functions.sqrtv), (Literal(25),), ()), Literal(5))
        self.assertSimplifiesTo(Call(Literal(functions.sqrtv), (), (Binding('xs', Literal(25)),)), Literal(5))

    def test_simple_inlining(self):
        """Calls to names that resolve to Function objects are inlined as let expressions"""
        func = Function('func', (Binding('x', Literal(null)),), Add(Name('x'), Literal(5)), captures=(), inlineable=True)
        self.assertSimplifiesTo(Call(Name('func'), (Add(Literal(1), Name('y')),), ()),
                                Let((PolyBinding(('x',), Add(Literal(1), Name('y'))),), Add(Name('x'), Literal(5))),
                                static={'func': func}, dynamic={'y'})

    def test_inlineable_recursive_non_literal(self):
        """Calls to inlineable, recursive functions are *not* inlined if arguments are not all literal"""
        func = Function(
            'func',
            (Binding('x', Literal(null)),),
            IfElse((IfCondition(GreaterThan(Name('x'), Literal(0)), Add(Name('x'), Call(Name('func'), (Subtract(Name('x'), Literal(1)),)))),), Literal(0)),
            captures=(), inlineable=True, recursive=True
        )
        self.assertSimplifiesTo(Call(Name('func'), (Name('y'),)), Call(Name('func'), (Name('y'),)), static={'func': func}, dynamic={'y'})

    def test_inlineable_recursive_literal(self):
        """Calls to inlineable, recursive functions *are* inlined if arguments are all literal"""
        func = Function(
            'func',
            (Binding('x', Literal(null)),),
            IfElse((IfCondition(GreaterThan(Name('x'), Literal(0)), Add(Name('x'), Call(Name('func'), (Subtract(Name('x'), Literal(1)),)))),), Literal(0)),
            captures=(), inlineable=True, recursive=True
        )
        self.assertSimplifiesTo(Call(Name('func'), (Literal(5),)), Literal(15), static={'func': func})


class TestFor(SimplifierTestCase):
    def test_dynamic(self):
        """Dynamic loop source left alone"""
        self.assertSimplifiesTo(For(('x',), Name('y'), Name('x')), For(('x',), Name('y'), Name('x')), dynamic={'y'})

    def test_recursive(self):
        """Source and body simplified"""
        self.assertSimplifiesTo(For(('x',), Name('y'), Add(Name('x'), Name('z'))),
                                For(('x',), Name('z'), Add(Name('x'), Name('z'))),
                                static={'y': Name('z')}, dynamic={'z'})
        self.assertSimplifiesTo(For(('x',), Name('y'), Add(Name('x'), Name('z'))),
                                For(('x',), Name('y'), Add(Name('x'), Name('w'))),
                                static={'z': Name('w')}, dynamic={'y', 'w'})

    def test_single_name_unroll(self):
        """Simple iteration of a single name over a literal vector"""
        self.assertSimplifiesTo(For(('x',), Literal([1, 2]), Add(Name('x'), Name('z'))),
                                Sequence((Add(Literal(1), Name('z')), Add(Literal(2), Name('z')))),
                                dynamic={'z'})

    def test_multiple_name_unroll(self):
        """Iteration of multiple names over a literal vector"""
        self.assertSimplifiesTo(For(('x', 'y'), Literal([1, 2, 3]), Call(Name('f'), (Name('x'), Name('y')), ())),
                                Sequence((Call(Name('f'), (Literal(1), Literal(2))), Call(Name('f'), (Literal(3), Literal(null))))),
                                dynamic={'f'})


class TestIfElse(SimplifierTestCase):
    def test_dynamic(self):
        """Dynamic condition left alone"""
        self.assertSimplifiesTo(IfElse((IfCondition(Name('x'), Literal(5)),), None), IfElse((IfCondition(Name('x'), Literal(5)),), None), dynamic={'x'})

    def test_recursive(self):
        """Sub expressions are all simplified"""
        self.assertSimplifiesTo(IfElse((IfCondition(Name('x'), Name('y')),), Name('z')),
                                IfElse((IfCondition(Name('w'), Name('y')),), Name('z')),
                                static={'x': Name('w')}, dynamic={'w', 'y', 'z'})
        self.assertSimplifiesTo(IfElse((IfCondition(Name('x'), Name('y')),), Name('z')),
                                IfElse((IfCondition(Name('x'), Name('w')),), Name('z')),
                                static={'y': Name('w')}, dynamic={'x', 'w', 'z'})
        self.assertSimplifiesTo(IfElse((IfCondition(Name('x'), Name('y')),), Name('z')),
                                IfElse((IfCondition(Name('x'), Name('y')),), Name('w')),
                                static={'z': Name('w')}, dynamic={'x', 'y', 'w'})

    def test_true_condition(self):
        """A true condition simplifies to the then expression"""
        self.assertSimplifiesTo(IfElse((IfCondition(Literal(true), Name('y')),), Name('z')), Name('y'), dynamic={'y', 'z'})

    def test_false_condition_else(self):
        """A false condition simplifies to the else expression"""
        self.assertSimplifiesTo(IfElse((IfCondition(Literal(false), Name('y')),), Name('z')), Name('z'), dynamic={'y', 'z'})

    def test_false_condition_no_else(self):
        """A false condition without an else simplifies to null"""
        self.assertSimplifiesTo(IfElse((IfCondition(Literal(false), Name('y')),), None), Literal(null), dynamic={'y'})

    def test_false_condition_1_of_2(self):
        """A false 1st condition is removed when more than one condition"""
        self.assertSimplifiesTo(IfElse((IfCondition(Literal(false), Name('x')), IfCondition(Name('w'), Name('y'))), Name('z')),
                                IfElse((IfCondition(Name('w'), Name('y')),), Name('z')), dynamic={'w', 'x', 'y', 'z'})

    def test_true_condition_2_of_2(self):
        """A true 2nd condition simplifies to an if with the 2nd then as the else"""
        self.assertSimplifiesTo(IfElse((IfCondition(Name('w'), Name('x')), IfCondition(Literal(true), Name('y'))), Name('z')),
                                IfElse((IfCondition(Name('w'), Name('x')),), Name('y')), dynamic={'w', 'x', 'y', 'z'})

    def test_false_condition_2_of_2(self):
        """A false 2nd condition is removed"""
        self.assertSimplifiesTo(IfElse((IfCondition(Name('w'), Name('x')), IfCondition(Literal(false), Name('y')),), Name('z')),
                                IfElse((IfCondition(Name('w'), Name('x')),), Name('z')), dynamic={'w', 'x', 'y', 'z'})

    def test_true_condition_2_of_3(self):
        """A true 2nd condition simplifies to an if with the 2nd then as the else; 3rd condition removed"""
        self.assertSimplifiesTo(IfElse((IfCondition(Name('w'), Name('x')),
                                        IfCondition(Literal(true), Name('y')),
                                        IfCondition(Name('a'), Name('b'))), Name('z')),
                                IfElse((IfCondition(Name('w'), Name('x')),), Name('y')), dynamic={'w', 'x', 'y', 'z', 'a', 'b'})


class TestImport(SimplifierTestCase):
    def test_recursive(self):
        """Imports are left alone except for the sub-expression being simplified"""
        self.assertSimplifiesTo(Import(('x', 'y'), Name('m'), Name('z')), Import(('x', 'y'), Literal('module.fl'), Literal(5)),
                                static={'m': 'module.fl', 'z': 5})


class TestFunction(SimplifierTestCase):
    def test_simple_inlineable(self):
        """A simple, unsimplifiable function with no external references will have empty captures and will be inlineable"""
        start_func = Function('func', (Binding('x', Literal(null)),), Add(Name('x'), Literal(5)))
        simpl_func = Function('func', (Binding('x', Literal(null)),), Add(Name('x'), Literal(5)), captures=(), inlineable=True)
        self.assertSimplifiesTo(start_func, simpl_func)

    def test_recursive(self):
        """Parameter defaults and the body of the function are simplified (again, this is an inlineable function)"""
        start_func = Function('func', (Binding('x', Name('y')),), Add(Name('x'), Name('z')))
        simpl_func = Function('func', (Binding('x', Literal(null)),), Add(Name('x'), Literal(5)), captures=(), inlineable=True)
        self.assertSimplifiesTo(start_func, simpl_func, static={'y': null, 'z': 5})

    def test_non_literal_default(self):
        """A function that has a dynamic parameter default won't be inlineable"""
        start_func = Function('func', (Binding('x', Name('y')),), Add(Name('x'), Literal(5)))
        simpl_func = Function('func', (Binding('x', Name('y')),), Add(Name('x'), Literal(5)), captures=(), inlineable=False)
        self.assertSimplifiesTo(start_func, simpl_func, dynamic={'y'})

    def test_simple_capture(self):
        """A function that references an external name will have that noted in its captures and won't be inlineable"""
        start_func = Function('func', (Binding('x', Literal(null)),), Add(Name('x'), Name('y')))
        simpl_func = Function('func', (Binding('x', Literal(null)),), Add(Name('x'), Name('y')), captures=('y',))
        self.assertSimplifiesTo(start_func, simpl_func)

    def test_simple_recursive(self):
        """A function that references only itself will be marked as recursive and will be inlineable"""
        start_func = Function('func', (Binding('x', Literal(null)),), Add(Name('x'), Call(Name('func'), (Name('x'),))))
        simpl_func = Function('func', (Binding('x', Literal(null)),), Add(Name('x'), Call(Name('func'), (Name('x'),))),
                              captures=(), inlineable=True, recursive=True)
        self.assertSimplifiesTo(start_func, simpl_func)


class TestExport(SimplifierTestCase):
    def test_empty(self):
        self.assertSimplifiesTo(Export(), Export())

    def test_explicit(self):
        self.assertSimplifiesTo(Export(), Export({'x': Vector(5)}), static={'x': 5})

    def test_let_explicit(self):
        self.assertSimplifiesTo(Let((PolyBinding(('x',), Literal(5)),), Export()), Export({'x': Vector(5)}))


class TestTop(SimplifierTestCase):
    def test_recursive(self):
        """Top expression is simplified"""
        self.assertSimplifiesTo(Top((), Name('x')), Top((), Name('y')), static={'x': Name('y')}, dynamic={'y'})
