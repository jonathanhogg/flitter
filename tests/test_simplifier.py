"""
Tests of the simplifier, which is part of the language AST
"""

import unittest

from flitter.model import Vector, null
from flitter.language.tree import (Literal, Name, Negative, Positive, Multiply, Divide)


# AST classes still to test:
#
# Top, Pragma, Import, Sequence, InlineSequence, FunctionName, Lookup, LookupLiteral,
# Range, Positive, Not, Add, Subtract, Multiply, Divide, FloorDivide, Modulo, Power,
# EqualTo, NotEqualTo, LessThan, GreaterThan, LessThanOrEqualTo, GreaterThanOrEqualTo, And, Or, Xor,
# Slice, FastSlice, Call, NodeModifier, Tag, Attributes,
# Append, Let, StoreGlobal, InlineLet, For, IfCondition, IfElse, Function


class TestLiteral(unittest.TestCase):
    def test_unchanged(self):
        """Literals should be unaffected by simplification."""
        expression = Literal(Vector([1, 2, 3]))
        simplified = expression.simplify()
        self.assertIsInstance(simplified, Literal)
        self.assertEqual(simplified.value, Vector([1, 2, 3]))


class TestName(unittest.TestCase):
    def test_static(self):
        """Names with static values should be replaced with their literal value."""
        expression = Name('x')
        simplified = expression.simplify(static={'x': Vector(5)})
        self.assertIsInstance(simplified, Literal)
        self.assertEqual(simplified.value, Vector(5))

    def test_undefined(self):
        """Undefined names should be replaced with literal nulls."""
        expression = Name('x')
        simplified = expression.simplify()
        self.assertIsInstance(simplified, Literal)
        self.assertEqual(simplified.value, null)

    def test_dynamic(self):
        """Names that are defined, but with unknown values - i.e., dynamic locals - should be unchanged."""
        expression = Name('x')
        simplified = expression.simplify(dynamic={'x'})
        self.assertIsInstance(simplified, Name)
        self.assertEqual(simplified.name, 'x')


class TestNegative(unittest.TestCase):
    def test_numeric_literal(self):
        """Numeric literals get negated"""
        expression = Negative(Literal(Vector(5)))
        simplified = expression.simplify()
        self.assertIsInstance(simplified, Literal)
        self.assertEqual(simplified.value, Vector(-5))

    def test_non_numeric_literal(self):
        """Non-numeric literals become nulls"""
        expression = Negative(Literal(Vector('foo')))
        simplified = expression.simplify()
        self.assertIsInstance(simplified, Literal)
        self.assertEqual(simplified.value, null)

    def test_double_negative(self):
        """Double-negatives become positive"""
        expression = Negative(Negative(Name('x')))
        simplified = expression.simplify(dynamic={'x'})
        self.assertIsInstance(simplified, Positive)
        self.assertIsInstance(simplified.expr, Name)
        self.assertEqual(simplified.expr.name, 'x')

    def test_multiplication(self):
        """Half-literal multiplication has negative pushed into literal"""
        expression = Negative(Multiply(Literal(Vector(5)), Name('x')))
        simplified = expression.simplify(dynamic={'x'})
        self.assertIsInstance(simplified, Multiply)
        self.assertIsInstance(simplified.left, Literal)
        self.assertEqual(simplified.left.value, Vector(-5))
        self.assertIsInstance(simplified.right, Name)
        self.assertEqual(simplified.right.name, 'x')
        # And the other way round:
        expression = Negative(Multiply(Name('x'), Literal(Vector(5))))
        simplified = expression.simplify(dynamic={'x'})
        self.assertIsInstance(simplified, Multiply)
        self.assertIsInstance(simplified.left, Name)
        self.assertEqual(simplified.left.name, 'x')
        self.assertIsInstance(simplified.right, Literal)
        self.assertEqual(simplified.right.value, Vector(-5))

    def test_division(self):
        """Half-literal division has negative pushed into literal"""
        expression = Negative(Divide(Literal(Vector(5)), Name('x')))
        simplified = expression.simplify(dynamic={'x'})
        self.assertIsInstance(simplified, Divide)
        self.assertIsInstance(simplified.left, Literal)
        self.assertEqual(simplified.left.value, Vector(-5))
        self.assertIsInstance(simplified.right, Name)
        self.assertEqual(simplified.right.name, 'x')
        # Note that round the other way, a second simplification rule comes into
        # play and the division is turned into a multiplication by the inverse of
        # the literal!
        expression = Negative(Divide(Name('x'), Literal(Vector(5))))
        simplified = expression.simplify(dynamic={'x'})
        self.assertIsInstance(simplified, Multiply)
        self.assertIsInstance(simplified.left, Name)
        self.assertEqual(simplified.left.name, 'x')
        self.assertIsInstance(simplified.right, Literal)
        self.assertEqual(simplified.right.value, Vector(-0.2))
