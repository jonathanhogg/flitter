"""
Tests of the language simplifier
"""

import unittest

from flitter.model import Vector, Node, null, true, false
from flitter.language.tree import (Top, Pragma, Import, Sequence, InlineSequence, Literal, Name, FunctionName, Lookup, LookupLiteral,
                                   Range, Negative, Positive, Not, Add, Subtract, Multiply, Divide, FloorDivide, Modulo, Power,
                                   EqualTo, NotEqualTo, LessThan, GreaterThan, LessThanOrEqualTo, GreaterThanOrEqualTo, And, Or, Xor,
                                   Slice, FastSlice, Call, NodeModifier, Tag, Attributes,
                                   Append, Let, StoreGlobal, InlineLet, For, IfCondition, IfElse, Function)


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
    def test_literal(self):
        expression = Negative(Literal(Vector(5)))
        simplified = expression.simplify()
        self.assertIsInstance(simplified, Literal)
        self.assertEqual(simplified.value, Vector(-5))
