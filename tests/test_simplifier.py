"""
Tests of the language simplifier
"""

import unittest

from flitter.model import Vector, Node, null, true, false
from flitter.language.tree import (Top, Pragma, Import, Sequence, InlineSequence, Literal, Name, FunctionName, Lookup,
                                   LookupLiteral, Range, UnaryOperation, Negative, Positive, Not, BinaryOperation,
                                   MathsBinaryOperation, Add, Subtract, Multiply, Divide, FloorDivide, Modulo, Power,
                                   Comparison, EqualTo, NotEqualTo, LessThan, GreaterThan, LessThanOrEqualTo, GreaterThanOrEqualTo,
                                   And, Or, Xor, Slice, FastSlice, Call, NodeModifier, Tag, Attributes, FastAttributes, Search,
                                   Append, Prepend, Let, StoreGlobal, InlineLet, For, IfCondition, IfElse, Function)


class TestName(unittest.TestCase):
    def setUp(self):
        self.expr = Name('x')

    def test_static_set(self):
        expr = self.expr.simplify(variables={'x': Vector(5)})
        self.assertIsInstance(expr, Literal)
        self.assertEqual(expr.value, Vector(5))

    def test_static_unset(self):
        expr = self.expr.simplify(variables={'y': Vector(5)})
        self.assertIsInstance(expr, Literal)
        self.assertEqual(expr.value, null)

    def test_dynamic(self):
        expr = self.expr.simplify(variables={'y': Vector(5)}, undefined={'x'})
        self.assertIsInstance(expr, Name)
        self.assertEqual(expr.name, 'x')
