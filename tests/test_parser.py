"""
Flitter parser tests
"""

import unittest

from flitter.language.tree import (
    Literal, Name, Top, Export, Sequence,
    Positive, Negative, Power, Add, Subtract, Multiply, Divide, FloorDivide, Modulo,
    # Ceil, Floor, Fract,
    # Contains, EqualTo, NotEqualTo, LessThan, GreaterThan, LessThanOrEqualTo, GreaterThanOrEqualTo,
    # Not, And, Or, Xor,
    # Range, Slice, Lookup,
    # Tag, Attributes, Append,
    # Let, Call, For, IfElse,
    # Import, Function,
    # Binding, PolyBinding, IfCondition
)
from flitter.language.parser import parse


class ParserTestCase(unittest.TestCase):
    def assertParsesTo(self, code, expression):
        self.maxDiff = 1000
        if not isinstance(expression, Top):
            if not isinstance(expression, Sequence):
                expression = Sequence((expression, Export(None)))
            elif not isinstance(expression.expressions[-1], Export):
                expression = Sequence(expression.expressions + (Export(None),))
            expression = Top((), expression)
        self.assertEqual(repr(parse(code)), repr(expression))


class TestPrecedence(ParserTestCase):
    def test_maths_literals(self):
        self.assertParsesTo("0-1//2%3++4*-5**6/7",
                            Add(Subtract(Literal(0), Modulo(FloorDivide(Literal(1), Literal(2)), Literal(3))),
                                Divide(Multiply(Positive(Literal(4)), Negative(Power(Literal(5), Literal(6)))), Literal(7))))

    def test_maths_names(self):
        self.assertParsesTo("a-b//c%d++e*-f**g/h",
                            Add(Subtract(Name('a'), Modulo(FloorDivide(Name('b'), Name('c')), Name('d'))),
                                Divide(Multiply(Positive(Name('e')), Negative(Power(Name('f'), Name('g')))), Name('h'))))
