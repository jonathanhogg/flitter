"""
Flitter parser tests
"""

import unittest

from flitter.language.tree import (
    Literal, Name, Top, Export, Sequence,
    Positive, Negative, Power, Add, Subtract, Multiply, Divide, FloorDivide, Modulo,
    # Contains, EqualTo, NotEqualTo, LessThan, GreaterThan, LessThanOrEqualTo, GreaterThanOrEqualTo,
    # Not, And, Or, Xor,
    # Range, Slice, Lookup,
    # Tag, Attributes, Append,
    # Let, Call, For, IfElse,
    # Import, Function,
    # Binding, PolyBinding, IfCondition
)
from flitter.model import Vector, Node
from flitter.language.parser import parse, ParseError


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


class TestAtoms(ParserTestCase):
    def test_numbers(self):
        self.assertParsesTo("0", Literal(0))
        self.assertParsesTo("1234567890", Literal(1234567890))
        self.assertParsesTo("0123456789", Literal(123456789))
        self.assertParsesTo("1_234_567_890", Literal(1234567890))
        self.assertParsesTo("1.234567890", Literal(1.23456789))
        self.assertParsesTo("1.234_567_890", Literal(1.23456789))
        self.assertParsesTo("1e9", Literal(1000000000))
        self.assertParsesTo("1.23456789e9", Literal(1234567890))
        self.assertParsesTo("1.234_567_890e9", Literal(1234567890))
        self.assertParsesTo("123.456_789e9", Literal(123456789000))
        self.assertParsesTo("1.23456789e-9", Literal(1.23456789e-9))
        self.assertParsesTo("1.23456789E-99", Literal(1.23456789e-99))
        self.assertParsesTo("1.2_3_4_5_6_7_8_9E1_0_0_0", Literal(1.23456789e1000))

    def test_timecodes(self):
        self.assertParsesTo("0:0", Literal(0))
        self.assertParsesTo("0:0.0", Literal(0))
        self.assertParsesTo("0:0:0", Literal(0))
        self.assertParsesTo("0:0:0.0", Literal(0))
        self.assertParsesTo("000:00:00.000", Literal(0))
        self.assertParsesTo("0:1", Literal(1))
        self.assertParsesTo("000:00:01.000", Literal(1))
        self.assertParsesTo("0:0.1", Literal(0.1))
        self.assertParsesTo("000:00:00.100", Literal(0.1))
        self.assertParsesTo("1:0", Literal(60))
        self.assertParsesTo("000:01:00.000", Literal(60))
        self.assertParsesTo("1:0:0", Literal(3600))
        self.assertParsesTo("001:00:00.000", Literal(3600))
        self.assertParsesTo("1:1:1", Literal(3661))
        self.assertParsesTo("001:01:01.000", Literal(3661))
        self.assertParsesTo("001:01:01.100", Literal(3661.1))
        self.assertParsesTo("999:59:59.999", Literal(3599999.999))
        with self.assertRaises(ParseError):
            parse("0:0:0:0")
        with self.assertRaises(ParseError):
            parse("0:60")
        with self.assertRaises(ParseError):
            parse("60:00")

    def test_si_prefixes(self):
        self.assertParsesTo('1T', Literal(1e12))
        self.assertParsesTo('1G', Literal(1e9))
        self.assertParsesTo('1M', Literal(1e6))
        self.assertParsesTo('1k', Literal(1e3))
        self.assertParsesTo('1m', Literal(1e-3))
        self.assertParsesTo('1u', Literal(1e-6))
        self.assertParsesTo('1µ', Literal(1e-6))
        self.assertParsesTo('1n', Literal(1e-9))
        self.assertParsesTo('1p', Literal(1e-12))

    def test_strings(self):
        self.assertParsesTo('"Hello world!"', Literal("Hello world!"))
        self.assertParsesTo("'Hello world!'", Literal("Hello world!"))
        self.assertParsesTo('''"Hello 'world!'"''', Literal("Hello 'world!'"))
        self.assertParsesTo("""'Hello "world!"'""", Literal('Hello "world!"'))
        self.assertParsesTo('''"""Hello
world!"""''', Literal("Hello\nworld!"))
        self.assertParsesTo("""'''Hello
world!'''""", Literal("Hello\nworld!"))
        self.assertParsesTo("'Hello\\nworld!'", Literal("Hello\nworld!"))
        self.assertParsesTo("'Hello Hafnarfjörður!'", Literal("Hello Hafnarfjörður!"))

    def test_symbols(self):
        self.assertParsesTo(":hello", Literal(Vector.symbol("hello")))
        self.assertParsesTo(":_world", Literal(Vector.symbol("_world")))
        self.assertParsesTo(":hello_world", Literal(Vector.symbol("hello_world")))
        self.assertParsesTo(":_", Literal(Vector.symbol("_")))
        self.assertParsesTo(":Hafnarfjörður", Literal(Vector.symbol("Hafnarfjörður")))
        self.assertParsesTo(":hello123", Literal(Vector.symbol("hello123")))

    def test_nodes(self):
        self.assertParsesTo("!hello", Literal(Node("hello")))
        self.assertParsesTo("!_world", Literal(Node("_world")))
        self.assertParsesTo("!hello_world", Literal(Node("hello_world")))
        self.assertParsesTo("!_", Literal(Node("_")))
        self.assertParsesTo("!Hafnarfjörður", Literal(Node("Hafnarfjörður")))
        self.assertParsesTo("!hello123", Literal(Node("hello123")))

    def test_names(self):
        self.assertParsesTo("hello", Name("hello"))
        self.assertParsesTo("_world", Name("_world"))
        self.assertParsesTo("hello_world", Name("hello_world"))
        self.assertParsesTo("_", Name("_"))
        self.assertParsesTo("Hafnarfjörður", Name("Hafnarfjörður"))
        self.assertParsesTo("hello123", Name("hello123"))


class TestPrecedence(ParserTestCase):
    def test_maths_literals(self):
        self.assertParsesTo("0-1//2%3++4*-5**6/7",
                            Add(Subtract(Literal(0), Modulo(FloorDivide(Literal(1), Literal(2)), Literal(3))),
                                Divide(Multiply(Positive(Literal(4)), Negative(Power(Literal(5), Literal(6)))), Literal(7))))

    def test_maths_names(self):
        self.assertParsesTo("a-b//c%d++e*-f**g/h",
                            Add(Subtract(Name('a'), Modulo(FloorDivide(Name('b'), Name('c')), Name('d'))),
                                Divide(Multiply(Positive(Name('e')), Negative(Power(Name('f'), Name('g')))), Name('h'))))

    def test_maths_parenthesised(self):
        self.assertParsesTo("(a-b)//c%d++e*(-f)**(g/h)",
                            Add(Modulo(FloorDivide(Subtract(Name('a'), Name('b')), Name('c')), Name('d')),
                                Multiply(Positive(Name('e')), Power(Negative(Name('f')), Divide(Name('g'), Name('h'))))))
