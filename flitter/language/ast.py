"""
Flitter language AST
"""

# pylama:ignore=R0903,E1123

from dataclasses import dataclass
from typing import Tuple, Any


class Expression:
    pass


@dataclass(frozen=True, slots=True)
class Pragma(Expression):
    name: str
    expr: Expression


@dataclass(frozen=True, slots=True)
class Sequence(Expression):
    expressions: Tuple[Expression, ...]


@dataclass(frozen=True, slots=True)
class Literal(Expression):
    value: Any


@dataclass(frozen=True, slots=True)
class Name(Expression):
    name: str


@dataclass(frozen=True, slots=True)
class Lookup(Expression):
    key: Expression


@dataclass(frozen=True, slots=True)
class Range(Expression):
    start: Expression
    stop: Expression
    step: Expression


@dataclass(frozen=True, slots=True)
class UnaryOperation(Expression):
    expr: Expression


class Negative(UnaryOperation):
    pass


class Positive(UnaryOperation):
    pass


class Not(UnaryOperation):
    pass


@dataclass(frozen=True, slots=True)
class BinaryOperation(Expression):
    left: Expression
    right: Expression


class MathsBinaryOperation(BinaryOperation):
    pass


class Add(MathsBinaryOperation):
    pass


class Subtract(MathsBinaryOperation):
    pass


class Multiply(MathsBinaryOperation):
    pass


class Divide(MathsBinaryOperation):
    pass


class FloorDivide(MathsBinaryOperation):
    pass


class Modulo(MathsBinaryOperation):
    pass


class Power(MathsBinaryOperation):
    pass


class Comparison(BinaryOperation):
    pass


class EqualTo(Comparison):
    pass


class NotEqualTo(Comparison):
    pass


class LessThan(Comparison):
    pass


class GreaterThan(Comparison):
    pass


class LessThanOrEqualTo(Comparison):
    pass


class GreaterThanOrEqualTo(Comparison):
    pass


class And(BinaryOperation):
    pass


class Or(BinaryOperation):
    pass


@dataclass(frozen=True, slots=True)
class Slice(Expression):
    expr: Expression
    index: Expression


@dataclass(frozen=True, slots=True)
class Call(Expression):
    function: Expression
    args: Tuple[Expression, ...]


@dataclass(frozen=True, slots=True)
class Node(Expression):
    kind: str
    tags: Tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Attribute(Expression):
    node: Expression
    name: str
    expr: Expression


@dataclass(frozen=True, slots=True)
class Search(Expression):
    query: str


@dataclass(frozen=True, slots=True)
class Append(Expression):
    node: Expression
    children: Expression


@dataclass(frozen=True, slots=True)
class Prepend(Expression):
    node: Expression
    children: Expression


@dataclass(frozen=True, slots=True)
class Binding:
    name: str
    expr: Expression


@dataclass(frozen=True, slots=True)
class Let(Expression):
    bindings: Tuple[Binding, ...]


@dataclass(frozen=True, slots=True)
class InlineLet(Expression):
    bindings: Tuple[Binding, ...]
    body: Expression


@dataclass(frozen=True, slots=True)
class For(Expression):
    name: str
    source: Expression
    body: Expression


@dataclass(frozen=True, slots=True)
class Test:
    condition: Expression
    then: Expression


@dataclass(frozen=True, slots=True)
class IfElse(Expression):
    tests: Tuple[Test, ...]
    else_: Expression
