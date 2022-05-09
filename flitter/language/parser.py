"""
Flitter language compiler
"""

# pylama:ignore=R0201,C0103

from pathlib import Path

from lark import Lark, Transformer
from lark.indenter import Indenter
from lark.visitors import v_args

from . import ast
from .. import model


class FlitterIndenter(Indenter):
    NL_type = '_NL'
    OPEN_PAREN_types = ['_LPAR', '_LBRA']
    CLOSE_PAREN_types = ['_RPAR', '_RBRA']
    INDENT_type = '_INDENT'
    DEDENT_type = '_DEDENT'
    tab_len = 8


@v_args(inline=True)
class FlitterTransformer(Transformer):
    NAME = str

    def SIGNED_NUMBER(self, token):
        return model.Vector((float(token),))

    def ESCAPED_STRING(self, token):
        return model.Vector((token[1:-1].encode('utf-8').decode('unicode_escape'),))

    def QUERY(self, token):
        return model.Query(token[1:-1])

    def range(self, start, stop, step):
        return ast.Range(ast.Literal(model.null) if start is None else start, stop, ast.Literal(model.null) if step is None else step)

    tuple = v_args(inline=False)(tuple)

    add = ast.Add
    append = ast.Append
    attribute = ast.Attribute
    binding = ast.Binding
    bool = ast.Literal
    call = ast.Call
    divide = ast.Divide
    eq = ast.EqualTo
    floor_divide = ast.FloorDivide
    ge = ast.GreaterThanOrEqualTo
    gt = ast.GreaterThan
    if_else = ast.IfElse
    inline_let = ast.InlineLet
    le = ast.LessThanOrEqualTo
    let = ast.Let
    literal = ast.Literal
    logical_and = ast.And
    logical_not = ast.Not
    logical_or = ast.Or
    lookup = ast.Lookup
    loop = ast.For
    lt = ast.LessThan
    modulo = ast.Modulo
    multiply = ast.Multiply
    name = ast.Name
    ne = ast.NotEqualTo
    neg = ast.Negative
    node = ast.Node
    power = ast.Power
    pos = ast.Positive
    pragma = ast.Pragma
    prepend = ast.Prepend
    search = ast.Search
    sequence = v_args(inline=False)(ast.Sequence)
    slice = ast.Slice
    subtract = ast.Subtract
    tags = v_args(inline=False)(tuple)
    test = ast.Test
    tests = v_args(inline=False)(tuple)


GRAMMAR = (Path(__file__).parent / 'grammar.lark').open('r').read()
PARSER = Lark(GRAMMAR, postlex=FlitterIndenter(), regex=True, start='sequence', maybe_placeholders=True)


def parse(source):
    return FlitterTransformer().transform(PARSER.parse(source))
