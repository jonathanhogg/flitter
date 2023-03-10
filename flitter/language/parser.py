"""
Flitter language compiler
"""

# pylama:ignore=R0201,C0103,R0903

from ast import literal_eval
from pathlib import Path
from sys import intern

from lark import Lark, Transformer
from lark.indenter import Indenter
from lark.visitors import v_args

from .. import model
from . import tree


class FlitterIndenter(Indenter):
    NL_type = '_NL'
    OPEN_PAREN_types = ['_LPAR', '_LBRA']
    CLOSE_PAREN_types = ['_RPAR', '_RBRA']
    INDENT_type = '_INDENT'
    DEDENT_type = '_DEDENT'
    tab_len = 8


@v_args(inline=True)
class FlitterTransformer(Transformer):
    def NAME(self, token):
        return intern(str(token))

    def NUMBER(self, token):
        return model.Vector.coerce(float(token))

    def TAG(self, token):
        return intern(str(token)[1:])

    def SYMBOL(self, token):
        return model.Vector(intern(str(token)[1:]))

    def STRING(self, token):
        return model.Vector(intern(literal_eval(token)))

    def QUERY(self, token):
        return model.Query(token[1:-1])

    def range(self, start, stop, step):
        return tree.Range(tree.Literal(model.null) if start is None else start, stop, tree.Literal(model.null) if step is None else step)

    def inline_if_else(self, then, condition, else_):
        return tree.IfElse((tree.Test(condition, then),), else_)

    def inline_loop(self, body, names, source):
        return tree.For(names, source, body)

    tuple = v_args(inline=False)(tuple)

    add = tree.Add
    append = tree.Append
    attributes = tree.Attributes
    binding = tree.Binding
    bool = tree.Literal
    call = tree.Call
    divide = tree.Divide
    eq = tree.EqualTo
    floor_divide = tree.FloorDivide
    function = tree.Function
    ge = tree.GreaterThanOrEqualTo
    gt = tree.GreaterThan
    if_else = tree.IfElse
    inline_let = tree.InlineLet
    inline_sequence = tree.InlineSequence
    le = tree.LessThanOrEqualTo
    let = tree.Let
    literal = tree.Literal
    logical_and = tree.And
    logical_not = tree.Not
    logical_or = tree.Or
    logical_xor = tree.Xor
    lookup = tree.Lookup
    loop = tree.For
    lt = tree.LessThan
    modulo = tree.Modulo
    multiply = tree.Multiply
    name = tree.Name
    ne = tree.NotEqualTo
    neg = tree.Negative
    node = tree.Node
    poly_binding = tree.PolyBinding
    pos = tree.Positive
    power = tree.Power
    pragma = tree.Pragma
    prepend = tree.Prepend
    search = tree.Search
    sequence = tree.Sequence
    slice = tree.Slice
    subtract = tree.Subtract
    tag = tree.Tag
    test = tree.Test
    top = tree.Top


GRAMMAR = (Path(__file__).parent / 'grammar.lark').open('r', encoding='utf8').read()
PARSER = Lark(GRAMMAR, postlex=FlitterIndenter(), regex=True, start='top', maybe_placeholders=True,
              parser='lalr', transformer=FlitterTransformer())


def parse(source):
    return PARSER.parse(source)
