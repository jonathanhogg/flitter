"""
Flitter language compiler
"""

from ast import literal_eval
from pathlib import Path
from sys import intern

from lark import Lark, Transformer
from lark.exceptions import UnexpectedInput
from lark.indenter import Indenter
from lark.visitors import v_args

from .. import model
from . import tree


SI_PREFIXES = {'p': 1e-12, 'n': 1e-9, 'u': 1e-6, 'µ': 1e-6, 'm': 1e-3, 'k': 1e3, 'M': 1e6, 'G': 1e9, 'T': 1e12}


class FlitterIndenter(Indenter):
    NL_type = '_NL'
    OPEN_PAREN_types = ['_LPAREN']
    CLOSE_PAREN_types = ['_RPAREN']
    INDENT_type = '_INDENT'
    DEDENT_type = '_DEDENT'
    tab_len = 8


class ParseError(Exception):
    def __init__(self, msg, line, column, context):
        super().__init__(msg)
        self.line = line
        self.column = column
        self.context = context


@v_args(inline=True)
class FlitterTransformer(Transformer):
    def NAME(self, token):
        return intern(str(token))

    def NODE(self, token):
        return model.Vector(model.Node(intern(str(token)[1:])))

    def NUMBER(self, token):
        multiplier = 1
        if token[-1] in SI_PREFIXES:
            multiplier = SI_PREFIXES[token[-1]]
            token = token[:-1]
        return model.Vector.coerce(float(token) * multiplier)

    def TAG(self, token):
        return intern(str(token)[1:])

    def SYMBOL(self, token):
        return model.Vector.symbol(intern(str(token)[1:]))

    def STRING(self, token):
        return model.Vector(literal_eval(token))

    def range(self, start, stop, step):
        return tree.Range(tree.Literal(model.null) if start is None else start, stop, tree.Literal(model.null) if step is None else step)

    def inline_if_else(self, then, condition, else_):
        return tree.IfElse((tree.IfCondition(condition, then),), else_)

    def inline_loop(self, body, names, source):
        return tree.For(names, source, body)

    def call(self, function, args):
        args = list(args)
        bindings = []
        while args and isinstance(args[-1], tree.Binding):
            bindings.insert(0, args.pop())
        for arg in args:
            if isinstance(arg, tree.Binding):
                raise TypeError("Cannot mix positional and keyword arguments")
        return tree.Call(function, tuple(args) if args else None, tuple(bindings) if bindings else None)

    def template_call(self, function, bindings, sequence):
        if sequence is not None:
            return tree.Call(function, (sequence,), bindings)
        return tree.Call(function, (tree.Literal(model.null),), bindings or None)

    tuple = v_args(inline=False)(tuple)

    add = tree.Add
    append = tree.Append
    attributes = tree.Attributes
    binding = tree.Binding
    bool = tree.Literal
    divide = tree.Divide
    eq = tree.EqualTo
    file_import = tree.Import
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
    poly_binding = tree.PolyBinding
    pos = tree.Positive
    power = tree.Power
    pragma = tree.Pragma
    sequence = tree.Sequence
    slice = tree.Slice
    subtract = tree.Subtract
    tag = tree.Tag
    test = tree.IfCondition
    top = tree.Top


GRAMMAR = (Path(__file__).parent / 'grammar.lark').read_text(encoding='utf8')
PARSER = Lark(GRAMMAR, postlex=FlitterIndenter(), regex=True, start='top', maybe_placeholders=True,
              parser='lalr', transformer=FlitterTransformer())


def parse(source):
    try:
        return PARSER.parse(source)
    except UnexpectedInput as exc:
        raise ParseError(f"Parse error in source at line {exc.line} column {exc.column}",
                         line=exc.line, column=exc.column, context=exc.get_context(source).rstrip()) from exc
