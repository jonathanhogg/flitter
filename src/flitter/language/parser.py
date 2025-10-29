"""
Flitter language compiler
"""

from ast import literal_eval
from pathlib import Path
from sys import intern

from lark import Lark, Transformer
from lark.exceptions import UnexpectedInput
from lark.indenter import Indenter
from lark.lexer import Token
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

    def process(self, stream):
        for token in super().process(stream):
            yield token
        yield Token('_EOF', '', start_pos=token.end_pos, end_pos=token.end_pos)


class ParseError(Exception):
    def __init__(self, msg, line, column, context):
        super().__init__(msg)
        self.line = line
        self.column = column
        self.context = context


def convert_timecode_to_float(t):
    parts = t.split(':')
    if not 1 <= len(parts) <= 3:
        raise ValueError("Incorrect format")
    seconds = float(parts[-1])
    if len(parts) > 1:
        seconds += 60*int(parts[-2])
    if len(parts) > 2:
        seconds += 3600*int(parts[-3])
    return seconds


def convert_number_to_float(t):
    multiplier = 1
    if len(t) > 1 and t[-1] in SI_PREFIXES:
        multiplier = SI_PREFIXES[t[-1]]
        t = t[:-1]
    return float(t) * multiplier


@v_args(inline=True)
class FlitterTransformer(Transformer):
    def NAME(self, token):
        return intern(str(token))

    def NODE(self, token):
        return model.Vector(model.Node(intern(str(token)[1:])))

    def NUMBER(self, token):
        return model.Vector.coerce(convert_number_to_float(token))

    def TIMECODE(self, token):
        return model.Vector.coerce(convert_timecode_to_float(token))

    def TAG(self, token):
        return intern(str(token)[1:])

    def SYMBOL(self, token):
        return model.Vector.symbol(str(token)[1:])

    def STRING(self, token):
        return model.Vector(intern(literal_eval(token)))

    def range(self, start, stop, step):
        return tree.Range(tree.Literal(model.null) if start is None else start, stop, tree.Literal(model.null) if step is None else step)

    def inline_let(self, expr, bindings):
        return tree.Let(bindings, expr)

    def inline_if_else(self, then, condition, else_):
        return tree.IfElse((tree.IfCondition(condition, then),), else_)

    def inline_loop(self, body, names, source):
        return tree.For(names, source, body)

    def loop(self, iterators, expr):
        while iterators:
            expr = tree.For(iterators[-2], iterators[-1], expr)
            iterators = iterators[:-2]
        return expr

    def call(self, function, args):
        args = list(args)
        bindings = []
        while args and isinstance(args[-1], tree.Binding):
            bindings.insert(0, args.pop())
        return tree.Call(function, tuple(args) if args else None, tuple(bindings) if bindings else None)

    def template_call(self, function, bindings, sequence):
        if sequence is not None:
            return tree.Call(function, (sequence,), bindings or None)
        return tree.Call(function, None, bindings or None)

    def function(self, name, parameters, body, sequence):
        return tree.Let((tree.PolyBinding((name,), tree.Function(name, parameters, body)),), sequence)

    def sequence_let(self, names, value, sequence):
        return tree.Let((tree.PolyBinding(names, value),), sequence)

    def anonymous_function(self, parameters, body):
        return tree.Function('<anon>', parameters, body)

    def sequence(self, *expressions):
        return tree.Sequence(expressions)

    tuple = v_args(inline=False)(tuple)

    add = tree.Add
    append = tree.Append
    attributes = tree.Attributes
    binding = tree.Binding
    bool = tree.Literal
    condition = tree.IfCondition
    contains = tree.Contains
    divide = tree.Divide
    export = tree.Export
    eq = tree.EqualTo
    floor_divide = tree.FloorDivide
    ge = tree.GreaterThanOrEqualTo
    gt = tree.GreaterThan
    if_else = tree.IfElse
    include = tree.Include
    le = tree.LessThanOrEqualTo
    let = tree.Let
    let_import = tree.Import
    let_stable = tree.LetStable
    literal = tree.Literal
    logical_and = tree.And
    logical_not = tree.Not
    logical_or = tree.Or
    logical_xor = tree.Xor
    lookup = tree.Lookup
    lt = tree.LessThan
    modulo = tree.Modulo
    multiply = tree.Multiply
    name = tree.Name
    ne = tree.NotEqualTo
    neg = tree.Negative
    poly_binding = tree.PolyBinding
    pos = tree.Positive
    power = tree.Power
    slice = tree.Slice
    subtract = tree.Subtract
    tag = tree.Tag
    top = tree.Top


GRAMMAR = (Path(__file__).parent / 'grammar.lark').read_text(encoding='utf8')
PARSER = Lark(GRAMMAR, postlex=FlitterIndenter(), regex=True, start='top', maybe_placeholders=True,
              parser='lalr', transformer=FlitterTransformer())


def parse(source):
    try:
        if not source.endswith('\n'):
            source += '\n'
        return PARSER.parse(source)
    except UnexpectedInput as exc:
        raise ParseError(f"Parse error in source at line {exc.line} column {exc.column}",
                         line=exc.line, column=exc.column, context=exc.get_context(source).rstrip()) from exc
