"""
Flitter language interpreter
"""

# pylama:skip=.

import operator

from . import ast
from ..model import values


OPERATORS = {
    ast.Negative: operator.neg,
    ast.Add: operator.add,
    ast.Subtract: operator.sub,
    ast.Multiply: operator.mul,
    ast.Divide: operator.truediv,
    ast.Modulo: operator.mod,
    ast.Power: operator.pow,
    ast.EqualTo: operator.eq,
    ast.NotEqualTo: operator.ne,
    ast.LessThan: operator.lt,
    ast.GreaterThan: operator.gt,
    ast.LessThanOrEqualTo: operator.le,
    ast.GreaterThanOrEqualTo: operator.ge,
}


def simplify_sequence(sequence, context):
    context = dict(context)
    remaining = []
    for expr in sequence:
        result = simplify_expression(expr, context)
        if isinstance(result, ast.Literal) and result.value == values.null:
            continue
        if isinstance(result, tuple):
            remaining.extend(result)
        else:
            remaining.append(result)
    return tuple(remaining)


def simplify_expression(expression, context):
    match expression:
        case ast.Literal() | ast.Node() | ast.Search():
            return expression

        case ast.Name(name=name):
            if name in context:
                return ast.Literal(context[name])
            return expression

        case ast.Let(bindings=bindings):
            remaining = []
            for binding in bindings:
                expr = simplify_expression(binding.expr, context)
                if isinstance(expr, ast.Literal):
                    context[binding.name] = expr.value
                else:
                    remaining.append(binding)
            if remaining:
                return ast.Let(bindings=tuple(remaining))
            return ast.Literal(values.null)

        case ast.Compose(left=left, right=right):
            left = simplify_expression(left, context)
            right = simplify_expression(right, context)
            if isinstance(left, ast.Literal) and isinstance(right, ast.Literal):
                return ast.Literal(values.Vector.compose(left.value, right.value))
            return ast.Compose(left=left, right=right)

        case ast.Range(start=start, stop=stop, step=step):
            start = simplify_expression(start, context) if start is not None else None
            stop = simplify_expression(stop, context) if stop is not None else None
            step = simplify_expression(step, context) if step is not None else None
            if (start is None or isinstance(start, ast.Literal)) and \
               (stop is None or isinstance(stop, ast.Literal)) and \
               (step is None or isinstance(step, ast.Literal)):
                value = values.Range(None if start is None else start.value, None if stop is None else stop.value, None if step is None else step.value)
                return ast.Literal(value)
            return ast.Range(start=start, stop=stop, step=step)

        case ast.Graph(node=node, children=children):
            node = simplify_expression(node, context)
            children = simplify_sequence(children, context) if children is not None else None
            return ast.Graph(node=node, children=children)

        case ast.Attribute(node=node, name=name, expr=expr):
            node = simplify_expression(node, context)
            expr = simplify_expression(expr, context)
            return ast.Attribute(node=node, name=name, expr=expr)

        case ast.For(name=name, source=source, body=body):
            source = simplify_expression(source, context)
            if not isinstance(source, ast.Literal):
                body = simplify_sequence(body, context)
                if not body:
                    return ast.Literal(value=values.null)
                return ast.For(name=name, source=source, body=body)
            context = dict(context)
            remaining = []
            for value in source.value:
                context[name] = value
                simplified_body = simplify_sequence(body, context)
                remaining.extend(simplified_body)
            return tuple(remaining)

        case ast.UnaryOperation(expr=expr):
            expr = simplify_expression(expr, context)
            if isinstance(expr, ast.Literal):
                return ast.Literal(OPERATORS[type(expression)](expr.value))
            return type(expression)(expr=expr)

        case ast.BinaryOperation(left=left, right=right):
            left = simplify_expression(left, context)
            right = simplify_expression(right, context)
            if isinstance(left, ast.Literal) and isinstance(right, ast.Literal):
                return ast.Literal(OPERATORS[type(expression)](left.value, right.value))
            return type(expression)(left=left, right=right)

    raise NotImplementedError(expression.__class__.__name__)
