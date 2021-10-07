"""
Flitter language interpreter
"""

# pylama:skip=.

import operator

from . import ast
from ..model import values


def sequence_pack(expressions):
    remaining = []
    for expr in expressions:
        if isinstance(expr, ast.Literal) and not expr.value:
            continue
        if isinstance(expr, ast.Sequence):
            remaining.extend(expr.expressions)
        else:
            remaining.append(expr)
    if all(isinstance(expr, ast.Literal) for expr in remaining):
        return ast.Literal(value=values.Vector.compose(*(expr.value for expr in remaining)))
    return ast.Sequence(expressions=tuple(remaining))


def simplify(expression, context):
    match expression:
        case ast.Sequence(expressions=expressions):
            context = dict(context)
            expressions = tuple(simplify(expr, context) for expr in expressions)
            return sequence_pack(expressions)

        case ast.Literal() | ast.Search():
            return expression

        case ast.Name(name=name):
            if name in context:
                return ast.Literal(context[name])
            return expression

        case ast.Let(bindings=bindings):
            remaining = []
            for binding in bindings:
                expr = simplify(binding.expr, context)
                if isinstance(expr, ast.Literal):
                    context[binding.name] = expr.value
                else:
                    remaining.append(binding)
            if remaining:
                return ast.Let(bindings=tuple(remaining))
            return ast.Literal(values.null)

        case ast.Range(start=start, stop=stop, step=step):
            start = simplify(start, context)
            stop = simplify(stop, context)
            step = simplify(step, context)
            if isinstance(start, ast.Literal) and isinstance(stop, ast.Literal) and isinstance(step, ast.Literal):
                return ast.Literal(values.Vector.range(start.value, stop.value, step.value))
            return ast.Range(start=start, stop=stop, step=step)

        case ast.Node(kind=kind, tags=tags):
            return ast.Literal(value=values.Vector((values.Node(kind, tags),)))

        case ast.Append(node=node, children=children):
            node = simplify(node, context)
            children = simplify(children, context)
            if isinstance(node, ast.Literal) and node.value.isinstance(values.Node) and \
               isinstance(children, ast.Literal) and children.value.isinstance(values.Node):
                 for n in node.value:
                     n.extend(children.value)
                 return node
            return ast.Append(node=node, children=children)

        case ast.Attribute(node=node, name=name, expr=expr):
            node = simplify(node, context)
            if isinstance(node, ast.Literal) and node.value.isinstance(values.Node):
                simplified_values = []
                for n in node.value:
                    attribute_context = dict(n)
                    attribute_context.update(context)
                    attribute_expr = simplify(expr, attribute_context)
                    if isinstance(attribute_expr, ast.Literal):
                        n[name] = attribute_expr.value
                        simplified_values.append(ast.Literal(value=values.Vector((n,))))
                    else:
                        simplified_values.append(ast.Attribute(node=ast.Literal(value=values.Vector((n,))), name=name, expr=attribute_expr))
                return sequence_pack(simplified_values)
            expr = simplify(expr, context)
            return ast.Attribute(node=node, name=name, expr=expr)

        case ast.For(name=name, source=source, body=body):
            source = simplify(source, context)
            if not isinstance(source, ast.Literal):
                body = simplify(body, context)
                return ast.For(name=name, source=source, body=body)
            context = dict(context)
            remaining = []
            for value in source.value:
                context[name] = values.Vector((value,))
                remaining.append(simplify(body, context))
            return sequence_pack(remaining)

        case ast.UnaryOperation(expr=expr):
            expr = simplify(expr, context)
            if isinstance(expr, ast.Literal):
                match expression:
                    case ast.Negative():
                        return ast.Literal(expr.value.neg())
            return type(expression)(expr=expr)

        case ast.Comparison(left=left, right=right):
            left = simplify(left, context)
            right = simplify(right, context)
            if isinstance(left, ast.Literal) and isinstance(right, ast.Literal):
                cmp = left.value.compare(right.value)
                match expression:
                    case ast.EqualTo():
                        return ast.Literal(values.Vector((1. if cmp == 0 else 0.,)))
                    case ast.NotEqualTo():
                        return ast.Literal(values.Vector((1. if cmp != 0 else 0.,)))
                    case ast.LessThan():
                        return ast.Literal(values.Vector((1. if cmp == -1 else 0.,)))
                    case ast.GreaterThan():
                        return ast.Literal(values.Vector((1. if cmp == 1 else 0.,)))
                    case ast.LessThanOrEqualTo():
                        return ast.Literal(values.Vector((1. if cmp != 1 else 0.,)))
                    case ast.GreaterThanOrEqualTo():
                        return ast.Literal(values.Vector((1. if cmp != -1 else 0.,)))
            return type(expression)(left=left, right=right)

        case ast.BinaryOperation(left=left, right=right):
            left = simplify(left, context)
            right = simplify(right, context)
            if isinstance(left, ast.Literal) and isinstance(right, ast.Literal):
                match expression:
                    case ast.Add():
                        return ast.Literal(left.value.add(right.value))
                    case ast.Subtract():
                        return ast.Literal(left.value.sub(right.value))
                    case ast.Multiply():
                        return ast.Literal(left.value.mul(right.value))
                    case ast.Divide():
                        return ast.Literal(left.value.truediv(right.value))
                    case ast.FloorDivide():
                        return ast.Literal(left.value.floordiv(right.value))
                    case ast.Modulo():
                        return ast.Literal(left.value.mod(right.value))
                    case ast.Power():
                        return ast.Literal(left.value.pow(right.value))
            return type(expression)(left=left, right=right)

        case ast.Call(function=function, args=args):
            function = simplify(function, context)
            args = tuple(simplify(arg, context) for arg in args)
            return ast.Call(function=function, args=args)

    raise NotImplementedError(expression.__class__.__name__)


def evaluate(expression, graph, context):
    match expression:
        case ast.Sequence(expressions=expressions):
            context = dict(context)
            for expr in expressions:
                vector = evaluate(expr, graph, context)
                for value in vector:
                    if isinstance(value, values.Node):
                        if not value.parent:
                            graph.append(value)

        case ast.Literal(value=value):
            return value

        case ast.Name(name=name):
            if name in context:
                return context[name]
            return values.null

        case ast.Let(bindings=bindings):
            for binding in bindings:
                context[binding.name] = evaluate(binding.expr, graph, context)
            return values.null

        case ast.Search(query=query):
            return values.Vector(graph.select_below(query))

        case ast.Node(kind=kind, tags=tags):
            return values.Vector((values.Node(kind, tags),))

        case ast.Append(node=node, children=children):
            node = evaluate(node, graph, context)
            children = evaluate(children, graph, context)
            for n in node:
                n.extend(children)
            return node

        case ast.Attribute(node=node, name=name, expr=expr):
            node = evaluate(node, graph, context)
            value = evaluate(expr, graph, context)
            for n in node:
                n[name] = value
            return node

        case ast.UnaryOperation(expr=expr):
            value = evaluate(expr, graph, context)
            match expression:
                case ast.Negative():
                    return value.neg()

        case ast.Comparison(left=left, right=right):
            left = evaluate(left, graph, context)
            right = evaluate(right, graph, context)
            cmp = left.compare(right)
            match expression:
                case ast.EqualTo():
                    return values.Vector((1. if cmp == 0 else 0.,))
                case ast.NotEqualTo():
                    return values.Vector((1. if cmp != 0 else 0.,))
                case ast.LessThan():
                    return values.Vector((1. if cmp == -1 else 0.,))
                case ast.GreaterThan():
                    return values.Vector((1. if cmp == 1 else 0.,))
                case ast.LessThanOrEqualTo():
                    return values.Vector((1. if cmp != 1 else 0.,))
                case ast.GreaterThanOrEqualTo():
                    return values.Vector((1. if cmp != -1 else 0.,))

        case ast.BinaryOperation(left=left, right=right):
            left = evaluate(left, graph, context)
            right = evaluate(right, graph, context)
            match expression:
                case ast.Add():
                    return left.add(right)
                case ast.Subtract():
                    return left.sub(right)
                case ast.Multiply():
                    return left.mul(right)
                case ast.Divide():
                    return left.truediv(right)
                case ast.FloorDivide():
                    return left.floordiv(right)
                case ast.Modulo():
                    return left.mod(right)
                case ast.Power():
                    return left.pow(right)

    raise NotImplementedError(expression.__class__.__name__)
