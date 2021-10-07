"""
Flitter language interpreter
"""

# pylama:skip=.

import operator

from . import ast
from .. import model


BUILTINS = {
    'true': model.true,
    'false': model.false,
    'null': model.null,
    'uniform': model.Vector((model.Uniform,)),
    'beta': model.Vector((model.Beta,)),
    'normal': model.Vector((model.Normal,)),
    'sine': model.Vector((model.sine,)),
    'bounce': model.Vector((model.bounce,)),
    'sharkfin': model.Vector((model.sharkfin,)),
    'sawtooth': model.Vector((model.sawtooth,)),
    'triangle': model.Vector((model.triangle,)),
    'square': model.Vector((model.square,)),
    'linear': model.Vector((model.linear,)),
    'quad': model.Vector((model.quad,)),
    'shuffle': model.Vector((model.shuffle,)),
}


class Context:
    def __init__(self, /, variables=None, state=None, graph=None):
        self.variables = variables if variables is not None else {}
        self.state = state if state is not None else {}
        self.graph = graph if graph is not None else model.Node('root', ())

    def copy(self, merge=None):
        variables = {} if merge is None else dict(merge)
        variables.update(self.variables)
        return Context(variables=variables, state=self.state, graph=self.graph)


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
        return ast.Literal(value=model.Vector.compose(*(expr.value for expr in remaining)))
    return ast.Sequence(expressions=tuple(remaining))


def simplify(expression, context):
    match expression:
        case ast.Sequence(expressions=expressions):
            context = context.copy()
            return sequence_pack(simplify(expr, context) for expr in expressions)

        case ast.Literal() | ast.Search():
            return expression

        case ast.Name(name=name):
            if name in context.variables:
                return ast.Literal(context.variables[name])
            if name in BUILTINS:
                return ast.Literal(BUILTINS[name])
            return expression

        case ast.Let(bindings=bindings):
            remaining = []
            for binding in bindings:
                expr = simplify(binding.expr, context)
                if isinstance(expr, ast.Literal):
                    context.variables[binding.name] = expr.value
                else:
                    remaining.append(binding)
            if remaining:
                return ast.Let(bindings=tuple(remaining))
            return ast.Literal(model.null)

        case ast.Range(start=start, stop=stop, step=step):
            start = simplify(start, context)
            stop = simplify(stop, context)
            step = simplify(step, context)
            if isinstance(start, ast.Literal) and isinstance(stop, ast.Literal) and isinstance(step, ast.Literal):
                return ast.Literal(model.Vector.range(start.value, stop.value, step.value))
            return ast.Range(start=start, stop=stop, step=step)

        case ast.Node(kind=kind, tags=tags):
            return ast.Literal(value=model.Vector((model.Node(kind, tags),)))

        case ast.Append(node=node, children=children):
            node = simplify(node, context)
            children = simplify(children, context)
            if isinstance(node, ast.Literal) and node.value.isinstance(model.Node) and \
               isinstance(children, ast.Literal) and children.value.isinstance(model.Node):
                 for n in node.value:
                     n.extend(children.value)
                 return node
            return ast.Append(node=node, children=children)

        case ast.Attribute(node=node, name=name, expr=expr):
            node = simplify(node, context)
            if isinstance(node, ast.Literal) and node.value.isinstance(model.Node):
                simplified_values = []
                for n in node.value:
                    attribute_expr = simplify(expr, context.copy(n))
                    if isinstance(attribute_expr, ast.Literal):
                        n[name] = attribute_expr.value
                        simplified_values.append(ast.Literal(value=model.Vector((n,))))
                    else:
                        simplified_values.append(ast.Attribute(node=ast.Literal(value=model.Vector((n,))), name=name, expr=attribute_expr))
                return sequence_pack(simplified_values)
            expr = simplify(expr, context)
            return ast.Attribute(node=node, name=name, expr=expr)

        case ast.For(name=name, source=source, body=body):
            source = simplify(source, context)
            if not isinstance(source, ast.Literal):
                body = simplify(body, context)
                return ast.For(name=name, source=source, body=body)
            context = context.copy()
            remaining = []
            for value in source.value:
                context.variables[name] = model.Vector((value,))
                remaining.append(simplify(body, context))
            return sequence_pack(remaining)

        case ast.UnaryOperation(expr=expr):
            expr = simplify(expr, context)
            if isinstance(expr, ast.Literal):
                match expression:
                    case ast.Negative():
                        return ast.Literal(expr.value.neg())
                    case ast.Not():
                        return ast.Literal(expr.value.not_())
            return type(expression)(expr=expr)

        case ast.Comparison(left=left, right=right):
            left = simplify(left, context)
            right = simplify(right, context)
            if isinstance(left, ast.Literal) and isinstance(right, ast.Literal):
                cmp = left.value.compare(right.value)
                match expression:
                    case ast.EqualTo():
                        return ast.Literal(model.true if cmp == 0 else model.false)
                    case ast.NotEqualTo():
                        return ast.Literal(model.true if cmp != 0 else model.false)
                    case ast.LessThan():
                        return ast.Literal(model.true if cmp == -1 else model.false)
                    case ast.GreaterThan():
                        return ast.Literal(model.true if cmp == 1 else model.false)
                    case ast.LessThanOrEqualTo():
                        return ast.Literal(model.true if cmp != 1 else model.false)
                    case ast.GreaterThanOrEqualTo():
                        return ast.Literal(model.true if cmp != -1 else model.false)
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
                    case ast.And():
                        return ast.Literal(left.value.and_(right.value))
                    case ast.Or():
                        return ast.Literal(left.value.or_(right.value))
            return type(expression)(left=left, right=right)

        case ast.Call(function=function, args=args):
            function = simplify(function, context)
            args = tuple(simplify(arg, context) for arg in args)
            if isinstance(function, ast.Literal) and all(isinstance(arg, ast.Literal) for arg in args):
                args = tuple(arg.value for arg in args)
                return ast.Literal(model.Vector.compose(*(func(*args) for func in function.values)))
            return ast.Call(function=function, args=args)

        case ast.Slice(expr=expr, index=index):
            expr = simplify(expr, context)
            index = simplify(index, context)
            if isinstance(expr, ast.Literal) and isinstance(index, ast.Literal):
                return ast.Literal(expr.slice(index))
            return ast.Slice(expr=expr, index=index)

    raise NotImplementedError(expression.__class__.__name__)


def evaluate(expression, context):
    match expression:
        case ast.Sequence(expressions=expressions):
            context = context.copy()
            return model.Vector.compose(*(evaluate(expr, context) for expr in expressions))

        case ast.Literal(value=value):
            if isinstance(value, model.Vector):
                return value.copynodes()
            return value

        case ast.Name(name=name):
            if name in context.variables:
                return context.variables[name]
            if name in BUILTINS:
                return BUILTINS[name]
            return model.null

        case ast.Let(bindings=bindings):
            for binding in bindings:
                context.variables[binding.name] = evaluate(binding.expr, context)
            return model.null

        case ast.Range(start=start, stop=stop, step=step):
            start = evaluate(start, context)
            stop = evaluate(stop, context)
            step = evaluate(step, context)
            return model.Vector.range(start, stop, step)

        case ast.Search(query=query):
            return model.Vector(context.graph.select_below(query))

        case ast.Node(kind=kind, tags=tags):
            return model.Vector((model.Node(kind, tags),))

        case ast.Append(node=node, children=children):
            node = evaluate(node, context)
            children = evaluate(children, context)
            for n in node:
                n.extend(children)
            return node

        case ast.Attribute(node=node, name=name, expr=expr):
            node = evaluate(node, context)
            for n in node:
                n[name] = evaluate(expr, context.copy(n))
            return node

        case ast.For(name=name, source=source, body=body):
            source = evaluate(source, context)
            context = context.copy()
            results = []
            for value in source:
                context.variables[name] = model.Vector((value,))
                results.append(evaluate(body, context))
            return model.Vector.compose(*results)

        case ast.UnaryOperation(expr=expr):
            value = evaluate(expr, context)
            match expression:
                case ast.Negative():
                    return value.neg()
                case ast.Not():
                    return value.not_()

        case ast.Comparison(left=left, right=right):
            left = evaluate(left, context)
            right = evaluate(right, context)
            cmp = left.compare(right)
            match expression:
                case ast.EqualTo():
                    return model.true if cmp == 0 else model.false
                case ast.NotEqualTo():
                    return model.true if cmp != 0 else model.false
                case ast.LessThan():
                    return model.true if cmp == -1 else model.false
                case ast.GreaterThan():
                    return model.true if cmp == 1 else model.false
                case ast.LessThanOrEqualTo():
                    return model.true if cmp != 1 else model.false
                case ast.GreaterThanOrEqualTo():
                    return model.true if cmp != -1 else model.false

        case ast.BinaryOperation(left=left, right=right):
            left = evaluate(left, context)
            right = evaluate(right, context)
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
                case ast.And():
                    return left.and_(right)
                case ast.Or():
                    return left.or_(right)

        case ast.Call(function=function, args=args):
            function = evaluate(function, context)
            args = tuple(evaluate(arg, context) for arg in args)
            return model.Vector.compose(*(func(*args) for func in function))

        case ast.Slice(expr=expr, index=index):
            expr = evaluate(expr, context)
            index = evaluate(index, context)
            return expr.slice(index)

    raise NotImplementedError(expression.__class__.__name__)


def run(tree, context):
    assert isinstance(tree, ast.Sequence)
    context = context.copy()
    for expr in tree.expressions:
        result = evaluate(expr, context)
        for value in result:
            if isinstance(value, model.Node) and value.parent is None:
                context.graph.append(value)
    return context.graph
