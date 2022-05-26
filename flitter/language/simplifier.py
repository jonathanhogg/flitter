"""
Flitter language simplifier
"""

# pylama:skip=.

from .. import model
from . import tree


def sequence_pack(expressions):
    remaining = []
    for expr in expressions:
        if isinstance(expr, tree.Literal) and not expr.value:
            continue
        if isinstance(expr, tree.Sequence):
            remaining.extend(expr.expressions)
        else:
            remaining.append(expr)
    if all(isinstance(expr, tree.Literal) for expr in remaining):
        return tree.Literal(value=model.Vector.compose(*(expr.value for expr in remaining)))
    return tree.Sequence(expressions=tuple(remaining))


def simplify(expression, context):
    match expression:
        case tree.Sequence(expressions=expressions):
            with context:
                return sequence_pack(simplify(expr, context) for expr in expressions)

        case tree.Literal() | tree.Search():
            return expression

        case tree.Name(name=name):
            if name in context:
                return tree.Literal(context[name])
            return expression

        case tree.Lookup(key=key):
            key = simplify(key, context)
            if isinstance(key, tree.Literal) and isinstance(key.value, model.Vector) and key.value in context.state:
                return tree.Literal(context.state[key.value])
            return tree.Lookup(key=key)

        case tree.Let(bindings=bindings):
            remaining = []
            for binding in bindings:
                expr = simplify(binding.expr, context)
                if isinstance(expr, tree.Literal):
                    context[binding.name] = expr.value
                else:
                    remaining.append(tree.Binding(binding.name, expr))
            if remaining:
                return tree.Let(bindings=tuple(remaining))
            return tree.Literal(model.null)

        case tree.Range(start=start, stop=stop, step=step):
            start = simplify(start, context)
            stop = simplify(stop, context)
            step = simplify(step, context)
            if isinstance(start, tree.Literal) and isinstance(stop, tree.Literal) and isinstance(step, tree.Literal):
                return tree.Literal(model.Vector.range(start.value, stop.value, step.value))
            return tree.Range(start=start, stop=stop, step=step)

        case tree.Node(kind=kind, tags=tags):
            return tree.Literal(value=model.Vector((model.Node(kind, tags),)))

        case tree.Append(node=node, children=children):
            node = simplify(node, context)
            children = simplify(children, context)
            if isinstance(node, tree.Literal) and node.value.isinstance(model.Node) and \
               isinstance(children, tree.Literal) and children.value.isinstance(model.Node):
                 for n in node.value:
                     n.extend(children.value.copynodes())
                 return node
            return tree.Append(node=node, children=children)

        case tree.Prepend(node=node, children=children):
            node = simplify(node, context)
            children = simplify(children, context)
            if isinstance(node, tree.Literal) and node.value.isinstance(model.Node) and \
               isinstance(children, tree.Literal) and children.value.isinstance(model.Node):
                 for n in node.value:
                     n.prepend(children.value.copynodes())
                 return node
            return tree.Prepend(node=node, children=children)

        case tree.Attribute(node=node, name=name, expr=expr):
            node = simplify(node, context)
            if isinstance(node, tree.Literal) and node.value.isinstance(model.Node):
                simplified_values = []
                for n in node.value:
                    with context:
                        context.merge_under(n)
                        attribute_expr = simplify(expr, context)
                    if isinstance(attribute_expr, tree.Literal):
                        n[name] = attribute_expr.value
                        simplified_values.append(tree.Literal(value=model.Vector((n,))))
                    else:
                        simplified_values.append(tree.Attribute(node=tree.Literal(value=model.Vector((n,))), name=name, expr=attribute_expr))
                return sequence_pack(simplified_values)
            expr = simplify(expr, context)
            return tree.Attribute(node=node, name=name, expr=expr)

        case tree.InlineLet(bindings=bindings, body=body):
            remaining = []
            with context:
                for binding in bindings:
                    expr = simplify(binding.expr, context)
                    if isinstance(expr, tree.Literal):
                        context[binding.name] = expr.value
                    else:
                        remaining.append(tree.Binding(binding.name, expr))
                body = simplify(body, context)
                if remaining:
                    return tree.InlineLet(bindings=tuple(remaining), body=body)
                return body

        case tree.For(name=name, source=source, body=body):
            source = simplify(source, context)
            if not isinstance(source, tree.Literal):
                body = simplify(body, context)
                return tree.For(name=name, source=source, body=body)
            with context:
                remaining = []
                for value in source.value:
                    context[name] = model.Vector((value,))
                    remaining.append(simplify(body, context))
            return sequence_pack(remaining)

        case tree.IfElse(tests=tests, else_=else_):
            remaining = []
            for test in tests:
                condition = simplify(test.condition, context)
                then = simplify(test.then, context)
                if isinstance(condition, tree.Literal):
                    if condition.value.istrue():
                        if not remaining:
                            return then
                        else:
                            return tree.IfElse(tests=remaining, else_=then)
                else:
                    remaining.append(tree.Test(condition=condition, then=then))
            else_ = simplify(else_, context) if else_ is not None else None
            if remaining:
                return tree.IfElse(tests=tuple(remaining), else_=else_)
            return tree.Literal(model.null) if else_ is None else else_

        case tree.UnaryOperation(expr=expr):
            expr = simplify(expr, context)
            if isinstance(expr, tree.Literal):
                match expression:
                    case tree.Negative():
                        return tree.Literal(expr.value.neg())
                    case tree.Positive():
                        return tree.Literal(expr.value.pos())
                    case tree.Not():
                        return tree.Literal(expr.value.not_())
            return type(expression)(expr=expr)

        case tree.Comparison(left=left, right=right):
            left = simplify(left, context)
            right = simplify(right, context)
            if isinstance(left, tree.Literal) and isinstance(right, tree.Literal):
                cmp = left.value.compare(right.value)
                match expression:
                    case tree.EqualTo():
                        return tree.Literal(model.true if cmp == 0 else model.false)
                    case tree.NotEqualTo():
                        return tree.Literal(model.true if cmp != 0 else model.false)
                    case tree.LessThan():
                        return tree.Literal(model.true if cmp == -1 else model.false)
                    case tree.GreaterThan():
                        return tree.Literal(model.true if cmp == 1 else model.false)
                    case tree.LessThanOrEqualTo():
                        return tree.Literal(model.true if cmp != 1 else model.false)
                    case tree.GreaterThanOrEqualTo():
                        return tree.Literal(model.true if cmp != -1 else model.false)
            return type(expression)(left=left, right=right)

        case tree.BinaryOperation(left=left, right=right):
            left = simplify(left, context)
            right = simplify(right, context)
            if isinstance(left, tree.Literal) and isinstance(right, tree.Literal):
                match expression:
                    case tree.Add():
                        return tree.Literal(left.value.add(right.value))
                    case tree.Subtract():
                        return tree.Literal(left.value.sub(right.value))
                    case tree.Multiply():
                        return tree.Literal(left.value.mul(right.value))
                    case tree.Divide():
                        return tree.Literal(left.value.truediv(right.value))
                    case tree.FloorDivide():
                        return tree.Literal(left.value.floordiv(right.value))
                    case tree.Modulo():
                        return tree.Literal(left.value.mod(right.value))
                    case tree.Power():
                        return tree.Literal(left.value.pow(right.value))
                    case tree.And():
                        if isinstance(left, tree.Literal):
                            return right if left.value.istrue() else left
                    case tree.Or():
                        if isinstance(left, tree.Literal):
                            return left if left.value.istrue() else right
            return type(expression)(left=left, right=right)

        case tree.Call(function=function, args=args):
            function = simplify(function, context)
            args = tuple(simplify(arg, context) for arg in args)
            if isinstance(function, tree.Literal) and all(isinstance(arg, tree.Literal) for arg in args):
                args = tuple(arg.value for arg in args)
                return tree.Literal(model.Vector.compose(*(func(*args) for func in function.value)))
            return tree.Call(function=function, args=args)

        case tree.Slice(expr=expr, index=index):
            expr = simplify(expr, context)
            index = simplify(index, context)
            if isinstance(expr, tree.Literal) and isinstance(index, tree.Literal):
                return tree.Literal(expr.value.slice(index.value))
            return tree.Slice(expr=expr, index=index)

        case tree.Pragma(name=name, expr=expr):
            expr = simplify(expr, context)
            return tree.Pragma(name=name, expr=expr)

    print(expression)
    raise NotImplementedError(expression.__class__.__name__)
