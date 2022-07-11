# cython: language_level=3, profile=True

"""
Language syntax/evaluation tree
"""

cimport cython

from .functions import FUNCTIONS
from .. cimport model


cdef dict builtins_ = {
    'true': model.true_,
    'false': model.false_,
    'null': model.null_,
}
builtins_.update(FUNCTIONS)


cdef Expression sequence_pack(list expressions):
    cdef Expression expr
    cdef Literal literal
    cdef model.Vector value
    cdef list remaining = []
    while expressions:
        expr = <Expression>expressions.pop(0)
        if isinstance(expr, Literal) and isinstance((<Literal>expr).value, model.Vector):
            value = model.Vector.__new__(model.Vector)
            while isinstance(expr, Literal) and isinstance((<Literal>expr).value, model.Vector):
                value.values.extend((<model.Vector>(<Literal>expr).value).values)
                if not expressions:
                    expr = None
                    break
                expr = <Expression>expressions.pop(0)
            remaining.append(Literal(value))
        if expr is not None:
            if isinstance(expr, Sequence):
                expressions[:0] = (<Sequence>expr).expressions
                continue
            remaining.append(expr)
    if len(remaining) == 0:
        return Literal(model.null_)
    if len(remaining) == 1:
        return remaining[0]
    return Sequence(tuple(remaining))


cdef class Expression:
    cpdef model.VectorLike evaluate(self, model.Context context):
        raise NotImplementedError()

    cpdef Expression partially_evaluate(self, model.Context context):
        raise NotImplementedError()


cdef class Pragma(Expression):
    cdef readonly str name
    cdef readonly Expression expr

    def __init__(self, str name, Expression expr):
        self.name = name
        self.expr = expr

    cpdef model.VectorLike evaluate(self, model.Context context):
        cdef model.Vector value = self.expr.evaluate(context)
        context.pragma(self.name, value)
        return model.null_

    cpdef Expression partially_evaluate(self, model.Context context):
        return Pragma(self.name, self.expr.partially_evaluate(context))

    def __repr__(self):
        return f'Pragma({self.name!r}, {self.expr!r})'


cdef class Sequence(Expression):
    cdef readonly tuple expressions

    def __init__(self, tuple expressions):
        self.expressions = expressions

    cpdef model.VectorLike evaluate(self, model.Context context):
        cdef model.Vector result = model.Vector.__new__(model.Vector)
        cdef model.Vector vector
        cdef Expression expr
        for expr in self.expressions:
            vector = expr.evaluate(context)
            result.values.extend(vector.values)
        return result

    cpdef Expression partially_evaluate(self, model.Context context):
        cdef list expressions = []
        cdef Expression expr
        with context:
            for expr in self.expressions:
                expressions.append(expr.partially_evaluate(context))
        return sequence_pack(expressions)

    def __repr__(self):
        return f'Sequence({self.expressions!r})'


cdef class Literal(Expression):
    cdef readonly model.VectorLike value

    def __init__(self, model.VectorLike value):
        self.value = value

    cpdef model.VectorLike evaluate(self, model.Context context):
        return self.value.copynodes()

    cpdef Expression partially_evaluate(self, model.Context context):
        return Literal(self.value.copynodes())

    def __repr__(self):
        return f'Literal({self.value!r})'


cdef class Name(Expression):
    cdef readonly str name

    def __init__(self, str name):
        self.name = name

    cpdef model.VectorLike evaluate(self, model.Context context):
        cdef model.VectorLike result
        result = context.variables.get(self.name)
        if result is not None:
            return result.copynodes()
        result = builtins_.get(self.name)
        if result is not None:
            return result
        return model.null_

    cpdef Expression partially_evaluate(self, model.Context context):
        cdef model.VectorLike value
        if self.name in context.variables:
            value = context.variables[self.name]
            return Literal(value.copynodes())
        if self.name in builtins_:
            value = builtins_[self.name]
            return Literal(value)
        return self

    def __repr__(self):
        return f'Name({self.name!r})'


cdef class Lookup(Expression):
    cdef readonly Expression key

    def __init__(self, Expression key):
        self.key = key

    cpdef model.VectorLike evaluate(self, model.Context context):
        cdef model.Vector key = self.key.evaluate(context)
        value = context.state.get(tuple(key.values))
        if value is None:
            return model.null_
        if isinstance(value, bool):
            return model.true_ if value else model.false_
        cdef model.Vector result = model.Vector.__new__(model.Vector)
        if isinstance(value, list):
            result.values = <list>value
        if isinstance(value, tuple):
            result.values.extend(value)
        else:
            result.values.append(value)
        return result

    cpdef Expression partially_evaluate(self, model.Context context):
        return Lookup(self.key.partially_evaluate(context))

    def __repr__(self):
        return f'Lookup({self.key!r})'


cdef class Range(Expression):
    cdef readonly Expression start
    cdef readonly Expression stop
    cdef readonly Expression step

    def __init__(self, Expression start, Expression stop, Expression step):
        self.start = start
        self.stop = stop
        self.step = step

    cpdef model.VectorLike evaluate(self, model.Context context):
        start = self.start.evaluate(context)
        stop = self.stop.evaluate(context)
        step = self.step.evaluate(context)
        return model.Vector.range(start, stop, step)

    cpdef Expression partially_evaluate(self, model.Context context):
        cdef Expression start = self.start.partially_evaluate(context)
        cdef Expression stop = self.stop.partially_evaluate(context)
        cdef Expression step = self.step.partially_evaluate(context)
        if isinstance(start, Literal) and isinstance(stop, Literal) and isinstance(step, Literal):
            return Literal(model.Vector.range((<Literal>start).value, (<Literal>stop).value, (<Literal>step).value))
        return Range(start, stop, step)

    def __repr__(self):
        return f'Range({self.start!r}, {self.stop!r}, {self.step!r})'


cdef class UnaryOperation(Expression):
    cdef readonly Expression expr

    def __init__(self, Expression expr):
        self.expr = expr

    cpdef Expression partially_evaluate(self, model.Context context):
        cdef Expression expr = self.expr.partially_evaluate(context)
        cdef Expression unary = type(self)(expr)
        if isinstance(expr, Literal):
            return Literal(unary.evaluate(context))
        return unary

    def __repr__(self):
        return f'{self.__class__.__name__}({self.expr!r})'


cdef class Negative(UnaryOperation):
    cpdef model.VectorLike evaluate(self, model.Context context):
        cdef model.Vector value = self.expr.evaluate(context)
        return value.neg()


cdef class Positive(UnaryOperation):
    cpdef model.VectorLike evaluate(self, model.Context context):
        cdef model.Vector value = self.expr.evaluate(context)
        return value.pos()


cdef class Not(UnaryOperation):
    cpdef model.VectorLike evaluate(self, model.Context context):
        cdef model.Vector value = self.expr.evaluate(context)
        return value.not_()


cdef class BinaryOperation(Expression):
    cdef readonly Expression left
    cdef readonly Expression right

    def __init__(self, Expression left, Expression right):
        self.left = left
        self.right = right

    cpdef Expression partially_evaluate(self, model.Context context):
        cdef Expression left = self.left.partially_evaluate(context)
        cdef Expression right = self.right.partially_evaluate(context)
        cdef Expression binary = type(self)(left, right)
        if isinstance(left, Literal) and isinstance(right, Literal):
            return Literal(binary.evaluate(context))
        return binary

    def __repr__(self):
        return f'{self.__class__.__name__}({self.left!r}, {self.right!r})'


cdef class MathsBinaryOperation(BinaryOperation):
    pass


cdef class Add(MathsBinaryOperation):
    cpdef model.VectorLike evaluate(self, model.Context context):
        cdef model.Vector left = self.left.evaluate(context)
        cdef model.Vector right = self.right.evaluate(context)
        return left.add(right)


cdef class Subtract(MathsBinaryOperation):
    cpdef model.VectorLike evaluate(self, model.Context context):
        cdef model.Vector left = self.left.evaluate(context)
        cdef model.Vector right = self.right.evaluate(context)
        return left.sub(right)


cdef class Multiply(MathsBinaryOperation):
    cpdef model.VectorLike evaluate(self, model.Context context):
        cdef model.Vector left = self.left.evaluate(context)
        cdef model.Vector right = self.right.evaluate(context)
        return left.mul(right)


cdef class Divide(MathsBinaryOperation):
    cpdef model.VectorLike evaluate(self, model.Context context):
        cdef model.Vector left = self.left.evaluate(context)
        cdef model.Vector right = self.right.evaluate(context)
        return left.truediv(right)


cdef class FloorDivide(MathsBinaryOperation):
    cpdef model.VectorLike evaluate(self, model.Context context):
        cdef model.Vector left = self.left.evaluate(context)
        cdef model.Vector right = self.right.evaluate(context)
        return left.floordiv(right)


cdef class Modulo(MathsBinaryOperation):
    cpdef model.VectorLike evaluate(self, model.Context context):
        cdef model.Vector left = self.left.evaluate(context)
        cdef model.Vector right = self.right.evaluate(context)
        return left.mod(right)


cdef class Power(MathsBinaryOperation):
    cpdef model.VectorLike evaluate(self, model.Context context):
        cdef model.Vector left = self.left.evaluate(context)
        cdef model.Vector right = self.right.evaluate(context)
        return left.pow(right)


cdef class Comparison(BinaryOperation):
    pass


cdef class EqualTo(Comparison):
    cpdef model.VectorLike evaluate(self, model.Context context):
        cdef model.Vector left = self.left.evaluate(context)
        cdef model.Vector right = self.right.evaluate(context)
        cdef int cmp = left.compare(right)
        return model.true_ if cmp == 0 else model.false_


cdef class NotEqualTo(Comparison):
    cpdef model.VectorLike evaluate(self, model.Context context):
        cdef model.Vector left = self.left.evaluate(context)
        cdef model.Vector right = self.right.evaluate(context)
        cdef int cmp = left.compare(right)
        return model.true_ if cmp != 0 else model.false_


cdef class LessThan(Comparison):
    cpdef model.VectorLike evaluate(self, model.Context context):
        cdef model.Vector left = self.left.evaluate(context)
        cdef model.Vector right = self.right.evaluate(context)
        cdef int cmp = left.compare(right)
        return model.true_ if cmp == -1 else model.false_


cdef class GreaterThan(Comparison):
    cpdef model.VectorLike evaluate(self, model.Context context):
        cdef model.Vector left = self.left.evaluate(context)
        cdef model.Vector right = self.right.evaluate(context)
        cdef int cmp = left.compare(right)
        return model.true_ if cmp == 1 else model.false_


cdef class LessThanOrEqualTo(Comparison):
    cpdef model.VectorLike evaluate(self, model.Context context):
        cdef model.Vector left = self.left.evaluate(context)
        cdef model.Vector right = self.right.evaluate(context)
        cdef int cmp = left.compare(right)
        return model.true_ if cmp != 1 else model.false_


cdef class GreaterThanOrEqualTo(Comparison):
    cpdef model.VectorLike evaluate(self, model.Context context):
        cdef model.Vector left = self.left.evaluate(context)
        cdef model.Vector right = self.right.evaluate(context)
        cdef int cmp = left.compare(right)
        return model.true_ if cmp != -1 else model.false_


cdef class And(BinaryOperation):
    cpdef model.VectorLike evaluate(self, model.Context context):
        cdef model.Vector left = self.left.evaluate(context)
        return self.right.evaluate(context) if left.istrue() else left


cdef class Or(BinaryOperation):
    cpdef model.VectorLike evaluate(self, model.Context context):
        cdef model.Vector left = self.left.evaluate(context)
        return left if left.istrue() else self.right.evaluate(context)


cdef class Xor(BinaryOperation):
    cpdef model.VectorLike evaluate(self, model.Context context):
        cdef model.Vector left = self.left.evaluate(context)
        cdef model.Vector right = self.right.evaluate(context)
        if not left.istrue():
            return right
        if not right.istrue():
            return left
        return model.false_


cdef class Slice(Expression):
    cdef readonly Expression expr
    cdef readonly Expression index

    def __init__(self, Expression expr, Expression index):
        self.expr = expr
        self.index = index

    cpdef model.VectorLike evaluate(self, model.Context context):
        cdef model.VectorLike expr = self.expr.evaluate(context)
        cdef model.Vector index = self.index.evaluate(context)
        return expr.slice(index)

    cpdef Expression partially_evaluate(self, model.Context context):
        cdef Expression expr = self.expr.partially_evaluate(context)
        cdef Expression index = self.index.partially_evaluate(context)
        cdef model.VectorLike expr_value
        cdef model.Vector index_value
        if isinstance(expr, Literal) and isinstance(index, Literal):
            expr_value = (<Literal>expr).value
            index_value = (<Literal>index).value
            return Literal(expr_value.slice(index_value))
        return Slice(expr, index)

    def __repr__(self):
        return f'Slice({self.expr!r}, {self.index!r})'


cdef class Call(Expression):
    cdef readonly Expression function
    cdef readonly tuple args

    def __init__(self, Expression function, tuple args):
        self.function = function
        self.args = args

    cpdef model.VectorLike evaluate(self, model.Context context):
        cdef model.Vector function = self.function.evaluate(context)
        if not function.values:
            return model.null_
        cdef list args = []
        cdef Expression arg
        cdef model.VectorLike value
        for arg in self.args:
            value = arg.evaluate(context)
            args.append(value)
        cdef list results = []
        cdef Function func_expr
        cdef dict saved, params
        cdef Binding parameter
        cdef int i
        for func in function.values:
            if isinstance(func, Function):
                func_expr = func
                saved = context.variables
                context.variables = {}
                for i, parameter in enumerate(func_expr.parameters):
                    if i < len(args):
                        context.variables[parameter.name] = args[i]
                    elif parameter.expr is not None:
                        context.variables[parameter.name] = (<Literal>parameter.expr).value
                    else:
                        context.variables[parameter.name] = model.null_
                results.append(func_expr.expr.evaluate(context))
                context.variables = saved
            else:
                results.append(func(*args))
        return model.Vector_compose(results)

    cpdef Expression partially_evaluate(self, model.Context context):
        cdef Expression function = self.function.partially_evaluate(context)
        cdef list args = []
        cdef Expression arg
        cdef bint literal = isinstance(function, Literal)
        for arg in self.args:
            arg = arg.partially_evaluate(context)
            args.append(arg)
            if not isinstance(arg, Literal):
                literal = False
        cdef Call call = Call(function, tuple(args))
        if literal:
            return Literal(call.evaluate(context))
        return Call(function, tuple(args))

    def __repr__(self):
        return f'Call({self.function!r}, {self.args!r})'


cdef class Node(Expression):
    cdef readonly str kind

    def __init__(self, str kind):
        self.kind = kind

    cpdef model.VectorLike evaluate(self, model.Context context):
        cdef model.Node node = model.Node.__new__(model.Node, self.kind)
        cdef model.Vector vector = model.Vector.__new__(model.Vector)
        vector.values.append(node)
        return vector

    cpdef Expression partially_evaluate(self, model.Context context):
        return Literal(self.evaluate(context))

    def __repr__(self):
        return f'Node({self.kind!r})'


cdef class Tag(Expression):
    cdef readonly Expression node
    cdef readonly str tag

    def __init__(self, Expression node, str tag):
        self.node = node
        self.tag = tag

    cpdef model.VectorLike evaluate(self, model.Context context):
        cdef model.Vector nodes = self.node.evaluate(context)
        cdef model.Node node
        for node in nodes.values:
            node._tags.add(self.tag)
        return nodes

    cpdef Expression partially_evaluate(self, model.Context context):
        cdef Expression node = self.node.partially_evaluate(context)
        cdef model.Vector nodes
        cdef model.Node n
        if isinstance(node, Literal):
            nodes = (<Literal>node).value
            if nodes.isinstance(model.Node):
                for n in nodes.values:
                    n.add_tag(self.tag)
                return node
        return Tag(node, self.tag)

    def __repr__(self):
        return f'Tag({self.node!r}, {self.tag!r})'


cdef class Attributes(Expression):
    cdef readonly Expression node
    cdef readonly tuple bindings

    def __init__(self, Expression node, tuple bindings):
        self.node = node
        self.bindings = bindings

    cpdef model.VectorLike evaluate(self, model.Context context):
        cdef model.Node node
        cdef model.Vector value
        cdef dict variables, saved
        cdef Binding binding
        cdef model.Vector nodes = self.node.evaluate(context)
        for node in nodes.values:
            saved = context.variables
            variables = saved.copy()
            for attr, value in node._attributes.items():
                variables.setdefault(attr, value)
            context.variables = variables
            for binding in self.bindings:
                value = binding.expr.evaluate(context)
                if value.values:
                    node._attributes[binding.name] = value
                    if binding.name not in saved:
                        variables[binding.name] = value
            context.variables = saved
        return nodes

    cpdef Expression partially_evaluate(self, model.Context context):
        cdef Expression node = self
        cdef list bindings = []
        cdef Attributes attrs
        cdef Binding binding
        while isinstance(node, Attributes):
            attrs = <Attributes>node
            for binding in reversed(attrs.bindings):
                bindings.insert(0, Binding(binding.name, binding.expr.partially_evaluate(context)))
            node = attrs.node
        node = node.partially_evaluate(context)
        cdef model.Vector nodes
        cdef model.Node n
        if isinstance(node, Literal):
            nodes = (<Literal>node).value
            if nodes.isinstance(model.Node):
                while bindings and isinstance((<Binding>bindings[0]).expr, Literal):
                    binding = bindings.pop(0)
                    for n in nodes.values:
                        n[binding.name] = (<Literal>binding.expr).value
        if not bindings:
            return node
        return Attributes(node, tuple(bindings))

    def __repr__(self):
        return f'Attributes({self.node!r}, {self.bindings!r})'


cdef class Search(Expression):
    cdef readonly model.Query query

    def __init__(self, model.Query query):
        self.query = query

    cpdef model.VectorLike evaluate(self, model.Context context):
        cdef model.Node node = context.graph.first_child
        cdef model.Vector nodes = model.Vector.__new__(model.Vector)
        while node is not None:
            node._select(self.query, nodes.values, False)
            node = node.next_sibling
        return nodes

    cpdef Expression partially_evaluate(self, model.Context context):
        return self

    def __repr__(self):
        return f'Search({self.query!r})'


cdef class Append(Expression):
    cdef readonly Expression node
    cdef readonly Expression children

    def __init__(self, Expression node, Expression children):
        self.node = node
        self.children = children

    cpdef model.VectorLike evaluate(self, model.Context context):
        cdef model.Vector nodes = self.node.evaluate(context)
        cdef model.Vector children = self.children.evaluate(context)
        cdef model.Node node, child
        cdef int i, n = len(nodes.values)
        for i in range(n):
            node = nodes[i]
            if i < n-1:
                for child in children.values:
                    node.append(child.copy())
            else:
                for child in children.values:
                    node.append(child)
        return nodes

    cpdef Expression partially_evaluate(self, model.Context context):
        cdef Expression node = self.node.partially_evaluate(context)
        cdef Expression children = self.children.partially_evaluate(context)
        cdef model.Vector nodes, childs
        cdef model.Node n, c
        if isinstance(node, Literal) and isinstance(children, Literal):
            nodes = (<Literal>node).value
            childs = (<Literal>children).value
            if nodes.isinstance(model.Node) and childs.isinstance(model.Node):
                for n in nodes.values:
                    for c in childs.values:
                        n.append(c.copy())
                return node
        return Append(node, children)

    def __repr__(self):
        return f'Append({self.node!r}, {self.children!r})'


cdef class Prepend(Expression):
    cdef readonly Expression node
    cdef readonly Expression children

    def __init__(self, Expression node, Expression children):
        self.node = node
        self.children = children

    cpdef model.VectorLike evaluate(self, model.Context context):
        cdef model.Vector nodes = self.node.evaluate(context)
        cdef model.Vector children = self.children.evaluate(context)
        cdef model.Node node, child
        cdef int i, n = len(nodes.values)
        for i in range(n):
            node = nodes[i]
            if i < n-1:
                for child in reversed(children.values):
                    node.insert(child.copy())
            else:
                for child in reversed(children.values):
                    node.insert(child)
        return nodes

    cpdef Expression partially_evaluate(self, model.Context context):
        cdef Expression node = self.node.partially_evaluate(context)
        cdef Expression children = self.children.partially_evaluate(context)
        cdef model.Vector nodes, childs
        cdef model.Node n, c
        if isinstance(node, Literal) and isinstance(children, Literal):
            nodes = (<Literal>node).value
            childs = (<Literal>children).value
            if nodes.isinstance(model.Node) and childs.isinstance(model.Node):
                for n in nodes.values:
                    for c in reversed(childs.values):
                        n.insert(c.copy())
                return node
        return Prepend(node, children)

    def __repr__(self):
        return f'Prepend({self.node!r}, {self.children!r})'


cdef class Binding:
    cdef readonly str name
    cdef readonly Expression expr

    def __init__(self, str name, Expression expr=None):
        self.name = name
        self.expr = expr

    def __repr__(self):
        return f'Binding({self.name!r}, {self.expr!r})'


cdef class Let(Expression):
    cdef readonly tuple bindings

    def __init__(self, tuple bindings):
        self.bindings = bindings

    cpdef model.VectorLike evaluate(self, model.Context context):
        cdef Binding binding
        for binding in self.bindings:
            context.variables[binding.name] = binding.expr.evaluate(context)
        return model.null_

    cpdef Expression partially_evaluate(self, model.Context context):
        cdef list remaining = []
        cdef Binding binding
        cdef Expression expr
        for binding in self.bindings:
            expr = binding.expr.partially_evaluate(context)
            if isinstance(expr, Literal):
                context.variables[binding.name] = (<Literal>expr).value
            else:
                if binding.name in context.variables:
                    del context.variables[binding.name]
                remaining.append(Binding(binding.name, expr))
        if remaining:
            return Let(tuple(remaining))
        return Literal(model.null_)

    def __repr__(self):
        return f'Let({self.bindings!r})'


cdef class InlineLet(Expression):
    cdef readonly Expression body
    cdef readonly tuple bindings

    def __init__(self, Expression body, tuple bindings):
        self.body = body
        self.bindings = bindings

    cpdef model.VectorLike evaluate(self, model.Context context):
        cdef Binding binding
        cdef dict saved = context.variables
        context.variables = saved.copy()
        for binding in self.bindings:
            context.variables[binding.name] = binding.expr.evaluate(context)
        cdef model.VectorLike result = self.body.evaluate(context)
        context.variables = saved
        return result

    cpdef Expression partially_evaluate(self, model.Context context):
        cdef list remaining = []
        cdef Binding binding
        cdef Expression body, expr
        with context:
            for binding in self.bindings:
                expr = binding.expr.partially_evaluate(context)
                if isinstance(expr, Literal):
                    context.variables[binding.name] = (<Literal>expr).value
                else:
                    if binding.name in context.variables:
                        del context.variables[binding.name]
                    remaining.append(Binding(binding.name, expr))
            body = self.body.partially_evaluate(context)
            if remaining:
                return InlineLet(body, tuple(remaining))
            return body

    def __repr__(self):
        return f'InlineLet({self.body!r}, {self.bindings!r})'


cdef class For(Expression):
    cdef readonly str name
    cdef readonly Expression source
    cdef readonly Expression body

    def __init__(self, str name, Expression source, Expression body):
        self.name = name
        self.source = source
        self.body = body

    cpdef model.VectorLike evaluate(self, model.Context context):
        cdef model.Vector source = self.source.evaluate(context)
        cdef model.Vector results = model.Vector.__new__(model.Vector)
        cdef model.Vector value
        cdef dict saved = context.variables
        context.variables = saved.copy()
        for v in source.values:
            value = model.Vector.__new__(model.Vector)
            value.values.append(v)
            context.variables[self.name] = value
            results.values.extend(self.body.evaluate(context))
        context.variables = saved
        return results

    cpdef Expression partially_evaluate(self, model.Context context):
        cdef Expression source = self.source.partially_evaluate(context)
        cdef list remaining = []
        cdef model.Vector values, single
        with context:
            if not isinstance(source, Literal):
                if self.name in context.variables:
                    del context.variables[self.name]
                return For(self.name, source, self.body.partially_evaluate(context))
            values = (<Literal>source).value
            for value in values.values:
                single = model.Vector.__new__(model.Vector)
                single.values.append(value)
                context.variables[self.name] = single
                remaining.append(self.body.partially_evaluate(context))
            return sequence_pack(remaining)

    def __repr__(self):
        return f'For({self.name!r}, {self.source!r}, {self.body!r})'


cdef class Test:
    cdef readonly Expression condition
    cdef readonly Expression then

    def __init__(self, Expression condition, Expression then):
        self.condition = condition
        self.then = then

    def __repr__(self):
        return f'Test({self.condition!r}, {self.then!r})'


cdef class IfElse(Expression):
    cdef readonly tuple tests
    cdef readonly Expression else_

    def __init__(self, tuple tests, Expression else_=None):
        self.tests = tests
        self.else_ = else_

    cpdef model.VectorLike evaluate(self, model.Context context):
        cdef Test test
        for test in self.tests:
            if test.condition.evaluate(context).istrue():
                return test.then.evaluate(context)
        return self.else_.evaluate(context) if self.else_ is not None else model.null_

    cpdef Expression partially_evaluate(self, model.Context context):
        cdef list remaining = []
        cdef Test test
        cdef Expression condition, then
        for test in self.tests:
            condition = test.condition.partially_evaluate(context)
            then = test.then.partially_evaluate(context)
            if isinstance(condition, Literal):
                if (<Literal>condition).value.istrue():
                    if not remaining:
                        return then
                    else:
                        return IfElse(tuple(remaining), then)
            else:
                remaining.append(Test(condition, then))
        else_ = self.else_.partially_evaluate(context) if self.else_ is not None else None
        if remaining:
            return IfElse(tuple(remaining), else_)
        return Literal(model.null_) if else_ is None else else_

    def __repr__(self):
        return f'IfElse({self.tests!r}, {self.else_!r})'


cdef class Function(Expression):
    cdef readonly str name
    cdef readonly tuple parameters
    cdef readonly Expression expr

    def __init__(self, str name, tuple parameters, Expression expr):
        self.name = name
        self.parameters = parameters
        self.expr = expr

    cpdef model.VectorLike evaluate(self, model.Context context):
        cdef list parameters = []
        cdef Binding parameter
        cdef dict saved=context.variables, variables=saved.copy()
        for parameter in self.parameters:
            if parameter.name in variables:
                del variables[parameter.name]
            if parameter.expr is not None and not isinstance(parameter.expr, Literal):
                parameters.append(Binding(parameter.name, Literal(parameter.expr.evaluate(context))))
            else:
                parameters.append(parameter)
        context.variables = variables
        cdef Expression expr = self.expr.partially_evaluate(context)
        context.variables = saved
        cdef model.Vector func = model.Vector.__new__(model.Vector)
        func.values.append(Function(self.name, tuple(parameters), expr))
        context.variables[self.name] = func
        return model.null_

    cpdef Expression partially_evaluate(self, model.Context context):
        cdef list parameters = []
        cdef Binding parameter
        cdef Expression expr
        for parameter in self.parameters:
            parameters.append(Binding(parameter.name, parameter.expr.partially_evaluate(context) if parameter.expr is not None else None))
        with context:
            for parameter in parameters:
                if parameter.name in context.variables:
                    del context.variables[parameter.name]
            expr = self.expr.partially_evaluate(context)
        return Function(self.name, tuple(parameters), expr)

    def __repr__(self):
        return f'Function({self.name!r}, {self.parameters!r}, {self.expr!r})'
