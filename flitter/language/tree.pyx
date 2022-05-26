# cython: language_level=3, profile=True

"""
Language syntax/evaluation tree
"""

cimport cython

from . import functions
from .. cimport model


BUILTINS = {
    'true': model.true_,
    'false': model.false_,
    'null': model.null_,
    'uniform': model.Vector((functions.Uniform,)),
    'beta': model.Vector((functions.Beta,)),
    'normal': model.Vector((functions.Normal,)),
    'len': model.Vector((functions.length,)),
    'sine': model.Vector((functions.sine,)),
    'bounce': model.Vector((functions.bounce,)),
    'sharkfin': model.Vector((functions.sharkfin,)),
    'sawtooth': model.Vector((functions.sawtooth,)),
    'triangle': model.Vector((functions.triangle,)),
    'square': model.Vector((functions.square,)),
    'linear': model.Vector((functions.linear,)),
    'quad': model.Vector((functions.quad,)),
    'shuffle': model.Vector((functions.shuffle,)),
    'round': model.Vector((functions.roundv,)),
    'min': model.Vector((functions.minv,)),
    'max': model.Vector((functions.maxv,)),
    'hypot': model.Vector((functions.hypot,)),
    'map': model.Vector((functions.mapv,)),
    'hsl': model.Vector((functions.hsl,)),
    'hsv': model.Vector((functions.hsv,)),
}


cdef class Expression:
    cpdef model.VectorLike evaluate(self, model.Context context):
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

    def __repr__(self):
        return f'Sequence({self.expressions!r})'


cdef class Literal(Expression):
    cdef readonly model.Vector value

    def __init__(self, model.Vector value):
        self.value = value

    cpdef model.VectorLike evaluate(self, model.Context context):
        return self.value.copynodes()

    def __repr__(self):
        return f'Literal({self.value!r})'


cdef class Name(Expression):
    cdef readonly str name

    def __init__(self, str name):
        self.name = name

    cpdef model.VectorLike evaluate(self, model.Context context):
        cdef model.VectorLike result
        if self.name in context.variables:
            result = context.variables[self.name]
            return result.copynodes()
        cdef dict builtins = BUILTINS
        result = builtins.get(self.name)
        if result is not None:
            return result
        return model.null_

    def __repr__(self):
        return f'Name({self.name!r})'


cdef class Lookup(Expression):
    cdef readonly Expression key

    def __init__(self, Expression key):
        self.key = key

    cpdef model.VectorLike evaluate(self, model.Context context):
        cdef model.Vector key = self.key.evaluate(context)
        cdef model.Vector value = context.state.get(key)
        if value is not None:
            return value
        return model.null_

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

    def __repr__(self):
        return f'Range({self.start!r}, {self.stop!r}, {self.step!r})'


cdef class UnaryOperation(Expression):
    cdef readonly Expression expr

    def __init__(self, Expression expr):
        self.expr = expr

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

    def __repr__(self):
        return f'Slice({self.index!r}, {self.index!r})'


cdef class Call(Expression):
    cdef readonly Expression function
    cdef readonly tuple args

    def __init__(self, Expression function, tuple args):
        self.function = function
        self.args = args

    cpdef model.VectorLike evaluate(self, model.Context context):
        cdef model.Vector value, function = self.function.evaluate(context)
        cdef list args = []
        cdef Expression arg
        for arg in self.args:
            value = arg.evaluate(context)
            args.append(value)
        if len(function.values) == 1:
            return function.values[0](*args)
        cdef model.Vector result = model.Vector.__new__(model.Vector)
        for func in function.values:
            result.values.extend(func(*args))
        return result

    def __repr__(self):
        return f'Call({self.function!r}, {self.args!r})'


cdef class Node(Expression):
    cdef readonly str kind
    cdef readonly tuple tags

    def __init__(self, str kind, tuple tags):
        self.kind = kind
        self.tags = tags

    cpdef model.VectorLike evaluate(self, model.Context context):
        return model.Vector((model.Node(self.kind, self.tags),))

    def __repr__(self):
        return f'Node({self.kind!r}, {self.tags!r})'


cdef class Attribute(Expression):
    cdef readonly Expression node
    cdef readonly str name
    cdef readonly Expression expr

    def __init__(self, Expression node, str name, Expression expr):
        self.node = node
        self.name = name
        self.expr = expr

    cpdef model.VectorLike evaluate(self, model.Context context):
        cdef list nodes = []
        cdef model.Vector nodes_vector = self.node.evaluate(context)
        cdef model.Node node
        cdef model.Vector value
        cdef dict variables
        for node in nodes_vector.values:
            try:
                context._stack.append(None)
                context.merge_under(node)
                value = self.expr.evaluate(context)
                if value.values:
                    node.attributes[self.name] = value
                nodes.append(node)
            finally:
                variables = context._stack.pop()
                if variables is not None:
                    context.variables = variables
        return model.Vector(nodes)

    def __repr__(self):
        return f'Attribute({self.node!r}, {self.name!r}, {self.expr!r})'


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

    def __repr__(self):
        return f'Prepend({self.node!r}, {self.children!r})'


cdef class Binding:
    cdef readonly str name
    cdef readonly Expression expr

    def __init__(self, str name, Expression expr):
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
            context.setitem(binding.name,binding.expr.evaluate(context))
        return model.null_

    def __repr__(self):
        return f'Let({self.bindings!r})'


cdef class InlineLet(Expression):
    cdef readonly tuple bindings
    cdef readonly Expression body

    def __init__(self, tuple bindings, Expression body):
        self.bindings = bindings
        self.body = body

    cpdef model.VectorLike evaluate(self, model.Context context):
        cdef Binding binding
        with context:
            for binding in self.bindings:
                context.setitem(binding.name, binding.expr.evaluate(context))
            return self.body.evaluate(context)

    def __repr__(self):
        return f'InlineLet({self.bindings!r}, {self.body!r})'


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
        with context:
            for v in source.values:
                value = model.Vector.__new__(model.Vector)
                value.values.append(v)
                context.setitem(self.name, value)
                results.values.extend(self.body.evaluate(context))
        return results

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

    def __init__(self, tuple tests, Expression else_):
        self.tests = tests
        self.else_ = else_

    cpdef model.VectorLike evaluate(self, model.Context context):
        cdef Test test
        for test in self.tests:
            if test.condition.evaluate(context).istrue():
                return test.then.evaluate(context)
        return self.else_.evaluate(context) if self.else_ is not None else model.null_

    def __repr__(self):
        return f'IfElse({self.tests!r}, {self.else_!r})'
