# cython: language_level=3, profile=False

"""
Language syntax/evaluation tree
"""

cimport cython

from loguru import logger

from .. import name_patch
from ..cache import SharedCache
from .functions import STATIC_FUNCTIONS, DYNAMIC_FUNCTIONS
from .. cimport model
from .noise import NOISE_FUNCTIONS
from .vm cimport Context, StateDict, Program, builtins, static_builtins, dynamic_builtins
from .vm import log


logger = name_patch(logger, __name__)

cdef Literal NoOp = Literal(model.null_)


cdef Expression sequence_pack(list expressions):
    cdef Expression expr
    cdef Literal literal
    cdef model.Vector value
    cdef list vectors, remaining = []
    cdef bint has_let = False
    while expressions:
        expr = <Expression>expressions.pop(0)
        if isinstance(expr, Literal) and type((<Literal>expr).value) == model.Vector:
            vectors = []
            while isinstance(expr, Literal) and type((<Literal>expr).value) == model.Vector:
                value = (<Literal>expr).value
                if value.length:
                    vectors.append(value)
                if not expressions:
                    expr = None
                    break
                expr = <Expression>expressions.pop(0)
            if vectors:
                remaining.append(Literal(model.Vector._compose(vectors)))
        if expr is not None:
            if isinstance(expr, InlineSequence):
                expressions[:0] = (<InlineSequence>expr).expressions
                continue
            if isinstance(expr, Let):
                has_let = True
            remaining.append(expr)
    if len(remaining) == 0:
        return NoOp
    if has_let:
        for expr in remaining:
            if not isinstance(expr, Let):
                return Sequence(tuple(remaining))
        return NoOp
    if len(remaining) == 1:
        return remaining[0]
    return InlineSequence(tuple(remaining))


cdef class Expression:
    cpdef Program compile(self):
        raise NotImplementedError()

    cpdef Expression partially_evaluate(self, Context context):
        raise NotImplementedError()

    cpdef object reduce(self, func):
        return func(self)


cdef class Top(Expression):
    cdef readonly str path
    cdef readonly tuple expressions
    cdef readonly Program program

    def __init__(self, tuple expressions):
        self.expressions = expressions
        self.program = None

    def set_path(self, str path):
        self.path = path

    def run(self, StateDict state=None, dict variables=None, bint compile=False):
        cdef dict context_vars = None
        cdef str key
        if variables is not None:
            context_vars = {}
            for key, value in variables.items():
                context_vars[key] = model.Vector._coerce(value)
        cdef Context context = Context(state=state, variables=context_vars, path=self.path)
        if compile:
            self.compile().execute(context)
        else:
            self.evaluate(context)
        return context

    def simplify(self, StateDict state=None, dict variables=None, undefined=None):
        cdef dict context_vars = {}
        cdef str key
        if variables is not None:
            for key, value in variables.items():
                context_vars[key] = model.Vector._coerce(value)
        if undefined is not None:
            for key in undefined:
                context_vars[key] = None
        cdef Context context = Context(state=state, variables=context_vars)
        cdef Top top
        try:
            top = self.partially_evaluate(context)
        except Exception as exc:
            logger.opt(exception=exc).warning("Unable to partially-evaluate program")
            return self
        cdef str error
        for error in context.errors:
            logger.warning("Partial-evaluation error: {}", error)
        return top

    cpdef Program compile(self):
        cdef Expression expr
        if self.program is None:
            self.program = Program.__new__(Program)
            for expr in self.expressions:
                self.program.extend(expr.compile())
                self.program.append_root()
            logger.debug("Compiled tree to {} instructions", len(self.program))
        return self.program

    cpdef Expression partially_evaluate(self, Context context):
        cdef list expressions = []
        cdef Expression expr
        for expr in self.expressions:
            expr = expr.partially_evaluate(context)
            if not isinstance(expr, Literal) or (<Literal>expr).value.length:
                expressions.append(expr)
        cdef str name
        cdef model.Vector value
        cdef list bindings = []
        for name, value in context.variables.items():
            if value is not None:
                bindings.append(PolyBinding((name,), Literal(value)))
        if bindings:
            expressions.append(Let(tuple(bindings)))
        cdef Top top = Top(tuple(expressions))
        top.set_path(self.path)
        return top

    cpdef object reduce(self, func):
        cdef list children = []
        cdef Expression expr
        for expr in self.expressions:
            children.append(expr.reduce(func))
        return func(self, *children)

    def __repr__(self):
        return f'Top({self.expressions!r})'


cdef class Pragma(Expression):
    cdef readonly str name
    cdef readonly Expression expr

    def __init__(self, str name, Expression expr):
        self.name = name
        self.expr = expr

    cpdef Program compile(self):
        cdef Program program = Program.__new__(Program)
        program.extend(self.expr.compile())
        program.pragma(self.name)
        return program

    cpdef Expression partially_evaluate(self, Context context):
        return Pragma(self.name, self.expr.partially_evaluate(context))

    cpdef object reduce(self, func):
        return func(self, self.expr.reduce(func))

    def __repr__(self):
        return f'Pragma({self.name!r}, {self.expr!r})'


cdef class Import(Expression):
    cdef readonly tuple names
    cdef readonly Expression filename

    def __init__(self, tuple names, Expression filename):
        self.names = names
        self.filename = filename

    cpdef Program compile(self):
        cdef Program program = Program.__new__(Program)
        program.extend(self.filename.compile())
        program.import_(self.names)
        return program

    cpdef Expression partially_evaluate(self, Context context):
        cdef str name
        for name in self.names:
            context.variables[name] = None
        return Import(self.names, self.filename.partially_evaluate(context))

    cpdef object reduce(self, func):
        return func(self, self.filename.reduce(func))

    def __repr__(self):
        return f'Import({self.names!r}, {self.filename!r})'


cdef class Sequence(Expression):
    cdef readonly tuple expressions

    def __init__(self, tuple expressions):
        self.expressions = expressions

    cpdef Program compile(self):
        cdef Program program = Program.__new__(Program)
        cdef Expression expr
        program.begin_scope()
        for expr in self.expressions:
            program.extend(expr.compile())
        program.compose(len(self.expressions))
        program.end_scope()
        return program

    cpdef Expression partially_evaluate(self, Context context):
        cdef list expressions = []
        cdef Expression expr
        cdef dict saved = context.variables
        context.variables = saved.copy()
        for expr in self.expressions:
            expressions.append(expr.partially_evaluate(context))
        context.variables = saved
        return sequence_pack(expressions)

    cpdef object reduce(self, func):
        cdef list children = []
        cdef Expression expr
        for expr in self.expressions:
            children.append(expr.reduce(func))
        return func(self, *children)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.expressions!r})'


cdef class InlineSequence(Sequence):
    cpdef Program compile(self):
        cdef Program program = Program.__new__(Program)
        cdef Expression expr
        for expr in self.expressions:
            program.extend(expr.compile())
        program.compose(len(self.expressions))
        return program

    cpdef Expression partially_evaluate(self, Context context):
        cdef list expressions = []
        cdef Expression expr
        for expr in self.expressions:
            expressions.append(expr.partially_evaluate(context))
        return sequence_pack(expressions)


cdef class Literal(Expression):
    cdef readonly model.Vector value
    cdef bint copynodes

    def __init__(self, model.Vector value):
        self.value = value
        self.copynodes = False
        if self.value.objects is not None:
            for obj in self.value.objects:
                if isinstance(obj, model.Node):
                    self.copynodes = True
                    break

    cpdef Program compile(self):
        cdef Program program = Program.__new__(Program)
        program.literal(self.value)
        return program

    cpdef Expression partially_evaluate(self, Context context):
        return Literal(self.value.copynodes()) if self.copynodes else self

    def __repr__(self):
        return f'Literal({self.value!r})'


cdef class Name(Expression):
    cdef readonly str name

    def __init__(self, str name):
        self.name = name

    cpdef Program compile(self):
        cdef Program program = Program.__new__(Program)
        program.name(self.name)
        return program

    cpdef Expression partially_evaluate(self, Context context):
        cdef model.Vector value
        if self.name in context.variables:
            value = context.variables[self.name]
            if value is not None:
                return Literal(value.copynodes())
            else:
                return self
        elif (value := static_builtins.get(self.name)) is not None:
            return Literal(value)
        elif self.name not in dynamic_builtins:
            context.unbound.add(self.name)
        return self

    def __repr__(self):
        return f'Name({self.name!r})'


cdef class Lookup(Expression):
    cdef readonly Expression key

    def __init__(self, Expression key):
        self.key = key

    cpdef Program compile(self):
        cdef Program program = Program.__new__(Program)
        program.extend(self.key.compile())
        program.lookup()
        return program

    cpdef Expression partially_evaluate(self, Context context):
        cdef Expression key = self.key.partially_evaluate(context)
        cdef model.Vector value
        if isinstance(key, Literal) and context.state is not None:
            value = context.state.get_item((<Literal>key).value)
            return Literal(value)
        return Lookup(key)

    cpdef object reduce(self, func):
        return func(self, self.key.reduce(func))

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

    cpdef Program compile(self):
        cdef Program program = Program.__new__(Program)
        program.extend(self.start.compile())
        program.extend(self.stop.compile())
        program.extend(self.step.compile())
        program.range()
        return program

    cpdef Expression partially_evaluate(self, Context context):
        cdef Expression start = self.start.partially_evaluate(context)
        cdef Expression stop = self.stop.partially_evaluate(context)
        cdef Expression step = self.step.partially_evaluate(context)
        cdef model.Vector result
        if isinstance(start, Literal) and isinstance(stop, Literal) and isinstance(step, Literal):
            result = model.Vector.__new__(model.Vector)
            result.fill_range((<Literal>start).value, (<Literal>stop).value, (<Literal>step).value)
            return Literal(result)
        return Range(start, stop, step)

    cpdef object reduce(self, func):
        return func(self, self.start.reduce(func), self.stop.reduce(func), self.step.reduce(func))

    def __repr__(self):
        return f'Range({self.start!r}, {self.stop!r}, {self.step!r})'


cdef class UnaryOperation(Expression):
    cdef readonly Expression expr

    def __init__(self, Expression expr):
        self.expr = expr

    cpdef object reduce(self, func):
        return func(self, self.expr.reduce(func))

    def __repr__(self):
        return f'{self.__class__.__name__}({self.expr!r})'


cdef class Negative(UnaryOperation):
    cpdef Program compile(self):
        cdef Program program = Program.__new__(Program)
        program.extend(self.expr.compile())
        program.neg()
        return program

    cpdef Expression partially_evaluate(self, Context context):
        cdef Expression expr = self.expr.partially_evaluate(context)
        if isinstance(expr, Literal):
            return Literal((<Literal>expr).value.neg())
        if isinstance(expr, Negative):
            expr = Positive((<Negative>expr).expr)
            return expr.partially_evaluate(context)
        cdef MathsBinaryOperation maths
        if isinstance(expr, (Multiply, Divide)):
            maths = expr
            if isinstance(maths.left, Literal):
                expr = type(expr)(Negative(maths.left), maths.right)
                return expr.partially_evaluate(context)
            if isinstance(maths.right, Literal):
                expr = type(expr)(maths.left, Negative(maths.right))
                return expr.partially_evaluate(context)
        return Negative(expr)


cdef class Positive(UnaryOperation):
    cpdef Program compile(self):
        cdef Program program = Program.__new__(Program)
        program.extend(self.expr.compile())
        program.pos()
        return program

    cpdef Expression partially_evaluate(self, Context context):
        cdef Expression expr = self.expr.partially_evaluate(context)
        if isinstance(expr, Literal):
            return Literal((<Literal>expr).value.pos())
        if isinstance(expr, (Negative, Positive, MathsBinaryOperation)):
            return expr.partially_evaluate(context)
        return Positive(expr)


cdef class Not(UnaryOperation):
    cpdef Program compile(self):
        cdef Program program = Program.__new__(Program)
        program.extend(self.expr.compile())
        program.not_()
        return program

    cpdef Expression partially_evaluate(self, Context context):
        cdef Expression expr = self.expr.partially_evaluate(context)
        if isinstance(expr, Literal):
            return Literal((<Literal>expr).value.not_())
        return Not(expr)


cdef class BinaryOperation(Expression):
    cdef readonly Expression left
    cdef readonly Expression right

    def __init__(self, Expression left, Expression right):
        self.left = left
        self.right = right

    cpdef Program compile(self):
        cdef Program program = Program.__new__(Program)
        program.extend(self.left.compile())
        program.extend(self.right.compile())
        self.compile_op(program)
        return program

    cdef void compile_op(self, Program program):
        raise NotImplementedError()

    cpdef Expression partially_evaluate(self, Context context):
        cdef Expression left = self.left.partially_evaluate(context)
        cdef Expression right = self.right.partially_evaluate(context)
        cdef bint literal_left = isinstance(left, Literal)
        cdef bint literal_right = isinstance(right, Literal)
        if literal_left and literal_right:
            return Literal(self.op((<Literal>left).value, (<Literal>right).value))
        elif literal_left:
            if (expr := self.constant_left((<Literal>left).value, right)) is not None:
                return expr.partially_evaluate(context)
        elif literal_right:
            if (expr := self.constant_right(left, (<Literal>right).value)) is not None:
                return expr.partially_evaluate(context)
        return type(self)(left, right)

    cdef model.Vector op(self, model.Vector left, model.Vector right):
        raise NotImplementedError()

    cdef Expression constant_left(self, model.Vector left, Expression right):
        return None

    cdef Expression constant_right(self, Expression left, model.Vector right):
        return None

    cpdef object reduce(self, func):
        return func(self, self.left.reduce(func), self.right.reduce(func))

    def __repr__(self):
        return f'{self.__class__.__name__}({self.left!r}, {self.right!r})'


cdef class MathsBinaryOperation(BinaryOperation):
    pass


cdef class Add(MathsBinaryOperation):
    cdef model.Vector op(self, model.Vector left, model.Vector right):
        return left.add(right)

    cdef void compile_op(self, Program program):
        program.add()

    cdef Expression constant_left(self, model.Vector left, Expression right):
        if left.eq(model.false_):
            return Positive(right)

    cdef Expression constant_right(self, Expression left, model.Vector right):
        return self.constant_left(right, left)


cdef class Subtract(MathsBinaryOperation):
    cdef model.Vector op(self, model.Vector left, model.Vector right):
        return left.sub(right)

    cdef void compile_op(self, Program program):
        program.sub()

    cdef Expression constant_left(self, model.Vector left, Expression right):
        if left.eq(model.false_):
            return Negative(right)

    cdef Expression constant_right(self, Expression left, model.Vector right):
        if right.eq(model.false_):
            return Positive(left)


cdef class Multiply(MathsBinaryOperation):
    cdef model.Vector op(self, model.Vector left, model.Vector right):
        return left.mul(right)

    cdef void compile_op(self, Program program):
        program.mul()

    cdef Expression constant_left(self, model.Vector left, Expression right):
        if left.eq(model.true_):
            return Positive(right)
        if left.eq(model.minusone_):
            return Negative(right)
        cdef MathsBinaryOperation maths
        if isinstance(right, Add) or isinstance(right, Subtract):
            maths = right
            if isinstance(maths.left, Literal) or isinstance(maths.right, Literal):
                return type(maths)(Multiply(Literal(left), maths.left), Multiply(Literal(left), maths.right))
        elif isinstance(right, Multiply):
            maths = right
            if isinstance(maths.left, Literal):
                return Multiply(Multiply(Literal(left), maths.left), maths.right)
            if isinstance(maths.right, Literal):
                return Multiply(maths.left, Multiply(Literal(left), maths.right))
        elif isinstance(right, Divide):
            maths = right
            if isinstance(maths.left, Literal):
                return Divide(Multiply(Literal(left), maths.left), maths.right)

    cdef Expression constant_right(self, Expression left, model.Vector right):
        return self.constant_left(right, left)


cdef class Divide(MathsBinaryOperation):
    cdef model.Vector op(self, model.Vector left, model.Vector right):
        return left.truediv(right)

    cdef void compile_op(self, Program program):
        program.truediv()

    cdef Expression constant_right(self, Expression left, model.Vector right):
        if right.eq(model.true_):
            return Positive(left)


cdef class FloorDivide(MathsBinaryOperation):
    cdef model.Vector op(self, model.Vector left, model.Vector right):
        return left.floordiv(right)

    cdef void compile_op(self, Program program):
        program.floordiv()


cdef class Modulo(MathsBinaryOperation):
    cdef model.Vector op(self, model.Vector left, model.Vector right):
        return left.mod(right)

    cdef void compile_op(self, Program program):
        program.mod()


cdef class Power(MathsBinaryOperation):
    cdef model.Vector op(self, model.Vector left, model.Vector right):
        return left.pow(right)

    cdef void compile_op(self, Program program):
        program.pow()

    cdef Expression constant_right(self, Expression left, model.Vector right):
        if right.eq(model.true_):
            return Positive(left)


cdef class Comparison(BinaryOperation):
    pass


cdef class EqualTo(Comparison):
    cdef model.Vector op(self, model.Vector left, model.Vector right):
        return left.eq(right)

    cdef void compile_op(self, Program program):
        program.eq()


cdef class NotEqualTo(Comparison):
    cdef model.Vector op(self, model.Vector left, model.Vector right):
        return left.ne(right)

    cdef void compile_op(self, Program program):
        program.ne()


cdef class LessThan(Comparison):
    cdef model.Vector op(self, model.Vector left, model.Vector right):
        return left.lt(right)

    cdef void compile_op(self, Program program):
        program.lt()


cdef class GreaterThan(Comparison):
    cdef model.Vector op(self, model.Vector left, model.Vector right):
        return left.gt(right)

    cdef void compile_op(self, Program program):
        program.gt()


cdef class LessThanOrEqualTo(Comparison):
    cdef model.Vector op(self, model.Vector left, model.Vector right):
        return left.le(right)

    cdef void compile_op(self, Program program):
        program.le()


cdef class GreaterThanOrEqualTo(Comparison):
    cdef model.Vector op(self, model.Vector left, model.Vector right):
        return left.ge(right)

    cdef void compile_op(self, Program program):
        program.ge()


cdef class And(BinaryOperation):
    cdef model.Vector op(self, model.Vector left, model.Vector right):
        return right if left.as_bool() else left

    cpdef Program compile(self):
        cdef Program program = Program.__new__(Program)
        end_label = program.new_label()
        program.extend(self.left.compile())
        program.dup()
        program.branch_false(end_label)
        program.drop()
        program.extend(self.right.compile())
        program.label(end_label)
        return program

    cdef Expression constant_left(self, model.Vector left, Expression right):
        if left.as_bool():
            return right
        else:
            return Literal(left)


cdef class Or(BinaryOperation):
    cdef model.Vector op(self, model.Vector left, model.Vector right):
        return left if left.as_bool() else right

    cpdef Program compile(self):
        cdef Program program = Program.__new__(Program)
        end_label = program.new_label()
        program.extend(self.left.compile())
        program.dup()
        program.branch_true(end_label)
        program.drop()
        program.extend(self.right.compile())
        program.label(end_label)
        return program

    cdef Expression constant_left(self, model.Vector left, Expression right):
        if left.as_bool():
            return Literal(left)
        else:
            return right


cdef class Xor(BinaryOperation):
    cdef model.Vector op(self, model.Vector left, model.Vector right):
        if not left.as_bool():
            return right
        if not right.as_bool():
            return left
        return model.false_

    cdef void compile_op(self, Program program):
        program.xor()

    cdef Expression constant_left(self, model.Vector left, Expression right):
        if not left.as_bool():
            return right


cdef class Slice(Expression):
    cdef readonly Expression expr
    cdef readonly Expression index

    def __init__(self, Expression expr, Expression index):
        self.expr = expr
        self.index = index

    cpdef Program compile(self):
        cdef Program program = Program.__new__(Program)
        program.extend(self.expr.compile())
        program.extend(self.index.compile())
        program.slice()
        return program

    cpdef Expression partially_evaluate(self, Context context):
        cdef Expression expr = self.expr.partially_evaluate(context)
        cdef Expression index = self.index.partially_evaluate(context)
        cdef model.Vector expr_value
        cdef model.Vector index_value
        cdef str name
        if isinstance(expr, Literal) and isinstance(index, Literal):
            expr_value = (<Literal>expr).value
            index_value = (<Literal>index).value
            return Literal(expr_value.slice(index_value))
        elif isinstance(index, Literal):
            index_value = (<Literal>index).value
            return FastSlice(expr, index_value)
        return Slice(expr, index)

    cpdef object reduce(self, func):
        return func(self, self.expr.reduce(func), self.index.reduce(func))

    def __repr__(self):
        return f'Slice({self.expr!r}, {self.index!r})'


cdef class FastSlice(Expression):
    cdef readonly Expression expr
    cdef readonly model.Vector index

    def __init__(self, Expression expr, model.Vector index):
        self.expr = expr
        self.index = index

    cpdef Program compile(self):
        cdef Program program = Program.__new__(Program)
        program.extend(self.expr.compile())
        program.slice_literal(self.index)
        return program

    cpdef Expression partially_evaluate(self, Context context):
        return self

    def __repr__(self):
        return f'FastSlice({self.expr!r}, {self.index!r})'


cdef class Call(Expression):
    cdef readonly Expression function
    cdef readonly tuple args

    def __init__(self, Expression function, tuple args):
        self.function = function
        self.args = args

    cpdef Program compile(self):
        cdef Program program = Program.__new__(Program)
        cdef Expression expr
        for expr in self.args:
            program.extend(expr.compile())
        program.extend(self.function.compile())
        program.call(len(self.args))
        return program

    cpdef Expression partially_evaluate(self, Context context):
        cdef Expression function = self.function.partially_evaluate(context)
        cdef list args = []
        cdef Expression arg
        cdef bint literal_function = isinstance(function, Literal) and (<Literal>function).value.objects is not None
        cdef bint literal = literal_function
        for arg in self.args:
            arg = arg.partially_evaluate(context)
            args.append(arg)
            if not isinstance(arg, Literal):
                literal = False
        cdef list vector_args, results
        cdef Literal literal_arg
        cdef Function func_expr
        cdef dict saved, params
        cdef Binding parameter
        cdef int i
        if literal:
            vector_args = [literal_arg.value for literal_arg in args]
            results = []
            for func in (<Literal>function).value.objects:
                if callable(func):
                    try:
                        if hasattr(func, 'state_transformer') and func.state_transformer:
                            results.append(Literal(func(context.state, *vector_args)))
                        else:
                            results.append(Literal(func(*vector_args)))
                    except Exception:
                        break
                elif isinstance(func, Function):
                    func_expr = func
                    saved = context.variables
                    context.variables = {}
                    for i, parameter in enumerate(func_expr.parameters):
                        if i < len(vector_args):
                            context.variables[parameter.name] = vector_args[i]
                        elif parameter.expr is not None:
                            context.variables[parameter.name] = (<Literal>parameter.expr).value
                        else:
                            context.variables[parameter.name] = model.null_
                    results.append(func_expr.expr.partially_evaluate(context))
                    context.variables = saved
            else:
                return sequence_pack(results)
        elif literal_function and (<Literal>function).value.length == 1:
            func = (<Literal>function).value.objects[0]
            if callable(func) and not hasattr(func, 'state_transformer') and func is not log:
                return FastCall(func, tuple(args))
        cdef Call call = Call(function, tuple(args))
        return call

    cpdef object reduce(self, func):
        cdef list children = []
        children.append(self.function.reduce(func))
        cdef Expression arg
        for arg in self.args:
            children.append(arg.reduce(func))
        return func(self, *children)

    def __repr__(self):
        return f'Call({self.function!r}, {self.args!r})'


cdef class FastCall(Expression):
    cdef readonly object function
    cdef readonly tuple args

    def __init__(self, function, tuple args):
        self.function = function
        self.args = args

    cpdef Program compile(self):
        cdef Program program = Program.__new__(Program)
        cdef Expression expr
        for expr in self.args:
            program.extend(expr.compile())
        program.literal(model.Vector._coerce(self.function))
        program.call(len(self.args))
        return program

    cpdef Expression partially_evaluate(self, Context context):
        cdef Call call = Call(Literal(model.Vector._coerce(self.function)), self.args)
        return call.partially_evaluate(context)

    cpdef object reduce(self, func):
        cdef list children = []
        cdef Expression arg
        for arg in self.args:
            children.append(arg.reduce(func))
        return func(self, *children)

    def __repr__(self):
        return f'FastCall({self.function!r}, {self.args!r})'


cdef class Tag(Expression):
    cdef readonly Expression node
    cdef readonly str tag

    def __init__(self, Expression node, str tag):
        self.node = node
        self.tag = tag

    cpdef Program compile(self):
        cdef Program program = Program.__new__(Program)
        program.extend(self.node.compile())
        program.tag(self.tag)
        return program

    cpdef Expression partially_evaluate(self, Context context):
        cdef Expression node = self.node.partially_evaluate(context)
        cdef model.Vector nodes
        cdef model.Node n
        if isinstance(node, Literal):
            nodes = (<Literal>node).value
            if nodes.isinstance(model.Node):
                for n in nodes.objects:
                    n.add_tag(self.tag)
                return node
        return Tag(node, self.tag)

    cpdef object reduce(self, func):
        return func(self, self.node.reduce(func))

    def __repr__(self):
        return f'Tag({self.node!r}, {self.tag!r})'


cdef class Attributes(Expression):
    cdef readonly Expression node
    cdef readonly tuple bindings

    def __init__(self, Expression node, tuple bindings):
        self.node = node
        self.bindings = bindings

    cpdef Program compile(self):
        cdef Program program = Program.__new__(Program)
        program.extend(self.node.compile())
        program.begin_for()
        START = program.new_label()
        END = program.new_label()
        program.label(START)
        program.push_next(END)
        program.set_node_scope()
        cdef Binding binding
        for binding in self.bindings:
            program.extend(binding.expr.compile())
            program.attribute(binding.name)
        program.jump(START)
        program.label(END)
        program.end_for()
        program.clear_node_scope()
        return program

    cpdef Expression partially_evaluate(self, Context context):
        cdef Expression node = self
        cdef list bindings = []
        cdef Attributes attrs
        cdef Binding binding
        cdef set unbound = context.unbound
        context.unbound = set()
        while isinstance(node, Attributes):
            attrs = <Attributes>node
            for binding in reversed(attrs.bindings):
                bindings.append(Binding(binding.name, binding.expr.partially_evaluate(context)))
            node = attrs.node
        cdef bint fast = not context.unbound
        context.unbound = unbound
        node = node.partially_evaluate(context)
        cdef model.Vector nodes
        cdef model.Node n
        if isinstance(node, Literal):
            nodes = (<Literal>node).value
            if nodes.isinstance(model.Node):
                while bindings and isinstance((<Binding>bindings[-1]).expr, Literal):
                    binding = bindings.pop()
                    for n in nodes.objects:
                        n[binding.name] = (<Literal>binding.expr).value
        if not bindings:
            return node
        bindings.reverse()
        if fast:
            return FastAttributes(node, tuple(bindings))
        return Attributes(node, tuple(bindings))

    cpdef object reduce(self, func):
        cdef list children = [self.node.reduce(func)]
        cdef Binding binding
        for binding in self.bindings:
            children.append(binding.expr.reduce(func))
        return func(self, *children)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.node!r}, {self.bindings!r})'


cdef class FastAttributes(Attributes):
    cpdef Program compile(self):
        cdef Program program = Program.__new__(Program)
        program.extend(self.node.compile())
        cdef Binding binding
        for binding in self.bindings:
            program.extend(binding.expr.compile())
            program.attribute(binding.name)
        return program


cdef class Search(Expression):
    cdef readonly model.Query query

    def __init__(self, model.Query query):
        self.query = query

    cpdef Program compile(self):
        cdef Program program = Program.__new__(Program)
        program.search(self.query)
        return program

    cpdef Expression partially_evaluate(self, Context context):
        return self

    def __repr__(self):
        return f'Search({self.query!r})'


cdef class Append(Expression):
    cdef readonly Expression node
    cdef readonly Expression children

    def __init__(self, Expression node, Expression children):
        self.node = node
        self.children = children

    cpdef Program compile(self):
        cdef Program program = Program.__new__(Program)
        program.extend(self.node.compile())
        program.extend(self.children.compile())
        program.append()
        return program

    cpdef Expression partially_evaluate(self, Context context):
        cdef Expression node = self.node.partially_evaluate(context)
        cdef Expression children = self.children.partially_evaluate(context)
        cdef model.Vector nodes, childs
        cdef model.Node n, c
        if isinstance(node, Literal) and isinstance(children, Literal):
            nodes = (<Literal>node).value
            childs = (<Literal>children).value
            if nodes.isinstance(model.Node) and childs.isinstance(model.Node):
                for n in nodes.objects:
                    for c in childs.objects:
                        n.append(c.copy())
                return node
        return Append(node, children)

    cpdef object reduce(self, func):
        return func(self, self.node.reduce(func), self.children.reduce(func))

    def __repr__(self):
        return f'{self.__class__.__name__}({self.node!r}, {self.children!r})'


cdef class Prepend(Append):
    cpdef Program compile(self):
        cdef Program program = Program.__new__(Program)
        program.extend(self.node.compile())
        program.extend(self.children.compile())
        program.prepend()
        return program

    cpdef Expression partially_evaluate(self, Context context):
        cdef Expression node = self.node.partially_evaluate(context)
        cdef Expression children = self.children.partially_evaluate(context)
        cdef model.Vector nodes, childs
        cdef model.Node n, c
        if isinstance(node, Literal) and isinstance(children, Literal):
            nodes = (<Literal>node).value
            childs = (<Literal>children).value
            if nodes.isinstance(model.Node) and childs.isinstance(model.Node):
                for n in nodes.objects:
                    for c in reversed(childs.objects):
                        n.insert(c.copy())
                return node
        return Prepend(node, children)


cdef class Binding:
    cdef readonly str name
    cdef readonly Expression expr

    def __init__(self, str name, Expression expr):
        self.name = name
        self.expr = expr

    def __repr__(self):
        return f'Binding({self.name!r}, {self.expr!r})'


cdef class PolyBinding:
    cdef readonly tuple names
    cdef readonly Expression expr

    def __init__(self, tuple names, Expression expr):
        self.names = names
        self.expr = expr

    def __repr__(self):
        return f'PolyBinding({self.names!r}, {self.expr!r})'


cdef class Let(Expression):
    cdef readonly tuple bindings

    def __init__(self, tuple bindings):
        self.bindings = bindings

    cpdef Program compile(self):
        cdef Program program = Program.__new__(Program)
        cdef PolyBinding binding
        for binding in self.bindings:
            program.extend(binding.expr.compile())
            program.let(binding.names)
        program.literal(model.null_)
        return program

    cpdef Expression partially_evaluate(self, Context context):
        cdef list remaining = []
        cdef PolyBinding binding
        cdef Expression expr
        cdef model.Vector value
        cdef str name
        cdef int i, n
        for binding in self.bindings:
            expr = binding.expr.partially_evaluate(context)
            if isinstance(expr, Literal):
                value = (<Literal>expr).value
                n = len(binding.names)
                if n == 1:
                    name = binding.names[0]
                    context.variables[name] = value
                else:
                    for i, name in enumerate(binding.names):
                        context.variables[name] = value.item(i)
            else:
                for name in binding.names:
                    context.variables[name] = None
                remaining.append(PolyBinding(binding.names, expr))
        if remaining:
            return Let(tuple(remaining))
        return NoOp

    cpdef object reduce(self, func):
        cdef list children = []
        cdef PolyBinding binding
        for binding in self.bindings:
            children.append(binding.expr.reduce(func))
        return func(self, *children)

    def __repr__(self):
        return f'Let({self.bindings!r})'


cdef class InlineLet(Expression):
    cdef readonly Expression body
    cdef readonly tuple bindings

    def __init__(self, Expression body, tuple bindings):
        self.body = body
        self.bindings = bindings

    cpdef Program compile(self):
        cdef Program program = Program.__new__(Program)
        cdef PolyBinding binding
        program.begin_scope()
        for binding in self.bindings:
            program.extend(binding.expr.compile())
            program.let(binding.names)
        program.extend(self.body.compile())
        program.end_scope()
        return program

    cpdef Expression partially_evaluate(self, Context context):
        cdef dict saved = context.variables
        context.variables = saved.copy()
        cdef list remaining = []
        cdef PolyBinding binding
        cdef Expression expr
        cdef model.Vector value
        cdef str name
        cdef int i, n
        for binding in self.bindings:
            expr = binding.expr.partially_evaluate(context)
            if isinstance(expr, Literal):
                value = (<Literal>expr).value
                n = len(binding.names)
                if n == 1:
                    name = binding.names[0]
                    context.variables[name] = value
                else:
                    for i, name in enumerate(binding.names):
                        context.variables[name] = value.item(i)
            else:
                for name in binding.names:
                    context.variables[name] = None
                remaining.append(PolyBinding(binding.names, expr))
        body = self.body.partially_evaluate(context)
        context.variables = saved
        if remaining:
            return InlineLet(body, tuple(remaining))
        return body

    cpdef object reduce(self, func):
        cdef list children = [self.body.reduce(func)]
        cdef PolyBinding binding
        for binding in self.bindings:
            children.append(binding.expr.reduce(func))
        return func(self, *children)

    def __repr__(self):
        return f'InlineLet({self.body!r}, {self.bindings!r})'


cdef class For(Expression):
    cdef readonly tuple names
    cdef readonly Expression source
    cdef readonly Expression body

    def __init__(self, tuple names, Expression source, Expression body):
        self.names = names
        self.source = source
        self.body = body

    cpdef Program compile(self):
        cdef Program program = Program.__new__(Program)
        program.extend(self.source.compile())
        program.begin_for()
        START = program.new_label()
        END = program.new_label()
        program.label(START)
        program.next(self.names, END)
        program.extend(self.body.compile())
        program.jump(START)
        program.label(END)
        program.end_for()
        return program

    cpdef Expression partially_evaluate(self, Context context):
        cdef Expression body, source=self.source.partially_evaluate(context)
        cdef list remaining = []
        cdef model.Vector values, single
        cdef dict saved = context.variables
        context.variables = saved.copy()
        cdef str name
        if not isinstance(source, Literal):
            for name in self.names:
                context.variables[name] = None
            body = self.body.partially_evaluate(context)
            context.variables = saved
            return For(self.names, source, body)
        values = (<Literal>source).value
        cdef int i=0, n=values.length
        while i < n:
            for name in self.names:
                context.variables[name] = values.item(i)
                i += 1
            remaining.append(self.body.partially_evaluate(context))
        context.variables = saved
        return sequence_pack(remaining)

    cpdef object reduce(self, func):
        return func(self, self.source.reduce(func), self.body.reduce(func))

    def __repr__(self):
        return f'For({self.names!r}, {self.source!r}, {self.body!r})'


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

    cpdef Program compile(self):
        cdef Program program = Program.__new__(Program)
        cdef Test test
        END = program.new_label()
        for test in self.tests:
            NEXT = program.new_label()
            program.extend(test.condition.compile())
            program.branch_false(NEXT)
            program.extend(test.then.compile())
            program.jump(END)
            program.label(NEXT)
        if self.else_ is not None:
            program.extend(self.else_.compile())
        else:
            program.literal(model.null_)
        program.label(END)
        return program

    cpdef Expression partially_evaluate(self, Context context):
        cdef list remaining = []
        cdef Test test
        cdef Expression condition, then
        for test in self.tests:
            condition = test.condition.partially_evaluate(context)
            then = test.then.partially_evaluate(context)
            if isinstance(condition, Literal):
                if (<Literal>condition).value.as_bool():
                    if not remaining:
                        return then
                    else:
                        return IfElse(tuple(remaining), then)
            else:
                remaining.append(Test(condition, then))
        else_ = self.else_.partially_evaluate(context) if self.else_ is not None else None
        if remaining:
            return IfElse(tuple(remaining), else_)
        return NoOp if else_ is None else else_

    cpdef object reduce(self, func):
        cdef list children = []
        cdef Test test
        for test in self.tests:
            children.append(test.condition.reduce(func))
            children.append(test.then.reduce(func))
        if self.else_ is not None:
            children.append(self.else_.reduce(func))
        return func(self, *children)

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

    cpdef Program compile(self):
        cdef Program program = Program.__new__(Program)
        program.literal(self.expr.compile())
        cdef Binding parameter
        cdef list names = []
        for parameter in self.parameters:
            names.append(parameter.name)
            if parameter.expr is None:
                program.literal(model.null_)
            else:
                program.extend(parameter.expr.compile())
        program.func((self.name, *names))
        program.let((self.name,))
        program.literal(model.null_)
        return program

    cpdef Expression partially_evaluate(self, Context context):
        cdef list parameters = []
        cdef Binding parameter
        cdef Expression expr
        cdef bint literal = True
        for parameter in self.parameters:
            expr = parameter.expr.partially_evaluate(context) if parameter.expr is not None else None
            if expr is not None and not isinstance(expr, Literal):
                literal = False
            parameters.append(Binding(parameter.name, expr))
        cdef dict saved = context.variables
        cdef set unbound = context.unbound
        cdef str key
        cdef model.Vector value
        context.variables = {}
        for key, value in saved.items():
            if value is not None:
                context.variables[key] = value
        for parameter in parameters:
            context.variables[parameter.name] = None
        context.unbound = set()
        expr = self.expr.partially_evaluate(context)
        context.variables = saved
        context.unbound = unbound
        context.variables[self.name] = None
        return Function(self.name, tuple(parameters), expr)

    cpdef object reduce(self, func):
        cdef list children = []
        cdef Binding parameter
        for parameter in self.parameters:
            if parameter.expr is not None:
                children.append(parameter.expr.reduce(func))
        children.append(self.expr.reduce(func))
        return func(self, *children)

    def __repr__(self):
        return f'Function({self.name!r}, {self.parameters!r}, {self.expr!r})'


cdef class TemplateCall(Expression):
    cdef readonly Expression function
    cdef readonly tuple args
    cdef readonly Expression children

    def __init__(self, Expression function, tuple args, Expression children):
        self.function = function
        self.args = args
        self.children = children

    cpdef Program compile(self):
        cdef Program program = Program.__new__(Program)
        cdef Binding arg
        cdef list names = []
        if self.children is not None:
            program.extend(self.children.compile())
        else:
            program.literal(model.null_)
        if self.args:
            for arg in self.args:
                names.append(arg.name)
                program.extend(arg.expr.compile())
        program.extend(self.function.compile())
        program.call(1, tuple(names))
        return program

    cpdef Expression partially_evaluate(self, Context context):
        cdef Expression function = self.function.partially_evaluate(context)
        cdef literal = isinstance(function, Literal)
        cdef Expression children = None
        if self.children is not None:
            children = self.children.partially_evaluate(context)
            literal = literal and isinstance(children, Literal)
        cdef list args = None
        cdef Binding arg
        cdef Expression value
        if self.args is not None:
            args = []
            for arg in self.args:
                value = arg.expr.partially_evaluate(context)
                if not isinstance(value, Literal):
                    literal = False
                args.append(Binding(arg.name, value))
        cdef TemplateCall call = TemplateCall(function, tuple(args) if args is not None else None, children)
        if literal:
            return Literal(call.evaluate(context))
        return call

    cpdef object reduce(self, func):
        cdef list children = [self.function.reduce(func)]
        cdef Binding binding
        if self.args is not None:
            for arg in self.args:
                children.append(arg.expr.reduce(func))
        if self.children is not None:
            children.append(self.children.reduce(func))
        return func(self, *children)

    def __repr__(self):
        return f'TemplateCall({self.function!r}, {self.args!r}, {self.children!r})'
