# cython: language_level=3, profile=False

"""
Language abstract syntax tree

The tree supports reasonably sophisticated simplification that will reduce
the tree down to a "simpler" form by propogating constants. As this can result
in unrolling loops, simpler does not necessarily mean "smaller."
"""

from loguru import logger

from libc.stdint cimport int64_t

from .. import name_patch
from ..model cimport Vector, Node, Context, StateDict, null_, true_, false_, minusone_
from .vm cimport Program, static_builtins, dynamic_builtins


logger = name_patch(logger, __name__)

cdef Literal NoOp = Literal(null_)


cdef Expression sequence_pack(list expressions):
    cdef Expression expr
    cdef Vector value
    cdef list vectors, remaining = []
    cdef bint has_let = False
    while expressions:
        expr = <Expression>expressions.pop(0)
        if isinstance(expr, Literal) and type((<Literal>expr).value) == Vector:
            vectors = []
            while isinstance(expr, Literal) and type((<Literal>expr).value) == Vector:
                value = (<Literal>expr).value
                if value.length:
                    vectors.append(value)
                if not expressions:
                    expr = None
                    break
                expr = <Expression>expressions.pop(0)
            if vectors:
                remaining.append(Literal(Vector._compose(vectors)))
        if expr is not None:
            if isinstance(expr, InlineSequence):
                expressions[:0] = (<InlineSequence>expr).expressions
                continue
            if isinstance(expr, (Let, Import, Function)):
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
    def compile(self, tuple initial_lnames=()):
        cdef Program program = Program.__new__(Program)
        program.initial_lnames = initial_lnames
        self._compile(program, list(initial_lnames))
        program.optimize()
        program.link()
        return program

    def simplify(self, StateDict state=None, dict static=None, dynamic=None):
        cdef dict context_vars = {}
        cdef str key
        if static is not None:
            for key, value in static.items():
                context_vars[key] = Vector._coerce(value)
        if dynamic is not None:
            for key in dynamic:
                context_vars[key] = None
        cdef Context context = Context(state=state, names=context_vars)
        cdef Expression expr
        try:
            expr = self._simplify(context)
        except Exception as exc:
            logger.opt(exception=exc).warning("Unable to simplify program")
            return self
        cdef str error
        for error in context.errors:
            logger.warning("Simplification error: {}", error)
        return expr

    cdef void _compile(self, Program program, list lnames):
        raise NotImplementedError()

    cdef Expression _simplify(self, Context context):
        raise NotImplementedError()


cdef class Top(Expression):
    cdef readonly tuple expressions

    def __init__(self, tuple expressions):
        self.expressions = expressions

    cdef void _compile(self, Program program, list lnames):
        cdef Expression expr
        cdef int64_t m = 0
        for expr in self.expressions:
            expr._compile(program, lnames)
            if not isinstance(expr, (Let, Import, Function, Pragma, StoreGlobal)):
                m += 1
        if m:
            program.append(m)
        cdef str name
        m = len(program.initial_lnames)
        cdef int64_t i, n=len(lnames)
        if n > m:
            for i in range(n-m):
                name = lnames[n-1-i]
                program.local_load(i)
                program.store_global(name)
            program.local_drop(n-m)

    cdef Expression _simplify(self, Context context):
        cdef list expressions = []
        cdef Expression expr
        cdef dict names = dict(context.names)
        for expr in self.expressions:
            expr = expr._simplify(context)
            if not isinstance(expr, Literal) or (<Literal>expr).value.length:
                expressions.append(expr)
        cdef str name
        cdef list bindings = []
        for name, value in context.names.items():
            if name not in names and value is not None and isinstance(value, Vector):
                bindings.append(Binding(name, Literal(value)))
        if bindings:
            expressions.append(StoreGlobal(tuple(bindings)))
        return Top(tuple(expressions))

    def __repr__(self):
        return f'Top({self.expressions!r})'


cdef class Pragma(Expression):
    cdef readonly str name
    cdef readonly Expression expr

    def __init__(self, str name, Expression expr):
        self.name = name
        self.expr = expr

    cdef void _compile(self, Program program, list lnames):
        self.expr._compile(program, lnames)
        program.pragma(self.name)

    cdef Expression _simplify(self, Context context):
        return Pragma(self.name, self.expr._simplify(context))

    def __repr__(self):
        return f'Pragma({self.name!r}, {self.expr!r})'


cdef class Import(Expression):
    cdef readonly tuple names
    cdef readonly Expression filename

    def __init__(self, tuple names, Expression filename):
        self.names = names
        self.filename = filename

    cdef void _compile(self, Program program, list lnames):
        self.filename._compile(program, lnames)
        program.import_(self.names)
        lnames.extend(self.names)

    cdef Expression _simplify(self, Context context):
        cdef str name
        for name in self.names:
            context.names[name] = None
        return Import(self.names, self.filename._simplify(context))

    def __repr__(self):
        return f'Import({self.names!r}, {self.filename!r})'


cdef class Sequence(Expression):
    cdef readonly tuple expressions

    def __init__(self, tuple expressions):
        self.expressions = expressions

    cdef void _compile(self, Program program, list lnames):
        cdef Expression expr
        cdef int64_t n=len(lnames), m=0
        for expr in self.expressions:
            expr._compile(program, lnames)
            if not isinstance(expr, (Let, Import, Function, Pragma)):
                m += 1
        if len(lnames) > n:
            program.local_drop(len(lnames)-n)
            while len(lnames) > n:
                lnames.pop()
        if m > 1:
            program.compose(m)
        elif m == 0:
            program.literal(null_)

    cdef Expression _simplify(self, Context context):
        cdef list expressions = []
        cdef Expression expr
        cdef dict saved = context.names
        context.names = saved.copy()
        for expr in self.expressions:
            expressions.append(expr._simplify(context))
        context.names = saved
        return sequence_pack(expressions)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.expressions!r})'


cdef class InlineSequence(Sequence):
    cdef void _compile(self, Program program, list lnames):
        cdef Expression expr
        for expr in self.expressions:
            expr._compile(program, lnames)
        program.compose(len(self.expressions))

    cdef Expression _simplify(self, Context context):
        cdef list expressions = []
        cdef Expression expr
        for expr in self.expressions:
            expressions.append(expr._simplify(context))
        return sequence_pack(expressions)


cdef class Literal(Expression):
    cdef readonly Vector value

    def __init__(self, Vector value):
        self.value = value

    cdef void _compile(self, Program program, list lnames):
        program.literal(self.value.intern())

    cdef Expression _simplify(self, Context context):
        return self

    def __repr__(self):
        return f'Literal({self.value!r})'


cdef class Name(Expression):
    cdef readonly str name

    def __init__(self, str name):
        self.name = name

    cdef void _compile(self, Program program, list lnames):
        cdef int64_t i, n=len(lnames)-1
        for i in range(len(lnames)):
            if self.name == <str>lnames[n-i]:
                program.local_load(i)
                break
        else:
            if self.name in dynamic_builtins:
                program.literal(dynamic_builtins[self.name])
            else:
                logger.warning("Name should have been removed by simplifier: {}", self.name)
                program.literal(null_)

    cdef Expression _simplify(self, Context context):
        if self.name in context.names:
            value = context.names[self.name]
            if value is None:
                return self
            elif isinstance(value, Function):
                return FunctionName(self.name)
            elif isinstance(value, Name):
                return (<Name>value)._simplify(context)
            else:
                return Literal(Vector.copy((<Vector>value)))
        elif (value := static_builtins.get(self.name)) is not None:
            return Literal(value)
        elif self.name not in dynamic_builtins:
            if context.captures is not None:
                context.captures.add(self.name)
            else:
                context.errors.add(f"Unbound name '{self.name}'")
                return NoOp
        return self

    def __repr__(self):
        return f'Name({self.name!r})'


cdef class FunctionName(Name):
    def __repr__(self):
        return f'FunctionName({self.name!r})'


cdef class Lookup(Expression):
    cdef readonly Expression key

    def __init__(self, Expression key):
        self.key = key

    cdef void _compile(self, Program program, list lnames):
        self.key._compile(program, lnames)
        program.lookup()

    cdef Expression _simplify(self, Context context):
        cdef Expression key = self.key._simplify(context)
        cdef Vector value
        if isinstance(key, Literal):
            if context.state is not None and context.state.contains((<Literal>key).value):
                value = context.state.get_item((<Literal>key).value)
                return Literal(value)
            return LookupLiteral((<Literal>key).value)
        return Lookup(key)

    def __repr__(self):
        return f'Lookup({self.key!r})'


cdef class LookupLiteral(Expression):
    cdef readonly Vector key

    def __init__(self, Vector key):
        self.key = key

    cdef void _compile(self, Program program, list lnames):
        program.lookup_literal(self.key.intern())

    cdef Expression _simplify(self, Context context):
        cdef Vector value
        if context.state is not None and context.state.contains(self.key):
            value = context.state.get_item(self.key)
            return Literal(value)
        return LookupLiteral(self.key)

    def __repr__(self):
        return f'LookupLiteral({self.key!r})'


cdef class Range(Expression):
    cdef readonly Expression start
    cdef readonly Expression stop
    cdef readonly Expression step

    def __init__(self, Expression start, Expression stop, Expression step):
        self.start = start
        self.stop = stop
        self.step = step

    cdef void _compile(self, Program program, list lnames):
        self.start._compile(program, lnames)
        self.stop._compile(program, lnames)
        self.step._compile(program, lnames)
        program.range()

    cdef Expression _simplify(self, Context context):
        cdef Expression start = self.start._simplify(context)
        cdef Expression stop = self.stop._simplify(context)
        cdef Expression step = self.step._simplify(context)
        cdef Vector result
        if isinstance(start, Literal) and isinstance(stop, Literal) and isinstance(step, Literal):
            result = Vector.__new__(Vector)
            result.fill_range((<Literal>start).value, (<Literal>stop).value, (<Literal>step).value)
            return Literal(result)
        return Range(start, stop, step)

    def __repr__(self):
        return f'Range({self.start!r}, {self.stop!r}, {self.step!r})'


cdef class UnaryOperation(Expression):
    cdef readonly Expression expr

    def __init__(self, Expression expr):
        self.expr = expr

    cdef void _compile(self, Program program, list lnames):
        self.expr._compile(program, lnames)
        self._compile_op(program)

    cdef void _compile_op(self, Program program):
        raise NotImplementedError()

    def __repr__(self):
        return f'{self.__class__.__name__}({self.expr!r})'


cdef class Negative(UnaryOperation):
    cdef void _compile_op(self, Program program):
        program.neg()

    cdef Expression _simplify(self, Context context):
        cdef Expression expr = self.expr._simplify(context)
        if isinstance(expr, Literal):
            return Literal((<Literal>expr).value.neg())
        if isinstance(expr, Negative):
            expr = Positive((<Negative>expr).expr)
            return expr._simplify(context)
        cdef MathsBinaryOperation maths
        if isinstance(expr, (Multiply, Divide)):
            maths = expr
            if isinstance(maths.left, Literal):
                expr = type(expr)(Negative(maths.left), maths.right)
                return expr._simplify(context)
            if isinstance(maths.right, Literal):
                expr = type(expr)(maths.left, Negative(maths.right))
                return expr._simplify(context)
        return Negative(expr)


cdef class Positive(UnaryOperation):
    cdef void _compile_op(self, Program program):
        program.pos()

    cdef Expression _simplify(self, Context context):
        cdef Expression expr = self.expr._simplify(context)
        if isinstance(expr, Literal):
            return Literal((<Literal>expr).value.pos())
        if isinstance(expr, (Negative, Positive, MathsBinaryOperation)):
            return expr._simplify(context)
        return Positive(expr)


cdef class Not(UnaryOperation):
    cdef void _compile_op(self, Program program):
        program.not_()

    cdef Expression _simplify(self, Context context):
        cdef Expression expr = self.expr._simplify(context)
        if isinstance(expr, Literal):
            return Literal(false_ if (<Literal>expr).value.as_bool() else true_)
        return Not(expr)


cdef class Ceil(UnaryOperation):
    cdef void _compile_op(self, Program program):
        program.ceil()

    cdef Expression _simplify(self, Context context):
        cdef Expression expr = self.expr._simplify(context)
        if isinstance(expr, Literal):
            return Literal((<Literal>expr).value.ceil())
        return Ceil(expr)


cdef class Floor(UnaryOperation):
    cdef void _compile_op(self, Program program):
        program.floor()

    cdef Expression _simplify(self, Context context):
        cdef Expression expr = self.expr._simplify(context)
        if isinstance(expr, Literal):
            return Literal((<Literal>expr).value.floor())
        return Floor(expr)


cdef class Fract(UnaryOperation):
    cdef void _compile_op(self, Program program):
        program.fract()

    cdef Expression _simplify(self, Context context):
        cdef Expression expr = self.expr._simplify(context)
        if isinstance(expr, Literal):
            return Literal((<Literal>expr).value.fract())
        return Fract(expr)


cdef class BinaryOperation(Expression):
    cdef readonly Expression left
    cdef readonly Expression right

    def __init__(self, Expression left, Expression right):
        self.left = left
        self.right = right

    cdef void _compile(self, Program program, list lnames):
        self.left._compile(program, lnames)
        self.right._compile(program, lnames)
        self._compile_op(program)

    cdef void _compile_op(self, Program program):
        raise NotImplementedError()

    cdef Expression _simplify(self, Context context):
        cdef Expression left = self.left._simplify(context)
        cdef Expression right = self.right._simplify(context)
        cdef bint literal_left = isinstance(left, Literal)
        cdef bint literal_right = isinstance(right, Literal)
        if literal_left and literal_right:
            return Literal(self.op((<Literal>left).value, (<Literal>right).value))
        elif literal_left:
            if (expr := self.constant_left((<Literal>left).value, right)) is not None:
                return expr._simplify(context)
        elif literal_right:
            if (expr := self.constant_right(left, (<Literal>right).value)) is not None:
                return expr._simplify(context)
        return type(self)(left, right)

    cdef Vector op(self, Vector left, Vector right):
        raise NotImplementedError()

    cdef Expression constant_left(self, Vector left, Expression right):
        return None

    cdef Expression constant_right(self, Expression left, Vector right):
        return None

    def __repr__(self):
        return f'{self.__class__.__name__}({self.left!r}, {self.right!r})'


cdef class MathsBinaryOperation(BinaryOperation):
    cdef Expression _simplify(self, Context context):
        cdef Expression expr=BinaryOperation._simplify(self, context)
        if isinstance(expr, MathsBinaryOperation):
            if isinstance(expr.left, Positive):
                return (<Expression>type(expr)(expr.left.expr, expr.right))._simplify(context)
            elif isinstance(expr.right, Positive):
                return (<Expression>type(expr)(expr.left, expr.right.expr))._simplify(context)
        return expr


cdef class Add(MathsBinaryOperation):
    cdef Vector op(self, Vector left, Vector right):
        return left.add(right)

    cdef void _compile_op(self, Program program):
        program.add()

    cdef Expression constant_left(self, Vector left, Expression right):
        if left.eq(false_):
            return Positive(right)
        if isinstance(right, Negative):
            return Subtract(Literal(left), right.expr)

    cdef Expression constant_right(self, Expression left, Vector right):
        return self.constant_left(right, left)


cdef class Subtract(MathsBinaryOperation):
    cdef Vector op(self, Vector left, Vector right):
        return left.sub(right)

    cdef void _compile_op(self, Program program):
        program.sub()

    cdef Expression constant_left(self, Vector left, Expression right):
        if left.eq(false_):
            return Negative(right)

    cdef Expression constant_right(self, Expression left, Vector right):
        if right.eq(false_):
            return Positive(left)
        return Add(left, Literal(right.neg()))


cdef class Multiply(MathsBinaryOperation):
    cdef Vector op(self, Vector left, Vector right):
        return left.mul(right)

    cdef void _compile_op(self, Program program):
        program.mul()

    cdef Expression constant_left(self, Vector left, Expression right):
        if left.eq(true_):
            return Positive(right)
        if left.eq(minusone_):
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
        elif isinstance(right, Negative):
            return Multiply(Literal(left.neg()), (<Negative>right).expr)

    cdef Expression constant_right(self, Expression left, Vector right):
        return self.constant_left(right, left)


cdef class Divide(MathsBinaryOperation):
    cdef Vector op(self, Vector left, Vector right):
        return left.truediv(right)

    cdef void _compile_op(self, Program program):
        program.truediv()

    cdef Expression constant_right(self, Expression left, Vector right):
        if right.eq(true_):
            return Positive(left)
        return Multiply(left, Literal(true_.truediv(right)))


cdef class FloorDivide(MathsBinaryOperation):
    cdef Vector op(self, Vector left, Vector right):
        return left.floordiv(right)

    cdef void _compile_op(self, Program program):
        program.floordiv()

    cdef Expression constant_right(self, Expression left, Vector right):
        if right.eq(true_):
            return Floor(left)


cdef class Modulo(MathsBinaryOperation):
    cdef Vector op(self, Vector left, Vector right):
        return left.mod(right)

    cdef void _compile_op(self, Program program):
        program.mod()

    cdef Expression constant_right(self, Expression left, Vector right):
        if right.eq(true_):
            return Fract(left)


cdef class Power(MathsBinaryOperation):
    cdef Vector op(self, Vector left, Vector right):
        return left.pow(right)

    cdef void _compile_op(self, Program program):
        program.pow()

    cdef Expression constant_right(self, Expression left, Vector right):
        if right.eq(true_):
            return Positive(left)


cdef class Comparison(BinaryOperation):
    pass


cdef class EqualTo(Comparison):
    cdef Vector op(self, Vector left, Vector right):
        return left.eq(right)

    cdef void _compile_op(self, Program program):
        program.eq()


cdef class NotEqualTo(Comparison):
    cdef Vector op(self, Vector left, Vector right):
        return left.ne(right)

    cdef void _compile_op(self, Program program):
        program.ne()


cdef class LessThan(Comparison):
    cdef Vector op(self, Vector left, Vector right):
        return left.lt(right)

    cdef void _compile_op(self, Program program):
        program.lt()


cdef class GreaterThan(Comparison):
    cdef Vector op(self, Vector left, Vector right):
        return left.gt(right)

    cdef void _compile_op(self, Program program):
        program.gt()


cdef class LessThanOrEqualTo(Comparison):
    cdef Vector op(self, Vector left, Vector right):
        return left.le(right)

    cdef void _compile_op(self, Program program):
        program.le()


cdef class GreaterThanOrEqualTo(Comparison):
    cdef Vector op(self, Vector left, Vector right):
        return left.ge(right)

    cdef void _compile_op(self, Program program):
        program.ge()


cdef class And(BinaryOperation):
    cdef void _compile(self, Program program, list lnames):
        end_label = program.new_label()
        self.left._compile(program, lnames)
        program.dup()
        program.branch_false(end_label)
        program.drop()
        self.right._compile(program, lnames)
        program.label(end_label)

    cdef Expression _simplify(self, Context context):
        cdef Expression left = self.left._simplify(context)
        if isinstance(left, Literal):
            if (<Literal>left).value.as_bool():
                return self.right._simplify(context)
            return left
        return And(left, self.right._simplify(context))


cdef class Or(BinaryOperation):
    cdef void _compile(self, Program program, list lnames):
        end_label = program.new_label()
        self.left._compile(program, lnames)
        program.dup()
        program.branch_true(end_label)
        program.drop()
        self.right._compile(program, lnames)
        program.label(end_label)

    cdef Expression _simplify(self, Context context):
        cdef Expression left = self.left._simplify(context)
        if isinstance(left, Literal):
            if not (<Literal>left).value.as_bool():
                return self.right._simplify(context)
            return left
        return Or(left, self.right._simplify(context))


cdef class Xor(BinaryOperation):
    cdef void _compile_op(self, Program program):
        program.xor()

    cdef Expression _simplify(self, Context context):
        cdef Expression left = self.left._simplify(context)
        cdef Expression right = self.right._simplify(context)
        cdef bint literal_left = isinstance(left, Literal)
        cdef bint literal_right = isinstance(right, Literal)
        if literal_left and not (<Literal>left).value.as_bool():
            return right
        if literal_right and not (<Literal>right).value.as_bool():
            return left
        if literal_left and literal_right:
            return Literal(false_)
        return Xor(left, right)


cdef class Slice(Expression):
    cdef readonly Expression expr
    cdef readonly Expression index

    def __init__(self, Expression expr, Expression index):
        self.expr = expr
        self.index = index

    cdef void _compile(self, Program program, list lnames):
        self.expr._compile(program, lnames)
        self.index._compile(program, lnames)
        program.slice()

    cdef Expression _simplify(self, Context context):
        cdef Expression expr = self.expr._simplify(context)
        cdef Expression index = self.index._simplify(context)
        cdef Vector expr_value
        cdef Vector index_value
        if isinstance(expr, Literal) and isinstance(index, Literal):
            expr_value = (<Literal>expr).value
            index_value = (<Literal>index).value
            return Literal(expr_value.slice(index_value))
        elif isinstance(index, Literal):
            index_value = (<Literal>index).value
            return FastSlice(expr, index_value)
        return Slice(expr, index)

    def __repr__(self):
        return f'Slice({self.expr!r}, {self.index!r})'


cdef class FastSlice(Expression):
    cdef readonly Expression expr
    cdef readonly Vector index

    def __init__(self, Expression expr, Vector index):
        self.expr = expr
        self.index = index

    cdef void _compile(self, Program program, list lnames):
        self.expr._compile(program, lnames)
        program.slice_literal(self.index.intern())

    cdef Expression _simplify(self, Context context):
        cdef Expression expr = self.expr._simplify(context)
        return FastSlice(expr, self.index)

    def __repr__(self):
        return f'FastSlice({self.expr!r}, {self.index!r})'


cdef class Call(Expression):
    cdef readonly Expression function
    cdef readonly tuple args
    cdef readonly tuple keyword_args

    def __init__(self, Expression function, tuple args, tuple keyword_args=None):
        self.function = function
        self.args = args
        self.keyword_args = keyword_args

    cdef void _compile(self, Program program, list lnames):
        cdef Expression expr
        cdef list names = []
        if self.args:
            for expr in self.args:
                expr._compile(program, lnames)
        cdef Binding keyword_arg
        if self.keyword_args:
            for keyword_arg in self.keyword_args:
                names.append(keyword_arg.name)
                keyword_arg.expr._compile(program, lnames)
        if not names and isinstance(self.function, Literal) \
                and (<Literal>self.function).value.length == 1 \
                and (<Literal>self.function).value.objects is not None \
                and not isinstance(function := (<Literal>self.function).value.objects[0], Function):
            program.call_fast(function, len(self.args) if self.args else 0)
        else:
            self.function._compile(program, lnames)
            program.call(len(self.args) if self.args else 0, tuple(names) if names else None)

    cdef Expression _simplify(self, Context context):
        cdef Expression function = self.function._simplify(context)
        cdef bint literal = isinstance(function, FunctionName) or \
            (isinstance(function, Literal) and (<Literal>function).value.objects is not None)
        cdef Expression arg, expr
        cdef list args = []
        if self.args:
            for arg in self.args:
                arg = arg._simplify(context)
                args.append(arg)
                if not isinstance(arg, Literal):
                    literal = False
        cdef list keyword_args = []
        cdef Binding binding
        if self.keyword_args:
            for binding in self.keyword_args:
                arg = binding.expr._simplify(context)
                keyword_args.append(Binding(binding.name, arg))
                if not isinstance(arg, Literal):
                    literal = False
        cdef list bindings
        cdef Function func_expr
        cdef dict kwargs
        cdef int64_t i
        if isinstance(function, FunctionName):
            func_expr = context.names[(<FunctionName>function).name]
            assert not func_expr.captures, "Cannot inline functions with captured names"
            if not func_expr.recursive or literal:
                kwargs = {binding.name: binding.expr for binding in keyword_args}
                bindings = [PolyBinding(((<FunctionName>function).name,), func_expr)]
                for i, binding in enumerate(func_expr.parameters):
                    if i < len(args):
                        bindings.append(PolyBinding((binding.name,), <Expression>args[i]))
                    elif binding.name in kwargs:
                        bindings.append(PolyBinding((binding.name,), <Expression>kwargs[binding.name]))
                    elif binding.expr is not None:
                        bindings.append(PolyBinding((binding.name,), binding.expr))
                    else:
                        bindings.append(PolyBinding((binding.name,), Literal(null_)))
                expr = InlineLet(func_expr.expr, tuple(bindings))._simplify(context)
                return expr
        cdef list vector_args, results
        cdef Literal literal_arg
        if literal:
            vector_args = [literal_arg.value for literal_arg in args]
            kwargs = {binding.name: (<Literal>binding.expr).value for binding in keyword_args}
            results = []
            for func in (<Literal>function).value.objects:
                if callable(func):
                    try:
                        if hasattr(func, 'context_func'):
                            results.append(Literal(func(context, *vector_args, **kwargs)))
                        else:
                            results.append(Literal(func(*vector_args, **kwargs)))
                    except Exception:
                        break
            else:
                return sequence_pack(results)
        if isinstance(function, Literal) and len(args) == 1:
            if (<Literal>function).value == static_builtins['ceil']:
                return Ceil(args[0])
            elif (<Literal>function).value == static_builtins['floor']:
                return Floor(args[0])
            if (<Literal>function).value == static_builtins['fract']:
                return Fract(args[0])
        cdef Call call = Call(function, tuple(args), tuple(keyword_args))
        return call

    def __repr__(self):
        return f'Call({self.function!r}, {self.args!r}, {self.keyword_args!r})'


cdef class NodeModifier(Expression):
    cdef readonly Expression node


cdef class Tag(NodeModifier):
    cdef readonly str tag

    def __init__(self, Expression node, str tag):
        self.node = node
        self.tag = tag

    cdef void _compile(self, Program program, list lnames):
        self.node._compile(program, lnames)
        program.tag(self.tag)

    cdef Expression _simplify(self, Context context):
        cdef Expression node = self.node._simplify(context)
        cdef Vector nodes
        cdef list objects
        cdef Node n
        cdef int64_t i
        if isinstance(node, Literal):
            nodes = (<Literal>node).value
            if nodes.isinstance(Node):
                objects = []
                for i in range(nodes.length):
                    n = nodes.objects[i].copy()
                    n.add_tag(self.tag)
                    objects.append(n)
                return Literal(Vector(objects))
        return Tag(node, self.tag)

    def __repr__(self):
        return f'Tag({self.node!r}, {self.tag!r})'


cdef class Attributes(NodeModifier):
    cdef readonly tuple bindings

    def __init__(self, Expression node, tuple bindings):
        self.node = node
        self.bindings = bindings

    cdef void _compile(self, Program program, list lnames):
        self.node._compile(program, lnames)
        cdef Binding binding
        for binding in self.bindings:
            binding.expr._compile(program, lnames)
            program.attribute(binding.name)

    cdef Expression _simplify(self, Context context):
        cdef Expression node = self
        cdef list bindings = []
        cdef Attributes attrs
        cdef Binding binding
        while isinstance(node, Attributes):
            attrs = <Attributes>node
            for binding in reversed(attrs.bindings):
                bindings.append(Binding(binding.name, binding.expr._simplify(context)))
            node = attrs.node
        node = node._simplify(context)
        cdef Vector nodes
        cdef int64_t i
        cdef list objects
        if isinstance(node, Literal):
            nodes = (<Literal>node).value
            if nodes.isinstance(Node):
                objects = []
                for i in range(nodes.length):
                    objects.append(nodes.objects[i].copy())
                while bindings and isinstance((<Binding>bindings[-1]).expr, Literal):
                    binding = bindings.pop()
                    for i in range(nodes.length):
                        (<Node>objects[i]).set_attribute(binding.name, (<Literal>binding.expr).value)
                node = Literal(Vector(objects))
        if not bindings:
            return node
        bindings.reverse()
        return Attributes(node, tuple(bindings))

    def __repr__(self):
        return f'{self.__class__.__name__}({self.node!r}, {self.bindings!r})'


cdef class Append(NodeModifier):
    cdef readonly Expression children

    def __init__(self, Expression node, Expression children):
        self.node = node
        self.children = children

    cdef void _compile(self, Program program, list lnames):
        self.node._compile(program, lnames)
        self.children._compile(program, lnames)
        program.append()

    cdef Expression _simplify(self, Context context):
        cdef Expression node = self.node._simplify(context)
        cdef Expression children = self.children._simplify(context)
        cdef Vector nodes, childs
        cdef Node c, n
        cdef int64_t i
        cdef list modifiers
        cdef Expression expr
        cdef NodeModifier modifier
        cdef list objects
        if isinstance(children, Literal):
            if isinstance(node, Literal):
                nodes = (<Literal>node).value
                childs = (<Literal>children).value
                if nodes.isinstance(Node) and childs.isinstance(Node):
                    objects = []
                    for i in range(nodes.length):
                        n = <Node>nodes.objects[i].copy()
                        for c in childs.objects:
                            n.append(c)
                        objects.append(n)
                    return Literal(Vector(objects))
            else:
                modifiers = []
                expr = node
                while isinstance(expr, NodeModifier):
                    modifiers.append(expr)
                    expr = (<NodeModifier>expr).node
                if isinstance(expr, Literal):
                    expr = Append(expr, children)._simplify(context)
                    while modifiers:
                        modifier = modifiers.pop()
                        if isinstance(modifier, Attributes):
                            expr = Attributes(expr, (<Attributes>modifier).bindings)
                        elif isinstance(modifier, Tag):
                            expr = Tag(expr, (<Tag>modifier).tag)
                        elif isinstance(modifier, Append):
                            expr = Append(expr, (<Append>modifier).children)
                    return expr
        elif isinstance(children, Sequence) and isinstance((<Sequence>children).expressions[0], Literal):
            node = Append(node, (<Sequence>children).expressions[0])
            children = Sequence((<Sequence>children).expressions[1:])
            return Append(node, children)._simplify(context)
        return Append(node, children)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.node!r}, {self.children!r})'


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

    cdef void _compile(self, Program program, list lnames):
        cdef PolyBinding binding
        for binding in self.bindings:
            binding.expr._compile(program, lnames)
            program.local_push(len(binding.names))
            lnames.extend(binding.names)

    cdef Expression _simplify(self, Context context):
        cdef list remaining = []
        cdef PolyBinding binding
        cdef Expression expr
        cdef Vector value
        cdef str name
        cdef int64_t i, n
        for binding in self.bindings:
            expr = binding.expr._simplify(context)
            if isinstance(expr, Literal):
                value = (<Literal>expr).value
                n = len(binding.names)
                if n == 1:
                    name = binding.names[0]
                    context.names[name] = value
                else:
                    for i, name in enumerate(binding.names):
                        context.names[name] = value.item(i)
            elif isinstance(expr, Name) and len(binding.names) == 1:
                name = binding.names[0]
                if (<Name>expr).name != name:
                    context.names[name] = expr
            else:
                for name in binding.names:
                    context.names[name] = None
                remaining.append(PolyBinding(binding.names, expr))
        if remaining:
            return Let(tuple(remaining))
        return NoOp

    def __repr__(self):
        return f'{self.__class__.__name__}({self.bindings!r})'


cdef class StoreGlobal(Expression):
    cdef readonly tuple bindings

    def __init__(self, tuple bindings):
        self.bindings = bindings

    cdef void _compile(self, Program program, list lnames):
        cdef Binding binding
        for binding in self.bindings:
            binding.expr._compile(program, lnames)
            program.store_global(binding.name)

    cdef Expression _simplify(self, Context context):
        return self

    def __repr__(self):
        return f'StoreGlobal({self.bindings!r})'


cdef class InlineLet(Expression):
    cdef readonly Expression body
    cdef readonly tuple bindings

    def __init__(self, Expression body, tuple bindings):
        self.body = body
        self.bindings = bindings

    cdef void _compile(self, Program program, list lnames):
        cdef PolyBinding binding
        cdef int64_t n=len(lnames)
        for binding in self.bindings:
            binding.expr._compile(program, lnames)
            program.local_push(len(binding.names))
            lnames.extend(binding.names)
        self.body._compile(program, lnames)
        program.local_drop(len(self.bindings))
        while len(lnames) > n:
            lnames.pop()

    cdef Expression _simplify(self, Context context):
        cdef dict saved = context.names
        context.names = saved.copy()
        cdef list remaining = []
        cdef PolyBinding binding
        cdef Expression expr
        cdef Vector value
        cdef str name
        cdef int64_t i, n
        for binding in self.bindings:
            expr = binding.expr._simplify(context)
            if isinstance(expr, Literal):
                value = (<Literal>expr).value
                n = len(binding.names)
                if n == 1:
                    name = binding.names[0]
                    context.names[name] = value
                else:
                    for i, name in enumerate(binding.names):
                        context.names[name] = value.item(i)
            elif isinstance(expr, Function) and len(binding.names) == 1:
                name = binding.names[0]
                context.names[name] = expr
            elif isinstance(expr, Name) and len(binding.names) == 1:
                name = binding.names[0]
                if (<Name>expr).name != name:
                    context.names[name] = expr
            else:
                for name in binding.names:
                    context.names[name] = None
                remaining.append(PolyBinding(binding.names, expr))
        body = self.body._simplify(context)
        context.names = saved
        if remaining:
            return InlineLet(body, tuple(remaining))
        return body

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

    cdef void _compile(self, Program program, list lnames):
        self.source._compile(program, lnames)
        program.begin_for()
        START = program.new_label()
        END = program.new_label()
        cdef int64_t i, n=len(self.names)
        lnames.extend(self.names)
        program.literal(null_)
        program.local_push(n)
        program.label(START)
        program.next(n, END)
        self.body._compile(program, lnames)
        program.jump(START)
        program.label(END)
        program.local_drop(n)
        program.end_for_compose()
        for i in range(n):
            lnames.pop()

    cdef Expression _simplify(self, Context context):
        cdef Expression body, source=self.source._simplify(context)
        cdef list remaining = []
        cdef Vector values
        cdef dict saved = context.names
        context.names = saved.copy()
        cdef str name
        if not isinstance(source, Literal):
            for name in self.names:
                context.names[name] = None
            body = self.body._simplify(context)
            context.names = saved
            return For(self.names, source, body)
        values = (<Literal>source).value
        cdef int64_t i=0, n=values.length
        while i < n:
            for name in self.names:
                context.names[name] = values.item(i)
                i += 1
            remaining.append(self.body._simplify(context))
        context.names = saved
        return sequence_pack(remaining)

    def __repr__(self):
        return f'For({self.names!r}, {self.source!r}, {self.body!r})'


cdef class IfCondition:
    cdef readonly Expression condition
    cdef readonly Expression then

    def __init__(self, Expression condition, Expression then):
        self.condition = condition
        self.then = then

    def __repr__(self):
        return f'IfCondition({self.condition!r}, {self.then!r})'


cdef class IfElse(Expression):
    cdef readonly tuple tests
    cdef readonly Expression else_

    def __init__(self, tuple tests, Expression else_):
        self.tests = tests
        self.else_ = else_

    cdef void _compile(self, Program program, list lnames):
        cdef IfCondition test
        END = program.new_label()
        for test in self.tests:
            NEXT = program.new_label()
            test.condition._compile(program, lnames)
            program.branch_false(NEXT)
            test.then._compile(program, lnames)
            program.jump(END)
            program.label(NEXT)
        if self.else_ is not None:
            self.else_._compile(program, lnames)
        else:
            program.literal(null_)
        program.label(END)

    cdef Expression _simplify(self, Context context):
        cdef list remaining = []
        cdef IfCondition test
        cdef Expression condition, then
        for test in self.tests:
            condition = test.condition._simplify(context)
            if isinstance(condition, Literal):
                if (<Literal>condition).value.as_bool():
                    then = test.then._simplify(context)
                    if not remaining:
                        return then
                    else:
                        return IfElse(tuple(remaining), then)
            else:
                then = test.then._simplify(context)
                remaining.append(IfCondition(condition, then))
        else_ = self.else_._simplify(context) if self.else_ is not None else None
        if remaining:
            return IfElse(tuple(remaining), else_)
        return NoOp if else_ is None else else_

    def __repr__(self):
        return f'IfElse({self.tests!r}, {self.else_!r})'


cdef class Function(Expression):
    cdef readonly str name
    cdef readonly tuple parameters
    cdef readonly Expression expr
    cdef readonly tuple captures
    cdef readonly bint recursive

    def __init__(self, str name, tuple parameters, Expression expr, tuple captures=None, bint recursive=False):
        self.name = name
        self.parameters = parameters
        self.expr = expr
        self.captures = captures
        self.recursive = recursive

    cdef void _compile(self, Program program, list lnames):
        cdef str name
        cdef list function_lnames = []
        assert self.captures is not None, "Simplifier must be run to correctly compile functions"
        cdef int64_t i, n=len(lnames)-1
        for name in self.captures:
            for i in range(len(lnames)):
                if name == <str>lnames[n-i]:
                    program.local_load(i)
                    break
            else:
                logger.warning("Name should have been removed by simplifier: {}", name)
                program.literal(null_)
            function_lnames.append(name)
        function_lnames.append(self.name)
        cdef list parameters = []
        cdef Binding parameter
        for parameter in self.parameters:
            parameters.append(parameter.name)
            if parameter.expr is None:
                program.literal(null_)
            else:
                parameter.expr._compile(program, lnames)
            function_lnames.append(parameter.name)
        START = program.new_label()
        END = program.new_label()
        program.jump(END)
        program.label(START)
        self.expr._compile(program, function_lnames)
        program.exit()
        program.label(END)
        program.func(START, self.name, tuple(parameters), len(self.captures))
        program.local_push(1)
        lnames.append(self.name)

    cdef Expression _simplify(self, Context context):
        cdef list parameters = []
        cdef Binding parameter
        cdef Expression expr
        cdef bint literal = True
        for parameter in self.parameters:
            expr = parameter.expr._simplify(context) if parameter.expr is not None else None
            if expr is not None and not isinstance(expr, Literal):
                literal = False
            parameters.append(Binding(parameter.name, expr))
        cdef dict saved = context.names
        cdef str key
        cdef set captures = context.captures
        context.captures = set()
        context.names = {}
        for key, value in saved.items():
            if value is not None and key != self.name:
                context.names[key] = value
        for parameter in parameters:
            context.names[parameter.name] = None
        expr = self.expr._simplify(context)
        cdef recursive = self.name in context.captures
        if recursive:
            context.captures.discard(self.name)
        cdef Function function = Function(self.name, tuple(parameters), expr, tuple(context.captures), recursive)
        context.names = saved
        context.names[self.name] = function if literal and not context.captures else None
        if captures is not None:
            context.captures.difference_update(saved)
            captures.update(context.captures)
        context.captures = captures
        return function

    def __repr__(self):
        return f'Function({self.name!r}, {self.parameters!r}, {self.expr!r}, {self.captures!r}, {self.recursive!r})'
