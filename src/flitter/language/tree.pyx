
"""
Language abstract syntax tree

The tree supports reasonably sophisticated simplification that will reduce
the tree down to a "simpler" form by propogating constants. As this can result
in unrolling loops, simpler does not necessarily mean "smaller."
"""

from loguru import logger

from libc.stdint cimport int64_t

from .. import name_patch
from ..cache import SharedCache
from ..model cimport Vector, Node, Context, StateDict, null_, true_, false_, minusone_
from .vm cimport Program, Instruction, InstructionInt, InstructionVector, OpCode, static_builtins, dynamic_builtins


logger = name_patch(logger, __name__)

cdef Literal NoOp = Literal(null_)


cdef bint sequence_pack(list expressions):
    cdef Expression expr
    cdef bint touched = False
    cdef list vectors, todo=[]
    while expressions:
        todo.append(expressions.pop())
    while todo:
        expr = <Expression>todo.pop()
        if todo and isinstance(expr, Literal):
            vectors = [(<Literal>expr).value]
            while todo and isinstance(todo[len(todo)-1], Literal):
                vectors.append((<Literal>todo.pop()).value)
            if len(vectors) > 1:
                expr = Literal(Vector._compose(vectors))
                touched = True
        elif isinstance(expr, Sequence):
            for expr in reversed((<Sequence>expr).expressions):
                todo.append(expr)
            touched = True
            continue
        expressions.append(expr)
    return touched


cdef class Expression:
    def compile(self, tuple initial_lnames=(), set initial_errors=None, bint log_errors=True):
        cdef Program program = Program.__new__(Program, initial_lnames)
        if initial_errors:
            program.compiler_errors.update(initial_errors)
        self._compile(program, list(initial_lnames))
        program.link()
        if log_errors:
            for error in program.compiler_errors:
                logger.warning("Compiler error: {}", error)
        return program

    def simplify(self, StateDict state=None, dict static=None, dynamic=None, Context parent=None, path=None, bint return_context=False):
        cdef dict context_vars = {}
        cdef str key
        if static is not None:
            for key, value in static.items():
                context_vars[key] = value if isinstance(value, Expression) else Vector._coerce(value)
        if dynamic is not None:
            for key in dynamic:
                context_vars[key] = None
        cdef Context context = Context(state=state, names=context_vars)
        context.path = path
        context.parent = parent
        cdef Expression expr = self
        try:
            expr = expr._simplify(context)
        except Exception as exc:                                             # pragma: no cover
            logger.opt(exception=exc).warning("Unable to simplify program")  # pragma: no cover
        if return_context:
            return expr, context
        else:
            for error in context.errors:
                logger.warning("Simplifier error: {}", error)
        return expr

    cdef void _compile(self, Program program, list lnames):
        raise NotImplementedError()

    cdef Expression _simplify(self, Context context):
        raise NotImplementedError()


cdef class Top(Expression):
    cdef readonly tuple pragmas
    cdef readonly Expression body
    cdef readonly set dependencies

    def __init__(self, tuple pragmas, Expression body, set dependencies=None):
        self.pragmas = pragmas
        self.body = body
        self.dependencies = dependencies if dependencies is not None else set()

    cdef void _compile(self, Program program, list lnames):
        cdef Binding binding
        for binding in self.pragmas:
            program.set_pragma(binding.name, (<Literal>binding.expr).value)
        self.body._compile(program, lnames)
        cdef Instruction instr = program.last_instruction()
        if instr.code == OpCode.Compose:
            instr = program.pop_instruction()
            program.append((<InstructionInt>instr).value)
        elif instr.code == OpCode.Literal and (<InstructionVector>instr).value.length == 0:
            program.pop_instruction()
        else:
            program.append()

    cdef Expression _simplify(self, Context context):
        cdef Expression body = self.body._simplify(context)
        if body is self.body and context.dependencies == self.dependencies:
            return self
        return Top(self.pragmas, body, self.dependencies ^ context.dependencies)

    def __repr__(self):
        return f'Top({self.pragmas!r}, {self.body!r}, {self.dependencies!r})'


cdef class Export(Expression):
    cdef readonly dict static_exports

    def __init__(self, dict static_exports=None):
        self.static_exports = static_exports

    cdef void _compile(self, Program program, list lnames):
        cdef str name
        cdef Vector value
        if self.static_exports:
            for name, value in self.static_exports.items():
                program.literal(value)
                program.export(name)
        m = len(program.initial_lnames)
        cdef int64_t i, n=len(lnames)
        if n > m:
            for i in range(n-m):
                name = lnames[n-1-i]
                program.local_load(i)
                program.export(name)
        program.literal(null_)

    cdef Expression _simplify(self, Context context):
        cdef str name
        cdef dict static_exports = dict(self.static_exports) if self.static_exports else {}
        cdef bint touched = False
        for name, value in context.names.items():
            if value is not None:
                context.exports[name] = value
            if isinstance(value, Vector) and (name not in static_exports or value != static_exports[name]):
                static_exports[name] = value
                touched = True
        if touched:
            return Export(static_exports)
        return self

    def __repr__(self):
        return f'Export({self.static_exports!r})'


cdef class Import(Expression):
    cdef readonly tuple names
    cdef readonly Expression filename
    cdef readonly Expression expr

    def __init__(self, tuple names, Expression filename, Expression expr):
        self.names = names
        self.filename = filename
        self.expr = expr

    cdef void _compile(self, Program program, list lnames):
        self.filename._compile(program, lnames)
        program.import_(self.names)
        self.expr._compile(program, lnames + list(self.names))
        cdef Instruction instr = program.last_instruction()
        if instr.code == OpCode.LocalDrop:
            program.pop_instruction()
            program.local_drop((<InstructionInt>instr).value + len(self.names))
        else:
            program.local_drop(len(self.names))

    cdef Expression _simplify(self, Context context):
        cdef str name
        cdef Expression filename = self.filename._simplify(context)
        cdef Top top
        cdef Context import_context
        cdef dict import_static_names = None
        if isinstance(filename, Literal) and context.path is not None:
            name = (<Literal>filename).value.as_string()
            path = SharedCache.get_with_root(name, context.path)
            import_context = context.parent
            while import_context is not None:
                if import_context.path is path:
                    context.errors.add(f"Circular import of '{name}'")
                    import_static_names = {name: null_ for name in self.names}
                    break
                import_context = import_context.parent
            else:
                if (top := path.read_flitter_top()) is not None:
                    top, import_context = top.simplify(path=path, parent=context, return_context=True)
                    import_static_names = import_context.exports
                    context.errors.update(import_context.errors)
                    context.dependencies.update(import_context.dependencies)
                    context.dependencies.add(path)
        cdef dict let_names = {}
        cdef dict saved = dict(context.names)
        cdef list remaining = []
        for name in self.names:
            if import_static_names is not None and name in import_static_names:
                let_names[name] = import_static_names[name]
            else:
                context.names[name] = None
                remaining.append(name)
        cdef Expression expr = self.expr
        if let_names:
            expr = Let(tuple(PolyBinding((name,), value if isinstance(value, Function) else Literal(value)) for name, value in let_names.items()), expr)
        expr = expr._simplify(context)
        context.names = saved
        if not remaining:
            return expr
        if filename is self.filename and expr is self.expr:
            return self
        return Import(tuple(remaining), filename, expr)

    def __repr__(self):
        return f'Import({self.names!r}, {self.filename!r}, {self.expr!r})'


cdef class Sequence(Expression):
    cdef readonly tuple expressions

    def __init__(self, tuple expressions):
        self.expressions = expressions

    cdef void _compile(self, Program program, list lnames):
        cdef Expression expr
        cdef InstructionInt instr = None
        cdef int64_t n=len(self.expressions)
        if n:
            for expr in self.expressions:
                expr._compile(program, lnames)
                if program.last_instruction().code == OpCode.Compose:
                    instr = program.pop_instruction()
                    n += instr.value - 1
            if n > 1:
                program.compose(n)
        else:
            program.literal(null_)

    cdef Expression _simplify(self, Context context):
        cdef list expressions = []
        cdef Expression expr, sexpr
        cdef bint touched = False
        for expr in self.expressions:
            sexpr = expr._simplify(context)
            expressions.append(sexpr)
            touched |= sexpr is not expr
        touched |= sequence_pack(expressions)
        if not expressions:
            return NoOp
        if len(expressions) == 1:
            return expressions[0]
        if not touched:
            return self
        return Sequence(tuple(expressions))

    def __repr__(self):
        return f'Sequence({self.expressions!r})'


cdef class Literal(Expression):
    cdef readonly Vector value

    def __init__(self, value):
        self.value = Vector._coerce(value)

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
            if self.name in static_builtins:
                program.literal(static_builtins[self.name])
            elif self.name in dynamic_builtins:
                program.literal(dynamic_builtins[self.name])
            else:
                program.compiler_errors.add(f"Unbound name '{self.name}'")
                program.literal(null_)

    cdef Expression _simplify(self, Context context):
        if self.name in context.names:
            value = context.names[self.name]
            if value is None or isinstance(value, Function):
                return self
            elif isinstance(value, Name):
                return (<Name>value)._simplify(context)
            elif type(value) is Vector:
                return Literal((<Vector>value).copy())
            else:
                return Literal(value)
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


cdef class Lookup(Expression):
    cdef readonly Expression key

    def __init__(self, Expression key):
        self.key = key

    cdef void _compile(self, Program program, list lnames):
        if isinstance(self.key, Literal):
            program.lookup_literal((<Literal>self.key).value.intern())
        else:
            self.key._compile(program, lnames)
            program.lookup()

    cdef Expression _simplify(self, Context context):
        cdef Expression key = self.key._simplify(context)
        cdef Vector value
        if isinstance(key, Literal):
            if context.state is not None and context.state.contains((<Literal>key).value):
                value = context.state.get_item((<Literal>key).value)
                return Literal(value)
        if key is self.key:
            return self
        return Lookup(key)

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
        if start is self.start and stop is self.stop and step is self.step:
            return self
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
        elif isinstance(expr, Add):
            maths = expr
            if isinstance(maths.left, Literal) or isinstance(maths.right, Literal):
                expr = Add(Negative(maths.left), Negative(maths.right))
                return expr._simplify(context)
        elif isinstance(expr, Subtract):
            maths = expr
            if isinstance(maths.left, Literal) or isinstance(maths.right, Literal):
                expr = Add(Negative(maths.left), maths.right)
                return expr._simplify(context)
        if expr is self.expr:
            return self
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
        if expr is self.expr:
            return self
        return Positive(expr)


cdef class Not(UnaryOperation):
    cdef void _compile_op(self, Program program):
        program.not_()

    cdef Expression _simplify(self, Context context):
        cdef Expression expr = self.expr._simplify(context)
        if isinstance(expr, Literal):
            return Literal(false_ if (<Literal>expr).value.as_bool() else true_)
        if expr is self.expr:
            return self
        return Not(expr)


cdef class Ceil(UnaryOperation):
    cdef void _compile_op(self, Program program):
        program.ceil()

    cdef Expression _simplify(self, Context context):
        cdef Expression expr = self.expr._simplify(context)
        if isinstance(expr, Literal):
            return Literal((<Literal>expr).value.ceil())
        if expr is self.expr:
            return self
        return Ceil(expr)


cdef class Floor(UnaryOperation):
    cdef void _compile_op(self, Program program):
        program.floor()

    cdef Expression _simplify(self, Context context):
        cdef Expression expr = self.expr._simplify(context)
        if isinstance(expr, Literal):
            return Literal((<Literal>expr).value.floor())
        if expr is self.expr:
            return self
        return Floor(expr)


cdef class Fract(UnaryOperation):
    cdef void _compile_op(self, Program program):
        program.fract()

    cdef Expression _simplify(self, Context context):
        cdef Expression expr = self.expr._simplify(context)
        if isinstance(expr, Literal):
            return Literal((<Literal>expr).value.fract())
        if expr is self.expr:
            return self
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
        cdef Expression expr
        if literal_left and literal_right:
            return Literal(self.op((<Literal>left).value, (<Literal>right).value))
        elif literal_left:
            if (expr := self.constant_left((<Literal>left).value, right)) is not None:
                return expr._simplify(context)
        elif literal_right:
            if (expr := self.constant_right(left, (<Literal>right).value)) is not None:
                return expr._simplify(context)
        if (expr := self.additional_rules(left, right)) is not None:
            return expr._simplify(context)
        if left is self.left and right is self.right:
            return self
        return type(self)(left, right)

    cdef Vector op(self, Vector left, Vector right):
        raise NotImplementedError()

    cdef Expression constant_left(self, Vector left, Expression right):
        return None

    cdef Expression constant_right(self, Expression left, Vector right):
        return None

    cdef Expression additional_rules(self, Expression left, Expression right):
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
        if program.last_instruction().code == OpCode.Mul:
            program.pop_instruction()
            program.mul_add()
        else:
            program.add()

    cdef Expression constant_left(self, Vector left, Expression right):
        if left.eq(false_):
            return Positive(right)

    cdef Expression constant_right(self, Expression left, Vector right):
        return self.constant_left(right, left)

    cdef Expression additional_rules(self, Expression left, Expression right):
        if not isinstance(right, Multiply) and isinstance(left, Multiply):
            return Add(right, left)
        if isinstance(right, Negative):
            return Subtract(left, (<Negative>right).expr)
        if isinstance(left, Negative):
            return Subtract(right, (<Negative>left).expr)


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

    cdef Expression additional_rules(self, Expression left, Expression right):
        if isinstance(right, Negative):
            return Add(left, (<Negative>right).expr)


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
                return Multiply(Multiply(Literal(left), maths.right), maths.left)
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
        return Multiply(Literal(true_.truediv(right)), left)


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
        cdef Instruction instr = program.last_instruction()
        if instr.code == OpCode.Literal and (<InstructionVector>instr).value.eq(Vector(2)):
            program.pop_instruction()
            program.dup()
            program.mul()
        else:
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
        cdef Expression right = self.right._simplify(context)
        if left is self.left and right is self.right:
            return self
        return And(left, right)


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
        cdef Expression right = self.right._simplify(context)
        if left is self.left and right is self.right:
            return self
        return Or(left, right)


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
        if left is self.left and right is self.right:
            return self
        return Xor(left, right)


cdef class Slice(Expression):
    cdef readonly Expression expr
    cdef readonly Expression index

    def __init__(self, Expression expr, Expression index):
        self.expr = expr
        self.index = index

    cdef void _compile(self, Program program, list lnames):
        self.expr._compile(program, lnames)
        if isinstance(self.index, Literal):
            program.slice_literal((<Literal>self.index).value)
        else:
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
        if expr is self.expr and index is self.index:
            return self
        return Slice(expr, index)

    def __repr__(self):
        return f'Slice({self.expr!r}, {self.index!r})'


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
        cdef bint touched = function is not self.function
        cdef Function func_expr = None
        if isinstance(function, Function):
            func_expr = <Function>function
        elif isinstance(function, Name) and (<Name>function).name in context.names:
            value = context.names[(<Name>function).name]
            if isinstance(value, Function):
                func_expr = <Function>value
        cdef bint literal_func = isinstance(function, Literal)
        if literal_func and not (<Literal>function).value.objects:
            return NoOp
        cdef bint literal_args = True
        cdef Expression arg, sarg, expr
        cdef list args = []
        if self.args:
            for arg in self.args:
                sarg = arg._simplify(context)
                touched |= sarg is not arg
                args.append(sarg)
                if not isinstance(sarg, Literal):
                    literal_args = False
        cdef list keyword_args = []
        cdef Binding binding
        if self.keyword_args:
            for binding in self.keyword_args:
                arg = binding.expr._simplify(context)
                touched |= arg is not binding.expr
                keyword_args.append(Binding(binding.name, arg))
                if not isinstance(arg, Literal):
                    literal_args = False
        cdef list bindings
        cdef dict kwargs
        cdef int64_t i
        if func_expr is not None and not func_expr.captures and (not func_expr.recursive or literal_args):
            kwargs = {binding.name: binding.expr for binding in keyword_args}
            bindings = []
            for i, binding in enumerate(func_expr.parameters):
                if i < len(args):
                    bindings.append(PolyBinding((binding.name,), <Expression>args[i]))
                elif binding.name in kwargs:
                    bindings.append(PolyBinding((binding.name,), <Expression>kwargs[binding.name]))
                elif binding.expr is not None:
                    bindings.append(PolyBinding((binding.name,), binding.expr))
                else:
                    bindings.append(PolyBinding((binding.name,), Literal(null_)))
            expr = Let(tuple(bindings), func_expr.body)._simplify(context)
            return expr
        cdef list vector_args, results
        cdef Literal literal_arg
        if literal_func and literal_args:
            vector_args = [literal_arg.value for literal_arg in args]
            kwargs = {binding.name: (<Literal>binding.expr).value for binding in keyword_args}
            results = []
            for func in (<Literal>function).value.objects:
                if callable(func):
                    try:
                        assert not hasattr(func, 'context_func')
                        results.append(func(*vector_args, **kwargs))
                    except Exception as exc:
                        context.errors.add(f"Error calling {func.__name__}: {str(exc)}")
            return Literal(Vector._compose(results))
        if isinstance(function, Literal) and len(args) == 1:
            if (<Literal>function).value == static_builtins['ceil']:
                return Ceil(args[0])
            elif (<Literal>function).value == static_builtins['floor']:
                return Floor(args[0])
            if (<Literal>function).value == static_builtins['fract']:
                return Fract(args[0])
        if not touched:
            return self
        return Call(function, tuple(args), tuple(keyword_args) if keyword_args else None)

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
            if nodes.objects is None:
                return node
            objects = []
            for i in range(nodes.length):
                obj = nodes.objects[i]
                if isinstance(obj, Node):
                    n = (<Node>obj).copy()
                    n.add_tag(self.tag)
                    objects.append(n)
                else:
                    objects.append(obj)
            return Literal(objects)
        if node is self.node:
            return self
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
        cdef list names = []
        for binding in self.bindings:
            binding.expr._compile(program, lnames)
            names.append(binding.name)
        program.attributes(tuple(names))

    cdef Expression _simplify(self, Context context):
        cdef Expression node = self
        cdef list bindings = []
        cdef Attributes attrs
        cdef Binding binding
        cdef Expression expr
        cdef bint touched = False
        while isinstance(node, Attributes):
            attrs = <Attributes>node
            for binding in reversed(attrs.bindings):
                expr = binding.expr._simplify(context)
                bindings.append(Binding(binding.name, expr))
                touched |= expr is not binding.expr
            node = attrs.node
        node = node._simplify(context)
        touched |= node is not self.node
        cdef Vector nodes, value
        cdef list objects
        if isinstance(node, Literal):
            nodes = (<Literal>node).value
            if nodes.objects is None:
                return node
            if bindings and isinstance((<Binding>bindings[-1]).expr, Literal):
                objects = []
                for obj in nodes.objects:
                    objects.append((<Node>obj).copy() if isinstance(obj, Node) else obj)
                while bindings and isinstance((<Binding>bindings[-1]).expr, Literal):
                    binding = bindings.pop()
                    value = (<Literal>binding.expr).value
                    for obj in objects:
                        if isinstance(obj, Node):
                            (<Node>obj).set_attribute(binding.name, value)
                node = Literal(objects)
                touched = True
        if not touched:
            return self
        if bindings:
            bindings.reverse()
            node = Attributes(node, tuple(bindings))
        return node

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
        cdef Instruction instr = program.last_instruction()
        if instr.code == OpCode.Compose:
            instr = program.pop_instruction()
            program.append((<InstructionInt>instr).value)
        elif instr.code == OpCode.Literal and (<InstructionVector>instr).value.length == 0:
            program.pop_instruction()
        else:
            program.append()

    cdef Expression _simplify(self, Context context):
        cdef Expression node = self.node._simplify(context)
        cdef Expression children = self.children._simplify(context)
        cdef Vector nodes, children_vector
        cdef int64_t i
        cdef list objects
        cdef tuple node_objects
        if (isinstance(node, Literal) and (<Literal>node).value.objects is None) or \
                (isinstance(children, Literal) and (<Literal>children).value.objects is None):
            return node
        if isinstance(node, Literal):
            nodes = (<Literal>node).value
            if isinstance(children, Literal):
                children_vector = (<Literal>children).value
                node_objects = nodes.objects
                objects = []
                for i in range(nodes.length):
                    obj = node_objects[i]
                    if isinstance(obj, Node):
                        obj = (<Node>obj).copy()
                        (<Node>obj).append_vector(children_vector)
                    objects.append(obj)
                return Literal(objects)
            elif isinstance(children, Sequence) and isinstance((<Sequence>children).expressions[0], Literal):
                node = Append(node, (<Sequence>children).expressions[0])
                children = Sequence((<Sequence>children).expressions[1:])
                return Append(node, children)._simplify(context)
        elif isinstance(node, Attributes) and isinstance((<Attributes>node).node, Literal):
            return Attributes(Append((<Attributes>node).node, children), (<Attributes>node).bindings)._simplify(context)
        if node is self.node and children is self.children:
            return self
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
    cdef readonly Expression body

    def __init__(self, tuple bindings, Expression body):
        self.bindings = bindings
        self.body = body

    cdef void _compile(self, Program program, list lnames):
        cdef PolyBinding binding
        cdef int64_t n=len(lnames)
        for binding in self.bindings:
            binding.expr._compile(program, lnames)
            program.local_push(len(binding.names))
            lnames.extend(binding.names)
        self.body._compile(program, lnames)
        cdef Instruction instr, compose=None
        if program.last_instruction().code == OpCode.Compose:
            compose = program.pop_instruction()
        instr = program.last_instruction()
        if instr.code == OpCode.LocalDrop:
            program.pop_instruction()
            program.local_drop((<InstructionInt>instr).value + len(lnames) - n)
        else:
            program.local_drop(len(lnames) - n)
        if compose is not None:
            program.push_instruction(compose)
        while len(lnames) > n:
            lnames.pop()

    cdef Expression _simplify(self, Context context):
        cdef dict saved = context.names
        context.names = saved.copy()
        cdef list bindings = list(self.bindings)
        cdef list remaining = []
        cdef PolyBinding binding
        cdef Expression expr, body=self.body
        cdef Vector value
        cdef str name, existing_name
        cdef int64_t i, j, n
        cdef touched = False
        cdef set shadowed=set(), discarded=set()
        while isinstance(body, Let):
            bindings.extend((<Let>body).bindings)
            body = (<Let>body).body
            touched = True
        for i, binding in enumerate(bindings):
            for existing_name in list(context.names):
                existing_value = context.names[existing_name]
                if existing_value is not None and isinstance(existing_value, Name):
                    for name in binding.names:
                        if name == (<Name>existing_value).name:
                            context.names[existing_name] = None
                            remaining.append(PolyBinding((existing_name,), Name(name)))
                            shadowed.add(name)
                            touched = True
            n = len(binding.names)
            expr = binding.expr._simplify(context)
            touched |= expr is not binding.expr
            if n == 1 and isinstance(expr, Function) and (<Function>expr).inlineable:
                context.names[binding.names[0]] = expr
                remaining.append(PolyBinding(binding.names, expr))
                continue
            if isinstance(expr, Literal):
                value = (<Literal>expr).value
                if n == 1:
                    name = binding.names[0]
                    context.names[name] = value
                    discarded.add(name)
                else:
                    for j, name in enumerate(binding.names):
                        context.names[name] = value.item(j)
                        discarded.add(name)
                touched = True
                continue
            if n == 1 and isinstance(expr, Name):
                name = binding.names[0]
                if (<Name>expr).name == name:
                    touched = True
                    continue
                for j in range(i+1, len(bindings)):
                    if (<Name>expr).name in (<Binding>bindings[j]).names:
                        context.names[name] = None
                        shadowed.add((<Name>expr).name)
                        break
                else:
                    context.names[name] = expr
                    discarded.add(name)
                    touched = True
                    continue
            for name in binding.names:
                context.names[name] = None
            remaining.append(PolyBinding(binding.names, expr))
        cdef bint resimplify = shadowed and shadowed & discarded
        cdef Expression sbody = body._simplify(context)
        touched |= sbody is not body
        context.names = saved
        if isinstance(sbody, Literal):
            return sbody
        if isinstance(sbody, Let):
            remaining.extend((<Let>sbody).bindings)
            sbody = (<Let>sbody).body
            resimplify = True
            touched = True
        if not touched:
            return self
        if remaining:
            sbody = Let(tuple(remaining), sbody)
        if resimplify:
            return sbody._simplify(context)
        return sbody

    def __repr__(self):
        return f'Let({self.bindings!r}, {self.body!r})'


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
        cdef int64_t i, n=len(self.names)
        program.begin_for(n)
        START = program.new_label()
        END = program.new_label()
        program.label(START)
        program.next(END)
        lnames.extend(self.names)
        self.body._compile(program, lnames)
        program.jump(START)
        program.label(END)
        program.end_for()
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
            if source is self.source and body is self.body:
                return self
            return For(self.names, source, body)
        values = (<Literal>source).value
        cdef int64_t i=0, n=values.length
        while i < n:
            for name in self.names:
                context.names[name] = values.item(i) if i < n else null_
                i += 1
            remaining.append(self.body._simplify(context))
        context.names = saved
        sequence_pack(remaining)
        if not remaining:
            return NoOp
        if len(remaining) == 1:
            return remaining[0]
        return Sequence(tuple(remaining))

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
        cdef bint touched = False
        for test in self.tests:
            condition = test.condition._simplify(context)
            touched |= condition is not test.condition
            if isinstance(condition, Literal):
                if (<Literal>condition).value.as_bool():
                    then = test.then._simplify(context)
                    if not remaining:
                        return then
                    else:
                        return IfElse(tuple(remaining), then)
                touched = True
            else:
                then = test.then._simplify(context)
                remaining.append(IfCondition(condition, then))
                touched |= then is not test.then
        else_ = self.else_._simplify(context) if self.else_ is not None else None
        touched |= else_ is not self.else_
        if not touched:
            return self
        if remaining:
            return IfElse(tuple(remaining), else_)
        return NoOp if else_ is None else else_

    def __repr__(self):
        return f'IfElse({self.tests!r}, {self.else_!r})'


cdef class Function(Expression):
    cdef readonly str name
    cdef readonly tuple parameters
    cdef readonly Expression body
    cdef readonly tuple captures
    cdef readonly bint inlineable
    cdef readonly bint recursive

    def __init__(self, str name, tuple parameters, Expression body, tuple captures=None, bint inlineable=False, bint recursive=False):
        self.name = name
        self.parameters = parameters
        self.body = body
        self.captures = captures
        self.inlineable = inlineable
        self.recursive = recursive

    cdef void _compile(self, Program program, list lnames):
        cdef str name
        cdef list function_lnames = []
        cdef tuple captures = self.captures if self.captures is not None else tuple(lnames)
        cdef int64_t i, n=len(lnames)-1
        for name in captures:
            for i in range(len(lnames)):
                if name == <str>lnames[n-i]:
                    program.local_load(i)
                    break
            else:
                program.compiler_errors.add(f"Unbound name '{name}'")
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
        self.body._compile(program, function_lnames)
        program.exit()
        program.label(END)
        program.func(START, self.name, tuple(parameters), len(captures))

    cdef Expression _simplify(self, Context context):
        cdef list parameters = []
        cdef Binding parameter
        cdef bint literal = True
        cdef Expression body
        cdef bint touched = False
        for parameter in self.parameters:
            expr = parameter.expr._simplify(context) if parameter.expr is not None else None
            if expr is not None and not isinstance(expr, Literal):
                literal = False
            parameters.append(Binding(parameter.name, expr))
            touched |= expr is not parameter.expr
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
        body = self.body._simplify(context)
        touched |= body is not self.body
        cdef recursive = self.name in context.captures
        if recursive:
            context.captures.discard(self.name)
        touched |= recursive != self.recursive
        cdef tuple captures_t = tuple(context.captures)
        cdef bint inlineable = literal and not captures_t
        touched |= inlineable != self.inlineable
        context.names = dict(saved)
        if captures is not None:
            context.captures.difference_update(saved)
            captures.update(context.captures)
        context.captures = captures
        touched |= captures_t != self.captures
        if not touched:
            return self
        return Function(self.name, tuple(parameters), body, captures_t, inlineable, recursive)

    def __repr__(self):
        return f'Function({self.name!r}, {self.parameters!r}, {self.body!r}, {self.captures!r}, {self.inlineable!r}, {self.recursive!r})'
