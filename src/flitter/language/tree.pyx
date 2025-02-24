
"""
Language abstract syntax tree

The tree supports reasonably sophisticated simplification that will reduce
the tree down to a "simpler" form by propogating constants. As this can result
in unrolling loops, simpler does not necessarily mean "smaller."
"""

from loguru import logger

from libc.stdint cimport int64_t
from cpython cimport PyObject
from cpython.dict cimport PyDict_GetItem

from .. import name_patch
from ..cache import SharedCache
from ..model cimport Vector, Node, Context, StateDict, null_, true_, false_, minusone_
from .vm cimport Program, Instruction, InstructionInt, InstructionVector, OpCode, static_builtins, dynamic_builtins


logger = name_patch(logger, __name__)

cdef frozenset EmptySet = frozenset()
cdef Literal NoOp = Literal(null_)
cdef int64_t MAX_RECURSIVE_CALL_DEPTH = 500
cdef Vector Two = Vector(2)


cdef bint sequence_pack(list expressions):
    cdef Expression expr
    cdef bint touched = False
    cdef list vectors, todo=expressions.copy()
    todo.reverse()
    expressions.clear()
    while todo:
        expr = <Expression>todo.pop()
        if type(expr) is Literal:
            vectors = [(<Literal>expr).value]
            while todo and type(todo[-1]) is Literal:
                vectors.append((<Literal>todo.pop()).value)
            if len(vectors) > 1:
                expr = Literal(Vector._compose(vectors))
                touched = True
        elif type(expr) is Sequence:
            for expr in reversed((<Sequence>expr).expressions):
                todo.append(expr)
            touched = True
            continue
        expressions.append(expr)
    return touched


cdef class Expression:
    cdef readonly frozenset unbound_names

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
        self.unbound_names = self.body.unbound_names

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
        self.unbound_names = frozenset([None])

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
        self.unbound_names = self.filename.unbound_names.union(self.expr.unbound_names.difference(self.names))

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
        if type(filename) is Literal and context.path is not None:
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
            expr = Let(tuple(PolyBinding((name,), value if type(value) is Function else Literal(value)) for name, value in let_names.items()), expr)
        try:
            expr = expr._simplify(context)
        finally:
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
        cdef set unbound = set()
        cdef Expression expr
        for expr in self.expressions:
            unbound.update(expr.unbound_names)
        self.unbound_names = frozenset(unbound)

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
        self.unbound_names = EmptySet

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
        self.unbound_names = frozenset([name])

    cdef void _compile(self, Program program, list lnames):
        cdef int64_t i
        cdef PyObject* ptr
        for i, name in enumerate(reversed(lnames)):
            if self.name == <str>name:
                program.local_load(i)
                break
        else:
            if (ptr := PyDict_GetItem(static_builtins, self.name)) != NULL:
                program.literal(<Vector>ptr)
            elif (ptr := PyDict_GetItem(dynamic_builtins, self.name)) != NULL:
                program.literal(<Vector>ptr)
            else:
                program.compiler_errors.add(f"Unbound name '{self.name}'")
                program.literal(null_)

    cdef Expression _simplify(self, Context context):
        cdef str name = self.name
        cdef PyObject* ptr
        cdef Literal literal
        if (ptr := PyDict_GetItem(context.names, name)) != NULL:
            value = <object>ptr
            if value is None or type(value) is Function:
                return self
            elif type(value) is Name:
                return (<Name>value)._simplify(context)
            elif type(value) is Vector:
                literal = Literal.__new__(Literal)
                literal.value = (<Vector>value).copy()
                literal.unbound_names = EmptySet
                return literal
            else:
                literal = Literal.__new__(Literal)
                literal.value = Vector._coerce(value)
                literal.unbound_names = EmptySet
                return literal
        elif (ptr := PyDict_GetItem(static_builtins, name)) != NULL:
            literal = Literal.__new__(Literal)
            literal.value = <Vector>ptr
            literal.unbound_names = EmptySet
            return literal
        elif PyDict_GetItem(dynamic_builtins, name) == NULL:
            context.errors.add(f"Unbound name '{name}'")
            return NoOp
        return self

    def __repr__(self):
        return f'Name({self.name!r})'


cdef class Lookup(Expression):
    cdef readonly Expression key

    def __init__(self, Expression key):
        self.key = key
        self.unbound_names = self.key.unbound_names

    cdef void _compile(self, Program program, list lnames):
        if type(self.key) is Literal:
            program.lookup_literal((<Literal>self.key).value.intern())
        else:
            self.key._compile(program, lnames)
            program.lookup()

    cdef Expression _simplify(self, Context context):
        cdef Expression key = self.key._simplify(context)
        cdef Vector value
        if type(key) is Literal:
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
        self.unbound_names = self.start.unbound_names.union(self.stop.unbound_names).union(self.step.unbound_names)

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
        if type(start) is Literal and type(stop) is Literal and type(step) is Literal:
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
        self.unbound_names = self.expr.unbound_names

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
        if type(expr) is Literal:
            return Literal((<Literal>expr).value.neg())
        if type(expr) is Negative:
            expr = Positive((<Negative>expr).expr)
            return expr._simplify(context)
        cdef MathsBinaryOperation maths
        if type(expr) is Multiply or type(expr) is Divide:
            maths = expr
            if type(maths.left) is Literal:
                expr = type(expr)(Negative(maths.left), maths.right)
                return expr._simplify(context)
            if type(maths.right) is Literal:
                expr = type(expr)(maths.left, Negative(maths.right))
                return expr._simplify(context)
        elif type(expr) is Add:
            maths = expr
            if type(maths.left) is Literal or type(maths.right) is Literal:
                expr = Add(Negative(maths.left), Negative(maths.right))
                return expr._simplify(context)
        elif type(expr) is Subtract:
            maths = expr
            if type(maths.left) is Literal or type(maths.right) is Literal:
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
        if type(expr) is Literal:
            return Literal((<Literal>expr).value.pos())
        if type(expr) is Negative or type(expr) is Positive or isinstance(expr, MathsBinaryOperation):
            return expr._simplify(context)
        if expr is self.expr:
            return self
        return Positive(expr)


cdef class Not(UnaryOperation):
    cdef void _compile_op(self, Program program):
        program.not_()

    cdef Expression _simplify(self, Context context):
        cdef Expression expr = self.expr._simplify(context)
        if type(expr) is Literal:
            return Literal(false_ if (<Literal>expr).value.as_bool() else true_)
        if expr is self.expr:
            return self
        return Not(expr)


cdef class Ceil(UnaryOperation):
    cdef void _compile_op(self, Program program):
        program.ceil()

    cdef Expression _simplify(self, Context context):
        cdef Expression expr = self.expr._simplify(context)
        if type(expr) is Literal:
            return Literal((<Literal>expr).value.ceil())
        if expr is self.expr:
            return self
        return Ceil(expr)


cdef class Floor(UnaryOperation):
    cdef void _compile_op(self, Program program):
        program.floor()

    cdef Expression _simplify(self, Context context):
        cdef Expression expr = self.expr._simplify(context)
        if type(expr) is Literal:
            return Literal((<Literal>expr).value.floor())
        if expr is self.expr:
            return self
        return Floor(expr)


cdef class Fract(UnaryOperation):
    cdef void _compile_op(self, Program program):
        program.fract()

    cdef Expression _simplify(self, Context context):
        cdef Expression expr = self.expr._simplify(context)
        if type(expr) is Literal:
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
        self.unbound_names = self.left.unbound_names.union(self.right.unbound_names)

    cdef void _compile(self, Program program, list lnames):
        self.left._compile(program, lnames)
        self.right._compile(program, lnames)
        self._compile_op(program)

    cdef void _compile_op(self, Program program):
        raise NotImplementedError()

    cdef Expression _simplify(self, Context context):
        cdef Expression left = self.left._simplify(context)
        cdef Expression right = self.right._simplify(context)
        cdef bint literal_left = type(left) is Literal
        cdef bint literal_right = type(right) is Literal
        cdef Expression expr
        cdef Literal literal
        if literal_left and literal_right:
            literal = Literal.__new__(Literal)
            literal.value = self.op((<Literal>left).value, (<Literal>right).value)
            literal.unbound_names = EmptySet
            return literal
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
        cdef type T = type(self)
        cdef BinaryOperation binary = <BinaryOperation>T.__new__(T)
        binary.left = left
        binary.right = right
        binary.unbound_names = left.unbound_names.union(right.unbound_names)
        return binary

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
    cdef Expression additional_rules(self, Expression left, Expression right):
        if type(left) is Positive:
            return (<Expression>type(self)((<Positive>left).expr, right))
        elif type(right) is Positive:
            return (<Expression>type(self)(left, (<Positive>right).expr))


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
        if left.numbers == NULL:
            return NoOp
        if left.eq(false_) is true_:
            return Positive(right)

    cdef Expression constant_right(self, Expression left, Vector right):
        return self.constant_left(right, left)

    cdef Expression additional_rules(self, Expression left, Expression right):
        if (expr := MathsBinaryOperation.additional_rules(self, left, right)) is not None:
            return expr
        if type(right) is not Multiply and type(left) is Multiply:
            return Add(right, left)
        if type(right) is Negative:
            return Subtract(left, (<Negative>right).expr)
        if type(left) is Negative:
            return Subtract(right, (<Negative>left).expr)


cdef class Subtract(MathsBinaryOperation):
    cdef Vector op(self, Vector left, Vector right):
        return left.sub(right)

    cdef void _compile_op(self, Program program):
        program.sub()

    cdef Expression constant_left(self, Vector left, Expression right):
        if left.numbers == NULL:
            return NoOp
        if left.eq(false_) is true_:
            return Negative(right)

    cdef Expression constant_right(self, Expression left, Vector right):
        if right.eq(false_) is true_:
            return Positive(left)
        return Add(left, Literal(right.neg()))

    cdef Expression additional_rules(self, Expression left, Expression right):
        if (expr := MathsBinaryOperation.additional_rules(self, left, right)) is not None:
            return expr
        if type(right) is Negative:
            return Add(left, (<Negative>right).expr)


cdef class Multiply(MathsBinaryOperation):
    cdef Vector op(self, Vector left, Vector right):
        return left.mul(right)

    cdef void _compile_op(self, Program program):
        program.mul()

    cdef Expression constant_left(self, Vector left, Expression right):
        if left.numbers == NULL:
            return NoOp
        if left.eq(true_) is true_:
            return Positive(right)
        if left.eq(minusone_) is true_:
            return Negative(right)
        cdef MathsBinaryOperation maths
        if type(right) is Add or type(right) is Subtract:
            maths = right
            if type(maths.left) is Literal or type(maths.right) is Literal:
                return type(maths)(Multiply(Literal(left), maths.left), Multiply(Literal(left), maths.right))
        elif type(right) is Multiply:
            maths = right
            if type(maths.left) is Literal:
                return Multiply(Multiply(Literal(left), maths.left), maths.right)
            if type(maths.right) is Literal:
                return Multiply(Multiply(Literal(left), maths.right), maths.left)
        elif type(right) is Divide:
            maths = right
            if type(maths.left) is Literal:
                return Divide(Multiply(Literal(left), maths.left), maths.right)
        elif type(right) is Negative:
            return Multiply(Literal(left.neg()), (<Negative>right).expr)

    cdef Expression constant_right(self, Expression left, Vector right):
        return self.constant_left(right, left)


cdef class Divide(MathsBinaryOperation):
    cdef Vector op(self, Vector left, Vector right):
        return left.truediv(right)

    cdef void _compile_op(self, Program program):
        program.truediv()

    cdef Expression constant_left(self, Vector left, Expression right):
        if left.numbers == NULL:
            return NoOp

    cdef Expression constant_right(self, Expression left, Vector right):
        if right.numbers == NULL:
            return NoOp
        if right.eq(true_) is true_:
            return Positive(left)
        return Multiply(Literal(true_.truediv(right)), left)


cdef class FloorDivide(MathsBinaryOperation):
    cdef Vector op(self, Vector left, Vector right):
        return left.floordiv(right)

    cdef void _compile_op(self, Program program):
        program.floordiv()

    cdef Expression constant_left(self, Vector left, Expression right):
        if left.numbers == NULL:
            return NoOp

    cdef Expression constant_right(self, Expression left, Vector right):
        if right.numbers == NULL:
            return NoOp
        if right.eq(true_) is true_:
            return Floor(left)


cdef class Modulo(MathsBinaryOperation):
    cdef Vector op(self, Vector left, Vector right):
        return left.mod(right)

    cdef void _compile_op(self, Program program):
        program.mod()

    cdef Expression constant_left(self, Vector left, Expression right):
        if left.numbers == NULL:
            return NoOp

    cdef Expression constant_right(self, Expression left, Vector right):
        if right.numbers == NULL:
            return NoOp
        if right.eq(true_) is true_:
            return Fract(left)


cdef class Power(MathsBinaryOperation):
    cdef Vector op(self, Vector left, Vector right):
        return left.pow(right)

    cdef void _compile_op(self, Program program):
        cdef Instruction instr = program.last_instruction()
        if instr.code == OpCode.Literal and (<InstructionVector>instr).value.eq(Two) is true_:
            program.pop_instruction()
            program.dup()
            program.mul()
        else:
            program.pow()

    cdef Expression constant_left(self, Vector left, Expression right):
        if left.numbers == NULL:
            return NoOp

    cdef Expression constant_right(self, Expression left, Vector right):
        if right.numbers == NULL:
            return NoOp
        if right.eq(true_) is true_:
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


cdef class Contains(BinaryOperation):
    cdef Vector op(self, Vector left, Vector right):
        return right.contains(left)

    cdef void _compile_op(self, Program program):
        program.contains()

    cdef Expression constant_left(self, Vector left, Expression right):
        if left.length == 0:
            return Literal(true_)


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
        if type(left) is Literal:
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
        if type(left) is Literal:
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
        cdef bint literal_left = type(left) is Literal
        cdef bint literal_right = type(right) is Literal
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
        self.unbound_names = self.expr.unbound_names.union(self.index.unbound_names)

    cdef void _compile(self, Program program, list lnames):
        self.expr._compile(program, lnames)
        if type(self.index) is Literal:
            program.slice_literal((<Literal>self.index).value.intern())
        else:
            self.index._compile(program, lnames)
            program.slice()

    cdef Expression _simplify(self, Context context):
        cdef Expression expr = self.expr._simplify(context)
        cdef Expression index = self.index._simplify(context)
        cdef Vector expr_value
        cdef Vector index_value
        if type(expr) is Literal and type(index) is Literal:
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
        cdef set unbound = set()
        unbound.update(self.function.unbound_names)
        cdef Expression arg
        if self.args is not None:
            for arg in self.args:
                unbound.update(arg.unbound_names)
        cdef Binding binding
        if self.keyword_args is not None:
            for binding in self.keyword_args:
                unbound.update(binding.expr.unbound_names)
        self.unbound_names = frozenset(unbound)

    cdef void _compile(self, Program program, list lnames):
        cdef Expression expr
        cdef list names = []
        if self.args is not None:
            for expr in self.args:
                expr._compile(program, lnames)
        cdef Binding keyword_arg
        if self.keyword_args is not None:
            for keyword_arg in self.keyword_args:
                names.append(keyword_arg.name)
                keyword_arg.expr._compile(program, lnames)
        if not names and type(self.function) is Literal \
                and (<Literal>self.function).value.length == 1 \
                and (<Literal>self.function).value.objects is not None \
                and not type(function := (<Literal>self.function).value.objects[0]) is Function:
            program.call_fast(function, len(self.args) if self.args is not None else 0)
        else:
            self.function._compile(program, lnames)
            program.call(len(self.args) if self.args is not None else 0, tuple(names) if names else None)

    cdef Expression _simplify(self, Context context):
        cdef Expression function = self.function._simplify(context)
        cdef bint touched = function is not self.function
        cdef Function func_expr = None
        if type(function) is Function:
            func_expr = <Function>function
        elif type(function) is Name and (<Name>function).name in context.names:
            value = context.names[(<Name>function).name]
            if type(value) is Function:
                func_expr = <Function>value
        cdef bint literal_func = type(function) is Literal
        if literal_func and (<Literal>function).value.length == 0:
            return NoOp
        cdef bint all_literal_args=True, all_dynamic_args=True
        cdef Expression arg, sarg, expr
        cdef list args = []
        if self.args is not None:
            for arg in self.args:
                sarg = arg._simplify(context)
                touched |= sarg is not arg
                args.append(sarg)
                if type(sarg) is Literal:
                    all_dynamic_args = False
                else:
                    all_literal_args = False
        cdef list keyword_args = []
        cdef Binding binding, sbinding
        if self.keyword_args is not None:
            for binding in self.keyword_args:
                arg = binding.expr._simplify(context)
                if arg is not binding.expr:
                    sbinding = Binding.__new__(Binding)
                    sbinding.name = binding.name
                    sbinding.expr = arg
                    keyword_args.append(sbinding)
                    touched = True
                else:
                    keyword_args.append(binding)
                if type(arg) is Literal:
                    all_dynamic_args = False
                else:
                    all_literal_args = False
        cdef list bindings, renames
        cdef dict kwargs
        cdef int64_t i, j=0
        cdef str name
        cdef tuple vector_args
        cdef Vector result
        cdef list results
        cdef Literal literal
        cdef PolyBinding polybinding
        if literal_func and all_literal_args:
            vector_args = tuple([literal.value for literal in args])
            kwargs = {binding.name: (<Literal>binding.expr).value for binding in keyword_args}
            results = []
            if (<Literal>function).value.objects is not None:
                for func in (<Literal>function).value.objects:
                    if callable(func):
                        try:
                            result = func(*vector_args, **kwargs)
                        except Exception as exc:
                            context.errors.add(f"Error calling {func.__name__}: {str(exc)}")
                        else:
                            results.append(result)
                    else:
                        context.errors.add(f"{func!r} is not callable")
            elif (<Literal>function).value.numbers != NULL:
                for i in range((<Literal>function).value.length):
                    context.errors.add(f"{(<Literal>function).value.numbers[i]!r} is not callable")
            literal = Literal.__new__(Literal)
            literal.value = Vector._compose(results)
            literal.unbound_names = EmptySet
            return literal
        if func_expr is not None and not func_expr.captures and not (func_expr.recursive and all_dynamic_args):
            kwargs = {binding.name: binding.expr for binding in keyword_args}
            bindings = []
            renames = []
            for i, binding in enumerate(func_expr.parameters):
                name = binding.name
                if i < len(args):
                    expr = <Expression>args[i]
                elif name in kwargs:
                    expr = <Expression>kwargs[name]
                elif binding.expr is not None:
                    expr = binding.expr
                else:
                    expr = NoOp
                while name in context.names:
                    name = f'__t{j}'
                    j += 1
                polybinding = PolyBinding.__new__(PolyBinding)
                polybinding.names = (name,)
                polybinding.expr = expr
                bindings.append(polybinding)
                if name is not binding.name:
                    polybinding = PolyBinding.__new__(PolyBinding)
                    polybinding.names = (binding.name,)
                    polybinding.expr = Name(name)
                    renames.append(polybinding)
            bindings.extend(renames)
            expr = Let(tuple(bindings), func_expr.body)
            if func_expr.recursive:
                if context.call_depth == 0:
                    context.call_depth = 1
                    try:
                        return expr._simplify(context)
                    except RecursionError:
                        logger.trace("Abandoned inline attempt of recursive function: {}", func_expr.name)
                    context.call_depth = 0
                elif context.call_depth == MAX_RECURSIVE_CALL_DEPTH:
                    raise RecursionError()
                else:
                    context.call_depth += 1
                    expr = expr._simplify(context)
                    context.call_depth -= 1
                    return expr
            else:
                return expr._simplify(context)
        if type(function) is Literal and len(args) == 1:
            if (<Literal>function).value == static_builtins['ceil']:
                return Ceil(args[0])
            if (<Literal>function).value == static_builtins['floor']:
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
        self.unbound_names = self.node.unbound_names

    cdef void _compile(self, Program program, list lnames):
        self.node._compile(program, lnames)
        program.tag(self.tag)

    cdef Expression _simplify(self, Context context):
        cdef Expression node = self.node._simplify(context)
        cdef Vector nodes
        cdef list objects
        cdef Node n
        cdef int64_t i
        if type(node) is Literal:
            nodes = (<Literal>node).value
            if nodes.objects is None:
                return node
            objects = []
            for i in range(nodes.length):
                obj = nodes.objects[i]
                if type(obj) is Node:
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
        cdef set unbound = set()
        unbound.update(self.node.unbound_names)
        cdef Binding binding
        for binding in self.bindings:
            unbound.update(binding.expr.unbound_names)
        self.unbound_names = frozenset(unbound)

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
        cdef Binding binding, simplified
        cdef Expression expr
        cdef bint touched = False
        while type(node) is Attributes:
            attrs = <Attributes>node
            for binding in reversed(attrs.bindings):
                expr = binding.expr._simplify(context)
                touched |= expr is not binding.expr
                simplified = Binding.__new__(Binding)
                simplified.name = binding.name
                simplified.expr = expr
                bindings.append(simplified)
            node = attrs.node
        node = node._simplify(context)
        touched |= node is not self.node
        cdef Vector nodes, value
        cdef list objects
        if type(node) is Literal:
            nodes = (<Literal>node).value
            if nodes.objects is None:
                return node
            if bindings and type((<Binding>bindings[-1]).expr) is Literal:
                objects = []
                for obj in nodes.objects:
                    objects.append((<Node>obj).copy() if type(obj) is Node else obj)
                while bindings and type((<Binding>bindings[-1]).expr) is Literal:
                    binding = <Binding>bindings.pop()
                    value = (<Literal>binding.expr).value
                    for obj in objects:
                        if type(obj) is Node:
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
        self.unbound_names = self.node.unbound_names.union(self.children.unbound_names)

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
        if (type(node) is Literal and (<Literal>node).value.objects is None) or \
                (type(children) is Literal and (<Literal>children).value.objects is None):
            return node
        if type(node) is Literal:
            nodes = (<Literal>node).value
            if type(children) is Literal:
                children_vector = (<Literal>children).value
                node_objects = nodes.objects
                objects = []
                for i in range(nodes.length):
                    obj = node_objects[i]
                    if type(obj) is Node:
                        obj = (<Node>obj).copy()
                        (<Node>obj).append_vector(children_vector)
                    objects.append(obj)
                return Literal(objects)
            elif type(children) is Sequence and type((<Sequence>children).expressions[0]) is Literal:
                node = Append(node, (<Sequence>children).expressions[0])
                children = Sequence((<Sequence>children).expressions[1:])
                return Append(node, children)._simplify(context)
        elif type(node) is Attributes and type((<Attributes>node).node) is Literal:
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
        cdef set bound = set()
        cdef set unbound = set()
        cdef PolyBinding binding
        for binding in self.bindings:
            unbound.update(binding.expr.unbound_names.difference(bound))
            bound.update(binding.names)
        unbound.update(self.body.unbound_names.difference(bound))
        self.unbound_names = frozenset(unbound)

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
        cdef str name, rename
        cdef int64_t i, j, n
        cdef bint touched = False
        cdef set shadowed=set(), discarded=set()
        while type(body) is Let:
            bindings.extend((<Let>body).bindings)
            body = (<Let>body).body
            touched = True
        cdef dict renames = {}
        for name, name_value in context.names.items():
            if type(name_value) is Name:
                rename = (<Name>name_value).name
                if rename in renames:
                    (<list>renames[rename]).append(name)
                else:
                    renames[rename] = [name]
        for i, binding in enumerate(bindings):
            for rename in binding.names:
                if rename not in shadowed and rename in renames:
                    for name in <list>renames.pop(rename):
                        context.names[name] = None
                        remaining.append(PolyBinding((name,), Name(rename)))
                    shadowed.add(rename)
                    touched = True
            n = len(binding.names)
            expr = binding.expr._simplify(context)
            touched |= expr is not binding.expr
            if n == 1 and type(expr) is Function and (<Function>expr).inlineable:
                context.names[binding.names[0]] = expr
                remaining.append(PolyBinding(binding.names, expr))
                continue
            if type(expr) is Literal:
                value = (<Literal>expr).value
                if n == 1:
                    name = <str>binding.names[0]
                    context.names[name] = value
                    discarded.add(name)
                else:
                    for j, name in enumerate(binding.names):
                        context.names[name] = value.item(j)
                        discarded.add(name)
                touched = True
                continue
            if n == 1 and type(expr) is Name:
                name = <str>binding.names[0]
                rename = (<Name>expr).name
                if name == rename:
                    touched = True
                    continue
                for j in range(i+1, len(bindings)):
                    if rename in (<Binding>bindings[j]).names:
                        shadowed.add(rename)
                        break
                else:
                    context.names[name] = expr
                    discarded.add(name)
                    touched = True
                    continue
            for name in binding.names:
                context.names[name] = None
            remaining.append(PolyBinding(binding.names, expr))
        cdef bint resimplify = not shadowed.isdisjoint(discarded)
        cdef Expression sbody
        try:
            sbody = body._simplify(context)
            touched |= sbody is not body
        finally:
            context.names = saved
        if type(sbody) is Literal:
            return sbody
        cdef set unbound
        cdef list original
        if None not in sbody.unbound_names:
            unbound = set(sbody.unbound_names)
            original = remaining
            remaining = []
            for binding in reversed(original):
                for name in binding.names:
                    if name in unbound:
                        remaining.insert(0, binding)
                        unbound.update(binding.expr.unbound_names)
                        break
                else:
                    touched = True
        if type(sbody) is Let:
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
        self.unbound_names = self.source.unbound_names.union(self.body.unbound_names.difference(self.names))

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
        if not type(source) is Literal:
            for name in self.names:
                context.names[name] = None
            try:
                body = self.body._simplify(context)
            finally:
                context.names = saved
            if source is self.source and body is self.body:
                return self
            return For(self.names, source, body)
        values = (<Literal>source).value
        cdef int64_t i=0, n=values.length
        try:
            while i < n:
                for name in self.names:
                    context.names[name] = values.item(i) if i < n else null_
                    i += 1
                remaining.append(self.body._simplify(context))
        finally:
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
        cdef set unbound = set()
        cdef IfCondition test
        for test in self.tests:
            unbound.update(test.condition.unbound_names)
            unbound.update(test.then.unbound_names)
        if self.else_:
            unbound.update(self.else_.unbound_names)
        self.unbound_names = frozenset(unbound)

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
            if type(condition) is Literal:
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
        cdef set bound = set([self.name])
        cdef set unbound = set()
        cdef Binding parameter
        for parameter in self.parameters:
            if parameter.expr is not None:
                unbound.update(parameter.expr.unbound_names)
            bound.add(parameter.name)
        unbound.update(self.body.unbound_names.difference(bound))
        self.unbound_names = frozenset(unbound)

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
            if expr is not None and not type(expr) is Literal:
                literal = False
            parameters.append(Binding(parameter.name, expr))
            touched |= expr is not parameter.expr
        cdef dict saved = context.names
        context.names = dict(saved)
        context.names[self.name] = None
        cdef set bound_names = set()
        bound_names.update(dynamic_builtins)
        bound_names.update(static_builtins)
        bound_names.difference_update(context.names)
        for parameter in parameters:
            context.names[parameter.name] = None
            bound_names.add(parameter.name)
        try:
            body = self.body._simplify(context)
        finally:
            context.names = saved
        cdef set captures = set(body.unbound_names.difference(bound_names))
        touched |= body is not self.body
        cdef bint recursive = self.name in captures
        if recursive:
            captures.discard(self.name)
        touched |= recursive != self.recursive
        cdef tuple captures_t = tuple(captures)
        cdef bint inlineable = literal and not captures_t
        touched |= inlineable != self.inlineable
        touched |= <bint>(captures_t != self.captures)
        if not touched:
            return self
        return Function(self.name, tuple(parameters), body, captures_t, inlineable, recursive)

    def __repr__(self):
        return f'Function({self.name!r}, {self.parameters!r}, {self.body!r}, {self.captures!r}, {self.inlineable!r}, {self.recursive!r})'
