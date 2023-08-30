# cython: language_level=3, profile=False

"""
Flitter language stack-based virtual machine
"""

import time

import cython
from loguru import logger

from .. import name_patch
from ..cache import SharedCache
from .functions import STATIC_FUNCTIONS, DYNAMIC_FUNCTIONS
from ..model cimport Vector, Node, Query, null_, true_, false_
from .noise import NOISE_FUNCTIONS

from libc.math cimport floor
from cpython cimport PyObject
from cpython.dict cimport PyDict_Update, PyDict_GetItem, PyDict_Size


logger = name_patch(logger, __name__)

cdef dict static_builtins = {
    'true': true_,
    'false': false_,
    'null': null_,
}
static_builtins.update(STATIC_FUNCTIONS)
static_builtins.update(NOISE_FUNCTIONS)

def log(value):
    return value

cdef dict dynamic_builtins = {
    'log': Vector(log),
}
dynamic_builtins.update(DYNAMIC_FUNCTIONS)

cdef dict builtins = {}
builtins.update(dynamic_builtins)
builtins.update(static_builtins)

cdef int NextLabel = 1

cdef enum OpCode:
    Add
    Append
    AppendRoot
    Attribute
    BeginFor
    BeginScope
    BranchFalse
    BranchTrue
    Call
    ClearNodeScope
    Compose
    Drop
    Dup
    EndFor
    EndScope
    Eq
    FloorDiv
    Func
    Ge
    Gt
    Import
    IndexLiteral
    Jump
    Label
    LiteralNode
    Le
    Let
    Literal
    Lookup
    Lt
    Mod
    Mul
    Name
    Ne
    Neg
    Next
    Not
    Pos
    Pow
    Pragma
    Prepend
    PushNext
    Range
    Search
    SetNodeScope
    Slice
    SliceLiteral
    Sub
    Tag
    TrueDiv
    Xor

cdef dict OpCodeNames = {
    OpCode.Add: 'Add',
    OpCode.Append: 'Append',
    OpCode.AppendRoot: 'AppendRoot',
    OpCode.Attribute: 'Attribute',
    OpCode.BeginFor: 'BeginFor',
    OpCode.BeginScope: 'BeginScope',
    OpCode.BranchFalse: 'BranchFalse',
    OpCode.BranchTrue: 'BranchTrue',
    OpCode.Call: 'Call',
    OpCode.ClearNodeScope: 'ClearNodeScope',
    OpCode.Compose: 'Compose',
    OpCode.Drop: 'Drop',
    OpCode.Dup: 'Dup',
    OpCode.EndFor: 'EndFor',
    OpCode.EndScope: 'EndScope',
    OpCode.Eq: 'Eq',
    OpCode.FloorDiv: 'FloorDiv',
    OpCode.Func: 'Func',
    OpCode.Ge: 'Ge',
    OpCode.Gt: 'Gt',
    OpCode.Import: 'Import',
    OpCode.IndexLiteral: 'IndexLiteral',
    OpCode.Jump: 'Jump',
    OpCode.Label: 'Label',
    OpCode.LiteralNode: 'LiteralNode',
    OpCode.Le: 'Le',
    OpCode.Let: 'Let',
    OpCode.Literal: 'Literal',
    OpCode.Lookup: 'Lookup',
    OpCode.Lt: 'Lt',
    OpCode.Mod: 'Mod',
    OpCode.Mul: 'Mul',
    OpCode.Name: 'Name',
    OpCode.Ne: 'Ne',
    OpCode.Neg: 'Neg',
    OpCode.Next: 'Next',
    OpCode.Not: 'Not',
    OpCode.Pos: 'Pos',
    OpCode.Pow: 'Pow',
    OpCode.Pragma: 'Pragma',
    OpCode.Prepend: 'Prepend',
    OpCode.PushNext: 'PushNext',
    OpCode.Range: 'Range',
    OpCode.Search: 'Search',
    OpCode.SetNodeScope: 'SetNodeScope',
    OpCode.Slice: 'Slice',
    OpCode.SliceLiteral: 'SliceLiteral',
    OpCode.Sub: 'Sub',
    OpCode.Tag: 'Tag',
    OpCode.TrueDiv: 'TrueDiv',
    OpCode.Xor: 'Xor',
}


cdef class Instruction:
    cdef readonly OpCode code

    def __init__(self, OpCode code):
        self.code = code

    def __str__(self):
        return OpCodeNames[self.code]


cdef class InstructionVector(Instruction):
    cdef readonly Vector value

    def __init__(self, OpCode code, Vector value):
        super().__init__(code)
        self.value = value

    def __str__(self):
        return f'{OpCodeNames[self.code]} {self.value!r}'


cdef class InstructionStr(Instruction):
    cdef readonly str value

    def __init__(self, OpCode code, str value):
        super().__init__(code)
        self.value = value

    def __str__(self):
        return f'{OpCodeNames[self.code]} {self.value!r}'


cdef class InstructionTuple(Instruction):
    cdef readonly tuple value

    def __init__(self, OpCode code, tuple value):
        super().__init__(code)
        self.value = value

    def __str__(self):
        return f'{OpCodeNames[self.code]} {self.value!r}'


cdef class InstructionInt(Instruction):
    cdef readonly int value

    def __init__(self, OpCode code, int value):
        super().__init__(code)
        self.value = value

    def __str__(self):
        return f'{OpCodeNames[self.code]} {self.value!r}'


cdef class InstructionIntTuple(Instruction):
    cdef readonly int ivalue
    cdef readonly tuple tvalue

    def __init__(self, OpCode code, int ivalue, tuple tvalue):
        super().__init__(code)
        self.ivalue = ivalue
        self.tvalue = tvalue

    def __str__(self):
        return f'{OpCodeNames[self.code]} {self.ivalue!r} {self.tvalue!r}'


cdef class InstructionQuery(Instruction):
    cdef readonly Query value

    def __init__(self, OpCode code, Query value):
        super().__init__(code)
        self.value = value

    def __str__(self):
        return f'{OpCodeNames[self.code]} {self.value!r}'


cdef class InstructionLabel(Instruction):
    cdef readonly int label

    def __init__(self, int label):
        super().__init__(OpCode.Label)
        self.label = label

    def __str__(self):
        return f'.L{self.label}'


cdef class InstructionJump(Instruction):
    cdef readonly int label
    cdef readonly int offset

    def __init__(self, OpCode code, int label):
        super().__init__(code)
        self.label = label

    def __str__(self):
        if self.offset:
            return f'{OpCodeNames[self.code]} .L{self.label} ({self.offset:+d})'
        return f'{OpCodeNames[self.code]} .L{self.label}'


cdef class InstructionJumpTuple(InstructionJump):
    cdef readonly tuple value

    def __init__(self, OpCode code, int label, tuple value):
        super().__init__(code, label)
        self.value = value

    def __str__(self):
        if self.offset:
            return f'{OpCodeNames[self.code]} {self.value!r} .L{self.label} ({self.offset:+d})'
        return f'{OpCodeNames[self.code]} {self.value!r} .L{self.label}'


cdef class Function:
    cdef readonly str __name__
    cdef readonly tuple parameters
    cdef readonly tuple defaults
    cdef readonly dict scope
    cdef readonly Program program


cdef class LoopSource:
    cdef Vector source
    cdef int position
    cdef int iterations


@cython.boundscheck(False)
@cython.wraparound(False)
cdef Vector call_helper(Context context, object function, tuple args, dict kwargs=None):
    cdef Context func_context
    cdef int i
    cdef list stack
    cdef Function func
    if isinstance(function, Function):
        func = <Function>function
        func_context = Context.__new__(Context)
        func_context.path = context.path
        func_context.errors = context.errors
        func_context.graph = context.graph
        for i, name in enumerate(func.parameters):
            if i < len(args):
                func_context.variables[name] = args[i]
            elif kwargs:
                func_context.variables[name] = kwargs.get(name, func.defaults[i])
            else:
                func_context.variables[name] = func.defaults[i]
        stack = func.program.execute(func_context, func.scope)
        assert len(stack) == 1
        return stack[0]
    elif function is log and len(args) == 1 and not kwargs:
        context.logs.add((<Vector>args[0]).repr())
        return <Vector>args[0]
    elif callable(function):
        try:
            if hasattr(function, 'state_transformer') and function.state_transformer:
                if kwargs is None:
                    return function(context.state, *args)
                else:
                    return function(context.state, *args, **kwargs)
            elif kwargs is None:
                return function(*args)
            else:
                return function(*args, **kwargs)
        except Exception as exc:
            context.errors.add(f"Error calling function {function.__name__}\n{str(exc)}")
    return null_


cdef class Program:
    def __cinit__(self):
        self.instructions = []
        self.linked = False
        self.stats = None

    def __len__(self):
        return len(self.instructions)

    def __str__(self):
        return '\n'.join(str(instruction) for instruction in self.instructions)

    def set_path(self, object path):
        self.path = path

    def set_top(self, object top):
        self.top = top

    def run(self, StateDict state=None, dict variables=None, bint record_stats=False):
        cdef dict context_vars = None
        cdef str key
        if variables is not None:
            context_vars = {}
            for key, value in variables.items():
                context_vars[key] = Vector._coerce(value)
        cdef Context context = Context(state=state, variables=context_vars, path=self.path)
        self.execute(context, record_stats=record_stats)
        return context

    @staticmethod
    def new_label():
        global NextLabel
        label = NextLabel
        NextLabel += 1
        return label

    def extend(self, Program program):
        self.instructions.extend(program.instructions)
        return self

    def log_stats(self):
        if self.stats:
            for duration, code in sorted([(duration, code) for (code, duration) in self.stats.items()], reverse=True):
                logger.info("{:20s} {:7.1f}s", OpCodeNames[code], duration)
        self.stats = None

    cpdef void link(self):
        cdef Instruction instruction
        cdef int label, address, target
        cdef list addresses
        cdef InstructionJump jump
        cdef dict jumps={}, labels={}
        for address, instruction in enumerate(self.instructions):
            if instruction.code == OpCode.Label:
                labels[(<InstructionLabel>instruction).label] = address
            elif isinstance(instruction, InstructionJump):
                (<list>jumps.setdefault((<InstructionJump>instruction).label, [])).append(address)
        for label, addresses in jumps.items():
            target = labels[label]
            for address in addresses:
                jump = self.instructions[address]
                jump.offset = target - address
        self.linked = True

    cdef dict import_module(self, Context context, str filename):
        cdef Context import_context
        program = SharedCache.get_with_root(filename, context.path).read_flitter_program()
        if program is not None:
            import_context = context.parent
            while import_context is not None:
                if import_context.path is program.path:
                    context.errors.add(f"Circular import of {filename}")
                    break
                import_context = import_context.parent
            else:
                import_context = Context(parent=context, path=program.path)
                program.execute(import_context)
                context.errors.update(import_context.errors)
                return import_context.variables
        return None

    def dup(self):
        self.instructions.append(Instruction(OpCode.Dup))

    def drop(self, int count=1):
        self.instructions.append(InstructionInt(OpCode.Drop, count))

    def label(self, int label):
        self.instructions.append(InstructionLabel(label))

    def jump(self, int label):
        self.instructions.append(InstructionJump(OpCode.Jump, label))

    def branch_true(self, int label):
        self.instructions.append(InstructionJump(OpCode.BranchTrue, label))

    def branch_false(self, int label):
        self.instructions.append(InstructionJump(OpCode.BranchFalse, label))

    def pragma(self, str name):
        self.instructions.append(InstructionStr(OpCode.Pragma, name))

    def import_(self, tuple names):
        self.instructions.append(InstructionTuple(OpCode.Import, names))

    def literal(self, value):
        if isinstance(value, Program):
            (<Program>value).link()
            value = (<Program>value).instructions
        cdef Vector vector = Vector._coerce(value)
        if vector.objects is not None:
            for obj in vector.objects:
                if isinstance(obj, Node):
                    self.instructions.append(InstructionVector(OpCode.LiteralNode, vector))
                    return
        self.instructions.append(InstructionVector(OpCode.Literal, vector))

    def name(self, str name):
        self.instructions.append(InstructionStr(OpCode.Name, name))

    def lookup(self):
        self.instructions.append(Instruction(OpCode.Lookup))

    def range(self):
        self.instructions.append(Instruction(OpCode.Range))

    def neg(self):
        self.instructions.append(Instruction(OpCode.Neg))

    def pos(self):
        self.instructions.append(Instruction(OpCode.Pos))

    def not_(self):
        self.instructions.append(Instruction(OpCode.Not))

    def add(self):
        self.instructions.append(Instruction(OpCode.Add))

    def sub(self):
        self.instructions.append(Instruction(OpCode.Sub))

    def mul(self):
        self.instructions.append(Instruction(OpCode.Mul))

    def truediv(self):
        self.instructions.append(Instruction(OpCode.TrueDiv))

    def floordiv(self):
        self.instructions.append(Instruction(OpCode.FloorDiv))

    def mod(self):
        self.instructions.append(Instruction(OpCode.Mod))

    def pow(self):
        self.instructions.append(Instruction(OpCode.Pow))

    def eq(self):
        self.instructions.append(Instruction(OpCode.Eq))

    def ne(self):
        self.instructions.append(Instruction(OpCode.Ne))

    def gt(self):
        self.instructions.append(Instruction(OpCode.Gt))

    def lt(self):
        self.instructions.append(Instruction(OpCode.Lt))

    def ge(self):
        self.instructions.append(Instruction(OpCode.Ge))

    def le(self):
        self.instructions.append(Instruction(OpCode.Le))

    def xor(self):
        self.instructions.append(Instruction(OpCode.Xor))

    def slice(self):
        self.instructions.append(Instruction(OpCode.Slice))

    def slice_literal(self, Vector value):
        if value.length == 1 and value.numbers != NULL:
            self.instructions.append(InstructionInt(OpCode.IndexLiteral, <int>floor(value.numbers[0])))
        else:
            self.instructions.append(InstructionVector(OpCode.SliceLiteral, value))

    def call(self, int count, tuple names=None):
        self.instructions.append(InstructionIntTuple(OpCode.Call, count, names))

    def tag(self, str name):
        self.instructions.append(InstructionStr(OpCode.Tag, name))

    def attribute(self, str name):
        self.instructions.append(InstructionStr(OpCode.Attribute, name))

    def append(self):
        self.instructions.append(Instruction(OpCode.Append))

    def prepend(self):
        self.instructions.append(Instruction(OpCode.Prepend))

    def compose(self, int count):
        self.instructions.append(InstructionInt(OpCode.Compose, count))

    def begin_scope(self):
        self.instructions.append(Instruction(OpCode.BeginScope))

    def end_scope(self):
        self.instructions.append(Instruction(OpCode.EndScope))

    def set_node_scope(self):
        self.instructions.append(Instruction(OpCode.SetNodeScope))

    def clear_node_scope(self):
        self.instructions.append(Instruction(OpCode.ClearNodeScope))

    def begin_for(self):
        self.instructions.append(Instruction(OpCode.BeginFor))

    def next(self, tuple names, int label):
        self.instructions.append(InstructionJumpTuple(OpCode.Next, label, names))

    def push_next(self, int label):
        self.instructions.append(InstructionJump(OpCode.PushNext, label))

    def end_for(self):
        self.instructions.append(Instruction(OpCode.EndFor))

    def let(self, tuple names):
        self.instructions.append(InstructionTuple(OpCode.Let, names))

    def search(self, Query query):
        self.instructions.append(InstructionQuery(OpCode.Search, query))

    def func(self, tuple names):
        self.instructions.append(InstructionTuple(OpCode.Func, names))

    def append_root(self):
        self.instructions.append(Instruction(OpCode.AppendRoot))

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef list execute(self, Context context, dict additional_scope=None, bint record_stats=False):
        cdef int i, n, pc=0, program_end=len(self.instructions)
        cdef Instruction instruction=None
        cdef int limit=0, top=-1
        cdef list stack=[], values, loop_sources=[]
        cdef list scopes
        if additional_scope:
            scopes = [None, builtins, additional_scope, context.variables]
        else:
            scopes = [None, builtins, context.variables]
        cdef int scopes_top = len(scopes) - 1
        cdef str name
        cdef tuple names, args
        cdef Vector r1, r2
        cdef LoopSource loop_source = None
        cdef dict scope, variables, kwargs
        cdef Query query
        cdef Function function
        cdef double now, timestamp
        cdef PyObject* objptr

        if not self.linked:
            self.link()

        if record_stats:
            if self.stats is None:
                self.stats = {}
            timestamp = time.perf_counter()

        while 0 <= pc < program_end:
            instruction = <Instruction>self.instructions[pc]
            pc += 1

            if instruction.code == OpCode.Dup:
                top += 1
                if top == limit:
                    stack.append(stack[top-1])
                    limit += 1
                else:
                    stack[top] = stack[top-1]

            elif instruction.code == OpCode.Drop:
                top -= (<InstructionInt>instruction).value

            elif instruction.code == OpCode.Label:
                pass

            elif instruction.code == OpCode.Jump:
                pc += (<InstructionJump>instruction).offset

            elif instruction.code == OpCode.BranchTrue:
                if (<Vector>stack[top]).as_bool():
                    pc += (<InstructionJump>instruction).offset
                top -= 1

            elif instruction.code == OpCode.BranchFalse:
                if not (<Vector>stack[top]).as_bool():
                    pc += (<InstructionJump>instruction).offset
                top -= 1

            elif instruction.code == OpCode.Pragma:
                context.pragmas[(<InstructionStr>instruction).value] = <Vector>stack[top]
                stack[top] = null_

            elif instruction.code == OpCode.Import:
                filename = (<Vector>stack[top]).as_string()
                stack[top] = null_
                names = (<InstructionTuple>instruction).value
                variables = self.import_module(context, filename)
                if variables is not None:
                    scope = <dict>scopes[scopes_top]
                    for name in names:
                        if name in variables:
                            scope[name] = variables[name]
                        else:
                            context.errors.add(f"Unable to import '{name}' from '{filename}'")
                            scope[name] = null_
                else:
                    context.errors.add(f"Unable to import from '{filename}'")

            elif instruction.code == OpCode.Literal:
                top += 1
                if top == limit:
                    stack.append((<InstructionVector>instruction).value)
                    limit += 1
                else:
                    stack[top] = (<InstructionVector>instruction).value

            elif instruction.code == OpCode.LiteralNode:
                top += 1
                if top == limit:
                    stack.append((<InstructionVector>instruction).value.copynodes())
                    limit += 1
                else:
                    stack[top] = (<InstructionVector>instruction).value.copynodes()

            elif instruction.code == OpCode.Name:
                top += 1
                if top == limit:
                    stack.append(None)
                    limit += 1
                name = (<InstructionStr>instruction).value
                for i in range(scopes_top, -1, -1):
                    scope = <dict>scopes[i]
                    if scope is not None and PyDict_Size(scope):
                        objptr = PyDict_GetItem(scope, name)
                        if objptr != NULL:
                            stack[top] = <Vector>objptr
                            break
                else:
                    stack[top] = null_
                    context.errors.add(f"Unbound name '{name}'")

            elif instruction.code == OpCode.Lookup:
                stack[top] = context.state.get_item(<Vector>stack[top]) if context.state is not None else null_

            elif instruction.code == OpCode.Range:
                r1 = Vector.__new__(Vector)
                r1.fill_range(<Vector>stack[top-2], <Vector>stack[top-1], <Vector>stack[top])
                top -= 2
                stack[top] = r1

            elif instruction.code == OpCode.Neg:
                stack[top] = (<Vector>stack[top]).neg()

            elif instruction.code == OpCode.Pos:
                stack[top] = (<Vector>stack[top]).pos()

            elif instruction.code == OpCode.Not:
                stack[top] = false_ if (<Vector>stack[top]).as_bool() else true_

            elif instruction.code == OpCode.Add:
                r1 = <Vector>stack[top-1]
                stack[top-1] = r1.add((<Vector>stack[top]))
                top -= 1

            elif instruction.code == OpCode.Sub:
                r1 = <Vector>stack[top-1]
                stack[top-1] = r1.sub((<Vector>stack[top]))
                top -= 1

            elif instruction.code == OpCode.Mul:
                r1 = <Vector>stack[top-1]
                stack[top-1] = r1.mul((<Vector>stack[top]))
                top -= 1

            elif instruction.code == OpCode.TrueDiv:
                r1 = <Vector>stack[top-1]
                stack[top-1] = r1.truediv((<Vector>stack[top]))
                top -= 1

            elif instruction.code == OpCode.FloorDiv:
                r1 = <Vector>stack[top-1]
                stack[top-1] = r1.floordiv((<Vector>stack[top]))
                top -= 1

            elif instruction.code == OpCode.Mod:
                r1 = <Vector>stack[top-1]
                stack[top-1] = r1.mod((<Vector>stack[top]))
                top -= 1

            elif instruction.code == OpCode.Pow:
                r1 = <Vector>stack[top-1]
                stack[top-1] = r1.pow((<Vector>stack[top]))
                top -= 1

            elif instruction.code == OpCode.Eq:
                r1 = <Vector>stack[top-1]
                stack[top-1] = r1.eq((<Vector>stack[top]))
                top -= 1

            elif instruction.code == OpCode.Ne:
                r1 = <Vector>stack[top-1]
                stack[top-1] = r1.ne((<Vector>stack[top]))
                top -= 1

            elif instruction.code == OpCode.Gt:
                r1 = <Vector>stack[top-1]
                stack[top-1] = r1.gt((<Vector>stack[top]))
                top -= 1

            elif instruction.code == OpCode.Lt:
                r1 = <Vector>stack[top-1]
                stack[top-1] = r1.lt((<Vector>stack[top]))
                top -= 1

            elif instruction.code == OpCode.Ge:
                r1 = <Vector>stack[top-1]
                stack[top-1] = r1.ge((<Vector>stack[top]))
                top -= 1

            elif instruction.code == OpCode.Le:
                r1 = <Vector>stack[top-1]
                stack[top-1] = r1.le((<Vector>stack[top]))
                top -= 1

            elif instruction.code == OpCode.Xor:
                r1 = <Vector>stack[top-1]
                r2 = <Vector>stack[top]
                top -= 1
                if not r1.as_bool():
                    stack[top] = r2
                elif r2.as_bool():
                    stack[top] = false_

            elif instruction.code == OpCode.Slice:
                r1 = <Vector>stack[top-1]
                stack[top-1] = r1.slice((<Vector>stack[top]))
                top -= 1

            elif instruction.code == OpCode.SliceLiteral:
                r1 = <Vector>stack[top]
                stack[top] = r1.slice((<InstructionVector>instruction).value)

            elif instruction.code == OpCode.IndexLiteral:
                r1 = <Vector>stack[top]
                stack[top] = r1.item((<InstructionInt>instruction).value)

            elif instruction.code == OpCode.Call:
                r1 = <Vector>stack[top]
                names = (<InstructionIntTuple>instruction).tvalue
                n = len(names) if names is not None else 0
                top -= n
                if n:
                    kwargs = {}
                    for i in range(n):
                        kwargs[names[i]] = stack[top+i]
                else:
                    kwargs = None
                n = (<InstructionIntTuple>instruction).ivalue
                if n == 1:
                    r2 = <Vector>stack[top-1]
                    args = (r2,)
                elif n:
                    args = tuple(stack[top-n:top])
                else:
                    args = ()
                top -= n
                if r1.objects:
                    if r1.length == 1:
                        stack[top] = call_helper(context, r1.objects[0], args, kwargs)
                    else:
                        values = []
                        for i in range(r1.length):
                            values.append(call_helper(context, r1.objects[i], args, kwargs))
                        stack[top] = Vector._compose(values)
                else:
                    stack[top] = null_

            elif instruction.code == OpCode.Func:
                function = Function.__new__(Function)
                function.scope = {}
                for i in range(2, len(scopes)):
                    PyDict_Update(function.scope, scopes[i])
                function.__name__ = <str>(<InstructionTuple>instruction).value[0]
                function.parameters = (<InstructionTuple>instruction).value[1:]
                n = len(function.parameters)
                function.defaults = tuple(stack[top-n+1:top+1])
                top -= n
                function.program = Program.__new__(Program)
                function.program.instructions = (<Vector>stack[top]).objects
                r1 = Vector.__new__(Vector)
                r1.objects = [function]
                r1.length = 1
                stack[top] = r1

            elif instruction.code == OpCode.Tag:
                name = (<InstructionStr>instruction).value
                r1 = <Vector>stack[top]
                if r1.objects is not None:
                    for node in r1.objects:
                        if isinstance(node, Node):
                            if (<Node>node)._tags is None:
                                (<Node>node)._tags = set()
                            (<Node>node)._tags.add(name)

            elif instruction.code == OpCode.Attribute:
                r2 = <Vector>stack[top]
                top -= 1
                r1 = <Vector>stack[top]
                if r1.objects is not None:
                    name = (<InstructionStr>instruction).value
                    for node in r1.objects:
                        if isinstance(node, Node):
                            if r2.length:
                                (<Node>node)._attributes[name] = r2
                            elif name in (<Node>node)._attributes:
                                del (<Node>node)._attributes[name]

            elif instruction.code == OpCode.Append:
                r2 = <Vector>stack[top]
                top -= 1
                r1 = <Vector>stack[top]
                if r1.objects is not None and r2.objects is not None:
                    n = r1.length - 1
                    for i, node in enumerate(r1.objects):
                        if isinstance(node, Node):
                            if i == n:
                                for child in r2.objects:
                                    if isinstance(child, Node):
                                        (<Node>node).append(<Node>child)
                            else:
                                for child in r2.objects:
                                    if isinstance(child, Node):
                                        (<Node>node).append((<Node>child).copy())

            elif instruction.code == OpCode.Prepend:
                r2 = <Vector>stack[top]
                top -= 1
                r1 = <Vector>stack[top]
                if r1.objects is not None and r2.objects is not None:
                    n = r1.length - 1
                    for i, node in enumerate(r1.objects):
                        if isinstance(node, Node):
                            if i == n:
                                for child in reversed(r2.objects):
                                    if isinstance(child, Node):
                                        (<Node>node).insert(<Node>child)
                            else:
                                for child in reversed(r2.objects):
                                    if isinstance(child, Node):
                                        (<Node>node).insert((<Node>child).copy())

            elif instruction.code == OpCode.Compose:
                n = (<InstructionInt>instruction).value - 1
                values = stack[top-n:top+1]
                top -= n
                stack[top] = Vector._compose(values)

            elif instruction.code == OpCode.BeginFor:
                if loop_source is not None:
                    loop_sources.append(loop_source)
                loop_source = LoopSource.__new__(LoopSource)
                loop_source.source = <Vector>stack[top]
                top -= 1
                loop_source.position = 0
                loop_source.iterations = 0
                scopes.append({})
                scopes_top += 1

            elif instruction.code == OpCode.Next:
                if loop_source.position >= loop_source.source.length:
                    pc += (<InstructionJump>instruction).offset
                else:
                    scope = <dict>scopes[scopes_top]
                    names = (<InstructionJumpTuple>instruction).value
                    n = len(names)
                    for i in range(n):
                        name = <str>names[i]
                        scope[name] = loop_source.source.item(loop_source.position + i)
                    loop_source.position += len(names)
                    loop_source.iterations += 1

            elif instruction.code == OpCode.PushNext:
                if loop_source.position == loop_source.source.length:
                    pc += (<InstructionJump>instruction).offset
                else:
                    r1 = loop_source.source.item(loop_source.position)
                    top += 1
                    if top == limit:
                        stack.append(r1)
                        limit += 1
                    else:
                        stack[top] = r1
                    loop_source.position += 1
                    loop_source.iterations += 1

            elif instruction.code == OpCode.EndFor:
                scopes.pop()
                scopes_top -= 1
                n = loop_source.iterations
                if n:
                    n -= 1
                    values = stack[top-n:top+1]
                    top -= n
                    stack[top] = Vector._compose(values)
                else:
                    top += 1
                    if top == limit:
                        stack.append(null_)
                        limit += 1
                    else:
                        stack[top] = null_
                loop_source = loop_sources.pop() if loop_sources else None

            elif instruction.code == OpCode.SetNodeScope:
                r1 = <Vector>stack[top]
                if r1.objects is not None and r1.length == 1 and isinstance(r1.objects[0], Node):
                    scopes[0] = (<Node>r1.objects[0])._attributes

            elif instruction.code == OpCode.ClearNodeScope:
                scopes[0] = None

            elif instruction.code == OpCode.BeginScope:
                scopes.append({})
                scopes_top += 1

            elif instruction.code == OpCode.EndScope:
                scopes.pop()
                scopes_top -= 1

            elif instruction.code == OpCode.Let:
                r1 = <Vector>stack[top]
                top -= 1
                names = (<InstructionTuple>instruction).value
                n = len(names)
                scope = <dict>scopes[scopes_top]
                if n == 1:
                    scope[names[0]] = r1
                else:
                    for i in range(n):
                        scope[names[i]] = r1.item(i)

            elif instruction.code == OpCode.Search:
                node = context.graph.first_child
                values = []
                query = (<InstructionQuery>instruction).value
                while node is not None:
                    (<Node>node)._select(query, values, False)
                    node = (<Node>node).next_sibling
                if values:
                    r1 = Vector.__new__(Vector)
                    r1.objects = values
                    r1.length = len(values)
                else:
                    r1 = null_
                top += 1
                if top == limit:
                    stack.append(r1)
                    limit += 1
                else:
                    stack[top] = r1

            elif instruction.code == OpCode.AppendRoot:
                r1 = <Vector>stack[top]
                top -= 1
                if r1.objects is not None:
                    for value in r1.objects:
                        if isinstance(value, Node):
                            if (<Node>value)._parent is None:
                                context.graph.append(<Node>value)

            else:
                raise ValueError(f"Unrecognised instruction: {instruction}")

            assert -1 <= top < limit

            if record_stats:
                now = time.perf_counter()
                self.stats[instruction.code] = (<double>self.stats.get(instruction.code, 0)) + now - timestamp
                timestamp = now

        assert pc == program_end
        return stack[:top+1]


cdef class StateDict:
    def __cinit__(self, state=None):
        cdef Vector value_vector
        self._state = {}
        self._changed_keys = set()
        if state is not None:
            for key, value in state.items():
                value_vector = Vector._coerce(value)
                if value_vector.length:
                    self._state[Vector._coerce(key)] = value_vector

    def __reduce__(self):
        return StateDict, (self._state,)

    @property
    def changed(self):
        return bool(self._changed_keys)

    @property
    def changed_keys(self):
        return frozenset(self._changed_keys)

    def clear_changed(self):
        self._changed_keys = set()

    cdef Vector get_item(self, Vector key):
        return <Vector>self._state.get(key, null_)

    cdef void set_item(self, Vector key, Vector value):
        cdef Vector current = <Vector>self._state.get(key, null_)
        if value.length:
            if value.ne(current):
                self._state[key] = value
                self._changed_keys.add(key)
        elif current.length:
            del self._state[key]
            self._changed_keys.add(key)

    cdef bint contains(self, Vector key):
        return key in self._state

    def __getitem__(self, key):
        return self.get_item(Vector._coerce(key))

    def __setitem__(self, key, value):
        self.set_item(Vector._coerce(key), Vector._coerce(value))

    def __contains__(self, key):
        return Vector._coerce(key) in self._state

    def __delitem__(self, key):
        cdef Vector key_vector = Vector._coerce(key)
        if key_vector in self._state:
            del self._state[key_vector]
            self._changed_keys.add(key_vector)

    def __iter__(self):
        return iter(self._state)

    def clear(self):
        cdef Vector key, value
        cdef dict new_state = {}
        for key, value in self._state.items():
            if key.length == 1 and key.objects is not None and isinstance(key.objects[0], str) and key.objects[0].startswith('_'):
                new_state[key] = value
        self._state = new_state
        self._changed_keys = set()

    def items(self):
        return self._state.items()

    def keys(self):
        return self._state.keys()

    def values(self):
        return self._state.values()

    def __repr__(self):
        return f"StateDict({self._state!r})"


cdef class Context:
    def __cinit__(self, dict variables=None, StateDict state=None, Node graph=None, dict pragmas=None, object path=None, Context parent=None):
        self.variables = variables if variables is not None else {}
        self.state = state
        self.graph = graph if graph is not None else Node('root')
        self.pragmas = pragmas if pragmas is not None else {}
        self.path = path
        self.parent = parent
        self.unbound = set()
        self.errors = set()
        self.logs = set()
