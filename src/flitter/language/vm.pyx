# cython: language_level=3, profile=False

import cython

from .functions import STATIC_FUNCTIONS, DYNAMIC_FUNCTIONS
from ..model cimport Vector, Node, null_, true_, false_
from .noise import NOISE_FUNCTIONS


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
    Compose
    Drop
    Dup
    EndFor
    EndScope
    Eq
    FloorDiv
    Ge
    Gt
    Jump
    Label
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
    NodeLiteral
    Not
    Pos
    Pow
    Pragma
    Prepend
    Range
    Slice
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
    OpCode.Compose: 'Compose',
    OpCode.Drop: 'Drop',
    OpCode.Dup: 'Dup',
    OpCode.EndFor: 'EndFor',
    OpCode.EndScope: 'EndScope',
    OpCode.Eq: 'Eq',
    OpCode.FloorDiv: 'FloorDiv',
    OpCode.Ge: 'Ge',
    OpCode.Gt: 'Gt',
    OpCode.Jump: 'Jump',
    OpCode.Label: 'Label',
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
    OpCode.NodeLiteral: 'NodeLiteral',
    OpCode.Not: 'Not',
    OpCode.Pos: 'Pos',
    OpCode.Pow: 'Pow',
    OpCode.Pragma: 'Pragma',
    OpCode.Prepend: 'Prepend',
    OpCode.Range: 'Range',
    OpCode.Slice: 'Slice',
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


cdef class LoopSource:
    cdef Vector source
    cdef int position
    cdef int iterations


cdef class Program:
    def __cinit__(self):
        self.instructions = []

    def __len__(self):
        return len(self.instructions)

    def __str__(self):
        return '\n'.join(str(instruction) for instruction in self.instructions)

    def free_count(self):
        cdef int i = 0
        cdef StackItem free = self.free
        while free is not None:
            assert free.value is null_
            free = free.prev
            i += 1
        return i

    @staticmethod
    def new_label():
        global NextLabel
        label = NextLabel
        NextLabel += 1
        return label

    def extend(self, Program program):
        self.instructions.extend(program.instructions)
        return self

    def link(self):
        cdef dict jumps = {}
        cdef dict labels = {}
        cdef Instruction instruction
        cdef int label, address, target
        for address, instruction in enumerate(self.instructions):
            if instruction.code == OpCode.Label:
                labels[(<InstructionLabel>instruction).label] = address
            elif isinstance(instruction, InstructionJump):
                (<list>jumps.setdefault((<InstructionJump>instruction).label, [])).append(address)
        cdef list addresses
        cdef InstructionJump jump
        for label, addresses in jumps.items():
            target = labels[label]
            for address in addresses:
                jump = self.instructions[address]
                jump.offset = target - address
        return self

    def dup(self):
        self.instructions.append(Instruction(OpCode.Dup))

    def drop(self):
        self.instructions.append(Instruction(OpCode.Drop))

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

    def literal(self, value):
        cdef Vector vector = Vector._coerce(value)
        if vector.objects is not None:
            for obj in vector.objects:
                if isinstance(obj, Node):
                    self.instructions.append(InstructionVector(OpCode.NodeLiteral, vector))
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

    def call(self, int count):
        self.instructions.append(InstructionInt(OpCode.Call, count))

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

    def begin_for(self):
        self.instructions.append(Instruction(OpCode.BeginFor))

    def next(self, tuple names, int label):
        self.instructions.append(InstructionJumpTuple(OpCode.Next, label, names))

    def end_for(self):
        self.instructions.append(Instruction(OpCode.EndFor))

    def let(self, tuple names):
        self.instructions.append(InstructionTuple(OpCode.Let, names))

    def append_root(self):
        self.instructions.append(Instruction(OpCode.AppendRoot))

    def run(self, StateDict state=None, dict variables=None):
        cdef dict context_vars = None
        cdef str key
        if variables is not None:
            context_vars = {}
            for key, value in variables.items():
                context_vars[key] = Vector._coerce(value)
        cdef Context context = Context(state=state, variables=context_vars)
        self.execute(context)
        return context

    @cython.wraparound(False)
    @cython.boundscheck(False)
    def execute(self, Context context):
        cdef int i, n, pc=0, program_end=len(self.instructions)
        cdef Instruction instruction=None
        cdef StackItem item=None, item2=None, top=None, free=self.free
        cdef list args, values, loop_sources=[]
        cdef list scopes = [dynamic_builtins, static_builtins, context.variables]
        cdef str name
        cdef tuple names
        cdef Vector vector
        cdef LoopSource loop_source = None
        while pc < program_end:
            instruction = <Instruction>self.instructions[pc]
            pc += 1

            if instruction.code == OpCode.Dup:
                if free is None:
                    free = StackItem.__new__(StackItem)
                item, free = free, free.prev
                item.value = top.value
                item.prev, top = top, item

            elif instruction.code == OpCode.Drop:
                item, top = top, top.prev
                item.prev, free = free, item
                free.value = null_

            elif instruction.code == OpCode.Label:
                pass

            elif instruction.code == OpCode.Jump:
                pc += (<InstructionJump>instruction).offset

            elif instruction.code == OpCode.BranchTrue:
                item, top = top, top.prev
                item.prev, free = free, item
                if item.value.as_bool():
                    pc += (<InstructionJump>instruction).offset
                free.value = null_

            elif instruction.code == OpCode.BranchFalse:
                item, top = top, top.prev
                item.prev, free = free, item
                if not item.value.as_bool():
                    pc += (<InstructionJump>instruction).offset
                free.value = null_

            elif instruction.code == OpCode.Pragma:
                context.pragmas[(<InstructionStr>instruction).value] = top.value
                top.value = null_

            elif instruction.code == OpCode.Literal:
                if free is None:
                    free = StackItem.__new__(StackItem)
                item, free = free, free.prev
                item.value = (<InstructionVector>instruction).value
                item.prev, top = top, item

            elif instruction.code == OpCode.NodeLiteral:
                if free is None:
                    free = StackItem.__new__(StackItem)
                item, free = free, free.prev
                item.value = (<InstructionVector>instruction).value.copynodes()
                item.prev, top = top, item

            elif instruction.code == OpCode.Name:
                if free is None:
                    free = StackItem.__new__(StackItem)
                item, free = free, free.prev
                item.prev, top = top, item
                name = (<InstructionStr>instruction).value
                for scope in reversed(scopes):
                    top.value = <Vector>(<dict>scope).get(name)
                    if top.value is not None:
                        break
                else:
                    top.value = null_

            elif instruction.code == OpCode.Lookup:
                top.value = context.state.get_item(top.value) if context.state is not None else null_

            elif instruction.code == OpCode.Range:
                item2, top = top, top.prev
                item2.prev = free
                item, top = top, top.prev
                item.prev, free = item2, item
                top.value = Vector.__new__(Vector)
                top.value.fill_range(top.value, item.value, item2.value)
                item.value = null_
                item2.value = null_

            elif instruction.code == OpCode.Neg:
                top.value = top.value.neg()

            elif instruction.code == OpCode.Pos:
                top.value = top.value.pos()

            elif instruction.code == OpCode.Not:
                top.value = false_ if top.value.as_bool() else true_

            elif instruction.code == OpCode.Add:
                item, top = top, top.prev
                item.prev, free = free, item
                top.value = top.value.add(item.value)
                free.value = null_

            elif instruction.code == OpCode.Sub:
                item, top = top, top.prev
                item.prev, free = free, item
                top.value = top.value.sub(item.value)
                free.value = null_

            elif instruction.code == OpCode.Mul:
                item, top = top, top.prev
                item.prev, free = free, item
                top.value = top.value.mul(item.value)
                free.value = null_

            elif instruction.code == OpCode.TrueDiv:
                item, top = top, top.prev
                item.prev, free = free, item
                top.value = top.value.truediv(item.value)
                free.value = null_

            elif instruction.code == OpCode.FloorDiv:
                item, top = top, top.prev
                item.prev, free = free, item
                top.value = top.value.floordiv(item.value)
                free.value = null_

            elif instruction.code == OpCode.Mod:
                item, top = top, top.prev
                item.prev, free = free, item
                top.value = top.value.mod(item.value)
                free.value = null_

            elif instruction.code == OpCode.Pow:
                item, top = top, top.prev
                item.prev, free = free, item
                top.value = top.value.pow(item.value)
                free.value = null_

            elif instruction.code == OpCode.Eq:
                item, top = top, top.prev
                item.prev, free = free, item
                top.value = top.value.eq(item.value)
                free.value = null_

            elif instruction.code == OpCode.Ne:
                item, top = top, top.prev
                item.prev, free = free, item
                top.value = top.value.ne(item.value)
                free.value = null_

            elif instruction.code == OpCode.Gt:
                item, top = top, top.prev
                item.prev, free = free, item
                top.value = top.value.gt(item.value)
                free.value = null_

            elif instruction.code == OpCode.Lt:
                item, top = top, top.prev
                item.prev, free = free, item
                top.value = top.value.lt(item.value)
                free.value = null_

            elif instruction.code == OpCode.Ge:
                item, top = top, top.prev
                item.prev, free = free, item
                top.value = top.value.ge(item.value)
                free.value = null_

            elif instruction.code == OpCode.Le:
                item, top = top, top.prev
                item.prev, free = free, item
                top.value = top.value.le(item.value)
                free.value = null_

            elif instruction.code == OpCode.Xor:
                item, top = top, top.prev
                item.prev, free = free, item
                if not top.value.as_bool():
                    top.value = item.value
                elif not item.value.as_bool():
                    pass
                else:
                    top.value = false_
                free.value = null_

            elif instruction.code == OpCode.Slice:
                item, top = top, top.prev
                item.prev, free = free, item
                top.value = top.value.slice(item.value)
                free.value = null_

            elif instruction.code == OpCode.Call:
                item2, top = top, top.prev
                item2.prev, free = free, item2
                vector, item2.value = item2.value, null_
                args = []
                for i in range((<InstructionInt>instruction).value - 1):
                    item, top = top, top.prev
                    item.prev, free = free, item
                    args.append(item.value)
                    item.value = null_
                args.append(top.value)
                args.reverse()
                values = []
                if vector.objects:
                    for function in vector.objects:
                        if callable(function):
                            try:
                                if hasattr(function, 'state_transformer') and function.state_transformer:
                                    values.append(function(context.state, *args))
                                else:
                                    values.append(function(*args))
                            except Exception as exc:
                                context.errors.add(f"Error calling function {function.__name__}\n{str(exc)}")
                        else:
                            raise NotImplementedError()
                top.value = Vector._compose(values)

            elif instruction.code == OpCode.Tag:
                name = (<InstructionStr>instruction).value
                for node in top.value.objects:
                    if isinstance(node, Node):
                        (<Node>node)._tags.add(name)

            elif instruction.code == OpCode.Attribute:
                item, top = top, top.prev
                item.prev, free = free, item
                if top.value.objects is not None:
                    for node in top.value.objects:
                        if isinstance(node, Node):
                            if item.value.length:
                                (<Node>node)._attributes[(<InstructionStr>instruction).value] = item.value
                            elif (<InstructionStr>instruction).value in (<Node>node)._attributes:
                                del (<Node>node)._attributes[(<InstructionStr>instruction).value]
                free.value = null_

            elif instruction.code == OpCode.Append:
                item, top = top, top.prev
                item.prev, free = free, item
                if top.value.objects is not None and item.value.objects is not None:
                    n = top.value.length - 1
                    for i, node in enumerate(top.value.objects):
                        if isinstance(node, Node):
                            if i == n:
                                for child in item.value.objects:
                                    if isinstance(child, Node):
                                        (<Node>node).append(<Node>child)
                            else:
                                for child in item.value.objects:
                                    if isinstance(child, Node):
                                        (<Node>node).append((<Node>child).copy())
                free.value = null_

            elif instruction.code == OpCode.Prepend:
                item, top = top, top.prev
                item.prev, free = free, item
                if top.value.objects is not None and item.value.objects is not None:
                    n = top.value.length - 1
                    for i, node in enumerate(top.value.objects):
                        if isinstance(node, Node):
                            if i == n:
                                for child in reversed(item.value.objects):
                                    if isinstance(child, Node):
                                        (<Node>node).insert(<Node>child)
                            else:
                                for child in reversed(item.value.objects):
                                    if isinstance(child, Node):
                                        (<Node>node).insert((<Node>child).copy())
                free.value = null_

            elif instruction.code == OpCode.Compose:
                values = []
                for i in range((<InstructionInt>instruction).value - 1):
                    item, top = top, top.prev
                    item.prev, free = free, item
                    values.append(item.value)
                    free.value = null_
                values.append(top.value)
                values.reverse()
                top.value = Vector._compose(values)

            elif instruction.code == OpCode.BeginFor:
                item, top = top, top.prev
                item.prev, free = free, item
                if loop_source is not None:
                    loop_sources.append(loop_source)
                loop_source = LoopSource.__new__(LoopSource)
                loop_source.source = item.value
                loop_source.position = 0
                loop_source.iterations = 0
                loop_sources.append(loop_source)
                item.value = null_
                scopes.append({})

            elif instruction.code == OpCode.Next:
                if loop_source.position >= loop_source.source.length:
                    pc += (<InstructionJump>instruction).offset
                else:
                    scope = <dict>scopes[len(scopes)-1]
                    names = (<InstructionJumpTuple>instruction).value
                    for i, name in enumerate(names):
                        scope[name] = loop_source.source.item(loop_source.position + i)
                    loop_source.position += len(names)
                    loop_source.iterations += 1

            elif instruction.code == OpCode.EndFor:
                scopes.pop()
                if loop_source.iterations:
                    values = []
                    for i in range(loop_source.iterations - 1):
                        item, top = top, top.prev
                        item.prev, free = free, item
                        values.append(item.value)
                        free.value = null_
                    values.append(top.value)
                    values.reverse()
                    top.value = Vector._compose(values)
                else:
                    if free is None:
                        free = StackItem.__new__(StackItem)
                    item, free = free, free.prev
                    item.value = null_
                    item.prev, top = top, item
                loop_source = loop_sources.pop() if loop_sources else None

            elif instruction.code == OpCode.BeginScope:
                scopes.append({})

            elif instruction.code == OpCode.EndScope:
                scopes.pop()

            elif instruction.code == OpCode.Let:
                item, top = top, top.prev
                item.prev, free = free, item
                names = (<InstructionTuple>instruction).value
                n = len(names)
                scope = <dict>scopes[len(scopes)-1]
                if n == 1:
                    scope[names[0]] = item.value
                else:
                    for i in range(n):
                        scope[names[i]] = item.value.item(i)
                free.value = null_

            elif instruction.code == OpCode.AppendRoot:
                item, top = top, top.prev
                item.prev, free = free, item
                if item.value.objects is not None:
                    for value in item.value.objects:
                        if isinstance(value, Node):
                            if (<Node>value)._parent is None:
                                context.graph.append(<Node>value)
                free.value = null_

            else:
                raise ValueError(f"Unrecognised instruction: {instruction}")

        self.free = free
        return top.value if top is not None else None


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
        if key in self._state:
            return self._state[key]
        return null_

    cdef void set_item(self, Vector key, Vector value):
        cdef Vector current = self.get_item(key)
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
    def __cinit__(self, dict variables=None, StateDict state=None, Node graph=None, dict pragmas=None, str path=None, Context parent=None):
        self.variables = variables if variables is not None else {}
        self.state = state
        self.graph = graph if graph is not None else Node.__new__(Node, 'root')
        self.pragmas = pragmas if pragmas is not None else {}
        self.path = path
        self.parent = parent
        self.unbound = set()
        self.errors = set()
        self.logs = set()
