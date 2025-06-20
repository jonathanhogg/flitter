# cython: wraparound=False, boundscheck=False

"""
Flitter language stack-based virtual machine
"""

from loguru import logger

from .. import name_patch
from ..cache import SharedCache
from .functions import STATIC_FUNCTIONS, DYNAMIC_FUNCTIONS
from ..model cimport StateDict, null_, true_, false_, inf_, nan_
from ..timer cimport perf_counter
from .noise import NOISE_FUNCTIONS

from libc.math cimport floor as c_floor
from libc.stdint cimport int64_t
from cpython cimport PyObject, Py_INCREF, Py_DECREF
from cpython.dict cimport PyDict_New, PyDict_GetItem, PyDict_SetItem, PyDict_DelItem, PyDict_Copy
from cpython.float cimport PyFloat_FromDouble
from cpython.list cimport PyList_New, PyList_GET_ITEM, PyList_SET_ITEM
from cpython.mem cimport PyMem_Malloc, PyMem_Free, PyMem_Realloc
from cpython.object cimport PyObject_Call, PyObject_CallObject, PyObject_HasAttrString
from cpython.ref cimport Py_REFCNT
from cpython.set cimport PySet_Add
from cpython.tuple cimport PyTuple_New, PyTuple_GET_ITEM, PyTuple_SET_ITEM, PyTuple_GET_SIZE, PyTuple_GetSlice


cdef extern from "Python.h":
    object PyObject_CallOneArg(object callable_object, object arg)


logger = name_patch(logger, __name__)

cdef const char* ContextFunc = "context_func\0"

cdef int64_t DEFAULT_STACK_SIZE = 32
cdef int64_t MAX_STACK_SIZE = 1 << 20
cdef int64_t MAX_CALL_DEPTH = 500

cdef dict dynamic_builtins = DYNAMIC_FUNCTIONS
cdef dict static_builtins = {
    'true': true_,
    'false': false_,
    'null': null_,
    'inf': inf_,
    'nan': nan_,
}
static_builtins.update(STATIC_FUNCTIONS)
static_builtins.update(NOISE_FUNCTIONS)

all_builtins = {}
all_builtins.update(dynamic_builtins)
all_builtins.update(static_builtins)

cdef int64_t* StatsCount = NULL
cdef double* StatsDuration = NULL
cdef double CallOutDuration = 0
cdef int64_t CallOutCount = 0


cdef dict OpCodeNames = {
    OpCode.Add: 'Add',
    OpCode.Append: 'Append',
    OpCode.Attributes: 'Attributes',
    OpCode.BeginFor: 'BeginFor',
    OpCode.BranchFalse: 'BranchFalse',
    OpCode.BranchTrue: 'BranchTrue',
    OpCode.Call: 'Call',
    OpCode.CallFast: 'CallFast',
    OpCode.Ceil: 'Ceil',
    OpCode.Compose: 'Compose',
    OpCode.Contains: 'Contains',
    OpCode.Drop: 'Drop',
    OpCode.Dup: 'Dup',
    OpCode.EndFor: 'EndFor',
    OpCode.Eq: 'Eq',
    OpCode.Floor: 'Floor',
    OpCode.FloorDiv: 'FloorDiv',
    OpCode.Fract: 'Fract',
    OpCode.Func: 'Func',
    OpCode.Exit: 'Exit',
    OpCode.Ge: 'Ge',
    OpCode.Gt: 'Gt',
    OpCode.Import: 'Import',
    OpCode.IndexLiteral: 'IndexLiteral',
    OpCode.Jump: 'Jump',
    OpCode.Label: 'Label',
    OpCode.Le: 'Le',
    OpCode.Literal: 'Literal',
    OpCode.LiteralNode: 'LiteralNode',
    OpCode.LiteralNodes: 'LiteralNodes',
    OpCode.LocalDrop: 'LocalDrop',
    OpCode.LocalLoad: 'LocalLoad',
    OpCode.LocalPush: 'LocalPush',
    OpCode.Lookup: 'Lookup',
    OpCode.LookupLiteral: 'LookupLiteral',
    OpCode.Lt: 'Lt',
    OpCode.Mod: 'Mod',
    OpCode.Mul: 'Mul',
    OpCode.MulAdd: 'MulAdd',
    OpCode.Ne: 'Ne',
    OpCode.Neg: 'Neg',
    OpCode.Next: 'Next',
    OpCode.Not: 'Not',
    OpCode.Pos: 'Pos',
    OpCode.Pow: 'Pow',
    OpCode.Range: 'Range',
    OpCode.Slice: 'Slice',
    OpCode.SliceLiteral: 'SliceLiteral',
    OpCode.Export: 'Export',
    OpCode.Sub: 'Sub',
    OpCode.Tag: 'Tag',
    OpCode.TrueDiv: 'TrueDiv',
    OpCode.Xor: 'Xor',
}


cdef initialize_stats():
    global StatsCount, StatsDuration
    cdef int64_t n = OpCode.MAX + 1
    StatsCount = <int64_t*>PyMem_Malloc(n * sizeof(int64_t))
    StatsDuration = <double*>PyMem_Malloc(n * sizeof(double))
    cdef int64_t i
    for i in range(n):
        StatsCount[i] = 0
        StatsDuration[i] = 0

initialize_stats()


cdef class Instruction:
    def __init__(self, OpCode code):
        self.code = code

    def __str__(self):
        return OpCodeNames[self.code]


cdef class InstructionVector(Instruction):
    def __init__(self, OpCode code, Vector value):
        super().__init__(code)
        self.value = value

    def __str__(self):
        return f'{OpCodeNames[self.code]} {self.value!r}'


cdef class InstructionNode(Instruction):
    def __init__(self, OpCode code, Node value):
        super().__init__(code)
        self.value = value

    def __str__(self):
        return f'{OpCodeNames[self.code]} {self.value!r}'


cdef class InstructionStr(Instruction):
    def __init__(self, OpCode code, str value):
        super().__init__(code)
        self.value = value

    def __str__(self):
        return f'{OpCodeNames[self.code]} {self.value!r}'


cdef class InstructionTuple(Instruction):
    def __init__(self, OpCode code, tuple value):
        super().__init__(code)
        self.value = value

    def __str__(self):
        return f'{OpCodeNames[self.code]} {self.value!r}'


cdef class InstructionInt(Instruction):
    def __init__(self, OpCode code, int64_t value):
        super().__init__(code)
        self.value = value

    def __str__(self):
        return f'{OpCodeNames[self.code]} {self.value!r}'


cdef class InstructionIntTuple(Instruction):
    def __init__(self, OpCode code, int64_t ivalue, tuple tvalue):
        super().__init__(code)
        self.ivalue = ivalue
        self.tvalue = tvalue

    def __str__(self):
        return f'{OpCodeNames[self.code]} {self.ivalue!r} {self.tvalue!r}'


cdef class InstructionLabel(Instruction):
    def __init__(self, int64_t label):
        super().__init__(OpCode.Label)
        self.label = label

    def __str__(self):
        return f'.L{self.label}'


cdef class InstructionJump(Instruction):
    def __init__(self, OpCode code, int64_t label):
        super().__init__(code)
        self.label = label

    def __str__(self):
        if self.offset:
            return f'{OpCodeNames[self.code]} .L{self.label} ({self.offset:+d})'
        return f'{OpCodeNames[self.code]} .L{self.label}'


cdef class InstructionFunc(InstructionJump):
    def __init__(self, OpCode code, int64_t label, str name, tuple parameters, int64_t ncaptures):
        super().__init__(code, label)
        self.name = name
        self.parameters = parameters
        self.ncaptures = ncaptures

    def __str__(self):
        return f'{OpCodeNames[self.code]} L{self.label!r} {self.name!r} {self.parameters!r} {self.ncaptures!r}'


cdef class InstructionObjectInt(Instruction):
    def __init__(self, OpCode code, object obj, int64_t value):
        super().__init__(code)
        self.obj = obj
        self.value = value

    def __str__(self):
        return f'{OpCodeNames[self.code]} {self.obj!r} {self.value!r}'


class StackError(Exception):
    pass


class StackUnderflow(StackError):
    pass


class StackOverflow(StackError):
    pass


cdef class VectorStack:
    def __cinit__(self, int64_t size=DEFAULT_STACK_SIZE, int64_t max_size=MAX_STACK_SIZE):
        self.vectors = <PyObject**>PyMem_Malloc(sizeof(PyObject*) * size)
        if self.vectors == NULL:
            raise MemoryError()
        self.size = size
        self.max_size = max_size
        self.top = -1

    def __dealloc__(self):
        cdef int64_t i
        for i in range(self.top+1):
            Py_DECREF(<Vector>self.vectors[i])
            self.vectors[i] = NULL
        PyMem_Free(self.vectors)

    def __len__(self):
        return self.top + 1

    def __str__(self):
        cdef list items = []
        cdef int64_t i
        for i in range(self.top+1):
            items.append(f"#{self.top+1-i}: {<Vector>self.vectors[i]!r}")
        return '\n'.join(items)

    cpdef VectorStack copy(self):
        cdef VectorStack new_stack = VectorStack.__new__(VectorStack, self.size)
        cdef int64_t i
        cdef Vector value
        for i in range(self.top+1):
            value = (<Vector>self.vectors[i]).copy()
            Py_INCREF(value)
            new_stack.vectors[i] = <PyObject*>value
        new_stack.top = self.top
        return new_stack

    cpdef void drop(self, int64_t count=1):
        drop(self, count)

    cpdef void push(self, Vector vector):
        push(self, vector)

    cpdef Vector pop(self):
        return pop(self)

    cpdef tuple pop_tuple(self, int64_t count):
        return pop_tuple(self, count)

    cpdef list pop_list(self, int64_t count):
        return pop_list(self, count)

    cpdef dict pop_dict(self, tuple keys):
        return pop_dict(self, keys)

    cpdef Vector pop_composed(self, int64_t count):
        return pop_composed(self, count)

    cpdef Vector peek(self):
        return peek(self)

    cpdef Vector peek_at(self, int64_t offset):
        return peek_at(self, offset)

    cpdef void poke(self, Vector vector):
        poke(self, vector)

    cpdef void poke_at(self, int64_t offset, Vector vector):
        poke_at(self, offset, vector)

cdef int64_t increase(VectorStack stack) except 0:
    if stack.size == stack.max_size:
        raise StackOverflow(f"Stack cannot grow beyond {stack.max_size} items")
    cdef int64_t new_size = min(stack.size * 2, stack.max_size)
    stack.vectors = <PyObject**>PyMem_Realloc(stack.vectors, sizeof(PyObject*) * new_size)
    if stack.vectors == NULL:
        raise MemoryError()
    stack.size = new_size
    return new_size

cdef inline void drop(VectorStack stack, int64_t n):
    if stack.top -n < -1:
        raise StackUnderflow(f"Cannot drop {n} items from {stack.top+1} depth stack")
    stack.top -= n
    cdef int64_t i
    for i in range(1, n+1):
        Py_DECREF(<Vector>stack.vectors[stack.top+i])
        stack.vectors[stack.top+i] = NULL

cdef inline int64_t push(VectorStack stack, Vector vector) except 0:
    assert vector is not None
    if stack.top + 1 == stack.size:
        increase(stack)
    stack.top += 1
    Py_INCREF(vector)
    stack.vectors[stack.top] = <PyObject*>vector
    return stack.size

cdef inline Vector pop(VectorStack stack):
    if stack.top == -1:
        raise StackUnderflow("Stack empty")
    cdef Vector vector = <Vector>stack.vectors[stack.top]
    stack.vectors[stack.top] = NULL
    stack.top -= 1
    Py_DECREF(vector)
    return vector

cdef inline tuple pop_tuple(VectorStack stack, int64_t n):
    if n == 0:
        return ()
    if stack.top - n < -1:
        raise StackUnderflow(f"Cannot pop {n} items from {stack.top+1} depth stack")
    cdef tuple t = PyTuple_New(n)
    stack.top -= n
    cdef PyObject* ptr
    cdef int64_t i, base=stack.top+1
    for i in range(n):
        ptr = stack.vectors[base+i]
        PyTuple_SET_ITEM(t, i, <Vector>ptr)
        stack.vectors[base+i] = NULL
    return t

cdef inline list pop_list(VectorStack stack, int64_t n):
    if stack.top - n < -1:
        raise StackUnderflow(f"Cannot pop {n} items from {stack.top+1} depth stack")
    cdef list t = PyList_New(n)
    stack.top -= n
    cdef PyObject* ptr
    cdef int64_t i, base=stack.top+1
    for i in range(n):
        ptr = stack.vectors[base+i]
        PyList_SET_ITEM(t, i, <Vector>ptr)
        stack.vectors[base+i] = NULL
    return t

cdef inline dict pop_dict(VectorStack stack, tuple keys):
    cdef int64_t n = len(keys)
    if stack.top - n < -1:
        raise StackUnderflow(f"Cannot pop {n} items from {stack.top+1} depth stack")
    cdef dict t = {}
    stack.top -= n
    cdef PyObject* ptr
    cdef int64_t i, base=stack.top+1
    for i in range(n):
        ptr = stack.vectors[base+i]
        PyDict_SetItem(t, <object>PyTuple_GET_ITEM(keys, i), <Vector>ptr)
        stack.vectors[base+i] = NULL
        Py_DECREF(<Vector>ptr)
    return t

cdef inline Vector pop_composed(VectorStack stack, int64_t m):
    if stack.top - m < -1:
        raise StackUnderflow(f"Cannot pop {m} items from {stack.top+1} depth stack")
    if m == 1:
        return pop(stack)
    if m == 0:
        return null_
    stack.top -= m
    cdef int64_t i, j=0, k, n=0, base=stack.top+1
    cdef bint numeric = True
    cdef PyObject* ptr
    for i in range(base, base+m):
        ptr = stack.vectors[i]
        if (<Vector>ptr).objects is not None:
            numeric = False
        n += (<Vector>ptr).length
    if n == 0:
        for i in range(base, base+m):
            ptr = stack.vectors[i]
            Py_DECREF(<object>ptr)
            stack.vectors[i] = NULL
        return null_
    cdef Vector result = Vector.__new__(Vector)
    if numeric:
        result.allocate_numbers(n)
        for i in range(base, base+m):
            ptr = stack.vectors[i]
            for k in range((<Vector>ptr).length):
                result.numbers[j] = (<Vector>ptr).numbers[k]
                j += 1
            Py_DECREF(<object>ptr)
            stack.vectors[i] = NULL
        return result
    cdef tuple vobjects, objects=PyTuple_New(n)
    cdef PyObject* itemptr
    cdef object obj
    for i in range(base, base+m):
        ptr = stack.vectors[i]
        if (<Vector>ptr).objects is None:
            for k in range((<Vector>ptr).length):
                obj = PyFloat_FromDouble((<Vector>ptr).numbers[k])
                Py_INCREF(obj)
                PyTuple_SET_ITEM(objects, j, obj)
                j += 1
        else:
            vobjects = (<Vector>ptr).objects
            for k in range((<Vector>ptr).length):
                itemptr = PyTuple_GET_ITEM(vobjects, k)
                Py_INCREF(<object>itemptr)
                PyTuple_SET_ITEM(objects, j, <object>itemptr)
                j += 1
        Py_DECREF(<object>ptr)
        stack.vectors[i] = NULL
    result.objects = objects
    result.length = n
    return result

cdef inline Vector peek(VectorStack stack):
    if stack.top == -1:
        raise StackUnderflow("Stack empty")
    return <Vector>stack.vectors[stack.top]

cdef inline Vector peek_at(VectorStack stack, int64_t offset):
    if stack.top - offset < 0:
        raise StackUnderflow(f"Cannot peek {offset} items below top of {stack.top+1} item stack")
    return <Vector>stack.vectors[stack.top-offset]

cdef inline void poke(VectorStack stack, Vector vector):
    assert vector is not None
    if stack.top == -1:
        raise StackUnderflow("Stack empty")
    Py_DECREF(<Vector>stack.vectors[stack.top])
    Py_INCREF(vector)
    stack.vectors[stack.top] = <PyObject*>vector

cdef inline void poke_at(VectorStack stack, int64_t offset, Vector vector):
    assert vector is not None
    if stack.top - offset < 0:
        raise StackUnderflow(f"Cannot poke {offset} items below top of {stack.top+1} item stack")
    Py_DECREF(<Vector>stack.vectors[stack.top-offset])
    Py_INCREF(vector)
    stack.vectors[stack.top-offset] = <PyObject*>vector


cdef class Function:
    cdef uint64_t hash(self):
        if self._hash == 0:
            self._hash = <uint64_t>(id(self.program) ^ self.address ^ hash(self.defaults) ^ hash(self.captures))
        return self._hash

    def __hash__(self):
        return self.hash()

    def __repr__(self):
        return f'<Function: {self.__name__}>'

    def __str__(self):
        return self.__name__

    def __call__(self, Context context, *args, **kwargs):
        if self.call_depth == MAX_CALL_DEPTH:
            PySet_Add(context.errors, f"Recursion depth exceeded for func '{self.__name__}'")
            return null_
        cdef int64_t i, lnames_top, stack_top, k=PyTuple_GET_SIZE(self.captures), m=PyTuple_GET_SIZE(self.parameters), n=PyTuple_GET_SIZE(args)
        cdef PyObject* objptr
        if context.state is None:
            context.state = StateDict()
        cdef VectorStack lnames = context.lnames
        if lnames is None:
            lnames = context.lnames = VectorStack()
        lnames_top = lnames.top
        cdef VectorStack stack = context.stack
        if stack is None:
            stack = context.stack = VectorStack()
        stack_top = stack.top
        for i in range(k):
            push(lnames, <Vector>PyTuple_GET_ITEM(self.captures, i))
        push(lnames, self.vself)
        for i in range(m):
            if i < n:
                push(lnames, <Vector>PyTuple_GET_ITEM(args, i))
            elif kwargs is not None and (objptr := PyDict_GetItem(kwargs, <object>PyTuple_GET_ITEM(self.parameters, i))) != NULL:
                push(lnames, <Vector>objptr)
            else:
                push(lnames, <Vector>PyTuple_GET_ITEM(self.defaults, i))
        self.call_depth += 1
        try:
            self.program._execute(context, self.address, self.record_stats)
        finally:
            self.call_depth -= 1
        drop(lnames, k + 1 + m)
        cdef Vector result = pop(stack)
        assert stack.top == stack_top, "Bad function return stack"
        assert lnames.top == lnames_top, "Bad function return lnames"
        return result

    cdef Vector call_one_fast(self, Context context, Vector arg):
        cdef int64_t i, k=PyTuple_GET_SIZE(self.captures), m=PyTuple_GET_SIZE(self.parameters)
        cdef VectorStack lnames = context.lnames
        cdef VectorStack stack = context.stack
        for i in range(k):
            push(lnames, <Vector>PyTuple_GET_ITEM(self.captures, i))
        push(lnames, self.vself)
        if m:
            push(lnames, arg)
            for i in range(1, m):
                push(lnames, <Vector>PyTuple_GET_ITEM(self.defaults, i))
        self.program._execute(context, self.address, self.record_stats)
        drop(lnames, k + 1 + m)
        return pop(stack)


cdef class LoopSource:
    cdef Vector source
    cdef int64_t stride
    cdef int64_t position
    cdef int64_t iterations


def log_vm_stats():
    cdef list stats = []
    cdef double duration, total=0
    cdef int64_t count
    cdef int64_t i
    cdef double start, end, overhead, per_execution
    start = perf_counter()
    for count in range(10000):
        end = perf_counter()
        StatsCount[<int64_t>OpCode.MAX] += 1
        StatsDuration[<int64_t>OpCode.MAX] += end
    overhead = (end - start) / 10000
    if CallOutCount:
        duration = max(0, CallOutDuration - CallOutCount*overhead)
        stats.append((duration, CallOutCount, '(native funcs)'))
        total += duration
    for i in range(OpCode.MAX):
        if StatsCount[i]:
            duration = max(0, StatsDuration[i] - StatsCount[i]*overhead)
            stats.append((duration, StatsCount[i], OpCodeNames[i]))
            total += duration
    stats.sort(reverse=True)
    cdef str name
    logger.info("VM execution statistics:")
    for duration, count, name in stats:
        per_execution = duration / count * 1e6
        if per_execution < 1:
            logger.info("- {:15s} {:9d} x {:8.0f}ns = {:7.3f}s ({:4.1f}%)", name, count, per_execution*1000,
                        duration, 100*duration/total)
        else:
            logger.info("- {:15s} {:9d} x {:8.3f}µs = {:7.3f}s ({:4.1f}%)", name, count, per_execution,
                        duration, 100*duration/total)


cdef inline void call_helper(Context context, VectorStack stack, object function, tuple args, dict kwargs, bint record_stats, double* duration):
    global CallOutDuration, CallOutCount
    cdef int64_t i, n=PyTuple_GET_SIZE(args)
    cdef double call_duration = 0
    cdef tuple context_args
    cdef bint is_func = type(function) is Function
    if is_func or PyObject_HasAttrString(function, ContextFunc):
        context_args = PyTuple_New(n+1)
        Py_INCREF(context)
        PyTuple_SET_ITEM(context_args, 0, context)
        for i in range(n):
            obj = PyTuple_GET_ITEM(args, i)
            Py_INCREF(<object>obj)
            PyTuple_SET_ITEM(context_args, i+1, <object>obj)
        args = context_args
    elif not callable(function):
        PySet_Add(context.errors, f"{function!r} is not callable")
        push(stack, null_)
        return
    if record_stats:
        call_duration = -perf_counter()
    try:
        if kwargs is None:
            push(stack, PyObject_CallObject(function, args))
        else:
            push(stack, PyObject_Call(function, args, kwargs))
    except (RecursionError, AssertionError, StackError):
        raise
    except Exception as exc:
        PySet_Add(context.errors, f"Error calling {function.__name__}: {str(exc)}")
        push(stack, null_)
    if record_stats:
        call_duration += perf_counter()
        duration[0] = duration[0] - call_duration
        if not is_func:
            CallOutDuration += call_duration
            CallOutCount += 1


cdef inline dict import_module(Context context, str filename, bint record_stats, double* duration, bint simplify):
    cdef Program program = SharedCache.get_with_root(filename, context.path).read_flitter_program(simplify=simplify)
    if program is None:
        PySet_Add(context.errors, f"Unable to import from '{filename}'")
        return None
    cdef Context import_context = context
    while import_context is not None:
        if import_context.path is program.path:
            PySet_Add(context.errors, f"Circular import of '{filename}'")
            return None
        import_context = import_context.parent
    context.errors.update(program.compiler_errors)
    cdef VectorStack stack=context.stack, lnames=context.lnames
    cdef int64_t stack_top=stack.top, lnames_top=lnames.top
    import_context = Context.__new__(Context)
    import_context.parent = context
    import_context.errors = context.errors
    import_context.logs = context.logs
    import_context.state = StateDict()
    import_context.names = {}
    import_context.exports = {}
    import_context.path = program.path
    import_context.stack = stack
    import_context.lnames = lnames
    push(stack, Vector(Node('root')))
    if record_stats:
        duration[0] += perf_counter()
    program._execute(import_context, 0, record_stats)
    if record_stats:
        duration[0] -= perf_counter()
    drop(stack, 1)
    assert stack.top == stack_top, "Bad stack"
    assert lnames.top == lnames_top, "Bad lnames"
    return import_context.exports


cdef inline void execute_append(VectorStack stack, int64_t count):
    cdef Vector nodes_vec = peek_at(stack, count)
    cdef Vector children
    cdef int64_t i, j, k, m=nodes_vec.length, n, o
    cdef tuple nodes = nodes_vec.objects
    cdef PyObject* nodeptr
    cdef PyObject* childrenptr
    cdef PyObject* objptr
    cdef tuple dest
    if nodes is not None:
        for i in range(m):
            nodeptr = PyTuple_GET_ITEM(nodes, i)
            if type(<object>nodeptr) is not Node:
                continue
            if Py_REFCNT(<Node>nodeptr) > 1:
                node = (<Node>nodeptr).copy()
                Py_DECREF(<Node>nodeptr)
                Py_INCREF(node)
                PyTuple_SET_ITEM(nodes, i, node)
                nodeptr = <PyObject*>node
            childrenptr = <PyObject*>((<Node>nodeptr)._children)
            n = o = PyTuple_GET_SIZE(<tuple>childrenptr)
            for j in range(count-1, -1, -1):
                n += peek_at(stack, j).length
            dest = PyTuple_New(n)
            for j in range(o):
                objptr = PyTuple_GET_ITEM(<tuple>childrenptr, j)
                Py_INCREF(<Node>objptr)
                PyTuple_SET_ITEM(dest, j, <Node>objptr)
            for j in range(count-1, -1, -1):
                children = peek_at(stack, j)
                childrenptr = <PyObject*>children.objects
                if childrenptr != <PyObject*>None:
                    for k in range(children.length):
                        objptr = PyTuple_GET_ITEM(<tuple>childrenptr, k)
                        if type(<object>objptr) is Node:
                            Py_INCREF(<Node>objptr)
                            PyTuple_SET_ITEM(dest, o, <Node>objptr)
                            o += 1
            if o < n:
                for k in range(o, n):
                    Py_INCREF(None)
                    PyTuple_SET_ITEM(dest, k, None)
                if o:
                    (<Node>nodeptr)._children = PyTuple_GetSlice(dest, 0, o)
                else:
                    (<Node>nodeptr)._children = ()
            else:
                (<Node>nodeptr)._children = dest
    drop(stack, count)


cdef inline execute_attributes(VectorStack stack, tuple names):
    cdef int64_t n = PyTuple_GET_SIZE(names)
    cdef Vector nodes_vec = peek_at(stack, n)
    cdef tuple nodes = nodes_vec.objects
    if nodes is None:
        drop(stack, n)
        return
    cdef Node copy
    cdef dict attributes
    cdef int64_t i, j, m=nodes_vec.length
    cdef PyObject* nodeptr
    cdef PyObject* attrptr
    cdef PyObject* nameptr
    cdef PyObject* valptr
    for i in range(m):
        nodeptr = PyTuple_GET_ITEM(nodes, i)
        if type(<object>nodeptr) is not Node:
            continue
        if Py_REFCNT(<Node>nodeptr) > 1:
            copy = (<Node>nodeptr).copy()
            Py_INCREF(copy)
            PyTuple_SET_ITEM(nodes, i, copy)
            Py_DECREF(<Node>nodeptr)
            nodeptr = <PyObject*>copy
        attrptr = <PyObject*>(<Node>nodeptr)._attributes
        if (<Node>nodeptr)._attributes_shared:
            attributes = PyDict_Copy(<object>attrptr)
            (<Node>nodeptr)._attributes = attributes
            (<Node>nodeptr)._attributes_shared = False
            attrptr = <PyObject*>(<Node>nodeptr)._attributes
        elif attrptr is <PyObject*>None:
            attributes = PyDict_New()
            attrptr = <PyObject*>attributes
            (<Node>nodeptr)._attributes = attributes
        for j in range(n):
            valptr = stack.vectors[stack.top-(n-1-j)]
            nameptr = PyTuple_GET_ITEM(names, j)
            if (<Vector>valptr).length:
                PyDict_SetItem(<object>attrptr, <object>nameptr, <Vector>valptr)
            elif PyDict_GetItem(<object>attrptr, <object>nameptr) != NULL:
                PyDict_DelItem(<object>attrptr, <object>nameptr)
    drop(stack, n)

cdef inline execute_tag(VectorStack stack, str name):
    cdef Vector nodes_vec = peek(stack)
    cdef int64_t i, m=nodes_vec.length
    cdef tuple nodes = nodes_vec.objects
    cdef PyObject* objptr
    cdef Node node
    if nodes is not None:
        for i in range(m):
            objptr = PyTuple_GET_ITEM(nodes, i)
            if type(<object>objptr) is not Node:
                continue
            if Py_REFCNT(<Node>objptr) > 1:
                node = (<Node>objptr).copy()
                Py_DECREF(<Node>objptr)
                Py_INCREF(node)
                PyTuple_SET_ITEM(nodes, i, node)
            else:
                node = <Node>objptr
            if (<Node>node)._tags is None:
                (<Node>node)._tags = set()
            PySet_Add((<Node>node)._tags, name)


cdef class Program:
    def __cinit__(self, lnames=()):
        self.pragmas = {}
        self.instructions = []
        self.initial_lnames = lnames
        self.linked = False
        self.next_label = 1
        self.simplify = True
        self.compiler_errors = set()

    def __len__(self):
        return len(self.instructions)

    def __str__(self):
        return '\n'.join(str(instruction) for instruction in self.instructions)

    cdef Instruction last_instruction(self):
        cdef int64_t n=len(self.instructions)
        return self.instructions[n-1]

    cdef Instruction pop_instruction(self):
        return self.instructions.pop()

    cdef Program push_instruction(self, Instruction instruction):
        self.instructions.append(instruction)
        return self

    def set_path(self, object path):
        self.path = path

    def set_top(self, object top):
        self.top = top

    def set_pragma(self, str name, Vector value):
        self.pragmas[name] = value
        return self

    def execute(self, Context context, list lnames=None, bint record_stats=False):
        """This is a test-harness function. Do not use."""
        assert self.initial_lnames == ()
        if not self.linked:
            self.link()
        if context.lnames is None:
            context.lnames = VectorStack()
        if lnames:
            for item in lnames:
                context.lnames.push(Vector._coerce(item))
        if context.stack is None:
            context.stack = VectorStack()
        self._execute(context, 0, record_stats)
        if lnames is not None:
            lnames.clear()
            while len(context.lnames):
                lnames.insert(0, context.lnames.pop())
        return context.stack.pop_list(len(context.stack))

    def run(self, Context context, bint record_stats=False):
        if context.stack is None:
            if self.stack is None:
                self.stack = VectorStack.__new__(VectorStack)
            context.stack = self.stack
        if context.lnames is None:
            if self.lnames is None:
                self.lnames = VectorStack.__new__(VectorStack)
            context.lnames = self.lnames
        context.path = self.path
        cdef int64_t i, n=PyTuple_GET_SIZE(self.initial_lnames)
        cdef PyObject* objptr
        for i in range(n):
            objptr = PyDict_GetItem(context.names, <object>PyTuple_GET_ITEM(self.initial_lnames, i))
            assert objptr != NULL, "Missing lname"
            push(context.lnames, <Vector>objptr)
        push(context.stack, Vector(context.root))
        try:
            self._execute(context, 0, record_stats)
        except StackOverflow:
            context.errors.add("Stack overflow in program")
            drop(context.lnames, len(context.lnames) - n)
            drop(context.stack, len(context.stack) - 1)
            return context
        assert (<VectorStack>context.lnames).top == n-1, "Bad lnames"
        drop(context.lnames, n)
        assert (<VectorStack>context.stack).top == 0, "Bad stack"
        cdef Vector result = pop(context.stack)
        assert result.length == 1 and result.objects is not None and isinstance(result.objects[0], Node), "Bad root node"
        context.root = result.objects[0]
        return context

    cpdef Program link(self):
        cdef Instruction instruction
        cdef int64_t label, address, target
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
        return self

    cpdef void use_simplifier(self, bint simplify):
        self.simplify = simplify

    cpdef int64_t new_label(self):
        cdef object label = self.next_label
        self.next_label += 1
        return label

    cpdef Program dup(self):
        self.instructions.append(Instruction(OpCode.Dup))
        return self

    cpdef Program drop(self, int64_t count=1):
        self.instructions.append(InstructionInt(OpCode.Drop, count))
        return self

    cpdef Program label(self, int64_t label):
        self.instructions.append(InstructionLabel(label))
        return self

    cpdef Program jump(self, int64_t label):
        self.instructions.append(InstructionJump(OpCode.Jump, label))
        return self

    cpdef Program branch_true(self, int64_t label):
        self.instructions.append(InstructionJump(OpCode.BranchTrue, label))
        return self

    cpdef Program branch_false(self, int64_t label):
        self.instructions.append(InstructionJump(OpCode.BranchFalse, label))
        return self

    cpdef Program import_(self, tuple names):
        self.instructions.append(InstructionTuple(OpCode.Import, names))
        return self

    cpdef Program literal(self, value):
        cdef Vector vector = Vector._coerce(value)
        cdef object obj
        if vector.objects is not None:
            if vector.length == 1:
                obj = vector.objects[0]
                if type(obj) is Node:
                    self.instructions.append(InstructionNode(OpCode.LiteralNode, <Node>obj))
                    return self
            else:
                for obj in vector.objects:
                    if type(obj) is Node:
                        self.instructions.append(InstructionTuple(OpCode.LiteralNodes, vector.objects))
                        return self
        self.instructions.append(InstructionVector(OpCode.Literal, vector))
        return self

    cpdef Program local_push(self, int64_t count):
        self.instructions.append(InstructionInt(OpCode.LocalPush, count))
        return self

    cpdef Program local_load(self, int64_t offset):
        self.instructions.append(InstructionInt(OpCode.LocalLoad, offset))
        return self

    cpdef Program local_drop(self, int64_t count):
        self.instructions.append(InstructionInt(OpCode.LocalDrop, count))
        return self

    cpdef Program lookup(self):
        self.instructions.append(Instruction(OpCode.Lookup))
        return self

    cpdef Program lookup_literal(self, Vector value):
        self.instructions.append(InstructionVector(OpCode.LookupLiteral, value))
        return self

    cpdef Program range(self):
        self.instructions.append(Instruction(OpCode.Range))
        return self

    cpdef Program neg(self):
        self.instructions.append(Instruction(OpCode.Neg))
        return self

    cpdef Program pos(self):
        self.instructions.append(Instruction(OpCode.Pos))
        return self

    cpdef Program ceil(self):
        self.instructions.append(Instruction(OpCode.Ceil))
        return self

    cpdef Program floor(self):
        self.instructions.append(Instruction(OpCode.Floor))
        return self

    cpdef Program fract(self):
        self.instructions.append(Instruction(OpCode.Fract))
        return self

    cpdef Program not_(self):
        self.instructions.append(Instruction(OpCode.Not))
        return self

    cpdef Program add(self):
        self.instructions.append(Instruction(OpCode.Add))
        return self

    cpdef Program sub(self):
        self.instructions.append(Instruction(OpCode.Sub))
        return self

    cpdef Program mul(self):
        self.instructions.append(Instruction(OpCode.Mul))
        return self

    cpdef Program mul_add(self):
        self.instructions.append(Instruction(OpCode.MulAdd))
        return self

    cpdef Program truediv(self):
        self.instructions.append(Instruction(OpCode.TrueDiv))
        return self

    cpdef Program floordiv(self):
        self.instructions.append(Instruction(OpCode.FloorDiv))
        return self

    cpdef Program mod(self):
        self.instructions.append(Instruction(OpCode.Mod))
        return self

    cpdef Program pow(self):
        self.instructions.append(Instruction(OpCode.Pow))
        return self

    cpdef Program contains(self):
        self.instructions.append(Instruction(OpCode.Contains))
        return self

    cpdef Program eq(self):
        self.instructions.append(Instruction(OpCode.Eq))
        return self

    cpdef Program ne(self):
        self.instructions.append(Instruction(OpCode.Ne))
        return self

    cpdef Program gt(self):
        self.instructions.append(Instruction(OpCode.Gt))
        return self

    cpdef Program lt(self):
        self.instructions.append(Instruction(OpCode.Lt))
        return self

    cpdef Program ge(self):
        self.instructions.append(Instruction(OpCode.Ge))
        return self

    cpdef Program le(self):
        self.instructions.append(Instruction(OpCode.Le))
        return self

    cpdef Program xor(self):
        self.instructions.append(Instruction(OpCode.Xor))
        return self

    cpdef Program slice(self):
        self.instructions.append(Instruction(OpCode.Slice))
        return self

    cpdef Program slice_literal(self, Vector value):
        if value.length == 1 and value.numbers != NULL:
            self.instructions.append(InstructionInt(OpCode.IndexLiteral, <int64_t>c_floor(value.numbers[0])))
        else:
            self.instructions.append(InstructionVector(OpCode.SliceLiteral, value))
        return self

    cpdef Program call(self, int64_t count, tuple names=None):
        self.instructions.append(InstructionIntTuple(OpCode.Call, count, names))
        return self

    cpdef Program call_fast(self, function, int64_t count):
        self.instructions.append(InstructionObjectInt(OpCode.CallFast, function, count))
        return self

    cpdef Program tag(self, str name):
        self.instructions.append(InstructionStr(OpCode.Tag, name))
        return self

    cpdef Program attributes(self, tuple names):
        self.instructions.append(InstructionTuple(OpCode.Attributes, names))
        return self

    cpdef Program append(self, int64_t count=1):
        self.instructions.append(InstructionInt(OpCode.Append, count))
        return self

    cpdef Program compose(self, int64_t count):
        self.instructions.append(InstructionInt(OpCode.Compose, count))
        return self

    cpdef Program begin_for(self, int64_t count):
        self.instructions.append(InstructionInt(OpCode.BeginFor, count))
        return self

    cpdef Program next(self, int64_t label):
        self.instructions.append(InstructionJump(OpCode.Next, label))
        return self

    cpdef Program end_for(self):
        self.instructions.append(Instruction(OpCode.EndFor))
        return self

    cpdef Program export(self, str name):
        self.instructions.append(InstructionStr(OpCode.Export, name))
        return self

    cpdef Program func(self, int64_t label, str name, tuple parameters, int64_t ncaptures=0):
        self.instructions.append(InstructionFunc(OpCode.Func, label, name, parameters, ncaptures))
        return self

    cpdef Program exit(self):
        self.instructions.append(Instruction(OpCode.Exit))
        return self

    cdef void _execute(self, Context context, int64_t pc, bint record_stats):
        global CallOutCount, CallOutDuration
        cdef VectorStack stack=context.stack, lnames=context.lnames
        cdef int64_t i, n, program_end=len(self.instructions)
        cdef dict exports=context.exports, state=context.state._state
        cdef list loop_sources=[]
        cdef LoopSource loop_source = None
        cdef double duration, call_duration

        cdef list instructions=self.instructions
        cdef Instruction instruction=None
        cdef str filename
        cdef tuple names, args, nodes
        cdef Vector r1, r2, r3
        cdef dict import_names, kwargs
        cdef Function function
        cdef PyObject* objptr

        assert self.linked, "Program has not been linked"

        try:
            while 0 <= pc < program_end:
                instruction = <Instruction>PyList_GET_ITEM(instructions, pc)
                pc += 1
                if record_stats:
                    duration = -perf_counter()

                if instruction.code == OpCode.Dup:
                    push(stack, peek(stack))

                elif instruction.code == OpCode.Drop:
                    drop(stack, (<InstructionInt>instruction).value)

                elif instruction.code == OpCode.Label:
                    pass

                elif instruction.code == OpCode.Jump:
                    pc += (<InstructionJump>instruction).offset

                elif instruction.code == OpCode.Exit:
                    pc = program_end

                elif instruction.code == OpCode.BranchTrue:
                    if pop(stack).as_bool():
                        pc += (<InstructionJump>instruction).offset

                elif instruction.code == OpCode.BranchFalse:
                    if not pop(stack).as_bool():
                        pc += (<InstructionJump>instruction).offset

                elif instruction.code == OpCode.Import:
                    filename = pop(stack).as_string()
                    names = (<InstructionTuple>instruction).value
                    n = PyTuple_GET_SIZE(names)
                    import_names = import_module(context, filename, record_stats, &duration, self.simplify)
                    if import_names is not None:
                        for i in range(n):
                            objptr = PyDict_GetItem(import_names, <object>PyTuple_GET_ITEM(names, i))
                            if objptr != NULL:
                                push(lnames, <Vector>objptr)
                            else:
                                PySet_Add(context.errors, f"Unable to import '{<str>PyTuple_GET_ITEM(names, i)}' from '{filename}'")
                                push(lnames, null_)
                    else:
                        for i in range(n):
                            push(lnames, null_)
                    filename = names = import_names = None

                elif instruction.code == OpCode.Literal:
                    push(stack, (<InstructionVector>instruction).value)

                elif instruction.code == OpCode.LiteralNode:
                    r1 = Vector.__new__(Vector)
                    r1.objects = ((<InstructionNode>instruction).value,)
                    r1.length = 1
                    push(stack, r1)
                    r1 = None

                elif instruction.code == OpCode.LiteralNodes:
                    nodes = (<InstructionTuple>instruction).value
                    n = PyTuple_GET_SIZE(nodes)
                    r1 = Vector.__new__(Vector)
                    r1.objects = PyTuple_New(n)
                    for i in range(n):
                        objptr = PyTuple_GET_ITEM(nodes, i)
                        Py_INCREF(<object>objptr)
                        PyTuple_SET_ITEM(r1.objects, i, <object>objptr)
                    r1.length = n
                    push(stack, r1)
                    r1 = nodes = None

                elif instruction.code == OpCode.LocalDrop:
                    drop(lnames, (<InstructionInt>instruction).value)

                elif instruction.code == OpCode.LocalLoad:
                    r1 = peek_at(lnames, (<InstructionInt>instruction).value)
                    push(stack, r1.copy() if r1.objects is not None else r1)
                    r1 = None

                elif instruction.code == OpCode.LocalPush:
                    r1 = pop(stack)
                    n = (<InstructionInt>instruction).value
                    if n == 1:
                        push(lnames, r1)
                    else:
                        for i in range(n):
                            push(lnames, r1.item(i))
                    r1 = None

                elif instruction.code == OpCode.Lookup:
                    objptr = PyDict_GetItem(state, peek(stack))
                    if objptr != NULL:
                        poke(stack, <Vector>objptr)
                    else:
                        poke(stack, null_)
                    objptr = NULL

                elif instruction.code == OpCode.LookupLiteral:
                    objptr = PyDict_GetItem(state, (<InstructionVector>instruction).value)
                    if objptr != NULL:
                        push(stack, <Vector>objptr)
                    else:
                        push(stack, null_)
                    objptr = NULL

                elif instruction.code == OpCode.Range:
                    r3 = pop(stack)
                    r2 = pop(stack)
                    r1 = Vector.__new__(Vector)
                    r1.fill_range(peek(stack), r2, r3)
                    poke(stack, r1)
                    r1 = r2 = r3 = None

                elif instruction.code == OpCode.Neg:
                    poke(stack, peek(stack).neg())

                elif instruction.code == OpCode.Pos:
                    if peek(stack).objects is not None:
                        poke(stack, null_)

                elif instruction.code == OpCode.Ceil:
                    poke(stack, peek(stack).ceil())

                elif instruction.code == OpCode.Floor:
                    poke(stack, peek(stack).floor())

                elif instruction.code == OpCode.Fract:
                    poke(stack, peek(stack).fract())

                elif instruction.code == OpCode.Not:
                    poke(stack, false_ if peek(stack).as_bool() else true_)

                elif instruction.code == OpCode.Add:
                    r1 = pop(stack)
                    poke(stack, peek(stack).add(r1))
                    r1 = None

                elif instruction.code == OpCode.Sub:
                    r1 = pop(stack)
                    poke(stack, peek(stack).sub(r1))
                    r1 = None

                elif instruction.code == OpCode.Mul:
                    r1 = pop(stack)
                    poke(stack, peek(stack).mul(r1))
                    r1 = None

                elif instruction.code == OpCode.MulAdd:
                    r2 = pop(stack)
                    r1 = pop(stack)
                    poke(stack, peek(stack).mul_add(r1, r2))
                    r1 = r2 = None

                elif instruction.code == OpCode.TrueDiv:
                    r1 = pop(stack)
                    poke(stack, peek(stack).truediv(r1))
                    r1 = None

                elif instruction.code == OpCode.FloorDiv:
                    r1 = pop(stack)
                    poke(stack, peek(stack).floordiv(r1))
                    r1 = None

                elif instruction.code == OpCode.Mod:
                    r1 = pop(stack)
                    poke(stack, peek(stack).mod(r1))
                    r1 = None

                elif instruction.code == OpCode.Pow:
                    r1 = pop(stack)
                    poke(stack, peek(stack).pow(r1))
                    r1 = None

                elif instruction.code == OpCode.Contains:
                    r1 = pop(stack)
                    poke(stack, r1.contains(peek(stack)))
                    r1 = None

                elif instruction.code == OpCode.Eq:
                    r1 = pop(stack)
                    poke(stack, peek(stack).eq(r1))
                    r1 = None

                elif instruction.code == OpCode.Ne:
                    r1 = pop(stack)
                    poke(stack, peek(stack).ne(r1))
                    r1 = None

                elif instruction.code == OpCode.Gt:
                    r1 = pop(stack)
                    poke(stack, peek(stack).gt(r1))
                    r1 = None

                elif instruction.code == OpCode.Lt:
                    r1 = pop(stack)
                    poke(stack, peek(stack).lt(r1))
                    r1 = None

                elif instruction.code == OpCode.Ge:
                    r1 = pop(stack)
                    poke(stack, peek(stack).ge(r1))
                    r1 = None

                elif instruction.code == OpCode.Le:
                    r1 = pop(stack)
                    poke(stack, peek(stack).le(r1))
                    r1 = None

                elif instruction.code == OpCode.Xor:
                    r2 = pop(stack)
                    r1 = peek(stack)
                    if not r1.as_bool():
                        poke(stack, r2)
                    elif r2.as_bool():
                        poke(stack, false_)
                    r1 = r2 = None

                elif instruction.code == OpCode.Slice:
                    r1 = pop(stack)
                    poke(stack, peek(stack).slice(r1))
                    r1 = None

                elif instruction.code == OpCode.SliceLiteral:
                    poke(stack, peek(stack).slice((<InstructionVector>instruction).value))

                elif instruction.code == OpCode.IndexLiteral:
                    poke(stack, peek(stack).item((<InstructionInt>instruction).value))

                elif instruction.code == OpCode.Call:
                    r1 = pop(stack)
                    if (<InstructionIntTuple>instruction).tvalue is not None:
                        kwargs = pop_dict(stack, (<InstructionIntTuple>instruction).tvalue)
                    else:
                        kwargs = None
                    n = (<InstructionIntTuple>instruction).ivalue
                    args = pop_tuple(stack, n) if n else ()
                    if r1.objects is not None:
                        for i in range(r1.length):
                            call_helper(context, stack, <object>PyTuple_GET_ITEM(r1.objects, i), args, kwargs, record_stats, &duration)
                        if r1.length > 1:
                            push(stack, pop_composed(stack, r1.length))
                    else:
                        push(stack, null_)
                    r1 = kwargs = args = None

                elif instruction.code == OpCode.CallFast:
                    n = (<InstructionObjectInt>instruction).value
                    if n == 1:
                        if record_stats:
                            call_duration = -perf_counter()
                        try:
                            poke(stack, PyObject_CallOneArg((<InstructionObjectInt>instruction).obj, peek(stack)))
                        except Exception as exc:
                            PySet_Add(context.errors, f"Error calling {(<InstructionObjectInt>instruction).obj.__name__}: {str(exc)}")
                            poke(stack, null_)
                    else:
                        args = pop_tuple(stack, n)
                        if record_stats:
                            call_duration = -perf_counter()
                        try:
                            push(stack, PyObject_CallObject((<InstructionObjectInt>instruction).obj, args))
                        except Exception as exc:
                            PySet_Add(context.errors, f"Error calling {(<InstructionObjectInt>instruction).obj.__name__}: {str(exc)}")
                            push(stack, null_)
                        args = None
                    if record_stats:
                        call_duration += perf_counter()
                        duration -= call_duration
                        CallOutDuration += call_duration
                        CallOutCount += 1

                elif instruction.code == OpCode.Func:
                    function = Function.__new__(Function)
                    function.__name__ = (<InstructionFunc>instruction).name
                    function.parameters = (<InstructionFunc>instruction).parameters
                    function.program = self
                    function.address = (<InstructionFunc>instruction).offset + pc
                    function.record_stats = record_stats
                    n = PyTuple_GET_SIZE(function.parameters)
                    function.defaults = pop_tuple(stack, n)
                    function.captures = pop_tuple(stack, (<InstructionFunc>instruction).ncaptures)
                    function.vself = Vector(function)
                    push(stack, function.vself)
                    function = None

                elif instruction.code == OpCode.Tag:
                    execute_tag(stack, (<InstructionStr>instruction).value)

                elif instruction.code == OpCode.Attributes:
                    execute_attributes(stack, (<InstructionTuple>instruction).value)

                elif instruction.code == OpCode.Append:
                    execute_append(stack, (<InstructionInt>instruction).value)

                elif instruction.code == OpCode.Compose:
                    push(stack, pop_composed(stack, (<InstructionInt>instruction).value))

                elif instruction.code == OpCode.BeginFor:
                    n = (<InstructionInt>instruction).value
                    if loop_source is not None:
                        loop_sources.append(loop_source)
                    loop_source = LoopSource.__new__(LoopSource)
                    loop_source.source = pop(stack)
                    loop_source.stride = n
                    loop_source.position = 0
                    loop_source.iterations = 0
                    for i in range(n):
                        push(lnames, null_)

                elif instruction.code == OpCode.Next:
                    if loop_source.position >= loop_source.source.length:
                        pc += (<InstructionJump>instruction).offset
                    else:
                        n = loop_source.stride
                        for i in range(n-1, -1, -1):
                            if loop_source.position >= loop_source.source.length:
                                poke_at(lnames, i, null_)
                            else:
                                poke_at(lnames, i, loop_source.source.item(loop_source.position))
                            loop_source.position += 1
                        loop_source.iterations += 1

                elif instruction.code == OpCode.EndFor:
                    drop(lnames, loop_source.stride)
                    push(stack, pop_composed(stack, loop_source.iterations))
                    if loop_sources:
                        loop_source = <LoopSource>loop_sources.pop()
                    else:
                        loop_source = None

                elif instruction.code == OpCode.Export:
                    PyDict_SetItem(exports, (<InstructionStr>instruction).value, pop(stack))

                else:
                    raise AssertionError(f"Unrecognised instruction: {instruction}")

                if record_stats:
                    duration += perf_counter()
                    StatsCount[<int64_t>instruction.code] += 1
                    StatsDuration[<int64_t>instruction.code] += duration

        except StackOverflow:
            raise

        except Exception:
            if instruction is not None:
                logger.exception("VM exception processing:\n{} <--",
                                 "\n".join(str(instruction) for instruction in self.instructions[pc-5:pc]))
            raise

        assert pc == program_end, "Jump outside of program"
