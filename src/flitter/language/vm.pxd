# cython: language_level=3, profile=False

from cpython cimport PyObject

from libc.stdint cimport int64_t

from ..model cimport Vector, Context


cdef dict static_builtins
cdef dict dynamic_builtins
cdef dict builtins


cdef class VectorStack:
    cdef PyObject** vectors
    cdef int64_t top
    cdef readonly int size

    cpdef VectorStack copy(self)
    cpdef void drop(self, int count=?)
    cpdef void push(self, Vector vector)
    cpdef Vector pop(self)
    cpdef tuple pop_tuple(self, int count)
    cpdef list pop_list(self, int count)
    cpdef dict pop_dict(self, tuple keys)
    cpdef Vector pop_composed(self, int count)
    cpdef Vector peek(self)
    cpdef Vector peek_at(self, int offset)
    cpdef void poke(self, Vector vector)
    cpdef void poke_at(self, int offset, Vector vector)


cdef class Program:
    cdef readonly list instructions
    cdef bint linked
    cdef readonly object path
    cdef readonly object top
    cdef readonly tuple initial_lnames
    cdef readonly VectorStack stack
    cdef readonly VectorStack lnames
    cdef int64_t next_label

    cpdef void link(self)
    cpdef void optimize(self)
    cpdef int new_label(self)
    cpdef void dup(self)
    cpdef void drop(self, int count=?)
    cpdef void label(self, int label)
    cpdef void jump(self, int label)
    cpdef void branch_true(self, int label)
    cpdef void branch_false(self, int label)
    cpdef void pragma(self, str name)
    cpdef void import_(self, tuple names)
    cpdef void literal(self, value)
    cpdef void local_push(self, int count)
    cpdef void local_load(self, int offset)
    cpdef void local_drop(self, int count)
    cpdef void lookup(self)
    cpdef void lookup_literal(self, Vector value)
    cpdef void range(self)
    cpdef void neg(self)
    cpdef void pos(self)
    cpdef void ceil(self)
    cpdef void floor(self)
    cpdef void fract(self)
    cpdef void not_(self)
    cpdef void add(self)
    cpdef void sub(self)
    cpdef void mul(self)
    cpdef void mul_add(self)
    cpdef void truediv(self)
    cpdef void floordiv(self)
    cpdef void mod(self)
    cpdef void pow(self)
    cpdef void eq(self)
    cpdef void ne(self)
    cpdef void gt(self)
    cpdef void lt(self)
    cpdef void ge(self)
    cpdef void le(self)
    cpdef void xor(self)
    cpdef void slice(self)
    cpdef void slice_literal(self, Vector value)
    cpdef void call(self, int count, tuple names=?)
    cpdef void call_fast(self, function, int count)
    cpdef void tag(self, str name)
    cpdef void attribute(self, str name)
    cpdef void append(self, int count=?)
    cpdef void compose(self, int count)
    cpdef void begin_for(self)
    cpdef void next(self, int count, int label)
    cpdef void end_for_compose(self)
    cpdef void store_global(self, str name)
    cpdef void func(self, int label, str name, tuple parameters, int ncaptures=?)
    cpdef void exit(self)
    cdef void _execute(self, Context context, int pc, bint record_stats)
