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
    cdef readonly int64_t size
    cdef readonly int64_t max_size

    cpdef VectorStack copy(self)
    cpdef void drop(self, int64_t count=?)
    cpdef void push(self, Vector vector)
    cpdef Vector pop(self)
    cpdef tuple pop_tuple(self, int64_t count)
    cpdef list pop_list(self, int64_t count)
    cpdef dict pop_dict(self, tuple keys)
    cpdef Vector pop_composed(self, int64_t count)
    cpdef Vector peek(self)
    cpdef Vector peek_at(self, int64_t offset)
    cpdef void poke(self, Vector vector)
    cpdef void poke_at(self, int64_t offset, Vector vector)


cdef class Program:
    cdef readonly dict pragmas
    cdef readonly list instructions
    cdef bint linked
    cdef readonly object path
    cdef readonly object top
    cdef readonly tuple initial_lnames
    cdef readonly VectorStack stack
    cdef readonly VectorStack lnames
    cdef readonly set compiler_errors
    cdef int64_t next_label
    cdef bint simplify

    cpdef Program link(self)
    cpdef Program optimize(self)
    cpdef void use_simplifier(self, bint simplify)
    cpdef int64_t new_label(self)
    cpdef Program dup(self)
    cpdef Program drop(self, int64_t count=?)
    cpdef Program label(self, int64_t label)
    cpdef Program jump(self, int64_t label)
    cpdef Program branch_true(self, int64_t label)
    cpdef Program branch_false(self, int64_t label)
    cpdef Program import_(self, tuple names)
    cpdef Program literal(self, value)
    cpdef Program local_push(self, int64_t count)
    cpdef Program local_load(self, int64_t offset)
    cpdef Program local_drop(self, int64_t count)
    cpdef Program lookup(self)
    cpdef Program lookup_literal(self, Vector value)
    cpdef Program range(self)
    cpdef Program neg(self)
    cpdef Program pos(self)
    cpdef Program ceil(self)
    cpdef Program floor(self)
    cpdef Program fract(self)
    cpdef Program not_(self)
    cpdef Program add(self)
    cpdef Program sub(self)
    cpdef Program mul(self)
    cpdef Program mul_add(self)
    cpdef Program truediv(self)
    cpdef Program floordiv(self)
    cpdef Program mod(self)
    cpdef Program pow(self)
    cpdef Program eq(self)
    cpdef Program ne(self)
    cpdef Program gt(self)
    cpdef Program lt(self)
    cpdef Program ge(self)
    cpdef Program le(self)
    cpdef Program xor(self)
    cpdef Program slice(self)
    cpdef Program slice_literal(self, Vector value)
    cpdef Program call(self, int64_t count, tuple names=?)
    cpdef Program call_fast(self, function, int64_t count)
    cpdef Program tag(self, str name)
    cpdef Program attribute(self, str name)
    cpdef Program append(self, int64_t count=?)
    cpdef Program compose(self, int64_t count)
    cpdef Program begin_for(self, int64_t count)
    cpdef Program next(self, int64_t label)
    cpdef Program end_for(self)
    cpdef Program export(self, str name)
    cpdef Program func(self, int64_t label, str name, tuple parameters, int64_t ncaptures=?)
    cpdef Program exit(self)
    cdef void _execute(self, Context context, int64_t pc, bint record_stats)
