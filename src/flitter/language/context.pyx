# cython: language_level=3, profile=False

"""
Flitter language context
"""


import cython

from ..model cimport Vector, null_


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
    def __init__(self, dict variables=None, StateDict state=None, Node graph=None, dict pragmas=None, object path=None, Context parent=None):
        self.variables = variables if variables is not None else {}
        self.state = state
        self.graph = graph if graph is not None else Node('root')
        self.pragmas = pragmas if pragmas is not None else {}
        self.path = path
        self.parent = parent
        self.unbound = None
        self.errors = set()
        self.logs = set()
