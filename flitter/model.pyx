# cython: language_level=3, profile=True

import cython
from weakref import ref as weak

from libc.math cimport isnan, floor


cdef union float_long:
    double f
    unsigned long long l


@cython.freelist(100)
cdef class VectorLike:
    cpdef VectorLike copynodes(self):
        raise NotImplementedError()

    cpdef Vector slice(self, Vector index):
        raise NotImplementedError()

    cpdef bint istrue(self):
        raise NotImplementedError()


@cython.final
cdef class Vector(VectorLike):
    @staticmethod
    def compose(*args):
        if len(args) == 1:
            return args[0]
        cdef Vector result = Vector.__new__(Vector)
        cdef Vector vector
        for arg in args:
            vector = arg
            result.values.extend(vector.values)
        return result

    @staticmethod
    def range(startv, stopv, stepv):
        cdef double start = float(startv) if startv is not None else float("nan")
        if isnan(start):
            start = 0
        cdef double stop = float(stopv) if stopv is not None else float("nan")
        if isnan(stop):
            return null_
        cdef double step = float(stepv) if stepv is not None else float("nan")
        if isnan(step):
            step = 1
        elif step == 0:
            return null_
        cdef Vector result = Vector.__new__(Vector)
        cdef int i, n = <int>floor((stop - start) / step)
        for i in range(n):
            result.values.append(start + step * i)
        return result

    def __cinit__(self, values=None):
        self.values = list(values) if values is not None else []

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        return iter(self.values)

    def __getitem__(self, int index):
        return self.values[index]

    def __bool__(self):
        return bool(self.values)

    def __float__(self):
        if len(self.values) == 1 and isinstance(self.values[0], (float, int)):
            return float(self.values[0])
        return float("nan")

    cdef unsigned long long _hash(self, bint floor_floats):
        # Compute a hash value using the SplitMix64 algorithm [http://xoshiro.di.unimi.it/splitmix64.c]
        cdef unsigned long long y, hash = 0xe220a8397b1dcdaf
        cdef float_long fl
        for value in self.values:
            if isinstance(value, str):
                # FNV-1a hash algorithm [https://en.wikipedia.org/wiki/Fowler–Noll–Vo_hash_function#FNV-1a_hash]
                y = <unsigned long long>(0xcbf29ce484222325)
                for c in <str>value:
                    y = (y ^ ord(c)) * <unsigned long long>(0x100000001b3)
            elif isinstance(value, float):
                if floor_floats:
                    y = <unsigned long long>(<long long>floor(value))
                else:
                    fl.f = value
                    y = fl.l
            elif isinstance(value, int):
                y = <unsigned long long>(<long long>value)
            else:
                raise TypeError("Unhashable value")
            hash ^= y
            hash += <unsigned long long>(0x9e3779b97f4a7c15)
            hash ^= hash >> 30
            hash *= <unsigned long long>(0xbf58476d1ce4e5b9)
            hash ^= hash >> 27
            hash *= <unsigned long long>(0x94d049bb133111eb)
            hash ^= hash >> 31
        return hash

    def __hash__(self):
        return self._hash(False)

    def __eq__(Node self, other):
        return isinstance(other, Vector) and self.compare(other) == 0

    cpdef bint istrue(self):
        for value in self.values:
            if isinstance(value, int):
                if <int>value != 0:
                    return True
            if isinstance(value, float):
                if <double>value != 0.:
                    return True
            elif isinstance(value, str):
                if <str>value != "":
                    return True
            else:
                return True
        return False

    def isinstance(self, t):
        for value in self.values:
            if not isinstance(value, t):
                return False
        return True

    cpdef VectorLike copynodes(self):
        cdef Vector result = self
        cdef Node node, parent
        cdef int i
        for i, value in enumerate(self.values):
            if isinstance(value, Node):
                node = value
                if node._parent is None:
                    if result is self:
                        result = Vector.__new__(Vector)
                        result.values.extend(self.values)
                    result.values[i] = node.copy()
        return result

    cpdef Vector withlen(self, int n, bint force_copy=False):
        cdef int m = len(self.values)
        if n == m:
            return Vector.__new__(Vector, self.values) if force_copy else self
        if m != 1:
            return null_
        cdef Vector result = Vector.__new__(Vector)
        value = self.values[0]
        for i in range(n):
            result.values.append(value)
        return result

    cpdef Vector neg(self):
        cdef Vector result = Vector.__new__(Vector)
        for value in self.values:
            result.values.append(-value)
        return result

    cpdef Vector pos(self):
        cdef Vector result = Vector.__new__(Vector)
        for value in self.values:
            result.values.append(+value)
        return result

    cpdef Vector not_(self):
        return false_ if self.istrue() else true_

    cpdef Vector add(self, Vector other):
        cdef Vector result = Vector.__new__(Vector)
        cdef int n = len(self.values), m = len(other.values)
        if n == m:
            for i in range(n):
                result.values.append(self.values[i] + other.values[i])
        elif n == 1:
            x = self.values[0]
            for i in range(m):
                result.values.append(x + other.values[i])
        elif m == 1:
            y = other.values[0]
            for i in range(n):
                result.values.append(self.values[i] + y)
        return result

    cpdef Vector sub(self, Vector other):
        cdef Vector result = Vector.__new__(Vector)
        cdef int n = len(self.values), m = len(other.values)
        if n == m:
            for i in range(n):
                result.values.append(self.values[i] - other.values[i])
        elif n == 1:
            x = self.values[0]
            for i in range(m):
                result.values.append(x - other.values[i])
        elif m == 1:
            y = other.values[0]
            for i in range(n):
                result.values.append(self.values[i] - y)
        return result

    cpdef Vector mul(self, Vector other):
        cdef Vector result = Vector.__new__(Vector)
        cdef int n = len(self.values), m = len(other.values)
        if n == m:
            for i in range(n):
                result.values.append(self.values[i] * other.values[i])
        elif n == 1:
            x = self.values[0]
            for i in range(m):
                result.values.append(x * other.values[i])
        elif m == 1:
            y = other.values[0]
            for i in range(n):
                result.values.append(self.values[i] * y)
        return result

    cpdef Vector truediv(self, Vector other):
        cdef Vector result = Vector.__new__(Vector)
        cdef int n = len(self.values), m = len(other.values)
        if n == m:
            for i in range(n):
                result.values.append(self.values[i] / other.values[i])
        elif n == 1:
            x = self.values[0]
            for i in range(m):
                result.values.append(x / other.values[i])
        elif m == 1:
            y = other.values[0]
            for i in range(n):
                result.values.append(self.values[i] / y)
        return result

    cpdef Vector floordiv(self, Vector other):
        cdef Vector result = Vector.__new__(Vector)
        cdef int n = len(self.values), m = len(other.values)
        if n == m:
            for i in range(n):
                result.values.append(self.values[i] // other.values[i])
        elif n == 1:
            x = self.values[0]
            for i in range(m):
                result.values.append(x // other.values[i])
        elif m == 1:
            y = other.values[0]
            for i in range(n):
                result.values.append(self.values[i] // y)
        return result

    cpdef Vector mod(self, Vector other):
        cdef Vector result = Vector.__new__(Vector)
        cdef int n = len(self.values), m = len(other.values)
        if n == m:
            for i in range(n):
                result.values.append(self.values[i] % other.values[i])
        elif n == 1:
            x = self.values[0]
            for i in range(m):
                result.values.append(x % other.values[i])
        elif m == 1:
            y = other.values[0]
            for i in range(n):
                result.values.append(self.values[i] % y)
        return result

    cpdef Vector pow(self, Vector other):
        cdef Vector result = Vector.__new__(Vector)
        cdef int n = len(self.values), m = len(other.values)
        if n == m:
            for i in range(n):
                result.values.append(self.values[i] ** other.values[i])
        elif n == 1:
            x = self.values[0]
            for i in range(m):
                result.values.append(x ** other.values[i])
        elif m == 1:
            y = other.values[0]
            for i in range(n):
                result.values.append(self.values[i] ** y)
        return result

    cpdef int compare(self, Vector other):
        cdef int n = len(self.values), m = len(other.values)
        for i in range(min(n, m)):
            x = self.values[i]
            y = other.values[i]
            if x < y:
                return -1
            if x > y:
                return 1
        if n < m:
            return -1
        if n > m:
            return 1
        return 0

    cpdef Vector slice(self, Vector index):
        cdef Vector result = Vector.__new__(Vector)
        cdef int j, n = len(self.values)
        for value in index.values:
            if isinstance(value, (float, int)):
                j = <int>floor(value)
                if j >= 0 and j < n:
                    result.values.append(self.values[j])
        return result

    def __repr__(self):
        return f"Vector({self.values!r})"


cdef Vector null_ = Vector.__new__(Vector)
cdef Vector true_ = Vector.__new__(Vector, (1.,))
cdef Vector false_ = Vector.__new__(Vector, (0.,))

null = null_
true = true_
false = false_


@cython.final
@cython.freelist(20)
cdef class Query:
    def __cinit__(self, str query=None, *, kind=None, tags=None):
        if query is None:
            self.kind = kind
            if tags is None:
                pass
            elif isinstance(tags, str):
                self.tags = frozenset(tags.split())
            else:
                self.tags = frozenset(tags)
            return
        cdef int i = 0, j, n = len(query)
        while n > 0 and query[n-1] in ' \t\r\n':
            n -= 1
        while i < n:
            if not self.strict and query[i] == '>':
                self.strict = True
            elif query[i] not in ' \t\r\n':
                break
            i += 1
        if i == n or query[i] in ('.', '|', '>'):
            raise ValueError("Bad query; contains empty element")
        cdef list tag_list = []
        for j in range(i, n+1):
            if j == n or query[j] in ('#', '.', '|', '>', ' '):
                if j > i:
                    if query[i] == '#':
                        if i+1 < j:
                            tag_list.append(query[i+1:j])
                    elif query[i] != '*':
                        self.kind = query[i:j]
                    i = j
                if j < n and query[j] in '.|> \t\r\n':
                    break
        if tag_list:
            self.tags = frozenset(tag_list)
        if i < n and query[i] == '.':
            self.stop = True
            i += 1
        if i < n and query[i] == '|':
            j = i + 1
            while j < n and query[j] not in '> \r\r\n':
                j += 1
            self.altquery = Query.__new__(Query, query[i+1:j])
            i = j
        if i < n:
            self.subquery = Query.__new__(Query, query[i:n])

    def __str__(self):
        cdef str text = ''
        if self.strict:
            text += '> '
        if self.kind is None and not self.tags:
            text += '*'
        else:
            if self.kind is not None:
                text += self.kind
            if self.tags:
                text += '#' + '#'.join(self.tags)
        if self.stop:
            text += '.'
        if self.altquery is not None:
            text += '|' + str(self.altquery)
        if self.subquery is not None:
            text += ' ' + str(self.subquery)
        return text

    def __repr__(self):
        return f"Query({str(self)!r})"


@cython.final
@cython.freelist(100)
cdef class Node:
    @staticmethod
    def from_dict(d):
        d = d.copy()
        cdef str kind = d.pop('_kind')
        cdef str tags = d.pop('_tags', '')
        cdef list children = d.pop('_children', [])
        cdef Node node = Node(kind, tags)
        for key, value in d.items():
            node.attributes[key] = Vector(value)
        cdef Node child
        for d in children:
            child = Node.from_dict(d)
            node.append(child)
        return node

    def __cinit__(self, str kind, tags, /):
        cdef str tag_str
        self.kind = kind
        if isinstance(tags, frozenset):
            self.tags = tags
        elif isinstance(tags, str):
            tag_str = tags
            self.tags = frozenset(tag_str.split())
        else:
            self.tags = frozenset(tags)
        self.attributes = {}

    @property
    def parent(self):
        return self._parent() if self._parent is not None else None

    @property
    def children(self):
        cdef Node node = self.first_child
        while node is not None:
            yield node
            node = node.next_sibling

    def to_dict(self):
        d = {'_kind': self.kind}
        if self.tags:
            d['_tags'] = ' '.join(self.tags)
        for key, value in self.attributes.items():
            d[key] = list(value)
        if self.first_child is not None:
            d['_children'] = [node.to_dict() for node in self.children]
        return d

    cpdef Node copy(self):
        cdef Node node = Node.__new__(Node, self.kind, self.tags)
        node.attributes = self.attributes.copy()
        cdef Node copy, child = self.first_child
        if child is not None:
            parent = weak(node)
            copy = child.copy()
            copy._parent = parent
            node.first_child = node.last_child = copy
            child = child.next_sibling
            while child is not None:
                copy = child.copy()
                copy._parent = parent
                node.last_child.next_sibling = node.last_child = copy
                child = child.next_sibling
        return node

    cpdef void append(self, Node node):
        cdef Node parent = node._parent() if node._parent is not None else None
        if parent is not None:
            parent.remove(node)
        node._parent = weak(self)
        if self.last_child is not None:
            self.last_child.next_sibling = node
            self.last_child = node
        else:
            self.first_child = self.last_child = node

    cpdef void insert(self, Node node):
        cdef Node parent = node._parent() if node._parent is not None else None
        if parent is not None:
            parent.remove(node)
        node._parent = weak(self)
        node.next_sibling = self.first_child
        self.first_child = node
        if self.last_child is None:
            self.last_child = self.first_child

    def extend(self, nodes):
        cdef Node node
        for node in nodes:
            self.append(node)

    def prepend(self, nodes):
        cdef Node node
        for node in reversed(nodes):
            self.insert(node)

    cpdef void remove(self, Node node):
        cdef Node parent = node._parent() if node._parent is not None else None
        if parent is not self:
            raise ValueError("Not a child of this node")
        cdef Node child = self.first_child, previous = None
        while child is not node:
            previous = child
            child = child.next_sibling
        if previous is None:
            self.first_child = node.next_sibling
        else:
            previous.next_sibling = node.next_sibling
        if node is self.last_child:
            self.last_child = previous
        node._parent = None
        node.next_sibling = None

    def delete(self):
        cdef Node parent = self._parent() if self._parent is not None else None
        if parent is None:
            raise TypeError("No parent")
        parent.remove(self)

    def select(self, query):
        cdef list nodes = []
        if not isinstance(query, Query):
            query = Query.__new__(Query, query)
        self._select(query, nodes, False)
        return nodes

    def select_below(self, query):
        cdef list nodes = []
        if not isinstance(query, Query):
            query = Query.__new__(Query, query)
        cdef Node node = self.first_child
        while node is not None:
            node._select(query, nodes, False)
            node = node.next_sibling
        return nodes

    cdef bint _select(self, Query query, list nodes, bint first):
        cdef frozenset tags
        cdef Query altquery = query
        cdef bint descend = not query.strict
        cdef Node node
        while altquery is not None:
            tags = altquery.tags
            if (altquery.kind is None or altquery.kind == self.kind) and \
               (tags is None or tags.issubset(self.tags)):
                if altquery.stop:
                    descend = False
                if query.subquery is not None:
                    node = self.first_child
                    while node is not None:
                        if node._select(query.subquery, nodes, first):
                            return True
                        node = node.next_sibling
                else:
                    nodes.append(self)
                    if first:
                        return True
                break
            altquery = altquery.altquery
        if descend:
            node = self.first_child
            while node is not None:
                if node._select(query, nodes, first):
                    return True
                node = node.next_sibling
        return False

    cdef bint _equals(self, Node other):
        if self.kind != other.kind:
            return False
        if self.tags != other.tags:
            return False
        if self.attributes != other.attributes:
            return False
        cdef Node child1 = self.first_child, child2 = other.first_child
        while child1 is not None and child2 is not None:
            if not child1._equals(child2):
                return False
            child1 = child1.next_sibling
            child2 = child2.next_sibling
        return child1 is None and child2 is None

    def __eq__(self, Node other not None):
        return self._equals(other)

    def __len__(self):
        return len(self.attributes)

    def __contains__(self, str name):
        return name in self.attributes

    def __getitem__(self, str name):
        return self.attributes[name]

    def __setitem__(self, str name, value):
        self.attributes[name] = value

    def keys(self):
        return self.attributes.keys()

    def values(self):
        return self.attributes.values()

    def items(self):
        return self.attributes.items()

    def get(self, str name, int n, type t, default=None, /):
        cdef Vector attr_vec
        cdef list attr_values, values
        cdef int m, i
        attr_vec = <Vector> self.attributes.get(name)
        if attr_vec is not None:
            attr_values = attr_vec.values
            m = len(attr_values)
            try:
                if m == 1:
                    value = attr_values[0]
                    if not isinstance(value, t):
                        value = t(value)
                    if n == 1:
                        return value
                    return [value] * n
                if m == n:
                    values = attr_values.copy()
                    for i, value in enumerate(values):
                        if not isinstance(value, t):
                            values[i] = t(value)
                    return values
            except:
                pass
        return default

    def __iter__(self):
        return iter(self.attributes)

    def __repr__(self):
        cdef str text = "Node('" + self.kind + "', {" + ",".join(repr(tag) for tag in self.tags) + "}"
        cdef str key
        cdef Node node
        for key, value in self.attributes.items():
            text += ", " + key + "=" + repr(value)
        text += ")"
        node = self.first_child
        while node is not None:
            for line in repr(node).split('\n'):
                text += "\n    " + line
            node = node.next_sibling
        return text


cdef class Context:
    def __cinit__(self, dict variables=None, dict state=None, Node graph=None, dict pragmas=None):
        self.variables = variables if variables is not None else {}
        self.state = state if state is not None else {}
        self.graph = graph if graph is not None else Node.__new__(Node, 'root', ())
        self.pragmas = pragmas if pragmas is not None else {}
        self._stack = []

    def __enter__(self):
        self._stack.append(None)
        return self

    def __exit__(self, *args):
        cdef dict variables = self._stack.pop()
        if variables is not None:
            self.variables = variables

    def __contains__(self, name):
        return name in self.variables

    def __getitem__(self, name):
        return self.variables[name]

    cdef void setitem(self, str name, value):
        if self._stack[-1] is None:
            self._stack[-1] = self.variables
            self.variables = self.variables.copy()
        self.variables[name] = value

    def __setitem__(self, str name, value):
        self.setitem(name, value)

    cpdef merge_under(self, Node node):
        if self._stack[-1] is None:
            self._stack[-1] = self.variables
            self.variables = self.variables.copy()
        for attr, value in node.attributes.items():
            if attr not in self.variables:
                self.variables[attr] = value

    def pragma(self, str name, value):
        self.pragmas[name] = value
