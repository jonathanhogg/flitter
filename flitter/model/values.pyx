# cython: language_level=3

import cython
import operator
from libc.math cimport isnan


@cython.final
@cython.freelist(8)
cdef class Vector:
    @staticmethod
    def compose(*args):
        cdef Vector result = Vector.__new__(Vector, ())
        cdef Vector vector
        for arg in args:
            vector = arg
            result.values.extend(vector.values)
        return result

    @staticmethod
    def range(startv, stopv, stepv):
        cdef Vector result = Vector.__new__(Vector, ())
        cdef float value = float(startv) if startv is not None else float("nan")
        if isnan(value):
            value = 0
        cdef float stop = float(stopv) if stopv is not None else float("nan")
        if isnan(stop):
            return result
        cdef float step = float(stepv) if stepv is not None else float("nan")
        if isnan(step):
            step = 1
        elif step == 0:
            return result
        while (step > 0 and value < stop) or (step < 0 and value > stop):
            result.values.append(value)
            value += step
        return result

    cdef list values

    def __cinit__(self, values):
        self.values = list(values)

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

    def isinstance(self, t):
        cdef int i
        for i in range(len(self.values)):
            if not isinstance(self.values[i], t):
                return False
        return True

    def neg(self):
        cdef Vector result = Vector.__new__(Vector, ())
        cdef int i
        for i in range(len(self.values)):
            result.values.append(-self.values[i])
        return result

    def add(self, Vector other not None):
        return self._binop(operator.add, other)

    def sub(self, Vector other not None):
        return self._binop(operator.sub, other)

    def mul(self, Vector other not None):
        return self._binop(operator.mul, other)

    def truediv(self, Vector other not None):
        return self._binop(operator.truediv, other)

    def floordiv(self, Vector other not None):
        return self._binop(operator.floordiv, other)

    def mod(self, Vector other not None):
        return self._binop(operator.mod, other)

    def pow(self, Vector other not None):
        return self._binop(operator.pow, other)

    cdef Vector _binop(self, op, Vector other):
        cdef Vector result = Vector.__new__(Vector, ())
        cdef int i
        if len(self.values) == len(other.values):
            for i in range(len(self.values)):
                result.values.append(op(self.values[i], other.values[i]))
        if len(self.values) == 1:
            x = self.values[0]
            for i in range(len(other.values)):
                result.values.append(op(x, other.values[i]))
        if len(other.values) == 1:
            y = other.values[0]
            for i in range(len(self.values)):
                result.values.append(op(self.values[i], y))
        return result

    def compare(self, Vector other not None):
        cdef int i, n = len(self.values), m = len(other.values)
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

    def slice(self, Vector index):
        cdef Vector result = Vector.__new__(Vector, ())
        cdef int i, j, n = len(self.values)
        for i in range(len(index.values)):
            value = index.values[i]
            if isinstance(value, (float, int)):
                j = int(value // 1)
                if j >= 0 and j < n:
                    result.values.append(self.values[j])
        return result

    def __repr__(self):
        return f"Vector({self.values!r})"


null = Vector(())


@cython.final
@cython.freelist(4)
cdef class Query:
    cdef str kind
    cdef frozenset tags
    cdef bint strict
    cdef bint stop
    cdef Query subquery, altquery

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
        if self.kind is None and self.id is None and not self.tags:
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
cdef class Node:
    cdef readonly str kind
    cdef readonly frozenset tags
    cdef dict attributes
    cdef readonly Node parent
    cdef Node next_sibling, first_child, last_child
    cdef object token

    @staticmethod
    def from_dict(d):
        d = d.copy()
        cdef str kind = d.pop('kind')
        cdef str tags = d.pop('tags', '')
        cdef list children = d.pop('children', [])
        cdef Node node = Node(kind=kind, id=id, tags=tags, parent=None)
        for key, value in d.items():
            node.attributes[key] = value
        cdef Node child
        for d in children:
            child = Node.from_dict(d)
            node.append(child)
        return node

    def __cinit__(self, str kind, tags, **attributes):
        self.kind = kind
        if isinstance(tags, str):
            self.tags = frozenset(tags.split())
        elif isinstance(tags, frozenset):
            self.tags = tags
        else:
            self.tags = frozenset(tags)
        self.attributes = {}
        for key, value in attributes.items():
            self.attributes[key] = value

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
            d[key] = value.to_dict() if hasattr(value, 'to_dict') else value
        if self.first_child is not None:
            d['_children'] = [node.to_dict() for node in self.children]
        return d

    cpdef Node copy(self):
        cdef Node node = Node(self.kind, self.tags, **self.attributes)
        cdef Node child = self.first_child
        while child is not None:
            node.append(child.copy())
            child = child.next_sibling
        return node

    cpdef void append(self, Node node):
        if node.parent is not None:
            node.parent.remove(self)
        node.parent = self
        if self.last_child is not None:
            self.last_child.next_sibling = node
            self.last_child = node
        else:
            self.first_child = self.last_child = node

    def extend(self, nodes):
        cdef Node node
        for node in nodes:
            self.append(node)

    cpdef void remove(self, Node node):
        if node.parent is not self:
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
        node.parent = None

    def delete(self):
        if self.parent is None:
            raise TypeError("No parent")
        self.parent.remove(self)

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
