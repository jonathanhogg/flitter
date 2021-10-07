# cython: language_level=3

import cython
from libc.math cimport isnan, floor, sin, cos


DEF PI = 3.141592653589793
DEF TwoPI = 6.283185307179586


@cython.final
@cython.freelist(16)
cdef class Vector:
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
        cdef float start = float(startv) if startv is not None else float("nan")
        if isnan(start):
            start = 0
        cdef float stop = float(stopv) if stopv is not None else float("nan")
        if isnan(stop):
            return null_
        cdef float step = float(stepv) if stepv is not None else float("nan")
        if isnan(step):
            step = 1
        elif step == 0:
            return null_
        cdef Vector result = Vector.__new__(Vector)
        cdef int i, n = <int>floor((stop - start) / step)
        for i in range(n):
            result.values.append(start + step * i)
        return result

    cdef list values

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

    def __eq__(self, other):
        return self.compare(other) == 0

    cpdef bint istrue_(self):
        for i in range(len(self.values)):
            if self.values[i]:
                return True
        return False

    def isinstance(self, t):
        for i in range(len(self.values)):
            if not isinstance(self.values[i], t):
                return False
        return True

    def copynodes(self):
        cdef int n = len(self.values)
        for i in range(n):
            if isinstance(self.values[i], Node):
                break
        else:
            return self
        cdef Vector result = Vector.__new__(Vector, self.values)
        cdef Node node
        for i in range(n):
            value = result.values[i]
            if isinstance(value, Node):
                node = value
                result.values[i] = node.copy()
        return result

    def neg(self):
        cdef Vector result = Vector.__new__(Vector)
        for i in range(len(self.values)):
            result.values.append(-self.values[i])
        return result

    def not_(self):
        return false_ if self.istrue_() else true_

    def add(self, Vector other not None):
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

    def sub(self, Vector other not None):
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

    def mul(self, Vector other not None):
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

    def truediv(self, Vector other not None):
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

    def floordiv(self, Vector other not None):
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

    def mod(self, Vector other not None):
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

    def pow(self, Vector other not None):
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

    def and_(self, Vector other not None):
        return true_ if self.istrue_() and other.istrue_() else false_

    def or_(self, Vector other not None):
        return true_ if self.istrue_() or other.istrue_() else false_

    def compare(self, Vector other not None):
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

    def slice(self, Vector index):
        cdef Vector result = Vector.__new__(Vector)
        cdef int j, n = len(self.values)
        for i in range(len(index.values)):
            value = index.values[i]
            if isinstance(value, (float, int)):
                j = <int>floor(value)
                if j >= 0 and j < n:
                    result.values.append(self.values[j])
        return result

    def __repr__(self):
        return f"Vector({self.values!r})"


cdef object null_ = Vector.__new__(Vector)
cdef object true_ = Vector.__new__(Vector, (1.,))
cdef object false_ = Vector.__new__(Vector, (0.,))

null = null_
true = true_
false = false_


@cython.freelist(16)
cdef class Uniform:
    cdef Vector keys
    cdef unsigned long long seed

    def __cinit__(self, Vector keys not None):
        cdef unsigned long long y, seed = 0xe220a8397b1dcdaf
        for i in range(len(keys.values)):
            key = keys.values[i]
            if isinstance(key, str):
                # FNV-1a hash algorithm [https://en.wikipedia.org/wiki/Fowler–Noll–Vo_hash_function#FNV-1a_hash]
                y = <unsigned long long>(0xcbf29ce484222325)
                for c in <str>key:
                    y = (y ^ ord(c)) * <unsigned long long>(0x100000001b3)
            elif isinstance(key, float):
                y = <unsigned long long>(<long long>floor(key))
            elif isinstance(key, int):
                y = <unsigned long long>(<long long>key)
            seed ^= y
            seed += <unsigned long long>(0x9e3779b97f4a7c15)
            seed ^= seed >> 30
            seed *= <unsigned long long>(0xbf58476d1ce4e5b9)
            seed ^= seed >> 27
            seed *= <unsigned long long>(0x94d049bb133111eb)
            seed ^= seed >> 31
        self.seed = seed
        self.keys = keys

    cdef double item(self, unsigned long long i):
        cdef unsigned long long x, y, z
        # Compute a 32bit float PRN using the Squares algorithm [https://arxiv.org/abs/2004.06278]
        x = y = i * self.seed
        z = y + self.seed
        x = x*x + y
        x = (x >> 32) | (x << 32)
        x = x*x + z
        x = (x >> 32) | (x << 32)
        x = x*x + y
        return <double>(x >> 32) / <double>(1<<32)

    def slice(self, Vector index not None):
        cdef Vector result = Vector.__new__(Vector)
        cdef int j
        for i in range(len(index.values)):
            value = index.values[i]
            if isinstance(value, (float, int)):
                j = <int>floor(value)
                if j >= 0:
                    result.values.append(self.item(j))
        return result

    def __repr__(self):
        return f"{self.__class__.__name__}({self.keys!r})"


cdef class Beta(Uniform):
    cdef double item(self, unsigned long long i):
        i <<= 2
        cdef double u1 = Uniform.item(self, i)
        cdef double u2 = Uniform.item(self, i + 1)
        cdef double u3 = Uniform.item(self, i + 2)
        if u1 <= u2 and u1 <= u3:
            return min(u2, u3)
        if u2 <= u1 and u2 <= u3:
            return min(u1, u3)
        return min(u1, u2)


cdef class Normal(Uniform):
    cdef double item(self, unsigned long long i):
        i <<= 4
        cdef double u = -6
        cdef int j
        for j in range(12):
            u += Uniform.item(self, i + j)
        return u


@cython.final
@cython.freelist(8)
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
@cython.freelist(16)
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


def sine(Vector xs not None):
    cdef Vector ys = Vector.__new__(Vector)
    cdef double x, y
    for i in range(len(xs.values)):
        x = xs.values[i]
        x -= floor(x)
        y = (1 - cos(TwoPI * x)) / 2
        ys.values.append(y)
    return ys


def bounce(Vector xs not None):
    cdef Vector ys = Vector.__new__(Vector)
    cdef double x, y
    for i in range(len(xs.values)):
        x = xs.values[i]
        x -= floor(x)
        y = sin(PI * x)
        ys.values.append(y)
    return ys


def sharkfin(Vector xs not None):
    cdef Vector ys = Vector.__new__(Vector)
    cdef double x, y
    for i in range(len(xs.values)):
        x = xs.values[i]
        x -= floor(x)
        y = sin(PI * x) if x < 0.5 else 1 - sin(PI * (x - 0.5))
        ys.values.append(y)
    return ys


def sawtooth(Vector xs not None):
    cdef Vector ys = Vector.__new__(Vector)
    cdef double x
    for i in range(len(xs.values)):
        x = xs.values[i]
        x -= floor(x)
        ys.values.append(x)
    return ys


def triangle(Vector xs not None):
    cdef Vector ys = Vector.__new__(Vector)
    cdef double x, y
    for i in range(len(xs.values)):
        x = xs.values[i]
        x -= floor(x)
        y = 1 - abs(x - 0.5) * 2
        ys.values.append(y)
    return ys


def square(Vector xs not None):
    cdef Vector ys = Vector.__new__(Vector)
    cdef double x, y
    for i in range(len(xs.values)):
        x = xs.values[i]
        x -= floor(x)
        y = 0 if x < 0.5 else 1
        ys.values.append(y)
    return ys


def linear(Vector xs not None):
    cdef Vector ys = Vector.__new__(Vector)
    cdef double x
    for i in range(len(xs.values)):
        x = xs.values[i]
        if x < 0:
            x = 0
        elif x > 1:
            x = 1
        ys.values.append(x)
    return ys


def quad(Vector xs not None):
    cdef Vector ys = Vector.__new__(Vector)
    cdef double x, y
    for i in range(len(xs.values)):
        x = xs.values[i]
        if x < 0:
            x = 0
        elif x > 1:
            x = 1
        y = (x * 2)**2 / 2 if x < 0.5 else 1 - ((1 - x) * 2)**2 / 2
        ys.values.append(y)
    return ys


def shuffle(Uniform source, Vector xs not None):
    cdef int j, n = len(xs.values)
    xs = Vector.__new__(Vector, xs)
    for i in range(n - 1):
        j = <int>floor(source.item(i) * n) + i
        n -= 1
        xs.values[i], xs.values[j] = xs.values[j], xs.values[i]
    return xs
