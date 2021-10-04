"""
Value classes
"""

# pylama:ignore=C0103

import operator


class Vector:
    @classmethod
    def compose(cls, left, right):
        return cls(cls(left).values + cls(right).values)

    @classmethod
    def _unop(cls, op, xs):
        if xs:
            return cls(op(x) for x in xs.values)
        return cls()

    @classmethod
    def _binop(cls, op, xs, ys):
        if not isinstance(xs, cls):
            xs = cls(xs)
        if not isinstance(ys, cls):
            ys = cls(ys)
        if len(xs.values) == len(ys.values):
            return cls(op(x, y) for x, y in zip(xs.values, ys.values))
        if len(xs.values) == 1:
            x = xs.values[0]
            return cls(op(x, y) for y in ys.values)
        if len(ys.values) == 1:
            y = ys.values[0]
            return cls(op(x, y) for x in xs.values)
        return cls()

    @classmethod
    def _cmp(cls, xs, ys):
        if not isinstance(xs, cls):
            xs = cls(xs)
        if not isinstance(ys, cls):
            ys = cls(ys)
        for x, y in zip(xs.values, ys.values):
            if x < y:
                return -1
            if x > y:
                return 1
        if len(xs.values) < len(ys.values):
            return -1
        if len(xs.values) > len(ys.values):
            return 1
        return 0

    def __init__(self, values=()):
        if isinstance(values, (int, float)):
            self.values = (float(values),)
        elif isinstance(values, str):
            self.values = (values,)
        elif isinstance(values, Vector):
            self.values = values.values
        elif isinstance(values, Range):
            self.values = sum((value.values for value in values), ())
        else:
            self.values = tuple(values)

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        for value in self:
            yield Vector((value,))

    def __getitem__(self, index):
        if isinstance(index, Range) and index.stop is None:
            index = Range(index.start, len(self.values), index.step)
        if not isinstance(index, Vector):
            index = Vector(index)
        values = []
        for idx in index.values:
            if isinstance(idx, float):
                idx = int(idx // 1)
                try:
                    values.append(self.values[idx])
                except IndexError:
                    pass
        return Vector(values)

    def __float__(self):
        if len(self) != 1 or not isinstance(self.values[0], (int, float)):
            raise TypeError("Not a single number")
        return self.values[0]

    def __add__(self, other): return self._binop(operator.add, self, other)  # noqa
    def __radd__(self, other): return self._binop(operator.add, other, self)  # noqa
    def __sub__(self, other): return self._binop(operator.sub, self, other)  # noqa
    def __rsub__(self, other): return self._binop(operator.sub, other, self)  # noqa
    def __mul__(self, other): return self._binop(operator.mul, self, other)  # noqa
    def __rmul__(self, other): return self._binop(operator.mul, other, self)  # noqa
    def __floordiv__(self, other): return self._binop(operator.floordiv, self, other)  # noqa
    def __rfloordiv__(self, other): return self._binop(operator.floordiv, other, self)  # noqa
    def __truediv__(self, other): return self._binop(operator.truediv, self, other)  # noqa
    def __rtruediv__(self, other): return self._binop(operator.truediv, other, self)  # noqa
    def __mod__(self, other): return self._binop(operator.mod, self, other)  # noqa
    def __rmod__(self, other): return self._binop(operator.mod, other, self)  # noqa
    def __neg__(self): return self._unop(operator.neg, self)  # noqa
    def __eq__(self, other): return Vector((self._cmp(self, other) == 0,))  # noqa
    def __ne__(self, other): return Vector((self._cmp(self, other) != 0,))  # noqa
    def __lt__(self, other): return Vector((self._cmp(self, other) == -1,))  # noqa
    def __gt__(self, other): return Vector((self._cmp(self, other) == 1,))  # noqa
    def __le__(self, other): return Vector((self._cmp(self, other) != 1,))  # noqa
    def __ge__(self, other): return Vector((self._cmp(self, other) != -1,))  # noqa

    def __repr__(self):
        if self:
            return ";".join(repr(value) for value in self.values)
        return 'null'


class Range:
    def __init__(self, start, stop, step):
        self.start = float(start) if start is not None and start != null else None
        self.stop = float(stop) if stop is not None and stop != null else None
        self.step = float(step) if step is not None and step != null else None

    def __len__(self):
        if self.stop is None:
            raise TypeError("Unbounded Range")
        start = 0.0 if self.start is None else self.start
        step = 1.0 if self.step is None else self.step
        return (self.stop - start) // step

    def __getitem__(self, index):
        if index == null:
            return null
        try:
            index = float(index) // 1
        except TypeError:
            return null
        start = 0.0 if self.start is None else self.start
        step = 1.0 if self.step is None else self.step
        if index < 0:
            if self.stop is None:
                return null
            index += (self.stop - start) // step
        if index < 0 or (self.stop is not None and index > (self.stop - start) // step - 1):
            return null
        return Vector((start + index * step,))

    def __iter__(self):
        if self.stop is None:
            return
        start = 0.0 if self.start is None else self.start
        step = 1.0 if self.step is None else self.step
        value = start
        while (value < self.stop) if step > 0 else (value > self.stop):
            yield Vector((value,))
            value += step

    def __repr__(self):
        text = '..' if self.start is None else f'{self.start}..'
        if self.stop is not None:
            text += repr(self.stop)
        if self.step is not None:
            text += f':{self.stop}'
        return text


null = Vector(())
