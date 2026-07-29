"""Dependency-free numerical substrate for Homo Silicus.

NumPy-inspired rather than NumPy-compatible in full: packed float64 storage,
explicit 1-D/2-D shapes, useful broadcasting, reductions, dot/matmul, norms,
and cosine scans. Unsupported semantics fail loudly instead of guessing.
"""
from __future__ import annotations

import builtins
import math
import operator
from array import array as packed_array


def _size(shape):
    result = 1
    for axis in shape:
        if int(axis) < 0:
            raise ValueError("shape dimensions cannot be negative")
        result *= int(axis)
    return result


class Tensor:
    def __init__(self, values, shape):
        self.shape = tuple(map(int, shape))
        if len(self.shape) not in (1, 2):
            raise ValueError("only 1-D and 2-D tensors are currently supported")
        self._data = values if isinstance(values, packed_array) and values.typecode == "d" else packed_array("d", values)
        if len(self._data) != _size(self.shape):
            raise ValueError(f"data size {len(self._data)} does not match shape {self.shape}")

    @property
    def ndim(self): return len(self.shape)

    @property
    def size(self): return len(self._data)

    @property
    def T(self):
        if self.ndim == 1:
            return self.copy()
        rows, columns = self.shape
        return Tensor((self._data[row * columns + column]
                       for column in range(columns) for row in range(rows)), (columns, rows))

    def __len__(self): return self.shape[0]

    def __iter__(self):
        return iter(self._data) if self.ndim == 1 else (self[row] for row in range(self.shape[0]))

    def __getitem__(self, key):
        if self.ndim == 1:
            if isinstance(key, slice):
                values = self._data[key]
                return Tensor(values, (len(values),))
            return self._data[key]
        rows, columns = self.shape
        if isinstance(key, tuple):
            if len(key) != 2:
                raise TypeError("matrix indexing expects (row, column)")
            row, column = key[0] % rows, key[1] % columns
            return self._data[row * columns + column]
        if isinstance(key, slice):
            selected = range(*key.indices(rows))
            return Tensor((self._data[row * columns + column]
                           for row in selected for column in range(columns)), (len(selected), columns))
        row = int(key) % rows
        return Tensor(self._data[row * columns:(row + 1) * columns], (columns,))

    def __repr__(self): return f"Tensor({self.tolist()!r}, shape={self.shape!r})"

    def copy(self): return Tensor(packed_array("d", self._data), self.shape)

    def tolist(self):
        if self.ndim == 1:
            return list(self._data)
        rows, columns = self.shape
        return [list(self._data[row * columns:(row + 1) * columns]) for row in range(rows)]

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        shape = list(shape)
        if shape.count(-1) > 1:
            raise ValueError("only one inferred dimension is allowed")
        if -1 in shape:
            known = _size(axis for axis in shape if axis != -1)
            if not known or self.size % known:
                raise ValueError("cannot infer requested shape")
            shape[shape.index(-1)] = self.size // known
        if _size(shape) != self.size:
            raise ValueError(f"cannot reshape size {self.size} into {tuple(shape)}")
        return Tensor(packed_array("d", self._data), shape)

    def _binary(self, other, operation):
        if isinstance(other, (int, float)):
            return Tensor((operation(value, float(other)) for value in self._data), self.shape)
        other = asarray(other)
        if other.shape == self.shape:
            return Tensor((operation(a, b) for a, b in zip(self._data, other._data)), self.shape)
        if self.ndim == 2 and other.ndim == 1 and other.size == self.shape[1]:
            return Tensor((operation(value, other._data[index % other.size])
                           for index, value in enumerate(self._data)), self.shape)
        raise ValueError(f"cannot broadcast shapes {self.shape} and {other.shape}")

    def __add__(self, other): return self._binary(other, operator.add)
    def __sub__(self, other): return self._binary(other, operator.sub)
    def __mul__(self, other): return self._binary(other, operator.mul)
    def __truediv__(self, other): return self._binary(other, operator.truediv)
    def __neg__(self): return Tensor((-value for value in self._data), self.shape)
    def sum(self, axis=None): return sum(self, axis)
    def mean(self, axis=None): return mean(self, axis)
    def min(self): return builtins.min(self._data)
    def max(self): return builtins.max(self._data)


def array(values, dtype=float, copy=True):
    if dtype not in (float, "float", "float64", "f8"):
        raise TypeError("the core currently supports float64 only")
    if isinstance(values, Tensor):
        return values.copy() if copy else values
    rows = list(values)
    if rows and isinstance(rows[0], (list, tuple, packed_array, Tensor)):
        nested = [list(row) for row in rows]
        columns = len(nested[0])
        if any(len(row) != columns for row in nested):
            raise ValueError("ragged arrays are not supported")
        return Tensor((value for row in nested for value in row), (len(nested), columns))
    return Tensor(rows, (len(rows),))


def asarray(values, dtype=float): return array(values, dtype, copy=False)


def zeros(shape, dtype=float):
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    return Tensor((0.0 for _ in range(_size(shape))), shape)


def ones(shape, dtype=float):
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    return Tensor((1.0 for _ in range(_size(shape))), shape)


def sum(value, axis=None):
    value = asarray(value)
    if axis is None:
        return builtins.sum(value._data)
    if value.ndim != 2 or axis not in (0, 1):
        raise ValueError("axis must be 0 or 1 for a matrix")
    rows, columns = value.shape
    if axis == 0:
        return Tensor((builtins.sum(value._data[row * columns + column] for row in range(rows))
                       for column in range(columns)), (columns,))
    return Tensor((builtins.sum(value._data[row * columns:(row + 1) * columns])
                   for row in range(rows)), (rows,))


def mean(value, axis=None):
    value = asarray(value)
    if axis is None:
        return sum(value) / value.size if value.size else math.nan
    reduced = sum(value, axis)
    divisor = value.shape[axis]
    return reduced / divisor if divisor else Tensor((math.nan for _ in range(reduced.size)), reduced.shape)


def abs(value):
    value = asarray(value)
    return Tensor((builtins.abs(item) for item in value._data), value.shape)


def clip(value, lower, upper):
    if lower > upper:
        raise ValueError("lower bound cannot exceed upper bound")
    value = asarray(value)
    return Tensor((builtins.max(float(lower), builtins.min(float(upper), item))
                   for item in value._data), value.shape)


def dot(left, right):
    left, right = asarray(left), asarray(right)
    if left.ndim == right.ndim == 1:
        if left.size != right.size:
            raise ValueError("vector dimensions do not align")
        return builtins.sum(a * b for a, b in zip(left._data, right._data))
    if left.ndim == 2 and right.ndim == 1:
        rows, columns = left.shape
        if columns != right.size:
            raise ValueError("matrix and vector dimensions do not align")
        return Tensor((builtins.sum(left._data[row * columns + column] * right._data[column]
                                    for column in range(columns)) for row in range(rows)), (rows,))
    if left.ndim == right.ndim == 2:
        rows, shared = left.shape
        right_rows, columns = right.shape
        if shared != right_rows:
            raise ValueError("matrix dimensions do not align")
        return Tensor((builtins.sum(left._data[row * shared + inner] * right._data[inner * columns + column]
                                    for inner in range(shared))
                       for row in range(rows) for column in range(columns)), (rows, columns))
    raise ValueError("dot supports vector-vector, matrix-vector, and matrix-matrix")


matmul = dot


def norm(value, axis=None):
    value = asarray(value)
    squared = value * value
    if axis is None:
        return math.sqrt(sum(squared))
    reduced = sum(squared, axis)
    return Tensor((math.sqrt(item) for item in reduced._data), reduced.shape)


def cosine_rows(matrix, query, epsilon=1e-8):
    matrix, query = asarray(matrix), asarray(query)
    if matrix.ndim != 2 or query.ndim != 1 or matrix.shape[1] != query.size:
        raise ValueError("expected matrix (n,d) and query (d,)")
    dots = dot(matrix, query)
    denominators = norm(matrix, axis=1) * norm(query)
    return Tensor((score / (denominator + epsilon)
                   for score, denominator in zip(dots._data, denominators._data)), dots.shape)
