"""Small pytest-compatible surface used by the dependency-free native runner."""
from __future__ import annotations

import importlib
import math
import os
import re
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest import SkipTest


class _Approx:
    def __init__(self, expected: Any, rel: float | None = None, abs: float | None = None) -> None:
        self.expected, self.rel, self.abs = expected, 1e-6 if rel is None else rel, 1e-12 if abs is None else abs

    def __eq__(self, actual: Any) -> bool:
        if isinstance(self.expected, dict) and isinstance(actual, dict):
            return self.expected.keys() == actual.keys() and all(_Approx(v, self.rel, self.abs) == actual[k] for k, v in self.expected.items())
        if isinstance(self.expected, (list, tuple)) and isinstance(actual, (list, tuple)):
            return len(self.expected) == len(actual) and all(_Approx(e, self.rel, self.abs) == a for e, a in zip(self.expected, actual))
        try: return math.isclose(float(actual), float(self.expected), rel_tol=self.rel, abs_tol=self.abs)
        except (TypeError, ValueError): return actual == self.expected

    def __repr__(self) -> str: return f"approx({self.expected!r})"


def approx(expected: Any, rel: float | None = None, abs: float | None = None) -> _Approx:
    return _Approx(expected, rel, abs)


class _Raises:
    def __init__(self, expected: type[BaseException] | tuple[type[BaseException], ...], match: str | None = None) -> None:
        self.expected, self.match, self.value = expected, match, None

    def __enter__(self): return self

    def __exit__(self, kind, value, traceback) -> bool:
        if kind is None: raise AssertionError(f"did not raise {self.expected}")
        if not issubclass(kind, self.expected): return False
        self.value = value
        if self.match and re.search(self.match, str(value)) is None:
            raise AssertionError(f"exception message {value!r} does not match {self.match!r}")
        return True


def raises(expected_exception, *, match: str | None = None): return _Raises(expected_exception, match)
def fail(reason: str = "") -> None: raise AssertionError(reason)
def skip(reason: str = "") -> None: raise SkipTest(reason)


def importorskip(name: str, minversion: str | None = None, reason: str | None = None) -> ModuleType:
    try: return importlib.import_module(name)
    except ImportError as exc: raise SkipTest(reason or str(exc)) from exc


class _Mark:
    def parametrize(self, names: str, values, **_kwargs):
        parsed = tuple(part.strip() for part in names.split(","))
        def decorate(function):
            specs = list(getattr(function, "__ina_parametrize__", ()))
            specs.append((parsed, tuple(values)))
            function.__ina_parametrize__ = specs
            return function
        return decorate

    def __getattr__(self, _name: str):
        def marker(*args, **kwargs):
            if len(args) == 1 and callable(args[0]) and not kwargs: return args[0]
            return lambda function: function
        return marker


mark = _Mark()


def fixture(function=None, **_kwargs):
    def decorate(target): target.__ina_fixture__ = True; return target
    return decorate(function) if function is not None else decorate


class MonkeyPatch:
    def __init__(self) -> None: self._undo = []

    def setattr(self, target, name=None, value=..., raising: bool = True) -> None:
        if isinstance(target, str):
            if value is ...: value, dotted = name, target
            else: dotted = target + "." + str(name)
            module_name, attribute = dotted.rsplit(".", 1)
            target, name = importlib.import_module(module_name), attribute
        if value is ...: raise TypeError("setattr requires a value")
        existed = hasattr(target, name)
        if raising and not existed: raise AttributeError(name)
        old = getattr(target, name, None)
        self._undo.append(lambda: setattr(target, name, old) if existed else delattr(target, name))
        setattr(target, name, value)

    def setitem(self, mapping, name, value) -> None:
        existed, old = name in mapping, mapping.get(name)
        self._undo.append(lambda: mapping.__setitem__(name, old) if existed else mapping.pop(name, None))
        mapping[name] = value

    def setenv(self, name: str, value: Any) -> None: self.setitem(os.environ, name, str(value))

    def delenv(self, name: str, raising: bool = True) -> None:
        if name not in os.environ:
            if raising: raise KeyError(name)
            return
        old = os.environ[name]; self._undo.append(lambda: os.environ.__setitem__(name, old)); del os.environ[name]

    def chdir(self, path: str | Path) -> None:
        old = Path.cwd(); self._undo.append(lambda: os.chdir(old)); os.chdir(path)

    def undo(self) -> None:
        while self._undo: self._undo.pop()()

    def __enter__(self): return self
    def __exit__(self, *_args): self.undo()
    @classmethod
    def context(cls): return cls()


__all__ = ["MonkeyPatch", "approx", "fail", "fixture", "importorskip", "mark", "raises", "skip"]
