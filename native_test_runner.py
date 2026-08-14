"""Dependency-free runner for Project Inazuma's pytest-style focused tests."""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import inspect
import itertools
import sys
import tempfile
import traceback
from contextlib import ExitStack
from pathlib import Path
from types import ModuleType
from unittest import SkipTest

import ina_test


def _load(path: Path) -> ModuleType:
    name = "ina_native_" + path.stem + "_" + str(abs(hash(path.resolve())))
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader: raise ImportError(path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
    return module


def _parameter_rows(function):
    rows = [dict()]
    for names, values in getattr(function, "__ina_parametrize__", ()):
        additions = []
        for value in values:
            values_tuple = value if len(names) > 1 and isinstance(value, (tuple, list)) else (value,)
            if len(values_tuple) != len(names): raise ValueError(f"parameter count mismatch for {function.__name__}")
            additions.append(dict(zip(names, values_tuple)))
        rows = [{**left, **right} for left, right in itertools.product(rows, additions)]
    return rows


def _fixture(name: str, module: ModuleType, stack: ExitStack, cache: dict):
    if name in cache: return cache[name]
    if name == "tmp_path":
        value = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="ina_test_")))
    elif name == "monkeypatch":
        value = stack.enter_context(ina_test.MonkeyPatch())
    else:
        provider = getattr(module, name, None)
        if not callable(provider) or not getattr(provider, "__ina_fixture__", False): raise TypeError(f"unsupported fixture: {name}")
        kwargs = {parameter: _fixture(parameter, module, stack, cache) for parameter in inspect.signature(provider).parameters}
        value = provider(**kwargs)
        if inspect.isgenerator(value):
            generator = value; value = next(generator)
            stack.callback(lambda: next(generator, None))
    cache[name] = value
    return value


def run(paths, *, match: str = "") -> dict[str, int]:
    sys.modules["pytest"] = ina_test
    stats = {"passed": 0, "failed": 0, "skipped": 0}
    for path in paths:
        try: module = _load(Path(path))
        except (SkipTest, ModuleNotFoundError) as exc:
            print(f"SKIP {path}: {exc}"); stats["skipped"] += 1; continue
        except Exception:
            print(f"FAIL {path} (collection)"); traceback.print_exc(); stats["failed"] += 1; continue
        for name, function in inspect.getmembers(module, inspect.isfunction):
            if not name.startswith("test_") or (match and match not in name): continue
            for index, parameters in enumerate(_parameter_rows(function)):
                label = f"{Path(path).name}::{name}" + (f"[{index}]" if getattr(function, "__ina_parametrize__", None) else "")
                with ExitStack() as stack:
                    try:
                        cache = {}
                        kwargs = dict(parameters)
                        for parameter in inspect.signature(function).parameters:
                            if parameter not in kwargs: kwargs[parameter] = _fixture(parameter, module, stack, cache)
                        result = function(**kwargs)
                        if inspect.isawaitable(result): asyncio.run(result)
                    except SkipTest as exc:
                        print(f"SKIP {label}: {exc}"); stats["skipped"] += 1
                    except Exception:
                        print(f"FAIL {label}"); traceback.print_exc(); stats["failed"] += 1
                    else:
                        print(f"PASS {label}"); stats["passed"] += 1
                    finally:
                        patcher = cache.get("monkeypatch") if "cache" in locals() else None
                        if patcher is not None:
                            patcher.undo()
    return stats


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=[])
    parser.add_argument("-k", "--match", default="")
    args = parser.parse_args(argv)
    paths = [Path(item) for item in args.paths] or sorted(Path("tests").glob("test_*.py"))
    stats = run(paths, match=args.match)
    print(f"{stats['passed']} passed, {stats['failed']} failed, {stats['skipped']} skipped")
    return 1 if stats["failed"] else 0


if __name__ == "__main__": raise SystemExit(main())
