"""Build the optional dependency-free C++ vector kernel."""
from __future__ import annotations

import argparse
import platform
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default="g++")
    parser.add_argument("--portable", action="store_true", help="omit -march=native")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output_dir = root / ".native"
    output_dir.mkdir(exist_ok=True)
    suffix = ".dll" if platform.system() == "Windows" else ".dylib" if platform.system() == "Darwin" else ".so"
    output = output_dir / f"libinazuma_vector{suffix}"
    command = [args.compiler, "-O3", "-std=c++20", "-shared", "-fPIC", "-Wall", "-Wextra",
               str(root / "native" / "vector_kernel.cpp"), "-o", str(output)]
    if not args.portable:
        command.insert(2, "-march=native")
    subprocess.run(command, check=True)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
