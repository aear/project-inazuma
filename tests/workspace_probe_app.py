"""Tiny manual/integration probe window; not part of Ina's runtime."""
from __future__ import annotations

import argparse
import json
import tkinter as tk
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state = {"text": "", "clicks": 0, "keys": []}

    def publish() -> None:
        args.output.write_text(json.dumps(state), encoding="utf-8")

    root = tk.Tk()
    root.title("Ina Workspace Input Probe")
    root.geometry("800x500+0+0")
    tk.Label(root, text="INA VIRTUAL WORKSPACE", font=("Sans", 28)).pack(pady=30)
    entry = tk.Entry(root, font=("Sans", 20))
    entry.pack(fill="x", padx=50, pady=30)
    entry.focus_force()

    def changed(*_args) -> None:
        state["text"] = entry.get()
        publish()

    def clicked(_event) -> None:
        state["clicks"] += 1
        publish()

    def keyed(event) -> None:
        state["keys"].append(event.keysym)
        state["keys"] = state["keys"][-100:]
        publish()

    variable = tk.StringVar()
    entry.configure(textvariable=variable)
    variable.trace_add("write", changed)
    root.bind("<ButtonPress>", clicked)
    root.bind("<KeyPress>", keyed)
    publish()
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
