"""Manual GUI for side-by-side benchmarks of retained module versions."""
from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

from module_benchmarks import benchmark_module, list_benchmark_modules


class ModuleBenchmarkWindow:
    def __init__(self, parent: tk.Misc, history_path: Path = Path("benchmark_results/module_versions.jsonl")) -> None:
        self.history_path = Path(history_path)
        self.registry = list_benchmark_modules()
        self.results: list[dict[str, Any]] = []
        self.window = tk.Toplevel(parent)
        self.window.title("Module Benchmark Suite")
        self.window.geometry("980x650")
        self.window.minsize(680, 430)

        outer = ttk.Frame(self.window, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(2, weight=1)

        controls = ttk.Frame(outer)
        controls.grid(row=0, column=0, sticky="ew")
        ttk.Label(controls, text="Module").pack(side=tk.LEFT)
        self.module = tk.StringVar(value=next(iter(self.registry), ""))
        combo = ttk.Combobox(controls, textvariable=self.module, values=tuple(self.registry), state="readonly", width=24)
        combo.pack(side=tk.LEFT, padx=8)
        combo.bind("<<ComboboxSelected>>", lambda _event: self._build_versions())
        ttk.Button(controls, text="Run comparison", command=self.run).pack(side=tk.LEFT)
        ttk.Button(controls, text="Copy results", command=self.copy_results).pack(side=tk.RIGHT)

        self.version_frame = ttk.LabelFrame(outer, text="Versions", padding=8)
        self.version_frame.grid(row=1, column=0, sticky="ew", pady=8)
        self.version_vars: dict[str, tk.BooleanVar] = {}

        body = ttk.Panedwindow(outer, orient=tk.VERTICAL)
        body.grid(row=2, column=0, sticky="nsew")
        table_frame = ttk.Frame(body)
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(table_frame, columns=("version", "score", "correct", "elapsed"), show="headings", selectmode="browse")
        for key, title, width in (("version", "Version", 120), ("score", "Accuracy", 140), ("correct", "Correct", 120), ("elapsed", "Elapsed", 140)):
            self.tree.heading(key, text=title)
            self.tree.column(key, width=width, stretch=True)
        self.tree.grid(row=0, column=0, sticky="nsew")
        self.tree.bind("<<TreeviewSelect>>", self._show_detail)
        detail = tk.Text(body, height=12, wrap=tk.WORD, state=tk.DISABLED, padx=8, pady=8)
        self.detail = detail
        body.add(table_frame, weight=3)
        body.add(detail, weight=2)
        self.status = tk.StringVar(value="Benchmarks run only when requested.")
        ttk.Label(outer, textvariable=self.status).grid(row=3, column=0, sticky="w", pady=(8, 0))
        self._build_versions()

    def _build_versions(self) -> None:
        for child in self.version_frame.winfo_children():
            child.destroy()
        self.version_vars = {}
        for spec in self.registry.get(self.module.get(), ()):
            var = tk.BooleanVar(value=True)
            self.version_vars[spec.version] = var
            ttk.Checkbutton(self.version_frame, text=f"{spec.version} · {spec.description}", variable=var).pack(anchor="w")

    def run(self) -> None:
        versions = tuple(version for version, var in self.version_vars.items() if var.get())
        if not versions:
            messagebox.showinfo("Module Benchmarks", "Select at least one version.", parent=self.window)
            return
        results = benchmark_module(self.module.get(), versions)
        self.results = [result.to_dict() for result in results]
        self.tree.delete(*self.tree.get_children())
        for index, result in enumerate(self.results):
            self.tree.insert("", tk.END, iid=str(index), values=(
                result["version"], f'{100.0 * result["accuracy"]:.1f}%',
                f'{result["correct"]}/{result["total"]}', f'{result["elapsed_seconds"]:.6f}s',
            ))
        self._append_history(self.results)
        self.status.set(f"Compared {len(self.results)} version(s); results appended to {self.history_path}")
        if self.results:
            self.tree.selection_set("0")
            self._show_detail()

    def _append_history(self, results: list[dict[str, Any]]) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result, sort_keys=True) + "\n")

    def _show_detail(self, _event: object = None) -> None:
        selection = self.tree.selection()
        payload = self.results[int(selection[0])] if selection else self.results
        self.detail.config(state=tk.NORMAL)
        self.detail.delete("1.0", tk.END)
        self.detail.insert("1.0", json.dumps(payload, indent=2, default=str))
        self.detail.config(state=tk.DISABLED)

    def copy_results(self) -> None:
        if not self.results:
            messagebox.showinfo("Module Benchmarks", "Run a comparison first.", parent=self.window)
            return
        self.window.clipboard_clear()
        self.window.clipboard_append(json.dumps(self.results, indent=2, default=str))
        self.status.set("Copied benchmark results to the clipboard")


__all__ = ["ModuleBenchmarkWindow"]
