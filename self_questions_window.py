"""Responsive, selectable self-question viewer with clipboard export."""
from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any, Mapping

from self_questions_format import format_question, format_questions

class SelfQuestionsWindow:
    def __init__(self, parent: tk.Misc, path: Path) -> None:
        self.path = Path(path)
        self.entries: list[dict[str, Any]] = []
        self.visible_indices: list[int] = []
        self.window = tk.Toplevel(parent)
        self.window.title("Self Questions")
        self.window.geometry("940x620")
        self.window.minsize(620, 400)

        outer = ttk.Frame(self.window, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(outer)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.show_resolved = tk.BooleanVar(value=True)
        ttk.Checkbutton(toolbar, text="Show resolved", variable=self.show_resolved, command=self._render).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Reload", command=self.reload).pack(side=tk.RIGHT)
        ttk.Button(toolbar, text="Copy all", command=self.copy_all).pack(side=tk.RIGHT, padx=(0, 6))
        ttk.Button(toolbar, text="Copy selected", command=self.copy_selected).pack(side=tk.RIGHT, padx=(0, 6))

        panes = ttk.Panedwindow(outer, orient=tk.VERTICAL)
        panes.grid(row=1, column=0, sticky="nsew")
        table_frame = ttk.Frame(panes)
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(
            table_frame, columns=("question", "asked", "status", "updated"),
            show="headings", selectmode="extended",
        )
        for key, title, width, stretch in (
            ("question", "Question", 480, True), ("asked", "Count", 70, False),
            ("status", "Status", 110, False), ("updated", "Updated", 190, True),
        ):
            self.tree.heading(key, text=title)
            self.tree.column(key, width=width, minwidth=60, stretch=stretch)
        scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        self.tree.bind("<<TreeviewSelect>>", self._show_selected)

        self.detail = tk.Text(panes, height=9, wrap=tk.WORD, state=tk.DISABLED, padx=8, pady=8)
        panes.add(table_frame, weight=4)
        panes.add(self.detail, weight=1)
        self.status = tk.StringVar(value="")
        ttk.Label(outer, textvariable=self.status).grid(row=2, column=0, sticky="w", pady=(7, 0))
        self.reload()

    def reload(self) -> None:
        try:
            if not self.path.is_file() or self.path.stat().st_size > 8 * 1024 * 1024:
                raise ValueError("Self-question store is missing or exceeds the 8 MiB display bound.")
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.entries = [dict(entry) for entry in raw if isinstance(entry, Mapping) and entry.get("question")]
        except Exception as exc:
            self.entries = []
            self.status.set(str(exc))
        self._render()

    def _render(self) -> None:
        self.tree.delete(*self.tree.get_children())
        self.visible_indices = []
        for index, entry in enumerate(self.entries):
            resolved = bool(entry.get("resolved_at"))
            if resolved and not self.show_resolved.get():
                continue
            iid = str(index)
            self.visible_indices.append(index)
            self.tree.insert("", tk.END, iid=iid, values=(
                entry.get("question"), int(entry.get("count", 1) or 1),
                "resolved" if resolved else "open",
                entry.get("last_updated") or entry.get("first_asked") or "",
            ))
        self.status.set(f"{len(self.visible_indices)} visible · {len(self.entries)} total")
        self._show_selected()

    def _selected_entries(self) -> list[dict[str, Any]]:
        return [self.entries[int(iid)] for iid in self.tree.selection() if iid.isdigit() and int(iid) < len(self.entries)]

    def _show_selected(self, _event: object = None) -> None:
        selected = self._selected_entries()
        content = format_questions(selected) if selected else "Select one or more questions to inspect or copy."
        self.detail.config(state=tk.NORMAL)
        self.detail.delete("1.0", tk.END)
        self.detail.insert("1.0", content)
        self.detail.config(state=tk.DISABLED)

    def _copy(self, entries: list[dict[str, Any]]) -> None:
        if not entries:
            messagebox.showinfo("Self Questions", "Nothing is selected to copy.", parent=self.window)
            return
        self.window.clipboard_clear()
        self.window.clipboard_append(format_questions(entries))
        self.status.set(f"Copied {len(entries)} question(s) to the clipboard")

    def copy_selected(self) -> None:
        self._copy(self._selected_entries())

    def copy_all(self) -> None:
        self._copy([self.entries[index] for index in self.visible_indices])


__all__ = ["SelfQuestionsWindow", "format_question", "format_questions"]
