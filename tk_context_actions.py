"""Reusable, bounded right-click text actions for Ina's Tk interfaces."""
from __future__ import annotations

import tkinter as tk
from typing import Any


def context_action_labels(*, has_selection: bool, editable: bool, has_text: bool) -> list[str]:
    """Return the visible action surface for a text widget state."""
    labels = []
    if has_selection:
        labels.append("Copy")
    if has_text:
        labels.append("Copy all")
    if editable:
        if has_selection:
            labels.append("Cut")
        labels.append("Paste")
    if has_text:
        labels.append("Select all")
    return labels


def _widget_text(widget: Any) -> str:
    try:
        if widget.winfo_class() == "Text":
            return widget.get("1.0", "end-1c")
        return widget.get()
    except (AttributeError, tk.TclError):
        return ""


def _selection(widget: Any) -> str:
    try:
        if widget.winfo_class() == "Text":
            return widget.get("sel.first", "sel.last")
        return widget.get("sel.first", "sel.last") if widget.selection_present() else ""
    except (AttributeError, tk.TclError):
        return ""


def _editable(widget: Any) -> bool:
    try:
        if widget.winfo_class() == "Text":
            return str(widget.cget("state")) == str(tk.NORMAL)
        return not widget.instate(("disabled", "readonly"))
    except (AttributeError, tk.TclError):
        return False


def _copy(widget: Any, text: str) -> None:
    if not text:
        return
    widget.clipboard_clear()
    widget.clipboard_append(text)


def _select_all(widget: Any) -> None:
    if widget.winfo_class() == "Text":
        widget.tag_add(tk.SEL, "1.0", "end-1c")
        widget.mark_set(tk.INSERT, "1.0")
    else:
        widget.selection_range(0, tk.END)
        widget.icursor(tk.END)


def _cut(widget: Any) -> None:
    selected = _selection(widget)
    _copy(widget, selected)
    if not selected:
        return
    if widget.winfo_class() == "Text":
        widget.delete("sel.first", "sel.last")
    else:
        widget.delete("sel.first", "sel.last")


def _paste(widget: Any) -> None:
    try:
        text = widget.clipboard_get()
    except tk.TclError:
        return
    try:
        if widget.winfo_class() == "Text":
            widget.insert(tk.INSERT, text)
        else:
            if _selection(widget):
                widget.delete("sel.first", "sel.last")
            widget.insert(tk.INSERT, text)
    except tk.TclError:
        return


def install_text_context_actions(root: Any) -> None:
    """Install copy/cut/paste/select actions for Tk Text and Entry classes."""
    def popup(event: Any):
        widget = event.widget
        text = _widget_text(widget)
        selected = _selection(widget)
        editable = _editable(widget)
        labels = context_action_labels(
            has_selection=bool(selected), editable=editable, has_text=bool(text),
        )
        if not labels:
            return None
        menu = tk.Menu(widget, tearoff=False)
        actions = {
            "Copy": lambda: _copy(widget, selected),
            "Copy all": lambda: _copy(widget, text),
            "Cut": lambda: _cut(widget),
            "Paste": lambda: _paste(widget),
            "Select all": lambda: _select_all(widget),
        }
        for label in labels:
            menu.add_command(label=label, command=actions[label])
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()
        return "break"

    for widget_class in ("Text", "Entry", "TEntry"):
        root.bind_class(widget_class, "<Button-3>", popup, add="+")


__all__ = ["context_action_labels", "install_text_context_actions"]
