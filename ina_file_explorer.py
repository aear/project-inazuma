"""Small data-only file explorer for Ina's virtual desktop."""
from __future__ import annotations

import argparse
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, simpledialog

from config_layers import load_config
from ina_desktop.files import VirtualFileSystem, configured_drives


class InaFileExplorer:
    def __init__(self, child: str, *, project_root: Path | str = ".") -> None:
        self.child = str(child)
        self.fs = VirtualFileSystem(configured_drives(load_config(), self.child, project_root=project_root))
        self.fs.ensure_writable_roots()
        self.current_drive = next(iter(self.fs.drives))
        self.current_folder = Path()
        self.current_file: Path | None = None
        self.root = tk.Tk()
        self.root.title(f"Ina's Files — {self.child}")
        self.root.geometry("1050x680")
        self._build()
        self.refresh()

    def _build(self) -> None:
        left = tk.Frame(self.root, bg="#171b26", width=210)
        left.pack(side="left", fill="y")
        tk.Label(left, text="DRIVES", bg="#171b26", fg="#94a3b8").pack(anchor="w", padx=12, pady=(12, 6))
        for drive in self.fs.drives.values():
            suffix = " · write" if drive.writable else " · read only"
            tk.Button(left, text=drive.label + suffix, anchor="w", command=lambda key=drive.id: self.select_drive(key)).pack(fill="x", padx=8, pady=2)
        tk.Label(left, text="Files are data only.\nThere is no Run or Open-with action.", justify="left", wraplength=180, bg="#171b26", fg="#94a3b8").pack(side="bottom", padx=12, pady=14)

        body = tk.Frame(self.root)
        body.pack(side="left", fill="both", expand=True)
        toolbar = tk.Frame(body)
        toolbar.pack(fill="x")
        tk.Button(toolbar, text="Up", command=self.up).pack(side="left", padx=4, pady=4)
        tk.Button(toolbar, text="New folder", command=self.new_folder).pack(side="left", padx=4)
        tk.Button(toolbar, text="New note", command=self.new_note).pack(side="left", padx=4)
        tk.Button(toolbar, text="Save text", command=self.save_text).pack(side="left", padx=4)
        self.path_var = tk.StringVar()
        tk.Label(toolbar, textvariable=self.path_var).pack(side="left", padx=10)
        panes = tk.PanedWindow(body, orient="horizontal")
        panes.pack(fill="both", expand=True)
        self.listbox = tk.Listbox(panes, width=36)
        self.listbox.bind("<Double-Button-1>", self.open_selected)
        panes.add(self.listbox)
        self.editor = tk.Text(panes, wrap="word", undo=True)
        panes.add(self.editor)
        self.entries: list[dict] = []

    def select_drive(self, drive_id: str) -> None:
        self.current_drive, self.current_folder, self.current_file = drive_id, Path(), None
        self.editor.delete("1.0", "end")
        self.refresh()

    def refresh(self) -> None:
        relative = "" if self.current_folder == Path() else str(self.current_folder)
        try:
            self.entries = self.fs.list(self.current_drive, relative)
        except Exception as exc:
            messagebox.showerror("Cannot list folder", str(exc), parent=self.root)
            return
        self.listbox.delete(0, "end")
        for item in self.entries:
            self.listbox.insert("end", ("📁 " if item["directory"] else "   ") + item["name"])
        drive = self.fs.drives[self.current_drive]
        self.path_var.set(f"{drive.label}: /{relative} · {'writable' if drive.writable else 'read only'} · never executable")

    def open_selected(self, _event=None) -> None:
        selected = self.listbox.curselection()
        if not selected:
            return
        item = self.entries[selected[0]]
        relative = self.current_folder / item["name"]
        if item["directory"]:
            self.current_folder, self.current_file = relative, None
            self.refresh()
            return
        try:
            content = self.fs.read(self.current_drive, str(relative)).decode("utf-8")
        except UnicodeDecodeError:
            content = "[Binary media file. It can be selected as data, but this text view does not decode it.]"
        except Exception as exc:
            messagebox.showerror("Cannot read file", str(exc), parent=self.root)
            return
        self.current_file = relative
        self.editor.delete("1.0", "end")
        self.editor.insert("1.0", content)

    def up(self) -> None:
        self.current_folder = self.current_folder.parent if self.current_folder != Path() else Path()
        self.current_file = None
        self.refresh()

    def new_folder(self) -> None:
        name = simpledialog.askstring("New folder", "Folder name:", parent=self.root)
        if name:
            try:
                self.fs.mkdir(self.current_drive, str(self.current_folder / name))
                self.refresh()
            except Exception as exc:
                messagebox.showerror("Cannot create folder", str(exc), parent=self.root)

    def new_note(self) -> None:
        name = simpledialog.askstring("New file", "File name:", initialvalue="thought.txt", parent=self.root)
        if name:
            self.current_file = self.current_folder / name
            self.editor.delete("1.0", "end")

    def save_text(self) -> None:
        if self.current_file is None:
            self.new_note()
        if self.current_file is None:
            return
        try:
            self.fs.write(self.current_drive, str(self.current_file), self.editor.get("1.0", "end-1c"))
            self.refresh()
        except Exception as exc:
            messagebox.showerror("Cannot save file", str(exc), parent=self.root)

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Open Ina's data-only virtual file explorer.")
    parser.add_argument("--child")
    args = parser.parse_args()
    config = load_config()
    InaFileExplorer(args.child or config.get("current_child") or "Inazuma_Yagami").run()


if __name__ == "__main__":
    main()
