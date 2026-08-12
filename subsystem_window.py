"""Manual-refresh GUI for supervised services and cognitive capabilities."""
from __future__ import annotations

import json
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Callable

from project_version import RELEASE


class SubsystemWindow:
    def __init__(
        self, parent: tk.Misc, *, status_provider: Callable[[], dict[str, Any]],
        services_provider: Callable[[], dict[str, Any]],
        restart_service: Callable[[str], Any],
        restart_capability: Callable[[str], dict[str, Any]],
        rollback_capability: Callable[[str], Any],
    ) -> None:
        self.status_provider = status_provider
        self.services_provider = services_provider
        self.restart_service = restart_service
        self.restart_capability = restart_capability
        self.rollback_capability = rollback_capability
        self.capabilities: dict[str, dict[str, Any]] = {}
        self.window = tk.Toplevel(parent)
        self.window.title(f"Ina Subsystems — {RELEASE}")
        self.window.geometry("1120x680")
        self.window.minsize(820, 480)
        header = ttk.Frame(self.window, padding=(14, 12))
        header.pack(fill="x")
        ttk.Label(header, text="Ina Subsystems", style="Title.TLabel").pack(side="left")
        ttk.Button(header, text="Refresh", command=self.refresh).pack(side="right")
        ttk.Button(header, text="Restart selected", command=self.restart_selected).pack(side="right", padx=6)
        ttk.Button(header, text="Rollback handler", command=self.rollback_selected).pack(side="right")
        self.summary = tk.StringVar(value="Loading subsystem state…")
        ttk.Label(self.window, textvariable=self.summary, padding=(14, 0, 14, 8)).pack(fill="x")
        notebook = ttk.Notebook(self.window)
        notebook.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.notebook = notebook
        self.service_tab = ttk.Frame(notebook, padding=8)
        self.capability_tab = ttk.Frame(notebook, padding=8)
        notebook.add(self.service_tab, text="Processes and services")
        notebook.add(self.capability_tab, text="Cognitive capabilities")
        self.service_tree = self._tree(
            self.service_tab, ("status", "pid", "restarts", "updated"),
            ("Status", "PID", "Restarts", "Updated"),
        )
        self.capability_tree = self._tree(
            self.capability_tab, ("status", "backend", "implementation", "ram", "generation"),
            ("Status", "Backend", "Implementation", "Expected RAM", "Patch generation"),
        )
        self.details = tk.Text(self.window, height=7, wrap="word", state="disabled")
        self.details.pack(fill="x", padx=12, pady=(0, 12))
        self.service_tree.bind("<<TreeviewSelect>>", self._selection_changed)
        self.capability_tree.bind("<<TreeviewSelect>>", self._selection_changed)
        self.refresh()

    @staticmethod
    def _tree(parent, columns, headings):
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True)
        tree = ttk.Treeview(frame, columns=columns, show="tree headings", selectmode="browse")
        tree.heading("#0", text="Subsystem")
        tree.column("#0", width=250, stretch=True)
        for name, heading in zip(columns, headings):
            tree.heading(name, text=heading)
            tree.column(name, width=130, stretch=True)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        return tree

    def exists(self) -> bool:
        return bool(self.window.winfo_exists())

    def lift(self) -> None:
        self.window.lift()
        self.window.focus_set()

    def _set_details(self, value: Any) -> None:
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("1.0", json.dumps(value, indent=2, default=str))
        self.details.configure(state="disabled")

    def _selection_changed(self, _event=None) -> None:
        focus = self.window.focus_get()
        tree = self.capability_tree if focus is self.capability_tree else self.service_tree
        selected = tree.selection()
        if selected:
            self._set_details(tree.item(selected[0], "tags") and self._detail_for(selected[0]))

    def _detail_for(self, item_id: str) -> Any:
        kind, _, name = item_id.partition(":")
        if kind == "capability":
            return self.capabilities.get(name, {})
        return (self.services_provider().get("services") or {}).get(name, {})

    def refresh(self) -> None:
        try:
            runtime = self.status_provider()
            services = self.services_provider()
        except Exception as exc:
            self.summary.set(f"Unable to read subsystem state: {exc}")
            return
        for item in self.service_tree.get_children():
            self.service_tree.delete(item)
        service_rows = services.get("services") if isinstance(services, dict) else {}
        service_rows = service_rows if isinstance(service_rows, dict) else {}
        for name, detail in sorted(service_rows.items()):
            detail = detail if isinstance(detail, dict) else {}
            self.service_tree.insert("", "end", iid=f"service:{name}", text=name.replace("_", " ").title(),
                                     values=(detail.get("status", "unknown"), detail.get("pid") or "—",
                                             detail.get("restart_count", 0), detail.get("updated_at", "—")),
                                     tags=("service",))
        scheduler = runtime.get("scheduler") if isinstance(runtime, dict) else {}
        running = {str(item.get("task_key")): item for item in scheduler.get("running", []) if isinstance(item, dict)}
        queued = {str(item.get("task_key")): item for item in scheduler.get("queue", []) if isinstance(item, dict)}
        patches = runtime.get("live_patches") if isinstance(runtime, dict) else {}
        capabilities = runtime.get("capabilities") if isinstance(runtime, dict) else []
        self.capabilities = {str(item.get("name")): item for item in capabilities if isinstance(item, dict)}
        for item in self.capability_tree.get_children():
            self.capability_tree.delete(item)
        for name, detail in sorted(self.capabilities.items()):
            process = running.get(name) or queued.get(name) or {}
            status = "running" if name in running else "queued" if name in queued else "available" if detail.get("available") else "unavailable"
            patch = patches.get(name) if isinstance(patches, dict) else {}
            cost = detail.get("expected_cost") if isinstance(detail.get("expected_cost"), dict) else {}
            ram = int(cost.get("ram_bytes") or 0)
            ram_text = f"{ram / (1024 ** 3):.2f} GB" if ram else "—"
            self.capability_tree.insert("", "end", iid=f"capability:{name}", text=name.replace("_", " ").title(),
                                        values=(status, detail.get("backend", "—"), detail.get("implementation", "—"),
                                                ram_text, patch.get("generation", "—") if isinstance(patch, dict) else "—"),
                                        tags=("capability",))
            detail["runtime_process"] = process
            detail["live_patch"] = patch
        envelope = runtime.get("resource_budget") if isinstance(runtime, dict) else {}
        verified = "verified" if envelope.get("enforced") else "UNVERIFIED"
        self.summary.set(
            f"Supervisor: {services.get('status', 'unavailable')} · "
            f"{len(service_rows)} services · {len(self.capabilities)} capabilities · envelope {verified}. "
            "State refreshes only when requested."
        )

    def _selected(self) -> tuple[str, str] | tuple[None, None]:
        tab = self.notebook.select()
        tree = self.service_tree if str(tab) == str(self.service_tab) else self.capability_tree
        selected = tree.selection()
        if not selected:
            return None, None
        return tuple(selected[0].split(":", 1))  # type: ignore[return-value]

    def restart_selected(self) -> None:
        kind, name = self._selected()
        if not name:
            return
        try:
            result = self.restart_service(name) if kind == "service" else self.restart_capability(name)
            if isinstance(result, dict) and result.get("ok") is False:
                messagebox.showerror("Restart failed", str(result.get("reason") or result))
            self.refresh()
        except Exception as exc:
            messagebox.showerror("Restart failed", str(exc))

    def rollback_selected(self) -> None:
        kind, name = self._selected()
        if kind != "capability" or not name:
            return
        try:
            self.rollback_capability(name)
            self.refresh()
        except Exception as exc:
            messagebox.showerror("Rollback unavailable", str(exc))
