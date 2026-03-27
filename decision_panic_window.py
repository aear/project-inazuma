import json
import os
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from io_utils import atomic_write_json, file_lock, load_json_dict

try:
    import fcntl  # type: ignore
except Exception:
    fcntl = None  # type: ignore


CONFIG_PATH = Path("config.json")
DEFAULT_GEOMETRY = "920x700+140+120"
REFRESH_MS = 5000
_LOCK_HANDLE = None


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json_dict(path: Path) -> Dict[str, Any]:
    return load_json_dict(path)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    atomic_write_json(path, payload, indent=4)


def _load_config() -> Dict[str, Any]:
    return _load_json_dict(CONFIG_PATH)


def _save_config_patch(patch: Dict[str, Any]) -> None:
    current = _load_config()
    current.update(patch)
    _atomic_write_json(CONFIG_PATH, current)


def _current_child() -> str:
    cfg = _load_config()
    child = cfg.get("current_child")
    if isinstance(child, str) and child.strip():
        return child.strip()
    return "Inazuma_Yagami"


def _inastate_path() -> Path:
    return Path("AI_Children") / _current_child() / "memory" / "inastate.json"


def _lock_path() -> Path:
    return Path("AI_Children") / _current_child() / "memory" / "decision_panic_window.lock"


def _acquire_lock() -> bool:
    global _LOCK_HANDLE
    if fcntl is None:
        return True
    path = _lock_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("w", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return False
        handle.write(str(os.getpid()))
        handle.flush()
        _LOCK_HANDLE = handle
        return True
    except Exception:
        return True


def _read_inastate() -> Dict[str, Any]:
    return _load_json_dict(_inastate_path())


def _update_inastate_fields(updates: Dict[str, Any]) -> None:
    path = _inastate_path()
    lock_path = path.with_name("inastate.lock")
    with file_lock(lock_path):
        current = _load_json_dict(path)
        current.update(updates)
        _atomic_write_json(path, current)


def _set_popup_active(active: bool) -> None:
    payload: Dict[str, Any] = {
        "decision_panic_popup_active": {
            "active": bool(active),
            "timestamp": _iso_now(),
            "pid": os.getpid(),
        }
    }
    if not active:
        payload["decision_panic_popup_last_closed"] = _iso_now()
    _update_inastate_fields(payload)


def _fmt_missing_inputs(items: List[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for idx, item in enumerate(items, start=1):
        mid = item.get("id", "unknown")
        sev = item.get("severity", "unknown")
        canonical = item.get("canonical_variable")
        symbolic = item.get("symbolic_tag")
        reason = item.get("reason", "")
        probe = item.get("suggested_probe") or item.get("probe", "")
        where = item.get("where_to_obtain") or item.get("source_location")
        providers = item.get("provider_modules") if isinstance(item.get("provider_modules"), list) else []
        expected = item.get("expected_output")
        lines.append(f"{idx}. [{sev}] {mid}")
        if canonical or symbolic:
            lines.append(f"   need : {symbolic or '-'} -> {canonical or '-'}")
        if reason:
            lines.append(f"   reason: {reason}")
        if probe:
            lines.append(f"   probe : {probe}")
        if where:
            lines.append(f"   where : {where}")
        if providers:
            lines.append(f"   module: {', '.join(str(x) for x in providers if x)}")
        if expected:
            lines.append(f"   output: {expected}")
    return lines


def _fmt_checks(items: List[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for idx, item in enumerate(items, start=1):
        cid = item.get("id", "unknown")
        value = item.get("value")
        status = item.get("status", "unknown")
        note = item.get("note", "")
        lines.append(f"{idx}. {cid}: value={value} status={status}")
        if note:
            lines.append(f"   note: {note}")
    return lines


def _render_state(state: Dict[str, Any]) -> str:
    panic = state.get("decision_panic")
    meta = state.get("meta_arbitration")
    advocacy = state.get("decision_need_advocacy")
    if not isinstance(panic, dict):
        panic = {}
    if not isinstance(meta, dict):
        meta = {}
    if not isinstance(advocacy, dict):
        advocacy = {}

    if not panic:
        return "No decision panic report found in inastate yet.\n\nWaiting for model manager updates..."

    lines: List[str] = []
    lines.append("Decision Panic Diagnostics")
    lines.append("=" * 30)
    lines.append(f"active      : {panic.get('active')}")
    lines.append(f"status      : {panic.get('status')}")
    lines.append(f"event       : {panic.get('event')}")
    lines.append(f"episode_id  : {panic.get('episode_id')}")
    lines.append(f"updated_at  : {panic.get('updated_at')}")
    lines.append(f"started_at  : {panic.get('started_at')}")
    lines.append(f"resolved_at : {panic.get('resolved_at')}")
    lines.append("")

    arbitration = panic.get("arbitration")
    if isinstance(arbitration, dict):
        lines.append("Arbitration Snapshot")
        lines.append("-" * 30)
        lines.append(f"status            : {arbitration.get('status')}")
        lines.append(f"top_signal        : {arbitration.get('top_signal')}")
        lines.append(f"runner_up_signal  : {arbitration.get('runner_up_signal')}")
        lines.append(f"winner_margin     : {arbitration.get('winner_margin')}")
        lines.append(f"conflict_score    : {arbitration.get('conflict_score')}")
        lines.append(f"indecision_seconds: {arbitration.get('indecision_seconds')}")
        lines.append(f"indecision_cost   : {arbitration.get('indecision_cost')}")
        lines.append(f"discomfort        : {arbitration.get('discomfort')}")
        allowed = arbitration.get("allowed_signals")
        if isinstance(allowed, list):
            lines.append(f"allowed_signals   : {', '.join(str(x) for x in allowed) or 'none'}")
        lines.append("")

    missing_inputs = panic.get("missing_inputs")
    if isinstance(missing_inputs, list) and missing_inputs:
        lines.append("Likely Missing Inputs")
        lines.append("-" * 30)
        lines.extend(_fmt_missing_inputs([x for x in missing_inputs if isinstance(x, dict)]))
        lines.append("")
    else:
        lines.append("Likely Missing Inputs")
        lines.append("-" * 30)
        lines.append("None identified in the latest panic payload.")
        lines.append("")

    checks = panic.get("diagnostic_checks")
    if isinstance(checks, list) and checks:
        lines.append("Diagnostic Checks")
        lines.append("-" * 30)
        lines.extend(_fmt_checks([x for x in checks if isinstance(x, dict)]))
        lines.append("")

    self_diag = panic.get("self_diagnosis")
    if isinstance(self_diag, list) and self_diag:
        lines.append("Self Diagnosis")
        lines.append("-" * 30)
        for idx, item in enumerate(self_diag, start=1):
            lines.append(f"{idx}. {item}")
        lines.append("")

    tools = panic.get("recommended_tools")
    if isinstance(tools, list) and tools:
        lines.append("Requested Support / Tools")
        lines.append("-" * 30)
        for idx, item in enumerate(tools, start=1):
            lines.append(f"{idx}. {item}")
        lines.append("")

    if advocacy:
        lines.append("Symbolic Need Report")
        lines.append("-" * 30)
        lines.append(f"mode        : {advocacy.get('mode')}")
        canonical = advocacy.get("canonical_variables")
        if isinstance(canonical, list):
            lines.append(f"canonical   : {', '.join(str(x) for x in canonical) or 'none'}")
        symbolic = advocacy.get("symbolic_tags")
        if isinstance(symbolic, list):
            lines.append(f"symbolic    : {', '.join(str(x) for x in symbolic) or 'none'}")
        bridge = advocacy.get("language_bridge")
        if bridge:
            lines.append(f"bridge      : {bridge}")
        entries = advocacy.get("entries")
        if isinstance(entries, list) and entries:
            lines.append("")
            lines.append("Need-to-Action Entries")
            for idx, entry in enumerate(entries, start=1):
                if not isinstance(entry, dict):
                    continue
                lines.append(
                    f"{idx}. {entry.get('symbolic_tag') or '-'} -> {entry.get('canonical_variable') or '-'}"
                )
                if entry.get("suggested_probe"):
                    lines.append(f"   probe : {entry.get('suggested_probe')}")
                if entry.get("where_to_obtain"):
                    lines.append(f"   where : {entry.get('where_to_obtain')}")
                providers = entry.get("provider_modules")
                if isinstance(providers, list) and providers:
                    lines.append(f"   module: {', '.join(str(x) for x in providers if x)}")
                if entry.get("expected_output"):
                    lines.append(f"   output: {entry.get('expected_output')}")
        lines.append("")

    if meta:
        lines.append("Latest Meta Arbitration")
        lines.append("-" * 30)
        lines.append(f"status      : {meta.get('status')}")
        lines.append(f"top_signal  : {meta.get('top_signal')}")
        lines.append(f"discomfort  : {meta.get('discomfort')}")
        lines.append(f"updated_at  : {meta.get('updated_at')}")

    return "\n".join(lines)


class DecisionPanicWindow(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Ina Decision Panic Diagnostics")
        cfg = _load_config()
        geometry = cfg.get("decision_panic_window_geometry")
        if isinstance(geometry, str) and geometry.strip():
            self.geometry(geometry)
        else:
            self.geometry(DEFAULT_GEOMETRY)

        self.minsize(760, 520)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        wrapper = ttk.Frame(self, padding=8)
        wrapper.pack(fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value="Loading panic diagnostics...")
        ttk.Label(wrapper, textvariable=self.status_var).pack(anchor=tk.W, pady=(0, 6))

        self.text = ScrolledText(wrapper, wrap=tk.WORD, font=("Courier", 10))
        self.text.pack(fill=tk.BOTH, expand=True)
        self.text.configure(state=tk.DISABLED)

        btns = ttk.Frame(wrapper)
        btns.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(btns, text="Refresh", command=self.refresh).pack(side=tk.LEFT)
        ttk.Button(btns, text="Acknowledge / Close", command=self.on_close).pack(side=tk.RIGHT)

        self._closed = False
        _set_popup_active(True)
        self.refresh()
        self.after(REFRESH_MS, self._poll)

    def _poll(self) -> None:
        if self._closed:
            return
        self.refresh()
        self.after(REFRESH_MS, self._poll)

    def refresh(self) -> None:
        state = _read_inastate()
        report_text = _render_state(state)
        panic = state.get("decision_panic") if isinstance(state, dict) else {}
        if isinstance(panic, dict):
            status = panic.get("status") or "unknown"
            updated = panic.get("updated_at") or "unknown"
            self.status_var.set(f"decision_panic status={status} updated_at={updated}")
        else:
            self.status_var.set("decision_panic status=missing")

        self.text.configure(state=tk.NORMAL)
        self.text.delete("1.0", tk.END)
        self.text.insert("1.0", report_text)
        self.text.configure(state=tk.DISABLED)

    def on_close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            _save_config_patch({"decision_panic_window_geometry": self.geometry()})
        except Exception:
            pass
        try:
            _set_popup_active(False)
        except Exception:
            pass
        self.destroy()


def main() -> None:
    if not _acquire_lock():
        return
    app = DecisionPanicWindow()
    app.mainloop()


if __name__ == "__main__":
    main()
