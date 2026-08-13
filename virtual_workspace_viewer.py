"""Detachable viewer/controller for Ina's persistent virtual desktop."""
from __future__ import annotations

import argparse
import json
import os
import time
import tkinter as tk
from pathlib import Path

from config_layers import load_config
from ina_desktop.client import send_command, workspace_status
from ina_desktop.paths import viewer_lock_path
from ina_desktop.x11 import X11Desktop

try:
    import fcntl
except Exception:  # pragma: no cover
    fcntl = None


class WorkspaceViewer:
    def __init__(self, child: str) -> None:
        self.child = str(child)
        self.desktop: X11Desktop | None = None
        self.source_size = (1920, 1080)
        self.root = tk.Tk()
        self.root.title(f"Ina Virtual Desktop — {self.child}")
        self.root.geometry("1280x760")
        self.root.minsize(640, 420)
        toolbar = tk.Frame(self.root, bg="#151923")
        toolbar.pack(fill="x")
        tk.Button(toolbar, text="Tile windows", command=self.tile).pack(side="left", padx=4, pady=4)
        tk.Button(toolbar, text="Next window", command=self.next_window).pack(side="left", padx=4, pady=4)
        tk.Button(toolbar, text="Refresh", command=self.refresh_now).pack(side="left", padx=4, pady=4)
        self.status_var = tk.StringVar(value="Connecting to Ina's workspace…")
        tk.Label(toolbar, textvariable=self.status_var, fg="#d8deef", bg="#151923").pack(side="left", padx=10)
        self.canvas = tk.Canvas(self.root, bg="#080a10", highlightthickness=0, takefocus=True)
        self.canvas.pack(fill="both", expand=True)
        self.image_id = self.canvas.create_image(0, 0, anchor="nw")
        self.photo = None
        self._last_motion = 0.0
        self._bind_input()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(50, self._connect)

    def _connect(self) -> None:
        status = workspace_status(self.child)
        if not status.get("ready"):
            if status.get("status") in {"failed", "blocked"}:
                self.status_var.set(f"Workspace unavailable: {status.get('error', 'unknown error')}")
                self.root.after(5000, self._connect)
                return
            self.root.after(250, self._connect)
            return
        try:
            self.desktop = X11Desktop(str(status["display"]))
            self.source_size = self.desktop.size()
        except Exception as exc:
            self.status_var.set(f"Display connection failed: {exc}")
            self.root.after(1000, self._connect)
            return
        audio = status.get("audio") if isinstance(status.get("audio"), dict) else {}
        audio_text = "audio isolated" if audio.get("ready") else "audio unavailable"
        self.status_var.set(f"{status['display']} · {self.source_size[0]}×{self.source_size[1]} · {audio_text}")
        self.refresh_now()

    def _bind_input(self) -> None:
        self.canvas.bind("<Motion>", self._motion)
        self.canvas.bind("<ButtonPress>", self._button_press)
        self.canvas.bind("<ButtonRelease>", self._button_release)
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.canvas.bind("<KeyPress>", self._key_press)
        self.canvas.bind("<KeyRelease>", self._key_release)

    def _coords(self, event) -> tuple[int, int]:
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        return (
            max(0, min(self.source_size[0] - 1, round(event.x * self.source_size[0] / canvas_width))),
            max(0, min(self.source_size[1] - 1, round(event.y * self.source_size[1] / canvas_height))),
        )

    def _motion(self, event) -> None:
        if self.desktop is None:
            return
        now = time.monotonic()
        if now - self._last_motion < 1 / 60:
            return
        self._last_motion = now
        self.desktop.mouse_move(*self._coords(event))

    def _button_press(self, event) -> None:
        if self.desktop is None:
            return
        self.canvas.focus_set()
        self.desktop.mouse_move(*self._coords(event))
        self.desktop.mouse_button(event.num, True)

    def _button_release(self, event) -> None:
        if self.desktop is not None:
            self.desktop.mouse_button(event.num, False)

    def _wheel(self, event) -> None:
        if self.desktop is None:
            return
        button = 4 if event.delta > 0 else 5
        self.desktop.mouse_button(button, True)
        self.desktop.mouse_button(button, False)

    def _key_press(self, event) -> None:
        if self.desktop is not None and event.keysym:
            try:
                self.desktop.key(event.keysym, True)
            except ValueError:
                pass

    def _key_release(self, event) -> None:
        if self.desktop is not None and event.keysym:
            try:
                self.desktop.key(event.keysym, False)
            except ValueError:
                pass

    def tile(self) -> None:
        result = send_command(self.child, {"action": "tile"})
        if not result.get("ok"):
            self.status_var.set(f"Tile failed: {result.get('error')}")

    def next_window(self) -> None:
        result = send_command(self.child, {"action": "next_window"})
        if not result.get("ok"):
            self.status_var.set(f"Window focus failed: {result.get('error')}")

    def refresh_now(self) -> None:
        if self.desktop is None:
            return
        try:
            frame = self.desktop.capture()
            height, width = frame.shape[:2]
            target_width = max(1, self.canvas.winfo_width())
            target_height = max(1, self.canvas.winfo_height())
            step = max(1, min(width // target_width if target_width else 1,
                              height // target_height if target_height else 1))
            if step > 1 and hasattr(frame, "__getitem__"):
                frame = frame[::step, ::step]
            if hasattr(frame, "ppm"):
                ppm = frame.ppm(max_width=target_width, max_height=target_height)
            else:
                header = f"P6\n{frame.shape[1]} {frame.shape[0]}\n255\n".encode("ascii")
                ppm = header + frame.tobytes()
            self.photo = tk.PhotoImage(data=ppm, format="PPM")
            self.canvas.itemconfigure(self.image_id, image=self.photo)
            self.canvas.scale("all", 0, 0, 1, 1)
        except Exception as exc:
            self.status_var.set(f"Capture failed: {exc}")
        self.root.after(250, self.refresh_now)

    def close(self) -> None:
        if self.desktop is not None:
            self.desktop.close()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def _viewer_lock(child: str):
    path = viewer_lock_path(child)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8")
    if fcntl is not None:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return None
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child")
    args = parser.parse_args()
    cfg = load_config()
    child = str(args.child or cfg.get("current_child") or "Inazuma_Yagami")
    lock = _viewer_lock(child)
    if lock is None:
        return 0
    try:
        WorkspaceViewer(child).run()
        return 0
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
