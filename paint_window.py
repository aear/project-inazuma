import json
import time
from datetime import datetime, timezone
from pathlib import Path
import tkinter as tk
from tkinter import colorchooser, messagebox, simpledialog

from PIL import Image, ImageDraw

from gui_hook import log_to_statusbox
from model_manager import (
    append_typed_outbox_entry,
    get_inastate,
    load_config,
    update_inastate,
)


CANVAS_WIDTH = 900
CANVAS_HEIGHT = 600
DEFAULT_BG = "#ffffff"
DEFAULT_COLOR = "#1b1b1b"
PALETTE = [
    "#1b1b1b",
    "#c0392b",
    "#d35400",
    "#f1c40f",
    "#27ae60",
    "#2980b9",
    "#8e44ad",
    "#ecf0f1",
]


class PaintWindow:
    def __init__(self) -> None:
        config = load_config()
        self.child = config.get("current_child", "default_child")
        discord_cfg = config.get("discord") if isinstance(config, dict) else None
        self.can_share_text_channel = bool(
            isinstance(discord_cfg, dict)
            and (discord_cfg.get("text_channel_id") or discord_cfg.get("text_channel_name"))
        )
        self.session_dir = Path("AI_Children") / self.child / "memory" / "paint_sessions"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.fragments_dir = Path("AI_Children") / self.child / "memory" / "fragments"
        self.log_path = self.session_dir / "paint_log.json"

        self.root = tk.Tk()
        self.root.title("Ina Paint")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.canvas = tk.Canvas(
            self.root,
            width=CANVAS_WIDTH,
            height=CANVAS_HEIGHT,
            bg=DEFAULT_BG,
            highlightthickness=1,
            highlightbackground="#d0d0d0",
            cursor="crosshair",
        )
        self.canvas.grid(row=1, column=0, columnspan=9, padx=10, pady=10)

        self._init_image()

        self.brush_size = tk.IntVar(value=6)
        self.is_eraser = tk.BooleanVar(value=False)
        self.active_color = DEFAULT_COLOR
        self.last_x = None
        self.last_y = None
        self.dirty = False
        self.last_saved_path = None

        self._build_toolbar()
        self._bind_canvas()
        self._set_window_state(True)

    def _set_window_state(self, is_open: bool) -> None:
        try:
            update_inastate("paint_window_open", is_open)
        except Exception:
            pass

    def _init_image(self) -> None:
        self.image = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), DEFAULT_BG)
        self.draw = ImageDraw.Draw(self.image)

    def _build_toolbar(self) -> None:
        tk.Label(self.root, text="Brush").grid(row=0, column=0, padx=(10, 6), pady=6, sticky="w")
        size_slider = tk.Scale(
            self.root,
            from_=2,
            to=36,
            orient=tk.HORIZONTAL,
            variable=self.brush_size,
            length=160,
        )
        size_slider.grid(row=0, column=1, padx=6, pady=6, sticky="w")

        tk.Label(self.root, text="Color").grid(row=0, column=2, padx=(10, 6), pady=6, sticky="w")
        palette_frame = tk.Frame(self.root)
        palette_frame.grid(row=0, column=3, padx=6, pady=6, sticky="w")
        for color in PALETTE:
            btn = tk.Button(
                palette_frame,
                bg=color,
                width=2,
                command=lambda c=color: self._set_color(c),
            )
            btn.pack(side=tk.LEFT, padx=1)

        tk.Button(self.root, text="Pick", command=self._pick_color).grid(row=0, column=4, padx=6, pady=6)
        tk.Checkbutton(self.root, text="Eraser", variable=self.is_eraser).grid(row=0, column=5, padx=6, pady=6)
        tk.Button(self.root, text="Clear", command=self._clear_canvas).grid(row=0, column=6, padx=6, pady=6)
        tk.Button(self.root, text="Save", command=self._save_image).grid(row=0, column=7, padx=6, pady=6)
        tk.Button(self.root, text="Share", command=self._share_image).grid(row=0, column=8, padx=(6, 10), pady=6)

    def _bind_canvas(self) -> None:
        self.canvas.bind("<ButtonPress-1>", self._start_draw)
        self.canvas.bind("<B1-Motion>", self._draw_motion)
        self.canvas.bind("<ButtonRelease-1>", self._end_draw)

    def _set_color(self, color: str) -> None:
        self.active_color = color
        self.is_eraser.set(False)

    def _pick_color(self) -> None:
        picked = colorchooser.askcolor(color=self.active_color, title="Pick a color")
        if picked and picked[1]:
            self._set_color(picked[1])

    def _current_color(self) -> str:
        return DEFAULT_BG if self.is_eraser.get() else self.active_color

    def _start_draw(self, event) -> None:
        self.last_x = event.x
        self.last_y = event.y

    def _draw_motion(self, event) -> None:
        if self.last_x is None or self.last_y is None:
            self.last_x = event.x
            self.last_y = event.y
            return
        color = self._current_color()
        width = self.brush_size.get()
        self.canvas.create_line(
            self.last_x,
            self.last_y,
            event.x,
            event.y,
            fill=color,
            width=width,
            capstyle=tk.ROUND,
            smooth=True,
            splinesteps=36,
        )
        self.draw.line(
            [self.last_x, self.last_y, event.x, event.y],
            fill=color,
            width=width,
        )
        self.last_x = event.x
        self.last_y = event.y
        self.dirty = True

    def _end_draw(self, _event) -> None:
        self.last_x = None
        self.last_y = None

    def _clear_canvas(self) -> None:
        if not messagebox.askyesno("Clear", "Clear the canvas?"):
            return
        self.canvas.delete("all")
        self._init_image()
        self.dirty = True

    def _save_image(self) -> str | None:
        timestamp = datetime.now(timezone.utc).isoformat().replace(":", "-")
        filename = f"ina_paint_{timestamp}.png"
        path = self.session_dir / filename
        try:
            self.image.save(path)
            self.last_saved_path = path
            self.dirty = False
            self._append_log_entry(path, timestamp)
            self._save_fragment(path, timestamp)
            log_to_statusbox(f"[Paint] Saved {path.name}")
            return str(path)
        except Exception as exc:
            messagebox.showerror("Save failed", f"Could not save image: {exc}")
            return None

    def _share_image(self) -> None:
        path = self._save_image() if self.dirty or self.last_saved_path is None else str(self.last_saved_path)
        if not path:
            return
        caption = simpledialog.askstring("Share", "Caption (optional):", parent=self.root)
        target = "owner_dm"
        if self.can_share_text_channel:
            choice = messagebox.askyesnocancel(
                "Share Target",
                "Send to owner DM? (No sends to the text channel)",
            )
            if choice is None:
                return
            if choice is False:
                target = "text_channel"
        metadata = {
            "source": "ina_paint",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "target": target,
        }
        queued = append_typed_outbox_entry(
            caption or "",
            target=target,
            metadata=metadata,
            allow_empty=True,
            attachment_path=path,
        )
        if queued:
            log_to_statusbox(f"[Paint] Queued share {queued}")
            messagebox.showinfo("Shared", "Queued for sharing.")
        else:
            messagebox.showwarning("Share", "Nothing queued for sharing.")

    def _append_log_entry(self, path: Path, timestamp: str) -> None:
        entry = {
            "timestamp": timestamp,
            "file": str(path),
            "brush_size": self.brush_size.get(),
            "color": self.active_color,
            "eraser": self.is_eraser.get(),
        }
        log = []
        if self.log_path.exists():
            try:
                log = json.loads(self.log_path.read_text(encoding="utf-8"))
            except Exception:
                log = []
        if not isinstance(log, list):
            log = []
        log.append(entry)
        log = log[-200:]
        try:
            self.log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _save_fragment(self, path: Path, timestamp: str) -> None:
        self.fragments_dir.mkdir(parents=True, exist_ok=True)
        frag_id = f"frag_paint_{int(time.time())}"
        snapshot = get_inastate("emotion_snapshot") or {}
        emotions = snapshot.get("values", snapshot) if isinstance(snapshot, dict) else {}
        frag = {
            "id": frag_id,
            "summary": "paint sketch",
            "tags": ["paint", "creative", "visual"],
            "timestamp": timestamp,
            "source": "paint_window",
            "image_file": path.name,
            "image_path": str(path),
            "emotions": emotions,
            "clarity": 0.5,
        }
        frag_path = self.fragments_dir / f"{frag_id}.json"
        try:
            frag_path.write_text(json.dumps(frag, indent=4), encoding="utf-8")
        except Exception:
            pass

    def _on_close(self) -> None:
        if self.dirty:
            if messagebox.askyesno("Save", "Save before closing?"):
                self._save_image()
        self._set_window_state(False)
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    PaintWindow().run()
