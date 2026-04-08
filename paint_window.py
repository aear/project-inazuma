import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
import tkinter as tk
from tkinter import colorchooser, messagebox, simpledialog

from PIL import Image, ImageColor, ImageDraw

from gui_hook import log_to_statusbox
from model_manager import (
    append_typed_outbox_entry,
    get_inastate,
    load_config,
    update_inastate,
)


CANVAS_WIDTH = 900
CANVAS_HEIGHT = 600
DEFAULT_BG = "#dfe3ea"
EMPTY_IMAGE_BG = (255, 255, 255, 0)
DEFAULT_COLOR = "#1b1b1b"
PAINT_API_QUEUE_KEY = "paint_command_queue"
PAINT_API_STATUS_KEY = "paint_api_status"
PAINT_API_LAST_RESULT_KEY = "paint_api_last_result"
PAINT_CANVAS_STATE_KEY = "paint_canvas_state"
PAINT_API_POLL_MS = 250
PAINT_API_MAX_COMMANDS_PER_TICK = 12
PAINT_CANVAS_RECENT_MARKS = 40
PAINT_CANVAS_HISTORY_LIMIT = 240
PAINT_SPATIAL_COLUMNS = 6
PAINT_SPATIAL_ROWS = 4
PAINT_REGION_COLUMNS = 3
PAINT_REGION_ROWS = 3
PAINT_API_PATTERNS = ["circle", "spiral", "wave", "star", "burst"]
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _coerce_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"1", "true", "yes", "on"}:
            return True
        if cleaned in {"0", "false", "no", "off"}:
            return False
    return bool(default)


def _coerce_float(value, default: float, low: float | None = None, high: float | None = None) -> float:
    try:
        number = float(value)
    except Exception:
        number = float(default)
    if low is not None:
        number = max(float(low), number)
    if high is not None:
        number = min(float(high), number)
    return number


def _coerce_int(value, default: int, low: int | None = None, high: int | None = None) -> int:
    number = int(round(_coerce_float(value, default)))
    if low is not None:
        number = max(int(low), number)
    if high is not None:
        number = min(int(high), number)
    return number


def _coerce_brush_size(value, default: int) -> int:
    return _coerce_int(value, default, low=1, high=72)


def _normalise_color(value, default: str = DEFAULT_COLOR) -> str:
    if not isinstance(value, str) or not value.strip():
        value = default
    try:
        rgb = ImageColor.getrgb(value.strip())
    except Exception:
        rgb = ImageColor.getrgb(default)
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def _paint_command_uses_normalized_space(command: dict) -> bool:
    space = str(command.get("space") or command.get("units") or "").strip().lower()
    if space in {"pixel", "pixels", "px", "absolute"}:
        return False
    if space in {"normalized", "normalised", "relative", "unit", "0..1", "0-1"}:
        return True
    if "normalized" in command:
        return _coerce_bool(command.get("normalized"), True)
    if "normalised" in command:
        return _coerce_bool(command.get("normalised"), True)
    return True


def normalise_paint_point(
    raw_point,
    *,
    width: int = CANVAS_WIDTH,
    height: int = CANVAS_HEIGHT,
    normalized: bool = True,
) -> tuple[float, float]:
    if isinstance(raw_point, dict):
        x_raw = raw_point.get("x")
        y_raw = raw_point.get("y")
    elif isinstance(raw_point, (list, tuple)) and len(raw_point) >= 2:
        x_raw = raw_point[0]
        y_raw = raw_point[1]
    else:
        raise ValueError("point must be [x, y] or {'x': x, 'y': y}")
    try:
        x = float(x_raw)
        y = float(y_raw)
    except Exception as exc:
        raise ValueError("point coordinates must be numeric") from exc
    if normalized:
        x *= width
        y *= height
    return (_clamp(x, 0.0, float(width)), _clamp(y, 0.0, float(height)))


def normalise_paint_points(
    raw_points,
    *,
    width: int = CANVAS_WIDTH,
    height: int = CANVAS_HEIGHT,
    normalized: bool = True,
) -> list[tuple[float, float]]:
    if not isinstance(raw_points, list):
        raise ValueError("points must be a list")

    points = [
        normalise_paint_point(item, width=width, height=height, normalized=normalized)
        for item in raw_points
    ]
    if not points:
        raise ValueError("points cannot be empty")
    return points


def _serialise_paint_point(point) -> list[float]:
    return [round(float(point[0]), 2), round(float(point[1]), 2)]


def _paint_points_preview(points, *, limit: int = 8) -> list[list[float]]:
    return [_serialise_paint_point(point) for point in list(points)[:limit]]


def _paint_bounds(points) -> dict | None:
    points = list(points)
    if not points:
        return None
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return {
        "x1": round(min(xs), 2),
        "y1": round(min(ys), 2),
        "x2": round(max(xs), 2),
        "y2": round(max(ys), 2),
    }


def _paint_cell_for_point(
    point,
    *,
    columns: int,
    rows: int,
    width: int = CANVAS_WIDTH,
    height: int = CANVAS_HEIGHT,
) -> tuple[int, int]:
    x = _clamp(float(point[0]), 0.0, float(width - 1))
    y = _clamp(float(point[1]), 0.0, float(height - 1))
    column = min(columns - 1, max(0, int((x / width) * columns)))
    row = min(rows - 1, max(0, int((y / height) * rows)))
    return row, column


def _paint_cells_for_stroke(
    stroke,
    *,
    columns: int,
    rows: int,
    width: int = CANVAS_WIDTH,
    height: int = CANVAS_HEIGHT,
) -> set[tuple[int, int]]:
    points = list(stroke)
    if not points:
        return set()
    if len(points) == 1:
        return {_paint_cell_for_point(points[0], columns=columns, rows=rows, width=width, height=height)}

    cell_width = width / columns
    cell_height = height / rows
    occupied = set()
    for start, end in zip(points, points[1:]):
        dx = float(end[0]) - float(start[0])
        dy = float(end[1]) - float(start[1])
        steps = max(1, int(math.ceil(max(abs(dx) / cell_width, abs(dy) / cell_height) * 2)))
        for idx in range(steps + 1):
            progress = idx / steps
            point = (float(start[0]) + dx * progress, float(start[1]) + dy * progress)
            occupied.add(_paint_cell_for_point(point, columns=columns, rows=rows, width=width, height=height))
    return occupied


def _serialise_canvas_mark(mark: dict) -> dict:
    return {key: value for key, value in mark.items() if not key.startswith("_")}


def _paint_region_name(row: int, column: int) -> str:
    vertical = ["top", "middle", "bottom"][min(PAINT_REGION_ROWS - 1, row)]
    horizontal = ["left", "center", "right"][min(PAINT_REGION_COLUMNS - 1, column)]
    if vertical == "middle" and horizontal == "center":
        return "center"
    if vertical == "middle":
        return horizontal
    if horizontal == "center":
        return vertical
    return f"{vertical}_{horizontal}"


def _paint_spatial_summary(marks: list[dict]) -> dict:
    points = []
    grid_cells = set()
    region_cells = set()
    for mark in marks:
        mark_grid_cells = set()
        mark_region_cells = set()
        mark_points = []
        for stroke in mark.get("_strokes", []):
            mark_points.extend(stroke)
            mark_grid_cells.update(
                _paint_cells_for_stroke(
                    stroke,
                    columns=PAINT_SPATIAL_COLUMNS,
                    rows=PAINT_SPATIAL_ROWS,
                )
            )
            mark_region_cells.update(
                _paint_cells_for_stroke(
                    stroke,
                    columns=PAINT_REGION_COLUMNS,
                    rows=PAINT_REGION_ROWS,
                )
            )
        if mark.get("eraser"):
            grid_cells.difference_update(mark_grid_cells)
            region_cells.difference_update(mark_region_cells)
        else:
            grid_cells.update(mark_grid_cells)
            region_cells.update(mark_region_cells)
            points.extend(mark_points)

    total_cells = PAINT_SPATIAL_COLUMNS * PAINT_SPATIAL_ROWS
    filled_ratio = round(len(grid_cells) / total_cells, 4) if total_cells else 0.0
    region_names = [
        _paint_region_name(row, column)
        for row in range(PAINT_REGION_ROWS)
        for column in range(PAINT_REGION_COLUMNS)
    ]
    regions = {name: 0 for name in region_names}
    for row, column in region_cells:
        regions[_paint_region_name(row, column)] += 1

    return {
        "method": "geometry_grid",
        "fill_ratio": filled_ratio,
        "empty_ratio": round(1.0 - filled_ratio, 4),
        "bounds": _paint_bounds(points),
        "grid": {
            "columns": PAINT_SPATIAL_COLUMNS,
            "rows": PAINT_SPATIAL_ROWS,
            "occupied_count": len(grid_cells),
            "total_cells": total_cells,
            "occupied_cells": [
                {"row": row, "column": column}
                for row, column in sorted(grid_cells)
            ],
        },
        "regions": regions,
    }


def _paint_distance(value, *, normalized: bool, width: int = CANVAS_WIDTH, height: int = CANVAS_HEIGHT, default: float = 0.2) -> float:
    number = _coerce_float(value, default, low=0.0)
    if normalized:
        number *= min(width, height)
    return number


def _build_circle_pattern(
    command: dict,
    *,
    normalized: bool,
    width: int = CANVAS_WIDTH,
    height: int = CANVAS_HEIGHT,
) -> list[list[tuple[float, float]]]:
    center = normalise_paint_point(command.get("center", [0.5, 0.5]), width=width, height=height, normalized=normalized)
    radius = _paint_distance(command.get("radius", 0.22), normalized=normalized, width=width, height=height, default=0.22)
    steps = _coerce_int(command.get("steps", 96), 96, low=12, high=360)
    points = []
    for idx in range(steps + 1):
        angle = (2.0 * math.pi * idx) / steps
        points.append((center[0] + math.cos(angle) * radius, center[1] + math.sin(angle) * radius))
    return [points]


def _build_spiral_pattern(
    command: dict,
    *,
    normalized: bool,
    width: int = CANVAS_WIDTH,
    height: int = CANVAS_HEIGHT,
) -> list[list[tuple[float, float]]]:
    center = normalise_paint_point(command.get("center", [0.5, 0.5]), width=width, height=height, normalized=normalized)
    radius = _paint_distance(command.get("radius", 0.32), normalized=normalized, width=width, height=height, default=0.32)
    turns = _coerce_float(command.get("turns", 3.0), 3.0, low=0.25, high=12.0)
    steps = _coerce_int(command.get("steps", 120), 120, low=12, high=720)
    points = []
    for idx in range(steps + 1):
        progress = idx / steps
        angle = 2.0 * math.pi * turns * progress
        r = radius * progress
        points.append((center[0] + math.cos(angle) * r, center[1] + math.sin(angle) * r))
    return [points]


def _build_wave_pattern(
    command: dict,
    *,
    normalized: bool,
    width: int = CANVAS_WIDTH,
    height: int = CANVAS_HEIGHT,
) -> list[list[tuple[float, float]]]:
    start = normalise_paint_point(command.get("start", [0.12, 0.5]), width=width, height=height, normalized=normalized)
    end = normalise_paint_point(command.get("end", [0.88, 0.5]), width=width, height=height, normalized=normalized)
    amplitude = _paint_distance(command.get("amplitude", 0.08), normalized=normalized, width=width, height=height, default=0.08)
    cycles = _coerce_float(command.get("cycles", 2.0), 2.0, low=0.25, high=16.0)
    steps = _coerce_int(command.get("steps", 120), 120, low=8, high=720)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy) or 1.0
    nx = -dy / length
    ny = dx / length
    points = []
    for idx in range(steps + 1):
        progress = idx / steps
        offset = math.sin(progress * cycles * 2.0 * math.pi) * amplitude
        points.append((start[0] + dx * progress + nx * offset, start[1] + dy * progress + ny * offset))
    return [points]


def _build_star_pattern(
    command: dict,
    *,
    normalized: bool,
    width: int = CANVAS_WIDTH,
    height: int = CANVAS_HEIGHT,
) -> list[list[tuple[float, float]]]:
    center = normalise_paint_point(command.get("center", [0.5, 0.5]), width=width, height=height, normalized=normalized)
    points_count = _coerce_int(command.get("points_count", command.get("points", 5)), 5, low=3, high=16)
    outer = _paint_distance(command.get("radius", 0.28), normalized=normalized, width=width, height=height, default=0.28)
    inner = _paint_distance(command.get("inner_radius", 0.12), normalized=normalized, width=width, height=height, default=0.12)
    vertices = []
    total = points_count * 2
    for idx in range(total + 1):
        radius = outer if idx % 2 == 0 else inner
        angle = (-math.pi / 2.0) + (2.0 * math.pi * idx / total)
        vertices.append((center[0] + math.cos(angle) * radius, center[1] + math.sin(angle) * radius))
    return [vertices]


def _build_burst_pattern(
    command: dict,
    *,
    normalized: bool,
    width: int = CANVAS_WIDTH,
    height: int = CANVAS_HEIGHT,
) -> list[list[tuple[float, float]]]:
    center = normalise_paint_point(command.get("center", [0.5, 0.5]), width=width, height=height, normalized=normalized)
    radius = _paint_distance(command.get("radius", 0.3), normalized=normalized, width=width, height=height, default=0.3)
    inner = _paint_distance(command.get("inner_radius", 0.0), normalized=normalized, width=width, height=height, default=0.0)
    rays = _coerce_int(command.get("rays", 12), 12, low=2, high=96)
    strokes = []
    for idx in range(rays):
        angle = (2.0 * math.pi * idx) / rays
        start = (center[0] + math.cos(angle) * inner, center[1] + math.sin(angle) * inner)
        end = (center[0] + math.cos(angle) * radius, center[1] + math.sin(angle) * radius)
        strokes.append([start, end])
    return strokes


def build_paint_pattern_strokes(command: dict, *, width: int = CANVAS_WIDTH, height: int = CANVAS_HEIGHT) -> list[list[tuple[float, float]]]:
    normalized = _paint_command_uses_normalized_space(command)
    pattern = str(command.get("pattern") or command.get("name") or command.get("action") or "").strip().lower()
    builders = {
        "circle": _build_circle_pattern,
        "spiral": _build_spiral_pattern,
        "wave": _build_wave_pattern,
        "star": _build_star_pattern,
        "burst": _build_burst_pattern,
    }
    builder = builders.get(pattern)
    if builder is None:
        raise ValueError(f"unknown pattern: {pattern or 'missing'}")
    strokes = builder(command, normalized=normalized, width=width, height=height)
    return [
        [(_clamp(x, 0.0, float(width)), _clamp(y, 0.0, float(height))) for x, y in stroke]
        for stroke in strokes
        if stroke
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
        self.canvas_revision = 0
        self.last_inspect_revision = 0
        self.next_mark_id = 1
        self.canvas_marks = []
        self._api_poll_active = True

        self._build_toolbar()
        self._bind_canvas()
        self._set_window_state(True)
        self._schedule_api_poll()

    def _set_window_state(self, is_open: bool) -> None:
        try:
            update_inastate("paint_window_open", is_open)
            self._publish_api_status("ready" if is_open else "closed")
            self._publish_canvas_state("window_open" if is_open else "window_closed", window_open=is_open)
        except Exception:
            pass

    def _init_image(self) -> None:
        self.image = Image.new("RGBA", (CANVAS_WIDTH, CANVAS_HEIGHT), EMPTY_IMAGE_BG)
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

    def _current_image_color(self):
        return EMPTY_IMAGE_BG if self.is_eraser.get() else self.active_color

    def _start_draw(self, event) -> None:
        self.last_x = event.x
        self.last_y = event.y

    def _draw_motion(self, event) -> None:
        if self.last_x is None or self.last_y is None:
            self.last_x = event.x
            self.last_y = event.y
            return
        start = (self.last_x, self.last_y)
        end = (event.x, event.y)
        color = self._current_color()
        image_color = self._current_image_color()
        width = self.brush_size.get()
        self._draw_segment(
            start[0],
            start[1],
            end[0],
            end[1],
            color=color,
            width=width,
            image_color=image_color,
        )
        self._record_canvas_mark(
            {
                "kind": "manual_stroke",
                "source": "manual",
                "points": 2,
                "_strokes": [[start, end]],
                "_image_color": image_color,
                "preview": [_serialise_paint_point(start), _serialise_paint_point(end)],
                "bounds": _paint_bounds([start, end]),
                "brush_size": width,
                "color": color,
                "eraser": self.is_eraser.get(),
            }
        )
        self.last_x = event.x
        self.last_y = event.y

    def _end_draw(self, _event) -> None:
        self.last_x = None
        self.last_y = None

    def _draw_segment(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        color: str,
        width: int,
        image_color=None,
    ) -> None:
        self.canvas.create_line(
            x1,
            y1,
            x2,
            y2,
            fill=color,
            width=width,
            capstyle=tk.ROUND,
            smooth=True,
            splinesteps=36,
        )
        image_fill = image_color if image_color is not None else color
        self.draw.line([x1, y1, x2, y2], fill=image_fill, width=width)
        self.dirty = True

    def _draw_dot(self, x: float, y: float, *, color: str, width: int, image_color=None) -> None:
        radius = max(1.0, width / 2.0)
        bounds = (x - radius, y - radius, x + radius, y + radius)
        self.canvas.create_oval(*bounds, fill=color, outline=color)
        image_fill = image_color if image_color is not None else color
        self.draw.ellipse(bounds, fill=image_fill, outline=image_fill)
        self.dirty = True

    def _reset_canvas(self) -> None:
        self.canvas.delete("all")
        self._init_image()
        self.dirty = True
        self.canvas_revision += 1
        self.canvas_marks = []
        self._publish_canvas_state("cleared")

    def _clear_canvas(self) -> None:
        if not messagebox.askyesno("Clear", "Clear the canvas?"):
            return
        self._reset_canvas()

    def _save_image(self, *, show_errors: bool = True) -> str | None:
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
            self._publish_canvas_state("saved")
            return str(path)
        except Exception as exc:
            if show_errors:
                messagebox.showerror("Save failed", f"Could not save image: {exc}")
            else:
                log_to_statusbox(f"[Paint] API save failed: {exc}")
            return None

    def _publish_api_status(self, status: str, **extra) -> None:
        payload = {
            "timestamp": _utc_now(),
            "window_open": bool(status != "closed"),
            "status": status,
            "queue_key": PAINT_API_QUEUE_KEY,
            "result_key": PAINT_API_LAST_RESULT_KEY,
            "canvas_state_key": PAINT_CANVAS_STATE_KEY,
            "commands": ["stroke", "pattern", "set_brush", "clear", "save", "close", "inspect", "undo"],
            "patterns": PAINT_API_PATTERNS,
            "coordinate_space": "normalized 0..1 by default; use space='pixels' for canvas pixels",
            "workspace_background": DEFAULT_BG,
            "image_background": "transparent",
            "undo_example": {"id": "undo_001", "action": "undo", "count": 1},
            "example": {
                "id": "sketch_001",
                "action": "pattern",
                "pattern": "spiral",
                "center": [0.5, 0.5],
                "radius": 0.32,
                "turns": 3,
                "color": "#d35400",
                "brush_size": 8,
            },
        }
        payload.update(extra)
        try:
            update_inastate(PAINT_API_STATUS_KEY, payload)
        except Exception:
            pass

    def _brush_state(self) -> dict:
        return {
            "size": self.brush_size.get(),
            "color": self.active_color,
            "eraser": self.is_eraser.get(),
        }

    def _canvas_state_payload(self, event: str, **extra) -> dict:
        since_revision = _coerce_int(
            extra.pop("since_revision", self.last_inspect_revision),
            self.last_inspect_revision,
            low=0,
        )
        spatial = _paint_spatial_summary(self.canvas_marks)
        recent_marks = [
            _serialise_canvas_mark(mark)
            for mark in self.canvas_marks[-PAINT_CANVAS_RECENT_MARKS:]
        ]
        delta_marks = [
            _serialise_canvas_mark(mark)
            for mark in self.canvas_marks
            if int(mark.get("revision", 0)) > since_revision
        ]
        payload = {
            "timestamp": _utc_now(),
            "event": event,
            "window_open": bool(extra.pop("window_open", self._api_poll_active)),
            "canvas": {
                "width": CANVAS_WIDTH,
                "height": CANVAS_HEIGHT,
                "workspace_background": DEFAULT_BG,
                "image_background": "transparent",
                "image_mode": getattr(self.image, "mode", None),
            },
            "dirty": self.dirty,
            "last_saved_path": str(self.last_saved_path) if self.last_saved_path else None,
            "revision": self.canvas_revision,
            "revision_delta": {
                "from": since_revision,
                "to": self.canvas_revision,
                "count": max(0, self.canvas_revision - since_revision),
                "mark_ids": [mark.get("mark_id") for mark in delta_marks],
                "marks": delta_marks,
            },
            "mark_count": len(self.canvas_marks),
            "recent_marks": recent_marks,
            "fill_ratio": spatial["fill_ratio"],
            "empty_ratio": spatial["empty_ratio"],
            "spatial": spatial,
            "brush": self._brush_state(),
        }
        payload.update(extra)
        return payload

    def _publish_canvas_state(self, event: str = "state", **extra) -> dict:
        payload = self._canvas_state_payload(event, **extra)
        try:
            update_inastate(PAINT_CANVAS_STATE_KEY, payload)
        except Exception:
            pass
        return payload

    def _record_canvas_mark(self, mark: dict) -> dict:
        self.canvas_revision += 1
        mark_id = str(mark.get("mark_id") or f"paint_mark_{self.next_mark_id:04d}")
        self.next_mark_id += 1
        entry = dict(mark)
        entry["mark_id"] = mark_id
        if entry.get("kind") in {"stroke", "manual_stroke"}:
            entry.setdefault("stroke_id", mark_id)
        elif entry.get("kind") == "pattern":
            entry.setdefault(
                "stroke_ids",
                [f"{mark_id}_stroke_{idx + 1:02d}" for idx, _stroke in enumerate(entry.get("_strokes", []))],
            )
        entry["revision"] = self.canvas_revision
        entry.setdefault("timestamp", _utc_now())
        self.canvas_marks.append(entry)
        self.canvas_marks = self.canvas_marks[-PAINT_CANVAS_HISTORY_LIMIT:]
        return self._publish_canvas_state(entry.get("kind", "mark"))

    def _redraw_canvas_from_marks(self) -> None:
        self.canvas.delete("all")
        self._init_image()
        for mark in self.canvas_marks:
            for stroke in mark.get("_strokes", []):
                self._render_api_points(
                    stroke,
                    color=mark.get("color", DEFAULT_COLOR),
                    width=mark.get("brush_size", self.brush_size.get()),
                    image_color=mark.get("_image_color"),
                )
        self.dirty = True

    def _api_inspect(self, command: dict) -> dict:
        since_revision = _coerce_int(
            command.get("since_revision", self.last_inspect_revision),
            self.last_inspect_revision,
            low=0,
        )
        canvas = self._publish_canvas_state("inspect", since_revision=since_revision)
        self.last_inspect_revision = self.canvas_revision
        return {"status": "ok", "canvas": canvas}

    def _api_undo(self, command: dict) -> dict:
        if not self.canvas_marks:
            canvas = self._publish_canvas_state("undo", undone_count=0, undone_marks=[])
            return {"status": "ok", "undone": 0, "undone_marks": [], "canvas": canvas}

        mark_id = str(command.get("mark_id") or command.get("stroke_id") or "").strip()
        undone = []
        if mark_id:
            index = next(
                (
                    idx for idx, mark in enumerate(self.canvas_marks)
                    if mark.get("mark_id") == mark_id
                    or mark.get("stroke_id") == mark_id
                    or mark_id in mark.get("stroke_ids", [])
                ),
                None,
            )
            if index is None:
                return {"status": "error", "error": f"mark not found: {mark_id}"}
            undone.append(self.canvas_marks.pop(index))
        else:
            count = _coerce_int(command.get("count", 1), 1, low=1, high=20)
            for _idx in range(min(count, len(self.canvas_marks))):
                undone.append(self.canvas_marks.pop())
            undone.reverse()

        self._redraw_canvas_from_marks()
        self.canvas_revision += 1
        undone_marks = [_serialise_canvas_mark(mark) for mark in undone]
        canvas = self._publish_canvas_state(
            "undo",
            undone_count=len(undone_marks),
            undone_marks=undone_marks,
        )
        return {
            "status": "ok",
            "undone": len(undone_marks),
            "undone_marks": undone_marks,
            "revision": self.canvas_revision,
            "canvas": canvas,
        }

    def _schedule_api_poll(self) -> None:
        if not self._api_poll_active:
            return
        try:
            self.root.after(PAINT_API_POLL_MS, self._poll_api_commands)
        except Exception:
            pass

    def _poll_api_commands(self) -> None:
        try:
            self._process_api_queue()
        finally:
            self._schedule_api_poll()

    def _process_api_queue(self) -> None:
        raw_queue = get_inastate(PAINT_API_QUEUE_KEY)
        if raw_queue in (None, [], ""):
            return
        if isinstance(raw_queue, dict):
            queue = [raw_queue]
        elif isinstance(raw_queue, list):
            queue = raw_queue
        else:
            result = {
                "timestamp": _utc_now(),
                "status": "error",
                "error": "paint_command_queue must be a command object or list of command objects",
            }
            update_inastate(PAINT_API_QUEUE_KEY, [])
            update_inastate(PAINT_API_LAST_RESULT_KEY, result)
            self._publish_api_status("error", last_error=result["error"])
            return

        batch = queue[:PAINT_API_MAX_COMMANDS_PER_TICK]
        remaining = queue[PAINT_API_MAX_COMMANDS_PER_TICK:]
        results = [self._process_api_command(command) for command in batch]
        payload = {
            "timestamp": _utc_now(),
            "processed": len(results),
            "remaining": len(remaining),
            "results": results,
        }
        update_inastate(PAINT_API_QUEUE_KEY, remaining)
        update_inastate(PAINT_API_LAST_RESULT_KEY, payload)
        self._publish_api_status("processed", processed=len(results), remaining=len(remaining))

    def _process_api_command(self, command) -> dict:
        if not isinstance(command, dict):
            return {"status": "error", "error": "command must be an object"}
        action = str(command.get("action") or command.get("type") or "").strip().lower()
        command_id = str(command.get("id") or f"paint_cmd_{int(time.time() * 1000)}")
        try:
            if action in {"stroke", "draw", "line"}:
                result = self._api_stroke(command)
            elif action in {"pattern", "shape", *PAINT_API_PATTERNS}:
                result = self._api_pattern(command, action=action)
            elif action == "set_brush":
                result = self._api_set_brush(command)
            elif action == "clear":
                self._reset_canvas()
                result = {"status": "ok", "cleared": True}
            elif action == "save":
                path = self._save_image(show_errors=False)
                result = {"status": "ok" if path else "error", "path": path}
                if not path:
                    result["error"] = "save failed"
            elif action in {"inspect", "state", "snapshot"}:
                result = self._api_inspect(command)
            elif action == "undo":
                result = self._api_undo(command)
            elif action in {"close", "finish", "done"}:
                result = self._api_close(command)
            else:
                result = {"status": "error", "error": f"unknown action: {action or 'missing'}"}
        except Exception as exc:
            result = {"status": "error", "error": str(exc)}
        result["id"] = command_id
        result["action"] = action or None
        result["timestamp"] = _utc_now()
        return result

    def _api_set_brush(self, command: dict) -> dict:
        if "brush_size" in command:
            self.brush_size.set(_coerce_brush_size(command.get("brush_size"), self.brush_size.get()))
        if "color" in command:
            self._set_color(_normalise_color(command.get("color"), self.active_color))
        if "eraser" in command:
            self.is_eraser.set(_coerce_bool(command.get("eraser"), self.is_eraser.get()))
        self._publish_canvas_state("brush")
        return {
            "status": "ok",
            "brush_size": self.brush_size.get(),
            "color": self.active_color,
            "eraser": self.is_eraser.get(),
        }

    def _api_stroke_style(self, command: dict):
        width = _coerce_brush_size(command.get("brush_size"), self.brush_size.get())
        eraser = _coerce_bool(command.get("eraser"), self.is_eraser.get())
        if eraser:
            return width, DEFAULT_BG, EMPTY_IMAGE_BG, eraser
        color = _normalise_color(command.get("color", self.active_color), self.active_color)
        return width, color, color, eraser

    def _render_api_points(
        self,
        points: list[tuple[float, float]],
        *,
        color: str,
        width: int,
        image_color=None,
    ) -> None:
        if len(points) == 1:
            self._draw_dot(points[0][0], points[0][1], color=color, width=width, image_color=image_color)
        else:
            for (x1, y1), (x2, y2) in zip(points, points[1:]):
                self._draw_segment(x1, y1, x2, y2, color=color, width=width, image_color=image_color)

    def _api_stroke(self, command: dict) -> dict:
        width, color, image_color, eraser = self._api_stroke_style(command)
        points = normalise_paint_points(
            command.get("points"),
            width=CANVAS_WIDTH,
            height=CANVAS_HEIGHT,
            normalized=_paint_command_uses_normalized_space(command),
        )
        self._render_api_points(points, color=color, width=width, image_color=image_color)
        self._record_canvas_mark(
            {
                "kind": "stroke",
                "source": "api",
                "command_id": command.get("id"),
                "points": len(points),
                "_strokes": [points],
                "_image_color": image_color,
                "preview": _paint_points_preview(points),
                "bounds": _paint_bounds(points),
                "brush_size": width,
                "color": color,
                "eraser": eraser,
            }
        )

        return {
            "status": "ok",
            "points": len(points),
            "brush_size": width,
            "color": color,
            "eraser": eraser,
            "dirty": self.dirty,
        }

    def _api_pattern(self, command: dict, *, action: str) -> dict:
        if not command.get("pattern") and action in PAINT_API_PATTERNS:
            command = dict(command)
            command["pattern"] = action
        width, color, image_color, eraser = self._api_stroke_style(command)
        strokes = build_paint_pattern_strokes(command)
        for stroke in strokes:
            self._render_api_points(stroke, color=color, width=width, image_color=image_color)
        all_points = [point for stroke in strokes for point in stroke]
        self._record_canvas_mark(
            {
                "kind": "pattern",
                "source": "api",
                "command_id": command.get("id"),
                "pattern": str(command.get("pattern") or action),
                "strokes": len(strokes),
                "points": len(all_points),
                "_strokes": strokes,
                "_image_color": image_color,
                "preview": _paint_points_preview(all_points),
                "bounds": _paint_bounds(all_points),
                "brush_size": width,
                "color": color,
                "eraser": eraser,
            }
        )
        return {
            "status": "ok",
            "pattern": str(command.get("pattern") or action),
            "strokes": len(strokes),
            "points": sum(len(stroke) for stroke in strokes),
            "brush_size": width,
            "color": color,
            "eraser": eraser,
            "dirty": self.dirty,
        }

    def _api_close(self, command: dict) -> dict:
        save_before_close = _coerce_bool(command.get("save"), True)
        saved_path = self._save_image(show_errors=False) if save_before_close and self.dirty else None
        self._api_poll_active = False
        self._set_window_state(False)
        try:
            self.root.destroy()
        except Exception:
            pass
        return {
            "status": "ok",
            "closed": True,
            "saved": bool(saved_path),
            "path": saved_path,
        }

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
        self._api_poll_active = False
        self._set_window_state(False)
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    PaintWindow().run()
