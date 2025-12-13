import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph.opengl as gl

from gui_hook import log_to_statusbox
from body_schema import get_region_anchors
from model_manager import (
    get_inastate,
    get_running_modules,
    is_dreaming,
    load_config as load_main_config,
)

CONFIG_FILE = "config.json"
DEFAULT_REFRESH_MS = 1200
RENDER_COOLDOWN = 0.35
DEFAULT_DISTANCE = 32.0
BRAIN_RADII = np.array([12.0, 9.0, 7.0], dtype=float)
SHIFT_STEP = 0.25
ROT_STEP_DEG = 2.0

DETAIL_LEVELS = {
    "Low": {"max_nodes": 4000, "max_edges": 12500, "show_edges": False},
    "Medium": {"max_nodes": 8000, "max_edges": 25000, "show_edges": True},
    "High": {"max_nodes": 12000, "max_edges": 50000, "show_edges": True},
}

NETWORK_COLORS: Dict[str, Tuple[float, float, float]] = {
    "emotion": (1.0, 0.38, 0.38),
    "memory_graph": (0.33, 0.82, 1.0),
    "meaning_map": (0.98, 0.78, 0.36),
    "audio": (0.62, 0.92, 0.58),
    "vision": (0.78, 0.58, 1.0),
    "prediction": (0.52, 1.0, 0.76),
    "logic": (0.52, 1.0, 0.76),
    "instinct": (0.98, 0.6, 0.56),
}

TYPE_COLORS: Dict[str, Tuple[float, float, float]] = {
    "sound": (0.32, 0.9, 0.78),
    "token": (0.98, 0.62, 0.24),
    "word": (0.98, 0.85, 0.3),
    "symbol": (0.82, 0.55, 0.98),
    "logic": (0.48, 0.9, 1.0),
}


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def save_config(update: Dict[str, Any]) -> None:
    cfg_path = Path(CONFIG_FILE)
    current: Dict[str, Any] = {}
    if cfg_path.exists():
        try:
            with cfg_path.open("r", encoding="utf-8") as fh:
                loaded = json.load(fh)
                if isinstance(loaded, dict):
                    current.update(loaded)
        except Exception:
            pass

    current.update(update)
    try:
        with cfg_path.open("w", encoding="utf-8") as fh:
            json.dump(current, fh, indent=4)
    except Exception as exc:
        log_to_statusbox(f"[EEG] Failed to persist config: {exc}")


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _parse_timestamp(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        dt = QtCore.QDateTime.fromString(value, QtCore.Qt.ISODate)
        if not dt.isValid():
            # Try replacing Z with offset
            dt = QtCore.QDateTime.fromString(value.replace("Z", "+00:00"), QtCore.Qt.ISODate)
        if dt.isValid():
            return dt.toSecsSinceEpoch()
    except Exception:
        return None
    return None


class BrainDataLoader:
    """
    Collects neuron/synapse data from existing runtime JSON sources without
    duplicating business logic. Positions are taken when available, otherwise
    derived from vectors or stable random placement inside a brain-like volume.
    """

    def __init__(self, child: str):
        self.child = child
        self.memory_root = Path("AI_Children") / child / "memory"
        self.neural_root = self.memory_root / "neural"
        schema = _load_json(Path("body_schema.json")) or {}
        bounds = schema.get("body_bounds") if isinstance(schema, dict) else None
        center = bounds.get("center") if isinstance(bounds, dict) else None
        try:
            self.origin_offset = -np.array([float(center[i]) for i in range(3)], dtype=float)
        except Exception:
            self.origin_offset = np.zeros(3, dtype=float)
        self.manual_offset = np.zeros(3, dtype=float)
        self.manual_rotation = np.zeros(3, dtype=float)  # yaw, pitch, roll in radians

    def load(self) -> Dict[str, Any]:
        neurons: List[Dict[str, Any]] = []
        synapses: List[Dict[str, Any]] = []

        state = self._load_state_snapshot()

        # Primary neural map
        map_nodes, map_edges = self._load_neural_map(self.neural_root / "neural_memory_map.json", "memory_graph")
        neurons.extend(map_nodes)
        synapses.extend(map_edges)

        # Logic / prediction network
        logic_nodes, logic_edges = self._load_neural_map(self.neural_root / "logic_neural_map.json", "logic")
        neurons.extend(logic_nodes)
        synapses.extend(logic_edges)

        # Typed neural graph (sound/token/word/symbol)
        typed_nodes, typed_edges = self._load_typed_graph(self.neural_root / "typed_neural_graph.json")
        neurons.extend(typed_nodes)
        synapses.extend(typed_edges)

        return {"neurons": neurons, "synapses": synapses, "state": state}

    def _load_state_snapshot(self) -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {"running_modules": get_running_modules(), "dreaming": is_dreaming()}
        emotion = get_inastate("emotion_snapshot") or {}
        current_mode = emotion.get("mode") or get_inastate("mode")
        values = emotion.get("values") if isinstance(emotion, dict) else None
        intensity = 0.0
        if isinstance(values, dict):
            for key in ("intensity", "_core_arousal", "_core_energy"):
                if isinstance(values.get(key), (int, float)):
                    intensity = max(intensity, float(values[key]))
        snapshot["mode"] = current_mode or "unknown"
        snapshot["emotion_intensity"] = clamp((intensity + 1.0) / 2.0) if intensity else 0.0
        return snapshot

    def _load_neural_map(self, path: Path, fallback_network: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        data = _load_json(path) or {}
        raw_nodes = data.get("neurons", [])
        raw_edges = data.get("synapses", [])
        neurons: List[Dict[str, Any]] = []
        synapses: List[Dict[str, Any]] = []

        for entry in raw_nodes:
            node = self._coerce_neuron(entry, fallback_network, seed_hint=path.name)
            if node:
                neurons.append(node)

        for entry in raw_edges:
            edge = self._coerce_edge(entry, fallback_network)
            if edge:
                synapses.append(edge)

        return neurons, synapses

    def _load_typed_graph(self, path: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        data = _load_json(path) or {}
        nodes_raw = data.get("nodes", {})
        edges_raw = data.get("edges", {})
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []

        if isinstance(nodes_raw, dict):
            iterable = nodes_raw.values()
        elif isinstance(nodes_raw, list):
            iterable = nodes_raw
        else:
            iterable = []

        for entry in iterable:
            if not isinstance(entry, dict):
                continue
            ntype = entry.get("type") or entry.get("node_type") or ""
            if ntype == "sound":
                network = "audio"
            elif ntype in ("token", "word", "symbol"):
                network = "meaning_map"
            else:
                network = "memory_graph"
            node = self._coerce_neuron(entry, network, seed_hint=path.name)
            if node:
                nodes.append(node)

        if isinstance(edges_raw, dict):
            edge_iter = edges_raw.values()
        elif isinstance(edges_raw, list):
            edge_iter = edges_raw
        else:
            edge_iter = []

        for entry in edge_iter:
            if not isinstance(entry, dict):
                continue
            rel = entry.get("relation") or ""
            net_hint = "audio" if rel.startswith("sound") else "meaning_map"
            edge = self._coerce_edge(entry, net_hint)
            if edge:
                edges.append(edge)

        return nodes, edges

    def _coerce_neuron(self, entry: Dict[str, Any], default_network: str, seed_hint: str) -> Optional[Dict[str, Any]]:
        neuron_id = entry.get("id") or entry.get("neuron_id")
        if not neuron_id:
            return None

        network_type = entry.get("network_type") or default_network
        pos = self._position_from_entry(entry, seed_hint=seed_hint + neuron_id)
        activation = self._activation_from_entry(entry)
        last_used = _parse_timestamp(entry.get("last_used"))
        label = entry.get("symbol") or entry.get("label") or entry.get("description") or ""
        return {
            "id": neuron_id,
            "pos": pos,
            "activation": activation,
            "network_type": network_type,
            "node_type": entry.get("type") or entry.get("node_type"),
            "label": label,
            "last_used": last_used or 0.0,
            "strength": float(entry.get("weight", 0.0)) if isinstance(entry.get("weight"), (int, float)) else 0.0,
            "tags": entry.get("tags", []),
        }

    def _coerce_edge(self, entry: Dict[str, Any], default_network: str) -> Optional[Dict[str, Any]]:
        source = entry.get("source") or entry.get("from_neuron_id")
        target = entry.get("target") or entry.get("to_neuron_id")
        if not source or not target:
            return None
        weight = entry.get("weight", entry.get("strength", 0.0))
        try:
            w_val = float(weight)
        except Exception:
            w_val = 0.0

        direction = entry.get("direction")
        dir_vec = direction if isinstance(direction, (list, tuple)) else None

        return {
            "source": source,
            "target": target,
            "weight": w_val,
            "network_type": entry.get("network_type") or default_network,
            "direction": dir_vec,
            "relation": entry.get("relation") or entry.get("type") or "",
        }

    def _activation_from_entry(self, entry: Dict[str, Any]) -> float:
        for key in ("activation_level", "activation", "activity"):
            if isinstance(entry.get(key), (int, float)):
                return clamp(float(entry[key]))
        if isinstance(entry.get("uses"), (int, float)):
            return clamp(math.log1p(float(entry["uses"])) / 10.0)
        if entry.get("activation_history"):
            try:
                values = [float(v) for v in entry.get("activation_history", []) if isinstance(v, (int, float))]
                if values:
                    return clamp(sum(values[-10:]) / max(1, len(values[-10:])))
            except Exception:
                pass
        ts = _parse_timestamp(entry.get("last_used"))
        if ts:
            age = max(1.0, time.time() - ts)
            return clamp(math.exp(-age / 3600.0))
        return 0.25

    def _position_from_entry(self, entry: Dict[str, Any], seed_hint: str) -> np.ndarray:
        # Priority: explicit position -> vector projection -> stable random in ellipsoid
        pos = entry.get("position")
        if isinstance(pos, (list, tuple)) and len(pos) >= 3:
            coords = [float(x) for x in pos[:3]]
            return self._shift_to_origin(np.array(coords, dtype=float))

        vec = entry.get("vector") or entry.get("embedding_hint") or entry.get("predicted_vector")
        if isinstance(vec, (list, tuple)) and len(vec) >= 3:
            coords = [float(x) for x in vec[:3]]
            return self._fit_to_brain(coords)

        rng = random.Random(hash(seed_hint) & 0xFFFFFFFF)
        theta = rng.uniform(0, 2 * math.pi)
        phi = rng.uniform(0, math.pi)
        r = rng.uniform(0.35, 1.0)
        x = r * math.sin(phi) * math.cos(theta)
        y = r * math.sin(phi) * math.sin(theta)
        z = r * math.cos(phi)
        return self._fit_to_brain([x, y, z])

    def _fit_to_brain(self, coords: List[float]) -> np.ndarray:
        vec = np.array(coords[:3], dtype=float)
        if vec.shape[0] < 3:
            vec = np.pad(vec, (0, 3 - vec.shape[0]))
        norm = np.linalg.norm(vec) + 1e-6
        unit = vec / norm
        return self._shift_to_origin(unit * BRAIN_RADII)

    def _shift_to_origin(self, vec: np.ndarray) -> np.ndarray:
        rotated = self._apply_rotation(vec)
        return rotated + self.origin_offset + self.manual_offset

    def _apply_rotation(self, vec: np.ndarray) -> np.ndarray:
        if vec.shape[0] < 3:
            vec = np.pad(vec, (0, 3 - vec.shape[0]))
        yaw, pitch, roll = self.manual_rotation  # yaw=z, pitch=y, roll=x
        cz, sz = math.cos(yaw), math.sin(yaw)
        cy, sy = math.cos(pitch), math.sin(pitch)
        cx, sx = math.cos(roll), math.sin(roll)
        # R = Rz(yaw) * Ry(pitch) * Rx(roll)
        rot = np.array(
            [
                [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
                [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
                [-sy, cy * sx, cy * cx],
            ],
            dtype=float,
        )
        return rot @ vec


class EEGWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ina - Neural Space EEG")
        self.config = load_main_config()
        self.current_child = self.config.get("current_child", "default_child")
        self.detail_level = self.config.get("eeg_detail_level", "Medium")
        if self.detail_level not in DETAIL_LEVELS:
            self.detail_level = "Medium"
        self.last_render = 0.0
        self.loader = BrainDataLoader(self.current_child)
        extra_offset = self.config.get("eeg_position_offset") or self.config.get("eeg_body_offset")
        if isinstance(extra_offset, (list, tuple)) and len(extra_offset) >= 3:
            try:
                self.loader.manual_offset = np.array([float(extra_offset[i]) for i in range(3)], dtype=float)
            except Exception:
                self.loader.manual_offset = np.zeros(3, dtype=float)
        rotation_deg = self.config.get("eeg_rotation_deg") or self.config.get("eeg_rotation")
        if isinstance(rotation_deg, (list, tuple)) and len(rotation_deg) >= 3:
            try:
                self.loader.manual_rotation = np.radians(
                    [float(rotation_deg[0]), float(rotation_deg[1]), float(rotation_deg[2])]
                )
            except Exception:
                self.loader.manual_rotation = np.zeros(3, dtype=float)
        self.node_item: Optional[gl.GLScatterPlotItem] = None
        self.edge_item: Optional[gl.GLLinePlotItem] = None
        self.type_buttons: Dict[str, QtWidgets.QPushButton] = {}
        self.network_buttons: Dict[str, QtWidgets.QPushButton] = {}

        self._init_ui()
        self._apply_geometry()
        self._start_timers()

    def _init_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        self.setLayout(layout)

        controls = QtWidgets.QHBoxLayout()
        layout.addLayout(controls)

        detail_label = QtWidgets.QLabel("Detail:")
        self.detail_combo = QtWidgets.QComboBox()
        self.detail_combo.addItems(list(DETAIL_LEVELS.keys()))
        self.detail_combo.setCurrentText(self.detail_level)
        self.detail_combo.currentTextChanged.connect(self._on_detail_changed)

        self.reset_button = QtWidgets.QPushButton("Reset view")
        self.reset_button.clicked.connect(self._reset_view)

        self.status_label = QtWidgets.QLabel("")
        self.status_label.setStyleSheet("color: #9aa0a6;")

        controls.addWidget(detail_label)
        controls.addWidget(self.detail_combo)
        controls.addSpacing(12)
        controls.addWidget(self.reset_button)
        controls.addStretch()
        controls.addWidget(self.status_label)

        self.legend_label = QtWidgets.QLabel("")
        self.legend_label.setTextFormat(QtCore.Qt.RichText)
        self.legend_label.setWordWrap(True)
        self.legend_label.setStyleSheet(
            "background: rgba(0, 0, 0, 140); color: #e8eaed; padding: 6px 8px; "
            "border-radius: 8px; font-size: 11px;"
        )

        layout.addWidget(self.legend_label)

        self.view = gl.GLViewWidget()
        self.view.opts["distance"] = DEFAULT_DISTANCE
        self.view.setBackgroundColor(QtGui.QColor(8, 10, 14))
        self.view.setCameraPosition(distance=DEFAULT_DISTANCE, elevation=18, azimuth=40)

        self.grid = gl.GLGridItem()
        self.grid.scale(2, 2, 1)
        self.grid.setSize(30, 30)
        self.grid.setDepthValue(10)
        self.view.addItem(self.grid)

        container = QtWidgets.QWidget()
        stack = QtWidgets.QStackedLayout(container)
        stack.setStackingMode(QtWidgets.QStackedLayout.StackAll)
        stack.addWidget(self.view)

        overlay = QtWidgets.QWidget()
        overlay.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        overlay.setStyleSheet("background: transparent;")
        overlay_layout = QtWidgets.QVBoxLayout(overlay)
        overlay_layout.setContentsMargins(12, 12, 12, 12)
        overlay_layout.setSpacing(6)

        self.info_label = QtWidgets.QLabel("")
        self.info_label.setTextFormat(QtCore.Qt.RichText)
        self.info_label.setStyleSheet(
            "background: rgba(0, 0, 0, 140); color: #e8eaed; padding: 8px 10px; border-radius: 8px; font-size: 12px;"
        )
        overlay_layout.addWidget(self.info_label, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        overlay_layout.addStretch()
        self._add_body_wireframe(overlay_layout)

        stack.addWidget(overlay)
        layout.addWidget(container)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # type: ignore[override]
        key = event.key()
        handled = False
        if key == QtCore.Qt.Key_Left:
            self._nudge_offset(dx=-SHIFT_STEP)
            handled = True
        elif key == QtCore.Qt.Key_Right:
            self._nudge_offset(dx=SHIFT_STEP)
            handled = True
        elif key == QtCore.Qt.Key_Up:
            self._nudge_offset(dy=SHIFT_STEP)
            handled = True
        elif key == QtCore.Qt.Key_Down:
            self._nudge_offset(dy=-SHIFT_STEP)
            handled = True
        elif key == QtCore.Qt.Key_PageUp:
            self._nudge_offset(dz=SHIFT_STEP)
            handled = True
        elif key == QtCore.Qt.Key_PageDown:
            self._nudge_offset(dz=-SHIFT_STEP)
            handled = True
        elif key == QtCore.Qt.Key_J:
            self._nudge_rotation(dyaw=-ROT_STEP_DEG)
            handled = True
        elif key == QtCore.Qt.Key_L:
            self._nudge_rotation(dyaw=ROT_STEP_DEG)
            handled = True
        elif key == QtCore.Qt.Key_I:
            self._nudge_rotation(dpitch=ROT_STEP_DEG)
            handled = True
        elif key == QtCore.Qt.Key_K:
            self._nudge_rotation(dpitch=-ROT_STEP_DEG)
            handled = True
        elif key == QtCore.Qt.Key_U:
            self._nudge_rotation(droll=-ROT_STEP_DEG)
            handled = True
        elif key == QtCore.Qt.Key_O:
            self._nudge_rotation(droll=ROT_STEP_DEG)
            handled = True
        elif key == QtCore.Qt.Key_R:
            self._reset_alignment()
            handled = True

        if handled:
            self.refresh_scene_force()
            self._persist_alignment()
            return
        super().keyPressEvent(event)

    def _add_body_wireframe(self, overlay_layout: QtWidgets.QVBoxLayout) -> None:
        anchors = get_region_anchors()
        offset = getattr(self.loader, "origin_offset", np.zeros(3, dtype=float)) + getattr(
            self.loader, "manual_offset", np.zeros(3, dtype=float)
        )
        schema = _load_json(Path("body_schema.json")) or {}
        regions = schema.get("body_regions", []) if isinstance(schema, dict) else []
        centers: Dict[str, List[float]] = {}
        for region in regions:
            rid = region.get("id")
            anchor = anchors.get(rid, {"center": [0.0, 0.0, 0.0]})
            base_center = np.array(anchor.get("center", [0.0, 0.0, 0.0]), dtype=float)
            center = self.loader._apply_rotation(base_center) + offset
            centers[rid] = center.tolist()

        segments = []
        colors = []
        for region in regions:
            rid = region.get("id")
            parent = region.get("parent")
            if parent and parent != "root" and parent in centers and rid in centers:
                segments.extend([centers[parent], centers[rid]])
                colors.extend(
                    [
                        (0.25, 0.32, 0.38, 0.4),
                        (0.25, 0.32, 0.38, 0.75),
                    ]
                )

        if segments:
            body_lines = gl.GLLinePlotItem(
                pos=np.array(segments),
                color=np.array(colors),
                width=2.0,
                mode="lines",
                antialias=True,
            )
            body_lines.setDepthValue(5)
            self.view.addItem(body_lines)

        if centers:
            pts = gl.GLScatterPlotItem(
                pos=np.array(list(centers.values())),
                size=np.array([0.6] * len(centers)),
                color=np.array([(0.7, 0.86, 1.0, 0.9)] * len(centers)),
                pxMode=False,
            )
            pts.setDepthValue(6)
            self.view.addItem(pts)

        self._build_filter_buttons(overlay_layout)

    def _build_filter_buttons(self, overlay_layout: QtWidgets.QVBoxLayout) -> None:
        def make_chip(title: str, color: Tuple[float, float, float]) -> QtWidgets.QPushButton:
            btn = QtWidgets.QPushButton(title)
            btn.setCheckable(True)
            btn.setChecked(True)
            rgb = tuple(int(c * 255) for c in color)
            btn.setStyleSheet(
                "QPushButton {"
                f"background-color: rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 210); "
                "color: #0b0c10; border: 1px solid rgba(0,0,0,60); "
                "border-radius: 10px; padding: 4px 8px; font-size: 11px;"
                "}"
                "QPushButton:checked {"
                f"background-color: rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 255); color: #0b0c10;"
                "}"
                "QPushButton:unchecked {"
                "background-color: rgba(120, 130, 140, 180); color: #e8eaed;"
                "}"
            )
            return btn

        filter_box = QtWidgets.QVBoxLayout()
        filter_box.setSpacing(6)
        filter_box.setContentsMargins(0, 0, 0, 0)

        type_row = QtWidgets.QHBoxLayout()
        type_row.setSpacing(4)
        type_label = QtWidgets.QLabel("Types:")
        type_label.setStyleSheet("color: #e8eaed; font-weight: bold;")
        type_row.addWidget(type_label)
        for t, color in TYPE_COLORS.items():
            btn = make_chip(t, color)
            btn.clicked.connect(self.refresh_scene_force)
            self.type_buttons[t] = btn
            type_row.addWidget(btn)
        type_row.addStretch()

        net_row = QtWidgets.QHBoxLayout()
        net_row.setSpacing(4)
        net_label = QtWidgets.QLabel("Networks:")
        net_label.setStyleSheet("color: #e8eaed; font-weight: bold;")
        net_row.addWidget(net_label)
        for net, color in NETWORK_COLORS.items():
            btn = make_chip(net, color)
            btn.clicked.connect(self.refresh_scene_force)
            self.network_buttons[net] = btn
            net_row.addWidget(btn)
        net_row.addStretch()

        filter_box.addLayout(type_row)
        filter_box.addLayout(net_row)

        overlay_layout.addLayout(filter_box)

    def _nudge_offset(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> None:
        self.loader.manual_offset = self.loader.manual_offset + np.array([dx, dy, dz], dtype=float)

    def _nudge_rotation(self, dyaw: float = 0.0, dpitch: float = 0.0, droll: float = 0.0) -> None:
        delta_rad = np.radians([dyaw, dpitch, droll])
        self.loader.manual_rotation = self.loader.manual_rotation + delta_rad

    def _reset_alignment(self) -> None:
        self.loader.manual_offset = np.zeros(3, dtype=float)
        self.loader.manual_rotation = np.zeros(3, dtype=float)
        self._persist_alignment()

    def _start_timers(self) -> None:
        self.refresh_timer = QtCore.QTimer(self)
        self.refresh_timer.setInterval(self.config.get("eeg_refresh_ms", DEFAULT_REFRESH_MS))
        self.refresh_timer.timeout.connect(self.refresh_scene)
        self.refresh_timer.start()
        QtCore.QTimer.singleShot(200, self.refresh_scene)

    def _apply_geometry(self) -> None:
        geom = self.config.get("eeg_window_geometry")
        if isinstance(geom, str):
            try:
                parts = [int(p) for p in geom.replace("+", "x").split("x") if p]
                if len(parts) == 4:
                    w, h, x, y = parts
                    self.setGeometry(x, y, w, h)
                elif len(parts) == 3:
                    w, h, x = parts
                    self.resize(w, h)
                    self.move(x, self.y())
            except Exception:
                pass

    def _persist_geometry(self) -> None:
        g = self.geometry()
        geom_str = f"{g.width()}x{g.height()}x{g.x()}x{g.y()}"
        save_config({"eeg_window_geometry": geom_str, "eeg_detail_level": self.detail_level})

    def _persist_alignment(self) -> None:
        save_config(
            {
                "eeg_position_offset": self.loader.manual_offset.tolist(),
                "eeg_rotation_deg": list(np.degrees(self.loader.manual_rotation)),
            }
        )

    def _on_detail_changed(self, value: str) -> None:
        self.detail_level = value if value in DETAIL_LEVELS else "Medium"
        self.refresh_scene(force=True)

    def _reset_view(self) -> None:
        self.view.setCameraPosition(distance=DEFAULT_DISTANCE, elevation=18, azimuth=40)

    def refresh_scene(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self.last_render < RENDER_COOLDOWN:
            return
        self.last_render = now
        try:
            payload = self.loader.load()
            self._render_payload(payload)
        except Exception as exc:
            self.status_label.setText("Render error")
            log_to_statusbox(f"[EEG] Render failed: {exc}")

    def refresh_scene_force(self) -> None:
        self.refresh_scene(force=True)

    def _render_payload(self, payload: Dict[str, Any]) -> None:
        neurons = payload.get("neurons") or []
        edges = payload.get("synapses") or []
        state = payload.get("state") or {}

        if not neurons:
            self._clear_scene()
            self.info_label.setText(
                "<b>No neural space data available yet.</b><br/>Waiting for maps or activity to load."
            )
            self.legend_label.setText("")
            self.status_label.setText("Idle")
            return

        settings = DETAIL_LEVELS.get(self.detail_level, DETAIL_LEVELS["Medium"])
        neurons = self._apply_filters(neurons)
        sampled_neurons = self._sample_neurons(neurons, settings["max_nodes"])
        sampled_edges = self._sample_edges(edges, settings["max_edges"], sampled_neurons, settings["show_edges"])

        pos_map = {n["id"]: n["pos"] for n in sampled_neurons}

        self._draw_nodes(sampled_neurons, state)
        self._draw_edges(sampled_edges, pos_map)
        self._update_overlay(sampled_neurons, neurons, sampled_edges, edges, state)

    def _sample_neurons(self, neurons: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        ranked = sorted(
            neurons,
            key=lambda n: (
                n.get("activation", 0.0),
                n.get("last_used", 0.0),
                n.get("strength", 0.0),
            ),
            reverse=True,
        )
        if len(ranked) <= limit:
            return ranked
        return ranked[:limit]

    def _sample_edges(
        self,
        edges: List[Dict[str, Any]],
        limit: int,
        neurons: List[Dict[str, Any]],
        allow_edges: bool,
    ) -> List[Dict[str, Any]]:
        if not allow_edges or limit <= 0:
            return []
        ids = {n["id"] for n in neurons}
        filtered = [e for e in edges if e.get("source") in ids and e.get("target") in ids]
        ranked = sorted(filtered, key=lambda e: e.get("weight", 0.0), reverse=True)
        return ranked[:limit] if len(ranked) > limit else ranked

    def _apply_filters(self, neurons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        enabled_types = {k for k, b in self.type_buttons.items() if b.isChecked()}
        enabled_networks = {k for k, b in self.network_buttons.items() if b.isChecked()}
        if not enabled_types and not enabled_networks:
            return neurons

        filtered: List[Dict[str, Any]] = []
        for n in neurons:
            node_type = (n.get("node_type") or "").lower()
            network = n.get("network_type", "memory_graph")

            type_ok = (not enabled_types) or (node_type in enabled_types) or (node_type == "" and network in enabled_networks)
            net_ok = (not enabled_networks) or (network in enabled_networks)

            if type_ok and net_ok:
                filtered.append(n)
        return filtered

    def _color_for_network(self, network: str) -> Tuple[float, float, float]:
        if network in NETWORK_COLORS:
            return NETWORK_COLORS[network]
        return (0.7, 0.78, 0.86)

    def _color_to_hex(self, color: Tuple[float, float, float]) -> str:
        r = int(clamp(color[0]) * 255)
        g = int(clamp(color[1]) * 255)
        b = int(clamp(color[2]) * 255)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _build_color_line(self, title: str, mapping: Dict[str, Tuple[float, float, float]], enabled: List[str]) -> str:
        entries = []
        for key in sorted(mapping.keys()):
            hex_color = self._color_to_hex(mapping[key])
            suffix = "" if not enabled or key in enabled else " (off)"
            entries.append(f'<span style="color:{hex_color};">&#11044;</span> {key}{suffix}')
        return f"<b>{title}</b>: " + "  ".join(entries)

    def _color_for_node(self, node: Dict[str, Any]) -> Tuple[float, float, float, str, str]:
        node_type = (node.get("node_type") or "").lower()
        if node_type in TYPE_COLORS:
            base = TYPE_COLORS[node_type]
            return base, node_type, "type"
        network = node.get("network_type", "memory_graph")
        return self._color_for_network(network), network, "network"

    def _draw_nodes(self, neurons: List[Dict[str, Any]], state: Dict[str, Any]) -> None:
        if self.node_item:
            self.view.removeItem(self.node_item)
            self.node_item = None

        positions = np.array([n["pos"] for n in neurons])
        sizes = []
        colors = []

        emotion_intensity = state.get("emotion_intensity", 0.0)
        pulse = 1.0 + 0.25 * emotion_intensity * math.sin(time.time() * 2.0)

        for n in neurons:
            activation = clamp(float(n.get("activation", 0.0)))
            base_color, _, _ = self._color_for_node(n)
            network = n.get("network_type", "memory_graph")
            base_r, base_g, base_b = base_color
            tint = pulse if network == "emotion" else 1.0
            alpha = clamp(0.35 + 0.55 * activation)
            color = (
                clamp(base_r * (0.65 + 0.45 * activation) * tint),
                clamp(base_g * (0.65 + 0.45 * activation) * tint),
                clamp(base_b * (0.65 + 0.45 * activation) * tint),
                alpha,
            )
            size = 0.5 + 1.8 * activation
            sizes.append(size)
            colors.append(color)

        self.node_item = gl.GLScatterPlotItem(
            pos=positions,
            size=np.array(sizes),
            color=np.array(colors),
            pxMode=False,
        )
        self.view.addItem(self.node_item)

    def _draw_edges(self, edges: List[Dict[str, Any]], pos_map: Dict[str, np.ndarray]) -> None:
        if self.edge_item:
            self.view.removeItem(self.edge_item)
            self.edge_item = None
        if not edges:
            return

        segments = []
        colors = []
        for edge in edges:
            src = pos_map.get(edge.get("source"))
            dst = pos_map.get(edge.get("target"))
            if src is None or dst is None:
                continue
            color_base = self._color_for_network(edge.get("network_type", "memory_graph"))
            alpha = 0.22 + 0.4 * clamp(float(edge.get("weight", 0.0)))
            color = (color_base[0], color_base[1], color_base[2], alpha)
            segments.extend([src, dst])
            colors.extend([color, color])

        if not segments:
            return

        self.edge_item = gl.GLLinePlotItem(
            pos=np.array(segments),
            color=np.array(colors),
            width=1.2,
            mode="lines",
            antialias=True,
        )
        self.view.addItem(self.edge_item)

    def _update_overlay(
        self,
        sampled_neurons: List[Dict[str, Any]],
        all_neurons: List[Dict[str, Any]],
        sampled_edges: List[Dict[str, Any]],
        all_edges: List[Dict[str, Any]],
        state: Dict[str, Any],
    ) -> None:
        mode = state.get("mode", "unknown")
        dreaming = state.get("dreaming", False)
        status_parts = [
            f"Mode: {mode}" + (" (dreaming)" if dreaming else ""),
            f"Detail: {self.detail_level}",
            f"Rendering {len(sampled_neurons)}/{len(all_neurons)} neurons",
        ]
        if self.detail_level != "Low":
            status_parts.append(f"{len(sampled_edges)}/{len(all_edges)} synapses")

        self.status_label.setText(" | ".join(status_parts))

        info_lines = [
            f"<b>Mode:</b> {mode}{' (dream)' if dreaming else ''}",
            f"<b>Detail:</b> {self.detail_level}",
            f"<b>Neurons:</b> {len(sampled_neurons)} of {len(all_neurons)}",
            f"<b>Synapses:</b> {len(sampled_edges)} of {len(all_edges)}",
        ]

        # Alignment readout
        offset = getattr(self.loader, "manual_offset", np.zeros(3, dtype=float))
        rotation = getattr(self.loader, "manual_rotation", np.zeros(3, dtype=float))
        offset_str = ", ".join(f"{v:.2f}" for v in offset)
        rotation_deg = np.degrees(rotation)
        rotation_str = ", ".join(f"{v:.1f}°" for v in rotation_deg)
        info_lines.append(f"<b>Offset:</b> [{offset_str}]")
        info_lines.append(f"<b>Rotation:</b> [{rotation_str}] (yaw, pitch, roll)")

        modules = state.get("running_modules") or []
        if modules:
            shown = ", ".join(modules[:6])
            more = "" if len(modules) <= 6 else f" (+{len(modules)-6} more)"
            info_lines.append(f"<b>Modules:</b> {shown}{more}")

        self.info_label.setText("<br/>".join(info_lines))

        active_networks = {}
        active_types = {}
        for n in sampled_neurons:
            net = n.get("network_type", "memory_graph")
            if net not in active_networks:
                color = self._color_for_network(net)
                active_networks[net] = self._color_to_hex(color)

            node_type = (n.get("node_type") or "").lower()
            if node_type in TYPE_COLORS and node_type not in active_types:
                color = TYPE_COLORS[node_type]
                active_types[node_type] = self._color_to_hex(color)

        legend_lines = []
        enabled_types = [t for t, b in self.type_buttons.items() if b.isChecked()]
        enabled_nets = [n for n, b in self.network_buttons.items() if b.isChecked()]
        if active_types:
            legend_lines.append("<b>Active types</b>: " + "  ".join(
                f'<span style="color:{active_types[t]};">&#11044;</span> {t}{" (off)" if t not in enabled_types else ""}'
                for t in sorted(active_types.keys())
            ))
        if active_networks:
            legend_lines.append("<b>Active networks</b>: " + "  ".join(
                f'<span style="color:{active_networks[net]};">&#11044;</span> {net}{" (off)" if net not in enabled_nets else ""}'
                for net in sorted(active_networks.keys())
            ))

        legend_lines.append('<span style="color:#9aa0a6;">Colour code</span>')
        legend_lines.append(self._build_color_line("Types", TYPE_COLORS, enabled_types))
        legend_lines.append(self._build_color_line("Networks", NETWORK_COLORS, enabled_nets))

        self.legend_label.setText("<br/>".join(legend_lines))

    def _clear_scene(self) -> None:
        if self.node_item:
            self.view.removeItem(self.node_item)
            self.node_item = None
        if self.edge_item:
            self.view.removeItem(self.edge_item)
            self.edge_item = None

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
        self._persist_geometry()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = EEGWindow()
    window.resize(1100, 800)
    window.show()
    sys.exit(app.exec_())
