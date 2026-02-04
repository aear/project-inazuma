#!/usr/bin/env python3
"""Reusable audio device selection panel."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Tuple

try:
    from PyQt5 import QtCore, QtWidgets
except Exception:  # pragma: no cover - optional dependency
    QtCore = None
    QtWidgets = None

try:
    import sounddevice as sd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    sd = None


def load_config() -> dict:
    path = Path("config.json")
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_config(config: dict) -> bool:
    path = Path("config.json")
    try:
        path.write_text(json.dumps(config, indent=4), encoding="utf-8")
        return True
    except Exception:
        return False


def _split_default_device(default_device: Any) -> Tuple[Optional[int], Optional[int]]:
    if isinstance(default_device, (list, tuple)) and len(default_device) >= 2:
        try:
            return int(default_device[0]), int(default_device[1])
        except Exception:
            return None, None
    if isinstance(default_device, (int, float)):
        try:
            as_int = int(default_device)
            return as_int, as_int
        except Exception:
            return None, None
    return None, None


def _pick_device_choice(
    config: dict,
    overrides: dict,
    *,
    label: str,
    name_key: str,
    index_key: str,
) -> Optional[Tuple[int, str]]:
    override = overrides.get(label)
    if override == "default":
        return None
    idx = config.get(index_key)
    name = config.get(name_key)
    try:
        idx_val = int(idx) if idx is not None else None
    except Exception:
        idx_val = None
    if idx_val is not None:
        return (idx_val, str(name or f"Device {idx_val}"))
    return None


def _select_combo_value(combo: QtWidgets.QComboBox, choice: Optional[Tuple[int, str]]) -> None:
    if choice is None:
        combo.setCurrentIndex(0)
        return
    target_idx = choice[0]
    for idx in range(combo.count()):
        data = combo.itemData(idx)
        if not data:
            continue
        if isinstance(data, tuple) and data[0] == target_idx:
            combo.setCurrentIndex(idx)
            return
    combo.setCurrentIndex(0)


def _apply_device_choice(
    config: dict,
    overrides: dict,
    *,
    label: str,
    name_key: str,
    index_key: str,
    choice: Optional[Tuple[int, str]],
) -> None:
    if choice is None:
        config[name_key] = "default"
        config[index_key] = None
        overrides[label] = "default"
        return
    idx, name = choice
    config[name_key] = name
    config[index_key] = int(idx)
    overrides.pop(label, None)


class AudioSettingsPanel(QtWidgets.QGroupBox):
    def __init__(self, title: str = "Audio Devices") -> None:
        if QtWidgets is None:
            raise RuntimeError("PyQt5 unavailable")
        super().__init__(title)
        self._input_combo = QtWidgets.QComboBox()
        self._output_combo = QtWidgets.QComboBox()
        self._status_label = QtWidgets.QLabel("")
        self._status_label.setStyleSheet("color: #b0b0b0; font-size: 11px;")
        self._build_ui()

    def _build_ui(self) -> None:
        self.setStyleSheet("QGroupBox { color: #d8d8d8; font-weight: 600; }")
        layout = QtWidgets.QGridLayout(self)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)

        layout.addWidget(QtWidgets.QLabel("Input"), 0, 0)
        layout.addWidget(self._input_combo, 0, 1)
        layout.addWidget(QtWidgets.QLabel("Output"), 1, 0)
        layout.addWidget(self._output_combo, 1, 1)

        apply_btn = QtWidgets.QPushButton("Apply")
        apply_btn.clicked.connect(self.apply_settings)
        layout.addWidget(apply_btn, 2, 0, 1, 2)
        layout.addWidget(self._status_label, 3, 0, 1, 2)

        if not self.refresh_devices():
            self._status_label.setText("sounddevice unavailable; using system defaults.")

    def refresh_devices(self) -> bool:
        self._input_combo.clear()
        self._output_combo.clear()

        self._input_combo.addItem("System Default", None)
        self._output_combo.addItem("System Default", None)

        if sd is None:
            return False

        try:
            devices = sd.query_devices()
            default_in, default_out = _split_default_device(sd.default.device)
        except Exception:
            return False

        for idx, device in enumerate(devices):
            name = device.get("name", f"Device {idx}")
            max_in = device.get("max_input_channels", 0) or 0
            max_out = device.get("max_output_channels", 0) or 0

            if max_in > 0:
                label = f"{idx}: {name}"
                if default_in == idx:
                    label += " (default)"
                self._input_combo.addItem(label, (idx, name))

            if max_out > 0:
                label = f"{idx}: {name}"
                if default_out == idx:
                    label += " (default)"
                self._output_combo.addItem(label, (idx, name))

        self._set_combo_defaults()
        return True

    def _set_combo_defaults(self) -> None:
        config = load_config()
        overrides = config.get("audio_device_overrides") or {}

        input_choice = _pick_device_choice(
            config,
            overrides,
            label="mic_headset",
            name_key="mic_headset_name",
            index_key="mic_headset_index",
        )
        output_choice = _pick_device_choice(
            config,
            overrides,
            label="output_headset",
            name_key="output_headset_name",
            index_key="output_headset_index",
        )

        _select_combo_value(self._input_combo, input_choice)
        _select_combo_value(self._output_combo, output_choice)

    def apply_settings(self) -> None:
        config = load_config()
        overrides = config.get("audio_device_overrides") or {}

        input_data = self._input_combo.currentData()
        output_data = self._output_combo.currentData()

        _apply_device_choice(
            config,
            overrides,
            label="mic_headset",
            name_key="mic_headset_name",
            index_key="mic_headset_index",
            choice=input_data,
        )
        _apply_device_choice(
            config,
            overrides,
            label="output_headset",
            name_key="output_headset_name",
            index_key="output_headset_index",
            choice=output_data,
        )

        config["audio_device_overrides"] = overrides
        if save_config(config):
            self._status_label.setText("Saved. Audio listener will reload shortly.")
        else:
            self._status_label.setText("Failed to write config.json.")
