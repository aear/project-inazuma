"""
audio_device_window.py

Tkinter window that lists available audio input/output devices (via
sounddevice) and highlights the entries that line up with the names/indices
stored in config.json. This helps verify Ina is pointed at the right devices.
"""

import json
import os
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Any, Dict, List, Optional, Tuple

CONFIG_FILE = "config.json"

# Map of the audio targets we care about and the config keys they use.
CONFIG_TARGETS = {
    "mic_headset": {
        "label": "Mic: Headset",
        "name_keys": ["mic_headset_name"],
        "index_keys": ["mic_headset_index"],
    },
    "mic_webcam": {
        "label": "Mic: Webcam",
        "name_keys": ["mic_webcam_name"],
        "index_keys": ["mic_webcam_index"],
    },
    "output_headset": {
        "label": "Output: Headset",
        "name_keys": ["output_headset_name", "ouput_headset_name"],
        "index_keys": ["output_headset_index", "ouput_headset_index"],
    },
    "output_TV": {
        "label": "Output: TV",
        "name_keys": ["output_TV_name", "ouput_TV_name"],
        "index_keys": ["output_TV_index", "ouput_TV_index"],
    },
}


def load_config() -> Dict[str, Any]:
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except json.JSONDecodeError:
            messagebox.showerror("Config Error", "config.json is not valid JSON.")
    return {}


def pick_first(config: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for key in keys:
        if key in config and config[key] not in (None, ""):
            return config[key]
    return None


class AudioDeviceWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Audio Devices")
        self.geometry("860x560")
        self.minsize(720, 480)

        self.config_data: Dict[str, Any] = load_config()
        self.targets: Dict[str, Dict[str, Any]] = self._gather_targets()
        self.target_vars: Dict[str, tk.StringVar] = {}
        self.status_var = tk.StringVar(value="")

        self._build_ui()
        self.refresh_devices()

    def _gather_targets(self) -> Dict[str, Dict[str, Any]]:
        targets: Dict[str, Dict[str, Any]] = {}
        for key, meta in CONFIG_TARGETS.items():
            name_value = pick_first(self.config_data, meta["name_keys"])
            raw_index = pick_first(self.config_data, meta["index_keys"])
            try:
                index_value = int(raw_index)
            except (TypeError, ValueError):
                index_value = None
            targets[key] = {
                "label": meta["label"],
                "name": name_value,
                "index": index_value,
                "index_raw": raw_index,
            }
        return targets

    def _format_target_text(self, target: Dict[str, Any]) -> str:
        if not target:
            return "(not set)"
        text = target.get("name") or "(not set)"
        index_raw = target.get("index_raw")
        if index_raw not in (None, ""):
            text += f" — index {index_raw}"
        return text

    def _build_ui(self) -> None:
        tk.Label(
            self,
            text="Detected audio devices",
            font=("Helvetica", 14, "bold"),
        ).pack(pady=(10, 4))
        tk.Label(
            self,
            text="Matches are based on the names/indices stored in config.json.",
            fg="gray25",
        ).pack(pady=(0, 8))

        self._build_config_section()
        self._build_device_section()

        control_row = tk.Frame(self)
        control_row.pack(fill=tk.X, pady=10, padx=10)
        tk.Button(control_row, text="Refresh", width=12, command=self.refresh_devices).pack(
            side=tk.LEFT, padx=5
        )
        tk.Button(control_row, text="Close", width=12, command=self.destroy).pack(
            side=tk.RIGHT, padx=5
        )

        tk.Label(self, textvariable=self.status_var, fg="gray25").pack(
            fill=tk.X, padx=12, pady=(0, 10)
        )

    def _build_config_section(self) -> None:
        frame = tk.LabelFrame(self, text="Configured Targets (config.json)")
        frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        for key, meta in CONFIG_TARGETS.items():
            row = tk.Frame(frame)
            row.pack(fill=tk.X, padx=8, pady=2)

            tk.Label(row, text=f"{meta['label']}:", width=18, anchor="w").pack(side=tk.LEFT)
            value_var = tk.StringVar(value=self._format_target_text(self.targets.get(key, {})))
            tk.Label(row, textvariable=value_var).pack(side=tk.LEFT)
            self.target_vars[key] = value_var

    def _build_device_section(self) -> None:
        container = tk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True, padx=10)

        self.input_tree = self._create_tree(
            container,
            "Input Devices",
            columns=("index", "name", "channels", "default", "matches"),
        )
        self.output_tree = self._create_tree(
            container,
            "Output Devices",
            columns=("index", "name", "channels", "default", "matches"),
        )

    def _create_tree(self, parent: tk.Widget, title: str, columns: Tuple[str, ...]) -> ttk.Treeview:
        frame = tk.LabelFrame(parent, text=title)
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        tree = ttk.Treeview(frame, columns=columns, show="headings", height=10)
        col_widths = {
            "index": 60,
            "name": 280,
            "channels": 90,
            "default": 70,
            "matches": 220,
        }
        headings = {
            "index": "Index",
            "name": "Name",
            "channels": "Channels",
            "default": "Default",
            "matches": "Matches",
        }
        for col in columns:
            tree.heading(col, text=headings.get(col, col.title()))
            tree.column(col, width=col_widths.get(col, 100), anchor="w", stretch=True)

        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.configure(yscrollcommand=scrollbar.set)

        return tree

    def _refresh_targets(self) -> None:
        self.config_data = load_config()
        self.targets = self._gather_targets()
        for key, var in self.target_vars.items():
            var.set(self._format_target_text(self.targets.get(key, {})))

    def _find_matches(self, device_name: str, index: int) -> str:
        matches: List[str] = []
        lowered = device_name.lower()
        for key, target in self.targets.items():
            label = target.get("label", key)
            name_value = target.get("name")
            index_value = target.get("index")

            if name_value and name_value.lower() in lowered:
                matches.append(f"{label} (name)")

            if index_value is not None and index_value == index:
                matches.append(f"{label} (index)")

        # Remove duplicates while preserving order
        seen = set()
        unique_matches = []
        for item in matches:
            if item not in seen:
                seen.add(item)
                unique_matches.append(item)
        return ", ".join(unique_matches)

    def _split_default_device(self, default_device: Any) -> Tuple[Optional[int], Optional[int]]:
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

    def refresh_devices(self) -> None:
        self._refresh_targets()
        try:
            import sounddevice as sd
        except ImportError:
            self.status_var.set("sounddevice is not installed (pip install sounddevice).")
            messagebox.showerror("Missing Dependency", "sounddevice is not installed.")
            return

        try:
            devices = sd.query_devices()
            default_in, default_out = self._split_default_device(sd.default.device)
        except Exception as exc:
            self.status_var.set(f"Failed to query audio devices: {exc}")
            return

        # Clear existing rows
        for tree in (self.input_tree, self.output_tree):
            for item in tree.get_children():
                tree.delete(item)

        input_count = 0
        output_count = 0

        for idx, device in enumerate(devices):
            name = device.get("name", f"Device {idx}")
            max_in = device.get("max_input_channels", 0) or 0
            max_out = device.get("max_output_channels", 0) or 0
            matches = self._find_matches(name, idx)

            if max_in > 0:
                input_count += 1
                self.input_tree.insert(
                    "",
                    tk.END,
                    values=(
                        idx,
                        name,
                        max_in,
                        "Yes" if default_in == idx else "",
                        matches,
                    ),
                )

            if max_out > 0:
                output_count += 1
                self.output_tree.insert(
                    "",
                    tk.END,
                    values=(
                        idx,
                        name,
                        max_out,
                        "Yes" if default_out == idx else "",
                        matches,
                    ),
                )

        self.status_var.set(
            f"Found {input_count} input device(s) and {output_count} output device(s)."
        )


if __name__ == "__main__":
    AudioDeviceWindow().mainloop()
