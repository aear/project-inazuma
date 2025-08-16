import tkinter as tk
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from transformers.fractal_multidimensional_transformers import FractalTransformer

CONFIG_FILE = "config.json"
    

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {}

class AttentionWindow(tk.Tk):
    def __init__(self, label="vision"):
        super().__init__()
        self.title(f"{label.capitalize()} Attention")
        self.label = label
        self.dragging = False

        self.config = load_config()
        self.child_name = self.config.get("current_child")
        if not self.child_name:
            raise ValueError("No current_child set in config.json")

        self.log_path = Path("AI_Children") / self.child_name / "memory" / "attention"
        self.attention_log = self.log_path / "attention_log.json"
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_path / f"{self.label}.json"

        self.geometry("200x150+100+100")
        self.configure(bg="lightblue")

        self.bind("<Configure>", self.on_configure)
        self.bind("<ButtonPress-1>", self.start_drag)
        self.bind("<B1-Motion>", self.on_drag)
        self.bind("<ButtonRelease-1>", self.end_drag)

        self.polygons = []
        transformer = FractalTransformer()  # Tuned via model

    def start_drag(self, event):
        self.dragging = True

    def end_drag(self, event):
        self.dragging = False
        self.log_position()

    def on_drag(self, event):
        self.log_position()

    def on_configure(self, event):
        if not self.dragging:
            self.log_position()

    def log_position(self):
        geo = self.geometry()
        width, height, x, y = self.parse_geometry(geo)
        timestamp = datetime.now(timezone.utc).isoformat()

        # Transformer scoring
        inputs = {"x": x, "y": y, "width": width, "height": height}
        score_vec = self.transformer.encode({"emotions": inputs})
        importance = round(sum(score_vec["vector"]) / len(score_vec["vector"]), 4)

        log_entry = {
            "timestamp": timestamp,
            "type": "window",
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "importance": importance
        }

        self.append_log(log_entry)

    def parse_geometry(self, geo_str):
        dims, pos = geo_str.split("+", 1)
        width, height = map(int, dims.split("x"))
        x, y = map(int, pos.split("+"))
        return width, height, x, y

    def add_polygon(self, vertices):
        timestamp = datetime.now(timezone.utc).isoformat()
        polygon_entry = {
            "timestamp": timestamp,
            "type": "polygon_add",
            "vertices": vertices,
            "label": self.label
        }
        self.polygons.append(polygon_entry)
        self.append_log(polygon_entry)

    def remove_polygon(self, index):
        if 0 <= index < len(self.polygons):
            removed = self.polygons.pop(index)
            timestamp = datetime.now(timezone.utc).isoformat()
            removal_log = {
                "timestamp": timestamp,
                "type": "polygon_remove",
                "removed": removed
            }
            self.append_log(removal_log)

    def append_log(self, entry):
        try:
            if self.log_file.exists():
                with open(self.log_file, "r") as f:
                    logs = json.load(f)
                    if not isinstance(logs, list):
                        logs = []
            else:
                logs = []
        except Exception:
            logs = []

        logs.append(entry)
        with open(self.log_file, "w") as f:
            json.dump(logs, f, indent=2)

if __name__ == "__main__":
    import threading

    def start_vision():
        app = AttentionWindow(label="vision")
        app.mainloop()

    def start_audio():
        app = AttentionWindow(label="audio")
        app.mainloop()

    threading.Thread(target=start_vision).start()
    threading.Thread(target=start_audio).start()

