import sys
import numpy as np
from PyQt5 import QtWidgets, QtGui
import pyqtgraph.opengl as gl
from pathlib import Path
import json
import os
from model_manager import get_running_modules, get_inastate, is_dreaming

CONFIG_FILE = "config.json"

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {}

def load_model_data(current_child):
    path = Path("AI_Children") / current_child / "ina_pretrained_model.json"
    if not path.exists():
        return []
    with open(path, "r") as f:
        model = json.load(f)
        return model.get("structure", [])

class EEGWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ina's EEG")
        self.resize(800, 600)
        self.config_data = load_config()
        self.current_child = self.config_data.get("current_child", "default_child")

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        self.view = gl.GLViewWidget()
        self.view.opts['distance'] = 40
        layout.addWidget(self.view)

        self.grid = gl.GLGridItem()
        self.grid.scale(2, 2, 1)
        self.view.addItem(self.grid)

        self.render_brain()

    def render_brain(self):
        structure = load_model_data(self.current_child)
        modules = get_running_modules()
        dreaming = is_dreaming()
        state = {
            "emotions": get_inastate("current_emotions") or {},
            "logic": get_inastate("last_logic_output"),
            "prediction": get_inastate("prediction_trace"),
            "expression": get_inastate("recent_expression")
        }

        points = []
        colors = []

        for layer in structure:
            for node in layer:
                x = np.random.normal(loc=node.get("depth", 1), scale=0.5)
                y = np.random.normal(loc=node.get("dimensions", 3), scale=1.0)
                z = np.random.normal(loc=node.get("activity", 0.0), scale=1.0)
                points.append([x, y, z])

                a = min(1.0, node.get("activity", 0.3))
                color = [0.2, 0.2, 1.0, a]

                if dreaming:
                    color = [0.6, 0.3, 1.0, a]

                if "emotion_engine" in modules and node.get("fragment_id", "").endswith("1"):
                    color = [1.0, 0.3, 0.3, a]
                elif "logic_engine" in modules and node.get("fragment_id", "").endswith("2"):
                    color = [0.3, 1.0, 0.3, a]
                elif "predictive_layer" in modules and node.get("fragment_id", "").endswith("3"):
                    color = [1.0, 1.0, 0.2, a]

                colors.append(color)

        if not points:
            print("[EEG] No nodes to render.")
            return

        pts = gl.GLScatterPlotItem(pos=np.array(points), color=np.array(colors), size=1.0)
        self.view.addItem(pts)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = EEGWindow()
    window.show()
    sys.exit(app.exec_())