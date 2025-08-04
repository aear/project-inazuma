
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from gui_hook import log_to_statusbox
from model_manager import load_config, update_inastate
from fractal_multidimensional_transformers import FractalTransformer

class HindsightTransformer:
    """
    Retrospective insight transformer:
      - Loads past prediction logs
      - Compares predicted clarity to subsequent clarity
      - Records errors and time deltas
      - Adjusts trust/clarity in inastate
      - Builds a 'why_wrong' map for self-correction
    """

    def __init__(self):
        config = load_config()
        self.child = config.get("current_child", "default_child")
        self.memory_path = Path("AI_Children") / self.child / "memory"
        self.pred_path = self.memory_path / "prediction_log.json"
        self.hindsight_map_path = self.memory_path / "hindsight_map.json"
        self.transformer = FractalTransformer()

    def load_predictions(self):
        if not self.pred_path.exists():
            log_to_statusbox("[Hindsight] No prediction log found.")
            return []
        try:
            preds = json.loads(self.pred_path.read_text())
            preds_sorted = sorted(preds, key=lambda e: e["timestamp"])
            return preds_sorted
        except Exception as e:
            log_to_statusbox(f"[Hindsight] Failed to load predictions: {e}")
            return []

    def compute_hindsight(self, predictions):
        insights = []
        for i in range(len(predictions) - 1):
            curr = predictions[i]
            nxt = predictions[i + 1]
            try:
                t1 = datetime.fromisoformat(curr["timestamp"])
                t2 = datetime.fromisoformat(nxt["timestamp"])
            except:
                continue
            c1 = curr.get("predicted_vector", {}).get("clarity", 0.0)
            c2 = nxt.get("predicted_vector", {}).get("clarity", 0.0)
            error = round(c2 - c1, 4)
            delta = (t2 - t1).total_seconds()
            insights.append({
                "prediction_time": curr["timestamp"],
                "next_time": nxt["timestamp"],
                "predicted_clarity": c1,
                "actual_clarity": c2,
                "error": error,
                "time_delta_s": delta
            })
        return insights

    def adjust_trust(self, insights):
        total_error = sum(abs(ins["error"]) for ins in insights)
        count = len(insights)
        avg_error = (total_error / count) if count else 0.0
        trust = round(max(0.0, 1 - avg_error), 4)
        update_inastate("hindsight_trust", trust)
        log_to_statusbox(f"[Hindsight] Updated trust: {trust}")

    def save_hindsight_map(self, insights):
        try:
            self.hindsight_map_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.hindsight_map_path, "w") as f:
                json.dump(insights, f, indent=2)
            log_to_statusbox(f"[Hindsight] Map saved ({len(insights)} entries).")
        except Exception as e:
            log_to_statusbox(f"[Hindsight] Failed to save map: {e}")

    def run(self):
        log_to_statusbox("[Hindsight] Running retrospective insight...")
        preds = self.load_predictions()
        if len(preds) < 2:
            log_to_statusbox("[Hindsight] Not enough predictions to analyze.")
            return
        insights = self.compute_hindsight(preds)
        if not insights:
            log_to_statusbox("[Hindsight] No insights generated.")
            return
        self.adjust_trust(insights)
        self.save_hindsight_map(insights)
        log_to_statusbox("[Hindsight] Retrospective analysis complete.")

if __name__ == "__main__":
    HindsightTransformer().run()
