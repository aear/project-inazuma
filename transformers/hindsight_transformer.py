
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
from gui_hook import log_to_statusbox
from model_manager import load_config, update_inastate
from .fractal_multidimensional_transformers import FractalTransformer
from symbol_generator import generate_symbol_from_parts

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
        self.emotion_log_path = self.memory_path / "emotion_log.json"
        self.emotion_drift_path = self.memory_path / "emotion_drift_log.json"
        self.pred_evo_path = self.memory_path / "prediction_evolution.json"
        self.transformer = FractalTransformer()
        self.intent_telemetry: Dict[str, Any] = {}

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

    def load_emotion_log(self):
        if not self.emotion_log_path.exists():
            return []
        try:
            return json.loads(self.emotion_log_path.read_text())
        except Exception:
            return []

    def get_emotion_snapshot(self, timestamp, log):
        """Return the latest emotion snapshot not after timestamp."""
        try:
            target = datetime.fromisoformat(timestamp)
        except Exception:
            return {}
        best = {}
        best_time = None
        for entry in log:
            try:
                t = datetime.fromisoformat(entry.get("timestamp", ""))
            except Exception:
                continue
            if t <= target and (best_time is None or t > best_time):
                best_time = t
                best = entry.get("snapshot", {})
        return best

    def compute_emotional_drift(self, t1, t2, log):
        snap1 = self.get_emotion_snapshot(t1, log)
        snap2 = self.get_emotion_snapshot(t2, log)
        drift = {}
        keys = set(snap1.keys()) | set(snap2.keys())
        for k in keys:
            drift[k] = round(snap2.get(k, 0.0) - snap1.get(k, 0.0), 4)
        return drift

    def generate_symbolic_tag(self, error):
        if error > 0:
            symbol = generate_symbol_from_parts("trust", "sharp", "change")
            tags = ["clarity_increase"]
        elif error < 0:
            symbol = generate_symbol_from_parts("fear", "sharp", "change")
            tags = ["clarity_decrease"]
        else:
            symbol = generate_symbol_from_parts("calm", "moderate", "pattern")
            tags = ["clarity_stable"]
        return symbol, tags

    def record_emotional_drift(self, drift, t1, t2):
        entry = {"from": t1, "to": t2, "drift": drift}
        try:
            if self.emotion_drift_path.exists():
                history = json.loads(self.emotion_drift_path.read_text())
            else:
                history = []
        except Exception:
            history = []
        history.append(entry)
        history = history[-200:]
        with open(self.emotion_drift_path, "w") as f:
            json.dump(history, f, indent=2)

        # pattern recognition: log repeated drift patterns
        key = tuple(sorted((k, round(v, 1)) for k, v in drift.items() if abs(v) > 0.2))
        if key:
            count = sum(1 for h in history if tuple(sorted((k, round(v, 1)) for k, v in h.get("drift", {}).items() if abs(v) > 0.2)) == key)
            if count >= 3:
                log_to_statusbox(f"[Hindsight] Repeated emotional shift pattern: {key}")

    def build_reflection(self, insight, curr, nxt):
        if abs(insight.get("error", 0.0)) < 0.1:
            return None
        causes = ["clarity mismatch"]
        missed = curr.get("fragments_used", [])
        adjustments = "increase attention to emotional context"
        return {
            "causes": causes,
            "missed_prediction_points": missed,
            "adjustments": adjustments,
            "symbolic_tag": insight.get("symbol"),
            "emotional_drift": insight.get("emotional_drift", {})
        }

    def store_prediction_lesson(self, symbol, reflection, timestamp):
        weight = 1.0
        try:
            t = datetime.fromisoformat(timestamp)
            age = (datetime.now(timezone.utc) - t).total_seconds()
            weight = round(1.0 / (1.0 + age / 3600.0), 4)  # hour decay
        except Exception:
            pass
        entry = {
            "timestamp": timestamp,
            "symbol": symbol,
            "adjustments": reflection.get("adjustments"),
            "weight": weight
        }
        try:
            if self.pred_evo_path.exists():
                history = json.loads(self.pred_evo_path.read_text())
            else:
                history = []
        except Exception:
            history = []
        history.append(entry)
        history = history[-200:]
        with open(self.pred_evo_path, "w") as f:
            json.dump(history, f, indent=2)

    def annotate_fragments(self, frag_ids, lesson, symbol):
        frag_dir = self.memory_path / "fragments"
        for fid in frag_ids:
            fpath = frag_dir / f"{fid}.json"
            if not fpath.exists():
                continue
            try:
                data = json.loads(fpath.read_text())
            except Exception:
                continue
            hindsight_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "lesson": lesson,
                "symbol": symbol
            }
            data.setdefault("hindsight", []).append(hindsight_entry)
            with open(fpath, "w") as f:
                json.dump(data, f, indent=4)

    def save_predictions(self, predictions):
        try:
            with open(self.pred_path, "w") as f:
                json.dump(predictions, f, indent=2)
        except Exception as e:
            log_to_statusbox(f"[Hindsight] Failed to update predictions: {e}")

    def compute_hindsight(self, predictions):
        insights = []
        emotion_log = self.load_emotion_log()
        for i in range(len(predictions) - 1):
            curr = predictions[i]
            nxt = predictions[i + 1]
            try:
                t1 = datetime.fromisoformat(curr["timestamp"])
                t2 = datetime.fromisoformat(nxt["timestamp"])
            except Exception:
                continue
            c1 = curr.get("predicted_vector", {}).get("clarity", 0.0)
            c2 = nxt.get("predicted_vector", {}).get("clarity", 0.0)
            error = round(c2 - c1, 4)
            delta = (t2 - t1).total_seconds()

            symbol, tags = self.generate_symbolic_tag(error)
            drift = self.compute_emotional_drift(curr["timestamp"], nxt["timestamp"], emotion_log)
            self.record_emotional_drift(drift, curr["timestamp"], nxt["timestamp"])

            insight = {
                "prediction_time": curr["timestamp"],
                "next_time": nxt["timestamp"],
                "predicted_clarity": c1,
                "actual_clarity": c2,
                "error": error,
                "time_delta_s": delta,
                "symbol": symbol,
                "tags": tags,
                "emotional_drift": drift,
                "related_fragments": curr.get("fragments_used", [])
            }

            curr.setdefault("hindsight_symbols", []).append(symbol)
            nxt.setdefault("hindsight_symbols", []).append(symbol)
            curr.setdefault("hindsight_tags", []).extend(tags)
            nxt.setdefault("hindsight_tags", []).extend(tags)

            reflection = self.build_reflection(insight, curr, nxt)
            if reflection:
                insight["reflection"] = reflection
                self.store_prediction_lesson(symbol, reflection, nxt["timestamp"])
                self.annotate_fragments(curr.get("fragments_used", []), reflection["adjustments"], symbol)

            insights.append(insight)
        return insights, predictions

    def adjust_trust(self, insights):
        total_error = sum(abs(ins["error"]) for ins in insights)
        count = len(insights)
        avg_error = (total_error / count) if count else 0.0
        trust = round(max(0.0, 1 - avg_error), 4)
        update_inastate("hindsight_trust", trust)
        log_to_statusbox(f"[Hindsight] Updated trust: {trust}")
        return trust, avg_error

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
        insights, updated_preds = self.compute_hindsight(preds)
        if not insights:
            log_to_statusbox("[Hindsight] No insights generated.")
            return
        trust_value, avg_error = self.adjust_trust(insights)
        self.save_hindsight_map(insights)
        self.save_predictions(updated_preds)
        self.intent_telemetry = {
            "intent": "trust_update",
            "insights": len(insights),
            "avg_error": round(avg_error, 4),
            "trust": trust_value,
        }
        log_to_statusbox("[Hindsight] Retrospective analysis complete.")

    def get_intent_telemetry(self) -> Dict[str, Any]:
        return dict(self.intent_telemetry)

if __name__ == "__main__":
    HindsightTransformer().run()
