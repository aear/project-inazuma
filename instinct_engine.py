import os
import json
from datetime import datetime, timezone
from pathlib import Path
from model_manager import seed_self_question, get_inastate
from gui_hook import log_to_statusbox
from fractal_multidimensional_transformers import FractalTransformer

CONFIG_FILE = "config.json"

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {}

def log_instinct_action(action, reason, precision=None, summary=None):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "reason": reason,
        "dreaming": get_inastate("dreaming"),
        "emotions": get_inastate("emotion_snapshot") or {},
        "precision": precision or get_inastate("current_precision"),
        "summary": summary or f"Instinct '{action}' triggered by {reason}"
    }

    path = Path("AI_Children") / get_inastate("current_child", "default_child") / "memory" / "instinct_log.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        try:
            with open(path, "r") as f:
                log = json.load(f)
        except:
            log = []
    else:
        log = []

    log.append(entry)
    log = log[-200:]

    with open(path, "w") as f:
        json.dump(log, f, indent=4)

    print(f"[Instinct] Logged action: {action} ({reason})")

def suggest_precision_override(score, reason="instinct threshold"):
    hint = {
        "override_precision": score,
        "reason": reason,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    path = Path("precision_hint.json")
    with open(path, "w") as f:
        json.dump(hint, f, indent=4)
    print(f"[Instinct] Suggested precision override → {score} due to: {reason}")

class InstinctEngine:
    def __init__(self):
        log_to_statusbox("[Instinct] Checking instinctual responses...")
        self.config = load_config()
        self.child = self.config.get("current_child", "default_child")
        self.memory_path = Path("AI_Children") / self.child / "memory"
        self.os_type = self.config.get("os", "unknown")

        transformer = FractalTransformer()
        transformer.load_precision_profile(self.child)
        self.precision = int(transformer.precision * 64)
        

        if self.precision >= 64:
            self.express_interval = 30
            self.reflect_interval = 300
        elif self.precision >= 32:
            self.express_interval = 20
            self.reflect_interval = 180
        else:
            self.express_interval = 10
            self.reflect_interval = 90

    def should_express(self):
        expr_log = self.memory_path / "expressions.json"
        if not expr_log.exists(): return True
        try:
            with open(expr_log, "r") as f:
                data = json.load(f)
                if data and "timestamp" in data[-1]:
                    last = datetime.fromisoformat(data[-1]["timestamp"].replace("Z", "+00:00"))
                    now = datetime.now(timezone.utc)
                    return (now - last).total_seconds() > self.express_interval
        except: return True
        return False

    def should_reflect(self):
        id_path = Path("AI_Children") / self.child / "identity" / "self_reflection.json"
        if not id_path.exists(): return True
        try:
            with open(id_path, "r") as f:
                data = json.load(f)
                last = datetime.fromisoformat(data.get("reflected_at", "").replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                return (now - last).total_seconds() > self.reflect_interval
        except: return True
        return False

    def detect_fragment_growth(self):
        frag_dir = self.memory_path / "fragments"
        if not frag_dir.exists(): return False
        return len(list(frag_dir.glob("frag_*.json"))) > 5

    def should_predict(self):
        frag_dir = self.memory_path / "fragments"
        return frag_dir.exists() and len(list(frag_dir.glob("frag_*.json"))) >= 2

    def poll(self):
        dreaming = get_inastate("dreaming", False)

        actions = []

        if self.should_express():
            actions.append({
                "type": "expression",
                "mode": "instinct",
                "reason": "silence_gap",
                "dream": dreaming
            })
            seed_self_question("Why didn’t I want to speak?")
            

        if self.should_reflect():
            actions.append({
                "type": "reflection",
                "module": "who_am_i.py",
                "dream": dreaming
            })

        if self.detect_fragment_growth():
            actions.append({
                "type": "training",
                "module": "train_fragments.py",
                "dream": dreaming
            })

        if self.should_predict():
            actions.append({
                "type": "prediction",
                "module": "predictive_layer.py",
                "dream": dreaming
            })

        frag_dir = self.memory_path / "fragments"
        identity_fragments = []

        if frag_dir.exists():
            for f in frag_dir.glob("frag_*.json"):
                try:
                    with open(f, "r", encoding="utf-8") as file:
                        frag = json.load(file)
                        if "identity" in frag.get("tags", []):
                            identity_fragments.append(frag)
                except:
                    continue

        if identity_fragments:
            unclear = [f for f in identity_fragments if f.get("clarity", 0.0) < 0.7]
            if len(unclear) >= 2:
                seed_self_question("Why do I keep hearing that name?")

                # === Curiosity Instinct ===
        curiosity_fragments = []
        for f in frag_dir.glob("frag_*.json"):
            try:
                with open(f, "r", encoding="utf-8") as file:
                    frag = json.load(file)
                    emotions = frag.get("emotions", {})
                    if emotions.get("curiosity", 0.0) > 0.5 and frag.get("clarity", 1.0) < 0.7:
                        curiosity_fragments.append(frag)
            except:
                continue

        if len(curiosity_fragments) >= 2:
            print("[Instinct] Curiosity-triggered reflection.")
            seed_self_question("Is this something I want to understand better?")

        # === Precision Tuning Hint (Instinctual Response)
        try:
            emo = get_inastate("current_emotions") or {}
            intensity = emo.get("intensity", 0.0)
            stress = emo.get("stress", 0.0)
            risk = emo.get("risk", 0.0)

            overload_score = (intensity + stress + risk) / 3
            threshold = 0.65

            hint_path = self.memory_path / "precision_hint.json"

            if overload_score > threshold:
                suggestion = {
                    "suggested_max_precision": 32,
                    "reason": "Instinctive response to emotional overload",
                    "score": overload_score,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                with open(hint_path, "w", encoding="utf-8") as f:
                    json.dump(suggestion, f, indent=4)
                print(f"[Instinct] Precision hint saved: overload={round(overload_score, 3)}")
        except Exception as e:
            print(f"[Instinct] Failed to evaluate precision hint: {e}")


        return actions

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="Perform a specific instinctual query")
    args = parser.parse_args()

    if args.query == "sleep_permission":
        emo = get_inastate("current_emotions") or {}
        anxiety = emo.get("anxiety", 0.0)
        fear = emo.get("fear", 0.0)
        trust = emo.get("trust", 0.5)

        if anxiety > 0.5 or fear > 0.5:
            result = {
                "permit_sleep": False,
                "reason": "high anxiety or fear",
                "emotions": emo
            }
        else:
            result = {
                "permit_sleep": True,
                "reason": "emotional state calm enough",
                "emotions": emo
            }

        print(json.dumps(result))
