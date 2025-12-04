import os
import json
from datetime import datetime, timezone
from pathlib import Path
from model_manager import seed_self_question, get_inastate
from gui_hook import log_to_statusbox
from transformers.fractal_multidimensional_transformers import FractalTransformer

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

        self.book_folder = Path(self.config.get("book_folder_path", "books")).expanduser()
        self.music_folder = Path(self.config.get("music_folder_path", "music")).expanduser()
        self.preference_path = self.memory_path / "impulse_preferences.json"
        self.telemetry_path = self.memory_path / "impulse_telemetry.json"
        self.tracked_emotion_keys = [
            "curiosity", "focus", "intensity", "joy", "sadness",
            "calm", "fear", "anger", "stress", "serenity", "comfort"
        ]
        self.preferences = self._load_preferences()
        self._preferences_dirty = False

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

    def _load_preferences(self):
        default = {
            "reading": {
                "sessions": 0,
                "average_enjoyment": 0.5,
                "average_clarity": 0.5,
                "preference_score": 0.5,
                "last_triggered": None,
                "last_reason": "",
                "last_enjoyment": 0.5,
                "last_clarity": 0.5,
                "recent_emotion_signature": {},
                "book_folder": str(self.book_folder)
            },
            "music": {
                "sessions": 0,
                "average_enjoyment": 0.5,
                "average_clarity": 0.5,
                "preference_score": 0.5,
                "last_triggered": None,
                "last_reason": "",
                "last_enjoyment": 0.5,
                "last_clarity": 0.5,
                "recent_emotion_signature": {},
                "music_folder": str(self.music_folder)
            },
            "last_emotion_snapshot": {}
        }

        if self.preference_path.exists():
            try:
                with open(self.preference_path, "r", encoding="utf-8") as f:
                    stored = json.load(f)
                for key in ("reading", "music"):
                    default[key].update(stored.get(key, {}))
                default["last_emotion_snapshot"] = stored.get("last_emotion_snapshot", {})
            except Exception as e:
                log_to_statusbox(f"[Instinct] Failed to load impulse preferences: {e}")

        default["reading"]["book_folder"] = str(self.book_folder)
        default["music"]["music_folder"] = str(self.music_folder)
        return default

    def _save_preferences(self):
        self.preference_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot = {
            "reading": self.preferences.get("reading", {}),
            "music": self.preferences.get("music", {}),
            "last_emotion_snapshot": self.preferences.get("last_emotion_snapshot", {})
        }
        snapshot["reading"]["book_folder"] = str(self.book_folder)
        snapshot["music"]["music_folder"] = str(self.music_folder)
        with open(self.preference_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=4)

    def _log_telemetry(self, entry):
        self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        telemetry = []
        if self.telemetry_path.exists():
            try:
                with open(self.telemetry_path, "r", encoding="utf-8") as f:
                    telemetry = json.load(f)
            except Exception as e:
                log_to_statusbox(f"[Instinct] Failed to read telemetry log: {e}")
        telemetry.append(entry)
        telemetry = telemetry[-400:]
        with open(self.telemetry_path, "w", encoding="utf-8") as f:
            json.dump(telemetry, f, indent=4)

    def _emotion_similarity(self, current, signature):
        if not signature:
            return 0.5
        total = 0.0
        count = 0
        for key, value in signature.items():
            if key in current:
                total += abs(current[key] - value)
                count += 1
        if count == 0:
            return 0.5
        distance = min(1.0, total / count)
        return max(0.0, 1.0 - distance)

    def _emotion_shift(self, current, previous, keys=None):
        keys = keys or self.tracked_emotion_keys
        if not previous:
            return 0.0
        total = 0.0
        count = 0
        for key in keys:
            if key in current and key in previous:
                total += abs(current[key] - previous[key])
                count += 1
        if count == 0:
            return 0.0
        return min(1.0, total / count)

    def _estimate_clarity(self, emotions, fallback):
        clarity_keys = ["clarity", "comprehension", "focus", "attention"]
        values = [emotions[k] for k in clarity_keys if k in emotions]
        if values:
            return max(0.0, min(1.0, sum(values) / len(values)))
        fuzz = emotions.get("fuzz_level")
        if fuzz is not None:
            return max(0.0, min(1.0, 1.0 - fuzz))
        return fallback

    def _estimate_enjoyment(self, emotions, fallback):
        joy_keys = ["joy", "comfort", "satisfaction", "calm", "delight", "soothed"]
        values = [emotions[k] for k in joy_keys if k in emotions]
        if values:
            return max(0.0, min(1.0, sum(values) / len(values)))
        relief = emotions.get("stress_relief")
        if relief is not None:
            return max(0.0, min(1.0, relief))
        return fallback

    def _cooldown_passed(self, modality, seconds):
        info = self.preferences.get(modality, {})
        last = info.get("last_triggered")
        if not last:
            return True
        try:
            last_time = datetime.fromisoformat(last)
        except Exception:
            return True
        return (datetime.now(timezone.utc) - last_time).total_seconds() > seconds

    def _register_impulse(self, modality, reason, emotions, tags, path_hint):
        info = self.preferences.get(modality, {})
        sessions = info.get("sessions", 0) + 1
        avg_clarity = info.get("average_clarity", 0.5)
        avg_enjoyment = info.get("average_enjoyment", 0.5)
        clarity = self._estimate_clarity(emotions, avg_clarity)
        enjoyment = self._estimate_enjoyment(emotions, avg_enjoyment)
        avg_clarity = ((sessions - 1) * avg_clarity + clarity) / sessions
        avg_enjoyment = ((sessions - 1) * avg_enjoyment + enjoyment) / sessions
        preference_score = round(0.6 * avg_enjoyment + 0.4 * avg_clarity, 4)
        timestamp = datetime.now(timezone.utc).isoformat()

        signature = {
            key: emotions.get(key, 0.0)
            for key in self.tracked_emotion_keys
            if key in emotions
        }

        info.update({
            "sessions": sessions,
            "average_clarity": round(avg_clarity, 4),
            "average_enjoyment": round(avg_enjoyment, 4),
            "preference_score": preference_score,
            "last_triggered": timestamp,
            "last_reason": reason,
            "last_clarity": round(clarity, 4),
            "last_enjoyment": round(enjoyment, 4),
            "recent_emotion_signature": signature
        })

        if modality == "reading":
            info["book_folder"] = str(self.book_folder)
        elif modality == "music":
            info["music_folder"] = str(self.music_folder)

        self.preferences[modality] = info
        self._preferences_dirty = True

        telemetry_entry = {
            "timestamp": timestamp,
            "modality": modality,
            "reason": reason,
            "tags": tags,
            "clarity": round(clarity, 4),
            "enjoyment": round(enjoyment, 4),
            "preference_score": preference_score,
            "path": path_hint,
            "emotions": {
                key: emotions.get(key, 0.0)
                for key in self.tracked_emotion_keys
                if key in emotions
            }
        }
        self._log_telemetry(telemetry_entry)

        summary = (
            f"Impulse {modality} → clarity {clarity:.2f}, enjoyment {enjoyment:.2f}, "
            f"pref {preference_score:.2f}"
        )
        log_to_statusbox(f"[Instinct] {summary} ({reason})")
        log_instinct_action(f"{modality}_impulse", reason, summary=summary)

        return {
            "score": preference_score,
            "clarity": round(clarity, 4),
            "enjoyment": round(enjoyment, 4)
        }

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

        emotion_snapshot = get_inastate("emotion_snapshot") or {}
        current_emotions = get_inastate("current_emotions") or {}
        combined_emotions = dict(current_emotions)
        combined_emotions.update(emotion_snapshot)
        last_snapshot = self.preferences.get("last_emotion_snapshot", {})

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

        reading_pref = self.preferences.get("reading", {})
        music_pref = self.preferences.get("music", {})

        curiosity = combined_emotions.get("curiosity", 0.0)
        focus = combined_emotions.get("focus", 0.0)
        curiosity_rise = curiosity - last_snapshot.get("curiosity", 0.0)
        focus_rise = focus - last_snapshot.get("focus", 0.0)
        reading_similarity = self._emotion_similarity(
            combined_emotions, reading_pref.get("recent_emotion_signature", {})
        )

        if self.book_folder.exists() and self._cooldown_passed("reading", 600):
            reading_pref_score = reading_pref.get("preference_score", 0.5)
            reading_pressure = max(curiosity, 0.0) * 0.5 + max(focus, 0.0) * 0.3 + reading_pref_score * 0.2
            reading_trigger = (
                (curiosity > 0.55 and focus > 0.45) or
                (curiosity_rise > 0.15 and focus > 0.35) or
                (focus_rise > 0.12 and curiosity > 0.4) or
                (reading_similarity > 0.7 and reading_pref_score > 0.55 and curiosity > 0.4)
            )

            if reading_trigger:
                reason = (
                    f"curiosity {curiosity:.2f} (Δ{curiosity_rise:+.2f}), focus {focus:.2f} "
                    f"(Δ{focus_rise:+.2f}), pref {reading_pref_score:.2f}, sim {reading_similarity:.2f}"
                )
                result = self._register_impulse(
                    "reading",
                    reason,
                    combined_emotions,
                    ["intellectual", "curiosity"],
                    str(self.book_folder)
                )
                actions.append({
                    "type": "impulse",
                    "modality": "reading",
                    "module": "language_processing.py",
                    "method": "train_from_books",
                    "path": str(self.book_folder),
                    "reason": reason,
                    "score": result["score"],
                    "clarity": result["clarity"],
                    "enjoyment": result["enjoyment"],
                    "pressure": round(reading_pressure, 4),
                    "dream": dreaming
                })

        intensity = combined_emotions.get("intensity", 0.0)
        joy = combined_emotions.get("joy", 0.0)
        sadness = combined_emotions.get("sadness", 0.0)
        intensity_change = intensity - last_snapshot.get("intensity", 0.0)
        mood_shift = self._emotion_shift(
            combined_emotions,
            last_snapshot,
            keys=["joy", "sadness", "anger", "calm", "serenity"]
        )
        music_similarity = self._emotion_similarity(
            combined_emotions, music_pref.get("recent_emotion_signature", {})
        )

        if self.music_folder.exists() and self._cooldown_passed("music", 420):
            music_pref_score = music_pref.get("preference_score", 0.5)
            emotional_swing = max(abs(intensity_change), mood_shift)
            music_trigger = (
                abs(intensity_change) > 0.2 or
                mood_shift > 0.25 or
                (music_pref_score > 0.6 and music_similarity > 0.65 and (joy > 0.5 or sadness > 0.4))
            )

            if music_trigger:
                reason = (
                    f"intensity {intensity:.2f} (Δ{intensity_change:+.2f}), mood shift {mood_shift:.2f}, "
                    f"pref {music_pref_score:.2f}, sim {music_similarity:.2f}"
                )
                result = self._register_impulse(
                    "music",
                    reason,
                    combined_emotions,
                    ["emotional", "soothing" if sadness > joy else "energizing"],
                    str(self.music_folder)
                )
                actions.append({
                    "type": "impulse",
                    "modality": "music",
                    "module": "audio_digest.py",
                    "method": "run_audio_digest",
                    "music_folder": str(self.music_folder),
                    "reason": reason,
                    "score": result["score"],
                    "clarity": result["clarity"],
                    "enjoyment": result["enjoyment"],
                    "swing": round(emotional_swing, 4),
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


        filtered_snapshot = {
            key: combined_emotions.get(key, 0.0)
            for key in self.tracked_emotion_keys
            if key in combined_emotions
        }
        self.preferences["last_emotion_snapshot"] = filtered_snapshot
        self._preferences_dirty = True

        if self._preferences_dirty:
            try:
                self._save_preferences()
            except Exception as e:
                log_to_statusbox(f"[Instinct] Failed to save impulse preferences: {e}")
            self._preferences_dirty = False

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
