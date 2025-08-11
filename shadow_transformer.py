import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from model_manager import load_config
from gui_hook import log_to_statusbox

# Optional integrations
try:
    from emotion_engine import get_current_emotion_state
except Exception:  # pragma: no cover
    def get_current_emotion_state():
        return {}

try:  # pragma: no cover
    from meaning_map import get_symbol_neighbors
except Exception:
    def get_symbol_neighbors(symbol_id=None, tags=None, k=5):
        return []

try:  # pragma: no cover
    from prediction_layer import predict_unspoken
except Exception:
    def predict_unspoken(fragment):
        return {}

try:  # pragma: no cover
    from logic_engine import register_shadow_hint
except Exception:
    def register_shadow_hint(envelope_id, summary):
        pass

try:  # pragma: no cover
    import memory_graph as mg
    fetch_fragment = getattr(mg, "fetch_fragment", None)
    upsert_fragment = getattr(mg, "upsert_fragment", None)
except Exception:
    fetch_fragment = None
    upsert_fragment = None


class ShadowTransformer:
    """Transformer handling suppressed/unresolved/high-conflict fragments.

    Fragments with shadow-related tags are sealed into envelopes stored in
    ``AI_Children/<child>/shadow/envelopes``. Each envelope contains the raw
    fragment, optional emotional context, symbol neighbours and predictions.
    Metadata for all envelopes is tracked in ``shadow_index.json`` and an
    operational log is appended to ``shadow_log.jsonl``.
    """

    def __init__(self):
        config = load_config()
        self.child = config.get("current_child", "default_child")
        self.shadow_path = Path("AI_Children") / self.child / "shadow"
        self.envelopes_path = self.shadow_path / "envelopes"
        self.index_path = self.shadow_path / "shadow_index.json"
        self.log_path = self.shadow_path / "shadow_log.jsonl"

        self.envelopes_path.mkdir(parents=True, exist_ok=True)
        self.shadow_path.mkdir(parents=True, exist_ok=True)

        self.index = self.load_index()

    # ------------------------------------------------------------------ utils
    def load_index(self):
        if self.index_path.exists():
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def save_index(self):
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self.index, f, indent=2)

    def log(self, action, detail=None):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "detail": detail or {}
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    # ------------------------------------------------------------------ core
    def find_shadow_candidates(self):
        frag_dir = Path("AI_Children") / self.child / "memory" / "fragments"
        if not frag_dir.exists():
            return []
        candidates = []
        for file in frag_dir.glob("*.json"):
            try:
                data = json.loads(file.read_text())
            except Exception:
                continue
            tags = set(data.get("tags", []))
            if tags & {"suppressed", "unresolved", "high_conflict"}:
                candidates.append(data)
        return candidates

    def create_envelope(self, fragment, emotion_state=None, neighbors=None, predictions=None):
        envelope_id = f"env_{uuid.uuid4().hex[:8]}"
        payload = {
            "fragment_id": fragment.get("id"),
            "fragment": fragment,
            "sealed": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "emotion": emotion_state or {},
            "neighbors": neighbors or [],
            "hypotheses": predictions or {},
        }
        out_path = self.envelopes_path / f"{envelope_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        self.index[envelope_id] = {
            "fragment_id": fragment.get("id"),
            "sealed": True,
            "created_at": payload["created_at"],
            "tags": fragment.get("tags", [])
        }
        self.save_index()
        self.log("create_envelope", {"envelope_id": envelope_id})

        # register hint for logic engine
        summary = fragment.get("summary")
        try:
            register_shadow_hint(envelope_id, summary)
        except Exception:
            pass

        # optionally update fragment with envelope reference
        if upsert_fragment:
            try:
                fragment.setdefault("shadow_envelope", envelope_id)
                upsert_fragment(fragment)
            except Exception:
                pass

    def process_fragment(self, fragment):
        # avoid duplicate processing
        if any(info.get("fragment_id") == fragment.get("id") for info in self.index.values()):
            return

        emotion_state = {}
        neighbors = []
        predictions = {}

        try:
            emotion_state = get_current_emotion_state()
        except Exception:
            pass

        try:
            neighbors = get_symbol_neighbors(tags=fragment.get("tags", []), k=5)
        except Exception:
            pass

        try:
            predictions = predict_unspoken(fragment)
        except Exception:
            pass

        self.create_envelope(fragment, emotion_state, neighbors, predictions)

    def run_sync(self):
        """Process all fragments with shadow tags synchronously."""
        fragments = self.find_shadow_candidates()
        for frag in fragments:
            self.process_fragment(frag)
        log_to_statusbox(f"[Shadow] Processed {len(fragments)} shadow fragments.")


if __name__ == "__main__":
    ShadowTransformer().run_sync()
