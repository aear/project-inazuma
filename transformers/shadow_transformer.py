import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from model_manager import load_config
from gui_hook import log_to_statusbox
from origin_record import make_origin

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
    memory_index_path = getattr(mg, "_memory_index_db_path", None)
    resolve_index_path = getattr(mg, "_resolve_index_path", None)
except Exception:
    fetch_fragment = None
    upsert_fragment = None
    memory_index_path = None
    resolve_index_path = None


class ShadowTransformer:
    """Transformer handling suppressed/unresolved/high-conflict fragments.

    Fragments with shadow-related tags are sealed into envelopes stored in
    ``AI_Children/<child>/shadow/envelopes``. Each envelope contains the raw
    fragment, optional emotional context, symbol neighbours and predictions.
    Metadata for all envelopes is tracked in ``shadow_index.json`` and an
    operational log is appended to ``shadow_log.jsonl``.
    """

    def __init__(self, child=None, root_path="AI_Children", index_db_path=None):
        config = load_config()
        self.child = child or config.get("current_child", "default_child")
        self.root = Path(root_path)
        self.index_db_path = Path(index_db_path) if index_db_path else None
        self.shadow_path = self.root / self.child / "shadow"
        self.envelopes_path = self.shadow_path / "envelopes"
        self.index_path = self.shadow_path / "shadow_index.json"
        self.log_path = self.shadow_path / "shadow_log.jsonl"
        self.candidate_queue_path = self.shadow_path / "candidate_queue.jsonl"

        self.envelopes_path.mkdir(parents=True, exist_ok=True)
        self.shadow_path.mkdir(parents=True, exist_ok=True)

        self.index = self.load_index()
        self.last_telemetry = {}

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
    def enqueue_candidate(self, fragment_id, *, tags=None, event_id=None):
        """Append one event-driven candidate reference; callers need not trigger a scan."""
        if not fragment_id:
            return
        row = {
            "fragment_id": str(fragment_id), "tags": list(tags or []),
            "event_id": event_id, "queued_at": datetime.now(timezone.utc).isoformat(),
        }
        with self.candidate_queue_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    def _load_candidate(self, fragment_id, meta=None):
        if fetch_fragment:
            try:
                payload = fetch_fragment(str(fragment_id))
                if isinstance(payload, dict):
                    return payload
            except Exception:
                pass
        path = None
        if resolve_index_path:
            try:
                path = resolve_index_path(self.child, str(fragment_id), meta or {})
            except Exception:
                path = None
        if path is None:
            filename = (meta or {}).get("filename") or f"{fragment_id}.json"
            path = self.root / self.child / "memory" / "fragments" / filename
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None

    def find_shadow_candidates(self, max_candidates=256):
        """Resolve queued/indexed candidates without walking the fragment directory."""
        shadow_tags = {"suppressed", "unresolved", "high_conflict"}
        limit = max(1, min(2048, int(max_candidates)))
        refs = {}
        source = "none"
        if self.candidate_queue_path.is_file():
            source = "queue"
            try:
                with self.candidate_queue_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        try:
                            row = json.loads(line)
                        except Exception:
                            continue
                        fragment_id = row.get("fragment_id") if isinstance(row, dict) else None
                        tags = {str(tag).lower() for tag in row.get("tags", [])} if isinstance(row, dict) else set()
                        if fragment_id and (not tags or tags & shadow_tags):
                            refs[str(fragment_id)] = row
                        if len(refs) >= limit:
                            break
            except Exception:
                refs = {}
        db_path = self.index_db_path or (
            memory_index_path(self.child) if memory_index_path and self.root == Path("AI_Children")
            else self.root / self.child / "memory" / "memory_map.sqlite"
        )
        if len(refs) < limit and Path(db_path).is_file():
            source = "queue+index" if refs else "index"
            try:
                with sqlite3.connect(str(db_path)) as connection:
                    has_tag_index = connection.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='fragment_tags'"
                    ).fetchone()
                    self._tag_index_used = bool(has_tag_index)
                    if has_tag_index:
                        placeholders = ",".join("?" for _ in shadow_tags)
                        rows = connection.execute(
                            "SELECT DISTINCT f.frag_id, f.tier, f.filename, f.tags_json "
                            "FROM fragment_tags t JOIN fragments f ON f.frag_id = t.frag_id "
                            f"WHERE t.tag IN ({placeholders}) LIMIT ?",
                            (*sorted(shadow_tags), limit - len(refs)),
                        ).fetchall()
                    else:
                        rows = connection.execute(
                            "SELECT frag_id, tier, filename, tags_json FROM fragments "
                            "WHERE tags_json LIKE ? OR tags_json LIKE ? OR tags_json LIKE ? LIMIT ?",
                            ("%suppressed%", "%unresolved%", "%high_conflict%", limit - len(refs)),
                        ).fetchall()
                for fragment_id, tier, filename, tags_json in rows:
                    try:
                        tags = json.loads(tags_json or "[]")
                    except Exception:
                        tags = []
                    if {str(tag).lower() for tag in tags} & shadow_tags:
                        refs.setdefault(str(fragment_id), {"tier": tier, "filename": filename, "tags": tags})
            except Exception:
                pass
        candidates = []
        for fragment_id, meta in list(refs.items())[:limit]:
            fragment = self._load_candidate(fragment_id, meta)
            if fragment is not None:
                candidates.append(fragment)
        self._candidate_source = source
        self._candidate_refs = len(refs)
        return candidates

    def create_envelope(self, fragment, emotion_state=None, neighbors=None, predictions=None):
        envelope_id = f"env_{uuid.uuid4().hex[:8]}"
        origin = make_origin(
            self.__class__.__name__, "V2", inputs={"tags": fragment.get("tags", [])},
            references=[fragment.get("id")] if fragment.get("id") else (),
            trigger="shadow_tag_index", metadata={"sealed": True},
        )
        payload = {
            "fragment_id": fragment.get("id"),
            "fragment": fragment,
            "sealed": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "emotion": emotion_state or {},
            "neighbors": neighbors or [],
            "hypotheses": predictions or {},
            "origins": [origin],
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
            return False

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
        return True

    def run_sync(self):
        """Process all fragments with shadow tags synchronously."""
        fragments = self.find_shadow_candidates()
        processed = 0
        for frag in fragments:
            processed += int(bool(self.process_fragment(frag)))
        self.last_telemetry = {
            "intent": "sealed_truth",
            "envelopes_created": processed,
            "shadow_index_size": len(self.index),
            "candidate_source": getattr(self, "_candidate_source", "none"),
            "candidate_references": getattr(self, "_candidate_refs", 0),
            "tag_index_used": bool(getattr(self, "_tag_index_used", False)),
        }
        log_to_statusbox(f"[Shadow] Processed {processed} shadow fragments.")

    def intent_telemetry(self):
        return dict(self.last_telemetry)


if __name__ == "__main__":
    ShadowTransformer().run_sync()
