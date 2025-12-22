from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_POLICY: Dict[str, Any] = {
    "enabled": True,
    "auto_compact": False,
    "symbol_limit": 12,
    "word_limit": 12,
    "tag_limit": 12,
    "token_sketch_bits": 64,
    "shard_importance_threshold": 0.4,
    "max_shards": 4,
    "always_shards": ["token_sketch"],
    "emotion_keys": ["intensity", "trust", "care", "stress", "risk", "novelty", "familiarity"],
    "quarantine_days": 2,
    "pending_delete_dir": "pending_delete",
    "verify_core_write": True,
    "require_shards_readable": True,
    "require_index_entry": True,
    "require_neighbor_links": True,
    "purge_pending_delete": False,
    "retain_full_fragment": False,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _tokenize(text: str) -> List[str]:
    return [tok.lower() for tok in re.findall(r"[A-Za-z0-9']+", text or "") if tok]


def _top_items(items: Iterable[str], limit: int) -> List[str]:
    counter = Counter(str(item) for item in items if item)
    ranked = [item for item, _ in counter.most_common(limit)]
    return ranked


def _hash_text(value: str, size: int = 12) -> str:
    raw = value.encode("utf-8", errors="ignore")
    return hashlib.sha1(raw).hexdigest()[:size]


def _json_checksum(payload: Any) -> str:
    try:
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        raw = json.dumps(str(payload), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()


def _guess_child_from_path(path: Path) -> Optional[str]:
    parts = path.parts
    try:
        idx = parts.index("AI_Children")
    except ValueError:
        return None
    if idx + 1 < len(parts):
        return parts[idx + 1]
    return None


def _cold_storage_root(child: str) -> Path:
    return Path("AI_Children") / child / "memory" / "cold_storage"


def _fragment_root(child: str) -> Path:
    return Path("AI_Children") / child / "memory" / "fragments"


def _pending_delete_root(child: str, policy: Dict[str, Any]) -> Path:
    pending_dir = policy.get("pending_delete_dir", DEFAULT_POLICY["pending_delete_dir"])
    return _fragment_root(child) / str(pending_dir)


def _write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = None, None
    try:
        fd, tmp_path = _temp_path(path)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=indent, ensure_ascii=False)
            handle.write("\n")
        os.replace(tmp_path, path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _temp_path(path: Path) -> Tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = None, None
    fd, tmp_path = os.open(str(path.with_suffix(path.suffix + ".tmp")), os.O_RDWR | os.O_CREAT | os.O_TRUNC), str(
        path.with_suffix(path.suffix + ".tmp")
    )
    return fd, tmp_path


def _read_last_line(path: Path, max_bytes: int = 8192) -> Optional[str]:
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            if size <= 0:
                return None
            read_size = min(size, max_bytes)
            handle.seek(-read_size, os.SEEK_END)
            data = handle.read(read_size)
    except Exception:
        return None
    text = data.decode("utf-8", errors="replace")
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else None


def _verify_core_write(core_path: Path, core_checksum: str) -> bool:
    last_line = _read_last_line(core_path)
    if not last_line:
        return False
    try:
        payload = json.loads(last_line)
    except Exception:
        return False
    checksums = payload.get("checksums")
    if isinstance(checksums, dict) and "core" in checksums:
        payload = dict(payload)
        checksums = dict(checksums)
        checksums.pop("core", None)
        payload["checksums"] = checksums
    return _json_checksum(payload) == core_checksum


def _verify_shards(shard_refs: List[Dict[str, Any]]) -> bool:
    if not shard_refs:
        return True
    for ref in shard_refs:
        path = ref.get("path")
        if not path:
            return False
        try:
            with open(path, "r", encoding="utf-8") as handle:
                json.load(handle)
        except Exception:
            return False
    return True


def _cleanup_shards(shard_refs: List[Dict[str, Any]]) -> None:
    for ref in shard_refs:
        path = ref.get("path")
        if not path:
            continue
        try:
            Path(path).unlink()
        except Exception:
            continue


def _safe_move(src: Path, dst: Path) -> bool:
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        return True
    except Exception:
        pass
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with src.open("rb") as handle:
            data = handle.read()
        with dst.open("wb") as handle:
            handle.write(data)
        src.unlink()
        return True
    except Exception:
        return False


def _move_to_pending_delete(
    fragment_path: Path,
    *,
    child: str,
    policy: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    root = _fragment_root(child)
    pending_root = _pending_delete_root(child, policy)
    try:
        relative = fragment_path.relative_to(root)
        target = pending_root / relative
    except ValueError:
        target = pending_root / fragment_path.name

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        target = target.with_name(f"{target.stem}__{stamp}{target.suffix}")

    moved = _safe_move(fragment_path, target)
    if not moved:
        return None

    moved_at = _now_iso()
    quarantine_days = int(policy.get("quarantine_days", DEFAULT_POLICY["quarantine_days"]))
    expires_at = None
    if quarantine_days > 0:
        expires_at = (datetime.now(timezone.utc)).timestamp() + (quarantine_days * 86400)
        expires_at = datetime.fromtimestamp(expires_at, timezone.utc).isoformat()

    return {
        "pending_path": str(target),
        "original_path": str(fragment_path),
        "moved_at": moved_at,
        "expires_at": expires_at,
    }


def purge_pending_delete(child: str, policy: Dict[str, Any]) -> Dict[str, int]:
    root = _pending_delete_root(child, policy)
    if not root.exists():
        return {"deleted": 0, "kept": 0}
    quarantine_days = int(policy.get("quarantine_days", DEFAULT_POLICY["quarantine_days"]))
    if quarantine_days <= 0:
        return {"deleted": 0, "kept": 0}
    cutoff = datetime.now(timezone.utc).timestamp() - (quarantine_days * 86400)
    deleted = 0
    kept = 0
    for path in root.rglob("*.json"):
        if not path.is_file():
            continue
        try:
            mtime = path.stat().st_mtime
        except Exception:
            kept += 1
            continue
        if mtime <= cutoff:
            try:
                path.unlink()
                deleted += 1
            except Exception:
                kept += 1
        else:
            kept += 1

    # Clean empty directories so pending_delete shrinks.
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        if dirnames or filenames:
            continue
        try:
            Path(dirpath).rmdir()
        except Exception:
            pass

    return {"deleted": deleted, "kept": kept}


def policy_from_config(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    policy = dict(DEFAULT_POLICY)
    if isinstance(cfg, dict):
        raw = cfg.get("cold_storage_policy")
        if isinstance(raw, dict):
            policy.update({k: raw.get(k, policy[k]) for k in policy.keys() if k in raw})

    policy["enabled"] = bool(policy.get("enabled", True))
    policy["auto_compact"] = bool(policy.get("auto_compact", False))

    for key in ("symbol_limit", "word_limit", "tag_limit", "token_sketch_bits", "max_shards"):
        try:
            policy[key] = max(1, int(policy.get(key)))
        except (TypeError, ValueError):
            policy[key] = DEFAULT_POLICY[key]

    try:
        policy["quarantine_days"] = max(0, int(policy.get("quarantine_days", 2)))
    except (TypeError, ValueError):
        policy["quarantine_days"] = DEFAULT_POLICY["quarantine_days"]

    try:
        policy["shard_importance_threshold"] = float(policy.get("shard_importance_threshold", 0.4))
    except (TypeError, ValueError):
        policy["shard_importance_threshold"] = DEFAULT_POLICY["shard_importance_threshold"]

    policy["verify_core_write"] = bool(policy.get("verify_core_write", True))
    policy["require_shards_readable"] = bool(policy.get("require_shards_readable", True))
    policy["require_index_entry"] = bool(policy.get("require_index_entry", True))
    policy["require_neighbor_links"] = bool(policy.get("require_neighbor_links", True))
    policy["purge_pending_delete"] = bool(policy.get("purge_pending_delete", False))
    policy["retain_full_fragment"] = bool(policy.get("retain_full_fragment", False))

    pending_dir = policy.get("pending_delete_dir", DEFAULT_POLICY["pending_delete_dir"])
    if isinstance(pending_dir, str) and pending_dir.strip():
        policy["pending_delete_dir"] = pending_dir.strip()
    else:
        policy["pending_delete_dir"] = DEFAULT_POLICY["pending_delete_dir"]

    always = policy.get("always_shards", DEFAULT_POLICY["always_shards"])
    if isinstance(always, (list, tuple)):
        policy["always_shards"] = [str(item) for item in always if item]
    else:
        policy["always_shards"] = list(DEFAULT_POLICY["always_shards"])

    emotion_keys = policy.get("emotion_keys", DEFAULT_POLICY["emotion_keys"])
    if isinstance(emotion_keys, (list, tuple)):
        policy["emotion_keys"] = [str(item) for item in emotion_keys if item]
    else:
        policy["emotion_keys"] = list(DEFAULT_POLICY["emotion_keys"])

    return policy


def _extract_metadata(fragment: Dict[str, Any]) -> Dict[str, Any]:
    meta = fragment.get("metadata")
    if isinstance(meta, dict):
        return meta
    return {}


def _extract_tags(fragment: Dict[str, Any]) -> List[str]:
    tags = fragment.get("tags")
    if isinstance(tags, list):
        return [str(t) for t in tags if t]
    meta = _extract_metadata(fragment)
    mtags = meta.get("tags")
    if isinstance(mtags, list):
        return [str(t) for t in mtags if t]
    return []


def _extract_flags(fragment: Dict[str, Any]) -> List[str]:
    meta = _extract_metadata(fragment)
    flags = meta.get("flags")
    if isinstance(flags, list):
        return [str(f) for f in flags if f]
    return []


def _extract_symbols(fragment: Dict[str, Any]) -> List[str]:
    symbols: List[str] = []
    for key in ("symbols", "symbols_spoken", "attempted_symbols"):
        value = fragment.get(key)
        if isinstance(value, list):
            symbols.extend(str(s) for s in value if s)
        elif isinstance(value, str):
            symbols.append(value)
    context = fragment.get("context")
    if isinstance(context, dict):
        ctx_symbols = context.get("symbols")
        if isinstance(ctx_symbols, list):
            symbols.extend(str(s) for s in ctx_symbols if s)
    return [s for s in symbols if s]


def _extract_text(fragment: Dict[str, Any]) -> str:
    for key in ("text", "transcript", "utterance", "content", "summary"):
        value = fragment.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    payload = fragment.get("payload")
    if isinstance(payload, dict):
        for key in ("text", "transcript", "summary", "content"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    context = fragment.get("context")
    if isinstance(context, dict):
        value = context.get("text")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_entities(fragment: Dict[str, Any]) -> List[str]:
    entities = fragment.get("entities")
    if isinstance(entities, list):
        return [str(ent) for ent in entities if ent]
    context = fragment.get("context")
    if isinstance(context, dict):
        centities = context.get("entities")
        if isinstance(centities, list):
            return [str(ent) for ent in centities if ent]
    return []


def _emotion_summary(fragment: Dict[str, Any], policy: Dict[str, Any]) -> Dict[str, float]:
    emotions = fragment.get("emotions")
    if not isinstance(emotions, dict):
        return {}
    summary: Dict[str, float] = {}
    for key in policy.get("emotion_keys", []):
        if key in emotions:
            summary[key] = _safe_float(emotions.get(key))
    if not summary and isinstance(emotions.get("summary"), dict):
        cooled = emotions["summary"].get("cooled_intensity")
        if cooled is not None:
            summary["intensity"] = _safe_float(cooled)
    return summary


def _topic_hash(anchors: Dict[str, List[str]]) -> str:
    parts = []
    for key in ("symbols", "words", "entities"):
        items = anchors.get(key) or []
        if items:
            parts.append(",".join(items))
    return _hash_text("|".join(parts) or "empty")


def build_cold_core(fragment: Dict[str, Any], policy: Dict[str, Any]) -> Dict[str, Any]:
    metadata = _extract_metadata(fragment)
    frag_id = fragment.get("id") or fragment.get("fragment_id") or ""
    symbols = _extract_symbols(fragment)
    anchors = {
        "symbols": _top_items(symbols, policy["symbol_limit"]),
        "words": _top_items(_tokenize(_extract_text(fragment)), policy["word_limit"]),
        "entities": _top_items(_extract_entities(fragment), policy["word_limit"]),
    }
    anchors["topic_hash"] = _topic_hash(anchors)

    timestamp = fragment.get("timestamp") or metadata.get("timestamp_start")
    timestamps = {
        "start": metadata.get("timestamp_start") or timestamp,
        "end": metadata.get("timestamp_end") or timestamp,
        "captured": timestamp,
    }

    provenance = {
        "source": fragment.get("source") or metadata.get("source"),
        "device": metadata.get("device_name") or fragment.get("device"),
        "module": fragment.get("module") or fragment.get("producer"),
        "session": fragment.get("session") or metadata.get("session_id"),
    }

    structure = {
        "prev": fragment.get("prev_id") or fragment.get("prev"),
        "next": fragment.get("next_id") or fragment.get("next"),
        "cluster_id": fragment.get("cluster_id") or fragment.get("cluster"),
        "synapse_edges": fragment.get("synapse_edges") or fragment.get("synapses"),
    }

    checksums = {
        "fragment": _json_checksum(fragment),
        "anchors": _hash_text(json.dumps(anchors, sort_keys=True, ensure_ascii=False)),
    }

    emotion_snapshot_id = metadata.get("emotion_snapshot_id") or fragment.get("emotion_snapshot_id")

    core = {
        "version": 1,
        "fragment_id": frag_id,
        "type": fragment.get("type"),
        "timestamps": timestamps,
        "tags": _extract_tags(fragment)[: policy["tag_limit"]],
        "trust": _safe_float(fragment.get("trust") or fragment.get("emotions", {}).get("trust")),
        "emotion_snapshot_id": emotion_snapshot_id,
        "emotion_summary": _emotion_summary(fragment, policy),
        "anchors": anchors,
        "structure": {k: v for k, v in structure.items() if v},
        "checksums": checksums,
        "provenance": {k: v for k, v in provenance.items() if v},
    }
    return core


def _simhash(tokens: List[str], bits: int) -> str:
    if not tokens:
        return "0" * (bits // 4)
    weights = [0] * bits
    for token in tokens:
        h = int(hashlib.md5(token.encode("utf-8", errors="ignore")).hexdigest(), 16)
        for i in range(bits):
            if h & (1 << i):
                weights[i] += 1
            else:
                weights[i] -= 1
    value = 0
    for i, weight in enumerate(weights):
        if weight > 0:
            value |= 1 << i
    width = bits // 4
    return f"{value:0{width}x}"


def _importance_score(fragment: Dict[str, Any]) -> float:
    importance = _safe_float(fragment.get("importance"))
    emotions = fragment.get("emotions")
    intensity = 0.0
    if isinstance(emotions, dict):
        intensity = abs(_safe_float(emotions.get("intensity")))
    centrality = _safe_float(fragment.get("graph_centrality"))
    recurrence = _safe_float(fragment.get("recurrence"))
    return max(importance, intensity, centrality, recurrence)


def _quantize_vector(vector: List[float], limit: int = 64, decimals: int = 3) -> List[float]:
    if not vector:
        return []
    trimmed = vector[:limit]
    return [round(_safe_float(v), decimals) for v in trimmed]


def build_shards(fragment: Dict[str, Any], policy: Dict[str, Any]) -> List[Dict[str, Any]]:
    shards: List[Dict[str, Any]] = []
    text = _extract_text(fragment)
    tokens = _tokenize(text)
    if tokens:
        shards.append(
            {
                "type": "token_sketch",
                "payload": {"simhash": _simhash(tokens, policy["token_sketch_bits"]), "count": len(tokens)},
            }
        )

    symbols = _extract_symbols(fragment)
    if symbols:
        counter = Counter(symbols)
        top = counter.most_common(policy["symbol_limit"])
        shards.append(
            {
                "type": "symbol_counts",
                "payload": {"counts": {sym: count for sym, count in top}, "total": len(symbols)},
            }
        )

    importance = _importance_score(fragment)
    high_value = importance >= policy.get("shard_importance_threshold", 0.4)

    embedding = None
    for key in ("embedding", "embedding_small", "audio_embedding", "vision_embedding", "embedding_vector"):
        value = fragment.get(key)
        if isinstance(value, list) and value:
            embedding = value
            break
    if embedding and high_value:
        shards.append(
            {
                "type": "embedding_sketch",
                "payload": {"vector": _quantize_vector(embedding), "dim": len(embedding)},
            }
        )

    feature_source = None
    for key in ("audio_features", "image_features", "video_features"):
        value = fragment.get(key)
        if value is not None:
            feature_source = value
            break
    if feature_source is not None and high_value:
        try:
            raw = json.dumps(feature_source, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            raw = str(feature_source)
        shards.append(
            {
                "type": "feature_hash",
                "payload": {"hash": _hash_text(raw, 16)},
            }
        )

    poem = fragment.get("memory_poem") or fragment.get("capture_poem") or fragment.get("poem")
    if isinstance(poem, str) and poem.strip():
        shards.append({"type": "memory_poem", "payload": {"text": poem.strip()}})

    always = set(policy.get("always_shards", []))
    kept: List[Dict[str, Any]] = []
    for shard in shards:
        if shard["type"] in always or high_value:
            kept.append(shard)

    max_shards = policy.get("max_shards", 4)
    return kept[:max_shards]


def _heal_reason_codes(fragment: Dict[str, Any]) -> List[str]:
    tags = {t.lower() for t in _extract_tags(fragment)}
    flags = {f.lower() for f in _extract_flags(fragment)}
    codes = []
    if tags & {"corrupt", "corruption", "truncated", "broken"}:
        codes.append("corruption")
    if tags & {"drift", "stale", "mismatch"}:
        codes.append("drift")
    if tags & {"missing", "unmapped", "orphaned"}:
        codes.append("missing_mappings")
    if tags & {"entropy", "symbol_soup", "noise", "noisy"}:
        codes.append("high_entropy")
    if flags & {"corrupt", "drift", "missing", "entropy"}:
        codes.extend(sorted(flags & {"corrupt", "drift", "missing", "entropy"}))
    return list(dict.fromkeys(codes))


def build_heal_ticket(fragment: Dict[str, Any], core: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    reason_codes = _heal_reason_codes(fragment)
    if not reason_codes:
        return None

    required_inputs = ["neighbors", "cluster_prototype"]
    if "corruption" in reason_codes:
        required_inputs.append("source_media")
    if "missing_mappings" in reason_codes:
        required_inputs.append("symbol_map")

    priority = min(1.0, _importance_score(fragment))

    ticket = {
        "fragment_id": core.get("fragment_id"),
        "created_at": _now_iso(),
        "reason_codes": reason_codes,
        "required_inputs": required_inputs,
        "priority": round(priority, 4),
        "context": {
            "summary": fragment.get("summary"),
            "tags": core.get("tags", []),
            "topic_hash": core.get("anchors", {}).get("topic_hash"),
        },
    }
    return ticket


def _write_shards(
    storage_root: Path,
    fragment_id: str,
    shards: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not shards:
        return []
    shard_dir = storage_root / "shards"
    shard_refs: List[Dict[str, Any]] = []
    for shard in shards:
        shard_type = shard.get("type", "unknown")
        filename = f"{fragment_id}__{shard_type}.json"
        path = shard_dir / filename
        _write_json(path, shard)
        shard_refs.append({"type": shard_type, "path": str(path), "checksum": _json_checksum(shard)})
    return shard_refs


def _append_cold_core(storage_root: Path, core: Dict[str, Any]) -> None:
    core_path = storage_root / "cold_core.jsonl"
    core_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(core, ensure_ascii=False)
    with core_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _append_heal_ticket(storage_root: Path, ticket: Dict[str, Any]) -> None:
    path = storage_root / "heal_tickets.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(ticket, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def compact_fragment(
    fragment: Dict[str, Any],
    *,
    policy: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    core = build_cold_core(fragment, policy)
    core_checksum = _json_checksum(core)
    core.setdefault("checksums", {})["core"] = core_checksum
    shards = build_shards(fragment, policy)
    ticket = build_heal_ticket(fragment, core)
    if ticket:
        shards.insert(
            0,
            {
                "type": "clue_context",
                "payload": {
                    "summary": fragment.get("summary"),
                    "tags": core.get("tags", []),
                    "symbols": core.get("anchors", {}).get("symbols", []),
                },
            },
        )
        max_shards = policy.get("max_shards", 4)
        shards = shards[:max_shards]
    stub = {
        "id": core.get("fragment_id") or fragment.get("id"),
        "type": fragment.get("type"),
        "source": fragment.get("source"),
        "timestamp": fragment.get("timestamp") or core.get("timestamps", {}).get("captured"),
        "summary": fragment.get("summary") or core.get("anchors", {}).get("topic_hash"),
        "tags": core.get("tags", []),
        "importance": _safe_float(fragment.get("importance")),
        "emotions": fragment.get("emotions") if isinstance(fragment.get("emotions"), dict) else {},
        "symbols": core.get("anchors", {}).get("symbols", []),
        "tier": "cold",
        "modality": fragment.get("modality") or fragment.get("metadata", {}).get("modality"),
        "cold_core": core,
        "cold_core_checksum": core_checksum,
        "reconstructed": False,
    }
    return stub, core, shards, ticket


def compact_fragment_file(
    fragment_path: Path,
    *,
    child: Optional[str] = None,
    policy: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    try:
        fragment = json.loads(fragment_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(fragment.get("cold_core"), dict) and fragment.get("tier") == "cold":
        return None

    if child is None:
        child = _guess_child_from_path(fragment_path) or "Inazuma_Yagami"

    policy = policy or dict(DEFAULT_POLICY)
    if not policy.get("enabled", True):
        return None
    if not fragment_path.exists():
        return None

    stub, core, shards, ticket = compact_fragment(fragment, policy=policy)
    storage_root = _cold_storage_root(child)
    shard_refs = _write_shards(storage_root, core.get("fragment_id", "unknown"), shards)
    if shard_refs:
        stub["cold_shards"] = shard_refs
    stub["cold_compacted_at"] = _now_iso()
    _append_cold_core(storage_root, core)

    if policy.get("verify_core_write", True):
        core_checksum = core.get("checksums", {}).get("core")
        if not core_checksum or not _verify_core_write(storage_root / "cold_core.jsonl", core_checksum):
            _cleanup_shards(shard_refs)
            return {"fragment_id": core.get("fragment_id"), "status": "failed", "reason": "core_write_unverified"}

    if policy.get("require_shards_readable", True):
        if not _verify_shards(shard_refs):
            _cleanup_shards(shard_refs)
            return {"fragment_id": core.get("fragment_id"), "status": "failed", "reason": "shard_unreadable"}

    if ticket:
        _append_heal_ticket(storage_root, ticket)

    if policy.get("retain_full_fragment", False):
        return {
            "fragment_id": core.get("fragment_id"),
            "status": "retained",
            "shards": len(shard_refs),
            "ticket": bool(ticket),
        }

    pending_info = _move_to_pending_delete(fragment_path, child=child, policy=policy)
    if not pending_info:
        return {
            "fragment_id": core.get("fragment_id"),
            "status": "failed",
            "reason": "pending_delete_move_failed",
        }
    stub["quarantine"] = pending_info
    _write_json(fragment_path, stub)
    return {
        "fragment_id": core.get("fragment_id"),
        "status": "compacted",
        "shards": len(shard_refs),
        "ticket": bool(ticket),
    }


__all__ = [
    "policy_from_config",
    "build_cold_core",
    "build_shards",
    "build_heal_ticket",
    "compact_fragment",
    "compact_fragment_file",
    "purge_pending_delete",
]
