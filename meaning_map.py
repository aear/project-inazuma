import json
import math
import time
import hashlib
import gc
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from transformers.fractal_multidimensional_transformers import FractalTransformer
from model_manager import load_config, seed_self_question, get_inastate, update_inastate
from gui_hook import log_to_statusbox
from symbol_generator import generate_symbol_from_parts, available_symbol_components
import random
from text_memory import build_text_symbol_links
from symbol_word_utils import proto_confidence

_CORRUPT_QUEUE_LIMIT = 120
_CORRUPT_PROMPT_STEP = 5
DEFAULT_MEANING_POLICY = {
    "fragment_burst": 96,
    "batch_size": 16,
    "build_budget_ms": 280.0,
    "cluster_threshold": 0.88,
    "word_merge_threshold": 0.9,
    "queue_max_items": 600,
    "max_components_per_word": 220,
    "max_new_words_per_run": 36,
    "max_tags_per_word": 32,
    "vector_round_digits": 6,
    "gc_every_batches": 4,
    "max_words_total": 0,
}


def _note_corrupt_fragment(path: Path, error: Exception) -> None:
    try:
        size = path.stat().st_size
    except Exception:
        size = None

    reason = "read_error"
    if isinstance(error, json.JSONDecodeError):
        reason = "invalid_json"
    if size == 0:
        reason = "empty"

    entry = {
        "id": path.stem,
        "file": path.name,
        "path": str(path),
        "reason": reason,
        "size_bytes": size,
        "error": str(error)[:200],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "meaning_map",
    }

    try:
        queue = get_inastate("corrupt_fragments") or []
        if not isinstance(queue, list):
            queue = []
        queue.append(entry)
        if len(queue) > _CORRUPT_QUEUE_LIMIT:
            queue = queue[-_CORRUPT_QUEUE_LIMIT:]
        update_inastate("corrupt_fragments", queue)

        stats = get_inastate("corrupt_fragment_stats") or {}
        if not isinstance(stats, dict):
            stats = {}
        total = int(stats.get("total", 0) or 0) + 1
        last_prompt_total = int(stats.get("last_prompt_total", 0) or 0)
        stats.update(
            {
                "total": total,
                "last_seen": entry["timestamp"],
                "last_reason": reason,
            }
        )
        update_inastate("corrupt_fragment_stats", stats)

        if total - last_prompt_total >= _CORRUPT_PROMPT_STEP:
            seed_self_question("I keep encountering corrupted fragments. Should I repair or quarantine them?")
            stats["last_prompt_total"] = total
            update_inastate("corrupt_fragment_stats", stats)
    except Exception:
        # Avoid cascading failures in a diagnostic helper.
        pass


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_iso_ts(value: Optional[str]) -> float:
    if not value:
        return 0.0
    raw = str(value)
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def _meaning_policy(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    policy = DEFAULT_MEANING_POLICY.copy()
    raw = cfg.get("meaning_map_policy", {}) if isinstance(cfg, dict) else {}
    if isinstance(raw, dict):
        for key in policy.keys():
            if key in raw:
                policy[key] = raw.get(key)
    policy["fragment_burst"] = max(8, int(_safe_float(policy.get("fragment_burst"), 96)))
    policy["batch_size"] = max(1, int(_safe_float(policy.get("batch_size"), 16)))
    policy["build_budget_ms"] = max(0.0, _safe_float(policy.get("build_budget_ms"), 280.0))
    policy["cluster_threshold"] = max(0.0, min(1.0, _safe_float(policy.get("cluster_threshold"), 0.88)))
    policy["word_merge_threshold"] = max(0.0, min(1.0, _safe_float(policy.get("word_merge_threshold"), 0.9)))
    policy["queue_max_items"] = max(40, int(_safe_float(policy.get("queue_max_items"), 600)))
    policy["max_components_per_word"] = max(20, int(_safe_float(policy.get("max_components_per_word"), 220)))
    policy["max_new_words_per_run"] = max(1, int(_safe_float(policy.get("max_new_words_per_run"), 36)))
    policy["max_tags_per_word"] = max(4, int(_safe_float(policy.get("max_tags_per_word"), 32)))
    policy["vector_round_digits"] = max(2, min(8, int(_safe_float(policy.get("vector_round_digits"), 6))))
    policy["gc_every_batches"] = max(0, int(_safe_float(policy.get("gc_every_batches"), 4)))
    policy["max_words_total"] = max(0, int(_safe_float(policy.get("max_words_total"), 0)))
    return policy


def _symbol_words_path(child: str) -> Path:
    return Path("AI_Children") / child / "memory" / "symbol_words.json"


def _meaning_state_path(child: str) -> Path:
    return Path("AI_Children") / child / "memory" / "meaning_map_state.json"


def _memory_index_path(child: str) -> Path:
    return Path("AI_Children") / child / "memory" / "memory_map.json"


def _load_json_dict(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return default
    return data if isinstance(data, dict) else default


def _save_json_dict(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    except Exception:
        return


def _load_symbol_words_payload(child: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    path = _symbol_words_path(child)
    payload = _load_json_dict(path, {})
    words_raw = payload.get("words")
    words: List[Dict[str, Any]] = []
    if isinstance(words_raw, list):
        for entry in words_raw:
            if isinstance(entry, dict):
                words.append(entry)
    preserved: Dict[str, Any] = {}
    for key in ("proto_words", "multi_symbol_words"):
        if key in payload:
            preserved[key] = payload[key]
    return words, preserved


def _next_word_index(words: List[Dict[str, Any]]) -> int:
    best = -1
    prefix = "sym_word_"
    for word in words:
        sym_id = str(word.get("symbol_word_id") or "")
        if not sym_id.startswith(prefix):
            continue
        try:
            idx = int(sym_id[len(prefix):])
        except ValueError:
            continue
        best = max(best, idx)
    return best + 1


def _word_vector(word: Dict[str, Any]) -> Optional[List[float]]:
    vec = word.get("vector")
    if isinstance(vec, list) and vec:
        cleaned = []
        for value in vec:
            if isinstance(value, (int, float)):
                cleaned.append(float(value))
        if cleaned:
            return cleaned
    return None


def _round_vector(values: List[float], digits: int) -> List[float]:
    return [round(float(value), digits) for value in values]


def _apply_word_caps(word: Dict[str, Any], policy: Dict[str, Any]) -> None:
    max_components = int(policy.get("max_components_per_word", 220))
    max_tags = int(policy.get("max_tags_per_word", 32))
    digits = int(policy.get("vector_round_digits", 6))

    components = [str(fid) for fid in (word.get("components") or []) if fid]
    if max_components > 0 and len(components) > max_components:
        components = components[-max_components:]
    word["components"] = components

    tags = sorted({str(tag) for tag in (word.get("tags") or []) if tag})
    if max_tags > 0 and len(tags) > max_tags:
        tags = tags[-max_tags:]
    word["tags"] = tags

    vector = _word_vector(word)
    if vector:
        word["vector"] = _round_vector(vector, digits)


def _enforce_word_budget(words: List[Dict[str, Any]], max_words_total: int) -> int:
    if max_words_total <= 0:
        return 0
    if len(words) <= max_words_total:
        return 0
    overflow = len(words) - max_words_total
    scored: List[Tuple[int, int, float, int]] = []
    for idx, word in enumerate(words):
        usage = int(_safe_float(word.get("usage_count"), 0))
        count = int(_safe_float(word.get("count"), 0))
        updated = _parse_iso_ts(word.get("updated_at") or word.get("birth_time"))
        scored.append((usage, count, updated, idx))
    scored.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    drop_indices = {idx for _, _, _, idx in scored[:overflow]}
    words[:] = [word for idx, word in enumerate(words) if idx not in drop_indices]
    return overflow


def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
    if not v1 or not v2:
        return 0.0
    length = min(len(v1), len(v2))
    if length <= 0:
        return 0.0
    dot = sum(v1[i] * v2[i] for i in range(length))
    norm1 = math.sqrt(sum(v1[i] * v1[i] for i in range(length)))
    norm2 = math.sqrt(sum(v2[i] * v2[i] for i in range(length)))
    return dot / (norm1 * norm2 + 1e-8)


def _merge_vectors(base: List[float], base_count: int, new: List[float], new_count: int) -> List[float]:
    length = min(len(base), len(new))
    if length <= 0:
        return []
    total = max(0, base_count) + max(0, new_count)
    if total <= 0:
        return [round(value, 6) for value in new[:length]]
    return [
        round(((base[i] * max(0, base_count)) + (new[i] * max(0, new_count))) / total, 6)
        for i in range(length)
    ]



def _coalesce_summary(chunks: List[str], fallback: str) -> str:
    seen: List[str] = []
    for chunk in chunks:
        value = str(chunk or "").strip()
        if value and value not in seen:
            seen.append(value)
        if len(seen) >= 2:
            break
    return " + ".join(seen) if seen else fallback



def _upsert_symbol_pair_entry(
    store: Dict[str, Any],
    key: str,
    *,
    sequence: List[str],
    pair_count: int,
    centroid: List[float],
    tags: List[str],
    summary_text: str,
    policy: Dict[str, Any],
    base_confidence: float,
    source: Optional[str] = None,
    stability_divisor: float = 10.0,
    flexible_limit: int = 5,
) -> Dict[str, Any]:
    entry = store.get(key) if isinstance(store.get(key), dict) else {}
    now = datetime.now(timezone.utc).isoformat()
    prior_uses = int(_safe_float(entry.get("uses"), 0))
    total_uses = prior_uses + max(1, int(pair_count))
    merged_tags = sorted({str(tag) for tag in (entry.get("tags") or []) if tag} | {str(tag) for tag in tags if tag})
    max_tags = int(policy.get("max_tags_per_word", 32))
    if max_tags > 0 and len(merged_tags) > max_tags:
        merged_tags = merged_tags[-max_tags:]
    prior_vec = _word_vector(entry)
    if prior_vec:
        merged_vec = _merge_vectors(prior_vec, prior_uses, list(centroid), pair_count)
    else:
        merged_vec = list(centroid)
    if summary_text and not entry.get("summary"):
        entry["summary"] = summary_text
    entry.update(
        {
            "sequence": list(sequence),
            "components": list(sequence),
            "uses": total_uses,
            "last_seen": now,
            "confidence": proto_confidence(total_uses, base=base_confidence),
            "flexible": total_uses < int(flexible_limit),
            "stability": round(min(1.0, total_uses / max(1.0, float(stability_divisor))), 3),
            "length": len(sequence),
            "tags": merged_tags,
            "vector": _round_vector([float(value) for value in merged_vec], int(policy.get("vector_round_digits", 6))),
            "updated_at": now,
        }
    )
    if summary_text:
        entry["summary"] = summary_text
    if source:
        entry["source"] = source
    if not entry.get("created"):
        entry["created"] = now
    store[key] = entry
    return entry



def _promote_symbol_pairs(
    preserved: Dict[str, Any],
    cluster_members: List[Dict[str, Any]],
    fragments_by_id: Dict[str, Dict[str, Any]],
    centroid: List[float],
    tag_set: List[str],
    summary_text: str,
    policy: Dict[str, Any],
) -> Tuple[int, int]:
    proto_store = preserved.get("proto_words") if isinstance(preserved.get("proto_words"), dict) else {}
    multi_store = preserved.get("multi_symbol_words") if isinstance(preserved.get("multi_symbol_words"), dict) else {}
    preserved["proto_words"] = proto_store
    preserved["multi_symbol_words"] = multi_store

    pair_counts: Dict[Tuple[str, str], int] = {}
    pair_tags: Dict[Tuple[str, str], set] = {}
    pair_summaries: Dict[Tuple[str, str], List[str]] = {}

    for member in cluster_members:
        frag_id = str(member.get("id") or "").strip()
        frag = fragments_by_id.get(frag_id)
        if not isinstance(frag, dict):
            continue
        sequence = frag.get("symbols_spoken")
        if not isinstance(sequence, list) or len(sequence) < 2:
            continue
        cleaned = [str(symbol_id) for symbol_id in sequence if symbol_id]
        if len(cleaned) < 2:
            continue
        frag_tags = {str(tag) for tag in (frag.get("tags") or []) if tag}
        frag_summary = str(frag.get("summary") or member.get("summary") or summary_text)
        for idx in range(len(cleaned) - 1):
            pair = (cleaned[idx], cleaned[idx + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
            pair_tags.setdefault(pair, set()).update(frag_tags)
            pair_summaries.setdefault(pair, []).append(frag_summary)

    proto_updates = 0
    multi_updates = 0
    for pair, pair_count in pair_counts.items():
        sequence = list(pair)
        proto_key = "_".join(sequence)
        pair_key = f"pair:{proto_key}"
        merged_tags = sorted(set(tag_set) | pair_tags.get(pair, set()))
        pair_summary = _coalesce_summary(pair_summaries.get(pair, []), summary_text)
        previous_proto_uses = int(_safe_float((proto_store.get(proto_key) or {}).get("uses"), 0)) if isinstance(proto_store.get(proto_key), dict) else 0
        proto_entry = _upsert_symbol_pair_entry(
            proto_store,
            proto_key,
            sequence=sequence,
            pair_count=pair_count,
            centroid=centroid,
            tags=merged_tags,
            summary_text=pair_summary,
            policy=policy,
            base_confidence=0.18,
            stability_divisor=10.0,
            flexible_limit=5,
        )
        if int(_safe_float(proto_entry.get("uses"), 0)) > previous_proto_uses:
            proto_updates += 1

        total_proto_uses = int(_safe_float(proto_entry.get("uses"), 0))
        if pair_count < 2 and total_proto_uses < 2:
            continue
        previous_multi_uses = int(_safe_float((multi_store.get(pair_key) or {}).get("uses"), 0)) if isinstance(multi_store.get(pair_key), dict) else 0
        multi_entry = _upsert_symbol_pair_entry(
            multi_store,
            pair_key,
            sequence=sequence,
            pair_count=pair_count,
            centroid=centroid,
            tags=merged_tags,
            summary_text=pair_summary,
            policy=policy,
            base_confidence=0.24,
            source="meaning_cluster",
            stability_divisor=8.0,
            flexible_limit=4,
        )
        if int(_safe_float(multi_entry.get("uses"), 0)) > previous_multi_uses:
            multi_updates += 1

    return proto_updates, multi_updates


def _fragment_signature(child: str, frag_id: str, meta: Dict[str, Any]) -> Optional[str]:
    filename = meta.get("filename") or f"frag_{frag_id}.json"
    tier = meta.get("tier")
    base = Path("AI_Children") / child / "memory" / "fragments"
    paths = []
    if tier:
        paths.append(base / str(tier) / str(filename))
    paths.append(base / str(filename))
    target = None
    for path in paths:
        if path.exists():
            target = path
            break
    if target is None:
        return None
    try:
        st = target.stat()
    except OSError:
        return None
    tags = meta.get("tags") if isinstance(meta.get("tags"), list) else []
    tag_text = ",".join(sorted(str(tag).lower() for tag in tags if tag))
    digest = hashlib.sha1(tag_text.encode("utf-8")).hexdigest()[:12]
    return f"{tier}:{filename}:{int(st.st_mtime_ns)}:{int(st.st_size)}:{digest}"


def _load_symbolic_index(child: str) -> Dict[str, Dict[str, Any]]:
    path = _memory_index_path(child)
    payload = _load_json_dict(path, {})
    symbolic: Dict[str, Dict[str, Any]] = {}
    for frag_id, meta in payload.items():
        if not isinstance(meta, dict):
            continue
        tags = meta.get("tags")
        if not isinstance(tags, list):
            continue
        lowered = {str(tag).lower() for tag in tags if tag}
        if "symbolic" not in lowered:
            continue
        if "sensor_incoherent" in lowered:
            continue
        symbolic[str(frag_id)] = meta
    return symbolic


def _collect_dirty_symbolic_ids(
    child: str,
    index: Dict[str, Dict[str, Any]],
    processed_signatures: Dict[str, str],
) -> Tuple[List[str], Dict[str, str], List[str]]:
    dirty: List[str] = []
    current_signatures: Dict[str, str] = {}
    for frag_id, meta in index.items():
        signature = _fragment_signature(child, frag_id, meta)
        if not signature:
            continue
        current_signatures[frag_id] = signature
        if processed_signatures.get(frag_id) != signature:
            dirty.append(frag_id)
    removed = [frag_id for frag_id in processed_signatures.keys() if frag_id not in current_signatures]
    return dirty, current_signatures, removed


def _resolve_index_path(child: str, frag_id: str, meta: Dict[str, Any]) -> Optional[Path]:
    base = Path("AI_Children") / child / "memory" / "fragments"
    filename = meta.get("filename") or f"frag_{frag_id}.json"
    tier = meta.get("tier")
    if tier:
        candidate = base / str(tier) / str(filename)
        if candidate.exists():
            return candidate
    candidate = base / str(filename)
    if candidate.exists():
        return candidate
    return None


def _load_fragments_by_ids(
    child: str,
    index: Dict[str, Dict[str, Any]],
    fragment_ids: List[str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    fragments: List[Dict[str, Any]] = []
    missing: List[str] = []
    for frag_id in fragment_ids:
        meta = index.get(frag_id)
        if not isinstance(meta, dict):
            missing.append(frag_id)
            continue
        path = _resolve_index_path(child, frag_id, meta)
        if path is None:
            missing.append(frag_id)
            continue
        try:
            with path.open("r", encoding="utf-8") as fh:
                frag = json.load(fh)
        except Exception as exc:
            _note_corrupt_fragment(path, exc)
            missing.append(frag_id)
            continue
        if not isinstance(frag, dict):
            missing.append(frag_id)
            continue
        frag["id"] = frag.get("id") or frag_id
        tags = frag.get("tags") if isinstance(frag.get("tags"), list) else []
        lowered_tags = {str(tag).lower() for tag in tags if tag}
        if "symbolic" not in lowered_tags:
            missing.append(frag_id)
            continue
        if "sensor_incoherent" in lowered_tags:
            missing.append(frag_id)
            continue
        fragments.append(frag)
    return fragments, missing


def _cluster_encoded(encoded: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
    clusters: List[Dict[str, Any]] = []
    for enc in encoded:
        vec = enc.get("vector")
        if not isinstance(vec, list) or not vec:
            continue
        member = {
            "id": enc.get("id"),
            "tags": list(enc.get("tags") or []),
            "summary": enc.get("summary"),
        }
        best_idx = -1
        best_score = 0.0
        for idx, cluster in enumerate(clusters):
            score = _cosine_similarity(vec, cluster["centroid"])
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx >= 0 and best_score >= threshold:
            target = clusters[best_idx]
            count = int(target["count"])
            centroid = target["centroid"]
            target["members"].append(member)
            target["centroid"] = _merge_vectors(centroid, count, vec, 1)
            target["count"] = count + 1
        else:
            clusters.append({"members": [member], "centroid": [float(v) for v in vec], "count": 1})
    return clusters

def load_base_model(child):
    path = Path.home() / "Projects" / "Project Inazuma" / "AI_Children" / child / "ina_pretrained_model.json"
    if not path.exists():
        log_to_statusbox(f"[Symbols] Base model not found at {path}")
        return {}
    try:
        with open(path, "r") as f:
            model = json.load(f)
            log_to_statusbox("[Symbols] Base model loaded successfully.")
            return model
    except Exception as e:
        log_to_statusbox(f"[Symbols] Failed to load base model: {e}")
        return {}

def evolve_unused_symbols(symbol_words, threshold=2):
    log_to_statusbox("[Symbols] Checking for underused symbols...")
    for word in symbol_words:
        usage = word.get("count", 0)
        if usage < threshold:
            seed_self_question(f"Why do I rarely use '{word['symbol_word_id']}'?")

def detect_conflicted_symbols(fragments, symbol_words):
    log_to_statusbox("[Symbols] Checking for symbol conflicts...")
    used = {}
    for frag in fragments:
        sid = frag.get("sound_symbol")
        sw = frag.get("symbol_word_id")
        if sid and sw:
            if sid not in used:
                used[sid] = set()
            used[sid].add(sw)

    for sid, seen in used.items():
        if len(seen) > 1:
            seed_self_question(f"Did I confuse meaning for {sid} between: {', '.join(seen)}?")

def cluster_symbols_and_generate_words(child: str, cfg: Optional[Dict[str, Any]] = None):
    policy = _meaning_policy(cfg)
    start_perf = time.perf_counter()
    index = _load_symbolic_index(child)
    if not index:
        log_to_statusbox("[Symbols] No symbolic fragments found in index.")
        return

    state = _load_json_dict(_meaning_state_path(child), {"pending": [], "processed_signatures": {}})
    processed_signatures = state.get("processed_signatures")
    if not isinstance(processed_signatures, dict):
        processed_signatures = {}
    pending = state.get("pending") if isinstance(state.get("pending"), list) else []
    pending_ids = [str(fid) for fid in pending if fid]

    dirty_ids, current_signatures, removed_ids = _collect_dirty_symbolic_ids(child, index, processed_signatures)
    if removed_ids:
        for frag_id in removed_ids:
            processed_signatures.pop(frag_id, None)
    if dirty_ids:
        def _queue_key(frag_id: str) -> Tuple[float, float]:
            meta = index.get(frag_id, {})
            ts = _parse_iso_ts(meta.get("last_seen") or meta.get("timestamp"))
            importance = _safe_float(meta.get("importance"), 0.0)
            return ts, importance
        dirty_ids = sorted(dirty_ids, key=_queue_key, reverse=True)

    existing_pending = set(pending_ids)
    for frag_id in dirty_ids:
        if frag_id not in existing_pending:
            pending_ids.append(frag_id)
            existing_pending.add(frag_id)
    if not pending_ids:
        log_to_statusbox("[Symbols] No dirty symbolic fragments to process.")
        return

    queue_max = policy["queue_max_items"]
    if len(pending_ids) > queue_max:
        pending_ids = pending_ids[:queue_max]

    words, preserved = _load_symbol_words_payload(child)
    for word in words:
        _apply_word_caps(word, policy)
    words_budget_pruned = _enforce_word_budget(words, policy["max_words_total"])
    next_word_idx = _next_word_index(words)
    word_vectors: List[Optional[List[float]]] = [_word_vector(word) for word in words]
    component_keys = available_symbol_components(child)
    transformer = FractalTransformer()

    burst_limit = policy["fragment_burst"]
    batch_size = min(policy["batch_size"], burst_limit)
    build_budget_ms = policy["build_budget_ms"]
    ground_fault = get_inastate("ground_sense_fault") or {}
    guard_active = isinstance(ground_fault, dict) and bool(ground_fault.get("active"))
    if guard_active:
        burst_limit = max(4, min(burst_limit, burst_limit // 6 or 4))
        batch_size = min(batch_size, burst_limit)
        if build_budget_ms <= 0:
            build_budget_ms = 120.0
        else:
            build_budget_ms = min(build_budget_ms, 120.0)
        log_to_statusbox("[Symbols] Ground sensor fault active: throttling meaning-map training.")
    cluster_threshold = policy["cluster_threshold"]
    merge_threshold = policy["word_merge_threshold"]
    max_components = policy["max_components_per_word"]
    max_new_words = policy["max_new_words_per_run"]
    max_tags = policy["max_tags_per_word"]
    vector_digits = policy["vector_round_digits"]
    gc_every_batches = policy["gc_every_batches"]

    processed_conflicts: List[Dict[str, Any]] = []
    processed_ids: List[str] = []
    encoded_total = 0
    clusters_total = 0
    merged_words = 0
    created_words = 0
    proto_promoted = 0
    multi_promoted = 0
    batches_run = 0
    budget_hit = False

    while pending_ids and len(processed_ids) < burst_limit:
        elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
        if build_budget_ms > 0 and elapsed_ms >= build_budget_ms and processed_ids:
            budget_hit = True
            break

        take = min(batch_size, burst_limit - len(processed_ids), len(pending_ids))
        batch_ids = pending_ids[:take]
        pending_ids = pending_ids[take:]
        fragments, missing_ids = _load_fragments_by_ids(child, index, batch_ids)
        for frag_id in missing_ids:
            current_signatures.pop(frag_id, None)
            processed_signatures.pop(frag_id, None)
        if not fragments:
            continue

        encoded = transformer.encode_many(fragments)
        encoded = [entry for entry in encoded if isinstance(entry, dict) and entry.get("id") and entry.get("vector")]
        if not encoded:
            continue
        fragments_by_id = {str(frag.get("id")): frag for frag in fragments if frag.get("id")}
        encoded_total += len(encoded)
        clusters = _cluster_encoded(encoded, cluster_threshold)
        clusters_total += len(clusters)

        for cluster in clusters:
            members = cluster.get("members") or []
            centroid = cluster.get("centroid") or []
            if not members or not centroid:
                continue
            component_ids = [str(member.get("id")) for member in members if member.get("id")]
            if not component_ids:
                continue
            component_set = set(component_ids)
            tag_set = {
                str(tag)
                for member in members
                for tag in (member.get("tags") or [])
                if tag
            }
            summary_parts = []
            for member in members[:2]:
                summary = member.get("summary")
                if summary:
                    summary_parts.append(str(summary))
            summary_text = " + ".join(summary_parts) if summary_parts else "symbolic cluster"
            new_count = len(component_ids)
            promoted_proto, promoted_multi = _promote_symbol_pairs(
                preserved,
                members,
                fragments_by_id,
                [float(value) for value in centroid],
                sorted(tag_set),
                summary_text,
                policy,
            )
            proto_promoted += promoted_proto
            multi_promoted += promoted_multi

            best_idx = -1
            best_score = 0.0
            for idx, vec in enumerate(word_vectors):
                if not vec:
                    continue
                score = _cosine_similarity(vec, centroid)
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx >= 0 and best_score >= merge_threshold:
                word = words[best_idx]
                prior_components = [str(fid) for fid in (word.get("components") or []) if fid]
                seen_components = set(prior_components)
                for frag_id in component_ids:
                    if frag_id not in seen_components:
                        prior_components.append(frag_id)
                        seen_components.add(frag_id)
                if len(prior_components) > max_components:
                    prior_components = prior_components[-max_components:]
                word["components"] = prior_components
                existing_tags = {str(tag) for tag in (word.get("tags") or []) if tag}
                existing_tags.update(tag_set)
                merged_tags = sorted(existing_tags)
                if len(merged_tags) > max_tags:
                    merged_tags = merged_tags[-max_tags:]
                word["tags"] = merged_tags
                prior_count = int(_safe_float(word.get("count"), len(prior_components)))
                word["count"] = prior_count + new_count
                prior_vec = word_vectors[best_idx]
                base_vec = prior_vec or list(centroid)
                base_vec_count = prior_count if prior_vec else 0
                merged_vec = _merge_vectors(base_vec, base_vec_count, list(centroid), new_count)
                word["vector"] = _round_vector(merged_vec, vector_digits)
                word_vectors[best_idx] = word["vector"]
                if not word.get("summary"):
                    word["summary"] = summary_text
                word["updated_at"] = datetime.now(timezone.utc).isoformat()
                _apply_word_caps(word, policy)
                merged_words += 1
                continue

            if created_words >= max_new_words:
                continue
            emotion = random.choice(component_keys["emotion"])
            modulation = random.choice(component_keys["modulation"])
            concept = random.choice(component_keys["concept"])
            symbol_id = generate_symbol_from_parts(emotion, modulation, concept, child=child)
            new_word = {
                "symbol_word_id": f"sym_word_{next_word_idx:04}",
                "components": list(component_set)[:max_components],
                "summary": summary_text,
                "tags": sorted(tag_set),
                "count": new_count,
                "birth_time": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol_id,
                "generated_word": "unknown",
                "usage_count": 0,
                "confidence": 0.0,
                "vector": _round_vector([float(value) for value in centroid], vector_digits),
            }
            _apply_word_caps(new_word, policy)
            words.append(new_word)
            word_vectors.append(_word_vector(new_word))
            next_word_idx += 1
            created_words += 1

        for frag in fragments:
            sid = frag.get("sound_symbol")
            sw = frag.get("symbol_word_id")
            if sid and sw:
                processed_conflicts.append({"sound_symbol": sid, "symbol_word_id": sw})
        processed_ids.extend([str(frag.get("id")) for frag in fragments if frag.get("id")])
        batches_run += 1

        dropped_words = _enforce_word_budget(words, policy["max_words_total"])
        if dropped_words:
            words_budget_pruned += dropped_words
            word_vectors = [_word_vector(word) for word in words]

        if gc_every_batches > 0 and (batches_run % gc_every_batches) == 0:
            gc.collect()

    for frag_id in processed_ids:
        signature = current_signatures.get(frag_id)
        if signature:
            processed_signatures[frag_id] = signature

    for word in words:
        _apply_word_caps(word, policy)

    state_payload = {
        "pending": pending_ids[:queue_max],
        "processed_signatures": processed_signatures,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "last_run": {
            "dirty_detected": len(dirty_ids),
            "processed_fragments": len(processed_ids),
            "remaining_queue": len(pending_ids),
            "budget_hit": budget_hit,
            "words_budget_pruned": words_budget_pruned,
        },
    }
    _save_json_dict(_meaning_state_path(child), state_payload)

    payload = {
        "words": words,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    payload.update(preserved)
    out_path = _symbol_words_path(child)
    try:
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=4)
    except Exception as exc:
        log_to_statusbox(f"[Symbols] Failed to save symbol words: {exc}")
        return

    log_to_statusbox(
        f"[Symbols] Processed {len(processed_ids)} symbolic fragment(s), encoded {encoded_total}, "
        f"clusters {clusters_total}, merged {merged_words}, created {created_words}, "
        f"proto_promoted {proto_promoted}, multi_promoted {multi_promoted}, "
        f"word_pruned {words_budget_pruned}, queue {len(pending_ids)}."
    )
    if budget_hit:
        log_to_statusbox("[Symbols] Build budget reached; continuing next cycle.")
    log_to_statusbox("[Symbols] Checking for drift and underuse...")
    evolve_unused_symbols(words)
    if processed_conflicts:
        detect_conflicted_symbols(processed_conflicts, words)
    log_to_statusbox("[Symbols] Completed symbol word update.")

def run_meaning_map():
    try:
        config = load_config()
        child = config.get("current_child", "default_child")
        log_to_statusbox("[Symbols] Meaning map update starting...")
        cluster_symbols_and_generate_words(child, cfg=config)
        build_text_symbol_links(child)
        log_to_statusbox("[Symbols] Meaning map update finished.")
    except Exception as e:
        log_to_statusbox(f"[Symbols] Error: {e}")
        print(f"[Symbols] Error: {e}")

if __name__ == "__main__":
    run_meaning_map()
