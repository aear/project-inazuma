import json
import math
import time
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from transformers.fractal_multidimensional_transformers import FractalTransformer
from model_manager import load_config, seed_self_question, get_inastate, update_inastate
from gui_hook import log_to_statusbox
from symbol_generator import generate_symbol_from_parts, available_symbol_components
import random
from text_memory import build_text_symbol_links

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
            target["members"].append(enc)
            target["centroid"] = _merge_vectors(centroid, count, vec, 1)
            target["count"] = count + 1
        else:
            clusters.append({"members": [enc], "centroid": [float(v) for v in vec], "count": 1})
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
    next_word_idx = _next_word_index(words)
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

    processed_fragments: List[Dict[str, Any]] = []
    processed_ids: List[str] = []
    encoded_total = 0
    clusters_total = 0
    merged_words = 0
    created_words = 0
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

            best_idx = -1
            best_score = 0.0
            for idx, word in enumerate(words):
                vec = _word_vector(word)
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
                word["tags"] = sorted(existing_tags)
                prior_count = int(_safe_float(word.get("count"), len(prior_components)))
                word["count"] = prior_count + new_count
                base_vec = _word_vector(word) or list(centroid)
                base_vec_count = prior_count if _word_vector(word) else 0
                word["vector"] = _merge_vectors(base_vec, base_vec_count, list(centroid), new_count)
                if not word.get("summary"):
                    word["summary"] = summary_text
                word["updated_at"] = datetime.now(timezone.utc).isoformat()
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
                "vector": [round(float(value), 6) for value in centroid],
            }
            words.append(new_word)
            next_word_idx += 1
            created_words += 1

        processed_fragments.extend(fragments)
        processed_ids.extend([str(frag.get("id")) for frag in fragments if frag.get("id")])

    for frag_id in processed_ids:
        signature = current_signatures.get(frag_id)
        if signature:
            processed_signatures[frag_id] = signature

    state_payload = {
        "pending": pending_ids[:queue_max],
        "processed_signatures": processed_signatures,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "last_run": {
            "dirty_detected": len(dirty_ids),
            "processed_fragments": len(processed_ids),
            "remaining_queue": len(pending_ids),
            "budget_hit": budget_hit,
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
        f"clusters {clusters_total}, merged {merged_words}, created {created_words}, queue {len(pending_ids)}."
    )
    if budget_hit:
        log_to_statusbox("[Symbols] Build budget reached; continuing next cycle.")
    log_to_statusbox("[Symbols] Checking for drift and underuse...")
    evolve_unused_symbols(words)
    if processed_fragments:
        detect_conflicted_symbols(processed_fragments, words)
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
