# === memory_graph.py (Logging Enhanced) ===

import os
import json
import math
import random
import heapq
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional, Set, Tuple, TYPE_CHECKING
from gui_hook import log_to_statusbox
from body_schema import get_region_anchors

if TYPE_CHECKING:  # pragma: no cover
    from transformers.fractal_multidimensional_transformers import FractalTransformer

MEMORY_TIERS = ["short", "working", "long", "cold"]

NEURAL_MAP_BURST_DEFAULT = 60
EXPERIENCE_GRAPH_BURST_DEFAULT = 200

DEFAULT_INCREMENTAL_POLICY = {
    "mode": "incremental",
    "fragment_batch": None,
    "position_blend": 0.25,
    "merge_slack": 0.03,
    "max_new_neurons": 120,
    "synapse_refresh_on_idle": True,
}

DEFAULT_TIER_POLICY = {
    "short": {"max_age_hours": 18.0, "target_count": 5000},
    "working": {"max_age_hours": 72.0, "target_count": 12000},
    "long": {"max_age_hours": 24.0 * 30.0, "target_count": 40000},
    "cold": {},
}


def _load_config():
    path = Path("config.json")
    if path.exists():
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _neural_settings():
    cfg = _load_config()
    cluster = float(cfg.get("neural_cluster_threshold", 0.88))
    synapse = float(cfg.get("neural_synapse_threshold", 0.84))
    tag_weight = float(cfg.get("neural_tag_weight", 0.25))
    tag_weight = max(0.0, min(1.0, tag_weight))
    return cluster, synapse, tag_weight


def _neural_policy(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if cfg is None:
        cfg = _load_config()
    raw = {}
    if isinstance(cfg, dict):
        raw = cfg.get("neural_map_policy", {}) or {}
    if not isinstance(raw, dict):
        raw = {}
    policy = DEFAULT_INCREMENTAL_POLICY.copy()
    policy.update({k: raw.get(k, policy[k]) for k in policy.keys() if k in raw})
    mode = str(policy.get("mode", "incremental")).lower().strip()
    incremental = mode not in {"rebuild", "overwrite", "legacy", "full"}
    fragment_batch = policy.get("fragment_batch")
    try:
        fragment_batch = int(fragment_batch) if fragment_batch is not None else None
    except (TypeError, ValueError):
        fragment_batch = None
    def _clamp(val: float, lo: float, hi: float) -> float:
        try:
            return max(lo, min(float(val), hi))
        except (TypeError, ValueError):
            return lo
    position_blend = _clamp(policy.get("position_blend", 0.25), 0.0, 1.0)
    merge_slack = _clamp(policy.get("merge_slack", 0.03), 0.0, 0.25)
    try:
        max_new = int(policy.get("max_new_neurons", 120))
        if max_new < 0:
            max_new = 0
    except (TypeError, ValueError):
        max_new = 0
    synapse_refresh = bool(policy.get("synapse_refresh_on_idle", True))
    return {
        "mode": mode,
        "incremental": incremental,
        "fragment_batch": fragment_batch,
        "position_blend": position_blend,
        "merge_slack": merge_slack,
        "max_new_neurons": max_new,
        "synapse_refresh_on_idle": synapse_refresh,
    }


def _memory_policy():
    """
    Pull tier policy (age caps + target counts) from config.json when present.
    Falls back to defaults tuned for keeping short-term lean.
    """
    cfg = _load_config()
    user_policy = cfg.get("memory_policy", {}) if isinstance(cfg, dict) else {}
    policy = {}

    def _coerce_positive_int(value: Any) -> Optional[int]:
        try:
            ivalue = int(value)
        except (TypeError, ValueError):
            return None
        return ivalue if ivalue > 0 else None

    def _coerce_positive_float(value: Any) -> Optional[float]:
        try:
            fvalue = float(value)
        except (TypeError, ValueError):
            return None
        return fvalue if fvalue > 0 else None

    for tier in MEMORY_TIERS:
        tier_defaults = DEFAULT_TIER_POLICY.get(tier, {}).copy()
        overrides = user_policy.get(tier, {}) if isinstance(user_policy, dict) else {}
        if isinstance(overrides, dict):
            age_override = _coerce_positive_float(overrides.get("max_age_hours"))
            if age_override is not None:
                tier_defaults["max_age_hours"] = age_override
            target_override = _coerce_positive_int(overrides.get("target_count"))
            if target_override is not None:
                tier_defaults["target_count"] = target_override
        policy[tier] = tier_defaults
    return policy


# === Spatial helpers (body schema → neural positions) ===
def _load_body_anchors() -> Dict[str, Dict[str, float]]:
    anchors = get_region_anchors()
    if not anchors:
        return {}
    return anchors


def _guess_region_from_tags(tags: List[str], anchors: Dict[str, Dict[str, float]]) -> str:
    tagset = {str(t).lower() for t in (tags or [])}

    if {"audio", "sound", "voice", "hearing"} & tagset:
        return "head" if "head" in anchors else next(iter(anchors.keys()), "head")
    if {"vision", "image", "video", "sight"} & tagset:
        return "head" if "head" in anchors else next(iter(anchors.keys()), "head")
    if {"emotion", "feeling", "heart"} & tagset:
        return "chest" if "chest" in anchors else next(iter(anchors.keys()), "chest")
    if {"core", "energy", "stomach", "gut"} & tagset:
        return "core" if "core" in anchors else next(iter(anchors.keys()), "core")
    if "left_arm" in tagset and "left_arm" in anchors:
        return "left_arm"
    if "right_arm" in tagset and "right_arm" in anchors:
        return "right_arm"
    if "left_leg" in tagset and "left_leg" in anchors:
        return "left_leg"
    if "right_leg" in tagset and "right_leg" in anchors:
        return "right_leg"

    # Default to head or first available anchor
    if "head" in anchors:
        return "head"
    return next(iter(anchors.keys()), "head")


def _project_vector_to_anchor(vector: List[float], anchor: Dict[str, float], seed: str) -> List[float]:
    """
    Map a latent vector into body space using the region's anchor.
    Keeps placement stable via a hash-based RNG when vectors are missing.
    """
    center = anchor.get("center", [0.0, 0.0, 0.0])
    radius = float(anchor.get("radius", 1.0) or 1.0)

    rng = random.Random(hash(seed) & 0xFFFFFFFF)
    if vector and len(vector) >= 3:
        base = [float(v) for v in vector[:3]]
        norm = math.sqrt(sum(v * v for v in base)) or 1e-6
        unit = [v / norm for v in base]
    else:
        theta = rng.uniform(0, 2 * math.pi)
        phi = rng.uniform(0, math.pi)
        unit = [
            math.sin(phi) * math.cos(theta),
            math.sin(phi) * math.sin(theta),
            math.cos(phi),
        ]

    r = radius * (0.35 + 0.6 * rng.random())
    return [center[i] + unit[i] * r for i in range(3)]


# === Experience Graph Utilities ===
def _experience_base(child: str, base_path: Optional[Path] = None) -> Path:
    root = Path(base_path) if base_path else Path("AI_Children")
    return root / child / "memory" / "experiences"


def load_experience_events(child: str, base_path: Optional[Path] = None, limit: Optional[int] = None) -> Tuple[List[Dict[str, Any]], int]:
    """Load structured events previously logged by the experience logger (optionally limited)."""

    events_dir = _experience_base(child, base_path) / "events"
    if not events_dir.exists():
        return [], 0

    limit_val = 0
    if limit is not None:
        try:
            limit_val = max(0, int(limit))
        except (TypeError, ValueError):
            limit_val = 0

    def _ts_value(payload: Dict[str, Any], path: Path) -> float:
        raw = payload.get("timestamp") or ""
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
        except Exception:
            try:
                return path.stat().st_mtime
            except Exception:
                return 0.0

    events: List[Dict[str, Any]] = []
    total = 0

    if limit_val > 0:
        heap: List[tuple] = []
        for path in sorted(events_dir.glob("evt_*.json")):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception:
                continue
            if "id" not in data:
                continue
            total += 1
            entry = (_ts_value(data, path), data)
            if len(heap) < limit_val:
                heapq.heappush(heap, entry)
            else:
                if entry[0] > heap[0][0]:
                    heapq.heapreplace(heap, entry)
        heap.sort()
        events = [item[1] for item in heap]
    else:
        for path in sorted(events_dir.glob("evt_*.json")):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception:
                continue
            if "id" not in data:
                continue
            events.append(data)
        total = len(events)

    return events, total


def build_experience_graph(child: str, base_path: Optional[Path] = None) -> Dict[str, Any]:
    """Construct a graph over events grounded in shared entities and words."""

    cfg = _load_config()
    burst_limit = cfg.get("experience_graph_burst")
    try:
        burst_limit = int(burst_limit)
    except (TypeError, ValueError):
        burst_limit = EXPERIENCE_GRAPH_BURST_DEFAULT
    if burst_limit <= 0:
        burst_limit = EXPERIENCE_GRAPH_BURST_DEFAULT

    events, total_events = load_experience_events(
        child,
        base_path=base_path,
        limit=burst_limit,
    )
    if not events:
        return {
            "events": [],
            "edges": [],
            "words_index": {},
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    if total_events > len(events):
        log_to_statusbox(
            f"[ExperienceGraph] Limiting to {len(events)} most recent events (burst={burst_limit}, total={total_events})."
        )

    nodes: List[Dict[str, Any]] = []
    words_index: Dict[str, Set[str]] = {}

    for raw in events:
        entity_labels = {
            entity.get("name") or entity.get("label")
            for entity in raw.get("perceived_entities", [])
            if entity.get("name") or entity.get("label")
        }
        node = {
            "id": raw.get("id"),
            "timestamp": raw.get("timestamp"),
            "situation_tags": raw.get("situation_tags", []),
            "entities": sorted(entity_labels),
            "episode_id": raw.get("episode_id"),
            "narrative": raw.get("narrative", ""),
            "word_usage": raw.get("word_usage", []),
        }
        nodes.append(node)
        for usage in node["word_usage"]:
            for word in usage.get("words", []):
                if not word:
                    continue
                words_index.setdefault(word.lower(), set()).add(node["id"])

    edges: List[Dict[str, Any]] = []
    for i, left in enumerate(nodes):
        left_tags = set(left.get("situation_tags", []))
        left_entities = set(left.get("entities", []))
        for right in nodes[i + 1 :]:
            shared_tags = left_tags.intersection(right.get("situation_tags", []))
            shared_entities = left_entities.intersection(right.get("entities", []))
            if not shared_tags and not shared_entities:
                continue
            edges.append(
                {
                    "source": left["id"],
                    "target": right["id"],
                    "shared_tags": sorted(shared_tags),
                    "shared_entities": sorted(shared_entities),
                }
            )

    graph = {
        "events": nodes,
        "edges": edges,
        "words_index": {word: sorted(list(event_ids)) for word, event_ids in words_index.items()},
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    out_path = _experience_base(child, base_path) / "experience_graph.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(graph, fh, indent=2)

    log_to_statusbox(
        f"[ExperienceGraph] {len(nodes)} events | {len(edges)} edges | {len(words_index)} grounded words."
    )
    return graph

# === Core Utilities ===
def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    return dot / (norm1 * norm2 + 1e-8)

def vector_average(vectors):
    if not vectors:
        return []
    length = len(vectors[0])
    avg = [0.0] * length
    for vec in vectors:
        for i in range(length):
            avg[i] += vec[i]
    return [round(x / len(vectors), 6) for x in avg]


def tag_similarity(tags_a, tags_b):
    a = set(tags_a or [])
    b = set(tags_b or [])
    union = a.union(b)
    if not union:
        return 0.0
    return len(a.intersection(b)) / len(union)


def _slim_fragment(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Keep only the fields needed for encoding/clustering to reduce memory pressure.
    """
    if not data or "id" not in data:
        return None
    keep_keys = {
        "id",
        "tags",
        "emotions",
        "summary",
        "tier",
        "modality",
        "audio_features",
        "image_features",
        "video_features",
        "timestamp",
        "source",
    }
    slim = {k: data.get(k) for k in keep_keys if k in data}
    # Preserve modality hints embedded in tags (e.g., "audio" / "vision") for clustering.
    return slim


def _iter_fragment_files(base: Path):
    for path in base.glob("frag_*.json"):
        yield path
    for tier in MEMORY_TIERS:
        tier_path = base / tier
        if tier_path.exists():
            for path in tier_path.glob("frag_*.json"):
                yield path


def load_fragments(child, limit: Optional[int] = None) -> Tuple[List[Dict[str, Any]], int]:
    base = Path("AI_Children") / child / "memory" / "fragments"
    limit_val = 0
    if limit is not None:
        try:
            limit_val = max(0, int(limit))
        except (TypeError, ValueError):
            limit_val = 0

    selected_paths: List[Path] = []
    total = 0
    if limit_val > 0:
        heap: List[tuple] = []
        for path in _iter_fragment_files(base):
            total += 1
            try:
                mtime = path.stat().st_mtime
            except Exception:
                continue
            entry = (mtime, path)
            if len(heap) < limit_val:
                heapq.heappush(heap, entry)
            else:
                if entry[0] > heap[0][0]:
                    heapq.heapreplace(heap, entry)
        heap.sort()
        selected_paths = [item[1] for item in heap]
    else:
        selected_paths = list(_iter_fragment_files(base))
        total = len(selected_paths)

    seen: Set[str] = set()

    def load(f: Path):
        try:
            with open(f, "r", encoding="utf-8") as file:
                data = json.load(file)
                if "tier" not in data:
                    parent_tier = f.parent.name
                    if parent_tier in MEMORY_TIERS:
                        data["tier"] = parent_tier
                slim = _slim_fragment(data)
                if slim and "emotions" in data:
                    frag_id = slim.get("id")
                    if frag_id and frag_id not in seen:
                        seen.add(frag_id)
                        return slim
        except Exception:
            return None
        return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        fragments = [frag for frag in pool.map(load, selected_paths) if frag]
    return fragments, total

def cluster_fragments(fragments, cache, threshold=0.92, tag_weight=0.25):
    clusters = []
    tag_weight = max(0.0, min(1.0, tag_weight))
    for frag in fragments:
        frag_id = frag["id"]
        vec = cache.get(frag_id)
        if vec is None:
            continue

        frag_tags = set(frag.get("tags", []))
        best = None
        best_score = 0.0

        for node in clusters:
            node_vec = [
                val / node["count"] for val in node["vector_sum"]
            ]
            vec_score = cosine_similarity(vec, node_vec)
            tag_score = tag_similarity(frag_tags, node["tags"])
            score = ((1 - tag_weight) * vec_score) + (tag_weight * tag_score)
            if score >= threshold and score > best_score:
                best_score = score
                best = node

        if best:
            best["fragments"].append(frag_id)
            best["tags"].update(frag_tags)
            best["count"] += 1
            best["vector_sum"] = [
                a + b for a, b in zip(best["vector_sum"], vec)
            ]
        else:
            clusters.append({
                "fragments": [frag_id],
                "tags": set(frag_tags),
                "vector_sum": list(vec),
                "count": 1
            })
    return clusters

def build_synaptic_links(neurons, threshold=0.91):
    synapses = []
    position_map = {n["id"]: n.get("position") for n in neurons}

    for i, source in enumerate(neurons):
        for j, target in enumerate(neurons):
            if j <= i:
                continue
            vec_a = source.get("vector")
            vec_b = target.get("vector")
            if vec_a and vec_b:
                sim = cosine_similarity(vec_a, vec_b)
                if sim >= threshold:
                    direction = None
                    pos_a = position_map.get(source["id"])
                    pos_b = position_map.get(target["id"])
                    if pos_a and pos_b:
                        delta = [pos_b[k] - pos_a[k] for k in range(3)]
                        norm = math.sqrt(sum(d * d for d in delta))
                        if norm > 1e-6:
                            direction = [d / norm for d in delta]
                    synapses.append({
                        "source": source["id"],
                        "target": target["id"],
                        "weight": round(sim, 4),
                        "direction": direction,
                        "network_type": "memory_graph",
                    })
    return synapses


def _neural_map_path(child: str) -> Path:
    return Path("AI_Children") / child / "memory" / "neural" / "neural_memory_map.json"


def _load_neural_map(child: str) -> Dict[str, Any]:
    path = _neural_map_path(child)
    if not path.exists():
        return {"neurons": [], "synapses": [], "converted_from_legacy": False, "updated_at": None}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return {"neurons": [], "synapses": [], "converted_from_legacy": False, "updated_at": None}
    if "neurons" not in data or "synapses" not in data:
        data.setdefault("neurons", [])
        data.setdefault("synapses", [])
    return data


def _save_neural_map(child: str, payload: Dict[str, Any]) -> None:
    path = _neural_map_path(child)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=4)


def _existing_fragment_ids(neurons: List[Dict[str, Any]]) -> Set[str]:
    known: Set[str] = set()
    for neuron in neurons:
        for frag_id in neuron.get("fragments", []):
            known.add(frag_id)
    return known


def _node_id_allocator(neurons: List[Dict[str, Any]]):
    prefix = "node_"
    max_idx = -1
    for neuron in neurons:
        node_id = str(neuron.get("id") or "")
        if node_id.startswith(prefix):
            try:
                idx = int(node_id[len(prefix):])
                max_idx = max(max_idx, idx)
            except ValueError:
                continue
    counter = max_idx + 1
    while True:
        yield f"{prefix}{counter:04}"
        counter += 1


def _blend_position(old: Optional[List[float]], new: Optional[List[float]], blend: float) -> Optional[List[float]]:
    if not new and not old:
        return None
    if not old:
        return list(new)
    if not new:
        return list(old)
    blend = max(0.0, min(1.0, blend))
    return [
        old[i] + (new[i] - old[i]) * blend for i in range(min(len(old), len(new)))
    ]


def _merge_vectors(base: List[float], base_count: int, new_vec: List[float], new_count: int) -> List[float]:
    if not base_count:
        return [round(v, 6) for v in new_vec]
    if not new_count:
        return [round(v, 6) for v in base]
    length = min(len(base), len(new_vec))
    merged = []
    total = base_count + new_count
    for i in range(length):
        merged.append(round((base[i] * base_count + new_vec[i] * new_count) / total, 6))
    return merged


def _materialize_candidate(node_id: str, candidate: Dict[str, Any]) -> Dict[str, Any]:
    now_iso = datetime.now(timezone.utc).isoformat()
    return {
        "id": node_id,
        "fragments": list(candidate["fragments"]),
        "vector": list(candidate["vector"]),
        "position": list(candidate["position"]),
        "region": candidate["region"],
        "network_type": "memory_graph",
        "symbolic_density": 0.0,
        "tags": list(candidate["tags"]),
        "activation_history": [],
        "last_used": now_iso,
    }


def _update_neuron_from_candidate(neuron: Dict[str, Any], candidate: Dict[str, Any], policy: Dict[str, Any]) -> None:
    existing_frags = neuron.setdefault("fragments", [])
    prior_count = len(existing_frags)
    new_ids = [fid for fid in candidate["fragments"] if fid not in existing_frags]
    if new_ids:
        existing_frags.extend(new_ids)
    base_vec = neuron.get("vector")
    base_count = prior_count if base_vec else 0
    base_vec = base_vec or candidate["vector"]
    new_count = max(len(new_ids), 1)
    combined_vec = _merge_vectors(
        base_vec,
        base_count,
        candidate["vector"],
        new_count,
    )
    neuron["vector"] = combined_vec
    neuron["position"] = _blend_position(neuron.get("position"), candidate["position"], policy["position_blend"]) or candidate["position"]
    tag_union = set(neuron.get("tags", []))
    tag_union.update(candidate["tags"])
    neuron["tags"] = sorted(tag_union)
    if not neuron.get("region"):
        neuron["region"] = candidate["region"]
    neuron["last_used"] = datetime.now(timezone.utc).isoformat()


def _score_candidate_match(neuron: Dict[str, Any], candidate: Dict[str, Any], tag_weight: float) -> float:
    vec_a = neuron.get("vector")
    vec_b = candidate["vector"]
    if not vec_a or not vec_b:
        return 0.0
    vec_score = cosine_similarity(vec_a, vec_b)
    tag_score = tag_similarity(neuron.get("tags", []), candidate["tags"])
    tag_weight = max(0.0, min(1.0, tag_weight))
    return ((1 - tag_weight) * vec_score) + (tag_weight * tag_score)


def _prepare_candidates(clusters: List[Dict[str, Any]], cache: Dict[str, List[float]], anchors: Dict[str, Dict[str, float]], fallback_anchor: Dict[str, Any]):
    candidates: List[Dict[str, Any]] = []
    for group in clusters:
        fragment_ids = group["fragments"]
        if not fragment_ids:
            continue
        vector_sum = group.get("vector_sum")
        count = group.get("count", len(fragment_ids))
        if vector_sum and count:
            avg_vec = [round(v / count, 6) for v in vector_sum]
        else:
            avg_vec = vector_average([cache[fid] for fid in fragment_ids if fid in cache])
        if not avg_vec:
            continue
        tags = sorted(group["tags"]) if isinstance(group.get("tags"), set) else list(group.get("tags", []))
        region = _guess_region_from_tags(tags, anchors)
        anchor = anchors.get(region, fallback_anchor)
        position = _project_vector_to_anchor(avg_vec, anchor, seed=fragment_ids[0])
        candidates.append({
            "fragments": fragment_ids,
            "tags": tags,
            "vector": avg_vec,
            "region": region,
            "position": position,
        })
    return candidates


def _merge_candidates_into_neurons(
    neurons: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    cluster_threshold: float,
    tag_weight: float,
    policy: Dict[str, Any],
) -> Tuple[int, int, int]:
    if not candidates:
        return 0, 0, 0
    merge_threshold = max(0.0, min(1.0, cluster_threshold - policy.get("merge_slack", 0.0)))
    merged = 0
    created = 0
    skipped = 0
    id_allocator = _node_id_allocator(neurons)
    for candidate in candidates:
        best = None
        best_score = 0.0
        for neuron in neurons:
            score = _score_candidate_match(neuron, candidate, tag_weight)
            if score > best_score:
                best_score = score
                best = neuron
        if best and best_score >= merge_threshold:
            _update_neuron_from_candidate(best, candidate, policy)
            merged += 1
            continue
        if created < policy.get("max_new_neurons", 0):
            node_id = next(id_allocator)
            neurons.append(_materialize_candidate(node_id, candidate))
            created += 1
            continue
        skipped += 1
    return merged, created, skipped

def build_fractal_memory(child):
    start_time = datetime.now()
    from transformers.fractal_multidimensional_transformers import FractalTransformer

    cluster_threshold, synapse_threshold, tag_weight = _neural_settings()
    cfg = _load_config()
    policy = _neural_policy(cfg)
    burst_limit = policy.get("fragment_batch")
    if burst_limit is None:
        burst_limit = cfg.get("neural_map_burst")
    try:
        burst_limit = int(burst_limit) if burst_limit is not None else NEURAL_MAP_BURST_DEFAULT
    except (TypeError, ValueError):
        burst_limit = NEURAL_MAP_BURST_DEFAULT
    if burst_limit <= 0:
        burst_limit = NEURAL_MAP_BURST_DEFAULT

    transformer = FractalTransformer()
    incremental = policy.get("incremental", True)
    existing_map = _load_neural_map(child) if incremental else {"neurons": [], "synapses": [], "converted_from_legacy": False}
    neurons = existing_map.get("neurons", []) if incremental else []
    known_fragments = _existing_fragment_ids(neurons) if incremental else set()

    fragments, total_count = load_fragments(child, limit=burst_limit)
    if not fragments:
        log_to_statusbox("[NeuralMap] No fragments available for neural map build.")
        return
    if total_count > len(fragments):
        log_to_statusbox(
            f"[NeuralMap] Limiting to {len(fragments)} recent fragments (burst={burst_limit}, total={total_count})."
        )
    else:
        log_to_statusbox(f"[NeuralMap] Loaded {len(fragments)} fragments.")

    target_fragments = [
        frag for frag in fragments if not incremental or frag.get("id") not in known_fragments
    ]
    if not target_fragments:
        if incremental and policy.get("synapse_refresh_on_idle", True):
            synapses = build_synaptic_links(neurons, threshold=synapse_threshold)
            existing_map["synapses"] = synapses
            existing_map["updated_at"] = datetime.now(timezone.utc).isoformat()
            _save_neural_map(child, existing_map)
            log_to_statusbox("[NeuralMap] No new fragments; refreshed synapses to keep map current.")
        else:
            log_to_statusbox("[NeuralMap] No new fragments detected; skipping rebuild.")
        return

    encoded = transformer.encode_many(target_fragments)
    if not encoded:
        log_to_statusbox("[NeuralMap] Encoder returned no vectors for the selected fragments.")
        return

    cache = {e["id"]: e["vector"] for e in encoded}
    clusters = cluster_fragments(
        target_fragments,
        cache,
        threshold=cluster_threshold,
        tag_weight=tag_weight,
    )
    log_to_statusbox(
        f"[NeuralMap] Clustered {len(target_fragments)} new fragments into {len(clusters)} nodes "
        f"(threshold={cluster_threshold:.2f}, tag_weight={tag_weight:.2f})."
    )

    anchors = _load_body_anchors()
    fallback_anchor = anchors.get("head", {"center": [0.0, 0.0, 0.0], "radius": 2.0})
    candidates = _prepare_candidates(clusters, cache, anchors, fallback_anchor)

    if incremental:
        merged, created, skipped = _merge_candidates_into_neurons(
            neurons,
            candidates,
            cluster_threshold,
            tag_weight,
            policy,
        )
        log_to_statusbox(
            f"[NeuralMap] Incremental update — merged {merged}, added {created}, skipped {skipped} (max_new={policy.get('max_new_neurons')})."
        )
    else:
        neurons = []
        id_allocator = _node_id_allocator(neurons)
        for candidate in candidates:
            neurons.append(_materialize_candidate(next(id_allocator), candidate))
        merged = len(candidates)
        created = len(candidates)
        skipped = 0

    needs_synapse_refresh = not incremental or merged > 0 or created > 0 or policy.get("synapse_refresh_on_idle", True)
    synapses = build_synaptic_links(neurons, threshold=synapse_threshold) if needs_synapse_refresh else existing_map.get("synapses", [])

    result = existing_map if incremental else {}
    result.update({
        "neurons": neurons,
        "synapses": synapses,
        "converted_from_legacy": existing_map.get("converted_from_legacy", False) if incremental else False,
        "updated_at": datetime.now(timezone.utc).isoformat()
    })
    _save_neural_map(child, result)

    duration = datetime.now() - start_time
    log_to_statusbox(
        f"[NeuralMap] {len(neurons)} neurons | {len(synapses)} synapses | Policy={policy.get('mode')} | Saved."
    )
    log_to_statusbox(f"[NeuralMap] Mapping time: {duration}.")

# === MemoryManager Class ===
class MemoryManager:
    def __init__(self, child="Inazuma_Yagami", tier_policy: Optional[Dict[str, Any]] = None):
        self.child = child
        self.base_path = Path("AI_Children") / child / "memory" / "fragments"
        self.index_path = Path("AI_Children") / child / "memory" / "memory_map.json"
        self.memory_map = {}
        self.policy = tier_policy or _memory_policy()
        self.load_map()

    def ensure_tier_directories(self):
        for tier in MEMORY_TIERS:
            (self.base_path / tier).mkdir(parents=True, exist_ok=True)

    def index_legacy_root(self):
        """
        Legacy fragments lived directly under memory/fragments without tier folders.
        Scan them so counts are accurate even if tiers are unused.
        """
        root_files = list(self.base_path.glob("frag_*.json"))
        for frag in root_files:
            try:
                with open(frag, "r") as f:
                    data = json.load(f)
                existing = self.memory_map.get(data["id"], {})
                self.memory_map[data["id"]] = {
                    "tier": existing.get("tier", "short"),
                    "tags": data.get("tags", []),
                    "importance": data.get("importance", 0),
                    "last_seen": existing.get(
                        "last_seen",
                        data.get("timestamp", datetime.now(timezone.utc).isoformat())
                    ),
                    "filename": frag.name,
                }
            except Exception:
                continue

    def load_map(self):
        if self.index_path.exists():
            try:
                with open(self.index_path, "r") as f:
                    self.memory_map = json.load(f)
            except:
                self.memory_map = {}
        else:
            self.memory_map = {}

    def save_map(self):
        with open(self.index_path, "w") as f:
            json.dump(self.memory_map, f, indent=2)

    def _resolve_fragment_path(self, frag_id: str, meta: Dict[str, Any]) -> Optional[Path]:
        """
        Resolve the on-disk path for a fragment given its metadata entry.
        """
        filename = meta.get("filename", f"frag_{frag_id}.json")
        tier = meta.get("tier")
        candidates = []
        if tier:
            candidates.append(self.base_path / tier / filename)
        candidates.append(self.base_path / filename)
        for path in candidates:
            if path.exists():
                return path
        return None

    @staticmethod
    def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            ts = datetime.fromisoformat(value)
        except Exception:
            try:
                ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except Exception:
                return None
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts

    def _tier_for_age(self, timestamp: Optional[datetime], now: datetime) -> str:
        """
        Decide target tier based on age cutoffs. Defaults to 'short' when unknown.
        """
        policy = self.policy or DEFAULT_TIER_POLICY
        age_hours: Optional[float] = None
        if timestamp:
            age_hours = (now - timestamp).total_seconds() / 3600.0

        short_cap = policy.get("short", {}).get("max_age_hours")
        working_cap = policy.get("working", {}).get("max_age_hours")
        long_cap = policy.get("long", {}).get("max_age_hours")

        if age_hours is None or short_cap is None:
            return "short"
        if age_hours <= short_cap:
            return "short"
        if working_cap is not None and age_hours <= working_cap:
            return "working"
        if long_cap is not None and age_hours <= long_cap:
            return "long"
        return "cold"

    @staticmethod
    def _timestamp_sort_key(ts: Optional[datetime]) -> datetime:
        """
        Provide a stable sort key, pushing unknown timestamps to the oldest end.
        """
        if ts is None:
            return datetime.fromtimestamp(0, tz=timezone.utc)
        return ts

    def index_tier(self, tier="short"):
        tier_path = self.base_path / tier
        if not tier_path.exists():
            return
        for frag in tier_path.glob("frag_*.json"):
            try:
                with open(frag, "r") as f:
                    data = json.load(f)
                existing = self.memory_map.get(data["id"], {})
                self.memory_map[data["id"]] = {
                    "tier": tier,
                    "tags": data.get("tags", []),
                    "importance": data.get("importance", 0),
                    "last_seen": existing.get(
                        "last_seen",
                        data.get("timestamp", datetime.now(timezone.utc).isoformat())
                    ),
                    "filename": frag.name,
                }
            except:
                continue

    def reindex_all(self, rebalance: bool = True):
        """
        Full reindex of all fragments. Optionally rebalance tiers afterward.
        """
        self.ensure_tier_directories()
        self.index_legacy_root()
        for tier in MEMORY_TIERS:
            self.index_tier(tier)
        self.save_map()
        log_to_statusbox(f"[Memory] Reindexed all fragments across {len(MEMORY_TIERS)} tiers.")
        log_to_statusbox(f"[Memory] Fragment count: {len(self.memory_map)}")
        if rebalance:
            self.rebalance_tiers()

    def reindex(self, new_only=True):
        added = 0
        self.ensure_tier_directories()
        self.index_legacy_root()
        for tier in MEMORY_TIERS:
            tier_path = self.base_path / tier
            if not tier_path.exists():
                continue
            for frag in tier_path.glob("frag_*.json"):
                try:
                    with open(frag, "r") as f:
                        data = json.load(f)
                    if not new_only or data["id"] not in self.memory_map:
                        self.memory_map[data["id"]] = {
                            "tier": tier,
                            "tags": data.get("tags", []),
                            "importance": data.get("importance", 0),
                            "last_seen": data.get(
                                "timestamp",
                                datetime.now(timezone.utc).isoformat()
                            ),
                            "filename": frag.name,
                        }
                        added += 1
                    else:
                        existing = self.memory_map.get(data["id"], {})
                        existing.update({
                            "tier": tier,
                            "tags": data.get("tags", []),
                            "importance": data.get("importance", 0),
                            "filename": frag.name,
                        })
                        self.memory_map[data["id"]] = existing
                except:
                    continue
        self.save_map()
        log_to_statusbox(f"[Memory] Reindexed {added} new fragments across tiers. Current total: {len(self.memory_map)}")

    def ingest_fragment_file(self, fragment_path, to_tier):
        try:
            with open(fragment_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return False

        frag_id = data.get("id")
        if not frag_id:
            return False

        destination = self.base_path / to_tier
        destination.mkdir(parents=True, exist_ok=True)
        target_path = destination / fragment_path.name

        try:
            fragment_path.rename(target_path)
        except FileNotFoundError:
            return False

        self.memory_map[frag_id] = {
            "tier": to_tier,
            "tags": data.get("tags", []),
            "importance": data.get("importance", 0),
            "last_seen": data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            "filename": target_path.name,
        }
        self.save_map()
        return True

    def prune_missing(self):
        removed = []
        for frag_id, meta in list(self.memory_map.items()):
            tier = meta.get("tier")
            filename = meta.get("filename", f"frag_{frag_id}.json")
            candidate_paths = []
            if tier:
                candidate_paths.append(self.base_path / tier / filename)
            candidate_paths.append(self.base_path / filename)
            if not any(path.exists() for path in candidate_paths):
                removed.append(frag_id)
                self.memory_map.pop(frag_id, None)
        if removed:
            self.save_map()
            log_to_statusbox(
                f"[Memory] Pruned {len(removed)} missing fragment entries from index."
            )
        return len(removed)

    def rebalance_tiers(self, now: Optional[datetime] = None):
        """
        Move fragments out of short-term when they age out or exceed target counts.
        """
        if not self.memory_map:
            return {"moved": 0, "missing": 0, "transitions": {}, "counts": {}}

        self.ensure_tier_directories()
        now = now or datetime.now(timezone.utc)
        buckets = {tier: [] for tier in MEMORY_TIERS}
        transitions: Dict[str, int] = {}
        missing = 0

        for frag_id, meta in self.memory_map.items():
            ts = self._parse_timestamp(meta.get("last_seen")) or self._parse_timestamp(meta.get("timestamp"))
            target_tier = self._tier_for_age(ts, now)
            buckets[target_tier].append(
                {
                    "id": frag_id,
                    "current_tier": meta.get("tier") or "short",
                    "timestamp": ts,
                    "path": self._resolve_fragment_path(frag_id, meta),
                    "last_seen": meta.get("last_seen"),
                }
            )

        # Apply target count caps so short/working stay lean.
        for tier, next_tier in [("short", "working"), ("working", "long"), ("long", "cold")]:
            try:
                cap = int(self.policy.get(tier, {}).get("target_count", 0))
            except (TypeError, ValueError):
                cap = 0
            if cap <= 0:
                continue
            bucket = buckets[tier]
            bucket.sort(key=lambda r: self._timestamp_sort_key(r["timestamp"]))
            if len(bucket) > cap:
                overflow = bucket[:-cap]
                buckets[tier] = bucket[-cap:]
                buckets[next_tier].extend(overflow)

        moved = 0
        for target_tier, records in buckets.items():
            for record in records:
                frag_id = record["id"]
                current_tier = record["current_tier"]
                path = record["path"]
                if path is None or not path.exists():
                    missing += 1
                    continue

                dest_dir = self.base_path / target_tier
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_path = dest_dir / path.name

                if current_tier == target_tier and path.parent == dest_dir:
                    continue

                try:
                    path.rename(dest_path)
                except Exception:
                    missing += 1
                    continue

                moved += 1
                transition_key = f"{current_tier}->{target_tier}"
                transitions[transition_key] = transitions.get(transition_key, 0) + 1

                meta = self.memory_map.get(frag_id, {})
                meta["tier"] = target_tier
                meta["filename"] = dest_path.name
                if record["last_seen"]:
                    meta["last_seen"] = record["last_seen"]
                self.memory_map[frag_id] = meta

        if moved or missing:
            self.save_map()

        transition_summary = ", ".join(f"{k}:{v}" for k, v in sorted(transitions.items()))
        if not transition_summary:
            transition_summary = "none"
        log_to_statusbox(
            f"[Memory] Rebalanced tiers: moved {moved} fragment(s); transitions {transition_summary}."
        )
        if missing:
            log_to_statusbox(f"[Memory] Rebalance skipped {missing} missing fragment files.")

        counts = self.stats()
        return {"moved": moved, "missing": missing, "transitions": transitions, "counts": counts}

    def promote(self, frag_id, to_tier, *, touch=True):
        if frag_id not in self.memory_map:
            return False
        old_tier = self.memory_map[frag_id]["tier"]
        if old_tier == to_tier:
            return True
        filename = self.memory_map[frag_id].get("filename", f"frag_{frag_id}.json")
        src = self.base_path / old_tier / filename
        dst = self.base_path / to_tier / filename
        if not src.exists():
            return False
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        self.memory_map[frag_id]["tier"] = to_tier
        self.memory_map[frag_id]["filename"] = dst.name
        if touch:
            self.memory_map[frag_id]["last_seen"] = datetime.now(timezone.utc).isoformat()
        self.save_map()
        return True

    def get_by_tag(self, tag, tier=None):
        results = []
        for fid, meta in self.memory_map.items():
            if tag in meta.get("tags", []) and (tier is None or meta.get("tier") == tier):
                results.append(fid)
        return results

    def stats(self):
        counts = {tier: 0 for tier in MEMORY_TIERS}
        for meta in self.memory_map.values():
            t = meta.get("tier")
            if t in counts:
                counts[t] += 1
        log_to_statusbox(f"[Memory] Stats: {json.dumps(counts)}")
        return counts

if __name__ == "__main__":
    mgr = MemoryManager()
    mgr.reindex_all()
    mgr.stats()
    build_fractal_memory(mgr.child)
    build_experience_graph(mgr.child)
