# === memory_graph.py (Logging Enhanced) ===

import os
import json
import math
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional, Set, TYPE_CHECKING
from gui_hook import log_to_statusbox

if TYPE_CHECKING:  # pragma: no cover
    from transformers.fractal_multidimensional_transformers import FractalTransformer

MEMORY_TIERS = ["short", "working", "long", "cold"]


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


# === Experience Graph Utilities ===
def _experience_base(child: str, base_path: Optional[Path] = None) -> Path:
    root = Path(base_path) if base_path else Path("AI_Children")
    return root / child / "memory" / "experiences"


def load_experience_events(child: str, base_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load structured events previously logged by the experience logger."""

    events_dir = _experience_base(child, base_path) / "events"
    if not events_dir.exists():
        return []

    events: List[Dict[str, Any]] = []
    for path in sorted(events_dir.glob("evt_*.json")):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                if "id" in data:
                    events.append(data)
        except Exception:
            continue
    return events


def build_experience_graph(child: str, base_path: Optional[Path] = None) -> Dict[str, Any]:
    """Construct a graph over events grounded in shared entities and words."""

    events = load_experience_events(child, base_path)
    if not events:
        return {
            "events": [],
            "edges": [],
            "words_index": {},
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

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


def load_fragments(child):
    base = Path("AI_Children") / child / "memory" / "fragments"
    files = list(base.glob("frag_*.json"))
    def load(f):
        try:
            with open(f, "r", encoding="utf-8") as file:
                data = json.load(file)
                slim = _slim_fragment(data)
                if slim and "emotions" in data:
                    return slim
        except:
            return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        return [f for f in pool.map(load, files) if f]

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
    for i, source in enumerate(neurons):
        for j, target in enumerate(neurons):
            if j <= i:
                continue
            vec_a = source.get("vector")
            vec_b = target.get("vector")
            if vec_a and vec_b:
                sim = cosine_similarity(vec_a, vec_b)
                if sim >= threshold:
                    synapses.append({
                        "source": source["id"],
                        "target": target["id"],
                        "weight": round(sim, 4)
                    })
    return synapses

def build_fractal_memory(child):
    start_time = datetime.now()
    from transformers.fractal_multidimensional_transformers import FractalTransformer

    cluster_threshold, synapse_threshold, tag_weight = _neural_settings()
    transformer = FractalTransformer()
    fragments = load_fragments(child)
    log_to_statusbox(f"[NeuralMap] Loaded {len(fragments)} fragments.")
    encoded = transformer.encode_many(fragments)
    cache = {e["id"]: e["vector"] for e in encoded}
    clusters = cluster_fragments(
        fragments,
        cache,
        threshold=cluster_threshold,
        tag_weight=tag_weight,
    )
    log_to_statusbox(
        f"[NeuralMap] Clustered into {len(clusters)} neurons "
        f"(threshold={cluster_threshold:.2f}, tag_weight={tag_weight:.2f})."
    )

    neurons = []
    for i, group in enumerate(clusters):
        fragment_ids = group["fragments"]
        tags = sorted(group["tags"]) if isinstance(group.get("tags"), set) else list(group.get("tags", []))
        vector_sum = group.get("vector_sum")
        count = group.get("count", len(fragment_ids))
        if vector_sum and count:
            avg_vec = [round(v / count, 6) for v in vector_sum]
        else:
            avg_vec = vector_average([cache[fid] for fid in fragment_ids if fid in cache])
        node_id = f"node_{i:04}"

        neurons.append({
            "id": node_id,
            "fragments": fragment_ids,
            "vector": avg_vec,
            "symbolic_density": 0.0,
            "tags": list(set(tags)),
            "activation_history": [],
            "last_used": datetime.now(timezone.utc).isoformat()
        })

    synapses = build_synaptic_links(neurons, threshold=synapse_threshold)
    result = {
        "neurons": neurons,
        "synapses": synapses,
        "converted_from_legacy": False,
        "updated_at": datetime.now(timezone.utc).isoformat()
    }

    out_path = Path("AI_Children") / child / "memory" / "neural" / "neural_memory_map.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    duration = datetime.now() - start_time
    log_to_statusbox(
        f"[NeuralMap] {len(neurons)} neurons | {len(synapses)} synapses | Saved to neural map."
    )
    log_to_statusbox(f"[NeuralMap] Mapping time: {duration}.")

# === MemoryManager Class ===
class MemoryManager:
    def __init__(self, child="Inazuma_Yagami"):
        self.child = child
        self.base_path = Path("AI_Children") / child / "memory" / "fragments"
        self.index_path = Path("AI_Children") / child / "memory" / "memory_map.json"
        self.memory_map = {}
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

    def reindex_all(self):
        self.index_legacy_root()
        for tier in MEMORY_TIERS:
            self.index_tier(tier)
        self.save_map()
        log_to_statusbox(f"[Memory] Reindexed all fragments across {len(MEMORY_TIERS)} tiers.")
        log_to_statusbox(f"[Memory] Fragment count: {len(self.memory_map)}")

    def reindex(self, new_only=True):
        added = 0
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
