# === memory_graph.py (Logging Enhanced) ===

import os
import json
import math
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from fractal_multidimensional_transformers import FractalTransformer
from gui_hook import log_to_statusbox

MEMORY_TIERS = ["short", "working", "long", "cold"]

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

def load_fragments(child):
    base = Path("AI_Children") / child / "memory" / "fragments"
    files = list(base.glob("frag_*.json"))
    def load(f):
        try:
            with open(f, "r", encoding="utf-8") as file:
                data = json.load(file)
                if "emotions" in data:
                    return data
        except:
            return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        return [f for f in pool.map(load, files) if f]

def cluster_fragments(fragments, transformer, cache, threshold=0.92):
    clusters = []
    for frag in fragments:
        frag_id = frag["id"]
        vec = cache.get(frag_id)
        matched = False
        for node in clusters:
            node_vec = vector_average([cache[fid] for fid in node["fragments"]])
            score = cosine_similarity(vec, node_vec)
            if score >= threshold:
                node["fragments"].append(frag_id)
                matched = True
                break
        if not matched:
            clusters.append({"fragments": [frag_id]})
    return clusters

def build_synaptic_links(neurons, cache, threshold=0.91):
    synapses = []
    for i, source in enumerate(neurons):
        for j, target in enumerate(neurons):
            if j <= i:
                continue
            vec_a = cache.get(source["id"])
            vec_b = cache.get(target["id"])
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
    transformer = FractalTransformer()
    fragments = load_fragments(child)
    log_to_statusbox(f"[NeuralMap] Loaded {len(fragments)} fragments.")
    encoded = transformer.encode_many(fragments)
    cache = {e["id"]: e["vector"] for e in encoded}
    clusters = cluster_fragments(fragments, transformer, cache)
    log_to_statusbox(f"[NeuralMap] Clustered into {len(clusters)} neurons.")

    neurons = []
    for i, group in enumerate(clusters):
        fragment_ids = group["fragments"]
        tags = [t for fid in fragment_ids for t in next((f["tags"] for f in fragments if f["id"] == fid), [])]
        avg_vec = vector_average([cache[fid] for fid in fragment_ids])
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

    synapses = build_synaptic_links(neurons, cache)
    result = {
        "neurons": neurons,
        "synapses": synapses,
        "converted_from_legacy": False,
        "updated_at": datetime.now(timezone.utc).isoformat()
    }

    out_path = Path.home() / "Projects" / "Project Inazuma" / "AI_Children" / child / "memory" / "neural" / "neural_memory_map.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    duration = datetime.now() - start_time
    log_to_statusbox(f"[NeuralMap] {len(neurons)} neurons | {len(synapses)} synapses | Saved to neural map.")
    log_to_statusbox(f"[NeuralMap] Mapping time: {duration}.")

# === MemoryManager Class ===
class MemoryManager:
    def __init__(self, child="Inazuma_Yagami"):
        self.child = child
        self.base_path = Path("AI_Children") / child / "memory" / "fragments"
        self.index_path = Path("AI_Children") / child / "memory" / "memory_map.json"
        self.memory_map = {}
        self.load_map()

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
                self.memory_map[data["id"]] = {
                    "tier": tier,
                    "tags": data.get("tags", []),
                    "importance": data.get("importance", 0),
                    "last_seen": data.get("timestamp", datetime.now(timezone.utc).isoformat())
                }
            except:
                continue

    def reindex_all(self):
        for tier in MEMORY_TIERS:
            self.index_tier(tier)
        self.save_map()
        log_to_statusbox(f"[Memory] Reindexed all fragments across {len(MEMORY_TIERS)} tiers.")
        log_to_statusbox(f"[Memory] Fragment count: {len(self.memory_map)}")

    def reindex(self, new_only=True):
        added = 0
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
                            "last_seen": data.get("timestamp", datetime.now(timezone.utc).isoformat())
                        }
                        added += 1
                except:
                    continue
        self.save_map()
        log_to_statusbox(f"[Memory] Reindexed {added} new fragments across tiers. Current total: {len(self.memory_map)}")

    def promote(self, frag_id, to_tier):
        if frag_id not in self.memory_map:
            return False
        old_tier = self.memory_map[frag_id]["tier"]
        if old_tier == to_tier:
            return True
        src = self.base_path / old_tier / f"frag_{frag_id}.json"
        dst = self.base_path / to_tier / f"frag_{frag_id}.json"
        if not src.exists():
            return False
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        self.memory_map[frag_id]["tier"] = to_tier
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
