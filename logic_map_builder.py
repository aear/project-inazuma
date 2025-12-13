# === logic_map_builder.py (Neural Rewrite) ===

import os
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from transformers.fractal_multidimensional_transformers import FractalTransformer
from model_manager import load_config
from gui_hook import log_to_statusbox
from symbol_generator import generate_symbol_from_parts
from body_schema import get_region_anchors

LOGIC_MAP_BURST_DEFAULT = 150  # neurons per pass before pausing

def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    return dot / (norm1 * norm2 + 1e-8)

def _flatten_numeric(value):
    """
    Collect numeric scalars from nested dicts/lists so old logic entries without
    prediction vectors still produce a usable embedding.
    """
    if isinstance(value, bool):
        return [1.0 if value else 0.0]
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, dict):
        items = []
        for v in value.values():
            items.extend(_flatten_numeric(v))
        return items
    if isinstance(value, (list, tuple)):
        items = []
        for v in value:
            items.extend(_flatten_numeric(v))
        return items
    return []


# --- spatial helpers -------------------------------------------------

def _load_body_anchors():
    anchors = get_region_anchors()
    if not anchors:
        return {}
    return anchors


def _guess_region_for_logic(entry, anchors):
    desc = (entry.get("description") or "").lower()
    if any(k in desc for k in ("emotion", "feeling", "heart")) and "chest" in anchors:
        return "chest"
    if any(k in desc for k in ("self", "identity", "core")) and "core" in anchors:
        return "core"
    if any(k in desc for k in ("speech", "voice")) and "throat" in anchors:
        return "throat"
    return "head" if "head" in anchors else next(iter(anchors.keys()), "head")


def _project_vector_to_anchor(vector, anchor, seed):
    center = anchor.get("center", [0.0, 0.0, 0.0])
    radius = float(anchor.get("radius", 1.0) or 1.0)
    rng = random.Random(hash(seed) & 0xFFFFFFFF)

    if vector:
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

def extract_logic_vector(entry):
    """
    Robustly derive a vector from whatever data the logic entry carries.
    """
    prediction = entry.get("prediction", {})
    vector = []

    if isinstance(prediction, dict) and prediction:
        vector = [prediction.get(k, 0.0) for k in sorted(prediction.keys())]
    elif isinstance(prediction, list):
        vector = [float(v) for v in prediction if isinstance(v, (int, float))]

    if not vector:
        alt = entry.get("prediction_vector") or entry.get("predicted_vector")
        if isinstance(alt, dict):
            vector = [alt.get(k, 0.0) for k in sorted(alt.keys())]
        elif isinstance(alt, list):
            vector = [float(v) for v in alt if isinstance(v, (int, float))]

    if not vector:
        traces = entry.get("trace_tests", [])
        flat = []
        for test in traces:
            for step in test.get("trace", []):
                flat.extend(_flatten_numeric(step.get("result")))
        vector = flat

    return vector

def load_logic_memory(child):
    path = Path("AI_Children") / child / "memory" / "logic_memory.json"
    if not path.exists():
        return []
    with open(path, "r") as f:
        return json.load(f)

def build_logic_neural_map(logic_entries):
    transformer = FractalTransformer()
    anchors = _load_body_anchors()
    fallback_anchor = anchors.get("head", {"center": [0.0, 0.0, 0.0], "radius": 2.0})
    neurons = []
    edges = []

    for i, entry in enumerate(logic_entries):
        vector = extract_logic_vector(entry)
        if not vector:
            continue

        # Generate procedural symbol per neuron
        emo = random.choice(["trust", "curiosity", "tension", "anger", "fear"])
        mod = random.choice(["spiral", "pulse", "soft", "sharp"])
        con = random.choice(["truth", "pattern", "change", "self", "unknown"])
        symbol = generate_symbol_from_parts(emo, mod, con)
        node_id = f"neuron_logic_{i:04}"
        region = _guess_region_for_logic(entry, anchors)
        anchor = anchors.get(region, fallback_anchor)
        position = _project_vector_to_anchor(vector, anchor, seed=node_id)
        neurons.append({
            "id": node_id,
            "timestamp": entry["timestamp"],
            "description": entry.get("description", ""),
            "symbol_word_id": entry.get("symbol_word_id", ""),
            "symbol": symbol,
            "vector": vector,
            "position": position,
            "region": region,
            "network_type": "logic"
        })

    pos_map = {n["id"]: n.get("position") for n in neurons}
    for i, a in enumerate(neurons):
        for j, b in enumerate(neurons):
            if j <= i:
                continue
            sim = cosine_similarity(a["vector"], b["vector"])
            edge_type = "reinforces" if sim > 0.8 else "contradicts" if sim < 0.4 else "curious"
            direction = None
            pos_a = pos_map.get(a["id"])
            pos_b = pos_map.get(b["id"])
            if pos_a and pos_b:
                delta = [pos_b[k] - pos_a[k] for k in range(3)]
                norm = math.sqrt(sum(d * d for d in delta))
                if norm > 1e-6:
                    direction = [d / norm for d in delta]
            edges.append({
                "source": a["id"],
                "target": b["id"],
                "weight": round(sim, 4),
                "type": edge_type,
                "direction": direction,
                "network_type": "logic"
            })

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "neurons": neurons,
        "synapses": edges
    }

def save_logic_neural_map(child, logic_map):
    out_path = Path.home() / "Projects" / "Project Inazuma" / "AI_Children" / child / "memory" / "neural" / "logic_neural_map.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(logic_map, f, indent=4)
    log_to_statusbox(f"[LogicMap] Neural logic map saved: {len(logic_map['neurons'])} neurons, {len(logic_map['synapses'])} synapses.")

def run_logic_map_builder():
    config = load_config()
    child = config.get("current_child", "default_child")
    logic_entries = load_logic_memory(child)
    burst_limit = config.get("logic_map_burst") or LOGIC_MAP_BURST_DEFAULT
    try:
        burst_limit = int(burst_limit)
    except (TypeError, ValueError):
        burst_limit = LOGIC_MAP_BURST_DEFAULT
    burst_limit = max(10, burst_limit)

    if not logic_entries:
        log_to_statusbox("[LogicMap] No logic entries found. Skipping map generation.")
        return

    if len(logic_entries) > burst_limit:
        logic_entries = logic_entries[-burst_limit:]
        log_to_statusbox(f"[LogicMap] Loaded {len(logic_entries)} recent logic entries (burst limit {burst_limit}).")
    else:
        log_to_statusbox(f"[LogicMap] Loaded {len(logic_entries)} logic entries.")

    logic_map = build_logic_neural_map(logic_entries)
    save_logic_neural_map(child, logic_map)
    log_to_statusbox("[LogicMap] Logic neural mapping complete.")

if __name__ == "__main__":
    run_logic_map_builder()
