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

def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    return dot / (norm1 * norm2 + 1e-8)

def load_logic_memory(child):
    path = Path("AI_Children") / child / "memory" / "logic_memory.json"
    if not path.exists():
        return []
    with open(path, "r") as f:
        return json.load(f)

def build_logic_neural_map(logic_entries):
    transformer = FractalTransformer()
    neurons = []
    edges = []

    for i, entry in enumerate(logic_entries):
        prediction = entry.get("prediction", {})
        vector = [prediction.get(k, 0.0) for k in sorted(prediction.keys())]
        if not vector:
            continue

        # Generate procedural symbol per neuron
        emo = random.choice(["trust", "curiosity", "tension", "anger", "fear"])
        mod = random.choice(["spiral", "pulse", "soft", "sharp"])
        con = random.choice(["truth", "pattern", "change", "self", "unknown"])
        symbol = generate_symbol_from_parts(emo, mod, con)

        neurons.append({
            "id": f"neuron_logic_{i:04}",
            "timestamp": entry["timestamp"],
            "description": entry.get("description", ""),
            "symbol_word_id": entry.get("symbol_word_id", ""),
            "symbol": symbol,
            "vector": vector
        })

    for i, a in enumerate(neurons):
        for j, b in enumerate(neurons):
            if j <= i:
                continue
            sim = cosine_similarity(a["vector"], b["vector"])
            edge_type = "reinforces" if sim > 0.8 else "contradicts" if sim < 0.4 else "curious"
            edges.append({
                "source": a["id"],
                "target": b["id"],
                "weight": round(sim, 4),
                "type": edge_type
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

    if not logic_entries:
        log_to_statusbox("[LogicMap] No logic entries found. Skipping map generation.")
        return

    log_to_statusbox(f"[LogicMap] Loaded {len(logic_entries)} logic entries.")
    logic_map = build_logic_neural_map(logic_entries)
    save_logic_neural_map(child, logic_map)
    log_to_statusbox("[LogicMap] Logic neural mapping complete.")

if __name__ == "__main__":
    run_logic_map_builder()
