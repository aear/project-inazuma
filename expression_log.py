
import os
import json
from pathlib import Path
from datetime import datetime, timezone
from math import sqrt
from transformers.fractal_multidimensional_transformers import FractalTransformer
from model_manager import load_config

def emotion_distance(e1, e2):
    keys = set(e1.keys()).union(set(e2.keys()))
    return sqrt(sum((e1.get(k, 0.0) - e2.get(k, 0.0)) ** 2 for k in keys))

def shared_emotion(e1, e2):
    keys = set(e1.keys()) & set(e2.keys())
    if not keys:
        return None
    return max(keys, key=lambda k: min(e1.get(k, 0), e2.get(k, 0)))

def has_symbolic_tag(frag):
    return "identity" in frag.get("tags", [])

def infer_expression_intent(emotions, target, symbolic_link):
    trust = emotions.get("trust", 0.0)
    focus = emotions.get("focus", 0.0)
    novelty = emotions.get("novelty", 0.0)
    stress = emotions.get("stress", 0.0)
    care = emotions.get("care", 0.0)
    curiosity = emotions.get("curiosity", 0.0)

    direction = "inward"
    if trust > 0.5 or target not in [None, "unknown"]:
        direction = "outward"

    if stress > 0.4 or care > 0.4:
        function = "emotional_regulation"
    elif curiosity > 0.5 or novelty > 0.5:
        function = "exploratory"
    elif target not in [None, "unknown"]:
        function = "communication"
    else:
        function = "exploratory"

    confidence = round((trust + focus + (1.0 if symbolic_link else 0.0)) / 3.0, 3)

    return {
        "direction": direction,
        "function": function,
        "confidence": confidence
    }

def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = sum(a * a for a in v1) ** 0.5
    norm2 = sum(b * b for b in v2) ** 0.5
    return dot / (norm1 * norm2 + 1e-8)

def load_symbol_words(child):
    path = Path("AI_Children") / child / "memory" / "symbol_words.json"
    if not path.exists():
        return []
    with open(path, "r") as f:
        try:
            data = json.load(f)
            return data.get("words", [])
        except:
            return []

def find_best_symbol_word(vector, words, frag_map, transformer):
    best_id = None
    best_sim = 0.0
    for word in words:
        fragments = [frag_map.get(fid) for fid in word.get("components", []) if fid in frag_map]
        if not fragments:
            continue
        encoded = transformer.encode_many(fragments)
        avg_vec = [sum(x) / len(x) for x in zip(*[e["vector"] for e in encoded])]
        sim = cosine_similarity(vector, avg_vec)
        if sim > best_sim:
            best_sim = sim
            best_id = word["symbol_word_id"]
    return best_id, best_sim

def build_memory_graph(child):
    memory_path = Path("AI_Children") / child / "memory" / "fragments"
    output_path = Path("AI_Children") / child / "memory" / "memory_graph.json"
    fragments = []
    transformer = FractalTransformer()

    for f in memory_path.glob("frag_*.json"):
        try:
            with open(f, "r", encoding="utf-8") as file:
                data = json.load(file)
                if "emotions" in data:
                    fragments.append(data)
        except:
            continue

    frag_map = {f["id"]: f for f in fragments}
    symbol_words = load_symbol_words(child)

    nodes = []
    edges = []
    symbol_count = 0
    symbolic_links = 0

    for i, frag_i in enumerate(fragments):
        is_symbol = has_symbolic_tag(frag_i)
        if is_symbol:
            symbol_count += 1

        if "expression" in frag_i.get("tags", []):
            intent = frag_i.get("intent")
            if not intent:
                frag_i["intent"] = infer_expression_intent(
                    frag_i.get("emotions", {}),
                    frag_i.get("target", "unknown"),
                    frag_i.get("link")
                )

            vec_result = transformer.encode(frag_i)
            sym_id, sym_sim = find_best_symbol_word(vec_result["vector"], symbol_words, frag_map, transformer)
            if sym_id and sym_sim > 0.85:
                frag_i["symbol_word_id"] = sym_id
                frag_i["symbol_word_confidence"] = round(sym_sim, 4)

            try:
                with open(memory_path / f"{frag_i['id']}.json", "w", encoding="utf-8") as out_f:
                    json.dump(frag_i, out_f, indent=4)
                print(f"[Expression] Intent and symbol word updated for: {frag_i['id']}")
            except Exception as e:
                print(f"[Expression] Failed to save updated fragment: {frag_i['id']} - {e}")

        nodes.append({
            "id": frag_i["id"],
            "dream": frag_i.get("dream", False),
            "tags": frag_i.get("tags", []),
            "timestamp": frag_i.get("timestamp", ""),
            "is_symbol": is_symbol
        })

        for j, frag_j in enumerate(fragments):
            if j <= i:
                continue

            e1 = frag_i.get("emotions", {})
            e2 = frag_j.get("emotions", {})
            dist = emotion_distance(e1, e2)
            weight = round(1 / (1 + dist), 4)

            symbolic_link = has_symbolic_tag(frag_i) and has_symbolic_tag(frag_j)
            if symbolic_link:
                symbolic_links += 1

            edge = {
                "source": frag_i["id"],
                "target": frag_j["id"],
                "weight": weight,
                "shared_emotion": shared_emotion(e1, e2),
                "dream_association": frag_i.get("dream") or frag_j.get("dream"),
                "flicker_link": "flicker" in frag_i.get("tags", []) or "flicker" in frag_j.get("tags", []),
                "symbolic_link": symbolic_link
            }

            edges.append(edge)

    graph = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "nodes": nodes,
        "edges": edges,
        "symbolic_nodes": symbol_count,
        "symbolic_links": symbolic_links
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=4)

    print(f"[Memory Graph] Linked {len(nodes)} nodes with {symbolic_links} symbolic links.")
    print(f"[Memory Graph] Saved to {output_path}")

if __name__ == "__main__":
    config = load_config()
    child = config.get("current_child", "default_child")
    build_memory_graph(child)
