import os
import json
import math
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
from transformers.fractal_multidimensional_transformers import FractalTransformer
from model_manager import load_config, seed_self_question
from gui_hook import log_to_statusbox
from symbol_generator import generate_symbol_from_parts
import random
from text_memory import build_text_symbol_links


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

def cluster_symbols_and_generate_words(child):
    frag_dir = Path("AI_Children") / child / "memory" / "fragments"
    fragments = []
    frag_map = {}
    log_to_statusbox("[Symbols] Scanning fragment directory...")

    for f in frag_dir.glob("frag_*.json"):
        try:
            with open(f, "r") as file:
                frag = json.load(file)
                if "symbolic" in frag.get("tags", []):
                    fragments.append(frag)
                    frag_map[frag["id"]] = frag
        except Exception as e:
            log_to_statusbox(f"[Symbols] Failed to load fragment {f.name}: {e}")

    if not fragments:
        log_to_statusbox("[Symbols] No symbolic fragments found.")
        return

    log_to_statusbox(f"[Symbols] Found {len(fragments)} symbolic fragments.")
    transformer = FractalTransformer()
    encoded = transformer.encode_many(fragments)
    cache = {e["id"]: e["vector"] for e in encoded}

    log_to_statusbox("[Symbols] Clustering symbols...")
    clusters = []
    for enc in encoded:
        matched = False
        for cluster in clusters:
            avg = [sum(x)/len(x) for x in zip(*[cache[i["id"]] for i in cluster])]
            score = sum(a*b for a, b in zip(enc["vector"], avg)) / (
                math.sqrt(sum(a*a for a in enc["vector"])) * math.sqrt(sum(b*b for b in avg)) + 1e-8)
            if score > 0.88:
                cluster.append(enc)
                matched = True
                break
        if not matched:
            clusters.append([enc])
    log_to_statusbox(f"[Symbols] Built {len(clusters)} clusters.")

    words = []
    for i, group in enumerate(clusters):
        log_to_statusbox(f"[Symbols] Processing cluster {i+1}/{len(clusters)} with {len(group)} members.")

        # Procedurally generate a unique symbol per cluster
        emotion = random.choice(list(generate_symbol_from_parts.__globals__["EMOTION_GLYPHS"].keys()))
        modulation = random.choice(list(generate_symbol_from_parts.__globals__["MODULATION_GLYPHS"].keys()))
        concept = random.choice(list(generate_symbol_from_parts.__globals__["CONCEPT_GLYPHS"].keys()))
        symbol_id = generate_symbol_from_parts(emotion, modulation, concept)

        word = {
            "symbol_word_id": f"sym_word_{i:04}",
            "components": [e["id"] for e in group],
            "summary": " + ".join(e["summary"] for e in group[:2]),
            "tags": list(set(t for e in group for t in e.get("tags", []))),
            "count": len(group),
            "birth_time": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol_id
        }

        log_to_statusbox(f"[Symbols] Assigned symbol '{symbol_id}' to cluster {i+1}.")
                # === Add symbolic word metadata ===
        word["generated_word"] = "unknown"
        word["usage_count"] = 0
        word["confidence"] = 0.0
        words.append(word)

    out_path = Path("AI_Children") / child / "memory" / "symbol_words.json"
    preserved = {}
    if out_path.exists():
        try:
            with open(out_path, "r") as prev_f:
                existing = json.load(prev_f)
            if isinstance(existing, dict):
                for key in ("proto_words", "multi_symbol_words"):
                    if key in existing:
                        preserved[key] = existing[key]
        except Exception:
            preserved = {}

    try:
        payload = {"words": words, "updated_at": datetime.now(timezone.utc).isoformat()}
        payload.update(preserved)
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=4)
        log_to_statusbox(f"[Symbols] Generated {len(words)} symbol words.")
    except Exception as e:
        log_to_statusbox(f"[Symbols] Failed to save symbol words: {e}")

    log_to_statusbox("[Symbols] Checking for drift and underuse...")
    evolve_unused_symbols(words)
    detect_conflicted_symbols(fragments, words)
    log_to_statusbox("[Symbols] Completed symbol word update.")

def run_meaning_map():
    try:
        config = load_config()
        child = config.get("current_child", "default_child")
        log_to_statusbox("[Symbols] Meaning map update starting...")
        cluster_symbols_and_generate_words(child)
        build_text_symbol_links(child)
        log_to_statusbox("[Symbols] Meaning map update finished.")
    except Exception as e:
        log_to_statusbox(f"[Symbols] Error: {e}")
        print(f"[Symbols] Error: {e}")

if __name__ == "__main__":
    run_meaning_map()
