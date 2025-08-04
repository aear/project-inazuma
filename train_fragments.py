# === train_fragments.py (Rewritten for Scale and Modular Use) ===

import os
import json
from pathlib import Path
from datetime import datetime, timezone
from fractal_multidimensional_transformers import FractalTransformer
from model_manager import load_config, mark_module_running, clear_module_running
from gui_hook import log_to_statusbox
from memory_graph import MemoryManager


# === Utility ===
def load_base_model(child):
    path = Path("AI_Children") / child / "ina_pretrained_model.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def save_base_model(child, model_data):
    path = Path("AI_Children") / child / "ina_pretrained_model.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model_data, f, indent=4)

# === Fragment Loader ===
def load_fragments_from_disk(child, limit=500):
    frag_dir = Path("AI_Children") / child / "memory" / "fragments"
    if not frag_dir.exists():
        return []

    loaded = []
    seen = set()
    for f in sorted(frag_dir.glob("frag_*.json")):
        try:
            with open(f, "r") as jf:
                data = json.load(jf)
                if data.get("id") not in seen:
                    loaded.append(data)
                    seen.add(data["id"])
        except:
            continue
        if len(loaded) >= limit:
            break
    return loaded

# === Fragment Encoder ===
def encode_fragments_batch(fragments, transformer):
    transformer.load_precision_profile()
    encoded = transformer.encode_many(fragments)
    for f, enc in zip(fragments, encoded):
        if "self_read" in f.get("tags", []):
            f["importance"] = round(enc["vector"][0] * 1.1, 4)
    return encoded

# === Public API: Pretrain Integration ===
def train_model(config=None, child=None, store=True):
    mark_module_running("train_fragments")
    if not config:
        config = load_config()
    if not child:
        child = config.get("current_child", "Inazuma_Yagami")

    fragments = load_fragments_from_disk(child, limit=800)
    if not fragments:
        log_to_statusbox("[Train] No fragments to train.")
        clear_module_running("train_fragments")
        return

    transformer = FractalTransformer()
    encoded = encode_fragments_batch(fragments, transformer)

    model_data = load_base_model(child)
    model_data["structure"] = encoded

    if store:
        save_base_model(child, model_data)
        log_to_statusbox(f"[Train] Trained on {len(encoded)} fragments and saved model.")
        # === Index new fragments after training
        log_to_statusbox(f"[Train] Sending call to memory manager for indexing and storage.")
        memory = MemoryManager(child)
        memory.reindex(new_only=True)
    else:
        log_to_statusbox(f"[Train] Trained on {len(encoded)} fragments (not saved).")



    clear_module_running("train_fragments")
    
# === Public API: Live/Module Training ===
def train_from_memory(fragments, child, store=False):
    transformer = FractalTransformer()
    encoded = encode_fragments_batch(fragments, transformer)
    if store:
        model_data = load_base_model(child)
        model_data.setdefault("structure", []).extend(encoded)
        save_base_model(child, model_data)
        # === Index new fragments after training
        log_to_statusbox(f"[Train] Sending call to memory manager for indexing and storage.")
        memory = MemoryManager(child)
        memory.reindex(new_only=True)
    return encoded
