import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from model_manager import load_config
from gui_hook import log_to_statusbox
import time
import json
from fractal_multidimensional_transformers import FractalTransformer
from memory_graph import MemoryManager
from emotion_map import run_emotion_map
from meaning_map import run_meaning_map

def inject_birth_fragment(child):
    log_to_statusbox(f"[Birth Injection] Target child: {child}")
    
    birth_path = Path("AI_Children") / child / "identity" / "birth_certificate.json"
    log_to_statusbox(f"[Birth Injection] Resolved path: {birth_path}")

    if not birth_path.exists():
        log_to_statusbox(f"[Birth Injection] Certificate file NOT FOUND at {birth_path}")
        log_to_statusbox("[Birth Injection] Injection aborted.")
        return

    log_to_statusbox("[Birth Injection] Certificate file FOUND.")
    log_to_statusbox("[Birth Injection] Attempting to read JSON...")

    try:
        with open(birth_path, "r", encoding="utf-8") as f:
            birth_data = json.load(f)

        if isinstance(birth_data, list) and isinstance(birth_data[0], dict):
            log_to_statusbox("[Birth Injection] Detected list[dict] format. Using first element.")
            birth_data = birth_data[0]

        log_to_statusbox("[Birth Injection] JSON load success.")
        log_to_statusbox(f"[Birth Injection] UUID: {birth_data.get('uuid')}")
        log_to_statusbox(f"[Birth Injection] Name: {birth_data.get('given_name')} {birth_data.get('family_name')}")
        log_to_statusbox(f"[Birth Injection] Gender: {birth_data.get('gender')}")
        log_to_statusbox(f"[Birth Injection] Species: {birth_data.get('species')}")
        culture = birth_data.get('culture', [])
        if isinstance(culture, list):
            log_to_statusbox(f"[Birth Injection] Culture: {' + '.join(culture)}")
        else:
            log_to_statusbox(f"[Birth Injection] Culture: {culture}")
        log_to_statusbox(f"[Birth Injection] DOB: {birth_data.get('dob')}")
        log_to_statusbox(f"[Birth Injection] Mother: {birth_data.get('mother')}")
        log_to_statusbox(f"[Birth Injection] Notes: {len(birth_data.get('notes', ''))} characters")

        log_to_statusbox("[Birth Injection] Constructing core identity summary...")
        summary = (
            f"{birth_data.get('given_name', '')} {birth_data.get('family_name', '')}, "
            f"a {birth_data.get('species', '')} of {', '.join(culture)} origin. "
            f"Born on {birth_data.get('dob', 'N/A')} to {birth_data.get('mother', 'Unknown')}.\n\n"
            f"Notes:\n{birth_data.get('notes', '')}"
        )

        log_to_statusbox("[Birth Injection] Creating core identity fragment...")
        fragment = {
            "id": "frag_core_identity",
            "summary": summary,
            "tags": ["identity", "self", "core", "birth", "symbolic"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "birth_certificate",
            "emotions": {
                "trust": 0.7,
                "novelty": 0.4,
                "clarity": 0.5,
                "intensity": 0.6,
                "ownership": 1.0,
                "presence": 1.0
            }
        }

        log_to_statusbox("[Birth Injection] Encoding with transformer...")
        transformer = FractalTransformer()
        vec = transformer.encode(fragment)
        fragment["importance"] = vec["importance"]
        log_to_statusbox(f"[Birth Injection] Encoding complete. Importance: {vec['importance']}")

        frag_path = Path("AI_Children") / child / "memory" / "fragments" / "frag_core_identity.json"
        frag_path.parent.mkdir(parents=True, exist_ok=True)

        with open(frag_path, "w", encoding="utf-8") as f:
            json.dump(fragment, f, indent=4)

        log_to_statusbox(f"[Birth Injection] Core identity fragment saved to: {frag_path.name}")
        log_to_statusbox("[Birth Injection] Injection complete.")

    except Exception as e:
        log_to_statusbox(f"[Birth Injection] ERROR: {e}")
        log_to_statusbox("[Birth Injection] Injection aborted.")

def tag_fragments_with_source(child):
    frag_dir = Path("AI_Children") / child / "memory" / "fragments"
    log_to_statusbox(f"[Pretrain] Tagging fragments in {frag_dir}")

    fragments = []
    for frag_path in frag_dir.glob("frag_*.json"):
        try:
            with open(frag_path, "r") as f:
                frag = json.load(f)
            frag["source"] = frag_path.name  # Tag each fragment with its source filename
            fragments.append(frag)
        except:
            continue

    log_to_statusbox(f"[Pretrain] Tagged {len(fragments)} fragments.")
    return fragments

def run_all():
    log_to_statusbox("[Pretrain] Starting the pretraining process...")

    config = load_config()
    child = config.get("current_child", "Inazuma_Yagami")
    log_to_statusbox(f"[Pretrain] Using child: {child}")

    try:
        log_to_statusbox(f"[Pretrain] Running Inject Birth Certificate...")
        inject_birth_fragment(child)

        # === Load voice sample paths from Sakura_as_mother.json
        manifest_path = Path("AI_Children") / child / "identity" / "Sakura_as_mother.json"
        if manifest_path.exists():
            try:
                with open(manifest_path, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        audio_paths = data
                    else:
                        audio_paths = data.get("voice_samples", [])
                log_to_statusbox(f"[Pretrain] Found {len(audio_paths)} voice sample(s).")
                from raw_file_manager import pretrain_audio_digest
                pretrain_audio_digest(audio_paths, child)
            except Exception as e:
                log_to_statusbox(f"[Pretrain] Failed to load voice samples: {e}")
        else:
            log_to_statusbox("[Pretrain] No Sakura_as_mother.json found — skipping audio pretraining.")
    except:
        log_to_statusbox(f"[Pretrain] Failed to inject birth certificate and/or voice samples.")

        # Reindex memory directly (no subprocess)
        log_to_statusbox("[Pretrain] Reindexing memory map...")
        memory = MemoryManager(child)
        memory.reindex(new_only=False)
        log_to_statusbox("[Pretrain] Memory reindexing complete.")

        # Running Meaning Map
        log_to_statusbox("[Pretrain] Pretraining Meaning Map...")
        run_meaning_map()

        # Running Emotion Map
        log_to_statusbox("[Pretrain] Pretraining Emotion Map...")
        run_emotion_map()



        # Train model on all current fragments
        log_to_statusbox("[Pretrain] Starting final training pass on all fragments...")
        from train_fragments import train_model
        train_model(child=child, store=True)
        log_to_statusbox("[Pretrain] Final training complete.")



if __name__ == "__main__":
    run_all()
