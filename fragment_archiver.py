
# === PATCH: fragment_archiver.py ===
# Runs emotion_engine on archive candidates before deletion to ensure tagging

import os
import json
import shutil
from pathlib import Path
from emotion_engine import tag_fragment_emotions
from model_manager import load_config

def archive_fragment(frag, archive_path):
    frag_id = frag.get("id")
    if not frag_id:
        return
    archive_path.mkdir(parents=True, exist_ok=True)
    with open(archive_path / f"{frag_id}.json", "w") as f:
        json.dump(frag, f, indent=4)
    print(f"[Archive] Archived fragment: {frag_id}")

def scan_and_archive(child, threshold=0.1):
    frag_path = Path("AI_Children") / child / "memory" / "fragments"
    archive_path = frag_path / "archived"

    count = 0
    for f in frag_path.glob("frag_*.json"):
        try:
            with open(f, "r") as file:
                frag = json.load(file)

            if frag.get("importance", 1.0) < threshold:
                # Ensure emotion tagging before archiving
                tagged = tag_fragment_emotions(frag)
                archive_fragment(tagged, archive_path)
                os.remove(f)
                count += 1
        except Exception as e:
            print(f"[Archive] Error processing {f.name}: {e}")

    print(f"[Archive] Archived {count} fragments.")

def main():
    config = load_config()
    child = config.get("current_child", "default_child")
    scan_and_archive(child)

if __name__ == "__main__":
    main()
