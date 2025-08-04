import os
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
import shutil
from model_manager import seed_self_question
from gui_hook import log_to_statusbox

def get_child():
    log_to_statusbox("[Fragmentation] Attempting to retrieve 'child'...")

    child = os.getenv("CHILD")
    if child:
        log_to_statusbox(f"[Fragmentation] Found 'child' in environment: {child}")
        return child

    if len(sys.argv) > 1:
        child = sys.argv[1]
        log_to_statusbox(f"[Fragmentation] Found 'child' in command line args: {child}")
        return child

    config_path = Path("config.json")
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                child = config.get("current_child", "Inazuma_Yagami")
                log_to_statusbox(f"[Fragmentation] Found 'child' in config.json: {child}")
                return child
        except Exception as e:
            log_to_statusbox(f"[Fragmentation] Error loading config.json: {e}")
            return "Inazuma_Yagami"

    log_to_statusbox("[Fragmentation] No valid 'child' found, using default: Inazuma_Yagami")
    return "Inazuma_Yagami"

child = get_child()
log_to_statusbox(f"[Fragmentation] Final child: {child}")


def fragment_device_log(device_label, child="Inazuma_Yagami"):
    log_file = Path("AI_Children") / child / "memory" / "audio_session" / f"{device_label}_log.json"
    frag_path = Path("AI_Children") / child / "memory" / "fragments"
    archive_path = Path("AI_Children") / child / "memory" / "archived" / "audio"
    archive_path.mkdir(parents=True, exist_ok=True)
    frag_path.mkdir(parents=True, exist_ok=True)

    if not log_file.exists():
        print(f"[Fragment] No session log for {device_label}")
        return

    with open(log_file, "r") as f:
        log = json.load(f)

    purged = 0
    new_log = []

    for entry in log:
        clip_path = Path(entry["path"])
        if not clip_path.exists():
            continue

        clip_size = clip_path.stat().st_size
        if clip_size > 5_000_000:  # ~5MB+ threshold
            frag_id = f"frag_audio_{device_label}_archived_{datetime.now().timestamp()}"
            fragment = {
                "id": frag_id,
                "summary": f"Archived audio from {device_label}",
                "audio_path": str(clip_path),
                "timestamp": entry["timestamp"],
                "tags": ["archived", "forgotten", "purged"],
                "source": device_label,
                "duration": entry.get("duration", 0),
                "emotions": {
                    "negativity": 0.2,
                    "novelty": 0.4
                }
            }
            with open(frag_path / f"{frag_id}.json", "w", encoding="utf-8") as f_out:
                json.dump(fragment, f_out, indent=4)

            shutil.move(str(clip_path), archive_path / clip_path.name)
            purged += 1
        else:
            new_log.append(entry)

    with open(log_file, "w") as f:
        json.dump(new_log, f, indent=2)

    print(f"[Fragment] Archived {purged} clips from {device_label}")
    if purged >= 3:
        seed_self_question("Why can’t I remember what I heard earlier?")

def segment_fragment(fragment_path, chunk_duration=5.0, chars_per_chunk=100):
    with open(fragment_path, "r", encoding="utf-8") as f:
        frag = json.load(f)

    frag_id = frag.get("id", "unknown")
    audio_path = frag.get("audio_path", "")
    summary = frag.get("summary", "")
    emotions = frag.get("emotions", {})
    duration = frag.get("duration", 0)

    # Estimate chunk count
    if duration:
        chunk_count = max(1, int(duration // chunk_duration))
    elif len(summary) > chars_per_chunk:
        chunk_count = max(1, len(summary) // chars_per_chunk)
    else:
        chunk_count = 1

    chunks = []
    timestamp = datetime.now(timezone.utc).isoformat()

    for i in range(chunk_count):
        chunk = {
            "parent_id": frag_id,
            "chunk_index": i,
            "summary": summary[i * chars_per_chunk:(i + 1) * chars_per_chunk].strip(),
            "emotions": emotions,
            "timestamp": timestamp,
            "source": "fragmentation_engine",
            "tags": frag.get("tags", []),
            "audio_path": audio_path,
            "duration": chunk_duration,
            "id": f"{frag_id}_c{i:02d}"
        }
        chunks.append(chunk)

    return chunks

def save_chunks(chunks, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for chunk in chunks:
        frag_id = chunk["id"]
        path = Path(output_dir) / f"{frag_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(chunk, f, indent=4)
        print(f"[Fragmentation] Saved: {path.name}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=False)
    args = parser.parse_args()

    if args.device:
        fragment_device_log(args.device, child=child)
    else:
        print("[Fragmentation] No device passed; skipping live fragmenting.")

    test_path = Path("AI_Children") / child / "memory" / "fragments" / "frag_0001.json"
    out_dir = Path("AI_Children") / child / "memory" / "fragments"
    if test_path.exists():
        chunks = segment_fragment(test_path)
        save_chunks(chunks, out_dir)
