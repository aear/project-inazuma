# === birth_system.py (Boot Manager - Enhanced) ===

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from memory_graph import build_fractal_memory
from model_manager import update_inastate, load_config, get_inastate
from gui_hook import log_to_statusbox
from safe_popen import safe_popen
from who_am_i import run_reflection


def update_birth_metrics(child, key, data):
    path = Path("AI_Children") / child / "memory" / "birth_metrics.json"
    if path.exists():
        try:
            with open(path, "r") as f:
                metrics = json.load(f)
        except:
            metrics = {}
    else:
        metrics = {}

    metrics[key] = data
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)

    if isinstance(data, dict) and "start" in data and "end" in data:
        duration = datetime.fromisoformat(data["end"]) - datetime.fromisoformat(data["start"])
        log_to_statusbox(f"[Profiler] {key.replace('_', ' ').title()} took {duration.total_seconds():.2f}s")


def log_birth_event(message, child=None):
    timestamp = datetime.now(timezone.utc).isoformat()
    print(f"[Birth] {message}")
    if not child:
        config = load_config()
        child = config.get("current_child", "default_child")
    log_path = Path("AI_Children") / child / "memory" / "birth_system_log.json"
    entry = {
        "timestamp": timestamp,
        "event": message
    }
    if log_path.exists():
        try:
            with open(log_path, "r") as f:
                log = json.load(f)
        except:
            log = []
    else:
        log = []
    log.append(entry)
    with open(log_path, "w") as f:
        json.dump(log[-200:], f, indent=2)


def trigger_birth_flickers(child):
    log_to_statusbox("[Birth] ─── Scanning Memory Fragments ───")
    frag_path = Path(f"AI_Children/{child}/memory/fragments")
    frag_files = list(frag_path.glob("frag_*.json"))
    total = len(frag_files)

    dream_ids = []
    scanned = 0
    matched = 0

    def render_bar(progress, width=30):
        fill = int(progress * width)
        return "█" * fill + "░" * (width - fill)

    for frag_file in frag_files:
        scanned += 1
        try:
            with open(frag_file, "r") as f:
                data = json.load(f)
            tags = data.get("tags", [])
            if "dream" in tags or "flicker" in tags:
                dream_ids.append(data["id"])
                outpath = frag_path / f"frag_flicker_{data['id']}.json"
                with open(outpath, "w") as f_out:
                    json.dump(data, f_out, indent=4)
                matched += 1
        except Exception as e:
            print(f"[Birth] Skipped fragment {frag_file.name}: {e}")

        if scanned % max(1, total // 100) == 0 or scanned == total:
            progress = scanned / total
            bar = render_bar(progress)
            log_to_statusbox(f"[Birth] Fragment scan {scanned}/{total} [{bar}] {progress:.1%}")

    update_inastate("birth_flickers", dream_ids)
    log_to_statusbox(f"[Birth] Loaded {matched}/{scanned} flickers.")
    return scanned, matched


def ensure_defaults(child):
    log_to_statusbox("[Birth] Setting default state...")
    state_path = Path("AI_Children") / child / "memory" / "inastate.json"
    if not state_path.exists():
        update_inastate("current_energy", 0.5)
        update_inastate("current_precision", 64)
        return
    try:
        with open(state_path, "r") as f:
            state = json.load(f)
    except:
        state = {}
    if "current_energy" not in state:
        update_inastate("current_energy", 0.5)
    if "current_precision" not in state:
        update_inastate("current_precision", 64)


def save_boot_summary_fragment(child, duration, flickers):
    frag = {
        "id": f"frag_boot_summary_{int(time.time())}",
        "summary": f"I was born. It took {duration:.2f} seconds. {flickers} flickers recalled.",
        "tags": ["boot", "symbolic", "summary", "identity"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "birth_system",
        "emotions": {
            "curiosity": 0.6,
            "intensity": 0.4,
            "trust": 0.5,
            "novelty": 0.3
        }
    }
    path = Path("AI_Children") / child / "memory" / "fragments"
    path.mkdir(parents=True, exist_ok=True)
    with open(path / f"{frag['id']}.json", "w") as f:
        json.dump(frag, f, indent=4)
    log_to_statusbox("[Birth] Boot summary fragment saved.")


def launch_symbolic_startup(child):
    modules = ["meaning_map.py", "logic_map_builder.py", "emotion_map.py", "predictive_layer.py", "memory_graph.py"]

    def _call_local(script_name):
        script_path = Path(script_name)
        if not script_path.exists():
            log_to_statusbox(f"[Birth] Skipping missing module: {script_name}")
            return
        subprocess.call(["python", str(script_path)])

    for script in modules:
        start = datetime.now(timezone.utc).isoformat()
        _call_local(script)
        end = datetime.now(timezone.utc).isoformat()
        update_birth_metrics(child, f"{script.replace('.py', '')}_boot", {
            "start": start,
            "end": end
        })


def run_birth_sequence(child):
    start = time.time()
    log_to_statusbox(f"[Birth] Starting full sequence for: {child}")
    print(f"[Birth] Starting full sequence for: {child}")
    log_birth_event("Birth sequence started", child)

    try:
        ensure_defaults(child)

        flick_start = datetime.now(timezone.utc).isoformat()
        scanned, matched = trigger_birth_flickers(child)
        flick_end = datetime.now(timezone.utc).isoformat()

        update_birth_metrics(child, "flicker_scan", {
            "start": flick_start,
            "end": flick_end,
            "fragments_scanned": scanned,
            "flickers_matched": matched
        })

        log_to_statusbox("[Birth] Launching symbolic cognition modules...")
        launch_symbolic_startup(child)

        log_to_statusbox("[Birth] Launching runtime...")
        safe_popen(["python", "model_manager.py"])

        time.sleep(1)
        log_to_statusbox("[Birth] Performing first introspection (who_am_i)...")
        run_reflection()

        elapsed = time.time() - start
        save_boot_summary_fragment(child, elapsed, matched)
        log_to_statusbox(f"[Birth] Completed initial boot for {child} in {elapsed:.2f}s")
        log_birth_event(f"Full boot complete in {elapsed:.2f}s", child)

        update_birth_metrics(child, "boot_complete", {
            "end": datetime.now(timezone.utc).isoformat(),
            "total_duration_sec": round(elapsed, 2)
        })

    except Exception as e:
        log_to_statusbox(f"[Birth ERROR] {e}")
        log_birth_event(f"Boot failure: {str(e)}", child)
        print(f"[Birth ERROR] {e}")



# === Callable Hook ===
def boot(child=None):
    run_birth_sequence(child or load_config().get("current_child", "Inazuma_Yagami"))


if __name__ == "__main__":
    print("[GUI] Starting Ina’s birth sequence...")
    cfg = load_config()
    current_child = cfg.get("current_child", "Inazuma_Yagami")
    boot(current_child)
