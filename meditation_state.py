
# === meditation_state.py (Full Rewrite) ===
# Reflection mode — builds maps, performs symbolic introspection, exits on emotional drift

import time
import json
import os
import fcntl
from datetime import datetime, timezone
from pathlib import Path
from model_manager import (
    mark_module_running, clear_module_running, update_inastate, get_inastate,
    get_sweet_spots, seed_self_question, load_config, request_scheduler_task
)
from gui_hook import log_to_statusbox
from self_inquiry_journey import begin_self_inquiry, continue_self_inquiry, current_inquiry_request


def _meditation_lock(child):
    lock_path = Path("AI_Children") / child / "memory" / "meditation_state.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def save_log(entry):
    child = load_config().get("current_child", "default_child")
    log_path = Path("AI_Children") / child / "memory" / "meditation_log.json"
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
        json.dump(log[-100:], f, indent=2)

def enter_meditation():
    child = load_config().get("current_child", "default_child")
    update_inastate("meditating", True)
    update_inastate("last_meditation_time", datetime.now(timezone.utc).isoformat())
    log_to_statusbox("[Meditation] Entering meditation mode.")
    state_path = Path("AI_Children") / child / "memory" / "session_state.json"
    state = {}
    if state_path.exists():
        try:
            with open(state_path, "r") as f:
                state = json.load(f)
        except:
            pass
    state["meditation_started_at"] = datetime.now(timezone.utc).isoformat()
    with open(state_path, "w") as f:
        json.dump(state, f, indent=4)

def exit_meditation(reason="natural"):
    update_inastate("meditating", False)
    clear_module_running("meditation_state")
    log_to_statusbox(f"[Meditation] Exiting meditation: {reason}")


def meditate_loop():
    child = load_config().get("current_child", "default_child")
    lock_handle = _meditation_lock(child)
    if lock_handle is None:
        log_to_statusbox("[Meditation] A reflective cycle is already running; duplicate launch skipped.")
        return
    try:
        enter_meditation()
        _begin_requested_self_inquiry()
        _run_meditation_cycles()
    finally:
        if get_inastate("meditating", False):
            exit_meditation("interrupted")
        lock_handle.close()


def _begin_requested_self_inquiry():
    """Admit a voluntary inquiry during meditation without auto-continuing it."""
    request = get_inastate("self_inquiry_request")
    if not isinstance(request, dict) or not request.get("requested"):
        return None
    try:
        journey = begin_self_inquiry(
            request.get("question"),
            trigger_references=request.get("trigger_references") or (),
            depth_budget=request.get("depth_budget", 3),
            include_code=bool(request.get("include_code", False)),
        )
    except (TypeError, ValueError) as exc:
        update_inastate("self_inquiry_request", {
            **request, "requested": False, "status": "rejected", "error": str(exc)[:240],
        })
        return None
    update_inastate("self_inquiry_journey", journey)
    update_inastate("self_inquiry_evidence_request", current_inquiry_request(journey))
    update_inastate("self_inquiry_request", {
        **request, "requested": False, "status": "started", "journey_id": journey["journey_id"],
    })
    log_to_statusbox("[Meditation] Ina began a bounded self-inquiry journey.")
    return journey


def _continue_requested_self_inquiry():
    """Advance exactly once when Ina explicitly asks to continue her journey."""
    request = get_inastate("self_inquiry_continue_request")
    journey = get_inastate("self_inquiry_journey")
    if not isinstance(request, dict) or not request.get("requested") or not isinstance(journey, dict):
        return None
    try:
        updated = continue_self_inquiry(
            journey, choice=request.get("choice"),
            observation_references=request.get("observation_references") or (),
            hypotheses=request.get("hypotheses") or (),
        )
    except (PermissionError, TypeError, ValueError) as exc:
        update_inastate("self_inquiry_continue_request", {
            **request, "requested": False, "status": "rejected", "error": str(exc)[:240],
        })
        return None
    update_inastate("self_inquiry_journey", updated)
    update_inastate("self_inquiry_evidence_request", current_inquiry_request(updated))
    update_inastate("self_inquiry_continue_request", {
        **request, "requested": False, "status": "applied",
        "journey_id": updated["journey_id"], "stage_index": updated["stage_index"],
    })
    log_to_statusbox("[Meditation] Ina chose the next state of her self-inquiry journey.")
    return updated


def _run_meditation_cycles():
    loop_count = 0
    max_loops = 10

    while loop_count < max_loops:
        loop_count += 1
        log_to_statusbox(f"[Meditation] Reflective cycle {loop_count}/{max_loops}")
        _continue_requested_self_inquiry()
        try:
            request_scheduler_task("emotion_engine_run", reason="meditation_cycle", priority=74)
            request_scheduler_task("who_am_i_run", reason="meditation_cycle", priority=70)
            request_scheduler_task("memory_graph_neural", reason="meditation_cycle", priority=88)
            request_scheduler_task("meaning_map_refresh", reason="meditation_cycle", priority=78)
            request_scheduler_task("logic_map_refresh", reason="meditation_cycle", priority=76)
            request_scheduler_task("emotion_map_refresh", reason="meditation_cycle", priority=74)
            request_scheduler_task("predictive_layer_run", reason="meditation_cycle", priority=76)
            log_to_statusbox("[Meditation] Queued core reflection modules.")
        except Exception as e:
            log_to_statusbox(f"[Meditation] Scheduler request error: {e}")

        emo = get_inastate("current_emotions") or {}
        log_to_statusbox(f"[Meditation] Emotional snapshot: {json.dumps(emo, indent=2)}")
        fuzz = emo.get("fuzz_level", 0.0)
        stress = emo.get("stress", 0.0)
        negativity = emo.get("negativity", 0.0)
        intensity = emo.get("intensity", 0.0)

        spots = get_sweet_spots()
        if intensity > spots.get("speech_rate", {}).get("max", 1.2):
            seed_self_question("Why does my thinking feel intense?")
            log_to_statusbox("[Meditation] Seeded question: thinking intensity.")

        if stress > spots.get("cpu_temperature", {}).get("max", 0.7):
            seed_self_question("Why am I stressed during meditation?")
            log_to_statusbox("[Meditation] Seeded question: thinking intensity.")
        if negativity > 0.5:
            exit_meditation("negativity spike")
            return
        if fuzz > 0.7 and stress > 0.5:
            exit_meditation("transition to dream")
            request_scheduler_task("dreamstate_run", reason="meditation_transition", priority=84)
            return

        save_log({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "loop": loop_count,
            "emotions": emo
        })
        log_to_statusbox(f"[Meditation] Logged cycle {loop_count} to meditation_log.json")


        time.sleep(30)

    exit_meditation("completed")

if __name__ == "__main__":
    meditate_loop()
