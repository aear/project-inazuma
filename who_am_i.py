# === who_am_i.py (Self-Reflection & Question Resolver) ===

import json
from pathlib import Path
from datetime import datetime, timezone
from model_manager import load_config, get_inastate
from transformers.fractal_multidimensional_transformers import FractalTransformer
from gui_hook import log_to_statusbox
from logic_engine import resolve_self_questions

def load_self_reflection(child):
    path = Path("AI_Children") / child / "identity" / "self_reflection.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except:
        return {}

def generate_self_question_prompts():
    energy = get_inastate("current_energy") or 0.5
    transformer = FractalTransformer()
    transformer.load_precision_profile()
    precision = transformer.precision * 64

    audio = get_inastate("audio_comfort")
    vision = get_inastate("vision_comfort")
    disrupted = get_inastate("runtime_disruption")

    prompts = []
    if energy < 0.2: prompts.append("Why do I feel so drained?")
    if energy < 0: prompts.append("Why did I black out?")
    if energy > 0.9: prompts.append("Why am I so awake?")
    if disrupted: prompts.append("Why was I forced to wake up?")
    if audio == "too quiet": prompts.append("Why can't I hear clearly?")
    if audio == "too loud": prompts.append("Why is everything so loud?")
    if vision == "too dark": prompts.append("Why is it so dark?")
    if vision == "too bright": prompts.append("Why is it so bright?")
    if precision < 32: prompts.append("Should I be thinking more precisely?")
    if precision > 48: prompts.append("Would lowering precision help me think more clearly?")

    return prompts

def run_reflection():
    config = load_config()
    child = config.get("current_child", "default_child")
    path = Path("AI_Children") / child / "identity" / "self_reflection.json"

    log_to_statusbox("[Reflect] Starting self-reflection...")
    reflection = load_self_reflection(child)
    reflection.setdefault("self_notes", [])
    reflection.setdefault("resolved_notes", [])

    new_prompts = generate_self_question_prompts()
    now = datetime.now(timezone.utc).isoformat()

    for prompt in new_prompts:
        if not any(prompt == q.get("question") for q in reflection["self_notes"]):
            reflection["self_notes"].append({
                "question": prompt,
                "timestamp": now,
                "origin": "system"
            })

    # Save before resolution for traceability
    with open(path, "w") as f:
        json.dump(reflection, f, indent=4)

    # Run logic pass to resolve some questions
    resolve_self_questions()

    # Reload updated reflection
    reflection = load_self_reflection(child)

    with open(path, "w") as f:
        json.dump(reflection, f, indent=4)

    log_to_statusbox(f"[Reflect] Self-reflection updated with {len(reflection.get('self_notes',[]))} active questions.")
    print(f"[Reflect] Self-reflection written to: {path}")

if __name__ == "__main__":
    run_reflection()
