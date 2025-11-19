# eq_engine.py

from typing import Dict, Any, List

def evaluate_emotional_state(emotion_vector: Dict[str, float],
                             context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Core entry point.
    emotion_vector: 24D sliders, e.g. {"intensity": 0.8, "stress": 0.7, ...}
    context: extra info from inastate, prediction_layer, etc.
    Returns an eq_report dict.
    """
    state_labels: List[str] = []

    risk_score = 0.0
    opportunity_score = 0.0

    # --- simple helpers ---
    def v(name: str) -> float:
        return emotion_vector.get(name, 0.0)

    # Example heuristics – tune later:
    # Overwhelm
    if v("intensity") > 0.7 and v("stress") > 0.7 and v("clarity") < -0.3:
        state_labels.append("overwhelm")
        risk_score = max(risk_score, 0.9)

    # Moral tension
    if v("care") > 0.6 and v("risk") > 0.4 and abs(v("positivity") - v("negativity")) < 0.3:
        state_labels.append("moral_tension")
        risk_score = max(risk_score, 0.7)

    # Numb / apathy
    if v("intensity") < -0.5 and v("care") < -0.3 and v("novelty") < -0.3:
        state_labels.append("apathy")
        risk_score = max(risk_score, 0.4)

    # Curiosity / exploration
    if v("novelty") > 0.5 and v("curiosity") > 0.5 and v("stress") < 0.4:
        state_labels.append("curious")
        opportunity_score = max(opportunity_score, 0.8)

    # Flow
    if 0.3 < v("intensity") < 0.8 and v("stress") < 0.3 and v("clarity") > 0.4:
        state_labels.append("flow")
        opportunity_score = max(opportunity_score, 0.9)

    # Transcendent
    if v("trust") > 0.6 and v("clarity") > 0.6 and v("awe") > 0.5 and v("stress") < 0.2:
        state_labels.append("transcendent")
        opportunity_score = max(opportunity_score, 1.0)

    # Build recommended actions based on labels
    recommended_actions: List[Dict[str, Any]] = []
    for label in state_labels:
        recommended_actions.extend(_actions_for_label(label, emotion_vector, context))

    return {
        "state_labels": state_labels,
        "risk_score": float(min(risk_score, 1.0)),
        "opportunity_score": float(min(opportunity_score, 1.0)),
        "recommended_actions": recommended_actions
    }


def _actions_for_label(label: str,
                       emotion_vector: Dict[str, float],
                       context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Map each detected label to suggested cross-module nudges.
    """
    actions: List[Dict[str, Any]] = []

    if label == "overwhelm":
        actions.append({"target": "instinct_engine", "action": "lower_precision"})
        actions.append({"target": "model_manager", "action": "reduce_io_load"})
        actions.append({"target": "meditation_state", "action": "enter_short"})

    elif label == "moral_tension":
        actions.append({
            "target": "logic_engine",
            "action": "seed_questions",
            "details": ["Who is affected?", "What harm could this cause?"]
        })
        actions.append({"target": "who_am_i", "action": "reflect_on_values"})

    elif label == "apathy":
        actions.append({"target": "boredom_state", "action": "start_exploration"})
        actions.append({"target": "self_reflection", "action": "log_apathy"})

    elif label == "curious":
        actions.append({"target": "boredom_state", "action": "deepen_exploration"})

    elif label == "flow":
        actions.append({"target": "precision_evolution", "action": "log_flow_state"})
        actions.append({"target": "memory_graph", "action": "mark_stable_fragments"})

    elif label == "transcendent":
        actions.append({"target": "symbol_generator", "action": "generate_symbols"})
        actions.append({"target": "self_reflection", "action": "snapshot_state"})

    return actions
