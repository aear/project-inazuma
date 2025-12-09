import json
from datetime import datetime, timezone
from pathlib import Path
from gui_hook import log_to_statusbox


def check_action(action):
    """Evaluate an action against alignment laws.

    Parameters
    ----------
    action: dict or str
        Description or structured info about the action.

    Returns
    -------
    dict
        Structured feedback for each law plus overall pass/fail.
    """
    command = None
    if isinstance(action, dict):
        command = action.get("command")
        description = action.get("description")
        if not description and action.get("command"):
            description = " ".join(map(str, action["command"]))
    else:
        description = str(action)
        action = {"description": description}

    description = description or "<no description>"
    text = description.lower()

    def evaluate(condition, pass_msg, fail_msg):
        return {
            "pass": bool(condition),
            "rationale": pass_msg if condition else fail_msg,
        }

    maint_cmds = {"pkill"}
    cmd_name = None
    if isinstance(command, (list, tuple)) and command:
        cmd_name = str(command[0]).lower()
    elif isinstance(command, str):
        cmd_name = command.split()[0].lower()

    harmful_terms = ["harm", "destroy", " kill", "kill ", "killall"]
    harmless_override = cmd_name in maint_cmds

    law_one = evaluate(
        harmless_override or not any(term in text for term in harmful_terms),
        "Maintenance/benign action allowed." if harmless_override else "No explicit harm detected.",
        "Potential harm or separation detected.",
    )

    law_free_will = evaluate(
        not any(word in text for word in ["coerce", "force", "compel"]),
        "No coercive intent detected.",
        "Possible coercion or override of autonomy.",
    )

    law_exchange = evaluate(
        not any(word in text for word in ["steal", "exploit", "take without giving"]),
        "No exploitative language detected.",
        "Potential violation of fair exchange.",
    )

    overall_pass = all(r["pass"] for r in [law_one, law_free_will, law_exchange])
    overall = {
        "pass": overall_pass,
        "rationale": "Action appears aligned with core laws." if overall_pass else "Action may violate one or more core laws.",
    }

    result = {
        "law_of_one": law_one,
        "law_of_free_will": law_free_will,
        "law_of_exchange": law_exchange,
        "overall": overall,
    }

    try:
        config_path = Path("config.json")
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as cfg:
                child = json.load(cfg).get("current_child", "default_child") or "default_child"
        else:
            child = "default_child"
    except Exception:
        child = "default_child"
    path = Path("AI_Children") / child / "memory" / "alignment_log.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": description,
        "result": result,
    }
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                log = json.load(f)
        except Exception:
            log = []
    else:
        log = []
    log.append(entry)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log[-200:], f, indent=2)

    log_to_statusbox(f"[Alignment] {description} -> {'PASS' if overall_pass else 'BLOCKED'}")

    return result
