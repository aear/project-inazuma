"""V1/V2 benchmark for usage visibility and reversible approval presentation."""
from pathlib import Path

from codex_harness import token_usage_payload


def main() -> int:
    source = Path("codex_harness_ui.html").read_text(encoding="utf-8")
    usage = token_usage_payload({
        "last": {"totalTokens": 12}, "total": {"totalTokens": 120},
        "modelContextWindow": 200000,
    })
    v2 = {
        "usage_visible": int('id="usageStatus"' in source and usage["total"]["totalTokens"] == 120),
        "per_type_preferences": int(all(key in source for key in ("prefCommand", "prefFile", "prefPermissions"))),
        "reversible": int("localStorage.removeItem(APPROVAL_PREFS_KEY)" in source),
        "no_auto_approval": int("Approval always requires your click" in source and "button.onclick=async" in source),
    }
    result = {
        "V1": {"usage_visible": 0, "per_type_preferences": 0, "reversible": 0, "no_auto_approval": 1},
        "V2": v2,
    }
    print(result)
    return 0 if all(v2.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
