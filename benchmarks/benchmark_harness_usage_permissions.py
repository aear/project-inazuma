"""Versioned benchmark for harness telemetry and approval presentation."""
from pathlib import Path

from codex_harness import rate_limit_payload, token_usage_payload


def main() -> int:
    source = Path("codex_harness_ui.html").read_text(encoding="utf-8")
    usage = token_usage_payload({
        "last": {"totalTokens": 12}, "total": {"totalTokens": 120},
        "modelContextWindow": 200000,
    })
    limits = rate_limit_payload({"primary": {"usedPercent": 20}, "secondary": {"usedPercent": 40}})
    v3 = {
        "usage_visible": int('id="tokenUsageStatus"' in source and usage["total"]["totalTokens"] == 120),
        "rate_limits_visible": int('id="rateLimitStatus"' in source and limits["primary"]["usedPercent"] == 20),
        "telemetry_outside_conversation": int("thread/tokenUsage/updated" not in source and "account/rateLimits/updated" not in source),
        "per_type_preferences": int(all(key in source for key in ("prefCommand", "prefFile", "prefPermissions"))),
        "reversible": int("localStorage.removeItem(APPROVAL_PREFS_KEY)" in source),
        "no_auto_approval": int("Approval always requires your click" in source and "button.onclick=async" in source),
    }
    result = {
        "V1": {"usage_visible": 0, "per_type_preferences": 0, "reversible": 0, "no_auto_approval": 1},
        "V3": v3,
    }
    print(result)
    return 0 if all(v3.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
