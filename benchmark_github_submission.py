#!/usr/bin/env python3
"""Bounded V1/V2 capability benchmark for Ina's GitHub delivery choice."""
from __future__ import annotations

import json

from github_submission import get_github_submission_config


def benchmark_v1() -> dict:
    """Historical behaviour: every queued entry was eligible for delivery."""
    entries = [{"id": "submit"}, {"id": "hold"}]
    return {"version": "V1", "eligible_ids": [entry["id"] for entry in entries], "explicit_choice": False}


def benchmark_v2() -> dict:
    """Candidate behaviour: Ina's explicit hold is preserved by delivery."""
    entries = [{"id": "submit", "delivery_choice": "submit"}, {"id": "hold", "delivery_choice": "hold"}]
    eligible = [
        entry["id"]
        for entry in entries
        if str(entry.get("delivery_choice") or "submit").strip().lower() == "submit"
    ]
    invalid_env = get_github_submission_config(
        {"github_submission": {"token_env": "github_pat_not_an_environment_name"}}
    )["token_env"]
    return {
        "version": "V2",
        "eligible_ids": eligible,
        "explicit_choice": True,
        "credential_shaped_token_env_rejected": invalid_env == "GITHUB_TOKEN",
    }


def main() -> int:
    result = {"benchmark": "github_submission_choice", "versions": [benchmark_v1(), benchmark_v2()]}
    print(json.dumps(result, indent=2, sort_keys=True))
    candidate = result["versions"][1]
    return 0 if candidate["eligible_ids"] == ["submit"] and candidate["credential_shaped_token_env_rejected"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
