"""Explicit one-batch migration of legacy outbox JSONL into SQLite."""
from __future__ import annotations

import argparse
import json

from config_layers import load_config
from discord_runtime import typed_outbox_archive_path, typed_outbox_history_path, typed_outbox_path
from github_submission import github_outbox_archive_path, github_outbox_history_path, github_outbox_path
from outbox_event_store import (
    backfill_jsonl_batch, durable_database_path, hot_database_path,
)


def profile_paths(child: str, cfg: dict):
    typed = typed_outbox_path(child, cfg)
    return {
        "typed_queue": (typed, "typed", "queued"),
        "typed_history": (typed_outbox_history_path(child, cfg), "typed", "history"),
        "typed_archive": (typed_outbox_archive_path(child, cfg), "typed", "archived"),
        "github_queue": (github_outbox_path(child), "github", "queued"),
        "github_history": (github_outbox_history_path(child), "github", "history"),
        "github_archive": (github_outbox_archive_path(child), "github", "archived"),
    }, typed


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one bounded outbox JSONL migration batch")
    parser.add_argument("profile")
    parser.add_argument("--records", type=int, default=1000)
    parser.add_argument("--bytes", type=int, default=8 * 1024 * 1024)
    args = parser.parse_args()
    cfg = load_config()
    child = str(cfg.get("current_child") or "Inazuma_Yagami")
    profiles, typed = profile_paths(child, cfg)
    if args.profile not in profiles:
        parser.error("profile must be one of: " + ", ".join(sorted(profiles)))
    source, channel, event_type = profiles[args.profile]
    result = backfill_jsonl_batch(
        source, channel=channel, event_type=event_type,
        durable_path=durable_database_path(child),
        hot_path=hot_database_path(child, typed_path=typed),
        max_records=args.records, max_bytes=args.bytes,
    )
    result["profile"] = args.profile
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
