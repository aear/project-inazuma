from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Optional

try:
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover - platform dependent
    fcntl = None

from github_submission import (
    archive_entry,
    get_current_child,
    get_github_submission_config,
    github_bridge_lock_path,
    load_completed_history_ids,
    load_config,
    load_submitted_count_for_day,
    log_history,
    read_pending_entries,
    submit_issue,
)

logger = logging.getLogger("github_bridge")
_LOCK_HANDLE = None


def _acquire_single_instance_lock(child: str) -> bool:
    if fcntl is None:
        return True
    lock_path = github_bridge_lock_path(child)
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("w", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return False
        handle.write(str(os.getpid()))
        handle.flush()
    except Exception:
        logger.exception("Failed to acquire GitHub bridge lock at %s", lock_path)
        return True
    global _LOCK_HANDLE
    _LOCK_HANDLE = handle
    return True


def process_once() -> int:
    cfg = load_config()
    child = get_current_child(cfg)
    policy = get_github_submission_config(cfg)
    if not policy.get("enabled", False):
        logger.info("GitHub submission disabled in config.")
        return 0
    if policy.get("delivery_mode") != "issues":
        logger.info("GitHub delivery mode is %s; queue will not be delivered.", policy.get("delivery_mode"))
        return 0

    daily_cap = int(policy.get("daily_issue_cap", 4) or 4)
    submitted_today = load_submitted_count_for_day(child)
    if submitted_today >= daily_cap:
        logger.info("Daily GitHub issue cap reached (%s).", daily_cap)
        return 0

    seen_ids = load_completed_history_ids(child)
    entries = read_pending_entries(child, cfg=cfg, seen_ids=seen_ids)
    if not entries:
        return 0

    submitted = 0
    for entry in entries:
        entry_id = str(entry.get("id") or "").strip()
        if not entry_id:
            continue
        if entry.get("_stale"):
            archive_entry(child, entry, "stale")
            log_history(child, entry_id, "archived", reason="stale")
            continue
        if submitted_today >= daily_cap:
            break
        try:
            result = submit_issue(entry, cfg=cfg)
        except Exception as exc:
            logger.warning("GitHub issue submission failed for %s: %s", entry_id, exc)
            log_history(child, entry_id, "failed", reason=str(exc)[:400])
            continue
        archive_entry(
            child,
            entry,
            "submitted",
            issue_number=result.get("issue_number"),
            issue_url=result.get("issue_url"),
        )
        log_history(
            child,
            entry_id,
            "submitted",
            issue_number=result.get("issue_number"),
            issue_url=result.get("issue_url"),
            title=result.get("title"),
        )
        submitted += 1
        submitted_today += 1
        logger.info("Submitted %s to GitHub issue %s", entry_id, result.get("issue_url") or result.get("issue_number"))
    return submitted


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Deliver Ina's queued GitHub optimisation proposals.")
    parser.add_argument("--once", action="store_true", help="Process the queue once and exit.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    cfg = load_config()
    child = get_current_child(cfg)
    if not _acquire_single_instance_lock(child):
        logger.error("github_bridge already running; exiting duplicate instance.")
        return 1

    if args.once:
        process_once()
        return 0

    while True:
        try:
            process_once()
        except Exception:
            logger.exception("GitHub bridge loop failed.")
        cfg = load_config()
        policy = get_github_submission_config(cfg)
        time.sleep(float(policy.get("poll_interval_sec", 60.0) or 60.0))


if __name__ == "__main__":
    raise SystemExit(main())
