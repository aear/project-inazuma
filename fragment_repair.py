from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from io_utils import atomic_write_json


_ACTION_LOG_LIMIT = 20


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def _memory_root(child: str) -> Path:
    return Path("AI_Children") / child / "memory"


def _fragments_root(child: str) -> Path:
    return _memory_root(child) / "fragments"


def _candidate_dirs(child: str) -> List[Path]:
    root = _fragments_root(child)
    return [
        root,
        root / "pending",
        root / "cold",
    ]


def _resolve_fragment_path(child: str, entry: Dict[str, Any]) -> Optional[Path]:
    root = _fragments_root(child)
    path_value = entry.get("path")
    if path_value:
        try:
            candidate = Path(path_value).expanduser()
            if not candidate.is_absolute():
                candidate = (Path.cwd() / candidate).resolve()
            if candidate.exists() and candidate.is_file() and _is_relative_to(candidate, root):
                return candidate
        except Exception:
            pass

    filename = entry.get("file") or entry.get("filename") or entry.get("id")
    if not filename:
        return None
    filename = str(filename)
    if not filename.endswith(".json"):
        filename = f"{filename}.json"

    for base in _candidate_dirs(child):
        candidate = base / filename
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _attempt_salvage(path: Path, max_bytes: int) -> Tuple[str, Optional[dict]]:
    try:
        size = path.stat().st_size
    except Exception:
        return "stat_failed", None

    if size == 0:
        return "empty", None
    if max_bytes > 0 and size > max_bytes:
        return "too_large", None

    try:
        data = path.read_text(encoding="utf-8")
    except Exception:
        return "decode_failed", None

    try:
        obj = json.loads(data)
        return "valid", obj
    except json.JSONDecodeError:
        try:
            obj, idx = json.JSONDecoder().raw_decode(data)
        except Exception:
            return "invalid", None
        if data[idx:].strip():
            return "invalid", None
        return "salvaged", obj
    except Exception:
        return "invalid", None


def _safe_move(path: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / path.name
    if dest.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        dest = dest_dir / f"{path.stem}_{stamp}{path.suffix}"
    path.replace(dest)
    return dest


def _append_action_log(child: str, payload: Dict[str, Any]) -> None:
    log_path = _memory_root(child) / "fragment_repair_log.jsonl"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def process_corrupt_queue(
    child: str,
    queue: List[Dict[str, Any]],
    policy: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    mode = str(policy.get("mode") or "inspect").lower()
    max_actions = int(policy.get("max_actions_per_pass") or 0)
    max_actions = max(1, max_actions)
    max_repair_bytes = int(policy.get("max_repair_bytes") or 0)
    def _process() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        memory_root = _memory_root(child)
        fragments_root = _fragments_root(child)
        quarantine_dir = policy.get("quarantine_dir") or "fragments/corrupt"
        pending_delete_dir = policy.get("pending_delete_dir") or "fragments/pending_delete"
        quarantine_path = Path(quarantine_dir)
        if not quarantine_path.is_absolute():
            quarantine_path = memory_root / quarantine_path
        pending_delete_path = Path(pending_delete_dir)
        if not pending_delete_path.is_absolute():
            pending_delete_path = memory_root / pending_delete_path

        summary = {
            "child": child,
            "started_at": _now_iso(),
            "mode": mode,
            "actions": [],
            "counts": {
                "processed": 0,
                "missing": 0,
                "skipped": 0,
                "repaired": 0,
                "quarantined": 0,
                "deleted": 0,
                "valid": 0,
                "invalid": 0,
            },
        }

        remaining: List[Dict[str, Any]] = []
        actions_taken = 0

        for entry in queue:
            if actions_taken >= max_actions:
                remaining.append(entry)
                continue

            path = _resolve_fragment_path(child, entry)
            if not path:
                summary["counts"]["missing"] += 1
                continue

            if not _is_relative_to(path, fragments_root):
                summary["counts"]["skipped"] += 1
                continue

            if mode == "inspect":
                remaining.append(entry)
                summary["counts"]["skipped"] += 1
                continue

            salvage_status = None
            salvage_obj = None
            if mode == "repair":
                salvage_status, salvage_obj = _attempt_salvage(path, max_repair_bytes)
                if salvage_status == "valid":
                    summary["counts"]["valid"] += 1
                    continue
                if salvage_status == "salvaged" and salvage_obj is not None:
                    try:
                        atomic_write_json(path, salvage_obj, indent=2, ensure_ascii=True)
                        summary["counts"]["repaired"] += 1
                        actions_taken += 1
                        summary["actions"].append(
                            {
                                "action": "repaired",
                                "path": str(path),
                                "timestamp": _now_iso(),
                            }
                        )
                        if len(summary["actions"]) > _ACTION_LOG_LIMIT:
                            summary["actions"] = summary["actions"][-_ACTION_LOG_LIMIT:]
                        _append_action_log(
                            child,
                            {
                                "action": "repaired",
                                "path": str(path),
                                "timestamp": _now_iso(),
                                "note": "json salvage rewrite",
                            },
                        )
                        continue
                    except Exception:
                        salvage_status = "invalid"
                if salvage_status not in {"valid", "salvaged"}:
                    summary["counts"]["invalid"] += 1

            target_dir = quarantine_path if mode != "delete" else pending_delete_path
            try:
                dest = _safe_move(path, target_dir)
                action_label = "quarantined" if mode != "delete" else "deleted"
                summary["counts"][action_label] += 1
                actions_taken += 1
                summary["actions"].append(
                    {
                        "action": action_label,
                        "path": str(path),
                        "dest": str(dest),
                        "timestamp": _now_iso(),
                        "reason": entry.get("reason") or salvage_status or "invalid",
                    }
                )
                if len(summary["actions"]) > _ACTION_LOG_LIMIT:
                    summary["actions"] = summary["actions"][-_ACTION_LOG_LIMIT:]
                _append_action_log(
                    child,
                    {
                        "action": action_label,
                        "path": str(path),
                        "dest": str(dest),
                        "timestamp": _now_iso(),
                        "reason": entry.get("reason") or salvage_status or "invalid",
                    },
                )
            except Exception:
                remaining.append(entry)
                summary["counts"]["skipped"] += 1

            summary["counts"]["processed"] += 1

        summary["remaining"] = len(remaining)
        summary["finished_at"] = _now_iso()
        return remaining, summary

    try:
        from precision_requests import precision_request
    except Exception:
        precision_request = None

    if precision_request:
        with precision_request(
            task="corruption_repair",
            child=child,
            ttl_sec=6.0,
            reason=f"fragment_repair:{mode}",
            integrity_threat=True,
            source="fragment_repair",
        ):
            return _process()

    return _process()
