"""Bounded, explainable daily reporting for Ina's storage placement policy."""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from adaptive_storage import load_state
from github_submission import append_typed_outbox_notice, report_github_finding
from storage_layout import format_child_path, storage_layout

DEFAULT_POLICY = {"enabled": False, "interval_hours": 24, "max_directory_entries_sampled": 50000, "queue_github_issue": True, "default_detail_level": "abstract", "default_delivery": "github", "allow_private_discord": True, "preference_path": "AI_Children/{child}/memory/storage_migration_report_preferences.json", "state_path": "AI_Children/{child}/memory/storage_migration_report_state.json"}


def report_policy(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    policy = dict(DEFAULT_POLICY)
    raw = config.get("storage_migration_reporting") if isinstance(config, dict) else None
    if isinstance(raw, dict):
        policy.update(raw)
    return policy


def _state_path(child: str, policy: Dict[str, Any]) -> Path:
    return Path(str(policy["state_path"]).format(child=child))


def load_report_preferences(child: str, policy: Dict[str, Any]) -> Dict[str, str]:
    """Load Ina-controlled disclosure preferences; absence means public abstract."""
    path = Path(str(policy.get("preference_path", DEFAULT_POLICY["preference_path"])).format(child=child))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    detail = str(payload.get("detail_level") or policy.get("default_detail_level") or "abstract").lower()
    delivery = str(payload.get("delivery") or policy.get("default_delivery") or "github").lower()
    if detail not in {"abstract", "detailed", "private"}:
        detail = "abstract"
    if delivery not in {"github", "discord", "none"}:
        delivery = "github"
    if detail == "private":
        delivery = "discord"
    if delivery == "discord" and not bool(policy.get("allow_private_discord", True)):
        delivery, detail = "github", "abstract"
    return {"detail_level": detail, "delivery": delivery}


def _load_report_state(child: str, policy: Dict[str, Any]) -> Dict[str, Any]:
    try:
        payload = json.loads(_state_path(child, policy).read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_report_state(child: str, policy: Dict[str, Any], payload: Dict[str, Any]) -> None:
    path = _state_path(child, policy)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _recent_migrations(child: str, limit: int = 20, max_bytes: int = 65536) -> list[Dict[str, Any]]:
    path = Path("AI_Children") / child / "memory" / "storage_migration_history.jsonl"
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes), os.SEEK_SET)
            text = handle.read(max_bytes).decode("utf-8", errors="replace")
    except OSError:
        return []
    records = []
    for line in text.splitlines()[-max(1, int(limit)):]:
        try:
            item = json.loads(line)
        except Exception:
            continue
        if isinstance(item, dict):
            records.append(item)
    return records


def _directory_summary(path: Path, limit: int) -> Dict[str, Any]:
    result = {"path": str(path), "available": path.is_dir(), "files": 0, "directories": 0, "sample_truncated": False}
    if not result["available"]:
        return result
    try:
        with os.scandir(path) as entries:
            for index, entry in enumerate(entries):
                if index >= limit:
                    result["sample_truncated"] = True
                    break
                key = "directories" if entry.is_dir(follow_symlinks=False) else "files"
                result[key] += 1
    except OSError as exc:
        result.update(available=False, error=str(exc))
    return result


def build_daily_migration_report(child: str, config: Dict[str, Any], *, now: Optional[datetime] = None) -> Dict[str, Any]:
    stamp = now or datetime.now(timezone.utc)
    policy, adaptive, layout = report_policy(config), load_state(child, config), storage_layout(config)
    limit = max(1000, int(policy.get("max_directory_entries_sampled", 50000)))
    fragment_root = Path("AI_Children") / child / "memory" / "fragments"
    fast_root = format_child_path(layout.get("fast_runtime_root") or layout.get("fast_root"), child)
    directories = {"fragment_root": _directory_summary(fragment_root, limit), "fast_runtime": _directory_summary(fast_root, limit) if fast_root else {"available": False}}
    tiers = {tier: _directory_summary(fragment_root / tier, limit) for tier in ("short", "working", "long", "cold")}
    root_files = int(directories["fragment_root"].get("files") or 0)
    recommendations = [f"Reindex/rebalance {root_files} legacy root fragment file(s) into deterministic memory tiers."] if root_files else []
    decisions = adaptive.get("decisions") if isinstance(adaptive.get("decisions"), dict) else {}
    for artifact, decision in sorted(decisions.items()):
        if isinstance(decision, dict):
            recommendations.append(f"Keep future rebuildable {artifact} artifacts on {decision.get('tier', 'unknown')} (fast={decision.get('fast_score', 'n/a')}, durable={decision.get('durable_score', 'n/a')}).")
    migrations = _recent_migrations(child)
    return {"date": stamp.date().isoformat(), "generated_at": stamp.isoformat(), "child": child, "adaptive_state_updated_at": adaptive.get("updated_at"), "last_probe_at": adaptive.get("last_probe_at"), "decisions": decisions, "devices": adaptive.get("devices", {}), "directories": directories, "memory_tiers": tiers, "recent_migrations": migrations, "recommendations": recommendations, "safety": "recommendation_only; durable memories were not moved"}


def _render_abstract(report: Dict[str, Any]) -> str:
    decisions = report.get("decisions", {})
    tiers = sorted({str(item.get("tier") or "unknown") for item in decisions.values() if isinstance(item, dict)})
    organisation_attention = bool(report.get("directories", {}).get("fragment_root", {}).get("files", 0))
    migrations = report.get("recent_migrations") or []
    migration_health = "attention needed" if any(item.get("status") != "ok" or item.get("failed") or item.get("conflicts") for item in migrations) else ("verified" if migrations else "no recent recorded activity")
    lines = [
        f"Ina's daily storage migration summary for {report['date']}.", "",
        f"- Placement policy: {'active' if decisions else 'awaiting observations'}",
        f"- Rebuildable storage tiers currently in use: {', '.join(tiers) if tiers else 'not disclosed'}",
        f"- Organisation status: {'maintenance recommended' if organisation_attention else 'settled'}",
        f"- Recent migration verification: {migration_health}",
        "- No durable memories were moved by this report.", "",
        "Further internal detail is withheld by default. Ina may choose to explain more publicly or send a private report to her guardian via Discord.",
    ]
    return "\n".join(lines)


def _render_detailed(report: Dict[str, Any]) -> str:
    lines = [f"Daily storage placement and organisation report for **{report['child']}** on {report['date']}.", "", "## Placement decisions"]
    for artifact, decision in sorted(report.get("decisions", {}).items()):
        if isinstance(decision, dict):
            lines.append(f"- `{artifact}` → **{decision.get('tier', 'unknown')}** (fast score {decision.get('fast_score', 'n/a')}; durable score {decision.get('durable_score', 'n/a')})")
    lines.extend(["", "## Organisation", f"- Legacy files directly in fragment root: {report['directories']['fragment_root'].get('files', 0)}"] )
    lines.extend(f"- `{tier}`: {summary.get('files', 0)} direct files" for tier, summary in report.get("memory_tiers", {}).items())
    lines.extend(["", "## Recent migration outcomes"] )
    migrations = report.get("recent_migrations") or []
    if migrations:
        for item in migrations:
            lines.append(f"- `{item.get('operation', 'unknown')}`: {item.get('status', 'unknown')} (verified={item.get('verified', 0)}, failed={item.get('failed', 0)}, conflicts={item.get('conflicts', 0)}, rolled_back={item.get('rolled_back', False)})")
    else:
        lines.append("- No recorded migration executions.")
    lines.extend(["", "## Recommendations"] )
    lines.extend(f"- {item}" for item in (report.get("recommendations") or ["No placement change recommended."]))
    lines.extend(["", f"Safety: `{report.get('safety')}`"] )
    return "\n".join(lines)


def maybe_queue_daily_migration_report(child: str, config: Dict[str, Any], *, now: Optional[datetime] = None, force: bool = False) -> Dict[str, Any]:
    policy = report_policy(config)
    if not bool(policy.get("enabled", False)):
        return {"queued": False, "reason": "disabled"}
    stamp, prior = now or datetime.now(timezone.utc), _load_report_state(child, policy)
    try:
        last = datetime.fromisoformat(str(prior.get("last_report_at") or "").replace("Z", "+00:00"))
    except Exception:
        last = None
    if not force and last and stamp - last.astimezone(timezone.utc) < timedelta(hours=max(1.0, float(policy.get("interval_hours", 24)))):
        return {"queued": False, "reason": "interval", "last_report_at": last.isoformat()}
    report = build_daily_migration_report(child, config, now=stamp)
    preference = load_report_preferences(child, policy)
    detail, delivery = preference["detail_level"], preference["delivery"]
    result = {"queued": False, "reason": "delivery_disabled", "delivery": delivery, "detail_level": detail}
    if delivery == "github" and bool(policy.get("queue_github_issue", True)):
        body = _render_detailed(report) if detail == "detailed" else _render_abstract(report)
        public_metadata = {"source": "daily_storage_migration_report", "disclosure": detail}
        result = report_github_finding(child, f"Daily storage migration report — {report['date']}", body, kind="issue", component="adaptive_storage", severity="low", confidence=1.0, evidence=[] if detail == "abstract" else [f"adaptive state updated at {report.get('adaptive_state_updated_at')}"], suggestion="Ina may disclose more publicly or route detail privately through Discord.", touched_files=[] if detail == "abstract" else ["adaptive_storage.py", "storage_layout.py", "memory_graph.py"], dedupe_key=f"daily-storage-migration:{report['date']}", metadata=public_metadata, cfg=config)
        result.update(delivery="github", detail_level=detail)
    elif delivery == "discord":
        entry_id = append_typed_outbox_notice(child, _render_detailed(report), target="owner_dm", metadata={"source": "daily_storage_migration_report", "privacy": "private", "chosen_by": "ina_preference"})
        result = {"queued": bool(entry_id), "entry_id": entry_id, "reason": "queued_private" if entry_id else "discord_queue_failed", "delivery": "discord", "detail_level": "private"}
    if result.get("queued"):
        _save_report_state(child, policy, {"last_report_at": stamp.isoformat(), "last_report_date": report["date"], "entry_id": result.get("entry_id"), "delivery": result.get("delivery"), "detail_level": result.get("detail_level")})
    result["report"] = report
    return result


__all__ = ["build_daily_migration_report", "load_report_preferences", "maybe_queue_daily_migration_report", "report_policy"]
