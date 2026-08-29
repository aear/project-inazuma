"""Bounded Discord speaker provenance and acoustic recognition evidence."""
from __future__ import annotations

from array import array
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any

from io_utils import atomic_write_json


PROFILE_VERSION = 1
SIGNATURE_BINS = 16
MAX_PROFILES = 64
MAX_OBSERVATIONS_PER_PROFILE = 256


def conversation_urge_influence(
    evidence: Any,
    *,
    now: float,
    curiosity: float,
    isolation: float,
) -> dict[str, Any]:
    """Translate fresh conversation evidence into a bounded, signed voice drive."""
    if not isinstance(evidence, dict):
        return {"adjustment": 0.0, "reason": "no_conversation_evidence"}
    raw_timestamp = evidence.get("timestamp")
    try:
        if isinstance(raw_timestamp, (int, float)):
            observed_at = float(raw_timestamp)
        else:
            observed_at = datetime.fromisoformat(str(raw_timestamp).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return {"adjustment": 0.0, "reason": "invalid_conversation_timestamp"}
    age_seconds = max(0.0, now - observed_at)
    if age_seconds > 120.0:
        return {"adjustment": 0.0, "reason": "stale_conversation", "age_seconds": round(age_seconds, 3)}

    speakers = evidence.get("speakers") if isinstance(evidence.get("speakers"), list) else []
    human_speakers = [row for row in speakers if isinstance(row, dict) and not row.get("is_bot")]
    if not human_speakers:
        return {"adjustment": 0.0, "reason": "no_human_speaker", "age_seconds": round(age_seconds, 3)}
    voiced_seconds = max(0.0, float(evidence.get("voiced_seconds", 0.0) or 0.0))
    engagement = min(1.0, voiced_seconds / 8.0)
    novelty = max(0.0, min(1.0, max(float(row.get("novelty", 0.0) or 0.0) for row in human_speakers)))
    group_size = len({str(row.get("discord_user_id") or "") for row in human_speakers})
    concurrency_load = max(0.0, min(1.0, float(evidence.get("concurrency_load", 0.0) or 0.0)))
    freshness = max(0.0, 1.0 - (age_seconds / 120.0))

    # Active, novel conversation and curiosity create room for a reply. Familiar
    # company can also satisfy some isolation-driven pressure. Neither outcome
    # is treated as reward, and a small deadband preserves a neutral result.
    group_invitation = min(0.045, max(0, group_size - 1) * 0.015) * curiosity
    reply_invitation = engagement * freshness * (
        0.035 + 0.085 * curiosity + 0.08 * novelty + group_invitation
    )
    presence_relief = engagement * freshness * (0.09 * isolation * (1.0 - novelty))
    turn_uncertainty = engagement * freshness * 0.05 * concurrency_load
    raw_adjustment = max(-0.12, min(0.18, reply_invitation - presence_relief - turn_uncertainty))
    adjustment = 0.0 if abs(raw_adjustment) < 0.015 else raw_adjustment
    reason = "neutral_conversation"
    if adjustment > 0.0:
        reason = "reply_invitation"
    elif adjustment < 0.0:
        reason = "social_presence_relief"
    return {
        "adjustment": round(adjustment, 6),
        "reason": reason,
        "age_seconds": round(age_seconds, 3),
        "engagement": round(engagement, 6),
        "novelty": round(novelty, 6),
        "group_size": group_size,
        "concurrency_load": round(concurrency_load, 6),
        "reply_invitation": round(reply_invitation, 6),
        "presence_relief": round(presence_relief, 6),
        "turn_uncertainty": round(turn_uncertainty, 6),
        "identity_basis": "discord_user_id",
    }


def pcm_s16le_signature(pcm_bytes: bytes, *, bins: int = SIGNATURE_BINS) -> list[float]:
    """Return a gain-resistant temporal/timbre sketch from bounded PCM bytes."""
    usable = len(pcm_bytes) - (len(pcm_bytes) % 2)
    if usable < 2:
        return []
    samples = array("h")
    samples.frombytes(pcm_bytes[:usable])
    if not samples:
        return []
    if samples.itemsize == 2 and __import__("sys").byteorder != "little":
        samples.byteswap()
    values = [float(value) / 32768.0 for value in samples]
    peak = max(max(abs(value) for value in values), 1e-6)
    normalized = [value / peak for value in values]
    width = max(1, math.ceil(len(normalized) / max(1, bins)))
    envelope = []
    for start in range(0, len(normalized), width):
        chunk = normalized[start : start + width]
        envelope.append(math.sqrt(sum(value * value for value in chunk) / len(chunk)))
        if len(envelope) >= bins:
            break
    envelope.extend([0.0] * (bins - len(envelope)))
    zero_cross = sum(
        1 for left, right in zip(normalized, normalized[1:])
        if (left < 0.0 <= right) or (right < 0.0 <= left)
    ) / max(1, len(normalized) - 1)
    mean_abs = sum(abs(value) for value in normalized) / len(normalized)
    return [round(value, 6) for value in envelope + [zero_cross, mean_abs]]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


def _safe_profiles(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError):
        return {}


def observe_discord_voice(
    path: Path,
    *,
    discord_user_id: str,
    display_name: str,
    signature: list[float],
    observed_at: str,
    guild_id: str | None,
    channel_id: str | None,
) -> dict[str, Any]:
    """Merge one Discord-attributed sample into a bounded acoustic profile."""
    user_id = str(discord_user_id or "").strip()
    if not user_id or not signature:
        return {"status": "skipped", "reason": "missing_identity_or_signature"}
    payload = _safe_profiles(path)
    profiles = payload.get("profiles") if isinstance(payload.get("profiles"), dict) else {}
    current = profiles.get(user_id) if isinstance(profiles.get(user_id), dict) else {}
    old = current.get("centroid") if isinstance(current.get("centroid"), list) else []
    count = max(0, int(current.get("observations", 0) or 0))
    weight = min(count, MAX_OBSERVATIONS_PER_PROFILE - 1)
    if len(old) == len(signature):
        centroid = [((old_value * weight) + new_value) / (weight + 1) for old_value, new_value in zip(old, signature)]
    else:
        centroid = list(signature)
    profiles[user_id] = {
        "discord_user_id": user_id,
        "display_name": str(display_name or user_id)[:128],
        "centroid": [round(float(value), 6) for value in centroid],
        "observations": min(MAX_OBSERVATIONS_PER_PROFILE, count + 1),
        "last_observed_at": observed_at,
        "last_guild_id": guild_id,
        "last_channel_id": channel_id,
        "identity_basis": "discord_user_id",
        "acoustic_role": "supporting_recognition_evidence",
    }
    if len(profiles) > MAX_PROFILES:
        ordered = sorted(profiles.items(), key=lambda item: str(item[1].get("last_observed_at") or ""), reverse=True)
        profiles = dict(ordered[:MAX_PROFILES])
    result = {"version": PROFILE_VERSION, "profiles": profiles}
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, result, indent=2, ensure_ascii=False)
    return profiles[user_id]


def recognition_candidates(path: Path, signature: list[float], *, limit: int = 3) -> list[dict[str, Any]]:
    """Return acoustic candidates without turning similarity into identity."""
    profiles = _safe_profiles(path).get("profiles")
    if not isinstance(profiles, dict) or not signature:
        return []
    rows = []
    for user_id, profile in profiles.items():
        if not isinstance(profile, dict):
            continue
        score = cosine_similarity(signature, profile.get("centroid") or [])
        rows.append({
            "discord_user_id": str(user_id),
            "display_name": str(profile.get("display_name") or user_id),
            "similarity": round(score, 6),
            "observations": int(profile.get("observations", 0) or 0),
            "status": "candidate_only",
        })
    rows.sort(key=lambda row: (-row["similarity"], row["discord_user_id"]))
    return rows[: max(1, min(8, int(limit)))]


__all__ = [
    "conversation_urge_influence", "observe_discord_voice",
    "pcm_s16le_signature", "recognition_candidates",
]
