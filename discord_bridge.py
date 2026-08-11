# discord_bridge.py

from __future__ import annotations

import json
import logging
import ast
import math
import os
import re
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional
import asyncio
import difflib
try:
    import fcntl  # type: ignore
except Exception:
    fcntl = None  # type: ignore

try:
    import discord
except ModuleNotFoundError as exc:
    if exc.name == "audioop":
        raise RuntimeError(
            "Python 3.13 removed the stdlib audioop module. Install dependencies from "
            "requirements.txt so audioop-lts is available for Discord voice support."
        ) from exc
    if exc.name == "discord":
        raise RuntimeError(
            "Discord support is not installed. Run `python -m pip install -r requirements.txt` "
            "to install py-cord[voice] and its voice dependencies."
        ) from exc
    raise

from comms_core import CommsCore, CommsResponse, load_secret
from conversation_scene import scene_with_memory_consideration
from backend_discord import (
    make_sender_info_from_discord,
    make_channel_info_from_discord,
    register_discord_backend,
)
from social_map import (
    get_owner_user_id,
    is_owner_friend,
    is_high_trust,
    get_high_trust_contacts,
    record_dm_attempt,
    update_social_entry,
)
from language_processing import (
    build_dual_symbolic_message,
    generate_symbolic_reply_from_text,
    load_generated_symbols,
    score_text_mirroring,
    select_symbolic_message_text,
)
from simple_image_fallback import ImageFallbackError, extract_image_features
from vector_math import cosine_similarity as visual_cosine_similarity
from visual_token_learning import observe_image as observe_visual_tokens
from visual_token_learning import observe_words as observe_visual_words
from live_experience_bridge import LiveExperienceBridge
from model_manager import get_inastate, update_inastate
from text_memory import (
    build_text_symbol_links,
    review_text_evidence,
)
from io_pressure import pressure_signal
from music_delivery import ensure_opus_sidecar
from discord_runtime import (
    load_root_config as load_layered_root_config,
    typed_outbox_path,
    typed_outbox_history_path,
    typed_outbox_archive_path,
)
try:
    from lm_studio_adapter import LMStudioAdapter
except Exception:
    LMStudioAdapter = None  # type: ignore
try:
    from PIL import Image
except Exception:
    Image = None  # type: ignore
try:
    import numpy as np
except Exception:
    np = None  # type: ignore
try:
    from transformers.fractal_multidimensional_transformers import FractalTransformer
except Exception:
    FractalTransformer = None  # type: ignore
try:
    from fragment_limits import get_memory_guard_level, should_accept_fragment  # type: ignore
except Exception:
    def get_memory_guard_level():  # type: ignore[redefinition]
        return "unknown"

    def should_accept_fragment(*args, **kwargs):  # type: ignore[redefinition]
        return True, "limits_unavailable"

# ---------------------------------------------------------------------------
# Basic logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("discord_bridge")
CONFIG_PATH = Path("config.json")
_CHAT_ADAPTER = None
IMAGE_ATTACHMENT_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
IMAGE_ATTACHMENT_MIME_MAP = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
}
DEFAULT_IMAGE_ATTACHMENT_MAX_BYTES = 25 * 1024 * 1024
DEFAULT_IMAGE_ATTACHMENT_MAX_COUNT = 4
DEFAULT_SONG_ATTACHMENT_MAX_BYTES = 25 * 1024 * 1024
AUDIO_ATTACHMENT_EXTENSIONS = {".wav", ".mp3", ".ogg", ".opus", ".flac", ".m4a", ".aac"}
DEFAULT_DISCORD_SEND_INTERVAL_SECONDS = 0.35
DEFAULT_DISCORD_RATE_LIMIT_PADDING_SECONDS = 0.25
DEFAULT_DISCORD_SEND_RETRIES = 3
DEFAULT_MAX_REPLY_SYMBOLS = 24  # attention ceiling, not a sentence target
DISCORD_MESSAGE_CHAR_LIMIT = 2000

_DISCORD_BRIDGE_LOCK_HANDLE = None


def split_discord_message(text: str, limit: int = DISCORD_MESSAGE_CHAR_LIMIT) -> list[str]:
    """Split only for transport, preferring paragraph and sentence boundaries."""
    payload = "" if text is None else str(text)
    try:
        bounded_limit = max(1, min(DISCORD_MESSAGE_CHAR_LIMIT, int(limit)))
    except (TypeError, ValueError):
        bounded_limit = DISCORD_MESSAGE_CHAR_LIMIT
    if len(payload) <= bounded_limit:
        return [payload]

    chunks = []
    remaining = payload
    while len(remaining) > bounded_limit:
        window = remaining[: bounded_limit + 1]
        useful_floor = max(1, bounded_limit // 3)
        cut = 0

        paragraph_break = window.rfind("\n\n", 0, bounded_limit + 1)
        if paragraph_break >= useful_floor:
            cut = paragraph_break + 2
        if not cut:
            line_break = window.rfind("\n", 0, bounded_limit + 1)
            if line_break >= useful_floor:
                cut = line_break + 1
        if not cut:
            sentence_breaks = [
                match.end()
                for match in re.finditer(r"""[.!?](?:["')\]]*)\s+""", window)
                if match.end() <= bounded_limit
            ]
            if sentence_breaks and sentence_breaks[-1] >= useful_floor:
                cut = sentence_breaks[-1]
        if not cut:
            whitespace = max(
                window.rfind(" ", 0, bounded_limit + 1),
                window.rfind("\t", 0, bounded_limit + 1),
            )
            if whitespace > 0:
                cut = whitespace + 1
        if not cut:
            cut = bounded_limit

        chunk = remaining[:cut]
        if not chunk:
            chunk = remaining[:bounded_limit]
            cut = bounded_limit
        chunks.append(chunk)
        remaining = remaining[cut:]

    if remaining:
        chunks.append(remaining)
    return chunks


def _acquire_single_instance_lock() -> bool:
    if fcntl is None:
        return True
    child = get_current_child()
    lock_path = Path("AI_Children") / child / "memory" / "discord_bridge.lock"
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
        logger.exception("Failed to acquire discord bridge lock at %s", lock_path)
        return True
    global _DISCORD_BRIDGE_LOCK_HANDLE
    _DISCORD_BRIDGE_LOCK_HANDLE = handle
    return True


def _load_discord_sinks():
    """
    Attempt to import Discord voice sinks, falling back to the legacy
    discord.ext.voice_recv extension if present. Emits a targeted warning with
    install guidance if neither is available.
    """
    version = getattr(discord, "__version__", "unknown")
    try:
        from discord import sinks as discord_sinks  # type: ignore
        return discord_sinks
    except Exception as first_exc:
        try:
            from discord.ext import voice_recv as voice_sinks  # type: ignore
        except Exception as exc:
            logger.warning(
                "discord voice receive modules not available (discord.py %s); voice capture disabled. "
                "Install py-cord[voice] (or discord-ext-voice-recv) to enable discord.sinks. import errors: %s / %s",
                version,
                first_exc,
                exc,
            )
            return None
        logger.info(
            "Loaded discord voice sinks from discord.ext.voice_recv extension (discord.py %s).",
            version,
        )
        return voice_sinks


sinks = _load_discord_sinks()


def log_discord_voice_capabilities():
    """Emit a one-time info log about discord voice support to aid debugging."""
    version = getattr(discord, "__version__", "unknown")
    sink_path = getattr(sinks, "__file__", None) if sinks else None
    has_start_recording = hasattr(getattr(discord, "VoiceClient", None), "start_recording")
    logger.info(
        "Discord voice capabilities: version=%s sinks=%s start_recording=%s sink_path=%s",
        version,
        bool(sinks),
        has_start_recording,
        sink_path,
    )


def _install_voice_debug_hooks():
    """
    Add lightweight logging on voice state/server updates to track session/token details.
    """
    try:
        vc_cls = discord.VoiceClient
    except Exception:
        return
    if getattr(vc_cls, "_ina_voice_hooks", False):
        return

    vc_cls._ina_voice_hooks = True

    orig_vs = vc_cls.on_voice_state_update
    orig_vserv = vc_cls.on_voice_server_update

    async def wrapped_vs(self, data, *args, **kwargs):
        logger.info(
            "Voice state update: session_id=%s channel_id=%s handshaking=%s reconnecting=%s",
            data.get("session_id"),
            data.get("channel_id"),
            getattr(self, "_handshaking", None),
            getattr(self, "_potentially_reconnecting", None),
        )
        return await orig_vs(self, data, *args, **kwargs)

    async def wrapped_vserv(self, data, *args, **kwargs):
        logger.info(
            "Voice server update: token_present=%s endpoint=%s guild_id=%s",
            bool(data.get("token")),
            data.get("endpoint"),
            data.get("guild_id"),
        )
        return await orig_vserv(self, data, *args, **kwargs)

    vc_cls.on_voice_state_update = wrapped_vs  # type: ignore
    vc_cls.on_voice_server_update = wrapped_vserv  # type: ignore


def load_root_config() -> dict:
    """Return the cached compatibility base plus owned config layers."""
    return load_layered_root_config(CONFIG_PATH)


def get_discord_config() -> dict:
    """Return the effective Discord policy block, or an empty dict."""
    cfg = load_root_config()
    section = cfg.get("discord") if isinstance(cfg, dict) else None
    return section if isinstance(section, dict) else {}


def get_language_review_policy() -> dict:
    """Return bounded, event-driven Discord language review settings."""
    raw = get_discord_config().get("language_review") or {}
    raw = raw if isinstance(raw, dict) else {}

    def positive_int(key: str, default: int, minimum: int = 1) -> int:
        try:
            return max(minimum, int(raw.get(key, default)))
        except (TypeError, ValueError):
            return default

    def nonnegative_float(key: str, default: float) -> float:
        try:
            return max(0.0, float(raw.get(key, default)))
        except (TypeError, ValueError):
            return default

    return {
        "enabled": raw.get("enabled", True) is not False,
        "pressure_messages": positive_int("pressure_messages", 24, 4),
        "history_limit": positive_int("history_limit", 75, 10),
        "mapping_batch": positive_int("mapping_batch", 160, 1),
        "revisit_mappings": positive_int("revisit_mappings", 24, 1),
        "cooldown_seconds": nonnegative_float("cooldown_seconds", 900.0),
    }


_NATIVE_HISTORY_LINE = re.compile(r"(?im)^\s*Native:\s*(.+?)\s*$")
_ENGLISH_HISTORY_LINE = re.compile(
    r"(?im)^\s*(?:Human guess|Word-for-word|English|Translation):\s*(.+?)\s*$"
)


def parse_symbolic_history_message(text: str) -> dict:
    """Pair each Native line only with its following human-gloss line."""
    pairs = []
    rejections = []
    pending_native = None
    for line_number, line in enumerate(str(text or "").splitlines(), start=1):
        native_match = _NATIVE_HISTORY_LINE.match(line)
        if native_match:
            if pending_native is not None:
                rejections.append(
                    {
                        "reason": "orphan_native",
                        "line": pending_native["line"],
                    }
                )
            pending_native = {
                "text": native_match.group(1).strip(),
                "line": line_number,
            }
            continue

        english_match = _ENGLISH_HISTORY_LINE.match(line)
        if not english_match:
            continue
        if pending_native is None:
            rejections.append(
                {
                    "reason": "orphan_human_guess",
                    "line": line_number,
                }
            )
            continue
        english_text = english_match.group(1).strip()
        if pending_native["text"] and english_text:
            pairs.append((pending_native["text"], english_text))
        pending_native = None

    if pending_native is not None:
        rejections.append(
            {
                "reason": "orphan_native",
                "line": pending_native["line"],
            }
        )
    rejection_counts = {}
    for rejection in rejections:
        reason = rejection["reason"]
        rejection_counts[reason] = int(rejection_counts.get(reason, 0)) + 1
    return {
        "pairs": pairs,
        "rejections": rejections,
        "rejection_counts": rejection_counts,
    }


def extract_symbolic_history_alignments(text: str) -> list[tuple[str, str]]:
    """Extract safely paired labelled native/English lines."""
    return parse_symbolic_history_message(text)["pairs"]


def format_history_review_result(result: dict) -> str:
    """Summarize reviewed work without making an empty unseen count look idle."""
    payload = result if isinstance(result, dict) else {}
    message = (
        "History review complete: "
        f"{int(payload.get('revisited_messages', 0) or 0)} revisited, "
        f"{int(payload.get('new_messages', 0) or 0)} unseen, "
        f"{int(payload.get('alignment_candidates', 0) or 0)} candidates reconsidered, "
        f"{int(payload.get('alignments', 0) or 0)} mapping changes."
    )
    rejections = payload.get("alignment_rejection_counts")
    if isinstance(rejections, dict) and rejections:
        detail = ", ".join(
            f"{reason}: {int(count or 0)}" for reason, count in sorted(rejections.items())
        )
        message += f" Rejections: {detail}."
    return message


def get_voice_io_config() -> dict:
    """
    Return voice input settings with sensible defaults.
    """
    cfg = load_root_config()
    child = cfg.get("current_child", "Inazuma_Yagami") if isinstance(cfg, dict) else "Inazuma_Yagami"
    discord_cfg = cfg.get("discord") if isinstance(cfg, dict) else None
    voice_cfg = discord_cfg if isinstance(discord_cfg, dict) else {}
    return {
        "voice_label": voice_cfg.get("voice_label", "discord_voice"),
        "voice_pipe_path": voice_cfg.get("voice_pipe_path"),
        "voice_buffer_dir": voice_cfg.get("voice_buffer_dir")
        or str(Path("AI_Children") / child / "memory" / "discord_voice"),
        "voice_chunk_seconds": max(5, int(voice_cfg.get("voice_chunk_seconds", 15) or 15)),
    }


def _coerce_nonnegative_float(value, default: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        return default
    return parsed if parsed >= 0.0 else default


def _coerce_positive_int(value, default: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        return default
    return parsed if parsed > 0 else default


def infer_message_edit(original: str, edited: str) -> dict:
    """Return an inspectable, non-psychological inference about an edit."""
    before = original or ""
    after = edited or ""
    before_words, after_words = before.split(), after.split()
    matcher = difflib.SequenceMatcher(a=before_words, b=after_words)
    added, removed = [], []
    for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
        if opcode in {"insert", "replace"}:
            added.extend(after_words[j1:j2])
        if opcode in {"delete", "replace"}:
            removed.extend(before_words[i1:i2])
    if not before.strip() and after.strip():
        likely_reason = "added_missing_content"
    elif before.strip() and not after.strip():
        likely_reason = "removed_content"
    elif added and not removed:
        likely_reason = "expanded_or_clarified"
    elif removed and not added:
        likely_reason = "shortened_or_retracted"
    elif before.casefold() == after.casefold():
        likely_reason = "capitalization_or_formatting"
    elif re.sub(r"\W+", "", before).casefold() == re.sub(r"\W+", "", after).casefold():
        likely_reason = "punctuation_or_formatting"
    else:
        likely_reason = "reworded_or_corrected"
    return {
        "original": before, "edited": after,
        "added_words": added[:32], "removed_words": removed[:32],
        "similarity": round(matcher.ratio(), 4),
        "likely_reason": likely_reason,
        "inference_kind": "surface_change_heuristic",
    }


def get_outbox_policy() -> dict:
    cfg = get_discord_config()
    policy = cfg.get("outbox_policy") if isinstance(cfg, dict) else None
    defaults = {
        "max_burst": 5,
        "max_age_minutes": 5.0,
        "archive_path": None,
        "flush_burst": 24,
        "flush_stale_mode": "drop",
        "min_send_interval_seconds": DEFAULT_DISCORD_SEND_INTERVAL_SECONDS,
        "rate_limit_padding_seconds": DEFAULT_DISCORD_RATE_LIMIT_PADDING_SECONDS,
        "max_send_retries": DEFAULT_DISCORD_SEND_RETRIES,
    }
    if not isinstance(policy, dict):
        return defaults
    result = defaults.copy()
    if policy.get("max_burst") is not None:
        try:
            result["max_burst"] = max(0, int(policy["max_burst"]))
        except Exception:
            logger.warning("Invalid discord.outbox_policy.max_burst value; using default %s", defaults["max_burst"])
    if policy.get("flush_burst") is not None:
        try:
            result["flush_burst"] = max(1, int(policy["flush_burst"]))
        except Exception:
            logger.warning("Invalid discord.outbox_policy.flush_burst value; using default %s", defaults["flush_burst"])
    if policy.get("max_age_minutes") is not None:
        try:
            result["max_age_minutes"] = max(0.0, float(policy["max_age_minutes"]))
        except Exception:
            logger.warning(
                "Invalid discord.outbox_policy.max_age_minutes; using default %s", defaults["max_age_minutes"]
            )
    stale_mode = str(policy.get("flush_stale_mode") or defaults["flush_stale_mode"]).strip().lower() or defaults["flush_stale_mode"]
    result["flush_stale_mode"] = stale_mode if stale_mode in {"drop", "archive"} else defaults["flush_stale_mode"]
    interval_raw = policy.get("min_send_interval_seconds", policy.get("send_interval_seconds"))
    if interval_raw is not None:
        result["min_send_interval_seconds"] = _coerce_nonnegative_float(
            interval_raw,
            defaults["min_send_interval_seconds"],
        )
    padding_raw = policy.get("rate_limit_padding_seconds")
    if padding_raw is not None:
        result["rate_limit_padding_seconds"] = _coerce_nonnegative_float(
            padding_raw,
            defaults["rate_limit_padding_seconds"],
        )
    retry_raw = policy.get("max_send_retries")
    if retry_raw is not None:
        result["max_send_retries"] = _coerce_positive_int(
            retry_raw,
            defaults["max_send_retries"],
        )
    archive_path = policy.get("archive_path")
    if archive_path:
        result["archive_path"] = str(archive_path)
    return result


def get_current_child() -> str:
    cfg = load_root_config()
    return cfg.get("current_child", "Inazuma_Yagami") if isinstance(cfg, dict) else "Inazuma_Yagami"


def _resolve_adjusted_urge_level(state: object) -> float:
    if not isinstance(state, dict):
        return 0.0
    try:
        base = float(state.get("level", 0.0))
    except Exception:
        base = 0.0
    try:
        adjusted = float(state.get("adjusted_level", base))
    except Exception:
        adjusted = base
    return max(0.0, min(1.0, adjusted))


def _emotion_values_from_state(state: object) -> dict[str, float]:
    if not isinstance(state, dict):
        return {}
    snapshot = state.get("emotion_snapshot")
    values = snapshot.get("values") if isinstance(snapshot, dict) else None
    if not isinstance(values, dict):
        return {}
    cleaned = {}
    for key, raw_value in values.items():
        name = str(key or "").strip()
        if not name or name.startswith("_"):
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            cleaned[name] = max(-1.0, min(1.0, value))
    return cleaned


def format_emotion_signal(state: object, max_items: Optional[int] = None) -> Optional[dict]:
    """Render a bounded, inspectable subset of Ina's current emotion sliders."""
    values = _emotion_values_from_state(state)
    if not values:
        return None
    intent = state.get("text_expression_intent") if isinstance(state, dict) else None
    intent = intent if isinstance(intent, dict) else {}
    requested = max_items if max_items is not None else intent.get("max_emotion_sliders")
    if requested is None and isinstance(state, dict):
        requested = state.get("emotion_signal_limit", 6)
    try:
        limit = max(1, min(24, int(requested or 6)))
    except (TypeError, ValueError):
        limit = 6
    ranked = sorted(values.items(), key=lambda item: (-abs(item[1]), item[0]))
    selected = ranked[:limit]

    def level(value: float) -> str:
        if value >= 0.67:
            return "very high"
        if value >= 0.25:
            return "raised"
        if value > -0.25:
            return "near neutral"
        if value > -0.67:
            return "low"
        return "very low"

    details = "; ".join(
        f"{name} {level(value)} ({value:+.2f})"
        for name, value in selected
    )
    return {
        "text": f"State signal ({len(selected)}/{len(values)} sliders): {details}",
        "sliders": {name: round(value, 4) for name, value in selected},
        "shown": len(selected),
        "available": len(values),
    }


def format_code_pointer_signal(
    state: object,
    *,
    repo_root: Optional[Path] = None,
    max_items: Optional[int] = None,
) -> Optional[dict]:
    """Validate and render bounded module/function pointers without code contents."""
    if not isinstance(state, dict):
        return None
    intent = state.get("text_expression_intent")
    intent = intent if isinstance(intent, dict) else {}
    raw_pointers = intent.get("pointers")
    if raw_pointers is None:
        raw_pointers = state.get("code_pointer_signal")
    if isinstance(raw_pointers, (str, dict)):
        raw_pointers = [raw_pointers]
    if not isinstance(raw_pointers, list) or not raw_pointers:
        return None
    requested = max_items if max_items is not None else intent.get("max_code_pointers", 4)
    try:
        limit = max(1, min(8, int(requested or 4)))
    except (TypeError, ValueError):
        limit = 4
    root = (repo_root or Path(".")).resolve()
    accepted = []
    rejections = []
    seen = set()

    for raw in raw_pointers[: limit * 3]:
        module = ""
        functions = []
        if isinstance(raw, str):
            module, separator, function_text = raw.strip().partition(":")
            if separator:
                functions = [part.strip() for part in function_text.split(",") if part.strip()]
        elif isinstance(raw, dict):
            module = str(raw.get("module") or raw.get("file") or raw.get("path") or "").strip()
            raw_functions = raw.get("functions", raw.get("function"))
            if isinstance(raw_functions, str):
                functions = [part.strip() for part in raw_functions.split(",") if part.strip()]
            elif isinstance(raw_functions, list):
                functions = [str(part).strip() for part in raw_functions if str(part).strip()]
        module = module.replace("\\", "/")
        module_parts = Path(module).parts
        if Path(module).is_absolute() or ".." in module_parts:
            rejections.append({"pointer": str(raw), "reason": "outside_repository"})
            continue
        if module.startswith("./"):
            module = module[2:]
        if not module or not module.endswith(".py") or module.startswith("AI_Children/"):
            rejections.append({"pointer": str(raw), "reason": "invalid_module"})
            continue
        path = (root / module).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            rejections.append({"pointer": str(raw), "reason": "outside_repository"})
            continue
        if not path.is_file():
            rejections.append({"pointer": str(raw), "reason": "module_not_found"})
            continue
        valid_functions = []
        if functions:
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
                available_functions = {
                    node.name
                    for node in ast.walk(tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
            except Exception:
                rejections.append({"pointer": str(raw), "reason": "module_unreadable"})
                continue
            missing = [name for name in functions if name not in available_functions]
            valid_functions = [name for name in functions if name in available_functions]
            for name in missing:
                rejections.append({
                    "pointer": f"{module}:{name}",
                    "reason": "function_not_found",
                })
            if not valid_functions:
                continue
        key = (module, tuple(valid_functions))
        if key in seen:
            continue
        seen.add(key)
        accepted.append({"module": module, "functions": valid_functions})
        if len(accepted) >= limit:
            break
    if not accepted:
        return None
    rendered = []
    for pointer in accepted:
        functions = pointer["functions"]
        rendered.append(
            pointer["module"] + (f" → {', '.join(functions)}" if functions else "")
        )
    return {
        "text": "Code pointer: " + "; ".join(rendered),
        "pointers": accepted,
        "rejections": rejections,
    }


def choose_text_expression_strategy(
    state: object,
    *,
    mapped_count: int,
    token_count: int,
    adapter_available: bool,
    expression_drive: float,
    emotion_available: bool,
    code_pointer_available: bool,
    song_available: bool = False,
    conversation_scene: object = None,
) -> dict:
    """Choose response, deliberate mirroring, a fallback signal, or silence."""
    state = state if isinstance(state, dict) else {}
    emotion_values = _emotion_values_from_state(state)

    def unit(name: str, default: float = 0.0) -> float:
        try:
            value = float(emotion_values.get(name, default))
        except (TypeError, ValueError):
            value = default
        return max(0.0, min(1.0, (value + 1.0) / 2.0))

    try:
        boredom = max(0.0, min(1.0, float(state.get("emotion_boredom", 0.0) or 0.0)))
    except (TypeError, ValueError):
        boredom = 0.0
    try:
        playfulness = max(0.0, min(1.0, float(state.get("emotion_playfulness_level", 0.0) or 0.0)))
    except (TypeError, ValueError):
        playfulness = 0.0
    drive = max(0.0, min(1.0, float(expression_drive or 0.0)))
    tokens = max(0, int(token_count or 0))
    mapped = max(0, int(mapped_count or 0))
    coverage = min(1.0, mapped / max(1, tokens))
    mirror_evaluation = score_text_mirroring(
        state,
        mapping_coverage=coverage,
        expression_drive=drive,
    )
    previous_streak = int(mirror_evaluation["previous_mirror_streak"])
    scene = conversation_scene if isinstance(conversation_scene, dict) else {}
    scene_signals = scene.get("signals") if isinstance(scene.get("signals"), dict) else {}
    has_prior_context = bool(scene_signals.get("has_prior_context"))
    reply_expected = bool(scene_signals.get("reply_expected"))
    continuity_count = len(scene_signals.get("continuity_terms") or [])
    relevant_memory_count = len(scene.get("memory_references") or [])
    scene_response_bonus = (
        (0.12 if reply_expected else 0.0)
        + (0.04 if has_prior_context else 0.0)
        + min(0.06, 0.02 * continuity_count)
        + min(0.06, 0.03 * relevant_memory_count)
    )

    scores = {
        "respond": 0.55 + 0.10 * unit("complexity") + 0.08 * unit("connection") + 0.07 * unit("interest") + scene_response_bonus
        if adapter_available else None,
        "mirror": mirror_evaluation["score"] if mapped and coverage >= 0.6 else None,
        "emotion": (
            0.12 + 0.20 * unit("intensity") + 0.18 * (1.0 - unit("clarity"))
            + 0.10 * unit("stress") + 0.08 * unit("isolation")
        ) if emotion_available else None,
        "code_pointer": 0.14 + 0.12 * unit("curiosity") + 0.08 * unit("clarity")
        if code_pointer_available else None,
        "song": (
            0.12 + 0.16 * unit("connection") + 0.14 * unit("interest")
            + 0.12 * playfulness + 0.08 * unit("intensity")
        ) if song_available else None,
        "silence": max(0.0, 0.06 + 0.18 * (1.0 - drive) + 0.10 * unit("isolation") + 0.10 * (1.0 - unit("connection")) - (0.08 if reply_expected else 0.0)),
    }
    available = {name: score for name, score in scores.items() if score is not None}
    raw_intent = state.get("text_expression_intent")
    intent_payload = raw_intent if isinstance(raw_intent, dict) else {}
    requested = intent_payload.get("strategy") or intent_payload.get("mode") or raw_intent
    aliases = {
        "reply": "respond", "original": "respond", "practice": "mirror",
        "mimic": "mirror", "state": "emotion", "feelings": "emotion",
        "module": "code_pointer", "modules": "code_pointer", "code": "code_pointer",
        "music": "song", "track": "song",
        "quiet": "silence",
    }
    normalized_request = str(requested or "").strip().lower()
    requested = aliases.get(normalized_request, normalized_request)
    if requested in available:
        strategy = requested
        reason = "explicit_intent"
    else:
        strategy = max(available, key=lambda name: (available[name], name))
        reason = "internal_state_scores"
    return {
        "strategy": strategy,
        "reason": reason,
        "requested_strategy": requested or None,
        "intent_available": requested in available if requested else None,
        "scores": {name: round(score, 4) if score is not None else None for name, score in scores.items()},
        "mapping_coverage": round(coverage, 4),
        "mapped_count": mapped,
        "token_count": tokens,
        "mirror_streak": previous_streak + 1 if strategy == "mirror" else 0,
        "signals": {
            "expression_drive": round(drive, 4), "boredom": round(boredom, 4),
            "playfulness": round(playfulness, 4), "curiosity": round(unit("curiosity"), 4),
            "novelty": round(unit("novelty"), 4), "familiarity": round(unit("familiarity"), 4),
            "scene_id": scene.get("scene_id"),
            "scene_turn_count": int(scene.get("turn_count", 0) or 0),
            "reply_expected": reply_expected,
            "continuity_term_count": continuity_count,
            "relevant_memory_count": relevant_memory_count,
        },
    }


def _is_emotion_or_sound_symbol(symbol: object) -> bool:
    symbol_id = str(symbol or "").strip().lower()
    return symbol_id.startswith(
        ("sound_symbol_", "sym_snd_", "combo_snd_", "sym_emotion_")
    )


def resolve_song_expression_candidate(
    state: object,
    *,
    child: str,
    base_path: Path = Path("AI_Children"),
) -> dict:
    """Resolve a remembered WAV without searching Ina's music directory."""
    state = state if isinstance(state, dict) else {}
    intent = state.get("text_expression_intent")
    intent = intent if isinstance(intent, dict) else {}
    workspace = state.get("daw_workspace_state")
    workspace = workspace if isinstance(workspace, dict) else {}
    project = workspace.get("project")
    project = project if isinstance(project, dict) else {}
    remembered = [
        ("text_expression_intent.song_path", intent.get("song_path")),
        ("text_expression_intent.path", intent.get("path")),
        ("daw_workspace_state.project.last_render", project.get("last_render")),
    ]
    studio_root = (base_path / child / "memory" / "music_studio").resolve()
    rejections = []
    for source, raw_path in remembered:
        if not raw_path:
            continue
        supplied = Path(str(raw_path)).expanduser()
        direct = supplied if supplied.is_absolute() else Path.cwd() / supplied
        try:
            direct.resolve().relative_to(studio_root)
            unresolved_path = direct
        except ValueError:
            unresolved_path = supplied if supplied.is_absolute() else studio_root / supplied
        if unresolved_path.is_symlink():
            rejections.append({
                "source": source, "path": str(raw_path),
                "reason": "symlink_rejected",
            })
            continue
        path = unresolved_path.resolve()
        try:
            path.relative_to(studio_root)
        except ValueError:
            rejections.append({
                "source": source, "path": str(raw_path),
                "reason": "outside_music_studio",
            })
            continue
        if path.suffix.lower() != ".wav":
            rejections.append({"source": source, "path": str(raw_path), "reason": "not_wav"})
            continue
        if not path.is_file():
            rejections.append({"source": source, "path": str(raw_path), "reason": "missing_file"})
            continue
        opus_path = path.with_suffix(".opus")
        if opus_path.is_symlink():
            rejections.append({
                "source": source, "path": str(opus_path),
                "reason": "opus_symlink_rejected",
            })
            continue
        return {
            "available": True,
            "source": source,
            "wav_path": str(path),
            "opus_path": str(opus_path),
            "rejections": rejections,
        }
    return {"available": False, "rejections": rejections}


def inspect_song_opus_sidecar(
    candidate: dict,
    *,
    max_bytes: int = DEFAULT_SONG_ATTACHMENT_MAX_BYTES,
) -> dict:
    """Accept only a current, already-rendered Opus copy for Discord."""
    wav_path = Path(str(candidate.get("wav_path") or ""))
    opus_path = Path(str(candidate.get("opus_path") or ""))
    if not candidate.get("available") or not wav_path.is_file():
        return {"status": "rejected", "reason": "candidate_unavailable"}
    try:
        if not opus_path.is_file():
            return {"status": "rejected", "reason": "opus_sidecar_missing"}
        if opus_path.stat().st_mtime_ns < wav_path.stat().st_mtime_ns:
            return {"status": "rejected", "reason": "opus_sidecar_stale"}
        size = opus_path.stat().st_size
        if size > max(1, int(max_bytes)):
            return {
                "status": "rejected",
                "reason": "opus_exceeds_attachment_limit",
                "opus_path": str(opus_path),
                "bytes": size,
                "max_bytes": int(max_bytes),
            }
        return {
            "status": "ready",
            "wav_path": str(wav_path),
            "opus_path": str(opus_path),
            "bytes": size,
        }
    except (OSError, ValueError) as exc:
        return {
            "status": "failed", "reason": type(exc).__name__,
            "detail": str(exc)[:300],
        }


def encode_selected_text_expression(
    text: str,
    *,
    child: str,
    language_preference: object,
    max_symbols: int,
    context: Optional[dict] = None,
) -> tuple[str, dict]:
    """Encode Ina's selected meaning after selection, never the inbound prompt."""
    selected_text = str(text or "").strip()
    if not selected_text:
        return "", {"effective_language_mode": "none"}
    encoded = generate_symbolic_reply_from_text(
        selected_text,
        child=child,
        base_path=Path("AI_Children"),
        max_symbols=max_symbols,
        context={
            **(context if isinstance(context, dict) else {}),
            "source": "discord_selected_expression",
            "source_text": selected_text,
            "tokens": _extract_tokens(selected_text),
            "tags": [
                "discord",
                "selected_expression",
                *(
                    list((context or {}).get("tags") or [])
                    if isinstance(context, dict)
                    else []
                ),
            ],
        },
        playback=False,
    )
    symbols = list(encoded.get("symbols") or []) if encoded else []
    if not symbols:
        return selected_text, {
            "effective_language_mode": "english_unmapped",
            "selected_expression_symbols": [],
            "native_translation_complete": False,
            "native_translation_rejections": ["no_mapped_symbols"],
        }
    dual = build_dual_symbolic_message(
        symbols,
        child=child,
        base_path=Path("AI_Children"),
        context=context,
        fallback_to_symbol_to_token=False,
        native_style="glyphs",
    )
    response_tokens = _extract_tokens(selected_text)
    unknown_words = list((encoded or {}).get("unknown") or [])
    native_sources = (dual or {}).get("native_sources") or {}
    gloss_sources = (dual or {}).get("gloss_sources") or {}
    native_text = str((dual or {}).get("native_text") or "").strip()
    aligned_native_tokens = list((dual or {}).get("native_tokens") or native_text.split())
    aligned_gloss_tokens = list(
        (dual or {}).get("gloss_tokens")
        or str((dual or {}).get("gloss_text") or "").split()
    )
    native_token_set = set(native_text.split())
    paralinguistic_indexes = {
        index
        for index, symbol in enumerate(symbols)
        if _is_emotion_or_sound_symbol(symbol)
    }
    linguistic_symbols = [
        symbol
        for index, symbol in enumerate(symbols)
        if index not in paralinguistic_indexes
    ]
    resolved_native_tokens = []
    resolved_gloss_tokens = []
    literal_gloss_tokens = []
    unresolved_linguistic_tokens = []
    emotion_sound_tokens = []
    for index, symbol in enumerate(symbols):
        if index >= len(aligned_native_tokens):
            continue
        native_token = str(aligned_native_tokens[index])
        if index < len(aligned_gloss_tokens) and symbol in gloss_sources:
            literal_gloss_tokens.append(str(aligned_gloss_tokens[index]))
        if index in paralinguistic_indexes:
            emotion_sound_tokens.append(str(symbol))
            continue
        if (
            index < len(aligned_gloss_tokens)
            and symbol in native_sources
            and symbol in gloss_sources
        ):
            resolved_native_tokens.append(native_token)
            resolved_gloss_tokens.append(str(aligned_gloss_tokens[index]))
        else:
            unresolved_linguistic_tokens.append(native_token)
    mapped_native_text = " ".join(resolved_native_tokens).strip()
    mapped_gloss_text = " ".join(resolved_gloss_tokens).strip()
    literal_gloss_text = " ".join(literal_gloss_tokens).strip()
    linguistic_signal_text = " ".join(unresolved_linguistic_tokens).strip()
    emotion_sound_text = " ".join(emotion_sound_tokens).strip()
    word_for_word_text = literal_gloss_text or "[no grounded word mappings]"
    complete_dual = {
        **(dual or {}),
        "text": f"Native: {mapped_native_text}\nHuman guess: {word_for_word_text}",
        "native_text": mapped_native_text,
        "gloss_text": word_for_word_text,
    }
    rejection_reasons = []
    if unknown_words:
        rejection_reasons.append("unknown_response_words")
    if len(linguistic_symbols) < len(response_tokens):
        rejection_reasons.append("incomplete_token_coverage")
    unresolved_native_symbols = [
        symbol
        for symbol in linguistic_symbols
        if symbol not in native_sources or symbol in native_token_set
    ]
    if unresolved_native_symbols:
        rejection_reasons.append("unresolved_native_symbols")
    unresolved_gloss_symbols = [
        symbol for symbol in linguistic_symbols if symbol not in gloss_sources
    ]
    if unresolved_gloss_symbols:
        rejection_reasons.append("unresolved_gloss_symbols")
    translation_complete = bool(
        dual and linguistic_symbols and not rejection_reasons
    )
    translation_metadata = {
        "selected_expression_symbols": symbols,
        "selected_expression_native_candidate": native_text or None,
        "selected_expression_gloss_text": selected_text,
        "selected_expression_unmapped_words": unknown_words,
        "native_translation_complete": translation_complete,
        "native_translation_rejections": rejection_reasons,
        "native_translation_unresolved_symbols": unresolved_native_symbols,
        "native_translation_unresolved_gloss_symbols": unresolved_gloss_symbols,
        "selected_expression_mapped_native_text": mapped_native_text or None,
        "selected_expression_mapped_gloss_text": mapped_gloss_text or None,
        "selected_expression_word_for_word_text": literal_gloss_text or None,
        "selected_expression_linguistic_signal": linguistic_signal_text or None,
        "selected_expression_emotion_sound_signal": emotion_sound_text or None,
        "native_translation_token_count": len(response_tokens),
        "native_translation_symbol_count": len(symbols),
    }
    if not translation_complete:
        _, requested_mode = select_symbolic_message_text(
            complete_dual, language_preference
        )
        layers = []
        if requested_mode == "english":
            if native_text:
                layers.append(f"Native: {native_text}")
            layers.append(f"Word-for-word: {word_for_word_text}")
            layers.append(f"English expression: {selected_text}")
            effective_mode = "english_incomplete_native"
        else:
            if native_text:
                layers.append(f"Native: {native_text}")
            layers.append(f"Word-for-word: {word_for_word_text}")
            layers.append(f"English expression: {selected_text}")
            effective_mode = "mixed_partial_native"
        if emotion_sound_text:
            layers.append(f"Emotion/sound signal: {emotion_sound_text}")
        rendered = "\n".join(layers) or selected_text
        return rendered, {
            **translation_metadata,
            "effective_language_mode": effective_mode,
        }
    _, effective_mode = select_symbolic_message_text(
        complete_dual, language_preference
    )
    layers = []
    if effective_mode == "english":
        if native_text:
            layers.append(f"Native: {native_text}")
        layers.append(f"Word-for-word: {word_for_word_text}")
        layers.append(f"English expression: {selected_text}")
    else:
        if native_text:
            layers.append(f"Native: {native_text}")
        layers.append(f"Word-for-word: {word_for_word_text}")
        layers.append(f"English expression: {selected_text}")
    if emotion_sound_text:
        layers.append(f"Emotion/sound signal: {emotion_sound_text}")
    return "\n".join(layers) or selected_text, {
        **translation_metadata,
        "effective_language_mode": effective_mode,
        "selected_expression_native_text": native_text,
    }


def get_chat_adapter():
    """
    Lazy-load a simple text responder. Uses LMStudioAdapter if available,
    otherwise leaves the expression arbiter to choose a bounded fallback.
    """
    global _CHAT_ADAPTER
    if _CHAT_ADAPTER is not None:
        return _CHAT_ADAPTER
    if LMStudioAdapter is None:
        return None
    try:
        _CHAT_ADAPTER = LMStudioAdapter(child=get_current_child())
    except Exception:
        logger.exception("Failed to initialise LMStudioAdapter; using expression fallback.")
        _CHAT_ADAPTER = None
    return _CHAT_ADAPTER


def _extract_tokens(text: str):
    return [tok.lower() for tok in re.findall(r"[A-Za-z0-9']+", text or "")]


def _clean_content_type(content_type):
    if not content_type:
        return None
    return content_type.split(";", 1)[0].strip().lower() or None


def _resolve_image_extension(filename, content_type):
    ext = Path(filename or "").suffix.lower()
    if ext in IMAGE_ATTACHMENT_EXTENSIONS:
        return ext
    cleaned = _clean_content_type(content_type)
    if cleaned in IMAGE_ATTACHMENT_MIME_MAP:
        return IMAGE_ATTACHMENT_MIME_MAP[cleaned]
    if cleaned and cleaned.startswith("image/") and ext:
        return ext
    return None


def _sanitize_attachment_basename(filename):
    base = Path(filename or "").stem.strip()
    if not base:
        base = "image"
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    safe = safe.strip("._-")
    return (safe or "image")[:64]


def _resolve_attachment_dir(child, cfg):
    raw = cfg.get("image_attachment_dir") or cfg.get("attachment_dir") or cfg.get("attachments_dir")
    if raw:
        return Path(raw)
    return Path("AI_Children") / child / "memory" / "discord_attachments"


def _resolve_attachment_limit(cfg):
    raw = (
        cfg.get("image_attachment_max_mb")
        or cfg.get("attachment_max_mb")
        or cfg.get("attachment_max_size_mb")
    )
    if raw is None:
        return DEFAULT_IMAGE_ATTACHMENT_MAX_BYTES
    try:
        limit = int(float(raw) * 1024 * 1024)
        return limit if limit > 0 else DEFAULT_IMAGE_ATTACHMENT_MAX_BYTES
    except Exception:
        logger.warning("Invalid discord image attachment max size; using default.")
        return DEFAULT_IMAGE_ATTACHMENT_MAX_BYTES


def _resolve_attachment_count(cfg):
    raw = cfg.get("max_image_attachments")
    if raw is None:
        return DEFAULT_IMAGE_ATTACHMENT_MAX_COUNT
    try:
        return max(0, int(raw))
    except Exception:
        logger.warning("Invalid discord max_image_attachments value; using default.")
        return DEFAULT_IMAGE_ATTACHMENT_MAX_COUNT


def _format_image_attachment_note(attachments):
    if not attachments:
        return ""
    names = []
    for entry in attachments:
        name = entry.get("original_filename") or entry.get("filename")
        if name:
            names.append(str(name))
    if not names:
        return f"[Image attachment(s): {len(attachments)}]"
    if len(names) > 4:
        shown = ", ".join(names[:4])
        return f"[Image attachment(s): {shown}, +{len(names) - 4} more]"
    return f"[Image attachment(s): {', '.join(names)}]"


def _collect_vision_context(attachments):
    perceptions = [
        entry.get("vision_perception")
        for entry in attachments or []
        if isinstance(entry, dict) and isinstance(entry.get("vision_perception"), dict)
    ]
    symbols = []
    event_ids = []
    visual_token_ids = []
    hypotheses = []
    for perception in perceptions:
        for symbol in perception.get("recognized_symbols") or []:
            if symbol and symbol not in symbols:
                symbols.append(str(symbol))
        event_id = perception.get("event_id")
        if event_id and event_id not in event_ids:
            event_ids.append(str(event_id))
        learning = perception.get("visual_token_learning") or {}
        for token_id in learning.get("candidate_ids") or []:
            if token_id and token_id not in visual_token_ids:
                visual_token_ids.append(str(token_id))
        for match in learning.get("matches") or []:
            if not isinstance(match, dict):
                continue
            for hypothesis in match.get("hypotheses") or []:
                if not isinstance(hypothesis, dict) or not hypothesis.get("word"):
                    continue
                row = {"cluster_id": match.get("cluster_id"), **hypothesis}
                key = (row.get("cluster_id"), row.get("word"))
                if not any((item.get("cluster_id"), item.get("word")) == key for item in hypotheses):
                    hypotheses.append(row)
    hypotheses.sort(
        key=lambda item: (-float(item.get("confidence", 0.0)), -int(item.get("support", 0)), item.get("word", ""))
    )
    return {
        "perceptions": perceptions,
        "recognized_symbols": symbols,
        "event_ids": event_ids,
        "visual_token_ids": visual_token_ids,
        "hypotheses": hypotheses[:16],
    }


def _format_image_perception_ack(attachments, vision_context):
    names = [
        entry.get("original_filename") or entry.get("filename")
        for entry in attachments or []
        if isinstance(entry, dict)
    ]
    names = [str(name) for name in names if name]
    subject = f" '{names[0]}'" if len(names) == 1 else ""
    perceptions = vision_context.get("perceptions") or []
    if not perceptions:
        return (
            f"I received and stored the image attachment{subject} as image memory, "
            "but my vision pass did not produce a usable perception."
        )

    perception = perceptions[0]
    brightness = float(perception.get("brightness", 0.5))
    contrast = float(perception.get("contrast", 0.0))
    light_word = "dark" if brightness < 0.33 else "bright" if brightness > 0.67 else "mid-lit"
    contrast_word = "high-contrast" if contrast > 0.25 else "low-contrast" if contrast < 0.10 else "moderate-contrast"
    orientation = perception.get("orientation") or "unknown-orientation"
    hypotheses = vision_context.get("hypotheses") or []
    strong = [item for item in hypotheses if float(item.get("confidence", 0.0)) >= 0.55]
    if strong:
        best = strong[0]
        return (
            f"I looked at and stored the image attachment{subject}. My vision registered "
            f"a {orientation}, {light_word}, {contrast_word} image. A recurring visual form "
            f"tentatively reminds me of '{best['word']}' "
            f"(confidence {float(best.get('confidence', 0.0)):.2f})."
        )
    return (
        f"I looked at and stored the image attachment{subject}. My vision registered "
        f"a {orientation}, {light_word}, {contrast_word} image, but it did not match "
        "a visual symbol I have learned yet."
    )


def _save_fragment(child, fragment):
    frag_path = Path("AI_Children") / child / "memory" / "fragments" / f"{fragment['id']}.json"
    frag_path.parent.mkdir(parents=True, exist_ok=True)
    with frag_path.open("w", encoding="utf-8") as fh:
        json.dump(fragment, fh, indent=4)


def _build_discord_image_fragment(
    *,
    path: Path,
    child: str,
    fragment_id: str,
    tags: list[str],
    summary: str,
    source_context: dict,
    rel_path: Optional[str],
):
    rgb_image = None
    decoder = "pillow_numpy"
    width = 0
    height = 0
    array = []

    if Image is not None and np is not None:
        try:
            with Image.open(path) as img:
                width, height = img.size
                rgb_image = np.array(img.convert("RGB"))
                array = np.array(img.convert("L")).flatten().tolist()
        except Exception:
            logger.exception("Failed to open image attachment through Pillow at %s", path)

    if not array:
        try:
            fallback = extract_image_features(path, limit=1024)
            array = fallback.get("features") or []
            width = int(fallback.get("width") or 0)
            height = int(fallback.get("height") or 0)
            decoder = str(fallback.get("decoder") or "simple_image_fallback")
        except ImageFallbackError as exc:
            logger.warning("Discord image vision decoder could not read %s: %s", path.name, exc)
            return None
        except Exception:
            logger.exception("Discord image vision decoder failed for %s", path.name)
            return None

    if not array:
        return None

    normalized = [max(0.0, min(255.0, float(value))) for value in array]
    brightness = sum(normalized) / (len(normalized) * 255.0)
    mean_pixel = sum(normalized) / len(normalized)
    variance = sum((value - mean_pixel) ** 2 for value in normalized) / len(normalized)
    contrast = (variance ** 0.5) / 255.0
    dominant_color = "unknown"
    if rgb_image is not None:
        channel_means = rgb_image.mean(axis=(0, 1)) / 255.0
        dominant_index = int(np.argmax(channel_means))
        dominant_color = ("red", "green", "blue")[dominant_index]
        if max(channel_means) - min(channel_means) < 0.08:
            dominant_color = "neutral"

    orientation = "square"
    if width > height * 1.1:
        orientation = "landscape"
    elif height > width * 1.1:
        orientation = "portrait"

    vision_symbols = []
    for entry in load_generated_symbols(child, base_path=Path("AI_Children")):
        stored = entry.get("image_features") if isinstance(entry, dict) else None
        symbol_id = entry.get("id") if isinstance(entry, dict) else None
        if not stored or not symbol_id:
            continue
        try:
            similarity = visual_cosine_similarity(normalized[:512], stored)
        except Exception:
            continue
        if similarity > 0.93 and symbol_id not in vision_symbols:
            vision_symbols.append(str(symbol_id))

    event_id = None
    try:
        vision_bridge = LiveExperienceBridge(child=child, base_path=Path("AI_Children"))
        event_id = vision_bridge.log_screen_snapshot(
            rgb_image if rgb_image is not None else normalized,
            tags=["discord", "vision", "image", "attachment", "inbound"],
            narrative=summary,
            metadata={**source_context, "vision_decoder": decoder},
        )
    except Exception:
        logger.exception("Failed to ground Discord image as a vision experience: %s", path.name)

    visual_token_learning = {}
    try:
        visual_token_learning = observe_visual_tokens(
            path,
            child=child,
            event_id=event_id,
            base_path=Path("AI_Children"),
        )
    except Exception:
        logger.exception("Discord visual-token learning failed for %s", path.name)

    perception = {
        "event_id": event_id,
        "recognized_symbols": vision_symbols,
        "width": int(width),
        "height": int(height),
        "orientation": orientation,
        "brightness": round(brightness, 4),
        "contrast": round(contrast, 4),
        "dominant_color": dominant_color,
        "decoder": decoder,
        "source": "ina_vision",
        "visual_token_learning": visual_token_learning,
    }
    try:
        update_inastate("last_discord_vision", perception)
    except Exception:
        logger.exception("Failed to publish Discord vision state for %s", path.name)

    fragment = {
        "id": fragment_id,
        "summary": summary,
        "tags": list(dict.fromkeys(tags)),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "discord_attachment",
        "modality": "image",
        "image_features": normalized[:1024],
        "image_path": rel_path or str(path),
        "emotions": {"curiosity": 0.3, "focus": 0.3},
        "source_context": source_context,
        "event_ref": event_id,
        "vision_perception": perception,
    }
    fragment["tags"].extend(
        symbol for symbol in vision_symbols if symbol not in fragment["tags"]
    )
    visual_token_ids = list(visual_token_learning.get("candidate_ids") or [])[:16]
    fragment["visual_token_ids"] = visual_token_ids
    fragment["tags"].extend(
        token_id for token_id in visual_token_ids if token_id not in fragment["tags"]
    )
    allowed, reason = should_accept_fragment(fragment=fragment)
    if not allowed:
        logger.info("Skipping discord image fragment %s (%s).", fragment_id, reason)
        fragment["stored"] = False
        fragment["storage_rejection_reason"] = reason
        return fragment

    if FractalTransformer is not None:
        transformer = FractalTransformer()
        vec = transformer.encode_image_fragment(fragment)
        fragment["importance"] = vec.get("importance")
    else:
        fragment["importance"] = round(sum(value / 255.0 for value in normalized) / len(normalized), 4)
    fragment["stored"] = True
    _save_fragment(child, fragment)
    return fragment


def _log_raw_outbound(msg):
    """
    Raw fallback: log outbound text when Discord send is unavailable.
    Keeps visibility into Ina's replies even if Discord is down.
    """
    chan = getattr(msg, "channel", None)
    chan_name = getattr(chan, "name", "unknown") if chan else "unknown"
    logger.info("[RAW OUTBOUND] channel=%s text=%s", chan_name, msg.text)


def _coerce_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _header_float(headers, key: str) -> Optional[float]:
    if not headers:
        return None
    try:
        raw = headers.get(key)
    except Exception:
        raw = None
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


def _discord_retry_after(exc: Exception) -> Optional[float]:
    retry_after = getattr(exc, "retry_after", None)
    if retry_after is not None:
        try:
            return max(0.0, float(retry_after))
        except (TypeError, ValueError):
            pass

    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    for header_name in ("Retry-After", "X-RateLimit-Reset-After"):
        parsed = _header_float(headers, header_name)
        if parsed is not None:
            return parsed
    return None


def _attachment_path_is_audio(path: Optional[str]) -> bool:
    if not path:
        return False
    return Path(path).suffix.lower() in AUDIO_ATTACHMENT_EXTENSIONS


def _find_channel_by_name(client: discord.Client, name: str, channel_type) -> discord.abc.GuildChannel | None:
    """
    Search across all guilds the bot can see to find a channel by exact name and type.
    """
    target = name.lower().strip()
    for guild in client.guilds:
        for channel in guild.channels:
            if isinstance(channel, channel_type) and channel.name.lower() == target:
                return channel
    return None


def resolve_configured_channels(client: discord.Client):
    """
    Resolve text/voice channel targets using IDs when present, otherwise by name.
    Returns a tuple (text_channel, voice_channel) which may contain None if not found.
    """
    cfg = get_discord_config()
    text_id = _coerce_int(cfg.get("text_channel_id"))
    voice_id = _coerce_int(cfg.get("voice_channel_id"))
    text_name = (cfg.get("text_channel_name") or "").strip()
    voice_name = (cfg.get("voice_channel_name") or "").strip()

    text_chan = client.get_channel(text_id) if text_id else None
    if not text_chan and text_name:
        text_chan = _find_channel_by_name(client, text_name, discord.TextChannel)

    voice_chan = client.get_channel(voice_id) if voice_id else None
    if not voice_chan and voice_name:
        voice_chan = _find_channel_by_name(client, voice_name, discord.VoiceChannel)

    if text_chan:
        logger.info("Resolved text channel: %s (id=%s, guild=%s)", text_chan.name, text_chan.id, text_chan.guild)
    else:
        logger.warning("Text channel not resolved. Set discord.text_channel_id or discord.text_channel_name in config.json.")

    if voice_chan:
        logger.info("Resolved voice channel: %s (id=%s, guild=%s)", voice_chan.name, voice_chan.id, voice_chan.guild)
    else:
        logger.warning("Voice channel not resolved. Set discord.voice_channel_id or discord.voice_channel_name in config.json.")

    return text_chan, voice_chan


# ---------------------------------------------------------------------------
# Config – change these for your setup
# ---------------------------------------------------------------------------

DEFAULT_OWNER_ID = 123456789012345678  # <-- replace via config.json -> discord.owner_user_id
VOICE_JOIN_COMMANDS = {"/ina join", "/ina voice", "/ina voice join", "/ina join voice"}
VOICE_LEAVE_COMMANDS = {"/ina leave", "/ina voice leave", "/ina leave voice", "/ina stop voice"}


def _resolve_primary_user_id() -> int:
    cfg = get_discord_config()
    raw_id = cfg.get("owner_user_id") if cfg else None
    if raw_id is not None:
        try:
            return int(raw_id)
        except (TypeError, ValueError):
            logger.warning("discord.owner_user_id is not an integer; checking social map next.")

    owner_from_social = get_owner_user_id(cfg)
    if owner_from_social:
        logger.info("Using owner user id from social map: %s", owner_from_social)
        return owner_from_social

    logger.warning(
        "discord.owner_user_id not set; using default placeholder (%s). "
        "Set config.json->discord.owner_user_id or tag an entry as 'owner' in social_map.json.",
        DEFAULT_OWNER_ID,
    )
    return DEFAULT_OWNER_ID


SAKURA_USER_ID = _resolve_primary_user_id()

# Optional: instance name for Ina, used in comms_core
INA_INSTANCE_NAME = "ina"

# Name of the backend as registered in CommsCore
BACKEND_NAME = "discord"


# ---------------------------------------------------------------------------
# Optional: custom processing pipeline hook
# ---------------------------------------------------------------------------

def process_inbound_message(msg) -> CommsResponse:
    """
    This is where you'd plug Ina's real brain in.

    For now, this tries a lightweight grounded-language adapter; if that is
    unavailable, it falls back to an echo so the bridge remains testable.
    """
    cfg = get_discord_config()
    if isinstance(cfg, dict) and cfg.get("allow_replies") is False:
        return CommsResponse(
            text=None,
            metadata={
                "adapter": "disabled",
                "reason": "discord.allow_replies=false",
            },
        )

    child = get_current_child()
    user_text = msg.text or ""
    attachments = []
    if msg.metadata:
        attachments = msg.metadata.get("image_attachments") or []
    attachment_note = _format_image_attachment_note(attachments)
    vision_context = _collect_vision_context(attachments)
    prompt_text = user_text
    if attachment_note:
        prompt_text = f"{user_text}\n\n{attachment_note}" if user_text.strip() else attachment_note

    # Give Ina the option to stay silent based on her urge to type/communicate.
    root_cfg: dict = {}
    try:
        root_cfg = load_root_config()
        min_urge = float(root_cfg.get("min_urge_to_type", 0.35))
    except Exception:
        min_urge = 0.35
    try:
        inastate_path = Path("AI_Children") / child / "memory" / "inastate.json"
        state = json.loads(inastate_path.read_text(encoding="utf-8")) if inastate_path.exists() else {}
    except Exception:
        state = {}
    urge_state = state.get("urge_to_type") or state.get("urge_to_communicate") or {}
    language_preference = (
        state.get("discord_language_preference")
        or state.get("communication_language_preference")
        or cfg.get("language_mode")
        or "auto"
    )
    urge_level = _resolve_adjusted_urge_level(urge_state)
    ignore_urge = bool(root_cfg.get("ignore_urge_for_typing", False)) if isinstance(root_cfg, dict) else False
    if not ignore_urge and urge_level < min_urge:
        return CommsResponse(
            text=None,
            metadata={
                "adapter": "urge_gate",
                "reason": "low_urge_to_reply",
                "urge_level": urge_level,
                "threshold": min_urge,
            },
        )

    reply_text = None
    adapter = get_chat_adapter()
    metadata = {"source": "discord_bridge.process_inbound_message", "adapter": "expression_arbiter"}
    conversation_scene = (msg.metadata or {}).get("conversation_scene") or {}
    prior_memory_candidates = [
        dict(reference)
        for reference in conversation_scene.get("memory_references", [])
        if isinstance(reference, dict)
    ]
    conversation_scene = {
        **conversation_scene,
        "memory_references": [],
        "memory_rejections": [],
    }
    memory_candidates = []
    memory_consideration = {"accepted": [], "rejected": []}
    recall_relevant = getattr(adapter, "recall_relevant", None)
    consider_memories = getattr(adapter, "consider_recalled_memories", None)
    if callable(recall_relevant) and user_text.strip():
        try:
            memory_limit = max(0, min(8, int(root_cfg.get("conversation_scene_memory_limit", 3))))
            memory_chars = max(0, min(2400, int(root_cfg.get("conversation_scene_memory_chars", 800))))
            memory_graph_bytes = max(1024, int(root_cfg.get("conversation_scene_memory_graph_max_bytes", 8 * 1024 * 1024)))
            retrieved_candidates = recall_relevant(
                user_text,
                max_items=memory_limit,
                max_chars=memory_chars,
                max_graph_bytes=memory_graph_bytes,
            )
            candidate_by_key = {}
            for candidate in [*prior_memory_candidates, *retrieved_candidates]:
                if not isinstance(candidate, dict):
                    continue
                key = (candidate.get("event_id"), candidate.get("cue"))
                candidate_by_key[key] = candidate
            memory_candidates = list(candidate_by_key.values())[:memory_limit]
            if callable(consider_memories):
                memory_consideration = consider_memories(
                    user_text, memory_candidates, scene=conversation_scene
                )
            elif memory_candidates:
                memory_consideration = {
                    "accepted": [],
                    "rejected": [
                        {
                            **candidate,
                            "consideration": {
                                "decision": "rejected",
                                "reason": "final_consideration_unavailable",
                            },
                        }
                        for candidate in memory_candidates
                    ],
                }
            conversation_scene = scene_with_memory_consideration(
                conversation_scene,
                memory_consideration,
                max_items=memory_limit,
                max_chars=memory_chars,
            )
            msg.metadata["conversation_scene"] = conversation_scene
        except Exception:
            logger.exception("Bounded conversational memory consideration failed.")
    metadata["conversation_scene"] = conversation_scene
    if vision_context.get("perceptions"):
        metadata["vision_context"] = vision_context

    tokens = _extract_tokens(user_text)
    if tokens and vision_context.get("event_ids") and vision_context.get("visual_token_ids"):
        try:
            learning_update = observe_visual_words(
                vision_context["event_ids"],
                tokens,
                child=child,
                base_path=Path("AI_Children"),
            )
            vision_context["word_learning"] = learning_update
            learned_hypotheses = learning_update.get("hypotheses") or []
            combined_hypotheses = [
                *learned_hypotheses,
                *vision_context.get("hypotheses", []),
            ]
            deduplicated = {}
            for hypothesis in combined_hypotheses:
                if not isinstance(hypothesis, dict) or not hypothesis.get("word"):
                    continue
                key = (hypothesis.get("cluster_id"), hypothesis.get("word"))
                current = deduplicated.get(key)
                if current is None or float(hypothesis.get("confidence", 0.0)) > float(current.get("confidence", 0.0)):
                    deduplicated[key] = hypothesis
            vision_context["hypotheses"] = sorted(
                deduplicated.values(),
                key=lambda item: (-float(item.get("confidence", 0.0)), -int(item.get("support", 0)), item.get("word", "")),
            )[:16]
        except Exception:
            logger.exception("Discord visual-token word learning failed")
    conversation_context = (msg.metadata or {}).get("conversation_context") or []
    edit_analysis = (msg.metadata or {}).get("message_edit")
    is_roleplay = bool((msg.metadata or {}).get("is_roleplay_context"))
    context_lines = [
        f"{turn.get('author_name', 'unknown')}: {turn.get('content', '')}"
        for turn in conversation_context[-12:]
        if isinstance(turn, dict)
    ]
    if context_lines:
        prompt_text = "Recent conversation (context only):\n" + "\n".join(context_lines) + "\n\nCurrent message:\n" + prompt_text
    if edit_analysis:
        prompt_text = (
            "The current message was edited. Original: " + edit_analysis.get("original", "")
            + "\nSurface-change inference: " + edit_analysis.get("likely_reason", "unknown")
            + "\nEdited message: " + prompt_text
        )

    visual_inference_words = []
    if not user_text.strip():
        for hypothesis in vision_context.get("hypotheses") or []:
            if float(hypothesis.get("confidence", 0.0)) < 0.70:
                continue
            word = str(hypothesis.get("word") or "").strip().lower()
            if word and word not in visual_inference_words:
                visual_inference_words.append(word)
            if len(visual_inference_words) >= 4:
                break
    symbolic_input = user_text if user_text.strip() else " ".join(visual_inference_words)

    symbolic_context = {
        "source": "discord",
        "source_text": user_text,
        "tokens": tokens,
        "visual_inference_words": visual_inference_words,
        "tags": [
            "discord",
            "text",
            "dm" if (msg.metadata or {}).get("is_dm") else "guild",
            *(["roleplay"] if is_roleplay else []),
            *(["edited"] if edit_analysis else []),
            *(["vision", "image"] if attachments else []),
            *vision_context.get("recognized_symbols", []),
        ],
        "channel": msg.channel.name,
        "current_speaker": {
            "id": msg.sender.backend_id,
            "display_name": msg.sender.display_name,
            "is_self": bool(msg.sender.is_self),
            "is_bot": bool((msg.metadata or {}).get("author_is_bot")),
        },
        "conversation_context": conversation_context,
        "conversation_scene": conversation_scene,
        "message_edit": edit_analysis,
        "expression_drive": urge_level,
        "is_high_trust": bool((msg.metadata or {}).get("is_high_trust")),
        "is_owner_friend": bool((msg.metadata or {}).get("is_owner_friend")),
        "language_state_signals": {
            "current_prediction": state.get("current_prediction"),
            "machine_semantics": state.get("machine_semantics"),
            "emotion_snapshot": state.get("emotion_snapshot"),
        },
        "vision": vision_context,
    }
    symbolic = generate_symbolic_reply_from_text(
        symbolic_input,
        child=child,
        base_path=Path("AI_Children"),
        max_symbols=_coerce_positive_int(
            cfg.get("max_reply_symbols", DEFAULT_MAX_REPLY_SYMBOLS),
            DEFAULT_MAX_REPLY_SYMBOLS,
        ),
        context=symbolic_context,
    )
    if isinstance(symbolic, dict):
        shadow = symbolic.get("language_context_shadow")
        audit = shadow.get("counterfactual_audit") if isinstance(shadow, dict) else None
        if isinstance(audit, dict):
            logger.info(
                "Language context shadow: changed=%d delta=%s coherent=%s "
                "reread=%s/%s helpful=%s output_unchanged=true",
                len(audit.get("changed_tokens") or []),
                audit.get("delta"),
                audit.get("counterfactual_more_coherent"),
                (shadow.get("snapshot_summary") or {}).get("reread_converged"),
                (shadow.get("snapshot_summary") or {}).get("reread_tokens"),
                (shadow.get("snapshot_summary") or {}).get("reread_helpful"),
            )
    vision_symbols = vision_context.get("recognized_symbols") or []
    if vision_symbols:
        text_symbols = list(symbolic.get("symbols") or []) if symbolic else []
        combined_symbols = list(dict.fromkeys([*text_symbols, *vision_symbols]))
        combined_symbols = combined_symbols[
            : _coerce_positive_int(
                cfg.get("max_reply_symbols", DEFAULT_MAX_REPLY_SYMBOLS),
                DEFAULT_MAX_REPLY_SYMBOLS,
            )
        ]
        visual_message = build_dual_symbolic_message(
            combined_symbols,
            child=child,
            base_path=Path("AI_Children"),
            context=symbolic_context,
            fallback_to_symbol_to_token=False,
            native_style="glyphs",
        )
        visual_text = (visual_message or {}).get("text") or " ".join(combined_symbols)
        if symbolic:
            symbolic = {
                **symbolic,
                "text": visual_text,
                "symbols": combined_symbols,
                "native_text": (visual_message or {}).get("native_text"),
                "gloss_text": (visual_message or {}).get("gloss_text"),
                "native_sources": (visual_message or {}).get("native_sources") or {},
                "gloss_sources": (visual_message or {}).get("gloss_sources") or {},
            }
        else:
            symbolic = {
                "text": visual_text,
                "symbols": combined_symbols,
                "unknown": [],
                "native_text": (visual_message or {}).get("native_text"),
                "gloss_text": (visual_message or {}).get("gloss_text"),
                "native_sources": (visual_message or {}).get("native_sources") or {},
                "gloss_sources": (visual_message or {}).get("gloss_sources") or {},
            }

    symbolic_unknown: list[str] = symbolic.get("unknown") if symbolic else []
    symbolic_text, effective_language_mode = select_symbolic_message_text(
        symbolic, language_preference
    )
    symbolic_native_text = symbolic.get("native_text") if symbolic else None
    symbolic_gloss_text = symbolic.get("gloss_text") if symbolic else None
    emotion_signal = format_emotion_signal(state)
    code_pointer_signal = format_code_pointer_signal(state)
    song_candidate = resolve_song_expression_candidate(state, child=child)
    adapter_can_respond = bool(adapter and (user_text.strip() or not attachments))
    has_unmapped_input = bool(tokens) and (
        symbolic is None or bool(symbolic_unknown)
    )
    constructive_probe = getattr(adapter, "has_constructive_reply", None)
    constructive_probe_used = callable(constructive_probe)
    constructive_probe_failed = False
    if adapter_can_respond and constructive_probe_used:
        try:
            adapter_has_constructive_reply = bool(constructive_probe(user_text))
        except Exception:
            logger.exception("Grounded reply capability probe failed.")
            adapter_has_constructive_reply = False
            constructive_probe_failed = True
    else:
        adapter_has_constructive_reply = (
            adapter_can_respond and not has_unmapped_input
        )
    adapter_rejection = None
    if adapter_can_respond and not adapter_has_constructive_reply:
        if constructive_probe_failed:
            adapter_rejection = "constructive_probe_failed"
        elif constructive_probe_used:
            adapter_rejection = "no_grounded_reply"
        elif has_unmapped_input:
            adapter_rejection = "unmapped_input"
        else:
            adapter_rejection = "no_constructive_reply"
    ordinary_response_available = adapter_has_constructive_reply or bool(attachments)
    expression_decision = choose_text_expression_strategy(
        state,
        mapped_count=len(symbolic.get("symbols") or []) if symbolic else 0,
        token_count=len(set(tokens)),
        adapter_available=ordinary_response_available,
        expression_drive=urge_level,
        emotion_available=emotion_signal is not None,
        code_pointer_available=code_pointer_signal is not None,
        song_available=bool(song_candidate.get("available")),
        conversation_scene=conversation_scene,
    )
    decision_record = {
        **expression_decision,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbolic_unknown_count": len(symbolic_unknown),
        "adapter_response_available": adapter_has_constructive_reply,
        "adapter_rejection": adapter_rejection,
    }
    metadata["expression_decision"] = decision_record
    metadata["song_expression_candidate"] = song_candidate
    try:
        update_inastate("last_text_expression_decision", decision_record)
        raw_intent = state.get("text_expression_intent") if isinstance(state, dict) else None
        if (
            expression_decision.get("reason") == "explicit_intent"
            and isinstance(raw_intent, dict)
            and raw_intent.get("once", True) is not False
        ):
            update_inastate("text_expression_intent", None)
    except Exception:
        logger.exception("Failed to persist text expression decision.")

    if symbolic:
        metadata.update(
            {
                "comprehension_adapter": "language_processing",
                "symbols": symbolic.get("symbols"),
                "unknown_words": symbolic.get("unknown"),
                "symbolic_native_text": symbolic_native_text,
                "symbolic_gloss_text": symbolic_gloss_text,
                "symbolic_native_sources": symbolic.get("native_sources"),
                "symbolic_gloss_sources": symbolic.get("gloss_sources"),
                "requested_language_mode": language_preference,
                "effective_language_mode": effective_language_mode,
                "vision_context": vision_context,
                "visual_inference_words": visual_inference_words,
            }
        )
    selected_strategy = expression_decision["strategy"]
    if selected_strategy == "song":
        try:
            song_limit = int(float(cfg.get("song_attachment_max_mb", 25)) * 1024 * 1024)
        except (TypeError, ValueError):
            song_limit = DEFAULT_SONG_ATTACHMENT_MAX_BYTES
        sidecar_refresh = ensure_opus_sidecar(song_candidate["wav_path"])
        metadata["song_sidecar_refresh"] = sidecar_refresh
        song_result = inspect_song_opus_sidecar(song_candidate, max_bytes=song_limit)
        metadata["song_expression"] = song_result
        if song_result.get("status") == "ready":
            raw_intent = state.get("text_expression_intent") if isinstance(state, dict) else None
            intent = raw_intent if isinstance(raw_intent, dict) else {}
            metadata.update({
                "adapter": "song_expression",
                "attachment_path": song_result["opus_path"],
                "attachment_kind": "music_opus",
                "source_wav_path": song_result["wav_path"],
            })
            return CommsResponse(text=str(intent.get("caption") or "♪"), metadata=metadata)
        logger.warning("Song expression unavailable after selection: %s", song_result)
        if emotion_signal:
            selected_strategy = "emotion"
        elif code_pointer_signal:
            selected_strategy = "code_pointer"
        else:
            selected_strategy = "silence"
        metadata["song_expression_fallback"] = selected_strategy
    if selected_strategy == "mirror" and symbolic_text:
        metadata["adapter"] = "deliberate_mirror"
        return CommsResponse(text=symbolic_text, metadata=metadata)
    if selected_strategy == "silence":
        metadata["adapter"] = "expression_silence"
        return CommsResponse(text=None, metadata=metadata)
    if selected_strategy == "emotion" and emotion_signal:
        reply_text = emotion_signal["text"]
        metadata["adapter"] = "emotion_signal"
        metadata["emotion_signal"] = emotion_signal
    elif selected_strategy == "code_pointer" and code_pointer_signal:
        reply_text = code_pointer_signal["text"]
        metadata["adapter"] = "code_pointer_signal"
        metadata["code_pointer_signal"] = code_pointer_signal

    if (
        not reply_text
        and selected_strategy == "respond"
        and adapter_has_constructive_reply
    ):
        try:
            entity_links = [
                {
                    "type": "discord_message",
                    "author_id": msg.sender.backend_id,
                    "author_name": msg.sender.display_name,
                    "channel_id": msg.channel.backend_id,
                    "channel_name": msg.channel.name,
                    "guild_id": msg.metadata.get("discord_guild_id") if msg.metadata else None,
                    "is_dm": msg.metadata.get("is_dm") if msg.metadata else None,
                }
            ]
            entity_links.extend(
                {
                    "type": "vision_perception",
                    "event_id": perception.get("event_id"),
                    "recognized_symbols": perception.get("recognized_symbols") or [],
                    "visual_token_ids": (
                        perception.get("visual_token_learning") or {}
                    ).get("candidate_ids") or [],
                    "visual_word_hypotheses": [
                        hypothesis
                        for match in (
                            (perception.get("visual_token_learning") or {}).get("matches") or []
                        )
                        if isinstance(match, dict)
                        for hypothesis in match.get("hypotheses") or []
                        if isinstance(hypothesis, dict)
                    ][:16],
                    "orientation": perception.get("orientation"),
                    "brightness": perception.get("brightness"),
                    "contrast": perception.get("contrast"),
                }
                for perception in vision_context.get("perceptions", [])
            )
            # The adapter is a grounded-memory responder, not an external
            # dictionary. Give it only sender-authored text: generated
            # instructions and context wrappers otherwise become bogus unknown
            # words in its exact-vocabulary fallback.
            explain_targets = symbolic_unknown or tokens
            if symbolic_unknown or (symbolic is None and explain_targets):
                reply_text = adapter.handle_prompt(
                    user_text,
                    speaker=msg.sender.display_name or msg.sender.internal_id,
                    tags=["discord", "text", "grounded_partial"],
                    entity_links=entity_links,
                    response_tags=["discord", "ina", "grounded_partial"],
                    include_clarification=False,
                )
                metadata["adapter"] = "lm_grounded_partial"
                metadata["unknown_words"] = explain_targets
                if symbolic_text and effective_language_mode != "english":
                    metadata["symbolic_hint"] = symbolic_text
            else:
                reply_text = adapter.handle_prompt(
                    user_text,
                    speaker=msg.sender.display_name or msg.sender.internal_id,
                    tags=["discord", "text"],
                    entity_links=entity_links,
                    response_tags=["discord", "ina"],
                    include_clarification=False,
                )
                metadata["adapter"] = "lmstudio"
        except Exception:
            logger.exception("LMStudioAdapter failed; using expression fallback.")

    if attachments:
        metadata["image_attachment_count"] = len(attachments)
        metadata["image_attachment_names"] = [
            name
            for name in (entry.get("original_filename") or entry.get("filename") for entry in attachments)
            if name
        ]

    if not reply_text and attachments and not user_text.strip():
        reply_text = _format_image_perception_ack(attachments, vision_context)
        metadata["adapter"] = "image_acknowledgement"
        metadata["vision_context"] = vision_context

    if not reply_text:
        fallback_signal = emotion_signal or code_pointer_signal
        if fallback_signal:
            reply_text = fallback_signal["text"]
            if fallback_signal is emotion_signal:
                metadata["adapter"] = "emotion_signal_fallback"
                metadata["emotion_signal"] = emotion_signal
            else:
                metadata["adapter"] = "code_pointer_signal_fallback"
                metadata["code_pointer_signal"] = code_pointer_signal

    if reply_text:
        reply_text, encoding_metadata = encode_selected_text_expression(
            reply_text,
            child=child,
            language_preference=language_preference,
            max_symbols=_coerce_positive_int(
                cfg.get("max_reply_symbols", DEFAULT_MAX_REPLY_SYMBOLS),
                DEFAULT_MAX_REPLY_SYMBOLS,
            ),
            context={
                "tags": ["discord", "selected_expression", selected_strategy],
                "channel": msg.channel.name,
                "expression_drive": urge_level,
            },
        )
        metadata.update(encoding_metadata)

    show_memory_consideration = bool(
        root_cfg.get("conversation_scene_show_memory_consideration", True)
    )
    if reply_text and show_memory_consideration:
        consideration_descriptions = []
        for reference in conversation_scene.get("memory_references", []):
            decision = reference.get("consideration") if isinstance(reference, dict) else {}
            description = decision.get("description") if isinstance(decision, dict) else None
            if description:
                consideration_descriptions.append(str(description))
        for rejection in conversation_scene.get("memory_rejections", []):
            decision = rejection.get("consideration") if isinstance(rejection, dict) else {}
            description = decision.get("description") if isinstance(decision, dict) else None
            if description:
                consideration_descriptions.append(str(description))
        if consideration_descriptions:
            visible_limit = max(1, min(8, _coerce_positive_int(root_cfg.get("conversation_scene_memory_limit", 3), 3)))
            reply_text += "\n\nMemory consideration:\n- " + "\n- ".join(
                consideration_descriptions[:visible_limit]
            )
            metadata["memory_consideration_visible"] = True
            metadata["memory_consideration_visible_count"] = min(
                len(consideration_descriptions), visible_limit
            )

    return CommsResponse(
        text=reply_text,
        metadata={
            **metadata,
            "debug": adapter is None,
        },
    )


# ---------------------------------------------------------------------------
# Discord client
# ---------------------------------------------------------------------------

def register_ina_application_commands(client) -> object:
    """Register suggested slash commands while reusing the text-command handlers."""
    raw_guild_ids = get_discord_config().get("slash_command_guild_ids") or []
    guild_ids = []
    for value in raw_guild_ids:
        try:
            guild_ids.append(int(value))
        except (TypeError, ValueError):
            continue
    command_kwargs = {"guild_ids": guild_ids} if guild_ids else {}
    ina = discord.SlashCommandGroup(
        "ina", "Talk to and manage Ina's Discord bridge.", **command_kwargs
    )
    learn = ina.create_subgroup("learn", "Review Ina's language learning.")

    @ina.command(name="status", description="Check whether Ina's bridge is listening.")
    async def ina_status(ctx):
        await client._handle_application_status(ctx)

    @ina.command(name="join", description="Ask Ina to join the configured voice channel.")
    async def ina_join(ctx):
        await client._handle_application_voice(ctx, join=True)

    @ina.command(name="leave", description="Ask Ina to leave voice chat.")
    async def ina_leave(ctx):
        await client._handle_application_voice(ctx, join=False)

    @learn.command(name="history", description="Review recent chat and revisit language mappings.")
    async def ina_learn_history(ctx):
        await client._handle_application_history_review(ctx)

    client.add_application_command(ina)
    return ina


class InaDiscordClient(discord.Bot):
    """
    Discord client that connects DMs and a configured text channel to Ina via CommsCore.
    """

    def __init__(self, comms: CommsCore, *args, **kwargs) -> None:
        intents = kwargs.pop("intents", None)
        if intents is None:
            intents = discord.Intents.default()
            intents.guilds = True
            intents.messages = True
            intents.message_content = True  # REQUIRED to read message content
            intents.dm_messages = True
            intents.voice_states = True

        super().__init__(intents=intents, *args, **kwargs)
        self.comms = comms
        self.text_channel = None
        self.voice_channel = None
        self.voice_client = None
        self.child = get_current_child()
        voice_cfg = get_voice_io_config()
        self.voice_label = voice_cfg["voice_label"]
        self.voice_pipe_path = Path(voice_cfg["voice_pipe_path"]) if voice_cfg.get("voice_pipe_path") else None
        self.voice_buffer_dir = Path(voice_cfg["voice_buffer_dir"])
        self.voice_buffer_dir.mkdir(parents=True, exist_ok=True)
        self.voice_chunk_seconds = voice_cfg["voice_chunk_seconds"]
        self._recording_active = False
        self._active_sink = None
        self.history_bridge = LiveExperienceBridge(child=self.child)
        self._typed_outbox_path = typed_outbox_path(self.child)
        self._typed_outbox_history_path = typed_outbox_history_path(self.child)
        self._outbox_policy = get_outbox_policy()
        archive_override = self._outbox_policy.get("archive_path")
        self._typed_archive_path = (
            Path(archive_override)
            if archive_override
            else typed_outbox_archive_path(self.child)
        )
        self._typed_outbox_seen = set()
        self._typed_outbox_history_offset = 0
        self._discord_send_lock = asyncio.Lock()
        self._next_discord_send_at = 0.0
        self._voice_playback_lock = asyncio.Lock()
        self._load_outbox_history()
        self._typed_outbox_task = None
        self._language_review_task = None
        self._ina_application_commands = register_ina_application_commands(self)

    def _application_command_allowed(self, ctx, *, owner_only: bool = False) -> bool:
        author = getattr(ctx, "author", None)
        author_id = getattr(author, "id", None)
        if owner_only:
            return author_id == SAKURA_USER_ID
        guild = getattr(ctx, "guild", None)
        if guild is None:
            return bool(
                author_id == SAKURA_USER_ID
                or is_owner_friend(author_id)
                or is_high_trust(author_id)
            )
        channel = getattr(ctx, "channel", None)
        channel_id = str(getattr(channel, "id", ""))
        channel_name = str(getattr(channel, "name", "")).strip().casefold()
        cfg = get_discord_config()
        if channel_id and channel_id == str(cfg.get("text_channel_id") or ""):
            return True
        roleplay = cfg.get("roleplay") or {}
        if not isinstance(roleplay, dict):
            return False
        reply_ids = {str(value) for value in roleplay.get("reply_channel_ids", [])}
        reply_names = {
            str(value).strip().casefold()
            for value in roleplay.get("reply_channel_names", ["ina-text"])
        }
        return channel_id in reply_ids or channel_name in reply_names

    async def _handle_application_status(self, ctx) -> None:
        if not self._application_command_allowed(ctx):
            await ctx.respond("Ina's commands are not enabled in this channel.", ephemeral=True)
            return
        await ctx.respond("Ina is listening here.", ephemeral=True)

    async def _handle_application_history_review(self, ctx) -> None:
        if not self._application_command_allowed(ctx, owner_only=True):
            await ctx.respond("Only Ina's operator can start a history review.", ephemeral=True)
            return
        await ctx.defer(ephemeral=True)
        result = await self._run_language_review(force=True)
        await ctx.followup.send(format_history_review_result(result), ephemeral=True)

    async def _handle_application_voice(self, ctx, *, join: bool) -> None:
        if not self._application_command_allowed(ctx):
            await ctx.respond("Ina's commands are not enabled in this channel.", ephemeral=True)
            return
        await ctx.defer()

        async def send_followup(text: str) -> None:
            await ctx.followup.send(text)

        # Existing voice behavior remains canonical; this small adapter only
        # changes where its status replies are delivered after interaction defer.
        target = type("InteractionCommandTarget", (), {})()
        target.author = ctx.author
        target.channel = type("InteractionResponseChannel", (), {})()
        target.channel.send = send_followup
        if join:
            await self._handle_voice_join(target)
        else:
            await self._handle_voice_leave(target)

    def _roleplay_mode(self, message: discord.Message) -> Optional[str]:
        """Return read_only/respond for configured RP spaces (Umani-compatible)."""
        if message.guild is None:
            return None
        cfg = get_discord_config().get("roleplay") or {}
        if not isinstance(cfg, dict) or cfg.get("enabled", True) is False:
            return None
        guild_ids = {str(value) for value in cfg.get("guild_ids", [])}
        guild_names = {str(value).strip().casefold() for value in cfg.get("guild_names", ["Umani RP", "Umani"])}
        channel_ids = {str(value) for value in cfg.get("channel_ids", [])}
        reply_channel_ids = {str(value) for value in cfg.get("reply_channel_ids", [])}
        reply_channel_names = {
            str(value).strip().casefold()
            for value in cfg.get("reply_channel_names", ["ina-text"])
        }
        guild_name = str(getattr(message.guild, "name", "")).strip().casefold()
        guild_match = str(message.guild.id) in guild_ids or guild_name in guild_names
        channel_match = not channel_ids or str(message.channel.id) in channel_ids
        if not (guild_match and channel_match):
            return None
        channel_name = str(getattr(message.channel, "name", "")).strip().casefold()
        reply_exception = (
            str(message.channel.id) in reply_channel_ids
            or channel_name in reply_channel_names
        )
        return "respond" if reply_exception or cfg.get("allow_replies", False) else "read_only"

    async def _recent_message_context(self, message: discord.Message) -> list[dict]:
        """Read a small, bounded slice of prior channel context."""
        cfg = get_discord_config()
        limit = _coerce_positive_int(cfg.get("history_context_limit", 12), 12)
        limit = min(limit, 50)
        turns = []
        try:
            async for prior in message.channel.history(limit=limit, before=message, oldest_first=False):
                content = (prior.content or "").strip()
                if not content:
                    continue
                turns.append({
                    "message_id": str(prior.id),
                    "author_id": str(prior.author.id),
                    "author_name": getattr(prior.author, "display_name", None) or str(prior.author),
                    "content": content[:2000],
                    "created_at": prior.created_at.replace(tzinfo=timezone.utc).isoformat(),
                    "is_self": bool(
                        self.user and str(prior.author.id) == str(self.user.id)
                    ),
                    "author_is_bot": bool(getattr(prior.author, "bot", False)),
                })
        except Exception:
            logger.exception("Failed to read recent Discord context for channel %s", message.channel.id)
        turns.reverse()
        return turns

    def _remember_roleplay_turn(self, message: discord.Message, context: list[dict], *, edited: bool = False) -> None:
        tags = ["discord", "roleplay", "umani_compatible", "history"]
        if edited:
            tags.append("edited")
        self.history_bridge.log_conversation_turn(
            message.content or "",
            speaker=getattr(message.author, "display_name", None) or str(message.author),
            tags=tags,
            entity_links=[{
                "type": "discord_roleplay_message",
                "message_id": str(message.id),
                "channel_id": str(message.channel.id),
                "guild_id": str(message.guild.id) if message.guild else None,
                "context_message_ids": [turn["message_id"] for turn in context],
            }],
            timestamp=message.edited_at.isoformat() if edited and message.edited_at else message.created_at.isoformat(),
        )

    async def on_ready(self) -> None:
        logger.info("Logged in as %s (ID: %s)", self.user, self.user and self.user.id)
        logger.info("Discord bridge is active. DMs from owner (%s) + configured text channel will be routed.", SAKURA_USER_ID)
        self.text_channel, self.voice_channel = resolve_configured_channels(self)
        if self._typed_outbox_task is None:
            self._typed_outbox_task = asyncio.create_task(self._watch_typed_outbox())
        if getattr(self, "_io_pressure_task", None) is None:
            self._io_pressure_task = asyncio.create_task(self._watch_io_pressure())
        await self._start_language_review_if_ready()

    async def on_message_edit(self, before: discord.Message, after: discord.Message) -> None:
        """Record and process meaningful Discord message edits."""
        if (before.content or "") == (after.content or ""):
            return
        if self.user and after.author.id == self.user.id:
            return
        if after.author.bot:
            rp_cfg = get_discord_config().get("roleplay") or {}
            if self._roleplay_mode(after) is None or rp_cfg.get("include_bot_messages", True) is False:
                return
        edit_analysis = infer_message_edit(before.content or "", after.content or "")
        is_dm = after.guild is None
        if is_dm:
            trusted = (
                after.author.id == SAKURA_USER_ID
                or is_owner_friend(after.author.id)
                or is_high_trust(after.author.id)
            )
            if not trusted:
                return
            roleplay_mode = None
        else:
            in_primary = self.text_channel is not None and after.channel.id == self.text_channel.id
            roleplay_mode = self._roleplay_mode(after)
            if not in_primary and roleplay_mode is None:
                return

        context = await self._recent_message_context(after)
        try:
            await asyncio.to_thread(
                self.history_bridge.log_conversation_turn,
                after.content or "",
                speaker=getattr(after.author, "display_name", None) or str(after.author),
                tags=["discord", "message_edit", edit_analysis["likely_reason"]],
                entity_links=[{
                    "type": "discord_message_edit",
                    "message_id": str(after.id),
                    **edit_analysis,
                }],
                timestamp=after.edited_at.isoformat() if after.edited_at else datetime.now(timezone.utc).isoformat(),
            )
        except Exception:
            logger.exception("Failed to retain Discord edit %s", after.id)
        if roleplay_mode == "read_only":
            return
        await asyncio.to_thread(
            self._route_to_comms,
            after, is_dm=is_dm,
            owner_friend=is_owner_friend(after.author.id) if is_dm else False,
            high_trust=is_high_trust(after.author.id) if is_dm else False,
            conversation_context=context, edit_analysis=edit_analysis,
            roleplay=bool(roleplay_mode),
        )

    async def on_message(self, message: discord.Message) -> None:
        # Never loop on Ina's own posts. RP proxy/webhook bots may be readable.
        if self.user and message.author.id == self.user.id:
            return
        if message.author.bot:
            rp_cfg = get_discord_config().get("roleplay") or {}
            if self._roleplay_mode(message) is None or rp_cfg.get("include_bot_messages", True) is False:
                return

        content = (message.content or "").strip()
        lower = content.lower()

        # DMs stay owner-only
        if message.guild is None:
            is_owner = message.author.id == SAKURA_USER_ID
            owner_friend = is_owner_friend(message.author.id)
            high_trust = is_high_trust(message.author.id)
            added = False
            if not is_owner:
                try:
                    added = record_dm_attempt(
                        user_id=message.author.id,
                        display_name=message.author.display_name or str(message.author),
                    )
                except Exception:
                    added = False
                    logger.exception(
                        "Failed to record DM attempt in social map for user %s (%s)",
                        message.author,
                        message.author.id,
                    )
            if not (is_owner or owner_friend or high_trust):
                logger.info(
                    "Ignoring DM from untrusted user %s (%s)%s",
                    message.author,
                    message.author.id,
                    " [logged to social_map]" if added else "",
                )
                return
            logger.info(
                "Inbound DM from %s: %s (channel %s)",
                "owner"
                if is_owner
                else "trusted friend"
                if owner_friend
                else "high-trust contact",
                message.content,
                message.channel.id,
            )
            self._record_social_contact(message)
            # inastate.json is shared with the memory graph and its advisory
            # lock can be held for a long time when the backing disk is busy.
            # Never wait for that synchronous lock on Discord's event loop:
            # doing so prevents gateway heartbeats and eventually disconnects
            # the client.
            await asyncio.to_thread(self._remember_last_dm_contact, message)
            if lower in VOICE_JOIN_COMMANDS:
                await self._handle_voice_join(message)
                return
            if lower in VOICE_LEAVE_COMMANDS:
                await self._handle_voice_leave(message)
                return
            image_attachments = await self._ingest_image_attachments(message)
            recent_context = await self._recent_message_context(message)
            await asyncio.to_thread(
                self._route_to_comms,
                message,
                is_dm=True,
                owner_friend=owner_friend,
                high_trust=high_trust,
                image_attachments=image_attachments,
                conversation_context=recent_context,
            )
            await self._note_language_review_stimulus(message)
            return

        # Guild messages: configured bridge channel or a compatible RP space.
        in_primary_channel = self.text_channel is not None and message.channel.id == self.text_channel.id
        roleplay_mode = self._roleplay_mode(message)
        if not in_primary_channel and roleplay_mode is None:
            return

        self._record_social_contact(message)
        recent_context = await self._recent_message_context(message)
        if roleplay_mode:
            await asyncio.to_thread(self._remember_roleplay_turn, message, recent_context)
            if roleplay_mode == "read_only":
                return

        if lower in VOICE_JOIN_COMMANDS:
            await self._handle_voice_join(message)
            return
        if lower in VOICE_LEAVE_COMMANDS:
            await self._handle_voice_leave(message)
            return
        if lower in {"/ina learn history", "/ina history learn"} and message.author.id == SAKURA_USER_ID:
            await message.channel.send("Scanning recent history and revisiting language mappings...")
            result = await self._run_language_review(force=True)
            await message.channel.send(format_history_review_result(result))
            return
        if lower in {"/ina status", "/ina ping"}:
            await message.channel.send("Ina is listening here.")
            return

        logger.info("Inbound guild message in text channel %s: %s", message.channel.id, content)
        image_attachments = await self._ingest_image_attachments(message)
        await asyncio.to_thread(
            self._route_to_comms,
            message, is_dm=False, image_attachments=image_attachments,
            conversation_context=recent_context, roleplay=bool(roleplay_mode),
        )
        await self._note_language_review_stimulus(message)

    async def _handle_voice_join(self, message: discord.Message) -> None:
        target_channel = self.voice_channel
        author_voice = getattr(message.author, "voice", None)
        if target_channel is None and author_voice and author_voice.channel:
            target_channel = author_voice.channel

        if target_channel is None:
            await message.channel.send("No voice channel configured or detected to join.")
            return

        async def _attempt_join(reason: str | None = None) -> tuple[bool, bool]:
            try:
                await self.ensure_voice_connected(target_channel)
                suffix = f" ({reason})" if reason else ""
                await message.channel.send(f"Joined voice channel: {target_channel.name}{suffix}")
                return True, True
            except discord.errors.ConnectionClosed as exc:
                logger.warning(
                    "Voice gateway closed while joining %s (code=%s, attempt=%s)",
                    target_channel,
                    exc.code,
                    reason or "initial",
                )
                if exc.code == 4006:
                    await message.channel.send(
                        "Discord reported an invalid voice session (4006). Resetting the voice client and retrying..."
                    )
                    await self._reset_voice_client()
                    return False, False
                await message.channel.send(
                    f"Voice gateway closed with code {exc.code}. "
                    "Make sure only py-cord[voice] is installed (no discord.py mix), then restart Ina."
                )
                return True, True
            except discord.errors.ClientException as exc:
                if "Already connected" in str(exc):
                    logger.info("Already connected to %s. Resetting voice client and retrying.", target_channel)
                    await message.channel.send(
                        "Discord thinks I'm still tied to an older voice session. Resetting and trying again..."
                    )
                    await self._reset_voice_client()
                    return False, False
                logger.exception("Voice client exception while joining %s: %s", target_channel, exc)
                await message.channel.send(f"Voice client error: {exc}")
                return True, True
            except Exception:
                logger.exception("Failed to join voice channel %s", target_channel)
                await message.channel.send(f"Failed to join voice channel: {target_channel.name}")
                return True, True

        completed, terminal = await _attempt_join()
        if not completed and not terminal:
            await asyncio.sleep(1.0)
            await _attempt_join("after reset")

    async def _handle_voice_leave(self, message: discord.Message) -> None:
        await self._reset_voice_client()
        await message.channel.send("Left voice channel.")

    async def _ingest_image_attachments(self, message: discord.Message) -> list[dict]:
        level = get_memory_guard_level()
        if level in {"soft", "hard"}:
            logger.info("Skipping Discord image attachments due to memory guard (%s).", level)
            return []
        cfg = get_discord_config()
        if isinstance(cfg, dict):
            allow_images = cfg.get("allow_image_attachments")
            if allow_images is None:
                allow_images = cfg.get("allow_attachments")
            if allow_images is False:
                return []

        attachments = list(message.attachments or [])
        if not attachments:
            return []

        child = get_current_child()
        attachment_dir = _resolve_attachment_dir(child, cfg)
        max_images = _resolve_attachment_count(cfg)
        max_bytes = _resolve_attachment_limit(cfg)
        if max_images == 0:
            return []

        can_process = True  # Native fallback supports PNG/BMP/PNM without optional media packages.
        saved: list[dict] = []
        attachment_dir.mkdir(parents=True, exist_ok=True)
        for idx, attachment in enumerate(attachments):
            if max_images and len(saved) >= max_images:
                break

            content_type = _clean_content_type(getattr(attachment, "content_type", None))
            ext = _resolve_image_extension(getattr(attachment, "filename", ""), content_type)
            if not ext:
                continue

            if attachment.size and max_bytes and attachment.size > max_bytes:
                logger.info(
                    "Skipping image attachment %s (%s bytes > %s limit).",
                    attachment.filename,
                    attachment.size,
                    max_bytes,
                )
                continue

            safe_base = _sanitize_attachment_basename(getattr(attachment, "filename", ""))
            safe_name = f"{message.id}_{idx}_{safe_base}{ext}"
            dest_path = attachment_dir / safe_name
            try:
                data = await attachment.read()
            except Exception:
                logger.exception("Failed to read Discord attachment %s", attachment.filename)
                continue

            if max_bytes and len(data) > max_bytes:
                logger.info(
                    "Skipping image attachment %s (%s bytes > %s limit).",
                    attachment.filename,
                    len(data),
                    max_bytes,
                )
                continue

            try:
                dest_path.write_bytes(data)
            except Exception:
                logger.exception("Failed to save Discord attachment to %s", dest_path)
                continue

            memory_root = Path("AI_Children") / child / "memory"
            try:
                rel_path = str(dest_path.relative_to(memory_root))
            except ValueError:
                rel_path = str(dest_path)

            source_context = {
                "discord_message_id": str(message.id),
                "discord_author_id": str(message.author.id),
                "discord_channel_id": str(message.channel.id),
                "discord_guild_id": str(message.guild.id) if message.guild else None,
                "attachment": {
                    "original_filename": attachment.filename,
                    "content_type": content_type,
                    "size_bytes": len(data),
                },
            }
            tags = ["discord", "image", "attachment", "inbound"]
            tags.append("dm" if message.guild is None else "guild")
            author_name = (
                getattr(message.author, "display_name", None)
                or getattr(message.author, "global_name", None)
                or getattr(message.author, "name", None)
                or str(message.author)
            )
            summary_name = attachment.filename or dest_path.name
            summary = f"Discord image attachment from {author_name}: {summary_name}"
            fragment_id = f"frag_discord_image_{message.id}_{idx}"
            fragment = None
            if can_process:
                fragment = await asyncio.to_thread(
                    _build_discord_image_fragment,
                    path=dest_path,
                    child=child,
                    fragment_id=fragment_id,
                    tags=tags,
                    summary=summary,
                    source_context=source_context,
                    rel_path=rel_path,
                )
            if fragment and fragment.get("stored"):
                logger.info("Discord image stored as fragment %s (%s).", fragment.get("id"), dest_path.name)
            saved.append(
                {
                    "filename": dest_path.name,
                    "original_filename": attachment.filename,
                    "content_type": content_type,
                    "size_bytes": len(data),
                    "saved_path": str(dest_path),
                    "relative_path": rel_path,
                    "fragment_id": fragment.get("id") if fragment and fragment.get("stored") else None,
                    "vision_perception": fragment.get("vision_perception") if fragment else None,
                    "vision_fragment_stored": bool(fragment and fragment.get("stored")),
                }
            )

        return saved

    def _route_to_comms(
        self,
        message: discord.Message,
        *,
        is_dm: bool,
        owner_friend: bool = False,
        high_trust: bool = False,
        image_attachments: Optional[list[dict]] = None,
        conversation_context: Optional[list[dict]] = None,
        edit_analysis: Optional[dict] = None,
        roleplay: bool = False,
    ) -> None:
        sender = make_sender_info_from_discord(
            message,
            backend_name=BACKEND_NAME,
            self_user_id=getattr(self.user, "id", None),
        )
        channel = make_channel_info_from_discord(message, backend_name=BACKEND_NAME)
        metadata = {
            "discord_author_id": str(message.author.id),
            "discord_channel_id": str(message.channel.id),
            "is_dm": is_dm,
            "is_owner_friend": owner_friend,
            "is_high_trust": high_trust,
            "author_is_bot": bool(getattr(message.author, "bot", False)),
            "conversation_context": list(conversation_context or []),
            "is_roleplay_context": roleplay,
        }
        if edit_analysis:
            metadata["message_edit"] = edit_analysis
            metadata["is_edited_message"] = True
        if image_attachments:
            metadata["image_attachments"] = image_attachments
            metadata["image_attachment_count"] = len(image_attachments)
        if message.guild:
            metadata["discord_guild_id"] = str(message.guild.id)

        # Hand this into Ina via CommsCore
        # This will synchronously run the processing pipeline and,
        # if a response is generated, CommsCore will trigger the outbound
        # path which sends a message back using the registered backend.
        self.comms.receive_inbound(
            backend=BACKEND_NAME,
            backend_message_id=str(message.id),
            sender=sender,
            channel=channel,
            text=message.content or "",
            reply_to_backend_id=str(message.id),
            metadata=metadata,
        )

    def _record_social_contact(self, message: discord.Message) -> None:
        """
        Touch the social map entry so trust and recency stay fresh.
        """
        display_name = (
            getattr(message.author, "display_name", None)
            or getattr(message.author, "global_name", None)
            or getattr(message.author, "name", None)
            or str(message.author)
        )
        try:
            update_social_entry(
                message.author.id,
                display_name=display_name,
                last_interaction=datetime.now(timezone.utc).isoformat(),
            )
        except Exception:
            logger.exception("Failed to update social map after contact from %s", message.author)

    def _remember_last_dm_contact(self, message: discord.Message) -> None:
        """
        Keep a lightweight hint in Ina's state about who last reached out via DM.
        """
        try:
            channel = message.channel
            payload = {
                "user_id": str(message.author.id),
                "display_name": getattr(message.author, "display_name", None)
                or getattr(message.author, "global_name", None)
                or getattr(message.author, "name", None)
                or str(message.author),
                "channel_id": str(getattr(channel, "id", "")),
                "channel_name": getattr(channel, "name", None) or "dm",
                "is_dm": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            update_inastate("last_heard_contact", payload)
        except Exception:
            logger.exception("Failed to record last DM contact in inastate.")

    def _read_outbox_flush_request(self) -> Optional[dict]:
        raw = get_inastate("discord_outbox_flush")
        if not isinstance(raw, dict):
            return None
        status = str(raw.get("status") or "requested").strip().lower() or "requested"
        if status in {"completed", "cancelled"}:
            return None
        try:
            burst = max(1, int(raw.get("burst") or self._outbox_policy.get("flush_burst") or self._outbox_policy.get("max_burst") or 24))
        except Exception:
            burst = max(1, int(self._outbox_policy.get("flush_burst") or self._outbox_policy.get("max_burst") or 24))
        stale_mode = str(raw.get("stale_mode") or self._outbox_policy.get("flush_stale_mode") or "drop").strip().lower() or "drop"
        if stale_mode not in {"drop", "archive"}:
            stale_mode = str(self._outbox_policy.get("flush_stale_mode") or "drop").strip().lower() or "drop"
        normalized = dict(raw)
        normalized["status"] = "active"
        normalized["burst"] = burst
        normalized["stale_mode"] = stale_mode
        normalized.setdefault("requested_at", datetime.now(timezone.utc).isoformat())
        now_iso = datetime.now(timezone.utc).isoformat()
        if status != "active" or raw.get("burst") != burst or raw.get("stale_mode") != stale_mode:
            normalized.setdefault("activated_at", now_iso)
            normalized["updated_at"] = now_iso
            update_inastate("discord_outbox_flush", normalized)
        return normalized

    def _complete_outbox_flush(self, request: Optional[dict], **extra) -> None:
        if not isinstance(request, dict):
            return
        payload = dict(request)
        payload.update(extra)
        payload["status"] = "completed"
        payload["completed_at"] = datetime.now(timezone.utc).isoformat()
        payload["updated_at"] = payload["completed_at"]
        update_inastate("discord_outbox_flush", payload)

    def _mark_outbox_entry_without_archive(self, entry: dict, reason: str, *, status: str = "flushed") -> None:
        entry_id = str(entry.get("id") or entry.get("uuid") or entry.get("created_at") or "")
        logger.info("Marked typed outbox entry %s as %s (%s) without archive", entry_id or "<unknown>", status, reason)
        if entry_id:
            self._log_outbox_history(entry_id, status, reason=reason)

    def _read_typed_outbox(self, flush_request: Optional[dict] = None):
        stats = {
            "pending_count": 0,
            "stale_count": 0,
            "flushed_stale_count": 0,
            "archived_stale_count": 0,
            "more_available": False,
        }
        if not self._typed_outbox_path.exists():
            return [], stats
        self._refresh_outbox_history()
        entries = []
        max_batch = int((flush_request or {}).get("burst") or self._outbox_policy.get("max_burst") or 0)
        max_age_minutes = float(self._outbox_policy.get("max_age_minutes") or 0.0)
        expiry_cutoff = (
            datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes) if max_age_minutes > 0 else None
        )
        stale_mode = str((flush_request or {}).get("stale_mode") or self._outbox_policy.get("flush_stale_mode") or "drop").strip().lower() or "drop"
        if stale_mode not in {"drop", "archive"}:
            stale_mode = "drop"
        try:
            with self._typed_outbox_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except Exception:
                        logger.exception("Failed to parse typed outbox line: %s", line[:120])
                        continue
                    entry_id = str(
                        entry.get("id") or entry.get("uuid") or entry.get("created_at") or len(self._typed_outbox_seen)
                    )
                    if entry_id in self._typed_outbox_seen:
                        continue
                    entry["id"] = entry_id
                    if expiry_cutoff and self._entry_is_stale(entry, expiry_cutoff):
                        stats["stale_count"] += 1
                        if flush_request and stale_mode == "drop":
                            self._mark_outbox_entry_without_archive(entry, "flush_stale", status="flushed")
                            stats["flushed_stale_count"] += 1
                        else:
                            # Defer archive to avoid blocking the async loop
                            if "_deferred_archives" not in stats:
                                stats["_deferred_archives"] = []
                            stats["_deferred_archives"].append((entry, "stale_buffer"))
                            stats["archived_stale_count"] += 1
                        continue
                    self._typed_outbox_seen.add(entry_id)
                    entries.append(entry)
                    stats["pending_count"] = len(entries)
                    if max_batch and len(entries) >= max_batch:
                        stats["more_available"] = True
                        break
        except Exception:
            logger.exception("Failed to read typed outbox at %s", self._typed_outbox_path)

        if len(self._typed_outbox_seen) > 5000:
            self._typed_outbox_seen = set(list(self._typed_outbox_seen)[-2000:])
        return entries, stats

    def _load_outbox_history(self) -> None:
        self._typed_outbox_history_offset = 0
        self._refresh_outbox_history()

    def _refresh_outbox_history(self) -> None:
        if not self._typed_outbox_history_path.exists():
            self._typed_outbox_history_offset = 0
            return
        try:
            size = self._typed_outbox_history_path.stat().st_size
            if size < self._typed_outbox_history_offset:
                self._typed_outbox_history_offset = 0
            with self._typed_outbox_history_path.open("r", encoding="utf-8") as fh:
                fh.seek(self._typed_outbox_history_offset)
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except Exception:
                        continue
                    entry_id = str(entry.get("id") or entry.get("entry_id") or "")
                    if entry_id:
                        self._typed_outbox_seen.add(entry_id)
                self._typed_outbox_history_offset = fh.tell()
        except Exception:
            logger.exception("Failed to refresh typed outbox history from %s", self._typed_outbox_history_path)

    def _log_outbox_history(self, entry_id: str, status: str, *, reason: Optional[str] = None) -> None:
        if not entry_id:
            return
        payload = {
            "id": str(entry_id),
            "status": status,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self._typed_outbox_history_path.parent.mkdir(parents=True, exist_ok=True)
            with self._typed_outbox_history_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            logger.exception("Failed to append typed outbox history for entry %s", entry_id)
        self._typed_outbox_seen.add(entry_id)

    def _entry_timestamp(self, entry: dict) -> Optional[datetime]:
        created_at = entry.get("created_at")
        if not created_at:
            return None
        try:
            stamp = datetime.fromisoformat(created_at)
        except Exception:
            return None
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return stamp

    def _entry_is_stale(self, entry: dict, cutoff: datetime) -> bool:
        stamp = self._entry_timestamp(entry)
        if not stamp:
            return False
        return stamp < cutoff

    def _archive_outbox_entry(self, entry: dict, reason: str) -> None:
        entry_id = str(entry.get("id") or entry.get("uuid") or entry.get("created_at") or "")
        archived = {
            **entry,
            "archive_reason": reason,
            "archived_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self._typed_archive_path.parent.mkdir(parents=True, exist_ok=True)
            with self._typed_archive_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(archived, ensure_ascii=False) + "\n")
            logger.info("Archived typed outbox entry %s (%s)", entry_id or "<unknown>", reason)
        except Exception:
            logger.exception("Failed to archive typed outbox entry %s", entry_id or "<unknown>")
        if entry.get("text"):
            try:
                self.history_bridge.log_conversation_turn(
                    entry.get("text", ""),
                    speaker=INA_INSTANCE_NAME,
                    tags=["typed_outbox", "archive", reason],
                    entity_links=[
                        {
                            "type": "typed_outbox_entry",
                            "id": entry_id or entry.get("id") or "",
                            "target": entry.get("target"),
                            "status": "archived",
                            "reason": reason,
                        }
                    ],
                    timestamp=entry.get("created_at") or archived["archived_at"],
                )
            except Exception:
                logger.exception("Failed to log archived outbox entry %s to history bridge", entry_id or "<unknown>")
        if entry_id:
            self._log_outbox_history(entry_id, "archived", reason=reason)

    async def _process_deferred_archives(self, deferred_archives: list) -> None:
        """Process deferred archive operations in a thread to avoid blocking the event loop."""
        if not deferred_archives:
            return
        
        loop = asyncio.get_event_loop()
        for entry, reason in deferred_archives:
            try:
                # Run the blocking archive operation in a thread pool
                await loop.run_in_executor(None, lambda e=entry, r=reason: self._archive_outbox_entry(e, r))
            except Exception:
                logger.exception("Failed to process deferred archive for entry %s", entry.get("id", "<unknown>"))

    async def _pace_discord_send(self) -> None:
        interval = _coerce_nonnegative_float(
            self._outbox_policy.get("min_send_interval_seconds"),
            DEFAULT_DISCORD_SEND_INTERVAL_SECONDS,
        )
        async with self._discord_send_lock:
            now = time.monotonic()
            wait_for = self._next_discord_send_at - now
            if wait_for > 0:
                await asyncio.sleep(wait_for)
            self._next_discord_send_at = time.monotonic() + interval

    async def send_discord_message(
        self,
        destination,
        text: str,
        *,
        file_factory=None,
        reason: str = "message",
        reply_to_message_id: Optional[str] = None,
    ) -> bool:
        """
        Send a Discord message with local pacing and 429-aware retries.

        py-cord already handles ordinary route buckets internally. This wrapper
        adds a small app-level gate for our outbox flushes and honors retry
        headers if Discord still returns a 429 or transient server error.

        The caller's full text remains one outbox/history expression. Splitting
        here represents Discord's transport limit and does not feed a shorter
        version back into Ina's language or memory state.
        """
        retries = _coerce_positive_int(
            self._outbox_policy.get("max_send_retries"),
            DEFAULT_DISCORD_SEND_RETRIES,
        )
        padding = _coerce_nonnegative_float(
            self._outbox_policy.get("rate_limit_padding_seconds"),
            DEFAULT_DISCORD_RATE_LIMIT_PADDING_SECONDS,
        )
        attempts = retries + 1
        chunks = split_discord_message(text)
        reply_reference = None
        get_partial_message = getattr(destination, "get_partial_message", None)
        if reply_to_message_id and callable(get_partial_message):
            try:
                partial_message = get_partial_message(int(reply_to_message_id))
                to_reference = getattr(partial_message, "to_reference", None)
                if callable(to_reference):
                    try:
                        reply_reference = to_reference(fail_if_not_exists=False)
                    except TypeError:
                        reply_reference = to_reference()
                else:
                    reply_reference = partial_message
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid Discord reply target for %s: %r; sending normally.",
                    reason,
                    reply_to_message_id,
                )
        for chunk_index, chunk in enumerate(chunks):
            sent_chunk = False
            for attempt in range(1, attempts + 1):
                file = None
                try:
                    await self._pace_discord_send()
                    file = file_factory() if file_factory and chunk_index == 0 else None
                    send_kwargs = {"file": file}
                    if reply_reference is not None and chunk_index == 0:
                        send_kwargs["reference"] = reply_reference
                        send_kwargs["mention_author"] = False
                    await destination.send(chunk, **send_kwargs)
                    sent_chunk = True
                    break
                except discord.HTTPException as exc:
                    status = getattr(exc, "status", None)
                    retry_after = _discord_retry_after(exc)
                    retryable = status == 429 or (isinstance(status, int) and 500 <= status < 600)
                    if not retryable or attempt >= attempts:
                        logger.exception(
                            "Discord send failed for %s chunk %s/%s after %s/%s attempts (status=%s).",
                            reason,
                            chunk_index + 1,
                            len(chunks),
                            attempt,
                            attempts,
                            status,
                        )
                        return False
                    delay = (retry_after if retry_after is not None else min(2 ** attempt, 30.0)) + padding
                    logger.warning(
                        "Discord send for %s chunk %s/%s hit status %s; retrying in %.2fs (attempt %s/%s).",
                        reason,
                        chunk_index + 1,
                        len(chunks),
                        status,
                        delay,
                        attempt,
                        attempts,
                    )
                    await asyncio.sleep(delay)
                except Exception:
                    logger.exception("Discord send failed for %s chunk %s/%s.", reason, chunk_index + 1, len(chunks))
                    return False
                finally:
                    if file is not None:
                        try:
                            file.close()
                        except Exception:
                            pass
            if not sent_chunk:
                return False
        if len(chunks) > 1:
            logger.info(
                "Delivered full %s-character expression as %s Discord transport chunks (%s).",
                len(text),
                len(chunks),
                reason,
            )
        return True

    def _entry_wants_voice_playback(self, entry: dict, attachment_path: Optional[str]) -> bool:
        if not _attachment_path_is_audio(attachment_path):
            return False
        cfg = get_discord_config()
        if cfg.get("voice_playback_enabled") is False:
            return False
        metadata = entry.get("metadata") if isinstance(entry, dict) else None
        metadata = metadata if isinstance(metadata, dict) else {}
        if metadata.get("voice_target") or metadata.get("delivery") == "discord_voice":
            return True
        if cfg.get("prefer_voice_for_sounds") and metadata.get("source") in {
            "symbol_sequence",
            "word_hint",
            "early_comm",
        }:
            return True
        return bool(cfg.get("voice_play_all_audio_attachments", False))

    async def _maybe_play_voice_attachment(self, entry: dict, attachment_path: Optional[str]) -> bool:
        if not self._entry_wants_voice_playback(entry, attachment_path):
            return False
        path = Path(str(attachment_path))
        if not path.exists() or not path.is_file():
            logger.warning("Voice attachment missing for entry %s: %s", entry.get("id"), attachment_path)
            return False

        ffmpeg_audio = getattr(discord, "FFmpegPCMAudio", None)
        if ffmpeg_audio is None:
            logger.warning("Discord FFmpegPCMAudio unavailable; cannot play %s into voice.", path)
            return False

        if self.voice_channel is None:
            _text_channel, self.voice_channel = resolve_configured_channels(self)
        if self.voice_channel is None:
            logger.warning("No configured voice channel available for voice attachment %s.", path)
            return False

        cfg = get_discord_config()
        timeout = _coerce_nonnegative_float(cfg.get("voice_playback_timeout_seconds"), 120.0) or 120.0
        async with self._voice_playback_lock:
            try:
                voice_client = await self.ensure_voice_connected(self.voice_channel)
                while voice_client.is_playing() or voice_client.is_paused():
                    await asyncio.sleep(0.25)

                loop = asyncio.get_running_loop()
                done = loop.create_future()

                def _after_playback(error):
                    def _finish():
                        if not done.done():
                            done.set_result(error)

                    loop.call_soon_threadsafe(_finish)

                source = ffmpeg_audio(str(path), before_options="-nostdin", options="-vn")
                voice_client.play(source, after=_after_playback)
                error = await asyncio.wait_for(done, timeout=timeout)
                if error:
                    logger.warning("Voice playback failed for %s: %s", path, error)
                    return False
                logger.info("Played Discord voice attachment %s for entry %s.", path, entry.get("id"))
                return True
            except asyncio.TimeoutError:
                logger.warning("Timed out playing Discord voice attachment %s.", path)
                try:
                    if self.voice_client and self.voice_client.is_playing():
                        self.voice_client.stop()
                except Exception:
                    pass
            except Exception:
                logger.exception("Failed to play Discord voice attachment %s.", path)
        return False

    async def _deliver_typed_outbox_entry(self, entry: dict) -> bool:
        text = entry.get("text")
        allow_empty = bool(entry.get("allow_empty"))
        attachment_path = entry.get("attachment_path")

        def _build_file():
            if not attachment_path:
                return None
            try:
                path = Path(attachment_path)
                if not path.exists() or not path.is_file():
                    logger.debug("Attachment path missing for entry %s: %s", entry.get("id"), attachment_path)
                    return None
                return discord.File(str(path), filename=path.name)
            except Exception:
                logger.exception("Failed to prepare attachment for entry %s", entry.get("id"))
                return None

        if text is None:
            if not allow_empty and not attachment_path:
                return False
            text = ""
        text_str = str(text)
        if not text_str.strip() and not allow_empty and not attachment_path:
            logger.debug("Skipping empty typed outbox entry %s", entry.get("id"))
            return False

        target = entry.get("target") or "owner_dm"
        channel_id = entry.get("channel_id")
        target_user_id = entry.get("user_id")
        sent = False
        voice_played = await self._maybe_play_voice_attachment(entry, attachment_path)

        async def _send_dm(user_id: int) -> bool:
            try:
                user = self.get_user(user_id) or await self.fetch_user(user_id)
                if not user:
                    return False
                return await self.send_discord_message(
                    user,
                    text_str,
                    file_factory=_build_file if attachment_path else None,
                    reason=f"typed_outbox:{entry.get('id')}:dm",
                )
            except Exception:
                logger.exception("Failed to DM user %s for typed outbox entry %s", user_id, entry.get("id"))
                return False

        if target_user_id:
            try:
                uid = int(target_user_id)
                if uid == SAKURA_USER_ID or is_high_trust(uid):
                    sent = await _send_dm(uid)
                else:
                    logger.info(
                        "Typed outbox entry %s targets user %s without high trust; skipping.",
                        entry.get("id"),
                        target_user_id,
                    )
            except Exception:
                logger.exception("Invalid user_id on typed outbox entry %s: %s", entry.get("id"), target_user_id)

        if not sent and target == "owner_dm":
            sent = await _send_dm(SAKURA_USER_ID)

        if not sent and target in {"trusted_dm", "high_trust_dm"}:
            contacts = get_high_trust_contacts(limit=1)
            if contacts:
                try:
                    uid = int(contacts[0].get("user_id"))
                    sent = await _send_dm(uid)
                except Exception:
                    logger.exception("Failed to DM high-trust contact for entry %s", entry.get("id"))

        if not sent and channel_id:
            try:
                channel = self.get_channel(int(channel_id)) or await self.fetch_channel(int(channel_id))
                if channel:
                    sent = await self.send_discord_message(
                        channel,
                        text_str,
                        file_factory=_build_file if attachment_path else None,
                        reason=f"typed_outbox:{entry.get('id')}:channel",
                    )
            except Exception:
                logger.exception(
                    "Failed to send typed outbox entry %s to channel %s", entry.get("id"), channel_id
                )

        if not sent and target == "text_channel" and self.text_channel:
            try:
                sent = await self.send_discord_message(
                    self.text_channel,
                    text_str,
                    file_factory=_build_file if attachment_path else None,
                    reason=f"typed_outbox:{entry.get('id')}:configured_text",
                )
            except Exception:
                logger.exception(
                    "Failed to send typed outbox entry %s to configured text channel", entry.get("id")
                )

        if sent or voice_played:
            logger.info(
                "Delivered typed outbox entry %s (target=%s, voice_played=%s, meta=%s)",
                entry.get("id"),
                target,
                voice_played,
                entry.get("metadata"),
            )
            entry_id = entry.get("id")
            if entry_id:
                self._log_outbox_history(str(entry_id), "sent")
        else:
            logger.warning("Unable to deliver typed outbox entry %s; no usable target.", entry.get("id"))
        return sent or voice_played

    async def _watch_io_pressure(self):
        """Measure gateway-loop stalls and ask managed batch I/O to yield."""
        interval = 2.0
        expected = asyncio.get_running_loop().time() + interval
        while not self.is_closed():
            await asyncio.sleep(interval)
            now = asyncio.get_running_loop().time()
            lag = max(0.0, now - expected)
            expected = now + interval
            policy = load_root_config().get("io_pressure", {})
            signal = pressure_signal("discord", lag, policy=policy)
            await asyncio.to_thread(update_inastate, "io_pressure", signal)

    async def _watch_typed_outbox(self):
        while not self.is_closed():
            flush_request = None
            sleep_sec = 3
            try:
                # Both helpers below perform synchronous disk I/O. In
                # particular, an outbox flush may wait on inastate.lock, which
                # is also used by the memory graph. Keep all of it off the
                # gateway event loop so Discord heartbeats remain responsive
                # during sustained storage contention.
                flush_request = await asyncio.to_thread(
                    self._read_outbox_flush_request
                )
                if flush_request:
                    sleep_sec = 1
                pending, stats = await asyncio.to_thread(
                    self._read_typed_outbox,
                    flush_request=flush_request,
                )
                # Process deferred archives asynchronously
                deferred = stats.pop("_deferred_archives", [])
                if deferred:
                    await self._process_deferred_archives(deferred)
                delivered = 0
                for entry in pending:
                    if await self._deliver_typed_outbox_entry(entry):
                        delivered += 1
                if flush_request:
                    now_iso = datetime.now(timezone.utc).isoformat()
                    if stats.get("more_available"):
                        payload = dict(flush_request)
                        payload.update(
                            {
                                "status": "active",
                                "updated_at": now_iso,
                                "last_polled_at": now_iso,
                                "last_batch_size": len(pending),
                                "last_delivered_count": delivered,
                                "last_stale_count": stats.get("stale_count", 0),
                                "last_flushed_stale_count": stats.get("flushed_stale_count", 0),
                                "last_archived_stale_count": stats.get("archived_stale_count", 0),
                            }
                        )
                        await asyncio.to_thread(
                            update_inastate,
                            "discord_outbox_flush",
                            payload,
                        )
                    else:
                        await asyncio.to_thread(
                            self._complete_outbox_flush,
                            flush_request,
                            last_polled_at=now_iso,
                            last_batch_size=len(pending),
                            delivered_count=delivered,
                            stale_count=stats.get("stale_count", 0),
                            flushed_stale_count=stats.get("flushed_stale_count", 0),
                            archived_stale_count=stats.get("archived_stale_count", 0),
                        )
            except Exception:
                logger.exception("Typed outbox dispatch loop failed.")
            await asyncio.sleep(sleep_sec)

    async def _note_language_review_stimulus(self, message: discord.Message) -> None:
        """Raise review pressure from meaningful new conversation evidence."""
        policy = get_language_review_policy()
        if not policy["enabled"] or not (message.content or "").strip():
            return
        raw = await asyncio.to_thread(get_inastate, "discord_language_review")
        state = dict(raw) if isinstance(raw, dict) else {}
        try:
            pending = max(0, int(state.get("pending_messages", 0) or 0)) + 1
        except (TypeError, ValueError):
            pending = 1
        threshold = policy["pressure_messages"]
        live_ids = [
            str(message_id)
            for message_id in state.get("live_vocab_message_ids", [])
            if message_id
        ] if isinstance(state.get("live_vocab_message_ids"), list) else []
        message_id = str(message.id)
        if message_id not in live_ids:
            live_ids.append(message_id)
        live_ids = live_ids[-(policy["history_limit"] * 2):]
        state.update(
            {
                "pending_messages": pending,
                "live_vocab_message_ids": live_ids,
                "pressure": round(min(1.0, pending / threshold), 3),
                "threshold_messages": threshold,
                "last_stimulus_at": datetime.now(timezone.utc).isoformat(),
                "last_message_id": str(message.id),
                "status": "ready" if pending >= threshold else "accumulating",
            }
        )
        await asyncio.to_thread(update_inastate, "discord_language_review", state)
        if pending >= threshold:
            await self._start_language_review_if_ready()

    async def _start_language_review_if_ready(self) -> bool:
        """Start one review only when evidence pressure crosses its threshold."""
        policy = get_language_review_policy()
        if not policy["enabled"]:
            return False
        if self._language_review_task and not self._language_review_task.done():
            return False

        raw = await asyncio.to_thread(get_inastate, "discord_language_review")
        state = dict(raw) if isinstance(raw, dict) else {}
        try:
            pending = max(0, int(state.get("pending_messages", 0) or 0))
        except (TypeError, ValueError):
            pending = 0
        if pending < policy["pressure_messages"]:
            return False
        now = time.time()
        last_review = float(state.get("last_review_epoch", 0.0) or 0.0)
        if last_review and now - last_review < policy["cooldown_seconds"]:
            state["status"] = "cooldown"
            await asyncio.to_thread(update_inastate, "discord_language_review", state)
            return False

        self._language_review_task = asyncio.create_task(self._run_language_review())
        return True

    async def _run_language_review(self, *, force: bool = False) -> dict:
        """Run one bounded history and mapping review pass."""
        current = asyncio.current_task()
        if (
            self._language_review_task
            and not self._language_review_task.done()
            and self._language_review_task is not current
        ):
            return {"status": "already_running", "new_messages": 0, "alignments": 0}

        policy = get_language_review_policy()
        raw = await asyncio.to_thread(get_inastate, "discord_language_review")
        state = dict(raw) if isinstance(raw, dict) else {}
        try:
            starting_pending = max(0, int(state.get("pending_messages", 0) or 0))
        except (TypeError, ValueError):
            starting_pending = 0
        state.update(
            {
                "status": "running",
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        await asyncio.to_thread(update_inastate, "discord_language_review", state)
        try:
            result = await self._ingest_message_history(
                limit=policy["history_limit"],
                mapping_batch=policy["mapping_batch"],
                revisit_mappings=policy["revisit_mappings"],
                force_history=force,
            )
        except Exception as exc:
            logger.exception("Discord language review failed.")
            failed_raw = await asyncio.to_thread(get_inastate, "discord_language_review")
            failed = dict(failed_raw) if isinstance(failed_raw, dict) else state
            failed.update(
                {
                    "status": "failed",
                    "last_error": type(exc).__name__,
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            await asyncio.to_thread(update_inastate, "discord_language_review", failed)
            return {"status": "failed", "new_messages": 0, "alignments": 0}

        latest_raw = await asyncio.to_thread(get_inastate, "discord_language_review")
        latest = dict(latest_raw) if isinstance(latest_raw, dict) else state
        if result.get("status") == "deferred_memory_pressure":
            latest["status"] = "deferred_memory_pressure"
        else:
            now = time.time()
            try:
                latest_pending = max(0, int(latest.get("pending_messages", 0) or 0))
            except (TypeError, ValueError):
                latest_pending = starting_pending
            remaining_pending = max(0, latest_pending - starting_pending)
            reviewed_ids = {
                str(message_id)
                for message_id in result.get("reviewed_message_ids", [])
                if message_id
            }
            live_ids = latest.get("live_vocab_message_ids")
            if isinstance(live_ids, list):
                latest["live_vocab_message_ids"] = [
                    str(message_id)
                    for message_id in live_ids
                    if str(message_id) not in reviewed_ids
                ]
            latest.update(
                {
                    "status": "accumulating" if remaining_pending else "complete",
                    "pending_messages": remaining_pending,
                    "pressure": round(
                        min(1.0, remaining_pending / policy["pressure_messages"]), 3
                    ),
                    "last_review_epoch": now,
                    "last_review_at": datetime.fromtimestamp(now, timezone.utc).isoformat(),
                    "history_cursors": result.get("history_cursors") or latest.get("history_cursors") or {},
                    "last_result": result,
                    "forced": bool(force),
                }
            )
        await asyncio.to_thread(update_inastate, "discord_language_review", latest)
        return result

    async def _ingest_message_history(
        self,
        *,
        limit: int = 75,
        mapping_batch: int = 160,
        revisit_mappings: int = 24,
        force_history: bool = False,
    ) -> dict:
        """Review a bounded, deduplicated slice of Discord history."""
        level = get_memory_guard_level()
        if level in {"soft", "hard"}:
            logger.info("Skipping Discord language review due to memory guard (%s).", level)
            return {
                "status": "deferred_memory_pressure",
                "new_messages": 0,
                "alignments": 0,
            }

        raw_state = await asyncio.to_thread(get_inastate, "discord_language_review")
        review_state = dict(raw_state) if isinstance(raw_state, dict) else {}
        cursors = review_state.get("history_cursors")
        cursors = dict(cursors) if isinstance(cursors, dict) else {}
        next_cursors = dict(cursors)
        live_vocab_ids = {
            str(message_id)
            for message_id in review_state.get("live_vocab_message_ids", [])
            if message_id
        } if isinstance(review_state.get("live_vocab_message_ids"), list) else set()
        reviewed_message_ids = []
        observations = []
        alignments = []
        new_messages = 0
        revisited_messages = 0
        history_diagnostics = {
            "self_authored_messages": 0,
            "human_authored_messages": 0,
            "other_bot_messages": 0,
            "structured_candidates_seen": 0,
            "structured_candidates_nonself": 0,
            "pairing_rejection_counts": {},
        }
        targets = []
        if self.text_channel:
            targets.append(("guild_text", self.text_channel))

        try:
            owner = self.get_user(SAKURA_USER_ID) or await self.fetch_user(SAKURA_USER_ID)
            if owner:
                dm = owner.dm_channel or await owner.create_dm()
                targets.append(("owner_dm", dm))
        except Exception:
            logger.exception("Failed to resolve owner DM channel for history review.")

        for label, channel in targets:
            channel_id = str(channel.id)
            try:
                cursor = int(cursors.get(channel_id, 0) or 0)
            except (TypeError, ValueError):
                cursor = 0
            newest_seen = cursor
            channel_messages = 0
            try:
                async for msg in channel.history(limit=limit, oldest_first=True):
                    try:
                        message_id = int(msg.id)
                    except (TypeError, ValueError):
                        continue
                    already_reviewed = message_id <= cursor
                    if already_reviewed and not force_history:
                        continue
                    if not already_reviewed:
                        newest_seen = max(newest_seen, message_id)
                    content = (msg.content or "").strip()
                    if not content:
                        continue
                    is_ina = bool(self.user and msg.author.id == self.user.id)
                    parsed_message = parse_symbolic_history_message(content)
                    extracted = parsed_message["pairs"]
                    for reason, count in parsed_message["rejection_counts"].items():
                        pairing_counts = history_diagnostics["pairing_rejection_counts"]
                        pairing_counts[reason] = int(pairing_counts.get(reason, 0)) + int(count)
                    history_diagnostics[
                        "self_authored_messages" if is_ina else
                        "other_bot_messages" if msg.author.bot else
                        "human_authored_messages"
                    ] += 1
                    history_diagnostics["structured_candidates_seen"] += len(extracted)
                    if extracted and not is_ina:
                        history_diagnostics["structured_candidates_nonself"] += len(extracted)
                    if msg.author.bot and not is_ina and not extracted:
                        continue
                    if content.lower().startswith("/ina "):
                        continue

                    if already_reviewed:
                        revisited_messages += 1
                    else:
                        new_messages += 1
                        channel_messages += 1
                    reviewed_message_ids.append(str(msg.id))
                    is_dm = label == "owner_dm"
                    tags = [
                        "discord",
                        "history",
                        "structured_alignment" if extracted else
                        "ina" if is_ina else "human",
                    ]
                    if is_dm:
                        tags.append("dm")
                    if extracted:
                        alignments.extend(extracted)
                        continue
                    if already_reviewed:
                        continue
                    if is_ina:
                        observations.append({"text": content, "tags": tags})
                        continue

                    if str(msg.id) in live_vocab_ids:
                        continue
                    observations.append({"text": content, "tags": tags})
                    await asyncio.to_thread(
                        self.history_bridge.log_conversation_turn,
                        content,
                        speaker=msg.author.display_name or str(msg.author),
                        tags=tags,
                        entity_links=[
                            {
                                "type": "discord_message",
                                "message_id": str(msg.id),
                                "author_id": str(msg.author.id),
                                "channel_id": channel_id,
                                "is_dm": is_dm,
                            }
                        ],
                        timestamp=msg.created_at.replace(tzinfo=timezone.utc).isoformat(),
                    )
                if newest_seen > cursor:
                    next_cursors[channel_id] = str(newest_seen)
                logger.info(
                    "Reviewed %s new Discord message(s) from %s.",
                    channel_messages,
                    label,
                )
            except Exception:
                logger.exception("Failed to review Discord history for %s", label)

        evidence_result = await asyncio.to_thread(
            review_text_evidence,
            observations,
            alignments,
            child=self.child,
            source="discord_history_review",
        )
        mapping_updated = await asyncio.to_thread(
            build_text_symbol_links,
            self.child,
            mapping_batch=mapping_batch,
            revisit_existing=revisit_mappings,
        )
        rejection_counts = {}
        for rejection in evidence_result.get("alignment_rejections") or []:
            reason = str(rejection.get("reason") or "unknown")
            rejection_counts[reason] = int(rejection_counts.get(reason, 0)) + 1
        return {
            "status": "complete",
            "new_messages": new_messages,
            "revisited_messages": revisited_messages,
            "force_history": bool(force_history),
            "observations": len(observations),
            "alignment_candidates": int(evidence_result.get("alignment_candidates", 0) or 0),
            "accepted_alignment_candidates": int(
                evidence_result.get("accepted_alignment_candidates", 0) or 0
            ),
            "alignments": len(evidence_result.get("pairs") or []),
            "alignment_rejection_counts": rejection_counts,
            "history_diagnostics": history_diagnostics,
            "mapping_updated": bool(mapping_updated),
            "revisited_budget": revisit_mappings,
            "history_cursors": next_cursors,
            "reviewed_message_ids": reviewed_message_ids,
        }

    def _guild_voice_client(self, guild: Optional[discord.Guild]) -> Optional[discord.VoiceClient]:
        if guild is None:
            return None
        for vc in getattr(self, "voice_clients", []):
            try:
                if vc.guild and vc.guild.id == guild.id:
                    return vc
            except Exception:
                continue
        return None

    async def ensure_voice_connected(self, channel: discord.VoiceChannel) -> discord.VoiceClient:
        """
        Connect or move Ina to the desired voice channel.
        """
        existing = self._guild_voice_client(channel.guild)
        if existing and existing is not self.voice_client:
            self.voice_client = existing

        if self.voice_client and self.voice_client.is_connected():
            if self.voice_client.channel and self.voice_client.channel.id == channel.id:
                await self._ensure_voice_capture()
                return self.voice_client
            await self.voice_client.move_to(channel)
        else:
            try:
                self.voice_client = await channel.connect(reconnect=True)
            except discord.errors.ClientException as exc:
                if "Already connected" in str(exc):
                    logger.info("Discord claims an existing voice session; forcing disconnect before retry.")
                    await self._reset_voice_client()
                    self.voice_client = await channel.connect(reconnect=True)
                else:
                    raise
        self.voice_channel = channel
        await self._ensure_voice_capture()
        return self.voice_client

    async def _ensure_voice_capture(self) -> None:
        """
        Start continuous chunked recording into a pipe/buffer directory if sinks are available.
        """
        if sinks is None:
            logger.warning("discord.sinks not available; voice capture disabled.")
            return
        if not self.voice_client or not self.voice_client.is_connected():
            return
        if not hasattr(self.voice_client, "start_recording"):
            logger.warning(
                "Discord client missing start_recording; install py-cord[voice] to enable voice capture support."
            )
            return
        if self._recording_active:
            return
        self._start_recording_segment()

    def _start_recording_segment(self):
        if sinks is None or not self.voice_client:
            return
        try:
            sink = getattr(sinks, "RawDataSink", None)
            if sink is None:
                sink = getattr(sinks, "RawSink", None)
            sink = sink() if sink else sinks.WaveSink()
        except Exception:
            logger.exception("Failed to create voice sink; voice capture disabled.")
            return
        self._active_sink = sink
        self._recording_active = True
        try:
            self.voice_client.start_recording(sink, self._on_record_complete)
        except Exception:
            self._recording_active = False
            logger.exception("Failed to start voice recording sink.")
            return
        loop = self.loop
        loop.call_later(self.voice_chunk_seconds, self._stop_recording_segment)

    def _stop_recording_segment(self):
        if not self.voice_client or not self._recording_active:
            return
        try:
            self.voice_client.stop_recording()
        except Exception:
            logger.exception("Failed to stop recording sink.")
            self._recording_active = False

    async def _reset_voice_client(self):
        """
        Forcefully disconnect the current voice client and reset recording state.
        """
        targets = set()
        if self.voice_client:
            targets.add(self.voice_client)
        for vc in getattr(self, "voice_clients", []):
            if vc:
                targets.add(vc)

        for vc in targets:
            try:
                await vc.disconnect(force=True)
            except Exception:
                logger.exception("Failed to disconnect voice client during reset.")

        self.voice_client = None
        self.voice_channel = None
        self._recording_active = False
        self._active_sink = None

    def _on_record_complete(self, sink, *args):
        """
        Called by discord.py when a recording segment completes.
        Writes PCM to pipe if configured and WAV chunks to buffer dir.
        """
        self.loop.create_task(self._after_record_complete(sink))

    async def _after_record_complete(self, sink):
        self._recording_active = False
        self._active_sink = None
        if not sink.audio_data:
            logger.warning("Voice sink produced no audio data for this segment.")
        else:
            await self._persist_audio_segment(sink)

        # Schedule next segment if still connected
        if self.voice_client and self.voice_client.is_connected():
            self._start_recording_segment()

    async def _persist_audio_segment(self, sink):
        """
        Persist the first user's audio to FIFO/buffer dir.
        """
        # pick first user entry
        audio_entry = next(iter(sink.audio_data.values()))
        try:
            audio_entry.file.seek(0)
        except Exception:
            pass
        pcm_bytes = audio_entry.file.read()

        # Write to FIFO/pipe if configured
        if self.voice_pipe_path and self.voice_pipe_path.exists():
            try:
                mode = self.voice_pipe_path.stat().st_mode
                is_fifo = (mode & 0o170000) == 0o010000  # stat.S_IFIFO
                if not is_fifo:
                    logger.warning("Configured voice_pipe_path is not a FIFO: %s", self.voice_pipe_path)
                fd = os.open(self.voice_pipe_path, os.O_WRONLY | os.O_NONBLOCK)
                try:
                    os.write(fd, pcm_bytes)
                finally:
                    os.close(fd)
            except Exception:
                logger.exception("Failed to write Discord voice segment to pipe %s", self.voice_pipe_path)

        # Always write to buffer directory as WAV-like raw bytes (named .pcm)
        ts = datetime.now(timezone.utc).isoformat().replace(":", "_")
        ext = ".wav" if sinks and isinstance(sink, getattr(sinks, "WaveSink", ())) else ".pcm"
        out_path = self.voice_buffer_dir / f"{self.voice_label}_{ts}{ext}"
        try:
            out_path.write_bytes(pcm_bytes)
            logger.info("Discord voice segment saved to %s (%d bytes)", out_path, len(pcm_bytes))
        except Exception:
            logger.exception("Failed to persist Discord voice segment to %s", out_path)


# ---------------------------------------------------------------------------
# Bot startup
# ---------------------------------------------------------------------------

def get_discord_token() -> str:
    """
    Load the Discord bot token from either:
    - config.json -> discord.bot_token
    - environment variable DISCORD_BOT_TOKEN
    - secrets.json file in the working directory: {"DISCORD_BOT_TOKEN": "..."}
    """
    cfg = get_discord_config()
    token = cfg.get("bot_token") if cfg else None
    if not token:
        token = load_secret("DISCORD_BOT_TOKEN")
    if not token:
        raise RuntimeError(
            "Discord token not found. Set discord.bot_token in config.json, set "
            "DISCORD_BOT_TOKEN in the environment, or create a secrets.json file "
            "with {'DISCORD_BOT_TOKEN': '...'}"
        )
    return token


def main() -> None:
    discord_cfg = get_discord_config()
    if not bool(discord_cfg.get("enabled", True)):
        logger.info("Discord bridge disabled in config; exiting.")
        return
    if not _acquire_single_instance_lock():
        logger.error("discord_bridge already running; exiting duplicate instance.")
        return
    # Python 3.12+ does not create a default event loop; set one explicitly
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    _install_voice_debug_hooks()
    # Create CommsCore with our custom process_inbound hook
    runtime_log_dir = (
        Path(discord_cfg["runtime_log_dir"])
        if discord_cfg.get("runtime_log_dir")
        else None
    )
    comms = CommsCore(
        instance_name=INA_INSTANCE_NAME,
        process_inbound=process_inbound_message,
        raw_fallback=_log_raw_outbound,
        log_dir=runtime_log_dir,
    )

    # Create Discord client
    client = InaDiscordClient(comms=comms, loop=loop)

    # Register Discord backend with CommsCore so outbound messages work
    register_discord_backend(
        comms,
        client,
        backend_name=BACKEND_NAME,
        reply_to_trigger_message=discord_cfg.get("reply_to_trigger_message", True) is not False,
    )

    log_discord_voice_capabilities()

    token = get_discord_token()

    # Run the Discord client
    client.run(token)


if __name__ == "__main__":
    main()
