
# === raw_file_manager.py (Multimodal Self-Read) ===

import os
import json
import wave
import contextlib
import math
import sys
import atexit
import signal
import io
import zipfile
import tarfile
import gzip
import hashlib
import bz2
import lzma
import uuid
import fnmatch
import random
import re
import time
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from urllib.parse import unquote, urldefrag
from tempfile import NamedTemporaryFile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

_IMAGE_IMPORT_ERROR = None
try:
    from PIL import Image
except Exception as e:  # pragma: no cover - optional dependency
    Image = None
    _IMAGE_IMPORT_ERROR = e

_NUMPY_IMPORT_ERROR = None
try:
    import numpy as np
except Exception as e:  # pragma: no cover - optional dependency
    np = None
    _NUMPY_IMPORT_ERROR = e
from transformers.fractal_multidimensional_transformers import FractalTransformer
from gui_hook import log_to_statusbox
from simple_image_fallback import ImageFallbackError, extract_image_features
from self_read_reporting import is_broken_pipe_error, report_self_read_broken_pipe
from self_read_policy import (
    SELF_READ_FOCUS_ENV,
    VALID_SELF_READ_FOCUS,
    self_read_focus_from_emotions,
)
from self_read_language import (
    annotate_music_language_evidence, attach_media_experience,
    media_seek_fraction, video_language_kind,
)
from text_memory import update_text_vocab
from language_intelligence import reading_span_metadata
from learned_media_lessons import record_media_lesson
from github_history_materializer import materialize_commit_history

_VIDEO_IMPORT_ERROR = None
try:
    import cv2  # type: ignore
except Exception as e:  # pragma: no cover - optional dependency
    cv2 = None
    _VIDEO_IMPORT_ERROR = e

_AUDIO_DIGEST_IMPORT_ERROR = None
try:
    from audio_digest import analyze_audio_clip, generate_fragment
except Exception as e:  # pragma: no cover - import guard
    analyze_audio_clip = None
    generate_fragment = None
    _AUDIO_DIGEST_IMPORT_ERROR = e

_PDF_IMPORT_ERROR = None
_AUDIO_METADATA_IMPORT_ERROR = None
try:
    from pydub.utils import mediainfo_json  # type: ignore
except Exception as e:  # pragma: no cover - optional dependency
    mediainfo_json = None
    _AUDIO_METADATA_IMPORT_ERROR = e

try:
    import fitz  # type: ignore
except Exception as e:  # pragma: no cover - optional dependency
    fitz = None
    _PDF_IMPORT_ERROR = e

try:
    import fcntl
except Exception:  # pragma: no cover - non-POSIX environments
    fcntl = None


FRAG_LIMIT = 1000
TEXT_EXTENSIONS = {".txt", ".md", ".json", ".py", ".lrc", ".srt", ".vtt"}
DOCUMENT_EXTENSIONS = {".pdf", ".odt", ".epub"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".pgm", ".ppm", ".pnm"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".opus"}
VIDEO_EXTENSIONS = {".mp4", ".m4v", ".mov", ".avi", ".webm"}
SIMPLE_COMPRESSED_EXTENSIONS = {".gz", ".bz2", ".xz"}

FILE_SIZE_LIMITS = {
    "text": 5 * 1024 * 1024,        # 5 MB
    "document": 50 * 1024 * 1024,   # 50 MB
    "image": 25 * 1024 * 1024,      # 25 MB
    "audio": 75 * 1024 * 1024,      # 75 MB
    "video": 800 * 1024 * 1024,     # 800 MB
    "archive": 800 * 1024 * 1024,   # 800 MB for compressed bundles
}

ARCHIVE_MEMBER_LIMIT = 50 * 1024 * 1024  # 50 MB per file inside an archive
ARCHIVE_MEMBER_COUNT_LIMIT = 256
ARCHIVE_TOTAL_UNCOMPRESSED_LIMIT = 256 * 1024 * 1024
ARCHIVE_FRAGMENT_LIMIT = 1000

SELF_READ_PREF_FILENAME = "self_read_preferences.json"
SELF_READ_SKIP_REQUESTS = "self_read_skip_requests.json"
SELF_READ_HISTORY_FILENAME = "read_history.json"
# EPUBs are ZIP containers, but they are read as ordered documents rather than
# generic archives. These limits keep malformed books from turning a sparse
# self-read pass into an unbounded decompression job.
EPUB_ENTRY_COUNT_LIMIT = 4096
EPUB_PACKAGE_SIZE_LIMIT = 2 * 1024 * 1024
EPUB_SECTION_SIZE_LIMIT = 4 * 1024 * 1024

SELF_READ_HISTORY_VERSION = 2
VALID_SOURCE_KEYS = {"code", "music", "books", "venv", "github_history"}
SELF_READ_SOURCE_ENV = "SELF_READ_SOURCE"
SELF_READ_REVISIT_LIMIT_ENV = "SELF_READ_REVISIT_LIMIT"
SELF_READ_INSPECTION_LIMIT_ENV = "SELF_READ_INSPECTION_LIMIT"
SELF_READ_SCAN_SECONDS_ENV = "SELF_READ_SCAN_SECONDS"
DEFAULT_SELF_READ_REVISIT_LIMIT = 3
DEFAULT_BALANCED_REVISIT_LIMIT = 1
MAX_SELF_READ_REVISIT_LIMIT = 25
SELF_READ_REVISIT_MIN_AGE_SECONDS = 6 * 3600
SEEN_FOCUS_REVISIT_FRAGMENT_RESERVE = 1
DEFAULT_SELF_READ_INSPECTION_LIMIT = 10_000
MAX_SELF_READ_INSPECTION_LIMIT = 100_000
DEFAULT_SELF_READ_SCAN_SECONDS = 45.0
MAX_SELF_READ_SCAN_SECONDS = 600.0
DEFAULT_CODE_SCAN_PRUNED_DIRS = frozenset(
    {
        "ai_children",
        ".git",
        ".hg",
        ".svn",
        "venv",
        ".venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        ".cache",
        ".eggs",
        "build",
        "dist",
    }
)
MUSIC_SCAN_EXTENSIONS = frozenset(
    AUDIO_EXTENSIONS | VIDEO_EXTENSIONS | IMAGE_EXTENSIONS | {".txt", ".md", ".json", ".lrc", ".srt", ".vtt", ".zip"}
)
# Compatibility name for the existing music-root traversal flag.
AUDIO_ONLY_SCAN_EXTENSIONS = MUSIC_SCAN_EXTENSIONS

DEFAULT_SELF_READ_PREFS = {
    "source_choices": {
        "code": True,
        "music": True,
        "books": True,
        "venv": False,
        "github_history": True,
    },
    # OS-managed swap files are storage infrastructure, not readable content.
    # Match by filename so the protection follows every configured drive.
    "skip_files": [
        "swapfile",
        "swapfile.*",
        "swapfile[0-9]*",
        ".swapfile",
        ".swapfile.*",
    ],
}

SIGNED_MUSIC_ARTISTS = ("rapidcrest",)


INA_MUSIC_TEXT_SOURCE_ANNOTATION = {
    "tags": ["ina_music", "music_language", "music_context"],
    "flags": ["music", "reading_context"],
    "provenance": "ina_music_language_context",
    "ownership": "self_creation",
}


MUSIC_TEXT_SOURCE_ANNOTATION = {
    "tags": ["music", "music_language", "music_context"],
    "flags": ["music", "reading_context"],
    "provenance": "music_language_context",
    "ownership": "unattributed",
}


SOURCE_ANNOTATIONS = {
    "code": {
        "tags": ["self_code", "project_source"],
        "flags": ["self_authored"],
        "provenance": "ina_project_work",
        "ownership": "self_creation",
    },
    "music": {
        "tags": ["ina_music", "self_voice", "audio_memory"],
        "flags": ["self_voice", "music"],
        "provenance": "ina_voice_library",
        "ownership": "self_voice",
    },
    "books": {
        "tags": ["book_library", "external_source"],
        "flags": ["reading", "external"],
        "provenance": "guardian_book_collection",
        "ownership": "external_author",
    },
    "venv": {
        "tags": ["environment", "dependency", "external_source"],
        "flags": ["environment", "external"],
        "provenance": "project_environment",
        "ownership": "environment_dependency",
    },
    "github_history": {
        "tags": ["project_history", "code_evolution", "github"],
        "flags": ["self_history", "read_only"],
        "provenance": "local_git_history",
        "ownership": "project_evolution",
    },
}


class InvalidChildIdentifierError(ValueError):
    """Raised before a child-derived path can escape the managed child tree."""


_CHILD_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")


def validate_child_identifier(value):
    if not isinstance(value, str):
        raise InvalidChildIdentifierError("child identifier must be text")
    identifier = value.strip()
    if not _CHILD_IDENTIFIER_PATTERN.fullmatch(identifier):
        raise InvalidChildIdentifierError(
            "child identifier must contain only letters, digits, '_' or '-'"
        )
    return identifier


def _child_root_path(child_name):
    identifier = validate_child_identifier(child_name)
    managed_root = Path("AI_Children").resolve()
    candidate = (managed_root / identifier).resolve()
    try:
        candidate.relative_to(managed_root)
    except ValueError as exc:
        raise InvalidChildIdentifierError(
            f"child path escapes managed root: {identifier!r}"
        ) from exc
    return candidate


def _child_memory_path(child_name, *parts):
    child_root = _child_root_path(child_name)
    memory_root = (child_root / "memory").resolve()
    try:
        memory_root.relative_to(child_root)
    except ValueError as exc:
        raise InvalidChildIdentifierError(
            "child memory root escapes the managed child directory"
        ) from exc

    candidate = memory_root.joinpath(*parts).resolve()
    try:
        candidate.relative_to(memory_root)
    except ValueError as exc:
        raise InvalidChildIdentifierError(
            f"child memory path escapes managed root: {parts!r}"
        ) from exc
    return candidate


def _default_self_read_prefs():
    return {
        "source_choices": dict(DEFAULT_SELF_READ_PREFS["source_choices"]),
        "skip_files": list(DEFAULT_SELF_READ_PREFS["skip_files"]),
    }


def _load_self_read_source_override():
    value = os.getenv(SELF_READ_SOURCE_ENV)
    if not value:
        return None
    source = value.strip().lower()
    if source in VALID_SOURCE_KEYS:
        return source
    log_to_statusbox(f"[SelfRead] Ignoring invalid {SELF_READ_SOURCE_ENV} '{value}'.")
    return None


def _load_self_read_emotion_values(child):
    path = _child_memory_path(child, "inastate.json")
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
    except Exception as exc:
        log_to_statusbox(f"[SelfRead] Emotion fallback unavailable: {exc}")
        return {}
    snapshot = state.get("emotion_snapshot") if isinstance(state, dict) else {}
    return snapshot if isinstance(snapshot, dict) else {}


def resolve_self_read_focus(child):
    override = str(os.getenv(SELF_READ_FOCUS_ENV) or "").strip().lower()
    snapshot = _load_self_read_emotion_values(child)
    decision = self_read_focus_from_emotions(snapshot or {})
    if override:
        if override in VALID_SELF_READ_FOCUS:
            decision["suggested_focus"] = decision["focus"]
            decision["focus"] = override
            decision["source"] = "environment"
            return decision
        log_to_statusbox(f"[SelfRead] Ignoring invalid {SELF_READ_FOCUS_ENV} '{override}'.")

    if snapshot:
        decision["source"] = "emotion_state"
        return decision
    decision["focus"] = "balanced"
    decision["source"] = "default"
    return decision


def _self_read_revisit_limit(focus):
    default = DEFAULT_SELF_READ_REVISIT_LIMIT
    raw = os.getenv(SELF_READ_REVISIT_LIMIT_ENV)
    if raw is not None:
        try:
            default = max(0, min(MAX_SELF_READ_REVISIT_LIMIT, int(raw)))
        except (TypeError, ValueError):
            log_to_statusbox(
                f"[SelfRead] Ignoring invalid {SELF_READ_REVISIT_LIMIT_ENV} '{raw}'."
            )
    if focus == "new":
        return 0
    if focus == "balanced":
        return min(DEFAULT_BALANCED_REVISIT_LIMIT, default)
    return default


def _bounded_positive_int_env(name, default, maximum):
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        value = int(raw)
        if value <= 0:
            raise ValueError
    except (TypeError, ValueError):
        log_to_statusbox(f"[SelfRead] Ignoring invalid {name} '{raw}'.")
        return int(default)
    return min(int(maximum), value)


def _bounded_positive_float_env(name, default, maximum):
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        value = float(raw)
        if not value > 0.0:
            raise ValueError
    except (TypeError, ValueError):
        log_to_statusbox(f"[SelfRead] Ignoring invalid {name} '{raw}'.")
        return float(default)
    return min(float(maximum), value)


def _self_read_inspection_limit():
    return _bounded_positive_int_env(
        SELF_READ_INSPECTION_LIMIT_ENV,
        DEFAULT_SELF_READ_INSPECTION_LIMIT,
        MAX_SELF_READ_INSPECTION_LIMIT,
    )


def _self_read_scan_seconds():
    return _bounded_positive_float_env(
        SELF_READ_SCAN_SECONDS_ENV,
        DEFAULT_SELF_READ_SCAN_SECONDS,
        MAX_SELF_READ_SCAN_SECONDS,
    )


def _self_read_pref_path(child):
    return _child_memory_path(child, SELF_READ_PREF_FILENAME)


def _skip_requests_path(child):
    return _child_memory_path(child, SELF_READ_SKIP_REQUESTS)


def save_self_read_preferences(child, prefs):
    path = _self_read_pref_path(child)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prefs, f, indent=4)


def load_self_read_preferences(child):
    prefs = _default_self_read_prefs()
    path = _self_read_pref_path(child)
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_save = not path.exists()
    data = {}

    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
        except Exception as e:
            log_to_statusbox(f"[SelfRead] Failed to load {path.name}: {e}")
            needs_save = True

    loaded_choices = data.get("source_choices", {})
    if isinstance(loaded_choices, dict):
        for key, default_value in DEFAULT_SELF_READ_PREFS["source_choices"].items():
            value = loaded_choices.get(key)
            if isinstance(value, bool):
                prefs["source_choices"][key] = value
            else:
                prefs["source_choices"][key] = default_value
                needs_save = True
    else:
        needs_save = True

    skip_files = data.get("skip_files", [])
    if isinstance(skip_files, list):
        sanitized = []
        for entry in skip_files:
            entry_str = str(entry).strip()
            if entry_str:
                sanitized.append(entry_str)
        if sanitized:
            prefs["skip_files"] = sanitized
    else:
        needs_save = True

    if needs_save:
        save_self_read_preferences(child, prefs)

    return prefs


def _apply_skip_requests(child, prefs):
    request_path = _skip_requests_path(child)
    if not request_path.exists():
        return prefs

    try:
        with open(request_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        log_to_statusbox(f"[SelfRead] Failed to read skip requests: {e}")
        return prefs

    if isinstance(payload, dict):
        candidates = payload.get("skip_files") or payload.get("skip") or []
    else:
        candidates = payload

    if not isinstance(candidates, list):
        log_to_statusbox("[SelfRead] Skip requests ignored due to invalid format.")
        try:
            request_path.unlink()
        except OSError:
            pass
        return prefs

    new_entries = []
    for entry in candidates:
        entry_str = str(entry).strip()
        if entry_str and entry_str not in prefs["skip_files"]:
            prefs["skip_files"].append(entry_str)
            new_entries.append(entry_str)

    if new_entries:
        log_to_statusbox(
            "[SelfRead] New skip rules added: " + ", ".join(new_entries[:5])
            + ("..." if len(new_entries) > 5 else "")
        )
        save_self_read_preferences(child, prefs)

    try:
        request_path.unlink()
    except OSError:
        pass

    return prefs


def _match_skip_pattern(path, relative_label, skip_patterns):
    if not skip_patterns:
        return None

    normalized_rel = relative_label.replace("\\", "/") if relative_label else ""
    absolute = str(path)
    filename = path.name

    for pattern in skip_patterns:
        pat = str(pattern).strip()
        if not pat:
            continue
        if (
            fnmatch.fnmatch(normalized_rel, pat)
            or fnmatch.fnmatch(filename, pat)
            or fnmatch.fnmatch(absolute, pat)
        ):
            return pat
    return None


def _derive_book_author_hint(relative_label):
    if not relative_label:
        return None

    clean = relative_label.strip().strip("/")
    if not clean:
        return None

    parts = [segment for segment in clean.split("/") if segment]
    if not parts:
        return None

    if len(parts) > 1:
        return parts[0]

    stem = Path(parts[0]).stem
    return stem or None


def _signed_music_artist_hint(relative_label):
    """Return a signed external artist identified by a music-library path."""
    normalized = str(relative_label or "").replace("\\", "/")
    parts = [part.strip().casefold() for part in normalized.split("/") if part.strip()]
    for artist in SIGNED_MUSIC_ARTISTS:
        artist_key = artist.casefold()
        if any(part.startswith(artist_key) for part in parts):
            return artist
    return None


def _ina_owned_music_path(relative_label):
    """Recognize the explicit Ina-authored namespace in a music library."""
    normalized = str(relative_label or "").replace("\\", "/")
    parts = [part.strip().casefold() for part in normalized.split("/") if part.strip()]
    return any(
        re.match(r"^ina[\s_-]+sings(?:$|[\s_:-])", part)
        for part in parts
    )


def _is_studio_stem_root(base_root):
    root_path = Path(base_root)
    return root_path.name == "stems" and root_path.parent.name == "music_studio"


def _studio_collection_context(relative_label):
    """Return collection and manifest labels for a studio-local relative path."""
    normalized = str(relative_label or "").replace("\\", "/").strip("/")
    if not normalized:
        return "", "manifest.json"
    parent = PurePosixPath(normalized).parent
    collection = "" if parent.as_posix() == "." else parent.as_posix()
    manifest = (
        (parent / "manifest.json").as_posix()
        if collection
        else "manifest.json"
    )
    return collection, manifest


def _source_annotations(
    source_key,
    relative_label,
    modality=None,
    *,
    ina_owned_music=False,
):
    artist = _signed_music_artist_hint(relative_label) if source_key == "music" else None
    if artist:
        return {
            "tags": ["music", "external_music", "signed_artist"],
            "flags": ["music", "external"],
            "provenance": "signed_artist_catalog",
            "ownership": "external_artist",
        }, artist
    if source_key == "music" and modality == "text":
        annotation = (
            INA_MUSIC_TEXT_SOURCE_ANNOTATION
            if ina_owned_music
            else MUSIC_TEXT_SOURCE_ANNOTATION
        )
        return annotation, None
    return SOURCE_ANNOTATIONS.get(source_key), None


def annotate_fragment_source(fragment, source_key, relative_label, base_root):
    fragment["self_read_origin"] = source_key
    context = fragment.setdefault("source_context", {})
    context.setdefault("self_read_origin", source_key)
    context.setdefault("relative_path", relative_label)
    context.setdefault("root_path", str(base_root))

    modality = str(fragment.get("modality") or "").casefold() or None
    studio_stem_root = _is_studio_stem_root(base_root)
    annotations, external_artist = _source_annotations(
        source_key,
        relative_label,
        modality,
        ina_owned_music=(
            source_key == "music"
            and (studio_stem_root or _ina_owned_music_path(relative_label))
        ),
    )
    if not annotations:
        return

    tags = fragment.setdefault("tags", [])
    for tag in annotations.get("tags", []):
        if tag not in tags:
            tags.append(tag)

    annotation_flags = annotations.get("flags", [])
    if annotation_flags:
        metadata = fragment.get("metadata")
        if isinstance(metadata, dict):
            meta_flags = metadata.get("flags") or []
            for flag in annotation_flags:
                if flag not in meta_flags:
                    meta_flags.append(flag)
            metadata["flags"] = meta_flags
        else:
            frag_flags = fragment.setdefault("flags", [])
            for flag in annotation_flags:
                if flag not in frag_flags:
                    frag_flags.append(flag)

    provenance = annotations.get("provenance")
    if provenance and not fragment.get("provenance"):
        fragment["provenance"] = provenance

    ownership = annotations.get("ownership")
    if ownership:
        context.setdefault("ownership_hint", ownership)

    if source_key == "books":
        hint = _derive_book_author_hint(relative_label)
        if hint:
            context.setdefault("external_author_hint", hint)
    elif source_key == "venv":
        component = (relative_label.split("/", 1)[0] if relative_label else "").strip()
        if component:
            context.setdefault("environment_component", component)
        env_file = Path(relative_label).name if relative_label else ""
        if env_file:
            context.setdefault("environment_file", env_file)
    elif source_key == "music":
        archive_category = str(context.get("archive_member_category") or "").casefold()
        archive_member = str(context.get("archive_member_path") or "")
        if external_artist:
            context.setdefault("external_artist_hint", external_artist)
            context.setdefault("catalog_relationship", "signed_artist")

        if modality == "text":
            context_label = archive_member or relative_label
            lowered = str(context_label).casefold()
            asset_kind = (
                "lyrics_style_context"
                if any(token in lowered for token in ("lyric", "style", "prompt"))
                else "music_context"
            )
            for tag in ("music_language", "music_context", asset_kind):
                if tag not in tags:
                    tags.append(tag)
            context.setdefault("music_asset_kind", asset_kind)
            context.setdefault("music_language_reference", Path(context_label).stem)
        elif not external_artist:
            context.setdefault("self_voice_hint", "ina_voice_reference")
            voice_name = Path(relative_label).stem if relative_label else ""
            if voice_name:
                context.setdefault("self_voice_reference", voice_name)

        if archive_category == "audio" or (studio_stem_root and modality == "audio"):
            if "music_stem" not in tags:
                tags.append("music_stem")
            stem_source = archive_member or relative_label
            context.setdefault("music_asset_kind", "stem")
            context.setdefault("stem_label", Path(stem_source).stem)
            if archive_member:
                context.setdefault("stem_container_relative_path", relative_label)

        if studio_stem_root:
            collection_path, manifest_path = _studio_collection_context(relative_label)
            if collection_path:
                context.setdefault("stem_collection_relative_path", collection_path)
            context.setdefault("stem_manifest_relative_path", manifest_path)
            if modality == "text":
                context.setdefault("stem_collection_context", True)

        annotate_music_language_evidence(fragment, relative_label)


def _read_limited(stream, limit):
    data = bytearray()
    while True:
        chunk = stream.read(64 * 1024)
        if not chunk:
            break
        data.extend(chunk)
        if len(data) > limit:
            raise ValueError("archive member exceeds limit")
    return bytes(data)

# === Core Config and State ===
def load_config():
    path = Path("config.json")
    if not path.exists():
        log_to_statusbox("[Pretrain] config.json not found.")
        return {}
    with open(path, "r") as f:
        return json.load(f)

config = load_config()

def _load_path_from_config(key):
    value = config.get(key)
    if not value:
        return None
    try:
        return Path(value).expanduser()
    except TypeError:
        return None

book_folder_path = _load_path_from_config("book_folder_path")
music_folder_path = _load_path_from_config("music_folder_path")
ina_work_path = _load_path_from_config("ina_work_path")
venv_path = _load_path_from_config("venv_path")
if venv_path is None:
    venv_path = Path("venv")

def _command_line_child():
    if __name__ != "__main__":
        return None
    args = list(sys.argv[1:])
    if not args:
        return None
    if args[0] == "--child":
        if len(args) < 2:
            raise InvalidChildIdentifierError("--child requires an identifier")
        return args[1]
    if args[0].startswith("-"):
        return None
    return args[0]


def get_child():
    log_to_statusbox("[RawFileManager] Attempting to retrieve 'child'...")

    environment_child = os.getenv("CHILD")
    if environment_child:
        identifier = validate_child_identifier(environment_child)
        log_to_statusbox(
            f"[RawFileManager] Found 'child' in environment: {identifier}"
        )
        return identifier

    argument_child = _command_line_child()
    if argument_child is not None:
        identifier = validate_child_identifier(argument_child)
        log_to_statusbox(
            f"[RawFileManager] Found 'child' in command line args: {identifier}"
        )
        return identifier

    config_path = Path("config.json")
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                stored_config = json.load(handle)
        except Exception as exc:
            log_to_statusbox(f"[RawFileManager] Error loading config.json: {exc}")
            return validate_child_identifier("Inazuma_Yagami")
        if not isinstance(stored_config, dict):
            raise InvalidChildIdentifierError("config.json must contain an object")
        identifier = validate_child_identifier(
            stored_config.get("current_child", "Inazuma_Yagami")
        )
        log_to_statusbox(
            f"[RawFileManager] Found 'child' in config.json: {identifier}"
        )
        return identifier

    log_to_statusbox(
        "[RawFileManager] No 'child' found, using default: Inazuma_Yagami"
    )
    return validate_child_identifier("Inazuma_Yagami")

child = get_child()

log_to_statusbox(f"[RawFileManager] Final child: {child}")

_SELF_READ_LOCK_HANDLE = None
_SELF_READ_LOCK_HELD = False
_SELF_READ_FINALIZED = False


def _memory_root(child_name=None):
    return _child_memory_path(child_name or child)


def _runtime_lock_path(child_name=None):
    return _memory_root(child_name) / "raw_file_manager.lock"


def _runtime_state_path(child_name=None):
    return _memory_root(child_name) / "raw_file_manager_state.json"


def _atomic_write_json(path, payload, *, sort_keys=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=sort_keys) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def _read_runtime_state(child_name=None):
    path = _runtime_state_path(child_name)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _pid_alive(pid):
    try:
        pid_int = int(pid or 0)
    except Exception:
        return False
    if pid_int <= 0:
        return False
    try:
        os.kill(pid_int, 0)
        return True
    except OSError:
        return False


def _write_runtime_state(status, *, source=None, error=None, **extra):
    state = _read_runtime_state()
    state.update(
        {
            "status": str(status or "unknown"),
            "pid": os.getpid(),
            "source": source or state.get("source") or os.getenv(SELF_READ_SOURCE_ENV) or "all",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    if status == "running":
        state.setdefault("started_at", state["updated_at"])
    if status in {"completed", "failed", "cancelled", "exited"}:
        state["finished_at"] = state["updated_at"]
    if error:
        state["error"] = str(error)[:500]
    for key, value in extra.items():
        state[key] = value
    _atomic_write_json(_runtime_state_path(), state)


def _acquire_runtime_lock():
    global _SELF_READ_LOCK_HANDLE, _SELF_READ_LOCK_HELD
    lock_path = _runtime_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("a+", encoding="utf-8")

    if fcntl is not None:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            state = _read_runtime_state()
            detail = f"pid {state.get('pid')}" if state.get("pid") else "another process"
            log_to_statusbox(f"[SelfRead] Raw file manager already running ({detail}); exiting duplicate.")
            lock_handle.close()
            return False
    else:
        state = _read_runtime_state()
        if str(state.get("status") or "").lower() == "running" and _pid_alive(state.get("pid")):
            log_to_statusbox(f"[SelfRead] Raw file manager already running (pid {state.get('pid')}); exiting duplicate.")
            lock_handle.close()
            return False

    source = os.getenv(SELF_READ_SOURCE_ENV) or "all"
    payload = {
        "pid": os.getpid(),
        "source": source,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    lock_handle.seek(0)
    lock_handle.truncate()
    lock_handle.write(json.dumps(payload, ensure_ascii=True))
    # The held flock is authoritative; the lock payload is only live telemetry.
    lock_handle.flush()

    _SELF_READ_LOCK_HANDLE = lock_handle
    _SELF_READ_LOCK_HELD = True
    _write_runtime_state("running", source=source)
    atexit.register(_release_runtime_lock)
    return True


def _release_runtime_lock(status="exited", *, error=None):
    global _SELF_READ_LOCK_HANDLE, _SELF_READ_LOCK_HELD, _SELF_READ_FINALIZED
    if not _SELF_READ_LOCK_HELD or _SELF_READ_FINALIZED:
        return
    _SELF_READ_FINALIZED = True
    try:
        _write_runtime_state(status, error=error)
    except Exception:
        pass
    lock_handle = _SELF_READ_LOCK_HANDLE
    _SELF_READ_LOCK_HANDLE = None
    _SELF_READ_LOCK_HELD = False
    if lock_handle is not None:
        try:
            if fcntl is not None:
                fcntl.flock(lock_handle, fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            lock_handle.close()
        except Exception:
            pass
    try:
        _runtime_lock_path().unlink(missing_ok=True)
    except Exception:
        pass


def _handle_runtime_signal(signum, frame):
    signal_name = getattr(signal.Signals(signum), "name", str(signum))
    _release_runtime_lock("cancelled", error=signal_name)
    raise SystemExit(128 + int(signum))


def _install_runtime_signal_handlers():
    for sig_name in ("SIGTERM", "SIGINT"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            signal.signal(sig, _handle_runtime_signal)


def classify_suffixes(suffixes):
    if not suffixes:
        return None
    ext = suffixes[-1].lower()
    if ext in TEXT_EXTENSIONS:
        return "text"
    if ext in DOCUMENT_EXTENSIONS:
        return "document"
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in SIMPLE_COMPRESSED_EXTENSIONS:
        return "archive"
    return None


def classify_path(path):
    category = classify_suffixes([s.lower() for s in path.suffixes])
    if category:
        return category
    try:
        if zipfile.is_zipfile(path) or tarfile.is_tarfile(path):
            return "archive"
    except Exception:
        return None
    return None


def is_readable_file(path):
    category = classify_path(path)
    if not category:
        return False
    size_limit = FILE_SIZE_LIMITS.get(category)
    if not size_limit:
        return False
    try:
        return path.stat().st_size <= size_limit
    except FileNotFoundError:
        return False


def _should_prune_default_code_scan(base_root, default_root, source_key):
    """Only the broad project-root scan excludes managed/generated trees."""
    if source_key != "code":
        return False
    try:
        return Path(base_root).resolve() == Path(default_root).resolve()
    except (OSError, RuntimeError):
        return Path(base_root) == Path(default_root)


def _iter_self_read_files_streaming(
    root,
    *,
    audio_only,
    prune_generated,
    stop_requested,
):
    """Depth-first scandir traversal that can stop between individual entries."""
    iterators = []
    try:
        if stop_requested():
            return
        try:
            iterators.append(os.scandir(root))
        except OSError:
            return

        while iterators:
            if stop_requested():
                return
            try:
                entry = next(iterators[-1])
            except StopIteration:
                iterators.pop().close()
                continue
            except OSError:
                iterators.pop().close()
                continue

            if stop_requested():
                return
            try:
                is_directory = entry.is_dir(follow_symlinks=False)
            except OSError:
                continue

            if is_directory:
                if (
                    prune_generated
                    and entry.name.casefold() in DEFAULT_CODE_SCAN_PRUNED_DIRS
                ):
                    continue
                try:
                    iterators.append(os.scandir(entry.path))
                except OSError:
                    continue
                continue

            path = Path(entry.path)
            if (
                audio_only
                and path.suffix.casefold() not in AUDIO_ONLY_SCAN_EXTENSIONS
            ):
                continue
            yield path
    finally:
        for iterator in reversed(iterators):
            try:
                iterator.close()
            except Exception:
                pass


def _iter_self_read_files(
    base_root,
    *,
    audio_only=False,
    prune_generated=False,
    stop_requested=None,
):
    """Yield files with deterministic legacy or entry-streaming bounded traversal."""
    root = Path(base_root)
    if stop_requested is not None:
        yield from _iter_self_read_files_streaming(
            root,
            audio_only=audio_only,
            prune_generated=prune_generated,
            stop_requested=stop_requested,
        )
        return

    for directory, dirnames, filenames in os.walk(
        root,
        topdown=True,
        followlinks=False,
    ):
        if prune_generated:
            dirnames[:] = [
                name
                for name in dirnames
                if name.casefold() not in DEFAULT_CODE_SCAN_PRUNED_DIRS
            ]
        dirnames.sort(key=str.casefold)
        filenames.sort(key=str.casefold)
        directory_path = Path(directory)
        for filename in filenames:
            path = directory_path / filename
            if audio_only and path.suffix.casefold() not in AUDIO_ONLY_SCAN_EXTENSIONS:
                continue
            yield path


class SelfReadHistoryLoadError(RuntimeError):
    """Raised when an existing history ledger cannot be trusted safely."""


def _empty_read_history():
    return {
        "version": SELF_READ_HISTORY_VERSION,
        "updated_at": None,
        "files": {},
    }


def _legacy_history_record():
    return {
        "read_count": 1,
        "first_read_at": None,
        "last_read_at": None,
        "last_read_reason": "legacy",
        "mtime_ns": None,
        "size_bytes": None,
        "legacy_migrated": True,
    }


def _validate_history_fingerprint(record, key, path):
    mtime = record.get("mtime_ns")
    size = record.get("size_bytes")
    if mtime is None and size is None:
        return
    if (
        isinstance(mtime, bool)
        or not isinstance(mtime, int)
        or mtime < 0
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
    ):
        raise SelfReadHistoryLoadError(
            f"invalid fingerprint for {key!r} in {path}"
        )


def _validate_history_continuation(record, key, path):
    if "continuation" not in record:
        return
    continuation = record.get("continuation")
    if not isinstance(continuation, dict):
        raise SelfReadHistoryLoadError(
            f"invalid continuation for {key!r} in {path}"
        )
    offset = continuation.get("offset")
    total = continuation.get("total_fragments")
    fingerprint = continuation.get("fingerprint")
    if (
        isinstance(offset, bool)
        or not isinstance(offset, int)
        or offset < 0
        or isinstance(total, bool)
        or not isinstance(total, int)
        or total <= 0
        or offset >= total
        or not isinstance(fingerprint, dict)
    ):
        raise SelfReadHistoryLoadError(
            f"invalid continuation for {key!r} in {path}"
        )
    document = continuation.get("document")
    if document is not None:
        cursor = document.get("cursor") if isinstance(document, dict) else None
        section = cursor.get("section") if isinstance(cursor, dict) else None
        char = cursor.get("char") if isinstance(cursor, dict) else None
        if (
            not isinstance(document, dict)
            or document.get("format") != "epub"
            or not isinstance(cursor, dict)
            or isinstance(section, bool)
            or not isinstance(section, int)
            or section < 0
            or isinstance(char, bool)
            or not isinstance(char, int)
            or char < 0
        ):
            raise SelfReadHistoryLoadError(
                f"invalid document continuation for {key!r} in {path}"
            )
    continuation_mtime = fingerprint.get("mtime_ns")
    continuation_size = fingerprint.get("size_bytes")
    if (
        isinstance(continuation_mtime, bool)
        or not isinstance(continuation_mtime, int)
        or continuation_mtime < 0
        or isinstance(continuation_size, bool)
        or not isinstance(continuation_size, int)
        or continuation_size < 0
        or continuation_mtime != record.get("mtime_ns")
        or continuation_size != record.get("size_bytes")
    ):
        raise SelfReadHistoryLoadError(
            f"invalid continuation fingerprint for {key!r} in {path}"
        )


def load_history(child):
    """Load the inspectable v2 per-file ledger, accepting the legacy string list."""
    path = _child_memory_path(child, SELF_READ_HISTORY_FILENAME)
    if not path.exists():
        return _empty_read_history()
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except Exception as exc:
        raise SelfReadHistoryLoadError(
            f"cannot read {path}: {exc}"
        ) from exc

    ledger = _empty_read_history()
    files = ledger["files"]
    if isinstance(raw, list):
        for value in raw:
            if not isinstance(value, str) or not value.strip():
                raise SelfReadHistoryLoadError(
                    f"invalid legacy entry in {path}"
                )
            files.setdefault(value.strip(), _legacy_history_record())
        ledger["migration"] = {
            "from": "legacy_string_list",
            "migrated_at": datetime.now(timezone.utc).isoformat(),
        }
        return ledger

    if not isinstance(raw, dict):
        raise SelfReadHistoryLoadError(f"invalid top-level format in {path}")

    version = raw.get("version")
    if version is not None and version != SELF_READ_HISTORY_VERSION:
        raise SelfReadHistoryLoadError(
            f"unsupported history version {version!r} in {path}"
        )

    if "files" in raw:
        raw_files = raw.get("files")
        if not isinstance(raw_files, dict):
            raise SelfReadHistoryLoadError(f"invalid files map in {path}")
    elif "entries" in raw:
        # Tolerate an early map-only ledger shape if one was written by a prototype.
        raw_files = raw.get("entries")
        if not isinstance(raw_files, dict):
            raise SelfReadHistoryLoadError(f"invalid entries map in {path}")
    else:
        raise SelfReadHistoryLoadError(f"missing files map in {path}")

    if "migration" in raw and not isinstance(raw.get("migration"), dict):
        raise SelfReadHistoryLoadError(f"invalid migration metadata in {path}")
    if "last_pass" in raw and not isinstance(raw.get("last_pass"), dict):
        raise SelfReadHistoryLoadError(f"invalid last-pass metadata in {path}")

    for raw_key, raw_record in raw_files.items():
        if not isinstance(raw_key, str) or not raw_key.strip():
            raise SelfReadHistoryLoadError(f"invalid file key in {path}")
        key = raw_key.strip()
        if not isinstance(raw_record, dict):
            raise SelfReadHistoryLoadError(
                f"invalid record for {key!r} in {path}"
            )
        record = dict(raw_record)
        try:
            record["read_count"] = max(1, int(record.get("read_count") or 1))
        except (TypeError, ValueError):
            record["read_count"] = 1
        record.setdefault("first_read_at", record.get("last_read_at"))
        record.setdefault("last_read_at", None)
        record.setdefault("last_read_reason", "legacy")
        record.setdefault("mtime_ns", None)
        record.setdefault("size_bytes", None)
        _validate_history_fingerprint(record, key, path)
        _validate_history_continuation(record, key, path)
        files[key] = record

    ledger["updated_at"] = raw.get("updated_at")
    if isinstance(raw.get("migration"), dict):
        ledger["migration"] = raw["migration"]
    if isinstance(raw.get("last_pass"), dict):
        ledger["last_pass"] = raw["last_pass"]
    return ledger


def save_history(child, history):
    """Atomically save a deterministic, non-truncated per-file read ledger."""
    path = _child_memory_path(child, SELF_READ_HISTORY_FILENAME)
    files = history.get("files") if isinstance(history, dict) else {}
    files = files if isinstance(files, dict) else {}
    payload = {
        "version": SELF_READ_HISTORY_VERSION,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    if isinstance(history, dict) and isinstance(history.get("migration"), dict):
        payload["migration"] = history["migration"]
    if isinstance(history, dict) and isinstance(history.get("last_pass"), dict):
        payload["last_pass"] = history["last_pass"]
    _atomic_write_json(path, payload, sort_keys=True)


def self_read_history_key(source_key, base_root, relative_path):
    """Build a collision-resistant key while keeping the source/path inspectable."""
    try:
        root_identity = str(Path(base_root).resolve())
    except OSError:
        root_identity = str(base_root)
    relative = str(relative_path or "").replace("\\", "/")
    return f"{source_key}|{root_identity}|{relative}"


def _old_self_read_history_key(base_root, relative_path):
    return f"{Path(base_root).name}/{relative_path}"


def _resolve_history_record(
    history_files,
    *,
    source_key,
    base_root,
    relative_path,
    allow_legacy_basename=False,
):
    """
    Find a canonical or legacy entry and migrate old keys without rereading.
    """
    canonical = self_read_history_key(source_key, base_root, relative_path)
    candidates = [canonical, _old_self_read_history_key(base_root, relative_path)]
    if allow_legacy_basename:
        candidates.append(Path(relative_path).name)

    for candidate_key in candidates:
        prior = history_files.get(candidate_key)
        if not isinstance(prior, dict):
            continue
        if candidate_key != canonical:
            migrated = dict(prior)
            migrated["migrated_from_key"] = candidate_key
            existing = history_files.get(canonical)
            if not isinstance(existing, dict):
                history_files[canonical] = migrated
                prior = migrated
            else:
                prior = existing
            history_files.pop(candidate_key, None)
        return canonical, prior
    return canonical, None


def _file_stamp(path):
    try:
        stat = path.stat()
    except OSError:
        return {}
    return {
        "mtime_ns": max(0, int(stat.st_mtime_ns)),
        "size_bytes": max(0, int(stat.st_size)),
    }


def classify_self_read_file(prior, stamp):
    """Return new/updated/resume, or None for an unchanged legacy/seen file."""
    if not isinstance(prior, dict):
        return "new"
    if not isinstance(stamp, dict) or not stamp:
        return None
    old_mtime = prior.get("mtime_ns")
    old_size = prior.get("size_bytes")
    if old_mtime is None or old_size is None:
        return None
    try:
        changed = (
            int(old_mtime) != int(stamp.get("mtime_ns"))
            or int(old_size) != int(stamp.get("size_bytes"))
        )
    except (TypeError, ValueError):
        return "updated"
    if changed:
        return "updated"
    return "resume" if _self_read_continuation(prior, stamp) is not None else None


def _fingerprint_matches(fingerprint, stamp):
    if not isinstance(fingerprint, dict) or not isinstance(stamp, dict):
        return False
    try:
        return (
            int(fingerprint.get("mtime_ns")) == int(stamp.get("mtime_ns"))
            and int(fingerprint.get("size_bytes")) == int(stamp.get("size_bytes"))
        )
    except (TypeError, ValueError):
        return False


def _self_read_continuation(prior, stamp):
    """Return a continuation only when it belongs to this file fingerprint."""
    continuation = prior.get("continuation") if isinstance(prior, dict) else None
    if not isinstance(continuation, dict):
        return None
    if not _fingerprint_matches(continuation.get("fingerprint"), stamp):
        return None
    return continuation


def _self_read_resume_offset(prior, stamp):
    continuation = _self_read_continuation(prior, stamp)
    if continuation is None:
        return 0
    try:
        offset = int(continuation.get("offset") or 0)
    except (TypeError, ValueError):
        return 0
    return max(0, offset)


def _epub_cursor_from_history(prior, stamp):
    continuation = _self_read_continuation(prior, stamp)
    document = continuation.get("document") if continuation else None
    if not isinstance(document, dict) or document.get("format") != "epub":
        return None
    return _normalize_epub_cursor(document.get("cursor"))


def _self_read_fragment_window(result, prior, stamp, budget):
    """Slice the next unsaved fragment window for a fingerprint-bound cursor."""
    fragments = list(result or [])
    start = min(_self_read_resume_offset(prior, stamp), len(fragments))
    try:
        capacity = max(0, int(budget))
    except (TypeError, ValueError):
        capacity = 0
    end = min(len(fragments), start + capacity)
    return fragments[start:end], start, end, len(fragments)


def _set_self_read_continuation(
    record,
    stamp,
    *,
    next_offset,
    total_fragments,
    document_cursor=None,
):
    """Persist a fragment cursor, an EPUB cursor, or both."""
    try:
        next_value = max(0, int(next_offset))
        total_value = max(0, int(total_fragments))
    except (TypeError, ValueError):
        next_value = 0
        total_value = 0

    normalized_document_cursor = (
        _normalize_epub_cursor(document_cursor)
        if isinstance(document_cursor, dict)
        else None
    )
    fragment_incomplete = next_value < total_value
    record.pop("continuation", None)
    if fragment_incomplete or normalized_document_cursor is not None:
        record["continuation"] = {
            "offset": next_value if fragment_incomplete else 0,
            "total_fragments": total_value if fragment_incomplete else 1,
            "fingerprint": {
                "mtime_ns": stamp.get("mtime_ns"),
                "size_bytes": stamp.get("size_bytes"),
            },
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if normalized_document_cursor is not None:
            record["continuation"]["document"] = {
                "format": "epub",
                "cursor": normalized_document_cursor,
            }
    return record


def _primary_fragment_ceiling(focus, fragment_limit=FRAG_LIMIT):
    """Keep one fragment slot for a due revisit only when seen focus asks for it."""
    try:
        limit = max(0, int(fragment_limit))
    except (TypeError, ValueError):
        limit = 0
    reserve = SEEN_FOCUS_REVISIT_FRAGMENT_RESERVE if focus == "seen" else 0
    return max(0, limit - min(limit, reserve))


def _parse_history_timestamp(value):
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError, OverflowError):
        return None


def _revisit_is_due(record, *, now_ts, min_age_seconds):
    last_ts = _parse_history_timestamp(record.get("last_read_at")) if isinstance(record, dict) else None
    if last_ts is None:
        return True
    return (now_ts - last_ts) >= max(0.0, float(min_age_seconds))


def select_revisit_candidates(
    candidates,
    focus,
    *,
    limit=None,
    now_ts=None,
    min_age_seconds=SELF_READ_REVISIT_MIN_AGE_SECONDS,
):
    """Choose oldest unchanged files deterministically and within the pass quota."""
    if focus == "new":
        return []
    capacity = _self_read_revisit_limit(focus) if limit is None else max(0, int(limit))
    if focus == "balanced":
        capacity = min(DEFAULT_BALANCED_REVISIT_LIMIT, capacity)
    if capacity <= 0:
        return []

    now_value = datetime.now(timezone.utc).timestamp() if now_ts is None else float(now_ts)
    eligible = [
        item
        for item in candidates
        if isinstance(item, dict)
        and _revisit_is_due(
            item.get("prior") or {},
            now_ts=now_value,
            min_age_seconds=min_age_seconds,
        )
    ]
    eligible.sort(
        key=lambda item: (
            _parse_history_timestamp((item.get("prior") or {}).get("last_read_at"))
            if _parse_history_timestamp((item.get("prior") or {}).get("last_read_at")) is not None
            else float("-inf"),
            str(item.get("history_key") or ""),
        )
    )
    return eligible[:capacity]


def _backfill_legacy_stamp(prior, stamp):
    record = dict(prior or {})
    if record.get("mtime_ns") is None:
        record["mtime_ns"] = stamp.get("mtime_ns")
    if record.get("size_bytes") is None:
        record["size_bytes"] = stamp.get("size_bytes")
    record["last_observed_at"] = datetime.now(timezone.utc).isoformat()
    record["legacy_migrated"] = True
    return record


def _next_history_record(prior, stamp, *, read_reason, source_key, relative_path, base_root):
    now_iso = datetime.now(timezone.utc).isoformat()
    prior = prior if isinstance(prior, dict) else {}
    try:
        prior_count = max(0, int(prior.get("read_count") or 0))
    except (TypeError, ValueError):
        prior_count = 0
    record = {
        "source": source_key,
        "relative_path": relative_path,
        "root_path": str(base_root),
        "mtime_ns": stamp.get("mtime_ns"),
        "size_bytes": stamp.get("size_bytes"),
        "first_read_at": prior.get("first_read_at") or prior.get("last_read_at") or now_iso,
        "last_read_at": now_iso,
        "read_count": prior_count + 1,
        "last_read_reason": read_reason,
    }
    if prior:
        record["previous_read"] = {
            "last_read_at": prior.get("last_read_at"),
            "read_count": prior_count,
            "last_read_reason": prior.get("last_read_reason"),
            "mtime_ns": prior.get("mtime_ns"),
            "size_bytes": prior.get("size_bytes"),
            "fragment_ids": list(prior.get("last_fragment_ids") or [])[:5],
        }
    return record


def annotate_fragment_read_lineage(fragment, *, read_reason, prior, record, focus):
    tags = fragment.setdefault("tags", [])
    reason_tag = f"self_read_{read_reason}"
    if reason_tag not in tags:
        tags.append(reason_tag)
    context = fragment.setdefault("source_context", {})
    context["read_reason"] = read_reason
    context["read_focus"] = focus
    context["read_count"] = int(record.get("read_count") or 1)
    context["source_fingerprint"] = {
        "mtime_ns": record.get("mtime_ns"),
        "size_bytes": record.get("size_bytes"),
    }
    if isinstance(prior, dict):
        context["prior_read"] = {
            "last_read_at": prior.get("last_read_at"),
            "read_count": int(prior.get("read_count") or 1),
            "last_read_reason": prior.get("last_read_reason"),
            "mtime_ns": prior.get("mtime_ns"),
            "size_bytes": prior.get("size_bytes"),
            "fragment_ids": list(prior.get("last_fragment_ids") or [])[:5],
        }
    return fragment

def log_reflection(child, fragment):
    path = _child_root_path(child) / "identity" / "self_reflection.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "r") as f:
            reflection = json.load(f)
    except:
        reflection = {}

    history = reflection.get("self_read_fragments", [])
    history.append({
        "timestamp": fragment["timestamp"],
        "summary": fragment.get("summary", "")[:60],
        "filename": fragment.get("source")
    })
    reflection["self_read_fragments"] = history[-100:]

    with open(path, "w") as f:
        json.dump(reflection, f, indent=4)

def _normalize_document_text(text):
    if not text:
        return ""
    cleaned = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = []
    current = []
    for raw_line in cleaned.split("\n"):
        line = " ".join(raw_line.split())
        if line:
            current.append(line)
        elif current:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)


def _document_chunk_starts(length, chunk_size, max_chunks, seed):
    if length <= chunk_size:
        return [0]
    if length <= chunk_size * max_chunks:
        return list(range(0, length, chunk_size))[:max_chunks]

    rng = random.Random(seed)
    starts = {
        0,
        max(0, (length // 2) - (chunk_size // 2)),
        max(0, length - chunk_size),
    }
    while len(starts) < max_chunks:
        starts.add(rng.randint(0, length - chunk_size))
    return sorted(starts)[:max_chunks]


def _written_passages(text, target_chars=1200):
    """Split storage at written boundaries while preserving each reading unit."""
    cleaned = _normalize_document_text(text)
    if not cleaned:
        return []
    target = max(200, int(target_chars))
    passages = []
    start = 0
    while start < len(cleaned):
        remaining = len(cleaned) - start
        if remaining <= target:
            end = len(cleaned)
        else:
            ceiling = start + target
            floor = start + max(80, target // 2)
            paragraph_boundary = cleaned.rfind("\n\n", floor, ceiling + 1)
            sentence_boundary = max(
                cleaned.rfind(mark, floor, ceiling + 1)
                for mark in (". ", "? ", "! ")
            )
            word_boundary = cleaned.rfind(" ", floor, ceiling + 1)
            boundary = next(
                (
                    candidate
                    for candidate in (
                        paragraph_boundary, sentence_boundary, word_boundary
                    )
                    if candidate >= floor
                ),
                -1,
            )
            if boundary < floor:
                end = ceiling
            else:
                marker = cleaned[boundary:boundary + 2]
                end = boundary + (1 if marker in {". ", "? ", "! "} else 0)
        passage = cleaned[start:end].strip()
        if passage:
            passages.append(passage)
        start = end
        while start < len(cleaned) and cleaned[start].isspace():
            start += 1
    return passages


def _document_chunks(text, source, chunk_size=1200, max_chunks=5):
    passages = _written_passages(text, target_chars=chunk_size)
    if len(passages) <= max_chunks:
        return passages
    seed = int(
        hashlib.sha256(str(source).encode("utf-8", errors="replace")).hexdigest()[:8],
        16,
    )
    indexes = _document_chunk_starts(len(passages), 1, max_chunks, seed)
    return [passages[index] for index in indexes]


def _limit_text(text, limit):
    if not text:
        return ""
    if limit and len(text) > limit:
        return text[:limit]
    return text


def _extract_pdf_text(path, *, max_pages=10, max_chars=12000):
    if fitz is None:
        log_to_statusbox(f"[RawFileManager] PDF support unavailable: {_PDF_IMPORT_ERROR}")
        return ""
    try:
        total = 0
        parts = []
        with fitz.open(path) as doc:
            for index, page in enumerate(doc):
                if index >= max_pages:
                    break
                text = page.get_text("text") or ""
                if not text:
                    continue
                parts.append(text)
                total += len(text)
                if total >= max_chars:
                    break
        return _limit_text("".join(parts), max_chars)
    except Exception as e:
        log_to_statusbox(f"[RawFileManager] Failed to read PDF {path}: {e}")
        return ""


def _extract_pdf_text_bytes(data, source_label, *, max_pages=10, max_chars=12000):
    if fitz is None:
        log_to_statusbox(f"[RawFileManager] PDF support unavailable: {_PDF_IMPORT_ERROR}")
        return ""
    try:
        total = 0
        parts = []
        with fitz.open(stream=data, filetype="pdf") as doc:
            for index, page in enumerate(doc):
                if index >= max_pages:
                    break
                text = page.get_text("text") or ""
                if not text:
                    continue
                parts.append(text)
                total += len(text)
                if total >= max_chars:
                    break
        return _limit_text("".join(parts), max_chars)
    except Exception as e:
        log_to_statusbox(f"[RawFileManager] Failed to read PDF {source_label}: {e}")
        return ""


def _extract_odt_text_bytes(data, source_label, *, max_chars=12000):
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if "content.xml" not in archive.namelist():
                log_to_statusbox(f"[RawFileManager] ODT missing content.xml: {source_label}")
                return ""
            raw = archive.read("content.xml")
    except Exception as e:
        log_to_statusbox(f"[RawFileManager] Failed to read ODT {source_label}: {e}")
        return ""

    try:
        root = ET.fromstring(raw)
        text = " ".join(root.itertext())
    except Exception as e:
        log_to_statusbox(f"[RawFileManager] Failed to parse ODT {source_label}: {e}")
        return ""

    return _limit_text(text, max_chars)


def _extract_odt_text(path, *, max_chars=12000):
    try:
        with open(path, "rb") as f:
            data = f.read()
    except Exception as e:
        log_to_statusbox(f"[RawFileManager] Failed to read ODT {path}: {e}")
        return ""
    return _extract_odt_text_bytes(data, str(path), max_chars=max_chars)


class _EPUBHTMLTextParser(HTMLParser):
    """Small dependency-free fallback for imperfect XHTML chapter files."""

    _IGNORED_TAGS = {"head", "script", "style", "noscript", "svg"}
    _BREAK_TAGS = {
        "address", "article", "aside", "blockquote", "br", "div", "footer",
        "h1", "h2", "h3", "h4", "h5", "h6", "header", "li", "main", "nav",
        "p", "pre", "section", "table", "tr",
    }

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self._ignored_depth = 0

    def handle_starttag(self, tag, attrs):
        tag = tag.casefold()
        if tag in self._IGNORED_TAGS:
            self._ignored_depth += 1
        elif not self._ignored_depth and tag in self._BREAK_TAGS:
            self.parts.append("\n")

    def handle_startendtag(self, tag, attrs):
        if not self._ignored_depth and tag.casefold() in self._BREAK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        tag = tag.casefold()
        if tag in self._IGNORED_TAGS and self._ignored_depth:
            self._ignored_depth -= 1
        elif not self._ignored_depth and tag in self._BREAK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._ignored_depth:
            self.parts.append(data)


def _xml_local_name(tag):
    return str(tag).rsplit("}", 1)[-1].casefold()


def _safe_epub_member_name(package_name, href):
    """Resolve an OPF-relative href without allowing it above the EPUB root."""
    raw_href = unquote(urldefrag(str(href or ""))[0]).replace("\\", "/")
    href_path = PurePosixPath(raw_href)
    if not raw_href or href_path.is_absolute():
        return None

    parts = []
    for part in PurePosixPath(package_name).parent.parts + href_path.parts:
        if part in ("", "."):
            continue
        if part == "..":
            if not parts:
                return None
            parts.pop()
        else:
            parts.append(part)
    return "/".join(parts) or None


def _read_epub_member(archive, member_name, size_limit):
    info = archive.getinfo(member_name)
    if info.is_dir() or info.file_size > size_limit:
        raise ValueError(f"EPUB member exceeds limit: {member_name}")
    with archive.open(info, "r") as member:
        return _read_limited(member, size_limit)


def _epub_section_text(data):
    try:
        root = ET.fromstring(data)
        body = next(
            (element for element in root.iter() if _xml_local_name(element.tag) == "body"),
            root,
        )
        ignored = {"head", "script", "style", "noscript", "svg"}
        block_tags = _EPUBHTMLTextParser._BREAK_TAGS
        parts = []

        def append_element_text(element):
            tag = _xml_local_name(element.tag)
            if tag in ignored:
                if element.tail:
                    parts.append(element.tail)
                return
            if tag in block_tags:
                parts.append("\n")
            if element.text:
                parts.append(element.text)
            for child_element in element:
                append_element_text(child_element)
            if tag in block_tags:
                parts.append("\n")
            if element.tail:
                parts.append(element.tail)

        append_element_text(body)
        return _normalize_document_text("".join(parts))
    except ET.ParseError:
        parser = _EPUBHTMLTextParser()
        parser.feed(data.decode("utf-8", errors="replace"))
        parser.close()
        return _normalize_document_text("".join(parser.parts))


def _normalize_epub_cursor(cursor):
    cursor = cursor if isinstance(cursor, dict) else {}
    try:
        section = max(0, int(cursor.get("section") or 0))
        char = max(0, int(cursor.get("char") or 0))
    except (TypeError, ValueError):
        section, char = 0, 0
    return {"section": section, "char": char}


def _extract_epub_archive_text(
    archive,
    source_label,
    *,
    max_chars=12000,
    cursor=None,
    with_progress=False,
):
    infos = archive.infolist()
    if len(infos) > EPUB_ENTRY_COUNT_LIMIT:
        raise ValueError(
            f"EPUB contains more than {EPUB_ENTRY_COUNT_LIMIT} entries"
        )
    names = {info.filename for info in infos}

    container_name = "META-INF/container.xml"
    if container_name not in names:
        raise ValueError("EPUB is missing META-INF/container.xml")
    container = ET.fromstring(
        _read_epub_member(archive, container_name, EPUB_PACKAGE_SIZE_LIMIT)
    )
    package_name = next(
        (
            element.attrib.get("full-path")
            for element in container.iter()
            if _xml_local_name(element.tag) == "rootfile"
            and element.attrib.get("full-path")
        ),
        None,
    )
    if not package_name or package_name not in names:
        raise ValueError("EPUB package document is missing")

    package = ET.fromstring(
        _read_epub_member(archive, package_name, EPUB_PACKAGE_SIZE_LIMIT)
    )
    manifest = {}
    for element in package.iter():
        if _xml_local_name(element.tag) != "item":
            continue
        item_id = element.attrib.get("id")
        member_name = _safe_epub_member_name(package_name, element.attrib.get("href"))
        media_type = element.attrib.get("media-type", "").casefold()
        if (
            item_id
            and member_name in names
            and (
                media_type == "application/xhtml+xml"
                or PurePosixPath(member_name).suffix.casefold()
                in {".xhtml", ".html", ".htm"}
            )
        ):
            manifest[item_id] = member_name

    spine = []
    for element in package.iter():
        if _xml_local_name(element.tag) != "itemref":
            continue
        member_name = manifest.get(element.attrib.get("idref"))
        if member_name and member_name not in spine:
            spine.append(member_name)
    if not spine:
        spine = list(dict.fromkeys(manifest.values()))
    if not spine:
        raise ValueError("EPUB contains no readable spine documents")

    start_cursor = _normalize_epub_cursor(cursor)
    section_index = min(start_cursor["section"], len(spine))
    char_offset = start_cursor["char"] if section_index < len(spine) else 0
    parts = []
    text_chars = 0
    unreadable_sections = []

    while section_index < len(spine) and text_chars < max_chars:
        member_name = spine[section_index]
        try:
            data = _read_epub_member(archive, member_name, EPUB_SECTION_SIZE_LIMIT)
            section_text = _epub_section_text(data)
        except (KeyError, OSError, ValueError, zipfile.BadZipFile) as exc:
            log_to_statusbox(
                f"[RawFileManager] Skipping EPUB section {member_name} "
                f"in {source_label}: {exc}"
            )
            unreadable_sections.append(section_index)
            section_index += 1
            char_offset = 0
            continue

        if char_offset >= len(section_text):
            section_index += 1
            char_offset = 0
            continue

        separator = 2 if parts else 0
        remaining = max(0, max_chars - text_chars - separator)
        if not remaining:
            break
        excerpt = section_text[char_offset:char_offset + remaining]
        if excerpt:
            parts.append(excerpt)
            text_chars += separator + len(excerpt)
        char_offset += len(excerpt)
        if char_offset >= len(section_text):
            section_index += 1
            char_offset = 0

    complete = section_index >= len(spine) and not unreadable_sections
    next_cursor = {"section": section_index, "char": char_offset}
    if unreadable_sections and section_index >= len(spine):
        next_cursor = {"section": unreadable_sections[0], "char": 0}
    progress = {
        "format": "epub",
        "complete": complete,
        "status": "complete" if complete else "partial",
        "sections_total": len(spine),
        "window_start": start_cursor,
        "next_cursor": next_cursor,
        "characters_in_window": sum(len(part) for part in parts),
        "unreadable_sections": unreadable_sections,
    }
    text = "\n\n".join(parts)
    return (text, progress) if with_progress else text


def _extract_epub_text_bytes(
    data,
    source_label,
    *,
    max_chars=12000,
    cursor=None,
    with_progress=False,
):
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            return _extract_epub_archive_text(
                archive,
                source_label,
                max_chars=max_chars,
                cursor=cursor,
                with_progress=with_progress,
            )
    except Exception as exc:
        log_to_statusbox(f"[RawFileManager] Failed to read EPUB {source_label}: {exc}")
        return ("", None) if with_progress else ""


def _extract_epub_text(
    path,
    *,
    max_chars=12000,
    cursor=None,
    with_progress=False,
):
    try:
        with zipfile.ZipFile(path) as archive:
            return _extract_epub_archive_text(
                archive,
                str(path),
                max_chars=max_chars,
                cursor=cursor,
                with_progress=with_progress,
            )
    except Exception as exc:
        log_to_statusbox(f"[RawFileManager] Failed to read EPUB {path}: {exc}")
        return ("", None) if with_progress else ""


def fragment_document_text(
    text,
    source,
    transformer,
    doc_type=None,
    *,
    vocab_source=None,
    document_progress=None,
    sequential=False,
):
    if sequential:
        chunks = _written_passages(text)
    else:
        chunks = _document_chunks(text, source)
    if not chunks:
        return []

    fragments = []
    passage_count = len(chunks)
    for passage_index, chunk in enumerate(chunks):
        frag_id = f"frag_text_{uuid.uuid4().hex[:10]}"
        tags = ["text", "self_read", "document"]
        if doc_type:
            tags.append(doc_type)
        if isinstance(document_progress, dict):
            tags.append("partial_document_read")
        tags = list(dict.fromkeys(tags))

        summary = f"Excerpt from {Path(source).name}: {chunk}"
        frag = {
            "id": frag_id,
            "modality": "text",
            "summary": summary,
            "text": chunk,
            "length": len(chunk),
            "tags": tags,
            "source": source,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "emotions": {"curiosity": 0.55, "focus": 0.35}
        }
        span_metadata = reading_span_metadata(
            str(source), passage_index, passage_count, chunk, document_progress
        )
        frag["written_example"] = {
            **span_metadata,
            "passage_index": passage_index,
            "passage_count": passage_count,
            "unit": "passage",
            "complete_text": True,
            "interpretation_unit": True,
            "storage_fragment": True,
            "transport_chunk": False,
            "boundary_policy": "written_structure",
            "paragraph_count": len([part for part in chunk.split("\n\n") if part.strip()]),
            "reread_eligible": True,
            "repetition_is_new_evidence": False,
        }
        if isinstance(document_progress, dict):
            fragment_progress = dict(document_progress)
            fragment_progress["window_reaches_end"] = bool(
                document_progress.get("complete")
            )
            fragment_progress["complete"] = False
            fragment_progress["status"] = "partial"
            frag["document_read_progress"] = fragment_progress
        vec = transformer.encode(frag)
        frag["importance"] = vec["importance"]
        try:
            update_text_vocab(
                chunk,
                child=child,
                tags=frag["tags"],
                emotions=frag.get("emotions"),
                source=vocab_source or "raw_file_manager",
            )
        except Exception:
            pass
        fragments.append(frag)
    return fragments


def fragment_document(
    path,
    transformer,
    *,
    vocab_source=None,
    document_cursor=None,
):
    ext = path.suffix.lower()
    document_progress = None
    if ext == ".pdf":
        text = _extract_pdf_text(path)
        doc_type = "pdf"
    elif ext == ".odt":
        text = _extract_odt_text(path)
        doc_type = "odt"
    elif ext == ".epub":
        text, document_progress = _extract_epub_text(
            path,
            cursor=document_cursor,
            with_progress=True,
        )
        doc_type = "epub"
    else:
        return []

    if not text:
        log_to_statusbox(f"[RawFileManager] No text extracted from {path}.")
        return []
    return fragment_document_text(
        text,
        path.name,
        transformer,
        doc_type=doc_type,
        vocab_source=vocab_source,
        document_progress=document_progress,
        sequential=(doc_type == "epub"),
    )


def fragment_document_bytes(data, source_label, transformer, suffix):
    ext = suffix.lower()
    document_progress = None
    if ext == ".pdf":
        text = _extract_pdf_text_bytes(data, source_label)
        doc_type = "pdf"
    elif ext == ".odt":
        text = _extract_odt_text_bytes(data, source_label)
        doc_type = "odt"
    elif ext == ".epub":
        text, document_progress = _extract_epub_text_bytes(
            data,
            source_label,
            with_progress=True,
        )
        doc_type = "epub"
    else:
        return []

    if not text:
        log_to_statusbox(f"[RawFileManager] No text extracted from {source_label}.")
        return []
    return fragment_document_text(
        text,
        source_label,
        transformer,
        doc_type=doc_type,
        document_progress=document_progress,
        sequential=(doc_type == "epub"),
    )


def fragment_text(text, source, transformer, *, vocab_source=None):
    chunks = [text[i:i+400] for i in range(0, len(text), 400)]
    fragments = []
    for chunk in chunks[:5]:
        frag_id = f"frag_text_{uuid.uuid4().hex[:10]}"
        tags = ["text", "self_read"]
        if source.endswith(".py"):
            tags.append("code")
        else:
            tags.append("text")
        tags = list(dict.fromkeys(tags))

        frag = {
            "id": frag_id,
            "modality": "text",
            "summary": chunk,
            "text": chunk,
            "length": len(chunk),
            "tags": tags,
            "source": source,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "emotions": {"curiosity": 0.6, "focus": 0.4}
        }
        vec = transformer.encode(frag)
        frag["importance"] = vec["importance"]
        try:
            update_text_vocab(
                chunk,
                child=child,
                tags=frag["tags"],
                emotions=frag.get("emotions"),
                source=vocab_source or "raw_file_manager",
            )
        except Exception:
            pass
        fragments.append(frag)
    return fragments

def fragment_image(image_source, transformer, source_label=None):
    try:
        if isinstance(image_source, (str, Path)):
            open_target = image_source
        else:
            image_source.seek(0)
            open_target = image_source

        if source_label:
            source = source_label
        elif isinstance(image_source, (str, Path)):
            source = str(image_source)
        else:
            source = getattr(image_source, "name", "<memory_image>")

        label = source_label or source
        array = None
        fallback_metadata = None
        pillow_error = None

        if Image is not None and np is not None:
            try:
                with Image.open(open_target) as img:
                    array = np.array(img.convert("L")).flatten().tolist()
            except Exception as e:
                pillow_error = e

        if array is None:
            if not isinstance(open_target, (str, Path)):
                try:
                    open_target.seek(0)
                except Exception:
                    pass
            try:
                fallback = extract_image_features(open_target, limit=1024)
            except ImageFallbackError as e:
                reason = e
                if pillow_error is not None:
                    reason = f"Pillow path failed ({pillow_error}); fallback path failed ({e})"
                elif Image is None or np is None:
                    missing = _IMAGE_IMPORT_ERROR if Image is None else _NUMPY_IMPORT_ERROR
                    reason = f"optional dependency unavailable ({missing}); fallback path failed ({e})"
                log_to_statusbox(f"[RawFileManager] Image support unavailable for {label}: {reason}")
                return []
            except Exception as e:
                log_to_statusbox(f"[RawFileManager] Fallback image decoder failed for {label}: {e}")
                return []

            array = fallback.get("features") or []
            if not array:
                log_to_statusbox(f"[RawFileManager] Fallback image decoder found no pixels for {label}.")
                return []
            fallback_metadata = {
                "decoder": fallback.get("decoder"),
                "format": fallback.get("format"),
                "width": fallback.get("width"),
                "height": fallback.get("height"),
                "feature_count": fallback.get("feature_count"),
                "source_pixels": fallback.get("source_pixels"),
            }
            if pillow_error is not None:
                fallback_metadata["pillow_error"] = str(pillow_error)[:250]
            elif Image is None or np is None:
                missing = _IMAGE_IMPORT_ERROR if Image is None else _NUMPY_IMPORT_ERROR
                fallback_metadata["dependency_gap"] = str(missing)[:250]

        summary_name = Path(source).name if isinstance(source, str) else "image"
        tags = ["self_read", "image"]
        if fallback_metadata:
            tags.append("fallback_image_decoder")

        frag = {
            "modality": "image",
            "image_features": array[:1024],
            "summary": f"Visual symbol or artifact from {summary_name}",
            "tags": tags,
            "source": source,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "emotions": {"focus": 0.3, "novelty": 0.5}
        }
        if fallback_metadata:
            frag["image_fallback"] = fallback_metadata
        vec = transformer.encode_image_fragment(frag)
        frag["importance"] = vec["importance"]
        return [frag]
    except Exception as e:
        label = source_label or image_source
        log_to_statusbox(f"[RawFileManager] Failed to process image {label}: {e}")
        return []

AUDIO_METADATA_TEXT_LIMIT = 2048
AUDIO_LYRICS_TEXT_LIMIT = 32 * 1024
AUDIO_METADATA_LIST_LIMIT = 32
AUDIO_FEATURE_VECTOR_LIMIT = 512


def _bounded_audio_metadata_text(value, limit=AUDIO_METADATA_TEXT_LIMIT):
    if value is None or isinstance(value, dict):
        return ""
    if isinstance(value, (list, tuple, set)):
        parts = [
            _bounded_audio_metadata_text(item, limit)
            for item in list(value)[:AUDIO_METADATA_LIST_LIMIT]
        ]
        value = "; ".join(part for part in parts if part)
    elif isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    else:
        value = str(value)
    return " ".join(value.replace("\x00", " ").split())[:limit]


def _bounded_audio_lyrics(value, limit=AUDIO_LYRICS_TEXT_LIMIT):
    if isinstance(value, (list, tuple, set)):
        parts = [
            _bounded_audio_lyrics(item, limit)
            for item in list(value)[:AUDIO_METADATA_LIST_LIMIT]
        ]
        value = "\n\n".join(part for part in parts if part)
    elif isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    elif value is None or isinstance(value, dict):
        return ""
    else:
        value = str(value)
    normalized_lines = [
        " ".join(line.split())
        for line in value.replace("\x00", " ").replace("\r\n", "\n").split("\n")
    ]
    return "\n".join(normalized_lines).strip()[:limit]


def _normalize_audio_metadata_value(value, limit=AUDIO_METADATA_TEXT_LIMIT):
    if isinstance(value, (list, tuple, set)):
        normalized = []
        for item in list(value)[:AUDIO_METADATA_LIST_LIMIT]:
            text = _bounded_audio_metadata_text(item, limit)
            if text and text not in normalized:
                normalized.append(text)
        if not normalized:
            return None
        return normalized[0] if len(normalized) == 1 else normalized
    text = _bounded_audio_metadata_text(value, limit)
    return text or None


def _audio_metadata_number(value, *, integer=False):
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0:
        return None
    return int(round(number)) if integer else round(number, 6)


def _normalized_audio_tag_key(value):
    return re.sub(
        r"_+",
        "_",
        str(value or "").strip().casefold().replace("-", "_").replace(" ", "_"),
    ).strip("_")


def _extract_audio_metadata(audio_path):
    """Return bounded descriptive and technical metadata from ffprobe/pydub."""
    if mediainfo_json is None:
        return {}

    try:
        probe = mediainfo_json(str(audio_path))
    except Exception as exc:
        log_to_statusbox(
            f"[RawFileManager] Audio metadata probe failed for {audio_path}: {exc}"
        )
        return {}
    if not isinstance(probe, dict):
        return {}

    format_info = probe.get("format")
    format_info = format_info if isinstance(format_info, dict) else {}
    streams = probe.get("streams")
    streams = streams if isinstance(streams, list) else []
    audio_stream = next(
        (
            stream
            for stream in streams
            if isinstance(stream, dict) and stream.get("codec_type") == "audio"
        ),
        {},
    )

    combined_tags = {}
    for source in (audio_stream.get("tags"), format_info.get("tags")):
        if not isinstance(source, dict):
            continue
        for raw_key, raw_value in source.items():
            key = _normalized_audio_tag_key(raw_key)
            if key:
                combined_tags[key] = raw_value

    lyrics_parts = []
    normalized_tags = {}
    for key, raw_value in combined_tags.items():
        is_lyrics = (
            key.startswith("lyrics")
            or "unsyncedlyrics" in key
            or "syncedlyrics" in key
        )
        if is_lyrics:
            lyrics = _bounded_audio_lyrics(raw_value)
            if lyrics and lyrics not in lyrics_parts:
                lyrics_parts.append(lyrics)
            continue
        value = _normalize_audio_metadata_value(raw_value)
        if value is not None:
            normalized_tags[key] = value

    aliases = {
        "title": ("title",),
        "artist": ("artist", "artists"),
        "album": ("album",),
        "album_artist": ("album_artist", "albumartist"),
        "genre": ("genre",),
        "date": ("date", "year"),
        "track": ("track", "tracknumber", "track_number"),
        "disc": ("disc", "discnumber", "disc_number"),
        "composer": ("composer",),
        "comment": ("comment", "description"),
        "copyright": ("copyright",),
    }
    metadata = {}
    for canonical, keys in aliases.items():
        for key in keys:
            value = normalized_tags.get(key)
            if value is not None:
                metadata[canonical] = value
                break

    if normalized_tags:
        metadata["tags"] = normalized_tags
    if lyrics_parts:
        metadata["lyrics"] = "\n\n".join(lyrics_parts)[:AUDIO_LYRICS_TEXT_LIMIT]

    technical = {}
    text_fields = {
        "format": format_info.get("format_name"),
        "codec": audio_stream.get("codec_name"),
    }
    for key, value in text_fields.items():
        normalized = _normalize_audio_metadata_value(value, 128)
        if normalized is not None:
            technical[key] = normalized

    numeric_fields = {
        "duration_seconds": (
            audio_stream.get("duration") or format_info.get("duration"),
            False,
        ),
        "bit_rate": (
            audio_stream.get("bit_rate") or format_info.get("bit_rate"),
            True,
        ),
        "sample_rate": (audio_stream.get("sample_rate"), True),
        "channels": (audio_stream.get("channels"), True),
        "file_size": (format_info.get("size"), True),
        "stream_index": (audio_stream.get("index"), True),
    }
    for key, (value, integer) in numeric_fields.items():
        normalized = _audio_metadata_number(value, integer=integer)
        if normalized is not None:
            technical[key] = normalized

    attached_pictures = [
        stream
        for stream in streams
        if isinstance(stream, dict)
        and isinstance(stream.get("disposition"), dict)
        and bool(stream["disposition"].get("attached_pic"))
    ]
    technical["attached_picture"] = bool(attached_pictures)
    if attached_pictures:
        artwork_codec = _normalize_audio_metadata_value(
            attached_pictures[0].get("codec_name"),
            128,
        )
        if artwork_codec:
            technical["artwork_codec"] = artwork_codec
    if technical:
        metadata["technical"] = technical
    return metadata


def _flat_audio_feature_vector(analysis):
    """Prefer the digest's bounded embedding; safely flatten legacy matrices."""
    if not isinstance(analysis, dict):
        return []

    candidates = (analysis.get("embedding"), analysis.get("frames"))
    for candidate in candidates:
        values = []

        def collect(value):
            if len(values) >= AUDIO_FEATURE_VECTOR_LIMIT:
                return
            if isinstance(value, (list, tuple)):
                for child_value in value:
                    collect(child_value)
                    if len(values) >= AUDIO_FEATURE_VECTOR_LIMIT:
                        break
                return
            if isinstance(value, bool):
                return
            try:
                number = float(value)
            except (TypeError, ValueError):
                return
            if math.isfinite(number):
                values.append(number)

        collect(candidate)
        if values:
            return values
    return []


def _audio_fragment_summary(audio_path, analysis, metadata):
    base = (
        analysis.get("summary")
        if isinstance(analysis, dict)
        else None
    ) or f"Sound fragment from {audio_path.name}"
    title = _bounded_audio_metadata_text(metadata.get("title"), 300)
    artist = _bounded_audio_metadata_text(metadata.get("artist"), 300)
    if title and artist:
        return f"{title} by {artist}. {base}"
    if title:
        return f"{title}. {base}"
    return base


def _attach_audio_analysis(frag, analysis, metadata):
    if metadata:
        frag["audio_metadata"] = metadata
        tags = frag.setdefault("tags", [])
        if "audio_metadata" not in tags:
            tags.append("audio_metadata")
        if metadata.get("lyrics") and "embedded_lyrics" not in tags:
            tags.append("embedded_lyrics")
        technical = metadata.get("technical")
        if (
            isinstance(technical, dict)
            and technical.get("attached_picture")
            and "embedded_cover_art" not in tags
        ):
            tags.append("embedded_cover_art")

    analysis_payload = {}
    for key in ("texture_signature", "diversity_boost", "unique_symbols", "language_hint", "analysis_window"):
        value = analysis.get(key) if isinstance(analysis, dict) else None
        if value is not None:
            analysis_payload[key] = value
    if isinstance(analysis, dict) and analysis.get("symbol_embedding"):
        analysis_payload["symbol_embedding"] = analysis["symbol_embedding"]
    frames = analysis.get("frames") if isinstance(analysis, dict) else None
    if isinstance(frames, list):
        analysis_payload["frame_count"] = len(frames)
        first_frame = frames[0] if frames else None
        if isinstance(first_frame, (list, tuple)):
            analysis_payload["feature_bins"] = len(first_frame)
    if analysis_payload:
        frag["audio_analysis"] = analysis_payload

    if isinstance(analysis, dict):
        frag["analysis_paths"] = {
            "symbol_map": analysis.get("symbol_map_path"),
            "symbol_words": analysis.get("symbol_words_path"),
        }
        frag["multi_symbol_pairs"] = analysis.get("multi_symbol_pairs", [])


def _probe_media_duration(audio_path, metadata=None):
    metadata = metadata if isinstance(metadata, dict) else {}
    technical = metadata.get("technical")
    technical = technical if isinstance(technical, dict) else {}
    duration = _audio_metadata_number(technical.get("duration_seconds"))
    if duration:
        return duration
    if Path(audio_path).suffix.casefold() == ".wav":
        try:
            with contextlib.closing(wave.open(str(audio_path), "r")) as wf:
                return wf.getnframes() / float(wf.getframerate() or 1)
        except Exception:
            return None
    probed = _extract_audio_metadata(audio_path)
    technical = probed.get("technical") if isinstance(probed, dict) else {}
    return _audio_metadata_number(technical.get("duration_seconds")) if isinstance(technical, dict) else None


def fragment_audio(
    audio_path, transformer, *, seek_fraction=None, seek_seconds=None,
    excerpt_seconds=60.0,
):
    ext = audio_path.suffix.lower()
    supported_digest_formats = {".wav", ".mp3", ".opus"}
    if ext not in supported_digest_formats:
        log_to_statusbox(
            f"[RawFileManager] Unsupported audio format for {audio_path.name}: {ext}"
        )
        return []

    metadata = (
        _extract_audio_metadata(audio_path) if ext in {".mp3", ".opus"} else {}
    )
    full_duration = _probe_media_duration(audio_path, metadata)
    try:
        selected_fraction = max(0.0, min(1.0, float(seek_fraction if seek_fraction is not None else 0.5)))
    except (TypeError, ValueError):
        selected_fraction = 0.5
    analysis_start = 0.0
    analysis_limit = None
    if seek_seconds is not None or seek_fraction is not None:
        try:
            requested_seconds = max(0.0, float(seek_seconds)) if seek_seconds is not None else None
        except (TypeError, ValueError):
            requested_seconds = None
        analysis_limit = max(1.0, min(120.0, float(excerpt_seconds)))
        centre = requested_seconds if requested_seconds is not None else (float(full_duration or 0.0) * selected_fraction)
        if requested_seconds is not None and full_duration:
            selected_fraction = max(0.0, min(1.0, requested_seconds / float(full_duration)))
        analysis_start = max(0.0, centre - analysis_limit / 2.0)
        if full_duration:
            analysis_start = min(analysis_start, max(0.0, float(full_duration) - analysis_limit))

    analysis = None
    if analyze_audio_clip is not None:
        try:
            try:
                analysis = analyze_audio_clip(
                    audio_path, transformer, child=child, label="self_read",
                    max_seconds=analysis_limit, start_seconds=analysis_start,
                )
            except TypeError as exc:
                if analysis_limit is not None or "unexpected keyword argument" not in str(exc):
                    raise
                analysis = analyze_audio_clip(audio_path, transformer)
        except Exception as exc:
            log_to_statusbox(
                f"[RawFileManager] Audio digest failed for {audio_path}: {exc}"
            )

    def merged_tags():
        tags = ["self_read", "audio"]
        raw_tags = analysis.get("tags", []) if isinstance(analysis, dict) else []
        if isinstance(raw_tags, str):
            raw_tags = [raw_tags]
        if not isinstance(raw_tags, (list, tuple, set)):
            raw_tags = []
        for tag in raw_tags:
            normalized = _bounded_audio_metadata_text(tag, 128)
            if normalized and normalized not in tags:
                tags.append(normalized)
        return tags

    if ext == ".wav":
        try:
            with contextlib.closing(wave.open(str(audio_path), "r")) as wf:
                frame_count = wf.getnframes()
                frames = wf.readframes(min(frame_count, 1024))
                sample_rate = wf.getframerate()
                channel_count = wf.getnchannels()
                sample_width_bits = wf.getsampwidth() * 8
        except Exception as exc:
            log_to_statusbox(
                f"[RawFileManager] Failed to process WAV {audio_path}: {exc}"
            )
            return []

        wave_duration = frame_count / float(sample_rate or 1)
        metadata = {
            "technical": {
                "format": "wav",
                "codec": "pcm",
                "duration_seconds": round(wave_duration, 6),
                "sample_rate": sample_rate,
                "channels": channel_count,
                "sample_width_bits": sample_width_bits,
                "attached_picture": False,
            }
        }
        digest_features = _flat_audio_feature_vector(analysis)
        audio_data = list(frames[:1024])
        fallback_features = [value / 255.0 for value in audio_data]
        duration = (
            full_duration
            or _audio_metadata_number(
                analysis.get("duration") if isinstance(analysis, dict) else None
            )
            or wave_duration
        )
        emotions = (
            analysis.get("emotions") if isinstance(analysis, dict) else None
        )
        if not isinstance(emotions, dict):
            emotions = {"attention": 0.5, "novelty": 0.6}

        frag = {
            "modality": "audio",
            "audio_features": digest_features or fallback_features,
            "summary": _audio_fragment_summary(audio_path, analysis, metadata),
            "tags": merged_tags(),
            "source": str(audio_path),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "emotions": emotions,
            "duration": duration,
            "symbols": (
                analysis.get("symbols", [])
                if isinstance(analysis, dict)
                else []
            ),
            "proto_words": (
                analysis.get("proto_words", [])
                if isinstance(analysis, dict)
                else []
            ),
        }
        _attach_audio_analysis(frag, analysis, metadata)
        observed_duration = _audio_metadata_number(analysis.get("duration")) if isinstance(analysis, dict) else duration
        attach_media_experience(
            frag, media_kind="audio", duration_seconds=duration, seek_fraction=selected_fraction,
            observed_start=analysis_start, observed_end=min(float(duration or 0.0), analysis_start + float(observed_duration or 0.0)),
        )

        vec = transformer.encode_audio_fragment(frag)
        clarity = analysis.get("clarity") if isinstance(analysis, dict) else None
        try:
            frag["importance"] = (
                round(float(clarity), 4)
                if clarity is not None
                else vec["importance"]
            )
        except (TypeError, ValueError):
            frag["importance"] = vec["importance"]
        return [frag]

    if analysis is None:
        if analyze_audio_clip is None:
            log_to_statusbox(
                "[RawFileManager] Compressed audio decoding unavailable: "
                f"{_AUDIO_DIGEST_IMPORT_ERROR}"
            )
        else:
            log_to_statusbox(
                f"[RawFileManager] Analysis returned no data for {audio_path.name}."
            )
        return []

    technical = metadata.get("technical")
    technical = technical if isinstance(technical, dict) else {}
    duration = (
        full_duration
        or _audio_metadata_number(analysis.get("duration"))
        or technical.get("duration_seconds")
        or 0
    )
    emotions = analysis.get("emotions")
    if not isinstance(emotions, dict):
        emotions = {"attention": 0.5}

    frag = {
        "modality": "audio",
        "audio_features": _flat_audio_feature_vector(analysis),
        "summary": _audio_fragment_summary(audio_path, analysis, metadata),
        "tags": merged_tags(),
        "source": str(audio_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "emotions": emotions,
        "symbols": analysis.get("symbols", []),
        "proto_words": analysis.get("proto_words", []),
        "duration": duration,
    }
    _attach_audio_analysis(frag, analysis, metadata)
    observed_duration = _audio_metadata_number(analysis.get("duration")) or duration
    attach_media_experience(
        frag, media_kind="audio", duration_seconds=duration, seek_fraction=selected_fraction,
        observed_start=analysis_start, observed_end=min(float(duration or 0.0), analysis_start + float(observed_duration or 0.0)),
    )

    vec = transformer.encode_audio_fragment(frag)
    clarity = analysis.get("clarity")
    try:
        frag["importance"] = (
            round(float(clarity), 4)
            if clarity is not None
            else vec["importance"]
        )
    except (TypeError, ValueError):
        frag["importance"] = vec["importance"]
    return [frag]


def fragment_video(
    video_path, transformer, source_label=None, *, seek_fraction=None,
    seek_seconds=None, excerpt_seconds=30.0,
):
    summary_parts = []
    try:
        selected_fraction = max(0.0, min(1.0, float(seek_fraction if seek_fraction is not None else 0.5)))
    except (TypeError, ValueError):
        selected_fraction = 0.5
    preview_features = []
    visual_sample_seconds = []
    duration_seconds = None
    resolution = None

    if cv2 is not None:
        capture = cv2.VideoCapture(str(video_path))
        if capture.isOpened():
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if frame_count and fps:
                duration_seconds = frame_count / fps
            if width and height:
                resolution = (width, height)

            try:
                requested_seconds = max(0.0, float(seek_seconds)) if seek_seconds is not None else None
            except (TypeError, ValueError):
                requested_seconds = None
            target_seconds = requested_seconds if requested_seconds is not None else (float(duration_seconds or 0.0) * selected_fraction)
            visual_window = min(max(1.0, float(excerpt_seconds)), 60.0, float(duration_seconds or 1.0))
            sample_points = [
                max(0.0, min(float(duration_seconds or 0.0), target_seconds + offset))
                for offset in (-visual_window / 2.0, 0.0, visual_window / 2.0)
            ]
            for sample_second in dict.fromkeys(round(value, 3) for value in sample_points):
                target_frame = min(frame_count - 1, int(sample_second * fps)) if frame_count and fps else 0
                if target_frame > 0:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                success, frame = capture.read()
                if not success or frame is None or Image is None or np is None:
                    continue
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    preview_image = Image.fromarray(frame_rgb).convert("L").resize((16, 16))
                    preview_features.extend((np.array(preview_image).astype(float).flatten() / 255.0).tolist())
                    visual_sample_seconds.append(sample_second)
                except Exception as frame_err:
                    log_to_statusbox(
                        f"[RawFileManager] Failed to extract frame from {video_path}: {frame_err}"
                    )
            capture.release()
        else:
            capture.release()
    elif _VIDEO_IMPORT_ERROR:
        log_to_statusbox(
            f"[RawFileManager] OpenCV unavailable for {video_path.name}: {_VIDEO_IMPORT_ERROR}"
        )

    if not duration_seconds:
        probed = _extract_audio_metadata(video_path)
        technical = probed.get("technical") if isinstance(probed, dict) else {}
        if isinstance(technical, dict):
            duration_seconds = _audio_metadata_number(technical.get("duration_seconds"))

    if duration_seconds:
        summary_parts.append(f"~{duration_seconds:.1f}s")
    if resolution:
        summary_parts.append(f"{resolution[0]}x{resolution[1]}")

    try:
        size_mb = video_path.stat().st_size / (1024 * 1024)
        summary_parts.append(f"{size_mb:.1f}MB")
    except Exception:
        pass

    summary_details = " (" + ", ".join(summary_parts) + ")" if summary_parts else ""
    source = source_label or str(video_path)

    video_kind = video_language_kind(duration_seconds)
    summary_kind = "Video essay" if video_kind == "video_essay" else "Channel video"
    frag = {
        "modality": "video",
        "summary": f"{summary_kind} from {Path(source).name}{summary_details}",
        "tags": ["self_read", "video"],
        "source": source,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "emotions": {"focus": 0.4, "curiosity": 0.55},
        "duration": duration_seconds or 0.0,
    }

    bounded_excerpt = min(max(1.0, float(excerpt_seconds)), 60.0, float(duration_seconds or 1.0))
    try:
        requested_seconds = max(0.0, float(seek_seconds)) if seek_seconds is not None else None
    except (TypeError, ValueError):
        requested_seconds = None
    excerpt_centre = requested_seconds if requested_seconds is not None else (float(duration_seconds or 0.0) * selected_fraction)
    if requested_seconds is not None and duration_seconds:
        selected_fraction = max(0.0, min(1.0, requested_seconds / float(duration_seconds)))
    excerpt_start = max(0.0, excerpt_centre - bounded_excerpt / 2.0)
    if duration_seconds:
        excerpt_start = min(excerpt_start, max(0.0, float(duration_seconds) - bounded_excerpt))

    if analyze_audio_clip is not None and duration_seconds:
        try:
            spoken_analysis = analyze_audio_clip(
                video_path, transformer, child=child,
                label="video_essay_spoken" if video_kind == "video_essay" else "channel_video_audio",
                max_seconds=bounded_excerpt, start_seconds=excerpt_start,
            )
        except Exception as exc:
            spoken_analysis = None
            log_to_statusbox(f"[RawFileManager] Video audio excerpt failed for {video_path}: {exc}")
        if isinstance(spoken_analysis, dict):
            frag["audio_features"] = _flat_audio_feature_vector(spoken_analysis)
            frag["symbols"] = spoken_analysis.get("symbols", [])
            frag["proto_words"] = spoken_analysis.get("proto_words", [])
            frag["tags"].append("bounded_audio_excerpt")
            _attach_audio_analysis(frag, spoken_analysis, {})

    if preview_features:
        frag["video_features"] = preview_features[:1024]
        frag["visual_sample_seconds"] = visual_sample_seconds[:3]
    else:
        metadata_features = []
        if duration_seconds:
            metadata_features.append(min(duration_seconds / 600.0, 1.0))
        if resolution:
            metadata_features.extend([
                min(resolution[0] / 4000.0, 1.0),
                min(resolution[1] / 4000.0, 1.0),
            ])
        if metadata_features:
            frag["video_features"] = metadata_features

    attach_media_experience(
        frag, media_kind="video", duration_seconds=duration_seconds, seek_fraction=selected_fraction,
        observed_start=excerpt_start, observed_end=min(float(duration_seconds or 0.0), excerpt_start + bounded_excerpt),
    )

    vec = transformer.encode_video_fragment(frag)
    frag["importance"] = vec.get("importance", 0.0)
    return [frag]


def _fragments_from_data_buffer(
    data, inner_path, container_path, category, transformer, *,
    seek_fraction=None, seek_seconds=None,
):
    source_label = f"{container_path.name}:{inner_path.as_posix()}"
    if category == "text":
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("latin-1", errors="ignore")
        return fragment_text(text, source_label, transformer)

    if category == "document":
        return fragment_document_bytes(data, source_label, transformer, inner_path.suffix)

    if category == "image":
        return fragment_image(io.BytesIO(data), transformer, source_label=source_label)

    if category in {"audio", "video"}:
        suffix = inner_path.suffix or ""
        with NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(data)
            tmp.flush()
            temp_path = Path(tmp.name)
            if category == "audio":
                results = fragment_audio(
                    temp_path, transformer, seek_fraction=seek_fraction, seek_seconds=seek_seconds
                )
            else:
                results = fragment_video(
                    temp_path, transformer, source_label=source_label,
                    seek_fraction=seek_fraction, seek_seconds=seek_seconds,
                )
            for frag in results:
                frag["source"] = source_label
            return results

    return []


def _zip_declared_entry_count(path):
    """Read ZIP end metadata without loading the complete central directory."""
    try:
        with open(path, "rb") as handle:
            end_record = zipfile._EndRecData(handle)
        if end_record is None:
            return None
        return int(end_record[zipfile._ECD_ENTRIES_TOTAL])
    except (AttributeError, IndexError, OSError, TypeError, ValueError):
        return None


def process_archive(
    path,
    transformer,
    *,
    member_limit=ARCHIVE_MEMBER_COUNT_LIMIT,
    aggregate_limit=ARCHIVE_TOTAL_UNCOMPRESSED_LIMIT,
    fragment_limit=ARCHIVE_FRAGMENT_LIMIT,
    allowed_categories=None,
    media_seek_fraction_value=None,
    media_seek_seconds=None,
):
    """Process a bounded archive sample without materializing an unbounded result."""
    fragments = []
    entries_inspected = 0
    decompressed_bytes = 0
    member_limit = max(1, min(ARCHIVE_MEMBER_COUNT_LIMIT, int(member_limit)))
    aggregate_limit = max(
        1,
        min(ARCHIVE_TOTAL_UNCOMPRESSED_LIMIT, int(aggregate_limit)),
    )
    fragment_limit = max(1, min(ARCHIVE_FRAGMENT_LIMIT, int(fragment_limit)))
    category_allowlist = (
        None
        if allowed_categories is None
        else frozenset(str(item).strip().casefold() for item in allowed_categories)
    )
    budget_notice = None

    def note_budget(reason):
        nonlocal budget_notice
        if budget_notice is None:
            budget_notice = reason
            log_to_statusbox(
                f"[RawFileManager] Archive budget reached for {path.name}: {reason}."
            )

    def append_member(data, inner_path, category):
        if category_allowlist is not None and category not in category_allowlist:
            return True
        generated = list(
            _fragments_from_data_buffer(
                data,
                inner_path,
                path,
                category,
                transformer,
                seek_fraction=media_seek_fraction_value,
                seek_seconds=media_seek_seconds,
            )
            or []
        )
        member_path = str(inner_path).replace("\\", "/")
        member_name = member_path.rsplit("/", 1)[-1]
        for fragment in generated:
            context = fragment.setdefault("source_context", {})
            context.setdefault("archive_container_name", path.name)
            context.setdefault("archive_member_path", member_path)
            context.setdefault("archive_member_name", member_name)
            context.setdefault("archive_member_category", category)
            tags = fragment.setdefault("tags", [])
            for tag in ("archive_member", f"archive_{category}_member"):
                if tag not in tags:
                    tags.append(tag)
        remaining = max(0, fragment_limit - len(fragments))
        fragments.extend(generated[:remaining])
        if len(generated) > remaining or len(fragments) >= fragment_limit:
            note_budget(f"{fragment_limit} fragments")
            return False
        return True

    def entry_allowed(declared_size):
        nonlocal entries_inspected
        if entries_inspected >= member_limit:
            note_budget(f"{member_limit} members")
            return False
        entries_inspected += 1
        if declared_size < 0 or declared_size > ARCHIVE_MEMBER_LIMIT:
            return None
        if decompressed_bytes + declared_size > aggregate_limit:
            note_budget(f"{aggregate_limit} decompressed bytes")
            return None
        return True

    try:
        if zipfile.is_zipfile(path):
            declared_entries = _zip_declared_entry_count(path)
            if (
                declared_entries is None
                or declared_entries > ARCHIVE_MEMBER_COUNT_LIMIT
            ):
                note_budget(
                    f"ZIP central directory exceeds "
                    f"{ARCHIVE_MEMBER_COUNT_LIMIT} entries"
                )
                return fragments
            with zipfile.ZipFile(path) as archive:
                for info in archive.infolist():
                    allowed = entry_allowed(int(info.file_size or 0))
                    if allowed is False:
                        break
                    if info.is_dir() or allowed is None:
                        continue
                    inner_path = Path(info.filename)
                    category = classify_suffixes(
                        [suffix.lower() for suffix in inner_path.suffixes]
                    )
                    if not category or category == "archive":
                        continue
                    if category_allowlist is not None and category not in category_allowlist:
                        continue
                    remaining_bytes = aggregate_limit - decompressed_bytes
                    try:
                        with archive.open(info, "r") as member:
                            data = _read_limited(
                                member,
                                min(ARCHIVE_MEMBER_LIMIT, remaining_bytes),
                            )
                    except ValueError:
                        note_budget(f"{aggregate_limit} decompressed bytes")
                        continue
                    decompressed_bytes += len(data)
                    if not append_member(data, inner_path, category):
                        break

        elif tarfile.is_tarfile(path):
            with tarfile.open(path, "r:*") as archive:
                for member in archive:
                    allowed = entry_allowed(int(member.size or 0))
                    if allowed is False:
                        break
                    if not member.isfile() or allowed is None:
                        continue
                    inner_path = Path(member.name)
                    category = classify_suffixes(
                        [suffix.lower() for suffix in inner_path.suffixes]
                    )
                    if not category or category == "archive":
                        continue
                    if category_allowlist is not None and category not in category_allowlist:
                        continue
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        continue
                    remaining_bytes = aggregate_limit - decompressed_bytes
                    try:
                        data = _read_limited(
                            extracted,
                            min(ARCHIVE_MEMBER_LIMIT, remaining_bytes),
                        )
                    except ValueError:
                        note_budget(f"{aggregate_limit} decompressed bytes")
                        continue
                    finally:
                        extracted.close()
                    decompressed_bytes += len(data)
                    if not append_member(data, inner_path, category):
                        break

        else:
            ext = path.suffix.lower()
            opener_map = {
                ".gz": gzip.open,
                ".bz2": bz2.open,
                ".xz": lzma.open,
            }
            opener = opener_map.get(ext)
            if opener:
                inner_name = Path(path.name).with_suffix("")
                category = classify_suffixes(
                    [suffix.lower() for suffix in inner_name.suffixes]
                )
                if (
                    category
                    and category != "archive"
                    and (
                        category_allowlist is None
                        or category in category_allowlist
                    )
                ):
                    try:
                        with opener(path, "rb") as compressed:
                            data = _read_limited(
                                compressed,
                                min(ARCHIVE_MEMBER_LIMIT, aggregate_limit),
                            )
                    except ValueError:
                        note_budget(
                            f"{min(ARCHIVE_MEMBER_LIMIT, aggregate_limit)} "
                            "decompressed bytes"
                        )
                    else:
                        decompressed_bytes = len(data)
                        append_member(data, inner_name, category)
    except Exception as exc:
        log_to_statusbox(f"[RawFileManager] Failed to process archive {path}: {exc}")

    return fragments

def self_read_and_train():
    child = get_child()
    default_root = Path(__file__).resolve().parent
    try:
        history_ledger = load_history(child)
    except SelfReadHistoryLoadError as exc:
        log_to_statusbox(
            f"[SelfRead] History load failed closed; no files will be read: {exc}"
        )
        if _SELF_READ_LOCK_HELD:
            _release_runtime_lock("failed", error=f"history_load_failed: {exc}")
        return False

    prefs = load_self_read_preferences(child)
    prefs = _apply_skip_requests(child, prefs)
    source_choices = prefs.get("source_choices", DEFAULT_SELF_READ_PREFS["source_choices"])
    skip_patterns = prefs.get("skip_files", [])
    source_override = _load_self_read_source_override()
    focus_decision = resolve_self_read_focus(child)
    read_focus = focus_decision["focus"]
    revisit_limit = _self_read_revisit_limit(read_focus)
    if source_override and not source_choices.get(source_override, False):
        log_to_statusbox(f"[SelfRead] Source override '{source_override}' ignored by preference.")
        source_override = None
    if _SELF_READ_LOCK_HELD:
        _write_runtime_state(
            "running",
            source=source_override or "all",
            phase="collect_roots",
            read_focus=read_focus,
            read_focus_source=focus_decision.get("source"),
            read_focus_scores={
                "new": focus_decision.get("new_score"),
                "seen": focus_decision.get("seen_score"),
            },
            read_focus_drivers=focus_decision.get("drivers") or {},
        )

    history_files = history_ledger["files"]
    read_reason_counts = {"new": 0, "updated": 0, "resume": 0, "revisit": 0}
    primary_fragment_ceiling = (
        _primary_fragment_ceiling(read_focus, FRAG_LIMIT)
        if revisit_limit > 0
        else FRAG_LIMIT
    )
    primary_fallback = None

    def collect_roots(override):
        roots = []
        seen_roots = set()

        def add_root(path, audio_only=False, source_key="code"):
            if override and source_key != override:
                return
            if path is None:
                return
            try:
                resolved = path.resolve()
            except FileNotFoundError:
                return
            if resolved in seen_roots:
                return
            seen_roots.add(resolved)
            roots.append((path, audio_only, source_key))

        if source_choices.get("code", True):
            if default_root.exists():
                add_root(default_root, audio_only=False, source_key="code")
            else:
                log_to_statusbox(f"[SelfRead] Project root not found: {default_root}")
        else:
            log_to_statusbox("[SelfRead] Preference: project code scan disabled.")

        if source_choices.get("books", True):
            if book_folder_path and book_folder_path.exists():
                add_root(book_folder_path, audio_only=False, source_key="books")
            elif book_folder_path:
                log_to_statusbox(f"[SelfRead] Book folder not found: {book_folder_path}")
        elif book_folder_path:
            log_to_statusbox("[SelfRead] Preference: book folder skipped by choice.")

        if source_choices.get("music", True):
            studio_stems_path = _child_memory_path(child, "music_studio", "stems")
            if studio_stems_path.exists():
                add_root(studio_stems_path, audio_only=True, source_key="music")
            if music_folder_path and music_folder_path.exists():
                add_root(music_folder_path, audio_only=True, source_key="music")
            elif music_folder_path:
                log_to_statusbox(f"[SelfRead] Music folder not found: {music_folder_path}")
        elif music_folder_path:
            log_to_statusbox("[SelfRead] Preference: music folder skipped by choice.")

        if source_choices.get("code", True):
            if ina_work_path and ina_work_path.exists():
                add_root(ina_work_path, audio_only=False, source_key="code")
            elif ina_work_path:
                log_to_statusbox(f"[SelfRead] Ina work folder not found: {ina_work_path}")

        if source_choices.get("venv", False):
            if venv_path and venv_path.exists():
                add_root(venv_path, audio_only=False, source_key="venv")
            elif venv_path:
                log_to_statusbox(f"[SelfRead] Virtual environment not found: {venv_path}")

        if source_choices.get("github_history", True) and source_override in (None, "github_history"):
            history_root = _child_memory_path(child, "github_history")
            try:
                materialize_commit_history(default_root, history_root, limit=24)
                add_root(history_root, audio_only=False, source_key="github_history")
            except Exception as exc:
                log_to_statusbox(f"[SelfRead] GitHub history unavailable: {exc}")

        return roots

    roots = collect_roots(source_override)
    if source_override:
        log_to_statusbox(f"[SelfRead] Source override: {source_override}")
        if not roots:
            log_to_statusbox(
                f"[SelfRead] No roots available for '{source_override}'; falling back to all sources."
            )
            roots = collect_roots(None)

    log_to_statusbox(f"[SelfRead] Child set to: {child}")
    log_to_statusbox(
        f"[SelfRead] File focus: {read_focus} "
        f"({focus_decision.get('source', 'unknown')}; revisit limit {revisit_limit})."
    )
    if roots:
        root_descriptions = ", ".join(
            f"{str(path)} [{source_key}]" for path, _, source_key in roots
        )
        log_to_statusbox("[SelfRead] Roots to scan: " + root_descriptions)
    else:
        log_to_statusbox("[SelfRead] No available roots to scan.")
        save_history(child, history_ledger)
        return
    log_to_statusbox(f"[SelfRead] Loaded {len(history_files)} previously seen files.")

    transformer = FractalTransformer()
    count = 0
    revisit_candidates = []
    unchanged_seen_count = 0
    scan_now_ts = datetime.now(timezone.utc).timestamp()
    inspection_limit = _self_read_inspection_limit()
    scan_seconds = _self_read_scan_seconds()
    scan_started_monotonic = time.monotonic()
    files_inspected = 0
    scan_stop_reason = None

    def _scan_time_stop_requested():
        nonlocal scan_stop_reason
        if (time.monotonic() - scan_started_monotonic) >= scan_seconds:
            scan_stop_reason = scan_stop_reason or "time_budget"
            return True
        return False

    seen_revisit_satisfied = False

    def _process_candidate(candidate, read_reason, *, fragment_ceiling=None):
        nonlocal count
        ceiling = FRAG_LIMIT if fragment_ceiling is None else fragment_ceiling
        ceiling = max(0, min(FRAG_LIMIT, int(ceiling)))
        if count >= ceiling:
            return False

        path = candidate["path"]
        category = candidate["category"]
        base_root = candidate["base_root"]
        source_key = candidate["source_key"]
        rel_str = candidate["relative_path"]
        history_key = candidate["history_key"]
        prior = candidate.get("prior")
        current_prior = history_files.get(history_key)
        if isinstance(current_prior, dict):
            prior = current_prior
            candidate["prior"] = prior
        stamp = candidate["stamp"]

        log_to_statusbox(
            f"[SelfRead] PROCESSING {path.name} [{category}; {read_reason}]"
        )

        try:
            navigation = prior.get("media_navigation") if isinstance(prior, dict) else {}
            navigation = navigation if isinstance(navigation, dict) else {}
            selected_seek_fraction = media_seek_fraction(read_reason, prior)
            requested_seek_seconds = navigation.get("requested_seek_seconds")
            if category == "text":
                with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                    text = handle.read()
                result = fragment_text(
                    text,
                    path.name,
                    transformer,
                    vocab_source=f"self_read:{source_key}",
                )

            elif category == "document":
                document_cursor = (
                    _epub_cursor_from_history(prior, stamp)
                    if path.suffix.casefold() == ".epub"
                    else None
                )
                result = fragment_document(
                    path,
                    transformer,
                    vocab_source=f"self_read:{source_key}",
                    document_cursor=document_cursor,
                )

            elif category == "image":
                result = fragment_image(path, transformer)

            elif category == "audio":
                result = fragment_audio(
                    path, transformer, seek_fraction=selected_seek_fraction,
                    seek_seconds=requested_seek_seconds,
                )

            elif category == "video":
                result = fragment_video(
                    path, transformer, seek_fraction=selected_seek_fraction,
                    seek_seconds=requested_seek_seconds,
                )

            elif category == "archive":
                result = process_archive(
                    path,
                    transformer,
                    allowed_categories=(
                        {"audio", "text", "image"}
                        if candidate.get("audio_only")
                        else None
                    ),
                    media_seek_fraction_value=selected_seek_fraction,
                    media_seek_seconds=requested_seek_seconds,
                )

            else:
                log_to_statusbox(
                    f"[SelfRead] SKIP {path.name} — unsupported processing category {category}."
                )
                return False

            result = list(result or [])
            if not result:
                return False

            document_progress = next(
                (
                    frag.get("document_read_progress")
                    for frag in result
                    if isinstance(frag.get("document_read_progress"), dict)
                ),
                None,
            )

            remaining = max(0, ceiling - count)
            fragments_to_save, start_offset, next_offset, total_fragments = (
                _self_read_fragment_window(result, prior, stamp, remaining)
            )
            if not fragments_to_save:
                if read_reason == "resume" and start_offset >= total_fragments:
                    completed = dict(prior or {})
                    completed.pop("continuation", None)
                    completed["continuation_cleared_at"] = datetime.now(
                        timezone.utc
                    ).isoformat()
                    history_files[history_key] = completed
                return False

            record = _next_history_record(
                prior,
                stamp,
                read_reason=read_reason,
                source_key=source_key,
                relative_path=rel_str,
                base_root=base_root,
            )
            saved_ids = []
            observed_media = []
            for frag in fragments_to_save:
                frag_id = frag.get("id") or f"frag_text_{uuid.uuid4().hex[:10]}"
                frag["id"] = frag_id

                annotate_fragment_source(frag, source_key, rel_str, base_root)
                annotate_fragment_read_lineage(
                    frag,
                    read_reason=read_reason,
                    prior=prior,
                    record=record,
                    focus=read_focus,
                )
                media_experience = frag.get("media_experience")
                if isinstance(media_experience, dict):
                    observed_media.append({
                        "fragment_id": frag_id,
                        "mode": media_experience.get("mode"),
                        "seek_fraction": media_experience.get("seek_fraction"),
                        "observed_spans": list(media_experience.get("observed_spans") or ())[:4],
                    })

                if source_key == "music":
                    try:
                        record_media_lesson(child, frag)
                    except Exception as exc:
                        log_to_statusbox(f"[SelfRead] Learned-media lesson index skipped for {path.name}: {exc}")

                frag_path = _child_memory_path(
                    child,
                    "fragments",
                    f"{frag_id}.json",
                )
                frag_path.parent.mkdir(parents=True, exist_ok=True)

                with open(frag_path, "w", encoding="utf-8") as handle:
                    json.dump(frag, handle, indent=4)

                log_to_statusbox(
                    f"[SelfRead] + Fragment saved: {frag_id} from {path.name} "
                    f"({read_reason}, read #{record['read_count']})"
                )
                log_reflection(child, frag)
                saved_ids.append(frag_id)
                count += 1

            record["last_fragment_ids"] = saved_ids[:5]
            record["last_fragment_range"] = {
                "start": start_offset,
                "end_exclusive": next_offset,
            }
            record["fragment_count_seen"] = total_fragments
            record["fragment_count_saved"] = len(fragments_to_save)
            if observed_media:
                record["media_navigation"] = {
                    "last_seek_fraction": observed_media[0].get("seek_fraction", selected_seek_fraction),
                    "last_observed": observed_media[:4],
                    "next_suggested_fraction": media_seek_fraction(
                        "revisit", {"read_count": record.get("read_count", 1)}
                    ),
                    "revisit_allowed": True,
                    "requested_seek_seconds": None,
                }

            document_cursor_for_continuation = None
            if isinstance(document_progress, dict):
                fragment_window_complete = next_offset >= total_fragments
                reaches_end = bool(document_progress.get("window_reaches_end"))
                document_complete = fragment_window_complete and reaches_end
                history_progress = dict(document_progress)
                history_progress["complete"] = document_complete
                history_progress["status"] = (
                    "complete" if document_complete else "partial"
                )
                record["document_read_progress"] = history_progress

                if not fragment_window_complete:
                    document_cursor_for_continuation = document_progress.get(
                        "window_start"
                    )
                elif not reaches_end:
                    document_cursor_for_continuation = document_progress.get(
                        "next_cursor"
                    )

                if not document_complete:
                    next_cursor = document_cursor_for_continuation or {}
                    log_to_statusbox(
                        f"[SelfRead] {path.name} remains partially read; "
                        f"resume cursor section {next_cursor.get('section', 0)}, "
                        f"character {next_cursor.get('char', 0)}."
                    )

            _set_self_read_continuation(
                record,
                stamp,
                next_offset=next_offset,
                total_fragments=total_fragments,
                document_cursor=document_cursor_for_continuation,
            )
            if next_offset < total_fragments:
                record["fragment_limit_truncated"] = True
                log_to_statusbox(
                    f"[SelfRead] Fragment limit saved range {start_offset}:{next_offset} "
                    f"of {total_fragments} from {path.name}; continuation recorded."
                )
            history_files[history_key] = record
            read_reason_counts[read_reason] += 1
            return True

        except Exception as exc:
            if is_broken_pipe_error(exc):
                report = report_self_read_broken_pipe(
                    child=child,
                    component="self_read",
                    operation=f"process_{category}",
                    error=exc,
                    source_message=(
                        f"[SelfRead] PROCESSING {path.name} "
                        f"[{category}; {read_reason}]"
                    ),
                    path_text=str(path),
                )
                note = (
                    "[SelfRead] Broken pipe explanation: "
                    f"{report.get('explanation') or str(exc)}"
                )
                issue_entry_id = str(report.get("issue_entry_id") or "").strip()
                if issue_entry_id:
                    note += f" GitHub queue entry: {issue_entry_id}."
                elif report.get("duplicate_within_cooldown"):
                    note += " Existing cooldown report reused."
                log_to_statusbox(note)
                return False
            log_to_statusbox(f"[SelfRead] ERROR processing {path.name}: {exc}")
            return False

    for base_root, audio_only, source_key in roots:
        if count >= FRAG_LIMIT:
            break
        if files_inspected >= inspection_limit:
            scan_stop_reason = scan_stop_reason or "file_budget"
            break
        if _scan_time_stop_requested():
            break
        log_to_statusbox(f"[SelfRead] Scanning: {base_root}")
        prune_generated = _should_prune_default_code_scan(
            base_root,
            default_root,
            source_key,
        )
        file_iter = _iter_self_read_files(
            base_root,
            audio_only=audio_only,
            prune_generated=prune_generated,
            stop_requested=_scan_time_stop_requested,
        )

        for path in file_iter:
            if count >= FRAG_LIMIT:
                break
            if files_inspected >= inspection_limit:
                scan_stop_reason = scan_stop_reason or "file_budget"
                break
            if _scan_time_stop_requested():
                break
            files_inspected += 1
            if not path.is_file():
                continue

            try:
                relative_path = path.relative_to(base_root)
            except ValueError:
                relative_path = path.name

            rel_str = (
                relative_path.as_posix()
                if isinstance(relative_path, Path)
                else str(relative_path)
            )
            if source_key == "code" and "/memory/github_history/" in f"/{rel_str}":
                continue

            log_to_statusbox(f"[SelfRead] Inspecting: {path}")

            skip_match = _match_skip_pattern(path, rel_str, skip_patterns)
            if skip_match:
                log_to_statusbox(
                    f"[SelfRead] SKIP {path.name} — preference skip rule '{skip_match}'."
                )
                continue

            category = classify_path(path)
            if not category:
                log_to_statusbox(f"[SelfRead] SKIP {path.name} — unrecognized type.")
                continue

            if not is_readable_file(path):
                log_to_statusbox(
                    f"[SelfRead] SKIP {path.name} — not a supported format or too large."
                )
                continue

            stamp = _file_stamp(path)
            if not stamp:
                log_to_statusbox(f"[SelfRead] SKIP {path.name} — file disappeared.")
                continue

            history_key, prior = _resolve_history_record(
                history_files,
                source_key=source_key,
                base_root=base_root,
                relative_path=rel_str,
                allow_legacy_basename=(base_root == default_root),
            )
            read_reason = classify_self_read_file(prior, stamp)
            candidate = {
                "path": path,
                "category": category,
                "base_root": base_root,
                "source_key": source_key,
                "relative_path": rel_str,
                "history_key": history_key,
                "prior": prior,
                "stamp": stamp,
                "audio_only": audio_only,
            }

            if read_reason in {"new", "updated", "resume"}:
                if count < primary_fragment_ceiling:
                    processed = _process_candidate(
                        candidate,
                        read_reason,
                        fragment_ceiling=primary_fragment_ceiling,
                    )
                    current = history_files.get(history_key) or {}
                    if processed and current.get("continuation") and primary_fallback is None:
                        primary_fallback = (candidate, "resume")
                elif primary_fallback is None:
                    primary_fallback = (candidate, read_reason)
                continue

            unchanged_seen_count += 1
            if prior.get("mtime_ns") is None or prior.get("size_bytes") is None:
                prior = _backfill_legacy_stamp(prior, stamp)
                prior.setdefault("source", source_key)
                prior.setdefault("relative_path", rel_str)
                prior.setdefault("root_path", str(base_root))
                history_files[history_key] = prior
                candidate["prior"] = prior

            if revisit_limit <= 0:
                log_to_statusbox(f"[SelfRead] SKIP {path.name} — already seen.")
                continue

            revisit_candidates = select_revisit_candidates(
                revisit_candidates + [candidate],
                read_focus,
                limit=revisit_limit,
                now_ts=scan_now_ts,
            )

        close_file_iter = getattr(file_iter, "close", None)
        if callable(close_file_iter):
            close_file_iter()
        if scan_stop_reason is not None:
            break

    if scan_stop_reason is not None:
        log_to_statusbox(
            f"[SelfRead] Inspection {scan_stop_reason.replace('_', ' ')} reached "
            f"after {files_inspected} file(s); ending this bounded pass."
        )

    if count < FRAG_LIMIT and revisit_candidates:
        selected_revisits = select_revisit_candidates(
            revisit_candidates,
            read_focus,
            limit=revisit_limit,
            now_ts=scan_now_ts,
        )
        log_to_statusbox(
            f"[SelfRead] Selected {len(selected_revisits)} unchanged file(s) "
            f"for a bounded {read_focus} revisit."
        )
        for candidate in selected_revisits:
            if count >= FRAG_LIMIT:
                break
            latest_stamp = _file_stamp(candidate["path"])
            if not latest_stamp:
                continue
            candidate["stamp"] = latest_stamp
            deferred_reason = classify_self_read_file(
                candidate.get("prior"),
                latest_stamp,
            )
            if deferred_reason in {"new", "updated", "resume"}:
                if primary_fallback is None:
                    primary_fallback = (candidate, deferred_reason)
                continue
            if _process_candidate(
                candidate,
                "revisit",
                fragment_ceiling=FRAG_LIMIT,
            ):
                seen_revisit_satisfied = True

    if count < FRAG_LIMIT and primary_fallback is not None:
        candidate, fallback_reason = primary_fallback
        latest_stamp = _file_stamp(candidate["path"])
        if latest_stamp:
            candidate["stamp"] = latest_stamp
            current_prior = history_files.get(candidate["history_key"])
            candidate["prior"] = current_prior
            current_reason = classify_self_read_file(current_prior, latest_stamp)
            if current_reason in {"new", "updated", "resume"}:
                fallback_reason = current_reason
            _process_candidate(
                candidate,
                fallback_reason,
                fragment_ceiling=FRAG_LIMIT,
            )

    if count >= FRAG_LIMIT:
        log_to_statusbox("[SelfRead] Fragment limit reached — stopping scan.")

    history_ledger["last_pass"] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source_override or "all",
        "read_focus": read_focus,
        "read_focus_source": focus_decision.get("source"),
        "read_focus_scores": {
            "new": focus_decision.get("new_score"),
            "seen": focus_decision.get("seen_score"),
        },
        "read_focus_drivers": focus_decision.get("drivers") or {},
        "primary_fragment_ceiling": primary_fragment_ceiling,
        "seen_revisit_fragment_reserve": FRAG_LIMIT - primary_fragment_ceiling,
        "seen_revisit_satisfied": seen_revisit_satisfied,
        "read_reason_counts": dict(read_reason_counts),
        "unchanged_seen_count": unchanged_seen_count,
        "fragments_saved": count,
        "files_inspected": files_inspected,
        "inspection_file_budget": inspection_limit,
        "inspection_time_budget_seconds": scan_seconds,
        "inspection_elapsed_seconds": round(
            max(0.0, time.monotonic() - scan_started_monotonic),
            3,
        ),
        "inspection_stop_reason": scan_stop_reason,
    }
    save_history(child, history_ledger)
    if _SELF_READ_LOCK_HELD:
        _write_runtime_state(
            "running",
            source=source_override or "all",
            read_focus_scores={
                "new": focus_decision.get("new_score"),
                "seen": focus_decision.get("seen_score"),
            },
            read_focus_drivers=focus_decision.get("drivers") or {},
            unchanged_seen_count=unchanged_seen_count,
            phase="training" if count > 0 else "complete",
            fragments_saved=count,
            files_processed=sum(read_reason_counts.values()),
            primary_fragment_ceiling=primary_fragment_ceiling,
            seen_revisit_fragment_reserve=FRAG_LIMIT - primary_fragment_ceiling,
            seen_revisit_satisfied=seen_revisit_satisfied,
            read_focus=read_focus,
            read_focus_source=focus_decision.get("source"),
            read_reason_counts=dict(read_reason_counts),
            files_inspected=files_inspected,
            inspection_file_budget=inspection_limit,
            inspection_time_budget_seconds=scan_seconds,
            inspection_stop_reason=scan_stop_reason,
        )
    log_to_statusbox(
        f"[SelfRead] Done. {count} fragments saved from "
        f"{read_reason_counts['new']} new, "
        f"{read_reason_counts['updated']} updated, "
        f"{read_reason_counts['resume']} resumed, and "
        f"{read_reason_counts['revisit']} revisited file(s)."
    )

    if count > 0:
        log_to_statusbox("[SelfRead] Calling training pipeline...")
        os.system("python train_fragments.py")
    else:
        log_to_statusbox("[SelfRead] No self-read fragments to train on.")


def pretrain_audio_digest(paths, child):
    log_to_statusbox(f"[PretrainDigest] Starting digest on {len(paths)} file(s) for {child}")
    transformer = FractalTransformer()

    if analyze_audio_clip is None or generate_fragment is None:
        log_to_statusbox(
            "[PretrainDigest] Audio digest unavailable: "
            f"{_AUDIO_DIGEST_IMPORT_ERROR}"
        )
        return

    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            log_to_statusbox(f"[PretrainDigest] File not found: {path}")
            continue

        if path.suffix.lower() not in [".mp3", ".wav", ".opus"]:
            log_to_statusbox(f"[PretrainDigest] Skipping unsupported file: {path.name}")
            continue

        try:
            log_to_statusbox(f"[PretrainDigest] Analyzing {path.name}...")
            result = analyze_audio_clip(path, transformer, child=child, label="pretrain")
            if result:
                generate_fragment(path, result, child=child, label="pretrain")
                log_to_statusbox(f"[PretrainDigest] + Fragment created for: {path.name}")
            else:
                log_to_statusbox(f"[PretrainDigest] Failed to analyze: {path.name}")
        except Exception as e:
            log_to_statusbox(f"[PretrainDigest] ERROR on {path.name}: {e}")


def main():
    if not _acquire_runtime_lock():
        return 0
    _install_runtime_signal_handlers()
    try:
        outcome = self_read_and_train()
    except Exception as exc:
        _release_runtime_lock("failed", error=exc)
        raise
    if outcome is False:
        _release_runtime_lock("failed", error="self_read_failed")
        return 1
    _release_runtime_lock("completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
