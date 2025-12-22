
# === raw_file_manager.py (Multimodal Self-Read) ===

import os
import json
import wave
import contextlib
import sys
import itertools
import io
import zipfile
import tarfile
import gzip
import bz2
import lzma
import uuid
import fnmatch
import random
import xml.etree.ElementTree as ET
from tempfile import NamedTemporaryFile
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image
import numpy as np
from transformers.fractal_multidimensional_transformers import FractalTransformer
from gui_hook import log_to_statusbox
from text_memory import update_text_vocab

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
try:
    import fitz  # type: ignore
except Exception as e:  # pragma: no cover - optional dependency
    fitz = None
    _PDF_IMPORT_ERROR = e


FRAG_LIMIT = 1000
TEXT_EXTENSIONS = {".txt", ".md", ".json", ".py"}
DOCUMENT_EXTENSIONS = {".pdf", ".odt"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
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

SELF_READ_PREF_FILENAME = "self_read_preferences.json"
SELF_READ_SKIP_REQUESTS = "self_read_skip_requests.json"
VALID_SOURCE_KEYS = {"code", "music", "books", "venv"}
SELF_READ_SOURCE_ENV = "SELF_READ_SOURCE"

DEFAULT_SELF_READ_PREFS = {
    "source_choices": {
        "code": True,
        "music": True,
        "books": True,
        "venv": False,
    },
    "skip_files": [],
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
}


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


def _self_read_pref_path(child):
    return Path("AI_Children") / child / "memory" / SELF_READ_PREF_FILENAME


def _skip_requests_path(child):
    return Path("AI_Children") / child / "memory" / SELF_READ_SKIP_REQUESTS


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


def annotate_fragment_source(fragment, source_key, relative_label, base_root):
    fragment["self_read_origin"] = source_key
    context = fragment.setdefault("source_context", {})
    context.setdefault("self_read_origin", source_key)
    context.setdefault("relative_path", relative_label)
    context.setdefault("root_path", str(base_root))

    annotations = SOURCE_ANNOTATIONS.get(source_key)
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
        context.setdefault("self_voice_hint", "ina_voice_reference")
        voice_name = Path(relative_label).stem if relative_label else ""
        if voice_name:
            context.setdefault("self_voice_reference", voice_name)


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

def get_child():
    log_to_statusbox("[RawFileManager] Attempting to retrieve 'child'...")

    # First try to get from environment variable
    child = os.getenv("CHILD")
    if child:
        log_to_statusbox(f"[RawFileManager] Found 'child' in environment: {child}")
        return child

    # If not found, try to get from command line argument
    if len(sys.argv) > 1:
        child = sys.argv[1]
        log_to_statusbox(f"[RawFileManager] Found 'child' in command line args: {child}")
        return child

    # Fallback to config.json if not set by environment or args
    config_path = Path("config.json")
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                child = config.get("current_child", "Inazuma_Yagami")
                log_to_statusbox(f"[RawFileManager] Found 'child' in config.json: {child}")
                return child
        except Exception as e:
            log_to_statusbox(f"[RawFileManager] Error loading config.json: {e}")
            return "Inazuma_Yagami"

    # If nothing works, return the default child
    log_to_statusbox("[RawFileManager] No valid 'child' found, using default: Inazuma_Yagami")
    return "Inazuma_Yagami"

child = get_child()

log_to_statusbox(f"[RawFileManager] Final child: {child}")


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

def load_history(child):
    path = Path("AI_Children") / child / "memory" / "read_history.json"
    if not path.exists():
        return []
    with open(path, "r") as f:
        return json.load(f)

def save_history(child, history):
    path = Path("AI_Children") / child / "memory" / "read_history.json"
    with open(path, "w") as f:
        json.dump(history[-250:], f, indent=4)

def log_reflection(child, fragment):
    path = Path("AI_Children") / child / "identity" / "self_reflection.json"
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
    cleaned = text.replace("\x00", " ")
    return " ".join(cleaned.split())


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


def _document_chunks(text, source, chunk_size=400, max_chunks=5):
    cleaned = _normalize_document_text(text)
    if not cleaned:
        return []
    seed = hash(source) & 0xFFFFFFFF
    starts = _document_chunk_starts(len(cleaned), chunk_size, max_chunks, seed)
    return [cleaned[start:start + chunk_size] for start in starts]


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


def fragment_document_text(text, source, transformer, doc_type=None):
    chunks = _document_chunks(text, source)
    if not chunks:
        return []

    fragments = []
    for chunk in chunks:
        frag_id = f"frag_text_{uuid.uuid4().hex[:10]}"
        tags = ["text", "self_read", "document"]
        if doc_type:
            tags.append(doc_type)
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
        vec = transformer.encode(frag)
        frag["importance"] = vec["importance"]
        try:
            update_text_vocab(
                chunk,
                child=child,
                tags=frag["tags"],
                emotions=frag.get("emotions"),
                source="raw_file_manager",
            )
        except Exception:
            pass
        fragments.append(frag)
    return fragments


def fragment_document(path, transformer):
    ext = path.suffix.lower()
    if ext == ".pdf":
        text = _extract_pdf_text(path)
        doc_type = "pdf"
    elif ext == ".odt":
        text = _extract_odt_text(path)
        doc_type = "odt"
    else:
        return []

    if not text:
        log_to_statusbox(f"[RawFileManager] No text extracted from {path}.")
        return []
    return fragment_document_text(text, path.name, transformer, doc_type=doc_type)


def fragment_document_bytes(data, source_label, transformer, suffix):
    ext = suffix.lower()
    if ext == ".pdf":
        text = _extract_pdf_text_bytes(data, source_label)
        doc_type = "pdf"
    elif ext == ".odt":
        text = _extract_odt_text_bytes(data, source_label)
        doc_type = "odt"
    else:
        return []

    if not text:
        log_to_statusbox(f"[RawFileManager] No text extracted from {source_label}.")
        return []
    return fragment_document_text(text, source_label, transformer, doc_type=doc_type)


def fragment_text(text, source, transformer):
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
                source="raw_file_manager",
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

        with Image.open(open_target) as img:
            array = np.array(img.convert("L")).flatten().tolist()

        if source_label:
            source = source_label
        elif isinstance(image_source, (str, Path)):
            source = str(image_source)
        else:
            source = getattr(image_source, "name", "<memory_image>")

        summary_name = Path(source).name if isinstance(source, str) else "image"
        frag = {
            "modality": "image",
            "image_features": array[:1024],
            "summary": f"Visual symbol or artifact from {summary_name}",
            "tags": ["self_read", "image"],
            "source": source,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "emotions": {"focus": 0.3, "novelty": 0.5}
        }
        vec = transformer.encode_image_fragment(frag)
        frag["importance"] = vec["importance"]
        return [frag]
    except Exception as e:
        label = source_label or image_source
        log_to_statusbox(f"[RawFileManager] Failed to process image {label}: {e}")
        return []

def fragment_audio(audio_path, transformer):
    ext = audio_path.suffix.lower()

    analysis = None
    if analyze_audio_clip is not None and ext in {".wav", ".mp3", ".opus"}:
        try:
            analysis = analyze_audio_clip(audio_path, transformer, child=child, label="self_read")
        except Exception as e:
            log_to_statusbox(f"[RawFileManager] Audio digest failed for {audio_path}: {e}")

    if ext == ".wav":
        try:
            with contextlib.closing(wave.open(str(audio_path), "r")) as wf:
                frames = wf.readframes(wf.getnframes())
                frame_count = wf.getnframes()
                sample_rate = wf.getframerate()
        except Exception as e:
            log_to_statusbox(f"[RawFileManager] Failed to process WAV {audio_path}: {e}")
            return []

        duration = frame_count / float(sample_rate or 1)
        audio_data = list(frames[:1024])
        tags = ["self_read", "audio"]

        if analysis:
            for tag in analysis.get("tags", []):
                if tag not in tags:
                    tags.append(tag)

        frag = {
            "modality": "audio",
            "audio_features": [x / 255.0 for x in audio_data],
            "summary": analysis.get("summary", f"Sound fragment from {audio_path.name}") if analysis else f"Sound fragment from {audio_path.name}",
            "tags": tags,
            "source": str(audio_path),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "emotions": (analysis.get("emotions") if analysis else None) or {"attention": 0.5, "novelty": 0.6},
            "duration": analysis.get("duration", duration) if analysis else duration,
        }

        if analysis:
            frag["symbols"] = analysis.get("symbols", [])
            frag["proto_words"] = analysis.get("proto_words", [])
            frag["analysis_paths"] = {
                "symbol_map": analysis.get("symbol_map_path"),
                "symbol_words": analysis.get("symbol_words_path"),
            }

        vec = transformer.encode_audio_fragment(frag)
        clarity = analysis.get("clarity") if analysis else None
        try:
            frag["importance"] = (
                round(float(clarity), 4) if clarity is not None else vec["importance"]
            )
        except (TypeError, ValueError):
            frag["importance"] = vec["importance"]

        return [frag]

    if ext in {".mp3", ".opus"}:
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

        tags = ["self_read", "audio"]
        for tag in analysis.get("tags", []):
            if tag not in tags:
                tags.append(tag)

        frag = {
            "modality": "audio",
            "summary": analysis.get("summary", f"Sound fragment from {audio_path.name}"),
            "tags": tags,
            "source": str(audio_path),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "emotions": analysis.get("emotions", {"attention": 0.5}),
            "symbols": analysis.get("symbols", []),
            "proto_words": analysis.get("proto_words", []),
            "duration": analysis.get("duration", 0),
        }

        frames = analysis.get("frames") or []
        if frames:
            frag["audio_features"] = frames[:256]

        vec = transformer.encode_audio_fragment(frag)
        clarity = analysis.get("clarity")
        try:
            frag["importance"] = (
                round(float(clarity), 4) if clarity is not None else vec["importance"]
            )
        except (TypeError, ValueError):
            frag["importance"] = vec["importance"]

        return [frag]

    log_to_statusbox(
        f"[RawFileManager] Unsupported audio format for {audio_path.name}: {ext}"
    )
    return []


def fragment_video(video_path, transformer, source_label=None):
    summary_parts = []
    preview_features = []
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

            target_frame = frame_count // 2 if frame_count else 0
            if target_frame:
                capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

            success, frame = capture.read()
            if success and frame is not None:
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    preview_image = Image.fromarray(frame_rgb).convert("L").resize((32, 32))
                    preview_features = (
                        np.array(preview_image).astype(float).flatten() / 255.0
                    ).tolist()
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

    frag = {
        "modality": "video",
        "summary": f"Video essay from {Path(source).name}{summary_details}",
        "tags": ["self_read", "video"],
        "source": source,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "emotions": {"focus": 0.4, "curiosity": 0.55},
    }

    if preview_features:
        frag["video_features"] = preview_features[:1024]
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

    vec = transformer.encode_video_fragment(frag)
    frag["importance"] = vec.get("importance", 0.0)
    return [frag]


def _fragments_from_data_buffer(data, inner_path, container_path, category, transformer):
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
                results = fragment_audio(temp_path, transformer)
            else:
                results = fragment_video(temp_path, transformer, source_label=source_label)
            for frag in results:
                frag["source"] = source_label
            return results

    return []


def process_archive(path, transformer):
    fragments = []

    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                for info in archive.infolist():
                    if info.is_dir() or info.file_size > ARCHIVE_MEMBER_LIMIT:
                        continue
                    inner_path = Path(info.filename)
                    category = classify_suffixes([s.lower() for s in inner_path.suffixes])
                    if not category or category == "archive":
                        continue
                    with archive.open(info, "r") as member:
                        data = member.read()
                    fragments.extend(
                        _fragments_from_data_buffer(
                            data, inner_path, path, category, transformer
                        )
                    )

        elif tarfile.is_tarfile(path):
            with tarfile.open(path, "r:*") as archive:
                for member in archive.getmembers():
                    if not member.isfile() or member.size > ARCHIVE_MEMBER_LIMIT:
                        continue
                    inner_path = Path(member.name)
                    category = classify_suffixes([s.lower() for s in inner_path.suffixes])
                    if not category or category == "archive":
                        continue
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        continue
                    data = extracted.read()
                    fragments.extend(
                        _fragments_from_data_buffer(
                            data, inner_path, path, category, transformer
                        )
                    )

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
                category = classify_suffixes([s.lower() for s in inner_name.suffixes])
                if category and category != "archive":
                    try:
                        with opener(path, "rb") as compressed:
                            data = _read_limited(compressed, ARCHIVE_MEMBER_LIMIT)
                    except ValueError:
                        log_to_statusbox(
                            f"[RawFileManager] Skipping {path} because the decompressed size exceeds the per-file limit"
                        )
                    else:
                        fragments.extend(
                            _fragments_from_data_buffer(
                                data, inner_name, path, category, transformer
                            )
                        )
    except Exception as e:
        log_to_statusbox(f"[RawFileManager] Failed to process archive {path}: {e}")

    return fragments

def self_read_and_train():
    child = get_child()
    default_root = Path.home() / "Projects" / "Project Inazuma"
    prefs = load_self_read_preferences(child)
    prefs = _apply_skip_requests(child, prefs)
    source_choices = prefs.get("source_choices", DEFAULT_SELF_READ_PREFS["source_choices"])
    skip_patterns = prefs.get("skip_files", [])
    source_override = _load_self_read_source_override()
    if source_override and not source_choices.get(source_override, False):
        log_to_statusbox(f"[SelfRead] Source override '{source_override}' ignored by preference.")
        source_override = None

    raw_history = load_history(child)
    history = {entry for entry in raw_history if "/" in entry}
    legacy_history = {entry for entry in raw_history if "/" not in entry}
    new_fragments = []

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
    if roots:
        root_descriptions = ", ".join(
            f"{str(path)} [{source_key}]" for path, _, source_key in roots
        )
        log_to_statusbox("[SelfRead] Roots to scan: " + root_descriptions)
    else:
        log_to_statusbox("[SelfRead] No available roots to scan.")
        return
    log_to_statusbox(f"[SelfRead] Loaded {len(history) + len(legacy_history)} previously seen files.")

    # Resolve roots once for provenance tagging
    def _safe_resolve(p):
        try:
            return p.resolve()
        except Exception:
            return None

    transformer = FractalTransformer()
    count = 0

    audio_patterns = ("*.wav", "*.mp3", "*.opus")

    for base_root, audio_only, source_key in roots:
        log_to_statusbox(f"[SelfRead] Scanning: {base_root}")
        if audio_only:
            file_iter = itertools.chain.from_iterable(base_root.rglob(pattern) for pattern in audio_patterns)
        else:
            file_iter = base_root.rglob("*")

        for path in file_iter:
            if not path.is_file():
                continue

            try:
                relative_path = path.relative_to(base_root)
            except ValueError:
                relative_path = path.name

            rel_str = relative_path.as_posix() if isinstance(relative_path, Path) else str(relative_path)
            history_key = f"{base_root.name}/{rel_str}"

            log_to_statusbox(f"[SelfRead] Inspecting: {path}")

            if history_key in history or (base_root == default_root and path.name in legacy_history):
                log_to_statusbox(f"[SelfRead] SKIP {path.name} — already seen.")
                continue

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

            log_to_statusbox(f"[SelfRead] PROCESSING {path.name} [{category}]")

            try:
                if category == "text":
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                    result = fragment_text(text, path.name, transformer)

                elif category == "document":
                    result = fragment_document(path, transformer)

                elif category == "image":
                    result = fragment_image(path, transformer)

                elif category == "audio":
                    result = fragment_audio(path, transformer)

                elif category == "video":
                    result = fragment_video(path, transformer)

                elif category == "archive":
                    result = process_archive(path, transformer)

                else:
                    log_to_statusbox(
                        f"[SelfRead] SKIP {path.name} — unsupported processing category {category}."
                    )
                    continue

                if result:
                    for frag in result:
                        frag_id = frag.get("id") or f"frag_text_{uuid.uuid4().hex[:10]}"
                        frag["id"] = frag_id

                        annotate_fragment_source(frag, source_key, rel_str, base_root)

                        frag_path = Path("AI_Children") / child / "memory" / "fragments" / f"{frag_id}.json"
                        frag_path.parent.mkdir(parents=True, exist_ok=True)

                        with open(frag_path, "w", encoding="utf-8") as f:
                            json.dump(frag, f, indent=4)

                        log_to_statusbox(f"[SelfRead] + Fragment saved: {frag_id} from {path.name}")
                        log_reflection(child, frag)
                        new_fragments.append(frag)

                    if base_root == default_root:
                        legacy_history.discard(path.name)
                    history.add(history_key)
                    count += len(result)

            except Exception as e:
                log_to_statusbox(f"[SelfRead] ERROR processing {path.name}: {e}")

            if count >= FRAG_LIMIT:
                log_to_statusbox("[SelfRead] Fragment limit reached — stopping scan.")
                break

        if count >= FRAG_LIMIT:
            break

    combined_history = list(history.union(legacy_history))
    save_history(child, combined_history)
    log_to_statusbox(f"[SelfRead] Done. {count} new fragments saved.")

    if count > 0:
        log_to_statusbox("[SelfRead] Calling training pipeline...")
        os.system("python train_fragments.py")
    else:
        log_to_statusbox("[SelfRead] No new fragments to train on.")


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


if __name__ == "__main__":
    self_read_and_train()
