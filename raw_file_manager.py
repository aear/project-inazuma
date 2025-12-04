
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
from tempfile import NamedTemporaryFile
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image
import numpy as np
from transformers.fractal_multidimensional_transformers import FractalTransformer
from gui_hook import log_to_statusbox

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


FRAG_LIMIT = 1000
TEXT_EXTENSIONS = {".txt", ".md", ".json", ".py"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}
VIDEO_EXTENSIONS = {".mp4", ".m4v", ".mov", ".avi", ".webm"}
SIMPLE_COMPRESSED_EXTENSIONS = {".gz", ".bz2", ".xz"}

FILE_SIZE_LIMITS = {
    "text": 5 * 1024 * 1024,        # 5 MB
    "image": 25 * 1024 * 1024,      # 25 MB
    "audio": 75 * 1024 * 1024,      # 75 MB
    "video": 800 * 1024 * 1024,     # 800 MB
    "archive": 800 * 1024 * 1024,   # 800 MB for compressed bundles
}

ARCHIVE_MEMBER_LIMIT = 50 * 1024 * 1024  # 50 MB per file inside an archive


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

def fragment_text(text, source, transformer):
    chunks = [text[i:i+400] for i in range(0, len(text), 400)]
    fragments = []
    for chunk in chunks[:5]:
        frag = {
            "summary": chunk,
            "tags": ["self_read", "code" if source.endswith(".py") else "text"],
            "source": source,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "emotions": {"curiosity": 0.6, "focus": 0.4}
        }
        vec = transformer.encode(frag)
        frag["importance"] = vec["importance"]
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
    if analyze_audio_clip is not None and ext in {".wav", ".mp3"}:
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

    if ext == ".mp3":
        if analysis is None:
            if analyze_audio_clip is None:
                log_to_statusbox(
                    "[RawFileManager] MP3 decoding unavailable: "
                    f"{_AUDIO_DIGEST_IMPORT_ERROR}"
                )
            else:
                log_to_statusbox(
                    f"[RawFileManager] MP3 analysis returned no data for {audio_path.name}."
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

    raw_history = load_history(child)
    history = {entry for entry in raw_history if "/" in entry}
    legacy_history = {entry for entry in raw_history if "/" not in entry}
    new_fragments = []

    roots = []
    seen_roots = set()

    def add_root(path, audio_only=False):
        try:
            resolved = path.resolve()
        except FileNotFoundError:
            return
        if resolved in seen_roots:
            return
        seen_roots.add(resolved)
        roots.append((path, audio_only))

    if default_root.exists():
        add_root(default_root, audio_only=False)
    else:
        log_to_statusbox(f"[SelfRead] Project root not found: {default_root}")

    if book_folder_path and book_folder_path.exists():
        add_root(book_folder_path, audio_only=False)
    elif book_folder_path:
        log_to_statusbox(f"[SelfRead] Book folder not found: {book_folder_path}")

    if music_folder_path and music_folder_path.exists():
        add_root(music_folder_path, audio_only=True)
    elif music_folder_path:
        log_to_statusbox(f"[SelfRead] Music folder not found: {music_folder_path}")

    if ina_work_path and ina_work_path.exists():
        add_root(ina_work_path, audio_only=False)
    elif ina_work_path:
        log_to_statusbox(f"[SelfRead] Ina work folder not found: {ina_work_path}")

    log_to_statusbox(f"[SelfRead] Child set to: {child}")
    if roots:
        log_to_statusbox("[SelfRead] Roots to scan: " + ", ".join(str(path) for path, _ in roots))
    else:
        log_to_statusbox("[SelfRead] No available roots to scan.")
        return
    log_to_statusbox(f"[SelfRead] Loaded {len(history) + len(legacy_history)} previously seen files.")

    transformer = FractalTransformer()
    count = 0

    audio_patterns = ("*.wav", "*.mp3")

    for base_root, audio_only in roots:
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
                        frag_id = f"frag_selfread_{abs(hash(frag['summary'])) % 10**12}"
                        frag["id"] = frag_id

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

        if not path.suffix.lower() in [".mp3", ".wav"]:
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
