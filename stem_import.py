"""Bounded, non-destructive imports for Ina Music Studio stem collections."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from typing import BinaryIO, Callable, Sequence
import uuid
import wave
import zipfile

from daw_engine import (
    MAX_AUDIO_STEMS,
    MAX_RENDER_SAMPLES,
    MAX_SAMPLE_RATE,
    MAX_WAV_CHANNELS,
    MAX_WAV_DECODE_SAMPLES,
)


MAX_STEM_ZIP_BYTES = 512 * 1024 * 1024
MAX_STEM_ARCHIVE_ENTRIES = 64
MAX_STEM_MEMBER_BYTES = 128 * 1024 * 1024
MAX_STEM_TOTAL_BYTES = 256 * 1024 * 1024
MAX_COMPANION_FILES = 8
MAX_COMPANION_TEXT_BYTES = 2 * 1024 * 1024
COPY_CHUNK_BYTES = 128 * 1024


class StemImportError(ValueError):
    """Raised when a selected stem collection is unsafe or unsupported."""


class StemImportCancelled(RuntimeError):
    """Raised when window shutdown cancels an in-progress import."""


@dataclass(frozen=True)
class WavInfo:
    sample_rate: int
    frames: int
    channels: int
    sample_width: int


@dataclass(frozen=True)
class ImportedStem:
    path: Path
    name: str
    role: str
    source_member: str
    sha256: str
    size_bytes: int
    wav: WavInfo


@dataclass(frozen=True)
class ImportedCompanion:
    path: Path
    name: str
    source_member: str
    sha256: str
    size_bytes: int
    kind: str = "music_context"


@dataclass(frozen=True)
class StemImportResult:
    collection_dir: Path
    collection_name: str
    stems: tuple[ImportedStem, ...]
    companions: tuple[ImportedCompanion, ...]
    manifest_path: Path


@dataclass(frozen=True)
class _SelectedWavSource:
    selected_path: Path
    resolved_path: Path
    size_bytes: int


def _cancelled(callback: Callable[[], bool] | None) -> bool:
    return bool(callback and callback())


def _check_cancelled(callback: Callable[[], bool] | None) -> None:
    if _cancelled(callback):
        raise StemImportCancelled("Stem import was cancelled.")


def _safe_component(value: object, default: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    text = text.strip("._-")[:80]
    return text or default


def infer_stem_role(label: object) -> str:
    """Infer a display-only role from a stem filename."""
    words = {
        part
        for part in re.split(r"[^a-z0-9]+", str(label or "").casefold())
        if part
    }
    if words & {"vocal", "vocals", "voice", "voices", "leadvox", "backingvox"}:
        return "vocals"
    if words & {"drum", "drums", "percussion", "kick", "snare", "cymbals"}:
        return "drums"
    if words & {"bass", "sub", "subbass"}:
        return "bass"
    if words & {"guitar", "guitars", "acoustic", "electric"}:
        return "guitar"
    if words & {"piano", "keys", "keyboard", "synth", "synths"}:
        return "keys"
    return "other"


def infer_companion_kind(label: object) -> str:
    """Classify filename-signalled lyric/style prompts without overclaiming TXT files."""
    words = {
        part
        for part in re.split(r"[^a-z0-9]+", str(label or "").casefold())
        if part
    }
    if words & {"lyric", "lyrics", "style", "styles", "prompt", "prompts"}:
        return "lyrics_style_context"
    return "music_context"


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except (OSError, ValueError):
        return False


def _zip_declared_entry_count(path: Path) -> int | None:
    """Read ZIP end metadata before ZipFile loads its central directory."""
    try:
        with path.open("rb") as handle:
            end_record = zipfile._EndRecData(handle)
        if end_record is None:
            return None
        return int(end_record[zipfile._ECD_ENTRIES_TOTAL])
    except (AttributeError, IndexError, TypeError, ValueError):
        return None


def _normalized_member_name(raw_name: str) -> str:
    if not isinstance(raw_name, str) or not raw_name or "\x00" in raw_name:
        raise StemImportError("Stem ZIP contains an invalid member name.")
    normalized = raw_name.replace("\\", "/")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        raise StemImportError(f"Stem ZIP member must be relative: {raw_name!r}")
    parts = PurePosixPath(normalized).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise StemImportError(f"Stem ZIP member has an unsafe path: {raw_name!r}")
    return PurePosixPath(*parts).as_posix()


def _reject_special_zip_entry(info: zipfile.ZipInfo) -> None:
    if info.flag_bits & 0x1:
        raise StemImportError(f"Encrypted ZIP member is not supported: {info.filename}")
    mode = (info.external_attr >> 16) & 0xFFFF
    file_type = stat.S_IFMT(mode)
    if file_type and file_type not in {stat.S_IFREG, stat.S_IFDIR}:
        raise StemImportError(f"ZIP links and special files are not supported: {info.filename}")


def _copy_stream(
    source: BinaryIO,
    destination: Path,
    *,
    maximum_bytes: int,
    cancelled: Callable[[], bool] | None,
) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    with destination.open("xb") as target:
        while True:
            _check_cancelled(cancelled)
            chunk = source.read(COPY_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > maximum_bytes:
                raise StemImportError(
                    f"Imported member exceeds the {maximum_bytes:,}-byte limit."
                )
            target.write(chunk)
            digest.update(chunk)
        target.flush()
        os.fsync(target.fileno())
    return total, digest.hexdigest()


def inspect_pcm_wav(path: Path | str) -> WavInfo:
    """Validate an engine-compatible PCM WAV without loading it into memory."""
    wav_path = Path(path)
    try:
        with wave.open(str(wav_path), "rb") as source:
            if source.getcomptype() != "NONE":
                raise StemImportError(f"Compressed WAV is not supported: {wav_path.name}")
            channels = int(source.getnchannels())
            sample_width = int(source.getsampwidth())
            sample_rate = int(source.getframerate())
            frames = int(source.getnframes())
            if not 1 <= channels <= MAX_WAV_CHANNELS:
                raise StemImportError(f"Invalid WAV channel count in {wav_path.name}.")
            if not 1 <= sample_width <= 4:
                raise StemImportError(f"Unsupported WAV sample width in {wav_path.name}.")
            if not 1 <= sample_rate <= MAX_SAMPLE_RATE:
                raise StemImportError(f"Invalid WAV sample rate in {wav_path.name}.")
            if not 0 <= frames <= MAX_RENDER_SAMPLES:
                raise StemImportError(f"WAV is longer than the DAW render limit: {wav_path.name}")
            if frames * channels > MAX_WAV_DECODE_SAMPLES:
                raise StemImportError(f"WAV exceeds the DAW decode budget: {wav_path.name}")

            frame_width = channels * sample_width
            expected = frames * frame_width
            received = 0
            remaining_frames = frames
            chunk_frames = max(1, COPY_CHUNK_BYTES // frame_width)
            while remaining_frames:
                chunk = source.readframes(min(chunk_frames, remaining_frames))
                if not chunk:
                    break
                if len(chunk) % frame_width:
                    raise StemImportError(f"WAV frame data is incomplete: {wav_path.name}")
                received += len(chunk)
                remaining_frames -= len(chunk) // frame_width
            if received != expected:
                raise StemImportError(f"WAV frame data is incomplete: {wav_path.name}")
    except StemImportError:
        raise
    except (EOFError, OSError, wave.Error) as exc:
        raise StemImportError(f"Could not read PCM WAV {wav_path.name}: {exc}") from exc
    return WavInfo(
        sample_rate=sample_rate,
        frames=frames,
        channels=channels,
        sample_width=sample_width,
    )


def _create_stage(destination_root: Path, cancelled: Callable[[], bool] | None) -> Path:
    _check_cancelled(cancelled)
    destination_root.mkdir(parents=True, exist_ok=True)
    resolved_root = destination_root.resolve()
    stage = Path(tempfile.mkdtemp(prefix=".stem_import_", dir=resolved_root))
    if not _is_within(stage, resolved_root):
        try:
            shutil.rmtree(stage)
        except Exception:
            pass
        raise StemImportError("Stem staging folder escaped the studio.")
    return stage


def _write_manifest(
    stage: Path,
    *,
    collection_name: str,
    source_kind: str,
    source_name: str,
    stems: Sequence[dict[str, object]],
    companions: Sequence[dict[str, object]],
) -> None:
    payload = {
        "schema_version": 1,
        "kind": "ina_music_stem_collection",
        "collection": collection_name,
        "imported_at": datetime.now(timezone.utc).isoformat(),
        "source": {"kind": source_kind, "name": source_name},
        "stems": list(stems),
        "companions": list(companions),
    }
    manifest = stage / "manifest.json"
    with manifest.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _commit_result(
    stage: Path,
    destination_root: Path,
    *,
    collection_name: str,
    stems: Sequence[dict[str, object]],
    companions: Sequence[dict[str, object]],
    cancelled: Callable[[], bool] | None,
) -> StemImportResult:
    _check_cancelled(cancelled)
    root = destination_root.resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory_name = (
        f"{_safe_component(collection_name, 'stem_collection')}_{timestamp}_"
        f"{uuid.uuid4().hex[:6]}"
    )
    final_dir = root / directory_name
    if not _is_within(final_dir, root) or final_dir.exists():
        raise StemImportError("Could not reserve a safe stem collection folder.")
    os.replace(stage, final_dir)

    imported_stems = tuple(
        ImportedStem(
            path=final_dir / str(item["relative_path"]),
            name=str(item["name"]),
            role=str(item["role"]),
            source_member=str(item["source_member"]),
            sha256=str(item["sha256"]),
            size_bytes=int(item["size_bytes"]),
            wav=WavInfo(**dict(item["wav"])),
        )
        for item in stems
    )
    imported_companions = tuple(
        ImportedCompanion(
            path=final_dir / str(item["relative_path"]),
            name=str(item["name"]),
            source_member=str(item["source_member"]),
            sha256=str(item["sha256"]),
            size_bytes=int(item["size_bytes"]),
            kind=str(item.get("kind") or "music_context"),
        )
        for item in companions
    )
    return StemImportResult(
        collection_dir=final_dir,
        collection_name=collection_name,
        stems=imported_stems,
        companions=imported_companions,
        manifest_path=final_dir / "manifest.json",
    )


def _cleanup_stage(stage: Path | None, destination_root: Path) -> None:
    try:
        if stage is not None and stage.exists() and _is_within(stage, destination_root):
            shutil.rmtree(stage)
    except Exception:
        # Cleanup is best-effort and must never hide the import error that led here.
        pass


def _validated_stem_limit(maximum_stems: int) -> int:
    if isinstance(maximum_stems, bool) or not isinstance(maximum_stems, int):
        raise StemImportError("maximum_stems must be an integer.")
    if not 1 <= maximum_stems <= MAX_AUDIO_STEMS:
        raise StemImportError(
            f"maximum_stems must be between 1 and {MAX_AUDIO_STEMS}."
        )
    return maximum_stems


def _selected_wav_sources(
    sources: Sequence[Path | str],
    *,
    stem_limit: int,
) -> tuple[_SelectedWavSource, ...]:
    selected_paths = tuple(Path(item).expanduser() for item in sources)
    if not selected_paths:
        raise StemImportError("Select at least one WAV stem.")
    if len(selected_paths) > stem_limit:
        raise StemImportError(f"This project can accept at most {stem_limit} more stems.")

    validated: list[_SelectedWavSource] = []
    canonical_seen: set[str] = set()
    total_size = 0
    for selected in selected_paths:
        if selected.suffix.casefold() != ".wav":
            raise StemImportError(f"Stem source must be a WAV file: {selected.name}")
        try:
            resolved = selected.resolve(strict=True)
            source_stat = resolved.stat()
        except OSError as exc:
            raise StemImportError(
                f"Could not read stem source {selected.name}: {exc}"
            ) from exc
        if not stat.S_ISREG(source_stat.st_mode):
            raise StemImportError(f"Stem source must be a WAV file: {selected.name}")

        canonical_key = os.path.normcase(os.fspath(resolved))
        if canonical_key in canonical_seen:
            raise StemImportError(
                f"The same WAV stem was selected more than once: {selected.name}"
            )
        canonical_seen.add(canonical_key)

        size = int(source_stat.st_size)
        if size < 0 or size > MAX_STEM_MEMBER_BYTES:
            raise StemImportError(f"Stem exceeds the per-file limit: {selected.name}")
        total_size += size
        if total_size > MAX_STEM_TOTAL_BYTES:
            raise StemImportError("Selected stems exceed the collection byte limit.")
        validated.append(
            _SelectedWavSource(
                selected_path=selected,
                resolved_path=resolved,
                size_bytes=size,
            )
        )
    return tuple(validated)


def import_wav_stems(
    sources: Sequence[Path | str],
    destination_root: Path | str,
    *,
    collection_name: str,
    maximum_stems: int = MAX_AUDIO_STEMS,
    cancelled: Callable[[], bool] | None = None,
) -> StemImportResult:
    """Copy selected WAVs into one atomic, studio-local stem collection."""
    stem_limit = _validated_stem_limit(maximum_stems)
    source_records = _selected_wav_sources(sources, stem_limit=stem_limit)
    display_collection = str(collection_name or "").strip()[:120] or "Stem collection"

    root = Path(destination_root).expanduser()
    stage: Path | None = None
    stem_records: list[dict[str, object]] = []
    copied_total = 0
    try:
        stage = _create_stage(root, cancelled)
        for index, source_record in enumerate(source_records, start=1):
            _check_cancelled(cancelled)
            selected = source_record.selected_path
            display_name = selected.stem.strip()[:120] or f"Stem {index}"
            relative_name = f"{index:02d}_{_safe_component(display_name, 'stem')}.wav"
            target = stage / relative_name
            try:
                with source_record.resolved_path.open("rb") as handle:
                    size, digest = _copy_stream(
                        handle,
                        target,
                        maximum_bytes=MAX_STEM_MEMBER_BYTES,
                        cancelled=cancelled,
                    )
            except (StemImportCancelled, StemImportError):
                raise
            except OSError as exc:
                raise StemImportError(
                    f"Could not copy stem source {selected.name}: {exc}"
                ) from exc
            if size != source_record.size_bytes:
                raise StemImportError(
                    f"Stem source size changed during import: {selected.name}"
                )
            copied_total += size
            if copied_total > MAX_STEM_TOTAL_BYTES:
                raise StemImportError(
                    "Selected stems exceed the collection byte limit."
                )
            wav = inspect_pcm_wav(target)
            stem_records.append(
                {
                    "relative_path": relative_name,
                    "name": display_name,
                    "role": infer_stem_role(display_name),
                    "source_member": selected.name,
                    "sha256": digest,
                    "size_bytes": size,
                    "wav": wav.__dict__,
                }
            )
        _write_manifest(
            stage,
            collection_name=display_collection,
            source_kind="selected_wavs",
            source_name=f"{len(source_records)} selected WAV file(s)",
            stems=stem_records,
            companions=[],
        )
        result = _commit_result(
            stage,
            root,
            collection_name=display_collection,
            stems=stem_records,
            companions=[],
            cancelled=cancelled,
        )
        stage = None
        return result
    finally:
        _cleanup_stage(stage, root)


def import_stem_zip(
    zip_path: Path | str,
    destination_root: Path | str,
    *,
    collection_name: str | None = None,
    maximum_stems: int = MAX_AUDIO_STEMS,
    cancelled: Callable[[], bool] | None = None,
) -> StemImportResult:
    """Import PCM WAV stems and small TXT companions from a bounded ZIP."""
    selected_source = Path(zip_path).expanduser()
    if selected_source.suffix.casefold() != ".zip":
        raise StemImportError("Select a ZIP stem collection.")
    try:
        source = selected_source.resolve(strict=True)
        source_stat = source.stat()
    except OSError as exc:
        raise StemImportError(
            f"Could not read stem ZIP {selected_source.name}: {exc}"
        ) from exc
    if not stat.S_ISREG(source_stat.st_mode):
        raise StemImportError("Select a ZIP stem collection.")
    source_size = int(source_stat.st_size)
    if source_size < 0 or source_size > MAX_STEM_ZIP_BYTES:
        raise StemImportError("Stem ZIP exceeds the input byte limit.")
    stem_limit = _validated_stem_limit(maximum_stems)
    try:
        declared_count = _zip_declared_entry_count(source)
    except OSError as exc:
        raise StemImportError(
            f"Could not read stem ZIP {selected_source.name}: {exc}"
        ) from exc
    if declared_count is None or declared_count > MAX_STEM_ARCHIVE_ENTRIES:
        raise StemImportError(
            f"Stem ZIP may contain at most {MAX_STEM_ARCHIVE_ENTRIES} entries."
        )

    display_collection = (
        str(collection_name or selected_source.stem).strip()[:120] or "Stem collection"
    )
    root = Path(destination_root).expanduser()
    stage: Path | None = None
    stem_records: list[dict[str, object]] = []
    companion_records: list[dict[str, object]] = []
    try:
        stage = _create_stage(root, cancelled)
        try:
            archive_handle = zipfile.ZipFile(source, "r")
        except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
            raise StemImportError(
                f"Could not read stem ZIP {selected_source.name}: {exc}"
            ) from exc
        with archive_handle as archive:
            infos = archive.infolist()
            if len(infos) != declared_count or len(infos) > MAX_STEM_ARCHIVE_ENTRIES:
                raise StemImportError("Stem ZIP entry count changed during import.")

            selected: list[tuple[zipfile.ZipInfo, str, str]] = []
            normalized_seen: set[str] = set()
            declared_total = 0
            stem_count = 0
            text_count = 0
            for info in infos:
                _check_cancelled(cancelled)
                _reject_special_zip_entry(info)
                member_name = _normalized_member_name(info.filename.rstrip("/"))
                if info.is_dir():
                    continue
                normalized_key = member_name.casefold()
                if normalized_key in normalized_seen:
                    raise StemImportError(f"Duplicate ZIP member path: {member_name}")
                normalized_seen.add(normalized_key)
                declared_size = int(info.file_size)
                if declared_size < 0:
                    raise StemImportError(f"Invalid member size: {member_name}")
                declared_total += declared_size
                if declared_total > MAX_STEM_TOTAL_BYTES:
                    raise StemImportError("Stem ZIP exceeds the expanded byte limit.")
                suffix = PurePosixPath(member_name).suffix.casefold()
                if suffix == ".wav":
                    stem_count += 1
                    if stem_count > stem_limit:
                        raise StemImportError(
                            f"This project can accept at most {stem_limit} more WAV stems."
                        )
                    if declared_size > MAX_STEM_MEMBER_BYTES:
                        raise StemImportError(f"Stem member is too large: {member_name}")
                    selected.append((info, member_name, "audio_stem"))
                elif suffix == ".txt":
                    text_count += 1
                    if text_count > MAX_COMPANION_FILES:
                        raise StemImportError(
                            f"A collection may contain at most {MAX_COMPANION_FILES} text companions."
                        )
                    if declared_size > MAX_COMPANION_TEXT_BYTES:
                        raise StemImportError(f"Text companion is too large: {member_name}")
                    selected.append((info, member_name, "text_companion"))

            if not stem_count:
                raise StemImportError("Stem ZIP contains no WAV stems.")

            stem_index = 0
            text_index = 0
            for info, member_name, kind in selected:
                _check_cancelled(cancelled)
                display_name = PurePosixPath(member_name).stem.strip()[:120]
                if kind == "audio_stem":
                    stem_index += 1
                    relative_name = (
                        f"{stem_index:02d}_{_safe_component(display_name, 'stem')}.wav"
                    )
                    limit = MAX_STEM_MEMBER_BYTES
                else:
                    text_index += 1
                    relative_name = (
                        f"context_{text_index:02d}_"
                        f"{_safe_component(display_name, 'notes')}.txt"
                    )
                    limit = MAX_COMPANION_TEXT_BYTES
                target = stage / relative_name
                try:
                    with archive.open(info, "r") as member:
                        size, digest = _copy_stream(
                            member,
                            target,
                            maximum_bytes=limit,
                            cancelled=cancelled,
                        )
                except (EOFError, OSError, RuntimeError, zipfile.BadZipFile) as exc:
                    raise StemImportError(
                        f"Could not safely read ZIP member {member_name}: {exc}"
                    ) from exc
                if size != int(info.file_size):
                    raise StemImportError(f"ZIP member size changed: {member_name}")

                if kind == "audio_stem":
                    wav = inspect_pcm_wav(target)
                    stem_records.append(
                        {
                            "relative_path": relative_name,
                            "name": display_name or f"Stem {stem_index}",
                            "role": infer_stem_role(display_name),
                            "source_member": member_name,
                            "sha256": digest,
                            "size_bytes": size,
                            "wav": wav.__dict__,
                        }
                    )
                else:
                    context_kind = infer_companion_kind(display_name)
                    companion_records.append(
                        {
                            "relative_path": relative_name,
                            "name": display_name or f"Notes {text_index}",
                            "source_member": member_name,
                            "sha256": digest,
                            "size_bytes": size,
                            "kind": context_kind,
                        }
                    )

        _write_manifest(
            stage,
            collection_name=display_collection,
            source_kind="zip",
            source_name=selected_source.name,
            stems=stem_records,
            companions=companion_records,
        )
        result = _commit_result(
            stage,
            root,
            collection_name=display_collection,
            stems=stem_records,
            companions=companion_records,
            cancelled=cancelled,
        )
        stage = None
        return result
    except (zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise StemImportError(f"Could not read stem ZIP {selected_source.name}: {exc}") from exc
    finally:
        _cleanup_stage(stage, root)


__all__ = [
    "MAX_STEM_ZIP_BYTES",
    "MAX_STEM_ARCHIVE_ENTRIES",
    "MAX_STEM_MEMBER_BYTES",
    "MAX_STEM_TOTAL_BYTES",
    "MAX_COMPANION_FILES",
    "MAX_COMPANION_TEXT_BYTES",
    "StemImportError",
    "StemImportCancelled",
    "WavInfo",
    "ImportedStem",
    "ImportedCompanion",
    "StemImportResult",
    "infer_stem_role",
    "infer_companion_kind",
    "inspect_pcm_wav",
    "import_wav_stems",
    "import_stem_zip",
]
