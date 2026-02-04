#!/usr/bin/env python3
"""
Bundle many small files into chunk files plus an index to reduce file count.

Default behavior is a dry run: it scans, estimates bundle counts, and reports
the projected file-count reduction without modifying data.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import stat
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

DEFAULT_MAX_FILES_PER_BUNDLE = 10_000
DEFAULT_MAX_BUNDLE_BYTES = 512 * 1024 * 1024
DEFAULT_MAX_FILE_BYTES = 256 * 1024
DEFAULT_BUNDLE_PREFIX = "bundle_"
DEFAULT_DATA_SUFFIX = ".pack"
DEFAULT_INDEX_SUFFIX = ".index.jsonl"
DEFAULT_MANIFEST_NAME = "bundle_manifest.json"
DEFAULT_EXCLUDE_DIRS = {".git", "__pycache__", ".bundles", "bundles"}
READ_CHUNK_SIZE = 64 * 1024


@dataclass
class PlanReport:
    total_files: int = 0
    eligible_files: int = 0
    eligible_bytes: int = 0
    ineligible_files: int = 0
    ineligible_bytes: int = 0
    skipped_files: int = 0
    bundles: int = 0
    bundle_overhead_files: int = 2
    global_overhead_files: int = 1

    @property
    def new_file_count(self) -> int:
        overhead = 0
        if self.bundles > 0:
            overhead = (self.bundles * self.bundle_overhead_files) + self.global_overhead_files
        return self.ineligible_files + self.skipped_files + overhead

    def to_dict(self) -> dict:
        return {
            "total_files": self.total_files,
            "eligible_files": self.eligible_files,
            "eligible_bytes": self.eligible_bytes,
            "ineligible_files": self.ineligible_files,
            "ineligible_bytes": self.ineligible_bytes,
            "skipped_files": self.skipped_files,
            "bundles": self.bundles,
            "bundle_overhead_files": self.bundle_overhead_files,
            "global_overhead_files": self.global_overhead_files,
            "new_file_count": self.new_file_count,
        }


class BundlePlanner:
    def __init__(self, max_files: int, max_bytes: int) -> None:
        self.max_files = max_files
        self.max_bytes = max_bytes
        self.current_files = 0
        self.current_bytes = 0
        self.bundles = 0

    def add(self, size: int) -> None:
        if self.current_files == 0:
            self.bundles += 1
        elif self.current_files + 1 > self.max_files or self.current_bytes + size > self.max_bytes:
            self.bundles += 1
            self.current_files = 0
            self.current_bytes = 0
        self.current_files += 1
        self.current_bytes += size


class BundleWriter:
    def __init__(
        self,
        bundle_dir: Path,
        bundle_id: str,
        max_files: int,
        max_bytes: int,
        data_suffix: str,
        index_suffix: str,
    ) -> None:
        self.bundle_dir = bundle_dir
        self.bundle_id = bundle_id
        self.max_files = max_files
        self.max_bytes = max_bytes
        self.data_path = bundle_dir / f"{bundle_id}{data_suffix}"
        self.index_path = bundle_dir / f"{bundle_id}{index_suffix}"
        self._data_fh = self.data_path.open("wb")
        self._index_fh = self.index_path.open("w", encoding="utf-8")
        self.offset = 0
        self.file_count = 0
        self.byte_count = 0

    def can_accept(self, size: int) -> bool:
        if self.file_count == 0:
            return size <= self.max_bytes
        return (self.file_count + 1 <= self.max_files) and (self.byte_count + size <= self.max_bytes)

    def add_file(self, path: Path, rel_path: str, st: os.stat_result) -> None:
        hasher = hashlib.sha256()
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(READ_CHUNK_SIZE)
                if not chunk:
                    break
                self._data_fh.write(chunk)
                hasher.update(chunk)
        entry = {
            "rel_path": rel_path,
            "bundle": self.bundle_id,
            "offset": self.offset,
            "size": st.st_size,
            "sha256": hasher.hexdigest(),
            "mtime_ns": int(st.st_mtime_ns),
            "mode": int(st.st_mode),
        }
        self._index_fh.write(json.dumps(entry, ensure_ascii=True) + "\n")
        self.offset += st.st_size
        self.file_count += 1
        self.byte_count += st.st_size

    def close(self) -> None:
        self._data_fh.close()
        self._index_fh.close()


def _merge_exclude_dirs(extra: Optional[Iterable[str]]) -> set[str]:
    merged = set(DEFAULT_EXCLUDE_DIRS)
    if not extra:
        return merged
    for entry in extra:
        if entry:
            merged.add(str(entry))
    return merged


def plan_bundle(
    root: Path,
    bundle_dir: Optional[Path],
    include_globs: list[str],
    exclude_globs: list[str],
    follow_symlinks: bool,
    max_file_bytes: int,
    max_bundle_bytes: int,
    max_files_per_bundle: int,
    exclude_dirs: Optional[Iterable[str]] = None,
    write_manifest: bool = True,
) -> PlanReport:
    report = _scan_plan(
        root=root,
        bundle_dir=bundle_dir,
        include_globs=include_globs,
        exclude_globs=exclude_globs,
        follow_symlinks=follow_symlinks,
        exclude_dirs=_merge_exclude_dirs(exclude_dirs),
        max_file_bytes=max_file_bytes,
        max_bundle_bytes=max_bundle_bytes,
        max_files_per_bundle=max_files_per_bundle,
        global_overhead_files=1 if write_manifest else 0,
    )
    return report


def apply_bundle(
    root: Path,
    bundle_dir: Path,
    include_globs: list[str],
    exclude_globs: list[str],
    follow_symlinks: bool,
    max_file_bytes: int,
    max_bundle_bytes: int,
    max_files_per_bundle: int,
    exclude_dirs: Optional[Iterable[str]] = None,
    write_manifest: bool = True,
) -> PlanReport:
    report = _bundle_files(
        root=root,
        bundle_dir=bundle_dir,
        include_globs=include_globs,
        exclude_globs=exclude_globs,
        follow_symlinks=follow_symlinks,
        exclude_dirs=_merge_exclude_dirs(exclude_dirs),
        max_file_bytes=max_file_bytes,
        max_bundle_bytes=max_bundle_bytes,
        max_files_per_bundle=max_files_per_bundle,
        data_suffix=DEFAULT_DATA_SUFFIX,
        index_suffix=DEFAULT_INDEX_SUFFIX,
        global_overhead_files=1 if write_manifest else 0,
    )
    if write_manifest and report.bundles > 0:
        _write_manifest(
            bundle_dir=bundle_dir,
            root=root,
            report=report,
            max_file_bytes=max_file_bytes,
            max_bundle_bytes=max_bundle_bytes,
            max_files_per_bundle=max_files_per_bundle,
            data_suffix=DEFAULT_DATA_SUFFIX,
            index_suffix=DEFAULT_INDEX_SUFFIX,
        )
    return report


def _load_current_child(config_path: Path) -> Optional[str]:
    try:
        with config_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    child = data.get("current_child")
    if not child:
        return None
    return str(child)


def _resolve_root(root_arg: Optional[str], child_arg: Optional[str]) -> Path:
    if root_arg:
        return Path(root_arg)
    if child_arg:
        return Path("AI_Children") / child_arg / "memory"
    config_path = Path("config.json")
    if config_path.exists():
        child = _load_current_child(config_path)
        if child:
            return Path("AI_Children") / child / "memory"
    return Path(".")


def _normalize_globs(values: Optional[Iterable[str]]) -> list[str]:
    if not values:
        return []
    return [value for value in values if value]


def _matches_globs(rel_path: str, globs: list[str]) -> bool:
    return any(fnmatch.fnmatch(rel_path, pattern) for pattern in globs)


def _is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.relative_to(other)
    except ValueError:
        return False
    return True


def _iter_files(
    root: Path,
    bundle_dir: Optional[Path],
    include_globs: list[str],
    exclude_globs: list[str],
    follow_symlinks: bool,
    exclude_dirs: set[str],
) -> Iterable[tuple[Path, str, Optional[os.stat_result], bool]]:
    root = root.resolve()
    bundle_dir_resolved = bundle_dir.resolve() if bundle_dir else None
    bundle_dir_rel: Optional[str] = None
    if bundle_dir_resolved and _is_relative_to(bundle_dir_resolved, root):
        bundle_dir_rel = bundle_dir_resolved.relative_to(root).as_posix()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        dirpath_path = Path(dirpath)
        if bundle_dir_resolved and _is_relative_to(dirpath_path.resolve(), bundle_dir_resolved):
            dirnames[:] = []
            continue
        pruned_dirs = [d for d in dirnames if d not in exclude_dirs]
        if exclude_globs:
            kept = []
            for d in pruned_dirs:
                rel_dir = (dirpath_path / d).relative_to(root).as_posix()
                if _matches_globs(rel_dir, exclude_globs):
                    continue
                kept.append(d)
            dirnames[:] = kept
        else:
            dirnames[:] = pruned_dirs
        for name in filenames:
            path = dirpath_path / name
            rel_path = path.relative_to(root).as_posix()
            if bundle_dir_rel and rel_path.startswith(f"{bundle_dir_rel}/"):
                continue
            if exclude_globs and _matches_globs(rel_path, exclude_globs):
                yield path, rel_path, None, True
                continue
            if include_globs and not _matches_globs(rel_path, include_globs):
                yield path, rel_path, None, True
                continue
            try:
                st = path.stat()
            except OSError:
                yield path, rel_path, None, True
                continue
            if not stat.S_ISREG(st.st_mode):
                yield path, rel_path, None, True
                continue
            yield path, rel_path, st, False


def _scan_plan(
    root: Path,
    bundle_dir: Optional[Path],
    include_globs: list[str],
    exclude_globs: list[str],
    follow_symlinks: bool,
    exclude_dirs: set[str],
    max_file_bytes: int,
    max_bundle_bytes: int,
    max_files_per_bundle: int,
    global_overhead_files: int,
) -> PlanReport:
    report = PlanReport(global_overhead_files=global_overhead_files)
    planner = BundlePlanner(max_files=max_files_per_bundle, max_bytes=max_bundle_bytes)
    for path, rel_path, st, skipped in _iter_files(
        root,
        bundle_dir,
        include_globs,
        exclude_globs,
        follow_symlinks,
        exclude_dirs,
    ):
        report.total_files += 1
        if skipped or st is None:
            report.skipped_files += 1
            continue
        if st.st_size > max_file_bytes or st.st_size > max_bundle_bytes:
            report.ineligible_files += 1
            report.ineligible_bytes += st.st_size
            continue
        report.eligible_files += 1
        report.eligible_bytes += st.st_size
        planner.add(st.st_size)
    report.bundles = planner.bundles
    return report


def _bundle_files(
    root: Path,
    bundle_dir: Path,
    include_globs: list[str],
    exclude_globs: list[str],
    follow_symlinks: bool,
    exclude_dirs: set[str],
    max_file_bytes: int,
    max_bundle_bytes: int,
    max_files_per_bundle: int,
    data_suffix: str,
    index_suffix: str,
    global_overhead_files: int,
) -> PlanReport:
    report = PlanReport(global_overhead_files=global_overhead_files)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle_index = 0
    writer: Optional[BundleWriter] = None

    def open_writer() -> BundleWriter:
        nonlocal bundle_index
        bundle_index += 1
        bundle_id = f"{DEFAULT_BUNDLE_PREFIX}{bundle_index:06d}"
        return BundleWriter(
            bundle_dir=bundle_dir,
            bundle_id=bundle_id,
            max_files=max_files_per_bundle,
            max_bytes=max_bundle_bytes,
            data_suffix=data_suffix,
            index_suffix=index_suffix,
        )

    for path, rel_path, st, skipped in _iter_files(
        root,
        bundle_dir,
        include_globs,
        exclude_globs,
        follow_symlinks,
        exclude_dirs,
    ):
        report.total_files += 1
        if skipped or st is None:
            report.skipped_files += 1
            continue
        if st.st_size > max_file_bytes or st.st_size > max_bundle_bytes:
            report.ineligible_files += 1
            report.ineligible_bytes += st.st_size
            continue
        if writer is None or not writer.can_accept(st.st_size):
            if writer is not None:
                writer.close()
            writer = open_writer()
            report.bundles += 1
        writer.add_file(path, rel_path, st)
        report.eligible_files += 1
        report.eligible_bytes += st.st_size
    if writer is not None:
        writer.close()
    return report


def _write_manifest(
    bundle_dir: Path,
    root: Path,
    report: PlanReport,
    max_file_bytes: int,
    max_bundle_bytes: int,
    max_files_per_bundle: int,
    data_suffix: str,
    index_suffix: str,
) -> None:
    manifest = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "bundle_dir": str(bundle_dir),
        "max_file_bytes": max_file_bytes,
        "max_bundle_bytes": max_bundle_bytes,
        "max_files_per_bundle": max_files_per_bundle,
        "data_suffix": data_suffix,
        "index_suffix": index_suffix,
        "report": report.to_dict(),
    }
    path = bundle_dir / DEFAULT_MANIFEST_NAME
    with path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=True, indent=2)
        fh.write("\n")


def _print_report(report: PlanReport, label: str, report_json: bool) -> None:
    if report_json:
        payload = report.to_dict()
        payload["label"] = label
        json.dump(payload, sys.stdout, ensure_ascii=True, indent=2)
        sys.stdout.write("\n")
        return
    print(f"[Bundler] {label}")
    print(f"[Bundler] Files scanned: {report.total_files}")
    print(f"[Bundler] Eligible files: {report.eligible_files} ({report.eligible_bytes} bytes)")
    print(f"[Bundler] Ineligible files: {report.ineligible_files} ({report.ineligible_bytes} bytes)")
    print(f"[Bundler] Skipped files: {report.skipped_files}")
    print(f"[Bundler] Bundles: {report.bundles}")
    print(
        "[Bundler] If applied: files go from "
        f"{report.total_files} -> {report.new_file_count}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bundle small files into chunk files plus an index."
    )
    parser.add_argument("--root", help="Root directory to scan (default: current child memory or cwd).")
    parser.add_argument("--child", help="Child name (uses AI_Children/<child>/memory).")
    parser.add_argument("--bundle-dir", help="Directory for bundles (default: <root>/bundles).")
    parser.add_argument("--max-files-per-bundle", type=int, default=DEFAULT_MAX_FILES_PER_BUNDLE)
    parser.add_argument("--max-bundle-bytes", type=int, default=DEFAULT_MAX_BUNDLE_BYTES)
    parser.add_argument("--max-file-bytes", type=int, default=DEFAULT_MAX_FILE_BYTES)
    parser.add_argument("--include", action="append", default=[], help="Glob to include (repeatable).")
    parser.add_argument("--exclude", action="append", default=[], help="Glob to exclude (repeatable).")
    parser.add_argument("--follow-symlinks", action="store_true", help="Follow symlinks during scan.")
    parser.add_argument("--apply", action="store_true", help="Write bundle files (no deletion).")
    parser.add_argument("--report-json", action="store_true", help="Emit report as JSON.")
    parser.add_argument("--no-manifest", action="store_true", help="Skip writing bundle_manifest.json.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.max_files_per_bundle < 1:
        print("[Bundler] max-files-per-bundle must be >= 1.", file=sys.stderr)
        return 2
    if args.max_bundle_bytes < 1 or args.max_file_bytes < 1:
        print("[Bundler] max-bundle-bytes and max-file-bytes must be >= 1.", file=sys.stderr)
        return 2
    root = _resolve_root(args.root, args.child)
    bundle_dir = Path(args.bundle_dir) if args.bundle_dir else root / "bundles"
    include_globs = _normalize_globs(args.include)
    exclude_globs = _normalize_globs(args.exclude)
    exclude_dirs = set(DEFAULT_EXCLUDE_DIRS)
    global_overhead_files = 0 if args.no_manifest else 1

    if not root.exists():
        print(f"[Bundler] Root not found: {root}", file=sys.stderr)
        return 2

    if args.apply:
        report = _bundle_files(
            root=root,
            bundle_dir=bundle_dir,
            include_globs=include_globs,
            exclude_globs=exclude_globs,
            follow_symlinks=args.follow_symlinks,
            exclude_dirs=exclude_dirs,
            max_file_bytes=args.max_file_bytes,
            max_bundle_bytes=args.max_bundle_bytes,
            max_files_per_bundle=args.max_files_per_bundle,
            data_suffix=DEFAULT_DATA_SUFFIX,
            index_suffix=DEFAULT_INDEX_SUFFIX,
            global_overhead_files=global_overhead_files,
        )
        if not args.no_manifest and report.bundles > 0:
            _write_manifest(
                bundle_dir=bundle_dir,
                root=root,
                report=report,
                max_file_bytes=args.max_file_bytes,
                max_bundle_bytes=args.max_bundle_bytes,
                max_files_per_bundle=args.max_files_per_bundle,
                data_suffix=DEFAULT_DATA_SUFFIX,
                index_suffix=DEFAULT_INDEX_SUFFIX,
            )
        _print_report(report, "Apply complete (no deletion).", args.report_json)
        return 0

    report = _scan_plan(
        root=root,
        bundle_dir=bundle_dir,
        include_globs=include_globs,
        exclude_globs=exclude_globs,
        follow_symlinks=args.follow_symlinks,
        exclude_dirs=exclude_dirs,
        max_file_bytes=args.max_file_bytes,
        max_bundle_bytes=args.max_bundle_bytes,
        max_files_per_bundle=args.max_files_per_bundle,
        global_overhead_files=global_overhead_files,
    )
    _print_report(report, "Dry run (no files written).", args.report_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
