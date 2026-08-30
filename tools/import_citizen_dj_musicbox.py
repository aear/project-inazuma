#!/usr/bin/env python3
"""Import the verified Citizen DJ MusicBox excerpt pack with provenance."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from public_music_library import MANIFEST_NAME, SCHEMA, sha256_file


SOURCE_URL = "https://citizen-dj.labs.loc.gov/loc-musicbox/use/"
LICENSE_URL = SOURCE_URL + "#rights--access"
COLLECTION_CREDIT = (
    "Citizen DJ Project, Dyann Arthur and Rick Arthur collection of MusicBox "
    "Project materials (AFC 2010/029), Library of Congress."
)


def _safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts


def import_pack(archive: Path, destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    tracks = {}
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        if not all(_safe_member(name) for name in names):
            raise ValueError("archive contains an unsafe path")
        rows = csv.DictReader(io.TextIOWrapper(bundle.open("excerpts.csv"), encoding="utf-8-sig"))
        for row in rows:
            filename = str(row.get("clipFilename") or "").strip()
            member = f"excerpts/{filename}"
            if not filename or member not in names:
                raise ValueError(f"missing excerpt {member!r}")
            relative = f"musicbox_project/{member}"
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, target.open("wb") as output:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(chunk)
            item_id = str(row.get("itemId") or "").strip()
            attribution_name = next(
                (name for name in names if name.startswith("attributions/") and name.endswith(f"_{item_id}.txt")),
                None,
            )
            attribution = bundle.read(attribution_name).decode("utf-8", "replace") if attribution_name else ""
            contributor_match = re.search(r"^Contributors:\s*(.+)$", attribution, re.MULTILINE)
            creator = contributor_match.group(1).strip() if contributor_match else "MusicBox Project contributor"
            tracks[relative] = {
                "title": str(row.get("itemTitle") or Path(filename).stem).strip(),
                "creator": creator,
                "source_url": str(row.get("itemUrl") or SOURCE_URL).strip(),
                "collection_url": SOURCE_URL,
                "license_id": "LOC-MUSICBOX-FREE-TO-USE",
                "license_url": LICENSE_URL,
                "recording_rights": "Copyright owners relinquished ownership; performer releases held by the Library of Congress.",
                "composition_rights": "Citizen DJ includes performances of songs in the public domain due to copyright expiration.",
                "suggested_credit": COLLECTION_CREDIT,
                "sha256": sha256_file(target),
            }
        readme_target = destination / "musicbox_project" / "README.txt"
        readme_target.parent.mkdir(parents=True, exist_ok=True)
        readme_target.write_bytes(bundle.read("README.txt"))
    manifest = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_archive": archive.name,
        "source_archive_sha256": sha256_file(archive),
        "collection": "Library of Congress Citizen DJ MusicBox Project",
        "tracks": tracks,
    }
    (destination / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {"tracks": len(tracks), "manifest": str(destination / MANIFEST_NAME)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(import_pack(args.archive, args.destination), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
