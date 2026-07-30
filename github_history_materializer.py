"""Materialize bounded local commit summaries for the self-read pipeline."""

from pathlib import Path
from typing import List

from github_history_bridge import commit_as_text, read_commit_history


def materialize_commit_history(
    repo_root: Path, output_root: Path, limit: int = 24
) -> List[Path]:
    """Write one stable text summary per commit; never include source diffs."""
    output_root.mkdir(parents=True, exist_ok=True)
    batch_limit = max(1, min(int(limit), 100))
    existing_hashes = {path.stem for path in output_root.glob("*.txt")}
    query_limit = min(1000, len(existing_hashes) + batch_limit)
    written = []
    for commit in read_commit_history(repo_root, limit=query_limit):
        if commit["hash"] in existing_hashes:
            continue
        if len(written) >= batch_limit:
            break
        path = output_root / f"{commit['hash']}.txt"
        if not path.exists():
            path.write_text(commit_as_text(commit), encoding="utf-8")
        written.append(path)
    return written
