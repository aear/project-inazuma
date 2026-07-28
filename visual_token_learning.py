"""Bounded, label-free visual token discovery and cross-modal learning.

This module does not perform OCR and contains no alphabet or pretrained text
labels.  It gives Ina tools to propose recurring visual forms, normalize them,
cluster repeated observations, and accumulate tentative associations with words
heard or typed during linked experience events.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from io_utils import atomic_write_json, file_lock
from simple_image_fallback import ImageFallbackError, extract_image_grid
from vector_math import cosine_similarity

GRID_LIMIT = 256
TOKEN_SIZE = 16
MAX_COMPONENTS = 128
MAX_PROPOSALS = 48
MAX_CLUSTERS = 512
MAX_EVENT_INDEX = 384
MAX_CLUSTER_EVENTS = 64
MAX_WORD_EVIDENCE = 48
MAX_CORPUS_WORDS = 4096
CLUSTER_THRESHOLD = 0.88
HYPOTHESIS_MIN_SUPPORT = 2
_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _registry_path(child: str, base_path: Optional[Path] = None) -> Path:
    root = Path(base_path) if base_path else Path("AI_Children")
    return root / child / "memory" / "vision" / "visual_token_registry.json"


def _empty_registry() -> Dict[str, Any]:
    return {
        "version": 1,
        "clusters": {},
        "event_index": {},
        "corpus": {"labeled_events": 0, "word_events": {}},
        "updated_at": _now_iso(),
    }


def _load_registry(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return _empty_registry()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _empty_registry()
    if not isinstance(payload, dict):
        return _empty_registry()
    payload.setdefault("version", 1)
    payload.setdefault("clusters", {})
    payload.setdefault("event_index", {})
    payload.setdefault("corpus", {"labeled_events": 0, "word_events": {}})
    return payload


def _bounded_grid(path: Path) -> Dict[str, Any]:
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as image:
            gray = image.convert("L")
            source_width, source_height = gray.size
            gray.thumbnail((GRID_LIMIT, GRID_LIMIT))
            width, height = gray.size
            pixels = [int(value) for value in gray.getdata()]
            return {
                "decoder": "pillow",
                "format": str(getattr(image, "format", "") or "").lower(),
                "source_width": int(source_width),
                "source_height": int(source_height),
                "width": int(width),
                "height": int(height),
                "pixels": pixels,
            }
    except Exception:
        return extract_image_grid(path, max_width=GRID_LIMIT, max_height=GRID_LIMIT)


def _otsu_threshold(pixels: Sequence[int]) -> int:
    histogram = [0] * 256
    for value in pixels:
        histogram[max(0, min(255, int(value)))] += 1
    total = len(pixels)
    if total <= 0:
        return 127
    weighted_total = sum(index * count for index, count in enumerate(histogram))
    background_weight = 0
    background_sum = 0.0
    best_variance = -1.0
    best_threshold = 127
    for threshold, count in enumerate(histogram):
        background_weight += count
        if background_weight == 0:
            continue
        foreground_weight = total - background_weight
        if foreground_weight == 0:
            break
        background_sum += threshold * count
        background_mean = background_sum / background_weight
        foreground_mean = (weighted_total - background_sum) / foreground_weight
        variance = background_weight * foreground_weight * (background_mean - foreground_mean) ** 2
        if variance > best_variance:
            best_variance = variance
            best_threshold = threshold
    return best_threshold


def _foreground_mask(pixels: Sequence[int], width: int, height: int) -> List[bool]:
    threshold = _otsu_threshold(pixels)
    border = []
    for x in range(width):
        border.append(pixels[x])
        if height > 1:
            border.append(pixels[(height - 1) * width + x])
    for y in range(1, max(1, height - 1)):
        border.append(pixels[y * width])
        if width > 1:
            border.append(pixels[y * width + width - 1])
    border_mean = sum(border) / max(1, len(border))
    dark_foreground = border_mean >= threshold
    if dark_foreground:
        mask = [value <= threshold for value in pixels]
    else:
        mask = [value > threshold for value in pixels]

    foreground_count = sum(mask)
    if foreground_count > len(mask) * 0.65:
        mask = [not value for value in mask]
    return mask


def _components(mask: Sequence[bool], width: int, height: int) -> List[Dict[str, Any]]:
    visited = bytearray(len(mask))
    found: List[Dict[str, Any]] = []
    for start, active in enumerate(mask):
        if not active or visited[start]:
            continue
        stack = [start]
        visited[start] = 1
        points: List[Tuple[int, int]] = []
        min_x = max_x = start % width
        min_y = max_y = start // width
        while stack:
            index = stack.pop()
            x = index % width
            y = index // width
            points.append((x, y))
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            for next_y in range(max(0, y - 1), min(height, y + 2)):
                offset = next_y * width
                for next_x in range(max(0, x - 1), min(width, x + 2)):
                    neighbour = offset + next_x
                    if mask[neighbour] and not visited[neighbour]:
                        visited[neighbour] = 1
                        stack.append(neighbour)
        box_width = max_x - min_x + 1
        box_height = max_y - min_y + 1
        area = len(points)
        if area < 2 or box_width < 1 or box_height < 2:
            continue
        if area > len(mask) * 0.45:
            continue
        aspect = box_width / float(box_height)
        if aspect < 0.06 or aspect > 14.0:
            continue
        found.append({
            "bbox": [min_x, min_y, max_x, max_y],
            "points": points,
            "area": area,
            "aspect": round(aspect, 4),
            "kind": "component",
        })
    found.sort(key=lambda item: (item["bbox"][1], item["bbox"][0], -item["area"]))
    return found[:MAX_COMPONENTS]


def _group_components(components: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    lines: List[List[Dict[str, Any]]] = []
    for component in components:
        x0, y0, x1, y1 = component["bbox"]
        center = (y0 + y1) / 2.0
        height = y1 - y0 + 1
        target = None
        for line in lines:
            centers = [(item["bbox"][1] + item["bbox"][3]) / 2.0 for item in line]
            heights = [item["bbox"][3] - item["bbox"][1] + 1 for item in line]
            if abs(center - sum(centers) / len(centers)) <= max(height, sum(heights) / len(heights)) * 0.55:
                target = line
                break
        if target is None:
            target = []
            lines.append(target)
        target.append(component)

    groups: List[Dict[str, Any]] = []
    for line in lines:
        ordered = sorted(line, key=lambda item: item["bbox"][0])
        current: List[Dict[str, Any]] = []
        for component in ordered:
            if current:
                previous = current[-1]
                gap = component["bbox"][0] - previous["bbox"][2] - 1
                heights = [item["bbox"][3] - item["bbox"][1] + 1 for item in current]
                typical_height = sum(heights) / len(heights)
                if gap > max(2.0, typical_height * 0.75):
                    if len(current) >= 2:
                        groups.append(_merge_group(current))
                    current = []
            current.append(component)
        if len(current) >= 2:
            groups.append(_merge_group(current))
    return groups


def _merge_group(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    points = [point for item in items for point in item["points"]]
    min_x = min(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_x = max(point[0] for point in points)
    max_y = max(point[1] for point in points)
    return {
        "bbox": [min_x, min_y, max_x, max_y],
        "points": points,
        "area": len(points),
        "aspect": round((max_x - min_x + 1) / float(max_y - min_y + 1), 4),
        "kind": "group",
        "part_count": len(items),
    }


def _normalize(proposal: Dict[str, Any]) -> List[float]:
    min_x, min_y, max_x, max_y = proposal["bbox"]
    source_width = max_x - min_x + 1
    source_height = max_y - min_y + 1
    buckets = [0.0] * (TOKEN_SIZE * TOKEN_SIZE)
    counts = [0] * (TOKEN_SIZE * TOKEN_SIZE)
    active = {(x, y) for x, y in proposal["points"]}
    for target_y in range(TOKEN_SIZE):
        y0 = min_y + int(target_y * source_height / TOKEN_SIZE)
        y1 = min_y + max(1, int((target_y + 1) * source_height / TOKEN_SIZE))
        y1 = min(max_y + 1, y1)
        for target_x in range(TOKEN_SIZE):
            x0 = min_x + int(target_x * source_width / TOKEN_SIZE)
            x1 = min_x + max(1, int((target_x + 1) * source_width / TOKEN_SIZE))
            x1 = min(max_x + 1, x1)
            index = target_y * TOKEN_SIZE + target_x
            for y in range(y0, y1):
                for x in range(x0, x1):
                    counts[index] += 1
                    if (x, y) in active:
                        buckets[index] += 1.0
    return [round(value / max(1, counts[index]), 4) for index, value in enumerate(buckets)]


def _cluster_id(vector: Sequence[float], kind: str) -> str:
    packed = bytes(max(0, min(255, round(value * 255))) for value in vector)
    digest = hashlib.sha256(kind.encode("utf-8") + packed).hexdigest()[:12]
    return f"vtoken_{digest}"


def _best_cluster(clusters: Dict[str, Any], vector: Sequence[float], kind: str, aspect: float):
    best_id = None
    best_similarity = 0.0
    for cluster_id, cluster in clusters.items():
        if cluster.get("kind") != kind:
            continue
        cluster_aspect = float(cluster.get("aspect", aspect) or aspect)
        ratio = max(aspect, cluster_aspect) / max(0.01, min(aspect, cluster_aspect))
        if ratio > 1.8:
            continue
        centroid = cluster.get("centroid") or []
        if not centroid:
            continue
        similarity = cosine_similarity(vector, centroid)
        if similarity > best_similarity:
            best_id = cluster_id
            best_similarity = similarity
    return best_id, best_similarity


def _cluster_hypotheses(cluster: Dict[str, Any], corpus: Dict[str, Any]) -> List[Dict[str, Any]]:
    evidence = cluster.get("word_evidence") or {}
    labeled_events = max(1, int(corpus.get("labeled_events", 0)))
    cluster_labeled = max(1, int(cluster.get("labeled_events", 0)))
    global_counts = corpus.get("word_events") or {}
    hypotheses = []
    for word, entry in evidence.items():
        support = len(entry.get("events") or [])
        if support < HYPOTHESIS_MIN_SUPPORT:
            continue
        coverage = support / cluster_labeled
        global_support = max(1, int(global_counts.get(word, 1)))
        specificity = math.log((labeled_events + 1) / global_support) + 1.0
        specificity /= math.log(labeled_events + 1) + 1.0
        repetition = min(1.0, support / 4.0)
        confidence = max(0.0, min(0.99, coverage * (0.5 + 0.5 * specificity) * repetition))
        hypotheses.append({
            "word": word,
            "support": support,
            "confidence": round(confidence, 4),
            "last_seen": entry.get("last_seen"),
        })
    hypotheses.sort(key=lambda item: (-item["confidence"], -item["support"], item["word"]))
    return hypotheses[:8]


def observe_image(
    image_path: Path,
    *,
    child: str,
    event_id: Optional[str] = None,
    base_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Propose and cluster anonymous visual forms from one image event."""

    try:
        grid = _bounded_grid(Path(image_path))
    except ImageFallbackError as exc:
        return {"status": "unsupported", "reason": str(exc), "candidate_ids": []}
    pixels = grid.get("pixels") or []
    width = int(grid.get("width") or 0)
    height = int(grid.get("height") or 0)
    if not pixels or width <= 0 or height <= 0:
        return {"status": "empty", "candidate_ids": []}

    mask = _foreground_mask(pixels, width, height)
    components = _components(mask, width, height)
    groups = _group_components(components)
    # Preserve word-like groups when a busy image reaches the proposal cap.
    proposals = [*groups, *components][:MAX_PROPOSALS]
    normalized = [(proposal, _normalize(proposal)) for proposal in proposals]
    registry_path = _registry_path(child, base_path)
    lock_path = registry_path.with_suffix(registry_path.suffix + ".lock")
    observed_ids: List[str] = []
    matches: List[Dict[str, Any]] = []

    with file_lock(lock_path):
        registry = _load_registry(registry_path)
        clusters = registry["clusters"]
        for proposal, vector in normalized:
            cluster_id, similarity = _best_cluster(
                clusters, vector, proposal["kind"], float(proposal["aspect"])
            )
            if cluster_id is None or similarity < CLUSTER_THRESHOLD:
                if len(clusters) >= MAX_CLUSTERS:
                    disposable = [
                        (candidate_id, candidate)
                        for candidate_id, candidate in clusters.items()
                        if not candidate.get("hypotheses")
                        and int(candidate.get("observations", 0)) <= 2
                    ]
                    if disposable:
                        disposable.sort(
                            key=lambda item: (
                                int(item[1].get("observations", 0)),
                                item[1].get("last_seen") or item[1].get("created_at") or "",
                            )
                        )
                        clusters.pop(disposable[0][0], None)
                    else:
                        continue
                cluster_id = _cluster_id(vector, proposal["kind"])
                if cluster_id in clusters:
                    suffix = 2
                    while f"{cluster_id}_{suffix}" in clusters:
                        suffix += 1
                    cluster_id = f"{cluster_id}_{suffix}"
                clusters[cluster_id] = {
                    "id": cluster_id,
                    "kind": proposal["kind"],
                    "centroid": vector,
                    "aspect": proposal["aspect"],
                    "observations": 0,
                    "event_count": 0,
                    "events": [],
                    "labeled_events": 0,
                    "word_evidence": {},
                    "hypotheses": [],
                    "created_at": _now_iso(),
                }
                similarity = 1.0
            cluster = clusters[cluster_id]
            count = int(cluster.get("observations", 0))
            centroid = cluster.get("centroid") or vector
            blend_count = min(count, 31)
            cluster["centroid"] = [
                round((float(old) * blend_count + float(new)) / (blend_count + 1), 4)
                for old, new in zip(centroid, vector)
            ]
            cluster["aspect"] = round(
                (float(cluster.get("aspect", proposal["aspect"])) * blend_count + float(proposal["aspect"]))
                / (blend_count + 1),
                4,
            )
            cluster["observations"] = count + 1
            cluster["last_seen"] = _now_iso()
            events = list(cluster.get("events") or [])
            if event_id and event_id not in events:
                events.append(event_id)
                cluster["event_count"] = int(cluster.get("event_count", 0)) + 1
            cluster["events"] = events[-MAX_CLUSTER_EVENTS:]
            cluster["hypotheses"] = _cluster_hypotheses(cluster, registry["corpus"])
            if cluster_id not in observed_ids:
                observed_ids.append(cluster_id)
            matches.append({
                "cluster_id": cluster_id,
                "kind": proposal["kind"],
                "similarity": round(float(similarity), 4),
                "bbox": proposal["bbox"],
                "hypotheses": cluster["hypotheses"][:3],
            })

        if event_id and observed_ids:
            registry["event_index"][event_id] = observed_ids
            while len(registry["event_index"]) > MAX_EVENT_INDEX:
                registry["event_index"].pop(next(iter(registry["event_index"])))
        registry["updated_at"] = _now_iso()
        atomic_write_json(registry_path, registry, indent=2, ensure_ascii=True)

    return {
        "status": "observed",
        "event_id": event_id,
        "candidate_ids": observed_ids,
        "matches": matches,
        "component_count": len(components),
        "proposal_count": len(proposals),
        "grid": {
            "width": width,
            "height": height,
            "source_width": grid.get("source_width"),
            "source_height": grid.get("source_height"),
            "decoder": grid.get("decoder"),
        },
    }


def observe_words(
    event_ids: Iterable[str],
    words: Iterable[str] | str,
    *,
    child: str,
    base_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Accumulate tentative word evidence for visual tokens in linked events."""

    if isinstance(words, str):
        tokens = [token.lower() for token in _WORD_RE.findall(words)]
    else:
        tokens = []
        for value in words:
            tokens.extend(token.lower() for token in _WORD_RE.findall(str(value)))
    tokens = list(dict.fromkeys(token for token in tokens if token))
    linked_events = list(dict.fromkeys(str(event) for event in event_ids if event))
    if not tokens or not linked_events:
        return {"status": "no_evidence", "updated_clusters": [], "hypotheses": []}

    registry_path = _registry_path(child, base_path)
    lock_path = registry_path.with_suffix(registry_path.suffix + ".lock")
    updated = []
    hypothesis_rows = []
    with file_lock(lock_path):
        registry = _load_registry(registry_path)
        event_index = registry["event_index"]
        clusters = registry["clusters"]
        corpus = registry["corpus"]
        # Labeled-event membership is stored separately to keep event_index values simple.
        labeled_events = list(corpus.get("events") or [])
        labeled_index = set(labeled_events)
        for event in linked_events:
            if event not in event_index or event in labeled_index:
                continue
            labeled_index.add(event)
            labeled_events.append(event)
            corpus["labeled_events"] = int(corpus.get("labeled_events", 0)) + 1
            for word in tokens:
                word_events = corpus.setdefault("word_events", {})
                word_events[word] = int(word_events.get(word, 0)) + 1
            for cluster_id in event_index.get(event, []):
                cluster = clusters.get(cluster_id)
                if not cluster:
                    continue
                cluster["labeled_events"] = int(cluster.get("labeled_events", 0)) + 1
                evidence = cluster.setdefault("word_evidence", {})
                for word in tokens:
                    entry = evidence.setdefault(word, {"events": [], "last_seen": None})
                    events = list(entry.get("events") or [])
                    if event not in events:
                        events.append(event)
                    entry["events"] = events[-MAX_CLUSTER_EVENTS:]
                    entry["last_seen"] = _now_iso()
                if len(evidence) > MAX_WORD_EVIDENCE:
                    ranked = sorted(
                        evidence.items(),
                        key=lambda item: (len(item[1].get("events") or []), item[1].get("last_seen") or ""),
                        reverse=True,
                    )[:MAX_WORD_EVIDENCE]
                    cluster["word_evidence"] = dict(ranked)
                cluster["hypotheses"] = _cluster_hypotheses(cluster, corpus)
                updated.append(cluster_id)
                for hypothesis in cluster["hypotheses"][:3]:
                    hypothesis_rows.append({"cluster_id": cluster_id, **hypothesis})
        corpus["events"] = labeled_events[-MAX_EVENT_INDEX:]
        registry["updated_at"] = _now_iso()
        atomic_write_json(registry_path, registry, indent=2, ensure_ascii=True)

    hypothesis_rows.sort(key=lambda item: (-item["confidence"], -item["support"], item["word"]))
    return {
        "status": "learned" if updated else "already_observed",
        "updated_clusters": list(dict.fromkeys(updated)),
        "hypotheses": hypothesis_rows[:16],
    }


def hypotheses_for_tokens(
    token_ids: Iterable[str],
    *,
    child: str,
    base_path: Optional[Path] = None,
    minimum_confidence: float = 0.0,
) -> List[Dict[str, Any]]:
    registry = _load_registry(_registry_path(child, base_path))
    rows = []
    for cluster_id in dict.fromkeys(str(value) for value in token_ids if value):
        cluster = registry.get("clusters", {}).get(cluster_id) or {}
        for hypothesis in cluster.get("hypotheses") or []:
            if float(hypothesis.get("confidence", 0.0)) >= minimum_confidence:
                rows.append({"cluster_id": cluster_id, **hypothesis})
    rows.sort(key=lambda item: (-item["confidence"], -item["support"], item["word"]))
    return rows


__all__ = ["observe_image", "observe_words", "hypotheses_for_tokens"]
