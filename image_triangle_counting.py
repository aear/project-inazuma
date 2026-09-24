"""Observable triangle detection feeding Ina's modality-neutral counter.

The detector receives pixels and region boundaries, never expected totals or OCR
captions.  Every accepted polygon becomes one observation for
``ExperientialCounter``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from experiential_counting import ExperientialCounter


SCHEMA = "ina.image_triangle_count/V1"


@dataclass(frozen=True)
class TriangleObservation:
    region: str
    identity: str
    vertices: tuple[tuple[int, int], ...]
    centroid: tuple[float, float]
    area: float
    state: str
    colour_fraction: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "region": self.region, "identity": self.identity,
            "vertices": [list(point) for point in self.vertices],
            "centroid": [round(value, 3) for value in self.centroid],
            "area": round(self.area, 3), "state": self.state,
            "colour_fraction": round(self.colour_fraction, 4),
        }


def _normalised_box(box: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
    if len(box) != 4:
        raise ValueError("region box must contain x0, y0, x1, y1")
    values = [float(value) for value in box]
    if all(0.0 <= value <= 1.0 for value in values):
        x0, y0, x1, y1 = values[0] * width, values[1] * height, values[2] * width, values[3] * height
    else:
        x0, y0, x1, y1 = values
    result = (max(0, int(x0)), max(0, int(y0)), min(width, int(x1)), min(height, int(y1)))
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("region box is empty or outside the image")
    return result


def _state(image: np.ndarray, polygon: np.ndarray) -> tuple[str, float]:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    # Erode away bright outlines: state is determined from the face, not border glow.
    radius = max(1, int(np.sqrt(max(cv2.contourArea(polygon), 1.0)) * 0.07))
    mask = cv2.erode(mask, np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    pixels = hsv[mask > 0]
    if not len(pixels):
        return "empty", 0.0
    # Lit faces have both strong chroma and luminance.  Saturated but dim glow
    # leaking from a neighbouring unit is not itself an active unit.
    saturated = pixels[(pixels[:, 1] >= 100) & (pixels[:, 2] >= 120)]
    fraction = len(saturated) / len(pixels)
    if fraction < 0.18:
        return "empty", fraction
    hues = saturated[:, 0]
    red = int(np.count_nonzero((hues <= 14) | (hues >= 166)))
    blue = int(np.count_nonzero((hues >= 85) & (hues <= 135)))
    if max(red, blue) / max(1, len(pixels)) < 0.16:
        return "other", fraction
    return ("red" if red >= blue else "blue"), max(red, blue) / len(pixels)


def _deduplicate(candidates: list[dict[str, Any]], distance: float) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: item["area"], reverse=True):
        cx, cy = candidate["centroid"]
        if any((cx - old["centroid"][0]) ** 2 + (cy - old["centroid"][1]) ** 2 <= distance ** 2 for old in kept):
            continue
        kept.append(candidate)
    return sorted(kept, key=lambda item: (item["centroid"][1], item["centroid"][0]))


def detect_triangles(image: np.ndarray, region: str, *, min_area: float = 80.0,
                     max_area_fraction: float = 0.20) -> list[TriangleObservation]:
    """Detect discrete triangular faces in one already-cropped scene."""
    if image is None or image.size == 0:
        raise ValueError("image region is empty")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    contour_sets = []
    for low, high in ((25, 90), (45, 140), (70, 180)):
        edges = cv2.Canny(blurred, low, high)
        contours, _hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contour_sets.extend(contours)
    ceiling = image.shape[0] * image.shape[1] * float(max_area_fraction)
    candidates: list[dict[str, Any]] = []
    for contour in contour_sets:
        perimeter = cv2.arcLength(contour, True)
        # Neon bloom, intersections, and partial occlusion can add a few points
        # to an otherwise triangular contour.  Try a bounded approximation
        # ladder and admit the first independently triangular result.
        for epsilon in (0.025, 0.035, 0.045, 0.055, 0.065):
            polygon = cv2.approxPolyDP(contour, epsilon * perimeter, True)
            area = abs(float(cv2.contourArea(polygon)))
            if len(polygon) != 3 or area < min_area or area > ceiling:
                continue
            hull_area = abs(float(cv2.contourArea(cv2.convexHull(polygon))))
            if hull_area <= 0 or area / hull_area < 0.92:
                continue
            moments = cv2.moments(polygon)
            if not moments["m00"]:
                continue
            candidates.append({
                "polygon": polygon,
                "area": area,
                "centroid": (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]),
            })
            break
    typical = np.sqrt(np.median([item["area"] for item in candidates])) if candidates else 4.0
    unique = _deduplicate(candidates, max(3.0, float(typical) * 0.20))
    observations = []
    for index, item in enumerate(unique):
        polygon = item["polygon"]
        state, fraction = _state(image, polygon)
        vertices = tuple((int(point[0][0]), int(point[0][1])) for point in polygon)
        observations.append(TriangleObservation(
            region=region, identity=f"{region}:{index}", vertices=vertices,
            centroid=item["centroid"], area=item["area"], state=state,
            colour_fraction=fraction,
        ))
    return observations


def count_image_regions(image_path: str | Path, regions: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Observe and count triangles in named regions without consulting answers."""
    path = Path(image_path)
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"could not decode image: {path}")
    height, width = image.shape[:2]
    rows = []
    all_observations: list[TriangleObservation] = []
    for spec in regions:
        name = str(spec.get("name") or "").strip()
        if not name:
            raise ValueError("each region requires a name")
        x0, y0, x1, y1 = _normalised_box(spec.get("box") or (), width, height)
        observed = detect_triangles(
            image[y0:y1, x0:x1], name,
            min_area=float(spec.get("min_area", 80.0)),
            max_area_fraction=float(spec.get("max_area_fraction", 0.20)),
        )
        all_observations.extend(observed)
        counter = ExperientialCounter(
            "triangle", rule="one detected triangular face in the named image region",
            mode="unique", identity_of=lambda item: item.identity,
            group_of=lambda item: item.state,
        )
        result = counter.observe_many(observed)
        rows.append({"name": name, "box": [x0, y0, x1, y1], "count": result,
                     "observations": [item.as_dict() for item in observed]})
    return {"schema": SCHEMA, "image": str(path), "image_size": [width, height],
            "regions": rows, "observation_count": len(all_observations),
            "ocr_used": False, "expected_totals_visible_to_detector": False}
