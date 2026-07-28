from pathlib import Path

import visual_token_learning as vtl
from simple_image_fallback import extract_image_grid


def _write_forms(path: Path, *, scale=1, invert=False, offset=0):
    width, height = 80 * scale, 30 * scale
    background, foreground = (15, 245) if invert else (245, 15)
    pixels = [background] * (width * height)
    boxes = [
        (8 + offset, 7, 13 + offset, 23),
        (22 + offset, 7, 27 + offset, 23),
        (45 + offset, 7, 50 + offset, 23),
    ]
    for x0, y0, x1, y1 in boxes:
        for y in range(y0 * scale, y1 * scale):
            for x in range(x0 * scale, x1 * scale):
                pixels[y * width + x] = foreground
    path.write_bytes(f"P5\n{width} {height}\n255\n".encode() + bytes(pixels))


def test_spatial_grid_is_bounded_and_preserves_shape(tmp_path):
    path = tmp_path / "large.pgm"
    _write_forms(path, scale=3)

    grid = extract_image_grid(path, max_width=64, max_height=64)

    assert grid["source_width"] == 240
    assert grid["source_height"] == 90
    assert grid["width"] <= 64
    assert grid["height"] <= 64
    assert len(grid["pixels"]) == grid["width"] * grid["height"]
    assert min(grid["pixels"]) < max(grid["pixels"])


def test_visual_forms_recur_across_scale_position_and_polarity(tmp_path):
    paths = [tmp_path / "base.pgm", tmp_path / "scaled.pgm", tmp_path / "inverse.pgm"]
    _write_forms(paths[0])
    _write_forms(paths[1], scale=2)
    _write_forms(paths[2], invert=True, offset=4)

    results = [
        vtl.observe_image(path, child="TestChild", event_id=f"evt_{index}", base_path=tmp_path)
        for index, path in enumerate(paths)
    ]

    common = set(results[0]["candidate_ids"])
    for result in results[1:]:
        common.intersection_update(result["candidate_ids"])
    assert common
    assert all(result["component_count"] >= 3 for result in results)


def test_word_hypothesis_needs_recurrence_and_weakens_on_contradiction(tmp_path):
    path = tmp_path / "forms.pgm"
    _write_forms(path)
    observations = []

    for index in range(1, 4):
        event_id = f"evt_{index}"
        observed = vtl.observe_image(
            path, child="TestChild", event_id=event_id, base_path=tmp_path
        )
        learned = vtl.observe_words(
            [event_id], ["zabble"], child="TestChild", base_path=tmp_path
        )
        observations.append((observed, learned))

    hypotheses = vtl.hypotheses_for_tokens(
        observations[-1][0]["candidate_ids"], child="TestChild", base_path=tmp_path
    )
    repeated = [row for row in hypotheses if row["word"] == "zabble"]
    assert repeated
    assert all(row["support"] == 3 for row in repeated)
    confidence_before = max(row["confidence"] for row in repeated)
    assert confidence_before >= 0.55

    observed = vtl.observe_image(
        path, child="TestChild", event_id="evt_4", base_path=tmp_path
    )
    vtl.observe_words(
        ["evt_4"], ["different"], child="TestChild", base_path=tmp_path
    )
    hypotheses = vtl.hypotheses_for_tokens(
        observed["candidate_ids"], child="TestChild", base_path=tmp_path
    )
    confidence_after = max(
        row["confidence"] for row in hypotheses if row["word"] == "zabble"
    )
    assert confidence_after < confidence_before


def test_single_exposure_cannot_label_visual_token(tmp_path):
    path = tmp_path / "forms.pgm"
    _write_forms(path)
    observed = vtl.observe_image(
        path, child="TestChild", event_id="evt_once", base_path=tmp_path
    )
    learned = vtl.observe_words(
        ["evt_once"], ["premature"], child="TestChild", base_path=tmp_path
    )

    assert learned["hypotheses"] == []
    assert vtl.hypotheses_for_tokens(
        observed["candidate_ids"], child="TestChild", base_path=tmp_path
    ) == []
