import cv2
import numpy as np

from image_triangle_counting import count_image_regions, detect_triangles


def _triangle_image():
    image = np.zeros((180, 360, 3), dtype=np.uint8)
    cv2.fillPoly(image, [np.array([[30, 130], [85, 30], [140, 130]], np.int32)], (255, 80, 20))
    cv2.polylines(image, [np.array([[210, 130], [265, 30], [320, 130]], np.int32)], True, (150, 150, 150), 4)
    return image


def test_detects_individual_filled_and_empty_triangles():
    observations = detect_triangles(_triangle_image(), "synthetic", min_area=500)
    assert len(observations) == 2
    assert sorted(item.state for item in observations) == ["blue", "empty"]


def test_region_results_are_experiential_counts(tmp_path):
    path = tmp_path / "triangles.png"
    assert cv2.imwrite(str(path), _triangle_image())
    report = count_image_regions(path, [
        {"name": "left", "box": [0.0, 0.0, 0.5, 1.0], "min_area": 500},
        {"name": "right", "box": [0.5, 0.0, 1.0, 1.0], "min_area": 500},
    ])
    assert report["ocr_used"] is False
    assert [row["count"]["value"] for row in report["regions"]] == [1, 1]
    assert all(row["count"]["counted_by_observation"] for row in report["regions"])
    assert report["regions"][0]["count"]["groups"] == {"blue": 1}
    assert report["regions"][1]["count"]["groups"] == {"empty": 1}
