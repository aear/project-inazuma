# Image Triangle Counting Benchmark V1

This benchmark asks Ina to count visible triangular faces in independently
named image regions. It measures two things for each region:

- the total number of geometric triangle slots;
- the number whose interiors are observably red, blue, empty, or another colour.

Captions and OCR are excluded from the detection path. Expected totals exist
only in the evaluator manifest; `image_triangle_counting.py` never receives
them. Each accepted polygon is passed individually to `ExperientialCounter`,
so the reported cardinality is produced by observation rather than copied from
a label or inferred from a known layout.

## Running the supplied case

Save the original, unscaled Player Status image as
`benchmarks/fixtures/image_triangle_counting/player_status_v1.png`, then run:

```bash
python -m benchmarks.benchmark_image_triangle_counting \
  benchmarks/image_triangle_counting_player_status_v1.json \
  benchmarks/fixtures/image_triangle_counting/player_status_v1.png \
  --output /tmp/player_status_triangle_report.json
```

Exit status `0` means every asserted measurement in every region matched.
Exit status `1` means at least one did not. The JSON report retains polygon
vertices, centroids, areas, colour fractions, expected values, and actual
values so a failure can be inspected rather than reduced to one score.

The attached chat image is not committed automatically. This avoids silently
reconstructing, recompressing, or redistributing an image without its original
bytes and provenance.

## Adding another public case

Create another manifest with schema `ina.benchmark.image_triangle_counting/V1`.
Each region requires a unique `name`, a `box`, and an `expected` mapping. Boxes
may be pixel coordinates or normalized `[x0, y0, x1, y1]` coordinates. Expected
keys may include `total`, `red`, `blue`, `empty`, and `other`.

Do not tune a region's detector settings from its expected answer. If unusual
scale requires `min_area` or `max_area_fraction`, derive those settings from
visible geometry and validate them on held-out images.

## Pass interpretation

A V1 pass establishes this bounded image and detector configuration only. It
does not establish unrestricted object counting, occlusion handling, arbitrary
triangle styles, or video counting. A stronger version should add rotated,
scaled, noisy, partially occluded, low-contrast, and adversarial non-triangle
cases while retaining this V1 result for comparison.

The first retained result is
`results/image_triangle_counting_player_status_v1.json`. It is a failing 3/7
baseline. In particular, the artwork contains 18 outer triangle faces rather
than the 12 claimed by its caption, and several damage captions likewise do
not match the illuminated faces. The benchmark follows observable geometry;
printed totals are not used as answers.
