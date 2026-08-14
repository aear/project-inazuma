import hashlib

from ina_ml import (
    cosine_similarity, deterministic_hash_bucket, mean_center, normalize_distribution,
    normalize_vector, numeric_summary, shannon_entropy,
)


def test_hash_bucket_preserves_existing_sha256_mapping():
    for text in ("ina", "trust", "Ω∆"):
        expected = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) % 64
        assert deterministic_hash_bucket(text, 64) == expected


def test_shared_vector_kernels_are_dependency_free_and_stable():
    assert normalize_vector([3, 4]) == [0.6, 0.8]
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) > 0.999999
    assert numeric_summary([1, 2, 3]) == [2.0, (2 / 3) ** 0.5, 1.0, 3.0, 2.0]
    assert normalize_distribution([2, 3]) == [0.4, 0.6]
    assert normalize_distribution([0, 0]) == [0.5, 0.5]
    assert mean_center([1, 3]) == [-1.0, 1.0]
    assert shannon_entropy([0.5, 0.5]) > 0.69


def test_rgb_frame_validates_and_downsamples_without_numpy():
    from ina_ml import RGBFrame

    frame = RGBFrame(2, 2, bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255]))
    assert frame.shape == (2, 2, 3)
    reduced = frame.downsample(2)
    assert reduced.shape == (1, 1, 3)
    assert reduced.tobytes() == bytes([255, 0, 0])
    assert reduced.ppm() == b"P6\n1 1\n255\n\xff\x00\x00"
