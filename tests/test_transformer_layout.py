from pathlib import Path


def test_transformer_implementations_live_only_in_transformers_package():
    root = Path(__file__).resolve().parents[1]
    package = root / "transformers"
    assert package.is_dir()
    assert not (root / "fractal_multidimensional_transformers.py").exists()
    assert (package / "fractal_multidimensional_transformers.py").exists()
