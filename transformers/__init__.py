try:  # pragma: no cover - optional dependency
    from .soul_drift import DriftConfig, DriftState, SoulDriftTransformer
except Exception:
    DriftConfig = DriftState = SoulDriftTransformer = None

try:
    from .mycelial_transformer import MycelialTransformer
except Exception:
    MycelialTransformer = None

try:
    from .heuristic_mirror_transformer import HeuristicMirrorTransformer
except Exception:
    HeuristicMirrorTransformer = None

try:
    from .fractal_multidimensional_transformers import (
        FractalTransformer,
        load_precision_profile,
    )
except Exception:
    FractalTransformer = load_precision_profile = None

try:
    from .seedling_transformer import SeedlingTransformer
except Exception:
    SeedlingTransformer = None

try:  # pragma: no cover - optional dependency
    from .QTransformer import QTransformer
except Exception:  # pragma: no cover
    QTransformer = None

try:
    from .hindsight_transformer import HindsightTransformer
except Exception:
    HindsightTransformer = None

try:
    from .shadow_transformer import ShadowTransformer
except Exception:
    ShadowTransformer = None

try:
    from .bridge_transformer import BridgeTransformer
except Exception:
    BridgeTransformer = None

__all__ = [
    "DriftConfig",
    "DriftState",
    "SoulDriftTransformer",
    "MycelialTransformer",
    "HeuristicMirrorTransformer",
    "FractalTransformer",
    "load_precision_profile",
    "SeedlingTransformer",
    "QTransformer",
    "HindsightTransformer",
    "ShadowTransformer",
    "BridgeTransformer",
]
