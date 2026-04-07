import importlib

_EXPORTS = {
    "DriftConfig": ("transformers.soul_drift", "DriftConfig"),
    "DriftState": ("transformers.soul_drift", "DriftState"),
    "SoulDriftTransformer": ("transformers.soul_drift", "SoulDriftTransformer"),
    "MycelialTransformer": ("transformers.mycelial_transformer", "MycelialTransformer"),
    "HeuristicMirrorTransformer": ("transformers.heuristic_mirror_transformer", "HeuristicMirrorTransformer"),
    "FractalTransformer": ("transformers.fractal_multidimensional_transformers", "FractalTransformer"),
    "load_precision_profile": ("transformers.fractal_multidimensional_transformers", "load_precision_profile"),
    "SeedlingTransformer": ("transformers.seedling_transformer", "SeedlingTransformer"),
    "QTransformer": ("transformers.QTransformer", "QTransformer"),
    "HindsightTransformer": ("transformers.hindsight_transformer", "HindsightTransformer"),
    "ShadowTransformer": ("transformers.shadow_transformer", "ShadowTransformer"),
    "BridgeTransformer": ("transformers.bridge_transformer", "BridgeTransformer"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    try:
        value = getattr(importlib.import_module(module_name), attr_name)
    except Exception:
        value = None
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
