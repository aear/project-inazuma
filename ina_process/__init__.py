"""Process metrics with a stdlib Linux fallback when psutil is unavailable."""
from __future__ import annotations

try:
    import psutil as psutil  # type: ignore
    USING_FALLBACK = False
except (ImportError, ModuleNotFoundError):
    from . import fallback as psutil
    USING_FALLBACK = True

__all__ = ["USING_FALLBACK", "psutil"]
