"""P0-A physics-audit asset construction package.

The package deliberately keeps legacy preprocessing immutable and adapts only
its public, low-level geometric and semantic helpers at runtime.
"""

from .config import load_config
from .types import P0Config

__all__ = ["P0Config", "load_config"]
