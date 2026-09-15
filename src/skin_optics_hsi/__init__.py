"""Real-HSI-constrained skin optics research package.

The package is intentionally separate from the frozen SO-0/SO-1 synthetic
pipeline.  Parameters produced here are model-constrained optical proxies, not
direct physiological measurements.
"""

from .config import Stage1Config, load_stage1_config
from .model_registry import Stage1ModelRegistry, load_model_registry
from .skin_forward import RegisteredSkinForwardModel, SkinForwardModel, skin_forward

__all__ = [
    "RegisteredSkinForwardModel",
    "SkinForwardModel",
    "Stage1Config",
    "Stage1ModelRegistry",
    "load_model_registry",
    "load_stage1_config",
    "skin_forward",
]
