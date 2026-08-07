"""B1 gate that prevents pilot execution without a passed B0 audit."""
from __future__ import annotations
from .config import P0BConfig
from .environment_audit import audit_environment
def require_b0_pass(config:P0BConfig)->None:
    """Raise a traceable failure instead of attempting DECA inference prematurely."""
    audit=audit_environment(config)
    if not audit['passed']: raise RuntimeError(f"failed_environment: {audit['block_reason']}")
