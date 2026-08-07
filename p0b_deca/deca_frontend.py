"""Late-bound official-DECA adapter; unavailable until real local source is audited."""
from __future__ import annotations
from pathlib import Path
class DecaFrontend:
    """Refuses model use until the installed official DECA interface is inspected."""
    def __init__(self,deca_root:Path,device:str)->None:
        if not deca_root.is_dir(): raise RuntimeError('failed_environment: third_party/DECA is unavailable')
        raise RuntimeError('failed_environment: audit the actual DECA encode/decode/renderer interface before enabling frontend')
