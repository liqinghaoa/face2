"""Permission-respecting local DECA asset inventory."""
from __future__ import annotations
import hashlib
from pathlib import Path
from .types import DecaAssetRecord

def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1<<20),b""): digest.update(chunk)
    return digest.hexdigest()

def audit_assets(deca_root: Path) -> list[DecaAssetRecord]:
    """Inventory expected local model assets without downloading or bypassing licences."""
    candidates={"deca_checkpoint":"data/deca_model.tar","flame_model":"data/generic_model.pkl","landmark_embedding":"data/landmark_embedding.npy","texture_model":"data/FLAME_albedo_from_BFM.npz","dense_template":"data/texture_data_256.npy","uv_face_mask":"data/uv_face_mask.png","uv_face_eye_mask":"data/uv_face_eye_mask.png","mean_texture":"data/mean_texture.jpg","fixed_displacement":"data/fixed_displacement_256.npy","head_template":"data/head_template.obj"}
    records=[]
    for name,relative in candidates.items():
        path=deca_root/relative; exists=path.is_file(); records.append(DecaAssetRecord(name,path,exists,path.stat().st_size if exists else None,sha256(path) if exists and path.stat().st_size else None,"not_loaded" if exists else "missing",True))
    return records
