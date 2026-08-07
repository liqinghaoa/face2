"""Strict reader for finalized P0-A tables and relative asset paths."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = {"ID","patient_group_id","camera_make","camera_model","binary_name","p0_usable","forehead_available","aligned_scene_relpath","face_valid_mask_relpath","skin_strict_mask_relpath","physics_core_skin_mask_relpath","physics_core_skin_pixel_count"}

def load_master(p0a_root: Path) -> pd.DataFrame:
    """Load the finalized 500-row P0-A master table without using labels for pixel work."""
    table = pd.read_csv(p0a_root / "metadata/master_index.csv", dtype={"ID":str,"patient_group_id":str}).fillna("")
    missing = REQUIRED_COLUMNS - set(table.columns)
    if missing: raise ValueError(f"P0-A master missing fields: {sorted(missing)}")
    if len(table) != 500 or table.ID.nunique() != 500: raise ValueError("P0-A master must have 500 unique rows")
    return table

def resolve_asset(p0a_root: Path, relpath: str) -> Path:
    """Resolve an existing relative P0-A asset and reject root escapes."""
    value = (p0a_root / relpath).resolve()
    if p0a_root.resolve() not in value.parents or not value.is_file(): raise FileNotFoundError(f"missing P0-A asset: {relpath}")
    return value
