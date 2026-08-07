"""Deterministic, metadata-only 4/4/4 pilot selection."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import P0BConfig
from .p0a_reader import resolve_asset
from .types import PilotSample

def _group(row: pd.Series, rules: dict[str, dict[str, str]]) -> str | None:
    """Use the audited make, model, and binary-field tuple rather than a prefix guess."""
    values = {"camera_make": str(row.camera_make).strip(), "camera_model": str(row.camera_model).strip(), "binary_name": str(row.binary_name).strip()}
    for name, expected in rules.items():
        if values == expected:
            return name
    return None

def select_pilot12(master: pd.DataFrame, config: P0BConfig) -> list[PilotSample]:
    """Select 4/4/4 samples using only pre-DECA metadata and deterministic diversity."""
    table = master.copy(); table["acquisition_group"] = table.apply(lambda row: _group(row, config.acquisition_group_rules), axis=1)
    pixels = pd.to_numeric(table.physics_core_skin_pixel_count, errors="coerce").fillna(0)
    usable = table[(table.p0_usable.astype(str)=="1") & (pixels > 0) & ~table.ID.isin(config.excluded_ids) & table.acquisition_group.notna()].copy()
    for column in ("aligned_scene_relpath","physics_core_skin_mask_relpath"):
        usable = usable[usable[column].map(lambda p: (config.p0a_root / str(p)).is_file())]
    usable = usable.sort_values(["acquisition_group","patient_group_id","ID"], kind="mergesort")
    selected: list[dict[str, object]] = []
    for group, count in config.selection_groups.items():
        candidates = usable[usable.acquisition_group.eq(group)].drop_duplicates("patient_group_id", keep="first").copy()
        if len(candidates) < count: raise ValueError(f"insufficient eligible samples for {group}: {len(candidates)}")
        for column in ("brightness_value","iso","exposure_time_s"):
            candidates[column] = pd.to_numeric(candidates[column], errors="coerce").fillna(candidates[column].median() if candidates[column].notna().any() else 0.0)
        selected.extend(_select_group(candidates, count, config.seed, group))
    selected = sorted(selected, key=lambda value: (str(value["acquisition_group"]), str(value["ID"])))
    return [PilotSample(f"P0B-{i:03d}", str(row["ID"]), str(row["patient_group_id"]), str(row["acquisition_group"]), int(row["SEX"]), _number(row.get("brightness_value")), _number(row.get("iso")), _number(row.get("exposure_time_s")), str(row.get("forehead_available")).lower() in {"true","1"}, resolve_asset(config.p0a_root,str(row["aligned_scene_relpath"])), resolve_asset(config.p0a_root,str(row["face_valid_mask_relpath"])), resolve_asset(config.p0a_root,str(row["skin_strict_mask_relpath"])), resolve_asset(config.p0a_root,str(row["physics_core_skin_mask_relpath"]))) for i,row in enumerate(selected,1)]

def _number(value: object) -> float | None:
    try: return float(value)
    except (TypeError, ValueError): return None


def _forehead_value(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1"}


def _select_group(candidates: pd.DataFrame, count: int, seed: int, group: str) -> list[dict[str, object]]:
    """Prioritize available sex/forehead strata, then greedy maximin EXIF distance.

    Selection has no access to DECA output, clinical severity, predictions, or visual QC.
    Each binary stratum is required only when it exists among eligible candidates.
    """
    table = candidates.copy().sort_values("ID", kind="mergesort").reset_index(drop=True)
    table["_sex"] = pd.to_numeric(table.SEX, errors="coerce").fillna(-1).astype(int)
    table["_forehead"] = table.forehead_available.map(_forehead_value)
    features = table[["brightness_value", "iso", "exposure_time_s"]].to_numpy(dtype=float)
    scale = np.ptp(features, axis=0)
    scale[scale < 1e-12] = 1.0
    features = (features - np.median(features, axis=0)) / scale
    rng = np.random.default_rng(seed + sum(ord(char) for char in group))
    table["_tie"] = rng.random(len(table))
    required_sex = set(table._sex.unique())
    required_forehead = set(table._forehead.unique())
    chosen: list[int] = []
    while len(chosen) < count:
        remaining = [index for index in range(len(table)) if index not in chosen]
        selected_sex = set(table.loc[chosen, "_sex"]) if chosen else set()
        selected_forehead = set(table.loc[chosen, "_forehead"]) if chosen else set()
        missing_sex = required_sex - selected_sex
        missing_forehead = required_forehead - selected_forehead
        scores: list[tuple[float, float, str, int]] = []
        for index in remaining:
            coverage_gain = int(table.at[index, "_sex"] in missing_sex) + int(table.at[index, "_forehead"] in missing_forehead)
            if chosen:
                min_distance = float(np.min(np.linalg.norm(features[index] - features[chosen], axis=1)))
            else:
                min_distance = float(np.linalg.norm(features[index]))
            # The large fixed multiplier makes categorical coverage take precedence over EXIF spread.
            scores.append((1000.0 * coverage_gain + min_distance, float(table.at[index, "_tie"]), str(table.at[index, "ID"]), index))
        _, _, _, winner = max(scores)
        chosen.append(winner)
    return table.loc[chosen].drop(columns=["_sex", "_forehead", "_tie"]).to_dict("records")
