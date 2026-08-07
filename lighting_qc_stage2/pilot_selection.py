from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def select_pilot32(split: pd.DataFrame, exif: pd.DataFrame, manifest: pd.DataFrame, *, pilot_size: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = split.merge(exif, on="sample_id", how="left").merge(
        manifest[["sample_id", "warning_codes", "face_valid_mask_path", "source_valid_mask_path", "parsing_label_path", "aligned_srgb_path", "metadata_path", "image_path"]],
        on="sample_id",
        how="left",
    )
    selected: dict[str, list[str]] = {}

    def add(sample_id: str, reason: str) -> None:
        selected.setdefault(str(sample_id), []).append(reason)

    for fold in sorted(base["fold"].unique()):
        for label in [0, 1]:
            pool = base[(base["fold"] == fold) & (base["binary_label"] == label)]
            if not pool.empty:
                add(str(pool.sample(n=1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]["sample_id"]), f"fold{fold}_label{label}_coverage")

    for col, low_reason, high_reason in [
        ("brightness_value_apex", "brightness_low_extreme", "brightness_high_extreme"),
        ("iso", "iso_low_extreme", "iso_high_extreme"),
        ("exposure_time", "exposure_time_low_extreme", "exposure_time_high_extreme"),
    ]:
        valid = base.dropna(subset=[col]).sort_values([col, "sample_id"], kind="mergesort")
        for sid in valid["sample_id"].head(2):
            add(str(sid), low_reason)
        for sid in valid["sample_id"].tail(2):
            add(str(sid), high_reason)

    warning_pool = base[base["warning_codes"].fillna("").astype(str).ne("")]
    for sid in warning_pool.sort_values(["warning_codes", "sample_id"], kind="mergesort")["sample_id"].head(6):
        add(str(sid), "scheme_b_warning_or_geometry_edge")

    remaining = base[~base["sample_id"].isin(selected)].copy()
    remaining = remaining.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    for sid in remaining["sample_id"]:
        if len(selected) >= pilot_size:
            break
        add(str(sid), "seeded_random_fill")

    ordered_ids = list(selected)[:pilot_size]
    pilot = base.set_index("sample_id").loc[ordered_ids].reset_index()
    pilot["selection_reason"] = pilot["sample_id"].map(lambda sid: ";".join(selected[str(sid)]))
    pilot["scheme_b_warning"] = pilot["warning_codes"].fillna("").astype(str)
    pilot["raw_scene_path"] = pilot["sample_id"].map(lambda sid: "")
    if len(pilot) != pilot_size or pilot["sample_id"].nunique() != pilot_size:
        raise ValueError("Pilot32 selection did not produce 32 unique IDs")
    if pilot["fold"].nunique() != 5 or set(pilot["binary_label"].unique()) != {0, 1}:
        raise ValueError("Pilot32 lacks fold or label coverage")
    return pilot
