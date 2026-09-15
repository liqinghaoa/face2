"""Deterministic R1 diagnostic sample selection from frozen SO-1 metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read_reused_ids(path: Path) -> list[str] | None:
    if path.is_file():
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return None


def _persist(path: Path, ids: list[str], selection: dict[str, Any]) -> list[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")
    selection_stem = path.stem.removesuffix("_id").removesuffix("_ids")
    path.with_name(selection_stem + "_selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return ids


def _eligible(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.loc[
        (frame["generation_status"].astype(str) == "SUCCESS")
        & (frame["retry_count"].fillna(99).astype(int) == 0)
        & (frame["total_clip_fraction"].fillna(1.0).astype(float) <= 0.01)
        & (frame["m_std"].fillna(0.0).astype(float) > 1.0e-4)
        & (frame["h_std"].fillna(0.0).astype(float) > 1.0e-4)
    ].copy()
    return result


def select_overfit1(metadata: pd.DataFrame, ids_path: str | Path) -> list[str]:
    path = Path(ids_path)
    reused = _read_reused_ids(path)
    if reused is not None:
        return reused
    eligible = _eligible(metadata)
    eligible = eligible.loc[(eligible["valid_mask_type"].astype(str) == "full") & (eligible["p_nonzero_fraction"] > 0)].copy()
    if eligible.empty:
        raise ValueError("No eligible deterministic overfit1 sample")
    fields = ["m_mean", "m_std", "h_mean", "h_std", "s_mean", "s_std", "p_mean", "p_std", "p_nonzero_fraction"]
    medians = eligible[fields].median()
    scale = (eligible[fields] - medians).abs().median().replace(0.0, 1.0)
    eligible["selection_distance"] = ((eligible[fields] - medians).abs() / scale).sum(axis=1)
    row = eligible.sort_values(["selection_distance", "split_index"], kind="mergesort").iloc[0]
    return _persist(path, [str(row.sample_id)], {"rule": "eligible sample closest to eligible target-statistic medians", "selected": row.to_dict(), "eligible_count": int(len(eligible))})


def select_paired_overfit2(metadata: pd.DataFrame, ids_path: str | Path) -> list[str]:
    path = Path(ids_path)
    reused = _read_reused_ids(path)
    if reused is not None:
        return reused
    eligible = _eligible(metadata)
    rows: list[pd.DataFrame] = []
    for _, group in eligible.groupby("base_latent_id", sort=True):
        if set(group["acquisition_variant_id"].astype(int)) != {0, 1}:
            continue
        pair = group.sort_values("acquisition_variant_id", kind="mergesort")
        a, b = pair.iloc[0], pair.iloc[1]
        same = all(a[field] == b[field] for field in ("m_seed", "h_seed", "mask_seed"))
        changed = all(a[field] != b[field] for field in ("s_seed", "p_seed", "final_camera_light_pair"))
        if same and changed and float(pair["p_nonzero_fraction"].max()) > 0.0:
            rows.append(pair)
    if not rows:
        raise ValueError("No eligible deterministic paired overfit2 sample pair")
    candidates = pd.concat(rows).sort_values(["base_latent_id", "acquisition_variant_id"], kind="mergesort")
    base_id = int(candidates.iloc[0].base_latent_id)
    selected = candidates.loc[candidates["base_latent_id"] == base_id]
    return _persist(path, selected["sample_id"].astype(str).tolist(), {"rule": "lowest eligible base_latent_id with exact paired metadata", "base_latent_id": base_id, "selected": selected.to_dict("records"), "eligible_pair_count": len(rows)})


def select_fixed_camera_light16(metadata: pd.DataFrame, ids_path: str | Path) -> list[str]:
    path = Path(ids_path)
    reused = _read_reused_ids(path)
    if reused is not None:
        return reused
    eligible = _eligible(metadata)
    candidates: list[tuple[float, str, pd.DataFrame]] = []
    global_median = eligible[["m_mean", "h_mean", "m_std", "h_std"]].median()
    for pair, group in eligible.groupby("final_camera_light_pair", sort=True):
        one_per_base = group.sort_values(["total_clip_fraction", "split_index"], kind="mergesort").drop_duplicates("base_latent_id")
        if len(one_per_base) < 16:
            continue
        score = float((one_per_base[["m_mean", "h_mean", "m_std", "h_std"]].median() - global_median).abs().sum())
        candidates.append((score, str(pair), one_per_base))
    if not candidates:
        raise ValueError("No fixed camera/light pair with 16 eligible base latents")
    _, pair, pool = sorted(candidates, key=lambda item: (item[0], item[1]))[0]
    pool = pool.sort_values(["m_mean", "h_mean", "split_index"], kind="mergesort")
    positions = [round(i * (len(pool) - 1) / 15) for i in range(16)]
    selected = pool.iloc[positions].sort_values("split_index", kind="mergesort")
    return _persist(path, selected["sample_id"].astype(str).tolist(), {"rule": "eligible final camera-light pair nearest global target medians; stratified 16 selection", "final_camera_light_pair": pair, "selected": selected.to_dict("records"), "candidate_pair_count": len(candidates)})
