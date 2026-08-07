from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .checkpoint_resolver import inference_contract, resolve_checkpoints
from .config import Stage3A0Config


def _exists(path: Path) -> bool:
    return path.is_file()


def run_preflight(config: Stage3A0Config, dirs: dict[str, Path]) -> dict[str, Any]:
    split = pd.read_csv(config.split_csv, dtype={"ID": str, "patient_group_id": str})
    oof = pd.read_csv(config.rgb_oof_csv, dtype={"sample_id": str, "patient_group_id": str})
    stage1 = pd.read_csv(config.stage1_master_csv, dtype={"sample_id": str, "patient_group_id": str})
    features = pd.read_csv(config.stage2_feature_csv, dtype={"sample_id": str, "patient_group_id": str})
    split = split.rename(columns={"ID": "sample_id"}) if "ID" in split.columns else split
    inventory_rows = []
    for _, row in split.iterrows():
        sid = str(row["sample_id"])
        image = config.input_image_dir / f"{sid}.png"
        mask = config.face_valid_mask_dir / f"{sid}.png"
        inventory_rows.append({
            "sample_id": sid,
            "image_path": str(image),
            "mask_path": str(mask),
            "image_exists": _exists(image),
            "mask_exists": _exists(mask),
        })
    inventory = pd.DataFrame(inventory_rows)
    shape_failures = []
    for _, row in inventory.iterrows():
        if not row["image_exists"] or not row["mask_exists"]:
            continue
        with Image.open(row["image_path"]) as im:
            image = np.asarray(im.convert("RGB"))
        with Image.open(row["mask_path"]) as m:
            mask = np.asarray(m.convert("L"))
        if image.shape != (320, 256, 3) or mask.shape != (320, 256):
            shape_failures.append(str(row["sample_id"]))
        vals = set(np.unique(mask).tolist())
        if not vals.issubset({0, 255}):
            shape_failures.append(str(row["sample_id"]) + ":mask_values")
        if np.count_nonzero(image[mask == 0]) != 0:
            shape_failures.append(str(row["sample_id"]) + ":background")
    merged = split[["sample_id", "patient_group_id", "fold", "binary_label"]].merge(
        oof[["sample_id", "patient_group_id", "fold", "binary_label"]], on="sample_id", suffixes=("_split", "_oof"), how="outer"
    )
    stage1_ids = set(stage1["sample_id"].astype(str))
    feature_ids = set(features["sample_id"].astype(str))
    camera = stage1[["sample_id", "camera_model", "binary_label"]].copy()
    camera_label = camera.groupby(["camera_model", "binary_label"], dropna=False).size().reset_index(name="n")
    checkpoints = resolve_checkpoints(config.rgb_oof_csv)
    contract = inference_contract(checkpoints)
    checkpoints.to_csv(dirs["preflight"] / "checkpoint_inventory.csv", index=False, encoding="utf-8-sig")
    inventory.to_csv(dirs["preflight"] / "input_asset_inventory.csv", index=False, encoding="utf-8-sig")
    camera_label.to_csv(dirs["preflight"] / "camera_label_distribution.csv", index=False, encoding="utf-8-sig")
    (dirs["preflight"] / "model_inference_contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "formal_sample_count": int(len(split)),
        "unique_sample_id": int(split["sample_id"].nunique()),
        "folds": sorted(pd.to_numeric(split["fold"], errors="coerce").dropna().astype(int).unique().tolist()),
        "patient_group_cross_fold": int(split.groupby("patient_group_id")["fold"].nunique().gt(1).sum()),
        "rgb_oof_coverage": int(split["sample_id"].astype(str).isin(set(oof["sample_id"].astype(str))).sum()),
        "input_image_coverage": int(inventory["image_exists"].sum()),
        "face_valid_mask_coverage": int(inventory["mask_exists"].sum()),
        "camera_model_coverage": int(stage1["camera_model"].notna().sum()),
        "stage1_id_match": set(split["sample_id"].astype(str)) == stage1_ids,
        "stage2_feature_id_match": set(split["sample_id"].astype(str)) == feature_ids,
        "label_conflicts": int((merged["binary_label_split"].astype(str) != merged["binary_label_oof"].astype(str)).sum()),
        "fold_conflicts": int((merged["fold_split"].astype(str) != merged["fold_oof"].astype(str)).sum()),
        "duplicate_oof": int(oof["sample_id"].duplicated().sum()),
        "critical_asset_missing": int((~inventory["image_exists"] | ~inventory["mask_exists"]).sum()),
        "shape_or_mask_failures": shape_failures[:20],
        "checkpoint_count": int(len(checkpoints)),
        "run_training": bool(config.run_training),
    }
    summary["pass"] = bool(
        summary["formal_sample_count"] == 500
        and summary["unique_sample_id"] == 500
        and summary["folds"] == [0, 1, 2, 3, 4]
        and summary["patient_group_cross_fold"] == 0
        and summary["rgb_oof_coverage"] == 500
        and summary["input_image_coverage"] == 500
        and summary["face_valid_mask_coverage"] == 500
        and summary["camera_model_coverage"] == 500
        and summary["label_conflicts"] == 0
        and summary["fold_conflicts"] == 0
        and summary["duplicate_oof"] == 0
        and summary["critical_asset_missing"] == 0
        and not summary["shape_or_mask_failures"]
        and summary["checkpoint_count"] == 5
        and not summary["run_training"]
    )
    (dirs["preflight"] / "stage3_a0_preflight_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (dirs["preflight"] / "stage3_a0_preflight_report.md").write_text(
        "# Stage3-A0 Preflight\n\n" + "\n".join(f"- {k}: {v}" for k, v in summary.items()) + "\n",
        encoding="utf-8",
    )
    (dirs["preflight"] / "checkpoint_resolution_report.md").write_text(
        "# Checkpoint Resolution\n\n" + "\n".join(f"- fold {r.fold}: `{r.checkpoint_path}` epoch={r.selected_epoch} sha256={r.checkpoint_sha256}" for r in checkpoints.itertuples()) + "\n",
        encoding="utf-8",
    )
    if not summary["pass"]:
        raise ValueError(f"Stage3-A0 preflight failed: {summary}")
    return {"summary": summary, "checkpoints": checkpoints, "contract": contract}
