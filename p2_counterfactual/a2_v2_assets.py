from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from p2_counterfactual.assets import PRESET_NAMES, ROOT, sha256_file, utc_now
from p2_counterfactual.io_utils import json_safe, write_json_atomic
from p2_counterfactual.path_utils import normalize_path_for_comparison


EXPERIMENT_DISPLAY_ID = "P2-A2-v2_R3DPR_MeanBG"
EXPERIMENT_ID = "p2_a2_v2_r3dpr_meanbg"
DATA_ROOT = ROOT / "data/processed/global_face/SixRelighting_OriginalCamera_meanfg"
P0_ORIGINAL_ROOT = ROOT / "data/processed/P0_Physics_Audit_v1/images/e0b_meanbg_224"
P1_MASTER_MANIFEST = ROOT / "data/processed/P1_Component_Audit_v1/manifests/p1_master_manifest.csv"
P0_MASTER_INDEX = ROOT / "data/processed/P0_Physics_Audit_v1/metadata/master_index.csv"
P1_QC_FLAGS = ROOT / "data/processed/P1_Component_Audit_v1/qc/p1_case_qc_flags.csv"

OUTPUT_DIR = DATA_ROOT / "p2_a2_v2_assets"
MANIFEST_PATH = DATA_ROOT / "p2_a2_v2_manifest.csv"
INVENTORY_PATH = OUTPUT_DIR / "dataset_inventory.csv"
AUDIT_PATH = OUTPUT_DIR / "dataset_audit.json"
MISSING_PATH = OUTPUT_DIR / "missing_or_invalid_cases.csv"
PRESET_MAPPING_PATH = OUTPUT_DIR / "preset_mapping.json"
AUDIT_RECORD_PATH = OUTPUT_DIR / "implementation_audit_record.md"


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _image_record(case_id: str, view_id: str, preset_index: int, path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "case_id": case_id,
        "view_id": view_id,
        "preset_index": preset_index,
        "path": _rel(path),
        "exists": path.is_file(),
        "file_size": int(path.stat().st_size) if path.is_file() else 0,
        "decode_ok": False,
        "mode": None,
        "width": None,
        "height": None,
        "channels": None,
        "all_black": None,
        "all_white": None,
        "pixel_std": None,
        "sha256": None,
        "error": "",
    }
    if not path.is_file():
        record["error"] = "missing"
        return record
    try:
        with Image.open(path) as image:
            record["mode"] = image.mode
            record["width"], record["height"] = image.size
            arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
        record["channels"] = 3
        record["decode_ok"] = True
        record["all_black"] = bool(np.all(arr == 0))
        record["all_white"] = bool(np.all(arr == 255))
        record["pixel_std"] = float(arr.std())
        record["sha256"] = sha256_file(path)
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def _load_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    master = pd.read_csv(P1_MASTER_MANIFEST, dtype=str)
    p0 = pd.read_csv(P0_MASTER_INDEX, dtype=str)
    qc = pd.read_csv(P1_QC_FLAGS, dtype=str)
    required = {"case_id", "group_id", "fold", "label_binary"}
    missing = sorted(required - set(master.columns))
    if missing:
        raise ValueError(f"P1 master manifest missing columns: {missing}")
    return master, p0, qc


def _original_consistency(case_id: str, new_original: Path, p0_original: Path) -> dict[str, Any]:
    result = {
        "case_id": case_id,
        "new_original_path": _rel(new_original),
        "p0_original_path": _rel(p0_original),
        "new_exists": new_original.is_file(),
        "p0_exists": p0_original.is_file(),
        "same_sha256": False,
        "max_abs_diff": None,
        "mean_abs_diff": None,
        "use_p0_original": True,
        "error": "",
    }
    if not new_original.is_file() or not p0_original.is_file():
        result["error"] = "missing_original"
        return result
    try:
        new_hash = sha256_file(new_original)
        p0_hash = sha256_file(p0_original)
        result["same_sha256"] = new_hash == p0_hash
        new_arr = _read_rgb(new_original)
        p0_arr = _read_rgb(p0_original)
        if new_arr.shape != p0_arr.shape:
            result["error"] = f"shape_mismatch {new_arr.shape} vs {p0_arr.shape}"
            return result
        diff = np.abs(new_arr.astype(np.int16) - p0_arr.astype(np.int16))
        result["max_abs_diff"] = int(diff.max())
        result["mean_abs_diff"] = float(diff.mean())
        result["use_p0_original"] = not bool(result["same_sha256"] or result["max_abs_diff"] == 0)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def build_a2_v2_assets() -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    master, p0, qc = _load_sources()
    p0_extra = p0[["ID", "jaw_neck_boundary_ambiguous", "boundary_ambiguity_status"]].rename(columns={"ID": "case_id"})
    qc_extra = qc[["case_id", "flag_count", "flags", "case_retained"]]
    merged = master.merge(p0_extra, on="case_id", how="left").merge(qc_extra, on="case_id", how="left")

    case_dirs = sorted([path for path in DATA_ROOT.iterdir() if path.is_dir() and path.name != OUTPUT_DIR.name], key=lambda p: p.name)
    case_ids_from_dirs = [path.name for path in case_dirs]
    master_ids = set(merged["case_id"].astype(str))
    inventory_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    original_consistency_rows: list[dict[str, Any]] = []

    for case_dir in case_dirs:
        case_id = case_dir.name
        row_match = merged[merged["case_id"].astype(str) == case_id]
        if row_match.empty:
            invalid_rows.append({"case_id": case_id, "reason": "case_absent_from_label_fold_manifest"})
            continue
        row = row_match.iloc[0]
        new_original = case_dir / "e0b_meanbg_224.png"
        p0_original = P0_ORIGINAL_ROOT / f"{case_id}.png"
        original_check = _original_consistency(case_id, new_original, p0_original)
        original_consistency_rows.append(original_check)
        original_for_training = p0_original if bool(original_check["use_p0_original"]) else new_original

        original_record = _image_record(case_id, "original", -1, original_for_training)
        inventory_rows.append(original_record)
        relight_paths = {preset: case_dir / f"relight_{preset}.png" for preset in PRESET_NAMES}
        relight_records = [
            _image_record(case_id, preset, index, relight_paths[preset])
            for index, preset in enumerate(PRESET_NAMES)
        ]
        inventory_rows.extend(relight_records)
        all_records = [original_record, *relight_records]
        failures = [
            rec["view_id"]
            for rec in all_records
            if not rec["exists"]
            or not rec["decode_ok"]
            or rec["width"] != 224
            or rec["height"] != 224
            or rec["channels"] != 3
            or rec["all_black"]
            or rec["all_white"]
            or float(rec["pixel_std"] or 0.0) <= 0.0
        ]
        if failures:
            invalid_rows.append({"case_id": case_id, "reason": "invalid_views", "views": json.dumps(failures)})
        flag_count_raw = str(row.get("flag_count", "")).strip()
        boundary_raw = str(row.get("jaw_neck_boundary_ambiguous", "")).strip().lower()
        manifest_rows.append(
            {
                "case_id": case_id,
                "patient_group_id": str(row["group_id"]),
                "fold": int(row["fold"]),
                "binary_label": int(row["label_binary"]),
                "original_path": _rel(original_for_training),
                "new_directory_original_path": _rel(new_original),
                "p0_original_path": _rel(p0_original),
                "original_rgb_path": _rel(original_for_training),
                **{f"relight_{preset}_path": _rel(relight_paths[preset]) for preset in PRESET_NAMES},
                **{f"preset_{index + 1}_id": preset for index, preset in enumerate(PRESET_NAMES)},
                **{f"relight_{index + 1}_path": _rel(relight_paths[preset]) for index, preset in enumerate(PRESET_NAMES)},
                "preset_names": json.dumps(list(PRESET_NAMES), ensure_ascii=False),
                "p1_qc_flag": flag_count_raw not in {"", "nan"} and int(float(flag_count_raw)) > 0,
                "boundary_uncertain": boundary_raw in {"1", "true", "yes"}
                or str(row.get("boundary_ambiguity_status", "")) == "jaw_neck_ambiguous",
                "source_asset_valid": True,
                "image_directory_manifest": True,
                "p2_training_ready": not failures,
            }
        )

    missing_from_dirs = sorted(master_ids - set(case_ids_from_dirs))
    for case_id in missing_from_dirs:
        invalid_rows.append({"case_id": case_id, "reason": "missing_case_directory"})

    manifest = pd.DataFrame(manifest_rows).sort_values(["fold", "case_id"], kind="stable").reset_index(drop=True)
    inventory = pd.DataFrame(inventory_rows).sort_values(["case_id", "preset_index"], kind="stable").reset_index(drop=True)
    invalid = pd.DataFrame(invalid_rows)
    original_consistency = pd.DataFrame(original_consistency_rows)
    manifest.to_csv(MANIFEST_PATH, index=False, encoding="utf-8")
    inventory.to_csv(INVENTORY_PATH, index=False, encoding="utf-8")
    invalid.to_csv(MISSING_PATH, index=False, encoding="utf-8")
    original_consistency.to_csv(OUTPUT_DIR / "original_consistency.csv", index=False, encoding="utf-8")

    patient_fold_counts = manifest.groupby("patient_group_id")["fold"].nunique() if not manifest.empty else pd.Series(dtype=int)
    checks = {
        "case_count_500": int(len(manifest)) == 500,
        "unique_case_id_500": int(manifest["case_id"].nunique()) == 500 if not manifest.empty else False,
        "directory_case_count_500": len(case_ids_from_dirs) == 500,
        "all_cases_labeled": not bool(set(case_ids_from_dirs) - master_ids),
        "all_manifest_cases_have_7_views": bool((inventory.groupby("case_id").size() == 7).all()) if not inventory.empty else False,
        "all_images_decode_rgb_224": bool(
            inventory["decode_ok"].all()
            and (inventory["width"].astype(int) == 224).all()
            and (inventory["height"].astype(int) == 224).all()
            and (inventory["channels"].astype(int) == 3).all()
        )
        if not inventory.empty
        else False,
        "no_blank_images": bool((~inventory["all_black"].astype(bool)).all() and (~inventory["all_white"].astype(bool)).all())
        if not inventory.empty
        else False,
        "patient_group_single_fold": bool((patient_fold_counts == 1).all()) if not manifest.empty else False,
        "no_old_deca_relighted_images_path": not any(
            "P2_Counterfactual_Relighting500_v1" in value or "P2_DetailPreserving_Relighting500_v2" in value
            for value in manifest.astype(str).to_numpy().ravel().tolist()
        )
        if not manifest.empty
        else False,
    }
    hard_failures = [key for key, value in checks.items() if not value]
    audit = {
        "experiment_display_id": EXPERIMENT_DISPLAY_ID,
        "experiment_id": EXPERIMENT_ID,
        "created_at": utc_now(),
        "data_root": str(DATA_ROOT),
        "manifest_path": str(MANIFEST_PATH),
        "case_count": int(len(manifest)),
        "directory_case_count": int(len(case_ids_from_dirs)),
        "inventory_rows": int(len(inventory)),
        "invalid_case_rows": int(len(invalid)),
        "preset_order": list(PRESET_NAMES),
        "original_consistency": {
            "same_sha256_count": int(original_consistency["same_sha256"].sum()) if not original_consistency.empty else 0,
            "use_p0_original_count": int(original_consistency["use_p0_original"].sum()) if not original_consistency.empty else 0,
            "max_abs_diff_max": int(original_consistency["max_abs_diff"].dropna().max()) if not original_consistency["max_abs_diff"].dropna().empty else None,
            "mean_abs_diff_max": float(original_consistency["mean_abs_diff"].dropna().max()) if not original_consistency["mean_abs_diff"].dropna().empty else None,
        },
        "fold_counts": {str(k): int(v) for k, v in manifest["fold"].value_counts().sort_index().to_dict().items()} if not manifest.empty else {},
        "label_counts": {str(k): int(v) for k, v in manifest["binary_label"].value_counts().sort_index().to_dict().items()} if not manifest.empty else {},
        "checks": checks,
        "hard_failures": hard_failures,
        "status": "passed" if not hard_failures and invalid.empty else "failed",
        "manifest_sha256": sha256_file(MANIFEST_PATH) if MANIFEST_PATH.is_file() else "missing",
    }
    write_json_atomic(AUDIT_PATH, audit)
    write_json_atomic(
        PRESET_MAPPING_PATH,
        {
            "preset_order": [
                {
                    "preset_index": index,
                    "canonical_preset_id": preset,
                    "filename": f"relight_{preset}.png",
                    "source": "R3DPR image-directory asset",
                }
                for index, preset in enumerate(PRESET_NAMES)
            ],
            "fixed_order_source": "p2_counterfactual.assets.PRESET_NAMES",
            "sh_ids_requested_in_prompt": [93, 62, 28, 39, 23, 81],
            "sh_ids_present_in_filenames": False,
        },
    )
    _write_audit_record(audit)
    return audit


def _write_audit_record(audit: dict[str, Any]) -> None:
    lines = [
        "# P2-A2-v2 Internal Implementation Audit",
        "",
        f"- Experiment: `{EXPERIMENT_DISPLAY_ID}` (`{EXPERIMENT_ID}`)",
        f"- New data root: `{DATA_ROOT}`",
        "- Reused code: `P2ASingleRGBResNet18`, `P2AFoldTrainer`, `P2AEvaluator`, `aggregate_p2_a_experiment`, `compute_stability_outputs`, `paired_cluster_bootstrap_comparison`, and `decide_p2_a2_gate`.",
        "- New code: image-directory asset audit/manifest builder and image-path loading branch in `P2ASingleRGBDataset`.",
        "- Config changes: independent experiment id/output root/manifest path, `input_mode=relight_mix`, `original_probability=0.5`, `color_jitter_enabled=false`, `consistency_enabled=false`.",
        "- Old P2-A2 path: `data/processed/P2_DetailPreserving_Relighting500_v2/manifests/p2_training_manifest.csv` and historical `p2_a2_relighting` outputs under `experiments/500Data/P2_Physics_Relighting_v2/rgb_benchmark`.",
        "- New P2-A2-v2 path: image columns in the newly generated manifest, sourced from the R3DPR `SixRelighting_OriginalCamera_meanfg` directory.",
        f"- Asset audit status: `{audit.get('status')}`; hard failures: `{audit.get('hard_failures')}`.",
        f"- Original consistency: `{audit.get('original_consistency')}`.",
    ]
    AUDIT_RECORD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    audit = build_a2_v2_assets()
    print(json.dumps(json_safe(audit), ensure_ascii=False, indent=2), flush=True)
    return 0 if audit["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
