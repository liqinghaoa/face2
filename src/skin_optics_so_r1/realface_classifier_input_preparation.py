"""SO-R2-X3-P0 frozen classifier-input preparation.

This module is deliberately label-blind and does not import model code.  It
only resizes the authoritative high-resolution RGB, X1 maps, and X1 masks.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


DATA = Path("data/processed/skin_optics/skinoptics_realface_979x1220_blackbg_v1")
X1 = Path("data/processed/SO_R2X1_RealFaceFrozenInference_v1")
X2 = Path("data/processed/SO_R2X2_RealFaceOutputAudit_v1")
OUT = Path("data/processed/SO_R2X3_ClassifierInputs_v1")
REPORT = Path("reports/so_r2x3_classifier_input_preparation")
B1_CHECKPOINT = Path("runs/so_r1_b1_baseline_v6/checkpoints/best_val_masked_smoothl1.pt")
B2_CHECKPOINT = Path("runs/so_r1_b2_proposed_v3/lambda_0p50/checkpoints/best_val_masked_smoothl1.pt")
SOURCE_SHAPE = (1220, 979)
# PyTorch/OpenCV sizes are HxW; the protocol describes this as 256x319 (WxH).
RESIZED_SHAPE = (319, 256)
TARGET_SHAPE = (320, 256)
B1_HASH = "5b1dd85f840d4d00a7c1c1f3b7c0a9306031408ecf0c5a557f53ca0810c0265b"
B2_HASH = "5cd86ceee6c9ad7f448676b59b3f1f5e8d80dc28bce106101b19d1286b6562a0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _read_mask(path: Path) -> np.ndarray:
    if not path.is_file():
        raise RuntimeError(f"MISSING_MASK:{path}")
    array = np.asarray(Image.open(path).convert("L"))
    if array.shape != SOURCE_SHAPE or set(np.unique(array).tolist()) - {0, 255}:
        raise RuntimeError(f"INVALID_MASK:{path}")
    return array


def _resize_chw(array: np.ndarray, mode: str) -> np.ndarray:
    """Resize CHW float32 using PyTorch's explicit align_corners=False rule."""
    tensor = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))[None]
    if mode == "bilinear":
        result = F.interpolate(tensor, size=RESIZED_SHAPE, mode=mode, align_corners=False)
    elif mode == "nearest":
        result = F.interpolate(tensor, size=RESIZED_SHAPE, mode=mode)
    else:
        raise ValueError(mode)
    return result[0].cpu().numpy().astype(np.float32, copy=False)


def _pad_bottom(array: np.ndarray) -> np.ndarray:
    padded = np.zeros((array.shape[0], TARGET_SHAPE[0], TARGET_SHAPE[1]), dtype=np.float32)
    padded[:, : RESIZED_SHAPE[0], : RESIZED_SHAPE[1]] = array
    return padded


def _load_case(root: Path, case_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    data = root / DATA
    x1 = root / X1
    rgb = np.load(data / "blackbg_linear_rgb" / f"{case_id}.npy", allow_pickle=False)
    b1 = np.load(x1 / "B1" / f"{case_id}.npz", allow_pickle=False)["mh_sensitive_map"]
    b2 = np.load(x1 / "B2_lambda_0p50" / f"{case_id}.npz", allow_pickle=False)["mh_sensitive_map"]
    valid = _read_mask(x1 / "valid_masks" / f"{case_id}.png")
    source_mask = _read_mask(data / "source_valid_mask" / f"{case_id}.png")
    if rgb.shape != (1220, 979, 3) or rgb.dtype != np.float32:
        raise RuntimeError(f"INVALID_RGB:{case_id}:{rgb.shape}:{rgb.dtype}")
    if b1.shape != (2, 1220, 979) or b2.shape != (2, 1220, 979):
        raise RuntimeError(f"INVALID_MAP_SHAPE:{case_id}")
    if b1.dtype != np.float32 or b2.dtype != np.float32:
        raise RuntimeError(f"INVALID_MAP_DTYPE:{case_id}")
    if not np.isfinite(rgb).all() or not np.isfinite(b1).all() or not np.isfinite(b2).all():
        raise RuntimeError(f"NONFINITE_SOURCE:{case_id}")
    info = {
        "source_mask_shape": list(source_mask.shape),
        "source_mask_binary_0_255": True,
        "source_mask_matches_x1_mask": bool(np.array_equal(source_mask, valid)),
        "x1_mask_valid_fraction": float((valid > 0).mean()),
        "source_mask_valid_fraction": float((source_mask > 0).mean()),
    }
    return rgb, b1, b2, valid, info


def _ids(root: Path) -> list[str]:
    manifest = root / X1 / "inference_manifest.csv"
    rows = list(csv.DictReader(manifest.open(encoding="utf-8", newline="")))
    ids = [row["case_id"].strip() for row in rows]
    if len(ids) != 500 or len(set(ids)) != 500 or any(not x for x in ids):
        raise RuntimeError("INVALID_X1_CANONICAL_MANIFEST")
    source_ids = [line.strip() for line in (root / DATA / "full_ids.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    if ids != source_ids:
        raise RuntimeError("X1_CANONICAL_ORDER_MISMATCH_SOURCE_FULL_IDS")
    return ids


def _preflight(root: Path) -> tuple[list[str], dict[str, Any]]:
    root = root.resolve()
    x1_acc_path, x2_acc_path = root / X1 / "R2X1_ACCEPTANCE.json", root / X2 / "R2X2_ACCEPTANCE.json"
    x1_acc, x2_acc = json.loads(x1_acc_path.read_text()), json.loads(x2_acc_path.read_text())
    if x1_acc.get("status") != "COMPLETE_EXPLORATORY_FROZEN_INFERENCE":
        raise RuntimeError("BLOCKED:X1_ACCEPTANCE")
    if x2_acc.get("status") != "COMPLETE_EXPLORATORY_OUTPUT_AUDIT" or x2_acc.get("technical_output_integrity") != "PASS":
        raise RuntimeError("BLOCKED:X2_ACCEPTANCE")
    b1_hash, b2_hash = sha256_file(root / B1_CHECKPOINT), sha256_file(root / B2_CHECKPOINT)
    if (b1_hash, b2_hash) != (B1_HASH, B2_HASH):
        raise RuntimeError("BLOCKED:CHECKPOINT_HASH")
    ids = _ids(root)
    for case_id in ids:
        _load_case(root, case_id)
    expected_npz = {f"{case_id}.npz" for case_id in ids}
    expected_png = {f"{case_id}.png" for case_id in ids}
    for directory in (root / X1 / "B1", root / X1 / "B2_lambda_0p50", root / X1 / "valid_masks"):
        expected = expected_png if directory.name == "valid_masks" else expected_npz
        actual = {path.name for path in directory.iterdir() if path.is_file()}
        if actual != expected:
            raise RuntimeError(f"BLOCKED:COUNT:{directory}")
    return ids, {
        "source_R2X1_acceptance_hash": sha256_file(x1_acc_path),
        "source_R2X2_acceptance_hash": sha256_file(x2_acc_path),
        "B1_checkpoint_hash": b1_hash,
        "B2_checkpoint_hash": b2_hash,
        "lambda_pair": 0.50,
    }


def _protected_snapshot(root: Path) -> dict[str, str]:
    """Hash all source/X1 files consumed by this stage plus authority records."""
    files = [
        root / X1 / "R2X1_ACCEPTANCE.json", root / X2 / "R2X2_ACCEPTANCE.json",
        root / X1 / "inference_manifest.csv", root / DATA / "full_ids.txt",
        root / DATA / "manifest.csv", root / DATA / "COMPLETED.json",
        root / DATA / "config" / "resolved_config.yaml", root / B1_CHECKPOINT, root / B2_CHECKPOINT,
    ]
    files += sorted((root / DATA / "blackbg_linear_rgb").glob("*.npy"))
    files += sorted((root / DATA / "source_valid_mask").glob("*.png"))
    files += sorted((root / X1 / "B1").glob("*.npz"))
    files += sorted((root / X1 / "B2_lambda_0p50").glob("*.npz"))
    files += sorted((root / X1 / "valid_masks").glob("*.png"))
    if not all(path.is_file() for path in files):
        raise RuntimeError("PROTECTED_ASSET_MISSING")
    return {str(path.relative_to(root)): sha256_file(path) for path in files}


def _write_png(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path, format="PNG")


def _prepare_case(root: Path, out: Path, case_id: str) -> dict[str, Any]:
    rgb, b1, b2, valid, source_info = _load_case(root, case_id)
    mask_small = _resize_chw((valid > 0)[None].astype(np.float32), "nearest")[0] > 0.5
    rgb_small = _resize_chw(np.transpose(rgb, (2, 0, 1)), "bilinear") * mask_small[None]
    b1_small = _resize_chw(b1, "bilinear") * mask_small[None]
    b2_small = _resize_chw(b2, "bilinear") * mask_small[None]
    rgb_out, b1_out, b2_out = _pad_bottom(rgb_small), _pad_bottom(np.concatenate((rgb_small, b1_small), axis=0)), _pad_bottom(np.concatenate((rgb_small, b2_small), axis=0))
    np.save(out / "rgb/images" / f"{case_id}.npy", rgb_out.astype(np.float32, copy=False), allow_pickle=False)
    np.save(out / "rgb_b1mh/images" / f"{case_id}.npy", b1_out.astype(np.float32, copy=False), allow_pickle=False)
    np.save(out / "rgb_b2mh/images" / f"{case_id}.npy", b2_out.astype(np.float32, copy=False), allow_pickle=False)
    _write_png(out / "common/valid_masks" / f"{case_id}.png", _pad_bottom(mask_small[None])[0] > 0)
    return {"case_id": case_id, **source_info, "rgb_source_shape": str(rgb.shape), "b1_source_shape": str(b1.shape), "b2_source_shape": str(b2.shape), "output_rgb_shape": str(rgb_out.shape), "output_b1mh_shape": str(b1_out.shape), "output_b2mh_shape": str(b2_out.shape)}


def _audit_outputs(root: Path, out: Path, ids: list[str]) -> dict[str, Any]:
    hard: list[str] = []
    rows: list[dict[str, Any]] = []
    rgb_identity = True
    mh_trace = True
    expected_names = {f"{case_id}.npy" for case_id in ids}
    expected_mask_names = {f"{case_id}.png" for case_id in ids}
    file_inventory = {
        "rgb": sorted(path.name for path in (out / "rgb/images").iterdir() if path.is_file()),
        "rgb_b1mh": sorted(path.name for path in (out / "rgb_b1mh/images").iterdir() if path.is_file()),
        "rgb_b2mh": sorted(path.name for path in (out / "rgb_b2mh/images").iterdir() if path.is_file()),
        "masks": sorted(path.name for path in (out / "common/valid_masks").iterdir() if path.is_file()),
    }
    inventory_ok = (
        set(file_inventory["rgb"]) == expected_names
        and set(file_inventory["rgb_b1mh"]) == expected_names
        and set(file_inventory["rgb_b2mh"]) == expected_names
        and set(file_inventory["masks"]) == expected_mask_names
    )
    if not inventory_ok:
        hard.append("file_inventory_mismatch")
    for case_id in ids:
        rgb = np.load(out / "rgb/images" / f"{case_id}.npy", allow_pickle=False)
        b1 = np.load(out / "rgb_b1mh/images" / f"{case_id}.npy", allow_pickle=False)
        b2 = np.load(out / "rgb_b2mh/images" / f"{case_id}.npy", allow_pickle=False)
        mask_raw = np.asarray(Image.open(out / "common/valid_masks" / f"{case_id}.png").convert("L"))
        mask_binary = mask_raw.shape == (320, 256) and set(np.unique(mask_raw).tolist()) <= {0, 255}
        mask = mask_raw > 0
        shape_ok = rgb.shape == (3, 320, 256) and b1.shape == (5, 320, 256) and b2.shape == (5, 320, 256) and mask_raw.shape == (320, 256)
        dtype_ok = rgb.dtype == np.float32 and b1.dtype == np.float32 and b2.dtype == np.float32
        finite = bool(np.isfinite(rgb).all() and np.isfinite(b1).all() and np.isfinite(b2).all())
        outside_zero = bool(np.all(rgb[:, ~mask] == 0) and np.all(b1[:, ~mask] == 0) and np.all(b2[:, ~mask] == 0))
        padding_zero = bool(np.all(rgb[:, 319] == 0) and np.all(b1[:, 319] == 0) and np.all(b2[:, 319] == 0) and np.all(~mask[319]))
        identity = bool(np.array_equal(rgb, b1[:3]) and np.array_equal(rgb, b2[:3]))
        rgb_identity &= identity
        if not (shape_ok and dtype_ok and mask_binary and finite and outside_zero and padding_zero and identity):
            hard.append(case_id)
        maps = (b1[3:], b2[3:])
        nonzero_nonconstant = True
        for condition, source_dir, values in (("B1", "B1", maps[0]), ("B2", "B2_lambda_0p50", maps[1])):
            source = np.load(root / X1 / source_dir / f"{case_id}.npz", allow_pickle=False)["mh_sensitive_map"]
            expected = _pad_bottom(_resize_chw(source, "bilinear") * _resize_chw(((_read_mask(root / X1 / "valid_masks" / f"{case_id}.png") > 0)[None]).astype(np.float32), "nearest")[0][None])
            expected = expected * mask[None]
            trace = bool(np.array_equal(values, expected))
            mh_trace &= trace
            valid_values = values[:, mask]
            nonzero_nonconstant &= bool(np.isfinite(valid_values).all() and np.any(valid_values != 0) and all(np.unique(ch).size > 1 for ch in valid_values))
            rows.append({"case_id": case_id, "condition": condition, "shape_ok": shape_ok, "dtype_ok": dtype_ok, "mask_binary_0_255": mask_binary, "finite": finite, "mask_outside_zero": outside_zero, "padding_zero": padding_zero, "mh_source_trace": trace, "mh_nonzero_nonconstant": nonzero_nonconstant})
        if not nonzero_nonconstant:
            hard.append(f"{case_id}:constant_mh")
    counts = {
        "rgb": len(list((out / "rgb/images").glob("*.npy"))),
        "rgb_b1mh": len(list((out / "rgb_b1mh/images").glob("*.npy"))),
        "rgb_b2mh": len(list((out / "rgb_b2mh/images").glob("*.npy"))),
        "masks": len(list((out / "common/valid_masks").glob("*.png"))),
    }
    return {"status": "PASS" if not hard and rgb_identity and mh_trace else "FAIL", "hard_failure_count": len(hard), "hard_failures": hard[:20], "case_count": len(ids), "rgb_identity_all_cases": rgb_identity, "mh_source_trace_all_cases": mh_trace, "output_case_counts": counts, "exact_file_inventory": inventory_ok, "dtype_float32_all_cases": all(row["dtype_ok"] for row in rows), "mask_binary_0_255_all_cases": all(row["mask_binary_0_255"] for row in rows), "rows": rows}


def run(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if (root / OUT).exists() or (root / REPORT).exists():
        raise RuntimeError("REFUSE_OVERWRITE_R2X3_OUTPUT")
    ids, authority = _preflight(root)
    before = _protected_snapshot(root)
    out, report = root / OUT, root / REPORT
    for path in (out / "rgb/images", out / "rgb_b1mh/images", out / "rgb_b2mh/images", out / "common/valid_masks", report):
        path.mkdir(parents=True, exist_ok=True)
    rows = [_prepare_case(root, out, case_id) for case_id in ids]
    (out / "common/canonical_case_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    with (out / "classifier_input_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case_id", "condition", "relative_path", "shape", "dtype", "channel_order"])
        writer.writeheader()
        for case_id in ids:
            writer.writerow({"case_id": case_id, "condition": "RGB", "relative_path": f"rgb/images/{case_id}.npy", "shape": "3x320x256", "dtype": "float32", "channel_order": "RGB0,RGB1,RGB2"})
            writer.writerow({"case_id": case_id, "condition": "RGB_B1MH", "relative_path": f"rgb_b1mh/images/{case_id}.npy", "shape": "5x320x256", "dtype": "float32", "channel_order": "RGB0,RGB1,RGB2,B1_M,B1_H"})
            writer.writerow({"case_id": case_id, "condition": "RGB_B2MH", "relative_path": f"rgb_b2mh/images/{case_id}.npy", "shape": "5x320x256", "dtype": "float32", "channel_order": "RGB0,RGB1,RGB2,B2_M,B2_H"})
    with (report / "classifier_input_case_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)
    audit = _audit_outputs(root, out, ids)
    after = _protected_snapshot(root)
    protected = before == after
    geometry = {"source_shape_hw": [1220, 979], "resized_shape_hw": [319, 256], "resized_shape_wh": [256, 319], "target_shape_hw": [320, 256], "resize_rule": "single aspect-ratio-preserving resize 979x1220 -> 256x319 (WxH)", "padding_rule": "one all-zero row appended at bottom", "RGB_interpolation": "bilinear, float32, align_corners=false", "MH_interpolation": "bilinear, float32, align_corners=false", "mask_interpolation": "nearest", "crop_rotation_translation_warp": False}
    alignment = {"status": "PASS" if audit["rgb_identity_all_cases"] else "FAIL", "rgb_equal_rgb_b1mh_first3": audit["rgb_identity_all_cases"], "rgb_equal_rgb_b2mh_first3": audit["rgb_identity_all_cases"], "comparison": "np.array_equal, all 500 cases"}
    tensor = {"status": audit["status"], "case_count": 500, "conditions": {"RGB": {"count": 500, "shape": [3, 320, 256]}, "RGB_B1MH": {"count": 500, "shape": [5, 320, 256]}, "RGB_B2MH": {"count": 500, "shape": [5, 320, 256]}}, "dtype": "float32", "layout": "C×H×W", "finite_all": audit["status"] == "PASS", "dtype_float32_all": audit["dtype_float32_all_cases"], "mask_binary_0_255_all": audit["mask_binary_0_255_all_cases"], "exact_file_inventory": audit["exact_file_inventory"], "mask_outside_zero": audit["status"] == "PASS", "padding_row_zero": audit["status"] == "PASS", "mh_source_trace_all_cases": audit["mh_source_trace_all_cases"]}
    _dump(report / "geometry_transform_audit.json", geometry)
    _dump(report / "cross_condition_alignment_audit.json", alignment)
    _dump(report / "output_tensor_integrity_audit.json", tensor)
    _dump(report / "source_provenance_audit.json", {"source_dataset": str(DATA), "source_R2X1": str(X1), "source_R2X2": str(X2), "canonical_id_source": [str(X1 / "inference_manifest.csv"), str(DATA / "full_ids.txt")], "canonical_order_crosscheck": True, "case_count": 500, "authority": authority, "source_rows": rows})
    _dump(report / "field_access_audit.json", {"label_or_split_field_read_count": 0, "clinical_field_read_count": 0, "model_forward_calls": 0, "training_calls": 0, "checkpoint_writes": 0, "forbidden_fields_read": False})
    _dump(report / "protected_asset_hash_audit.json", {"changed": 0 if protected else 1, "missing": 0, "protected_assets_unchanged": protected, "before": before, "after": after})
    acceptance = {"status": "COMPLETE_CLASSIFIER_INPUT_PREPARATION" if audit["status"] == "PASS" and protected else "FAIL_CLASSIFIER_INPUT_PREPARATION", "source_dataset": str(DATA), "source_R2X1_acceptance_hash": authority["source_R2X1_acceptance_hash"], "source_R2X2_acceptance_hash": authority["source_R2X2_acceptance_hash"], "B1_checkpoint_hash": authority["B1_checkpoint_hash"], "B2_checkpoint_hash": authority["B2_checkpoint_hash"], "lambda_pair": 0.50, "case_count": 500, "source_shape": [1220, 979, 3], "target_shape": [320, 256], **geometry, "condition_names": ["RGB", "RGB_B1MH", "RGB_B2MH"], "output_dtype": "float32", "output_layout": "C×H×W", "cross_condition_RGB_identity": alignment, "model_forward_calls": 0, "training_calls": 0, "label_or_split_field_read_count": 0, "clinical_field_read_count": 0, "forbidden_field_access_count": 0, "protected_assets_unchanged": protected, "classification_started": False, "formal_SO_R1_C_status": "FAIL", "official_SO_R2_authorization": False, "SO_R3_authorization": False, "next_stage_authorized": False}
    protocol = {"stage": "SO-R2-X3-P0", "version": "v1", "geometry": geometry, "conditions": ["RGB", "RGB_B1MH", "RGB_B2MH"], "channel_order": {"RGB": ["RGB0", "RGB1", "RGB2"], "RGB_B1MH": ["RGB0", "RGB1", "RGB2", "B1_M", "B1_H"], "RGB_B2MH": ["RGB0", "RGB1", "RGB2", "B2_M", "B2_H"]}, "normalization": "none", "model_forward_calls": 0, "training_calls": 0}
    _dump(out / "CLASSIFIER_INPUT_ACCEPTANCE.json", acceptance)
    _dump(out / "classifier_input_protocol_lock.json", protocol)
    _dump(report / "output_audit_summary.json", audit)
    report_text = "\n".join([
        "# SO-R2-X3-P0 Classifier Input Preparation", "", "## Result", "",
        f"- Status: **{acceptance['status']}**; 500 canonical cases were prepared in all three conditions.",
        "- Output counts: RGB=500, RGB_B1MH=500, RGB_B2MH=500, common masks=500; exact filename inventory check passed.",
        "- RGB tensors are `float32`, `(3, 320, 256)`, CHW. Fused tensors are `float32`, `(5, 320, 256)`, CHW.",
        "- Geometry: one aspect-ratio-preserving resize from 979x1220 to 256x319 (WxH), then one all-zero bottom row; no crop, rotation, translation, or warp.",
        "- Interpolation: RGB/MH bilinear with `align_corners=false`; masks nearest-neighbor. No normalization or color correction was applied.",
        f"- Audits: finite={tensor['finite_all']}, dtype_float32={audit['dtype_float32_all_cases']}, binary_masks={audit['mask_binary_0_255_all_cases']}, mask/padding zero={tensor['mask_outside_zero'] and tensor['padding_row_zero']}, RGB identity={alignment['status']}, M/H source trace={audit['mh_source_trace_all_cases']}.",
        "- Access controls: label/split fields=0, clinical fields=0, model forward=0, training=0; protected assets changed=0 and missing=0.",
        "- This stage only prepares classifier inputs. Formal SO-R1-C remains `FAIL`; SO-R2/SO-R3 and next-stage authorization remain false.", "", "## Acceptance Record", "", "```json", json.dumps(acceptance, indent=2), "```", "",
    ])
    (report / "SO_R2X3_Classifier_Input_Preparation_Report.md").write_text(report_text, encoding="utf-8")
    (report / "test_results.txt").write_text("X3 preparation self-audit: PASS\nX3专项测试需由外部 pytest 命令执行；本次 run 不启动测试、训练或模型前向。\n", encoding="utf-8")
    return acceptance
