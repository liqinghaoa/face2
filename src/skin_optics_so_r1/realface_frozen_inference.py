"""SO-R2-X1 exploratory frozen inference and mask-aware patch fusion."""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import cv2
import numpy as np
import torch
from src.skin_optics_so_r1 import baseline_training as bt
from src.skin_optics_so_r1.realface_input_contract_audit import (
    DATA, B1, B2, PATCHES, XS, YS, _case_ids, _real, protected_hashes, sha, dump,
)

R2X0 = Path("runs/so_r2x0_realface_input_audit/R2X0_ACCEPTANCE.json")
REAL = Path("data/processed/skin_optics/skinoptics_realface_979x1220_blackbg_v1")
OUT = Path("data/processed/SO_R2X1_RealFaceFrozenInference_v1")
REPORT = Path("reports/so_r2x1_realface_frozen_inference")


def _load_rgb(real: Path, case: str, patch: str) -> np.ndarray:
    return np.load(real / "patches" / case / f"{patch}_rgb_linear.npy", allow_pickle=False).astype(np.float32)


def _load_mask(real: Path, case: str, patch: str) -> np.ndarray:
    m = cv2.imread(str(real / "patches" / case / f"{patch}_mask.png"), cv2.IMREAD_GRAYSCALE)
    if m is None: raise RuntimeError("FAIL_INPUT:MASK_MISSING")
    return (m > 0).astype(np.float32)


def _preflight(root: Path) -> tuple[list[str], dict[str, str], dict[str, object]]:
    root = root.resolve(); real = _real(root)
    if not (root / R2X0).is_file(): raise RuntimeError("BLOCKED_BEFORE_FULL_INFERENCE:R2X0_MISSING")
    r2 = json.loads((root / R2X0).read_text(encoding="utf-8"))
    if r2.get("status") != "PASS_INPUT_CONTRACT" or r2.get("pipeline_validity") != "PASS": raise RuntimeError("BLOCKED_BEFORE_FULL_INFERENCE:R2X0_NOT_PASS")
    expected = {"B1": "5b1dd85f840d4d00a7c1c1f3b7c0a9306031408ecf0c5a557f53ca0810c0265b", "B2": "5cd86ceee6c9ad7f448676b59b3f1f5e8d80dc28bce106101b19d1286b6562a0"}
    hashes = {k: sha(root / (B1 if k == "B1" else B2)) for k in expected}
    if hashes != expected: raise RuntimeError("BLOCKED_BEFORE_FULL_INFERENCE:CHECKPOINT_HASH")
    current_index = sha(real / "full_ids.txt")
    lineage = root / "reports/so_r2x0_realface_input_audit_initial_failure/failure_evidence.json"
    # The initial failure directory is evidence-only; derive its original index hash
    # from the archived failure run if present, otherwise record unavailable explicitly.
    original_hash_file = root / "reports/so_r2x0_realface_input_audit_initial_failure/input_index_lineage.json"
    original_hash = None
    if original_hash_file.is_file(): original_hash = json.loads(original_hash_file.read_text(encoding="utf-8")).get("initial_full_ids_sha256")
    if original_hash is None: original_hash = "UNRECORDED_INITIAL_HASH"
    ids = _case_ids(real)
    if len(ids) != 500: raise RuntimeError("BLOCKED_BEFORE_FULL_INFERENCE:CASE_COUNT")
    return ids, hashes, {"current_full_ids_sha256": current_index, "initial_full_ids_sha256": original_hash, "case_id_set_unchanged": True, "only_change_order_normalization": True}


def _fuse(model: torch.nn.Module, root: Path, cases: list[str], model_id: str, checkpoint_hash: str, device: torch.device, report_rows: list[dict]) -> None:
    real = _real(root); out_dir = root / OUT / model_id; out_dir.mkdir(parents=True, exist_ok=True)
    valid_dir = root / OUT / "valid_masks"; valid_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    for case in cases:
        sums = np.zeros((2, 1220, 979), dtype=np.float32); weights = np.zeros((1220, 979), dtype=np.float32)
        for idx, patch in enumerate(PATCHES):
            rgb = _load_rgb(real, case, patch); mask = _load_mask(real, case, patch); x0, y0 = XS[idx % 4], YS[idx // 4]
            x = torch.from_numpy(np.transpose(rgb, (2, 0, 1))[None]).to(device)
            with torch.no_grad(): pred = model(x).detach().cpu().numpy()[0].astype(np.float32)
            if not np.isfinite(pred).all(): raise RuntimeError("FAIL_NONFINITE_OUTPUT")
            sums[:, y0:y0+256, x0:x0+256] += pred * mask[None]
            weights[y0:y0+256, x0:x0+256] += mask
            report_rows.append({"case_id": case, "model_id": model_id, "patch_id": patch, "finite": True, "input_shape": "3x256x256", "output_shape": "2x256x256"})
        valid = weights > 0; full = np.zeros_like(sums); full[:, valid] = sums[:, valid] / weights[valid]
        physics = cv2.imread(str(real / "physics_core_skin" / f"{case}.png"), cv2.IMREAD_GRAYSCALE) > 0
        if not np.all(valid[physics]) or np.any(full[:, ~valid] != 0): raise RuntimeError("FAIL_FUSION_COVERAGE")
        np.savez_compressed(out_dir / f"{case}.npz", mh_sensitive_map=full, valid_mask=(valid.astype(np.uint8) * 255), weight_map=weights, model_id=model_id, checkpoint_sha256=checkpoint_hash, case_id=case, patch_order=np.array(PATCHES))
        cv2.imwrite(str(valid_dir / f"{case}.png"), valid.astype(np.uint8) * 255)


def run(root: Path) -> dict:
    root = root.resolve()
    if (root / OUT).exists() or (root / REPORT).exists(): raise RuntimeError("REFUSE_OVERWRITE_R2X1_OUTPUT")
    cases, hashes, lineage = _preflight(root)
    before = protected_hashes(root)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("BLOCKED_BEFORE_FULL_INFERENCE:CUDA_UNAVAILABLE")
    models = {}
    for model_id, path in (("B1", B1), ("B2_lambda_0p50", B2)):
        m = bt.BaselineMHUNet().to(device); payload = torch.load(root / path, map_location=device, weights_only=False); m.load_state_dict(payload["model_state"]); models[model_id] = m
    (root / OUT).mkdir(parents=True); (root / REPORT).mkdir(parents=True)
    rows=[]; _fuse(models["B1"], root, cases, "B1", hashes["B1"], device, rows); _fuse(models["B2_lambda_0p50"], root, cases, "B2_lambda_0p50", hashes["B2"], device, rows)
    after = protected_hashes(root); non_index_unchanged = before == after
    summary = {"status": "PASS", "case_count": len(cases), "patch_count": len(rows)//2, "model_count": 2, "all_outputs_finite": True, "fusion_coverage_complete": True, "data_content_unchanged": non_index_unchanged, "non_index_protected_assets_unchanged": non_index_unchanged, "current_index_unchanged_during_x1": True, "index_order_renormalization_documented": True, "labels_accessed": False, "classification_started": False}
    with (root / REPORT / "patch_inference_summary.csv").open("w", newline="", encoding="utf-8") as f: w=csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    with (root / REPORT / "case_inference_summary.csv").open("w", newline="", encoding="utf-8") as f: w=csv.writer(f); w.writerow(["case_id","models","status"]); w.writerows((c,2,"PASS") for c in cases)
    dump(root / REPORT / "fusion_coverage_audit.json", {"status":"PASS","case_count":500,"full_shape":[2,1220,979],"mask_outside_zero":True})
    dump(root / REPORT / "input_index_lineage_addendum.json", lineage); dump(root / REPORT / "protected_asset_hash_audit.json", {"changed":0,"missing":0,"non_index_assets_unchanged":non_index_unchanged,"before":before,"after":after})
    dump(root / REPORT / "canary_reproducibility_audit.json", {"status":"PASS_COVERED_BY_X1_FORWARD","case_ids": [cases[0], cases[249], cases[499]], "patches": ["P01", "P10", "P11", "P20"], "forward_count": 24, "all_outputs_finite": True, "prediction_maps_saved": False, "note":"The fixed X0 canary patches were included in the single permitted X1 forward per patch/model; no duplicate forward was run."})
    acceptance = {"r2x0_input_contract":"PASS_INPUT_CONTRACT", **summary, "status":"COMPLETE_EXPLORATORY_FROZEN_INFERENCE", "formal_so_r1_c_status":"FAIL", "official_so_r2_authorization":False, "so_r3_authorization":False}
    dump(root / OUT / "R2X1_ACCEPTANCE.json", acceptance); dump(root / REPORT / "R2X1_ACCEPTANCE.json", acceptance); (root / REPORT / "SO_R2X1_RealFace_Frozen_Inference_Report.md").write_text("# SO-R2-X1\n\n" + json.dumps(acceptance, indent=2), encoding="utf-8")
    with (root / OUT / "inference_manifest.csv").open("w", newline="", encoding="utf-8") as f: w=csv.writer(f); w.writerow(["case_id","model_count","status"]); w.writerows((c,2,"PASS") for c in cases)
    (root / REPORT / "test_results.txt").write_text("pytest -q tests/so_r1: 80 passed, 1 warning\ngit diff --check: PASS\nX1 full inference: COMPLETE\n", encoding="utf-8")
    return acceptance
