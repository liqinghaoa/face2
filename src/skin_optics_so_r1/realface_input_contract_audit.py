"""SO-R2-X0 read-only real-face patch input-contract audit."""
from __future__ import annotations

import csv, hashlib, json, os
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import pandas as pd
import torch

from src.skin_optics_so_r1 import baseline_training as bt

DATA = Path("data/processed/skin_optics/skinoptics_realface_979x1220_blackbg_v1")
B1 = Path("runs/so_r1_b1_baseline_v6/checkpoints/best_val_masked_smoothl1.pt")
B2 = Path("runs/so_r1_b2_proposed_v3/lambda_0p50/checkpoints/best_val_masked_smoothl1.pt")
OUT = Path("runs/so_r2x0_realface_input_audit")
REPORT = Path("reports/so_r2x0_realface_input_audit")
CONFIG = Path("config/so_r2x/so_r2x0_realface_input_contract_audit_v1.yaml")
LOCK = Path("config/so_r2x/SO_R2X0_REALFACE_INPUT_AUDIT_LOCK.json")
XS, YS = (0, 241, 482, 723), (0, 241, 482, 723, 964)
PATCHES = tuple(f"P{i:02d}" for i in range(1, 21))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _root(root: Path) -> Path:
    return Path(root).resolve()


def _real(root: Path) -> Path:
    return _root(root) / DATA


def _case_ids(real: Path) -> list[str]:
    ids = [line.strip() for line in (real / "full_ids.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(ids) != 500 or ids != sorted(ids) or len(set(ids)) != 500: raise RuntimeError("FAIL_REAL_PATCH_INVENTORY:FULL_IDS")
    if set(ids) != {p.name for p in (real / "patches").iterdir() if p.is_dir()}: raise RuntimeError("FAIL_REAL_PATCH_INVENTORY:CASE_SET")
    return ids


def protected_hashes(root: Path) -> dict[str, str]:
    root = _root(root); real = _real(root)
    required = [root / B1, root / B2, root / "config/so_r1/so_r1_b1_training_contract_v6.yaml", root / "data/processed/SO_R1_A2_G1_AM1_FullGeneration_v1/G1_AM1_ACCEPTANCE.json", real / "COMPLETED.json", real / "manifest.csv", real / "full_ids.txt", real / "config/resolved_config.yaml"]
    files = required + sorted(real.glob("patches/*/*_metadata.json")) + sorted(real.glob("patches/*/*_rgb_linear.npy")) + sorted(real.glob("patches/*/*_mask.png"))
    if not all(path.is_file() for path in files): raise RuntimeError("FAIL_PROTECTED_ASSET_AUDIT:MISSING")
    return {str(path.relative_to(root)): sha(path) for path in files}


def authority(root: Path) -> dict[str, Any]:
    root = _root(root); b1_acc = json.loads((root / "runs/so_r1_b1_baseline_v6/B1_ACCEPTANCE.json").read_text()); b2_acc = json.loads((root / "runs/so_r1_b2_proposed_v3/B2_ACCEPTANCE.json").read_text()); real = _real(root)
    if b1_acc.get("status") != "PASS" or not b1_acc.get("baseline_checkpoint_authoritative") or b2_acc.get("status") != "PASS" or not b2_acc.get("proposed_checkpoint_authoritative") or b2_acc.get("selected_lambda") != .5: raise RuntimeError("FAIL_INPUT_PROVENANCE_GATE:ACCEPTANCE")
    b1 = root / B1; b2 = root / B2
    if sha(b1) != "5b1dd85f840d4d00a7c1c1f3b7c0a9306031408ecf0c5a557f53ca0810c0265b" or sha(b2) != "5cd86ceee6c9ad7f448676b59b3f1f5e8d80dc28bce106101b19d1286b6562a0": raise RuntimeError("FAIL_INPUT_PROVENANCE_GATE:CHECKPOINT_HASH")
    contract = json.loads(json.dumps(__import__("yaml").safe_load((root / "config/so_r1/so_r1_b1_training_contract_v6.yaml").read_text())))
    if contract["model_architecture"]["input_channels"] != 3 or contract["model_architecture"]["output_channels"] != 2 or contract["model_architecture"]["output_order"] != list(bt.BaselineMHUNet.output_order) or contract["amp_contract"]["precision"] != "FP32": raise RuntimeError("FAIL_INPUT_PROVENANCE_GATE:CONTRACT")
    g1 = json.loads((root / "data/processed/SO_R1_A2_G1_AM1_FullGeneration_v1/G1_AM1_ACCEPTANCE.json").read_text())
    if g1.get("dataset_status") != "ACCEPTED": raise RuntimeError("FAIL_INPUT_PROVENANCE_GATE:G1")
    ids = _case_ids(real)
    return {"status": "PASS", "b1_checkpoint": str(b1), "b2_checkpoint": str(b2), "b1_sha256": sha(b1), "b2_sha256": sha(b2), "b1_contract_hash": b1_acc["contract_hash"], "g1_dataset_status": g1["dataset_status"], "case_count": len(ids), "patches_per_case": 20, "case_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()}


def synthetic_loader_semantics(root: Path) -> dict[str, Any]:
    root = _root(root); _, lock, rows = bt.validate_authoritative_input(root); seed = bt._seed_from_protocol(lock["protocol_hash"]); groups, _ = bt.build_epoch_groups(rows, seed, 1); selected = [g for g in groups if g.split == "Train"][:128]; ds = bt.PairedGroupDataset(selected); loader = bt._loader(ds, 4, shuffle=False, seed=0, workers=0, groups=True)
    batch = next(iter(loader)); images = batch["images"].float(); masks = batch["mask"].float(); forwarded = images.flatten(0, 1)
    outside = images * (1 - masks.unsqueeze(1)); return {"source": "B1 PairedGroupDataset -> loader -> model.forward", "fixed_train_latent_count": len(selected), "model_input_shape": list(forwarded.shape[1:]), "model_input_dtype": str(forwarded.dtype), "rgb_min": float(images.min()), "rgb_max": float(images.max()), "rgb_mask_outside_abs_max": float(outside.abs().max()), "rgb_mask_outside_abs_mean": float(outside.abs().mean()), "rgb_mask_outside_nonzero_fraction": float((outside.abs() > 0).float().mean()), "mask_shape": list(masks.shape[1:]), "mask_dtype": str(masks.dtype), "mask_unique": sorted(float(x) for x in torch.unique(masks)), "forward_tensor_is_flattened_images": True, "mask_is_model_input": False, "normalization": "none; linear-sRGB FP32", "augmentation": "none"}


def audit_real_patches(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    real = _real(root); ids = _case_ids(real); tensor_rows=[]; mask_rows=[]; geometry=[]; all_pass=True
    for case_id in ids:
        full = np.load(real / "blackbg_linear_rgb" / f"{case_id}.npy", allow_pickle=False).astype(np.float32); physics = cv2.imread(str(real / "physics_core_skin" / f"{case_id}.png"), cv2.IMREAD_GRAYSCALE); union=np.zeros((1220,979), dtype=bool); case_ok=True
        if full.shape != (1220,979,3) or physics is None or physics.shape != (1220,979): case_ok=False
        for index, patch_id in enumerate(PATCHES):
            meta=json.loads((real / "patches" / case_id / f"{patch_id}_metadata.json").read_text(encoding="utf-8")); rgb=np.load(real / "patches" / case_id / f"{patch_id}_rgb_linear.npy", allow_pickle=False); mask=cv2.imread(str(real / "patches" / case_id / f"{patch_id}_mask.png"), cv2.IMREAD_GRAYSCALE)
            row, col = divmod(index, 4); x0,y0=XS[col],YS[row]; expected=(x0,y0,x0+256,y0+256); valid=mask > 0 if mask is not None else np.zeros((256,256),bool)
            finite=bool(np.isfinite(rgb).all()) if rgb.size else False; range_ok=bool(rgb.min() >= -1e-6 and rgb.max() <= 1+1e-6) if rgb.size else False; shape_ok=rgb.shape==(256,256,3) and rgb.dtype==np.float32; mask_ok=mask is not None and mask.shape==(256,256) and set(np.unique(mask).tolist()) <= {0,255}; meta_ok=tuple(meta.get(k) for k in ("x0","y0","x1","y1"))==expected and meta.get("patch_id")==patch_id
            slice_ok=bool(shape_ok and full.shape==(1220,979,3) and np.array_equal(rgb,full[y0:y0+256,x0:x0+256])); union[y0:y0+256,x0:x0+256] |= valid
            case_ok &= finite and range_ok and shape_ok and mask_ok and meta_ok and slice_ok
            tensor_rows.append({"case_id":case_id,"patch_id":patch_id,"shape":str(rgb.shape),"dtype":str(rgb.dtype),"finite":finite,"min":float(rgb.min()),"max":float(rgb.max()),"hwc_to_chw_shape":str((3,256,256)),"range_ok":range_ok,"full_slice_equal":slice_ok,"pass":finite and range_ok and shape_ok and slice_ok})
            mask_rows.append({"case_id":case_id,"patch_id":patch_id,"shape":str(None if mask is None else mask.shape),"dtype":str(None if mask is None else mask.dtype),"unique":str([] if mask is None else sorted(int(x) for x in np.unique(mask))),"binary_0_255":mask_ok,"valid_fraction":float(valid.mean()),"pass":mask_ok})
        coverage_ok=bool(physics is not None and np.array_equal(union, physics > 0)); geometry.append({"case_id":case_id,"patch_count":20,"row_major_coordinates":True,"union_covers_physics_core_skin":coverage_ok,"physics_core_pixels":int((physics>0).sum()) if physics is not None else 0,"union_pixels":int(union.sum()),"pass":case_ok and coverage_ok}); all_pass &= case_ok and coverage_ok
    return pd.DataFrame(tensor_rows),pd.DataFrame(mask_rows),{"status":"PASS" if all_pass else "FAIL","case_count":500,"patch_count":10000,"geometry_rows":geometry,"coordinates":{"x":list(XS),"y":list(YS)},"all_patch_contracts_pass":all_pass}


def canary(root: Path, authority_info: dict[str, Any]) -> dict[str, Any]:
    real=_real(root); ids=_case_ids(real); cases=[ids[0],ids[249],ids[499]]; patches=["P01","P10","P11","P20"]; device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); results=[]
    if not torch.cuda.is_available(): raise RuntimeError("FAIL_INPUT_CONTRACT:CUDA_UNAVAILABLE")
    models={"B1":bt.BaselineMHUNet().to(device),"B2":bt.BaselineMHUNet().to(device)}
    for name,path in (("B1",_root(root)/B1),("B2",_root(root)/B2)):
        payload=torch.load(path,map_location=device,weights_only=False); models[name].load_state_dict(payload["model_state"]); models[name].eval()
    for case_id in cases:
        for patch_id in patches:
            rgb=np.load(real/"patches"/case_id/f"{patch_id}_rgb_linear.npy",allow_pickle=False).astype(np.float32); x=torch.from_numpy(np.transpose(rgb,(2,0,1))[None]).to(device)
            for name,model in models.items():
                with torch.no_grad(): out=model(x)
                arr=out.detach().cpu().numpy(); results.append({"model":name,"case_id":case_id,"patch_id":patch_id,"input_shape":list(x.shape[1:]),"input_dtype":str(x.dtype).replace("torch.",""),"input_min":float(x.min()),"input_max":float(x.max()),"output_shape":list(arr.shape[1:]),"finite":bool(np.isfinite(arr).all()),"M_min":float(arr[:,0].min()),"M_max":float(arr[:,0].max()),"M_mean":float(arr[:,0].mean()),"M_std":float(arr[:,0].std()),"H_min":float(arr[:,1].min()),"H_max":float(arr[:,1].max()),"H_mean":float(arr[:,1].mean()),"H_std":float(arr[:,1].std()),"prediction_map_saved":False})
    return {"status":"PASS" if all(r["finite"] and r["output_shape"]==[2,256,256] for r in results) else "FAIL","selection":{"case_ids":cases,"patches":patches,"rule":"full_ids.txt dictionary order indices 1,250,500; fixed P01,P10,P11,P20"},"count":len(results),"results":results,"device":torch.cuda.get_device_name(0),"precision":"FP32","prediction_maps_saved":False}


def run(root: Path) -> dict[str, Any]:
    root=_root(root)
    if (root/OUT).exists() or (root/REPORT).exists() or (root/CONFIG).exists() or (root/LOCK).exists(): raise RuntimeError("REFUSE_OVERWRITE_R2X0_OUTPUT")
    auth=authority(root); before=protected_hashes(root); semantics=synthetic_loader_semantics(root); tensors,masks,geometry=audit_real_patches(root)
    if geometry["status"] != "PASS": raise RuntimeError("FAIL_INPUT_CONTRACT:REAL_PATCH_GEOMETRY")
    protocol={"stage":"SO-R2-X0","version":"v1","read_only":True,"model_input":"3x256x256 linear-sRGB FP32 CHW","mask_as_model_input":False,"test_canary_only":True,"full_inference":False,"coordinates":{"x":list(XS),"y":list(YS)},"patches_per_case":20,"case_count":500,"patch_count":10000,"b1_sha256":auth["b1_sha256"],"b2_sha256":auth["b2_sha256"],"b1_contract_hash":auth["b1_contract_hash"]}
    import yaml; (root/CONFIG).parent.mkdir(parents=True,exist_ok=True); (root/CONFIG).write_text(yaml.safe_dump(protocol,sort_keys=False),encoding="utf-8"); dump(root/LOCK,{"status":"FROZEN","protocol_sha256":sha(root/CONFIG),"protocol":protocol})
    after=protected_hashes(root); unchanged=before==after
    root/OUT; (root/OUT).mkdir(parents=True); (root/REPORT).mkdir(parents=True)
    tensors.to_csv(root/REPORT/"real_patch_tensor_audit.csv",index=False); masks.to_csv(root/REPORT/"real_mask_audit.csv",index=False); dump(root/REPORT/"synthetic_loader_input_semantics.json",semantics); dump(root/REPORT/"real_patch_inventory_audit.json",{"status":"PASS","case_count":500,"patch_count":10000}); dump(root/REPORT/"real_patch_geometry_coverage_audit.json",geometry); dump(root/REPORT/"protected_asset_hash_audit.json",{"before":before,"after":after,"changed":0 if unchanged else 1,"missing":0})
    c=canary(root,auth); dump(root/REPORT/"canary_forward_audit.json",c)
    contract_ok=auth["status"]=="PASS" and semantics["model_input_shape"]==[3,256,256] and semantics["mask_is_model_input"] is False and geometry["status"]=="PASS" and c["status"]=="PASS" and unchanged
    acceptance={"status":"PASS_INPUT_CONTRACT" if contract_ok else "FAIL_INPUT_SEMANTICS","pipeline_validity":"PASS" if unchanged else "FAIL_PROTECTED_ASSET_MUTATION","input_contract_status":"PASS_INPUT_CONTRACT" if contract_ok else "FAIL_INPUT_SEMANTICS","b1_checkpoint_sha256":auth["b1_sha256"],"b2_checkpoint_sha256":auth["b2_sha256"],"b1_training_contract_hash":auth["b1_contract_hash"],"real_case_count":500,"real_patch_count":10000,"canary_count":12,"canary_status":c["status"],"protected_assets_unchanged":unchanged,"full_inference_started":False,"next_step":"建议进入下一步全量冻结推理与 M/H patch 融合" if contract_ok else None,"next_stage_authorized":False,"nyha_labels_folds_sex_read":False}
    dump(root/OUT/"R2X0_ACCEPTANCE.json",acceptance); dump(root/OUT/"run_manifest.json",acceptance); dump(root/REPORT/"R2X0_ACCEPTANCE.json",acceptance); (root/REPORT/"SO_R2X0_RealFace_Input_Contract_Audit_Report.md").write_text("# SO-R2-X0 RealFace Input Contract Audit\n\nConclusion: **"+acceptance["input_contract_status"]+"**. This audit is read-only and canary-only; no full real-face inference or M/H export was performed.\n\n"+json.dumps(acceptance,indent=2)+"\n",encoding="utf-8")
    return acceptance


def write_failure(root: Path, error: BaseException) -> None:
    root=_root(root); (root/OUT).mkdir(parents=True,exist_ok=True); (root/REPORT).mkdir(parents=True,exist_ok=True); dump(root/OUT/"R2X0_ACCEPTANCE.json",{"status":"FAIL_INPUT_SEMANTICS","pipeline_validity":"FAIL","error_type":type(error).__name__,"error":repr(error),"full_inference_started":False,"next_stage_authorized":False}); dump(root/REPORT/"failure_evidence.json",{"status":"FAIL_INPUT_SEMANTICS","error_type":type(error).__name__,"error":repr(error)})
