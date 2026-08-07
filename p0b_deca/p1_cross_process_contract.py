"""Versioned P0B cross-process regression contract (v2).

Only run_01/run_02/run_03 calibrate thresholds.  A separately launched run_04
is held out for validation and can never alter a generated contract.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .p1_generation import _environment, _read_any, load_config, sha
from .p1_reproducibility_audit import ALPHA_THRESHOLD, LATENTS, MAPS, RUN_NAMES, _map_metrics, _pair_metrics, _regions, _run_environment, _load_run, audit_root, worker, write_manifest

VERSION = "p0b_cross_process_v2"
PRESETS = ("neutral_front", "left", "right", "top", "dim_front", "bright_front")


def base(root: Path) -> Path: return root / "data/processed/P0B_cross_process_regression_baseline_v2"
def _json(path: Path) -> dict[str, Any]: return json.loads(path.read_text(encoding="utf-8"))
def _write(path: Path, value: Any) -> None: path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2), encoding="utf-8")
def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fields = sorted({k for r in rows for k in r}) or ["metric"]
    with path.open("w", newline="", encoding="utf-8") as h: w=csv.DictWriter(h, fieldnames=fields); w.writeheader(); w.writerows(rows)
def _paths(root: Path, run: str) -> Path: return audit_root(root, load_config(root / "config/p1/p1_deca_full500_v1.yaml", root)) / run
def _case_rows(root: Path, cfg: dict[str, Any]) -> list[dict[str, str]]:
    path=root/cfg["output_root"] / "metadata/reproducibility_audit/pilot12_case_manifest.csv"
    rows=list(csv.DictReader(path.open(newline="",encoding="utf-8")))
    if len(rows)!=12 or {r["fixed_input_mode"] for r in rows}!={"direct_p0_aligned"}: raise ValueError("fixed Pilot12 manifest required")
    return rows
def _q(a: np.ndarray) -> np.ndarray: return np.rint(np.clip(a,0,1)*255).astype(np.uint8)
def _finite(v: float) -> bool: return math.isfinite(float(v))


def validate_inputs(root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    rows=_case_rows(root,cfg)
    for run in RUN_NAMES:
        p=_paths(root,run)
        if not (p/"metadata/summary.json").is_file() or _json(p/"metadata/summary.json")["success_cases"] != 12: raise ValueError(f"incomplete calibration run: {run}")
        for r in rows:
            for name in ("latents.npz","maps.npz","relighting.npz","diagnostics.json"):
                if not (p/"cases"/r["case_id"]/name).is_file(): raise FileNotFoundError(name)
    return {"status":"passed","calibration_runs":list(RUN_NAMES),"holdout_excluded":True,"case_count":12,"full_500_started":False}


def _values_for_pair(root: Path, cfg: dict[str, Any], left: str, right: str) -> list[dict[str, Any]]:
    rows=[]
    for row in _case_rows(root,cfg):
        cid=row["case_id"]; la,ma,ra=_load_run(_paths(root,left).parent,left,cid); lb,mb,rb=_load_run(_paths(root,right).parent,right,cid)
        for key in LATENTS:
            p=_pair_metrics(la[key],lb[key])
            for metric in ("max_abs","MAE","RMSE","relative_L2"):
                rows.append({"case_id":cid,"category":"latent","output":key,"metric":metric,"value":p[metric]})
            rows.append({"case_id":cid,"category":"latent","output":key,"metric":"cosine_distance","value":1-p["cosine_similarity"]})
        p0=Path(row["physics_core_skin_path"]); f0=Path(row["face_valid_path"]); face=_read_any(f0)>0; physics=_read_any(p0)>0; regions=_regions(ma["alpha"],mb["alpha"],face,physics)
        for output in ("albedo_like","normal_coarse"):
            keys=[("physics_core_skin","physics_core"),("alpha_intersection_erode_3","eroded_alpha_3")]
            for region,prefix in keys:
                m=_map_metrics(ma[output],mb[output],regions[region],normal=output=="normal_coarse")
                for metric in ("MAE","p99_abs","SSIM"):
                    value=1-m[metric] if metric=="SSIM" else m[metric]
                    rows.append({"case_id":cid,"category":output,"output":output,"metric":f"{prefix}_{metric if metric!='SSIM' else 'SSIM_difference'}","value":value})
                if output=="normal_coarse":
                    for metric in ("mean_angular_error_deg","p99_angular_error_deg"):
                        rows.append({"case_id":cid,"category":output,"output":output,"metric":f"{prefix}_{metric}","value":m[metric]})
            # Full-image maximum remains diagnostic and is deliberately omitted.
        names_a=[str(x) for x in ra["preset_names"].tolist()]; names_b=[str(x) for x in rb["preset_names"].tolist()]
        if names_a != list(PRESETS) or names_b != list(PRESETS): raise ValueError("fixed six-preset contract violated")
        for i,preset in enumerate(PRESETS):
            for region,mask in (("full_image",np.ones(ma["alpha"].shape,bool)),("physics_core",physics),("eroded_alpha_3",regions["alpha_intersection_erode_3"])):
                m=_map_metrics(ra["relighted_images"][i],rb["relighted_images"][i],mask)
                for metric in ("MAE","SSIM"):
                    rows.append({"case_id":cid,"category":"relighting","output":preset,"metric":f"{region}_{metric if metric!='SSIM' else 'SSIM_difference'}","value":1-m[metric] if metric=="SSIM" else m[metric]})
            d=np.abs(_q(ra["relighted_images"][i]).astype(np.int16)-_q(rb["relighted_images"][i]).astype(np.int16))
            for level in (1,2,3): rows.append({"case_id":cid,"category":"relighting","output":preset,"metric":f"fraction_gt_{level}_gray_levels","value":float((d>level).mean())})
    return rows


def _thresholds(values: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    result=[]; index={}
    groups={}
    for r in values: groups.setdefault((r["category"],r["output"],r["metric"]),[]).append(float(r["value"]))
    eps=np.finfo(np.float32).eps
    for key,raw in sorted(groups.items()):
        a=np.asarray(raw,float); scale=max(float(np.median(np.abs(a))),1e-12); floor=10*eps*max(1.,scale); maximum=float(a.max()); p95=float(np.quantile(a,.95)); p99=float(np.quantile(a,.99))
        row={"category":key[0],"output":key[1],"metric":key[2],"count":len(a),"median":float(np.median(a)),"p95":p95,"p99":p99,"observed_maximum":maximum,"reference_scale":scale,"numerical_floor":floor,"warning_limit":maximum,"hard_limit":max(1.5*maximum,maximum+floor),"cohort_median_limit":max(1.25*p95,p95+floor),"cohort_p95_limit":max(1.25*maximum,maximum+floor),"formula":"warning=observed_maximum; hard=max(1.5*maximum,maximum+floor); cohort_median=max(1.25*p95,p95+floor); cohort_p95=max(1.25*maximum,maximum+floor)","source_runs":"run_01,run_02,run_03"}
        result.append(row); index["|".join(key)] = row
    return result,index


def _medoid(root: Path,cfg:dict[str,Any], calibration:list[dict[str,Any]], thresholds:dict[str,dict[str,float]]) -> dict[str,Any]:
    candidates=[]
    for run in RUN_NAMES:
        pair=[r for r in calibration if (r["left"]==run or r["right"]==run)]
        normalized=[]
        for r in pair:
            t=thresholds["|".join((r["category"],r["output"],r["metric"]))]; normalized.append(float(r["value"])/max(float(t["median"]),float(t["numerical_floor"])))
        candidates.append({"run":run,"score":float(np.median(normalized))})
    candidates.sort(key=lambda x:(x["score"],x["run"])); return {"candidate_scores":candidates,"selected_reference_run":candidates[0]["run"],"selection_method":"global medoid: calibration metric / empirical median; equal output weighting via metric medians","included_outputs":[*LATENTS,"albedo_like","normal_coarse",*PRESETS],"normalization_method":"value/max(metric_median,numerical_floor)","tie_break_rule":"lexicographic run id"}


def build_contract(root: Path,cfg:dict[str,Any]) -> dict[str,Any]:
    validate_inputs(root,cfg); out=base(root)
    if (out/"metadata/FROZEN.json").exists(): raise FileExistsError("v2 is frozen; create a new version")
    out.mkdir(parents=True,exist_ok=True); rows=_case_rows(root,cfg); write_manifest(root,cfg)
    calibration=[]
    for left,right in (("run_01","run_02"),("run_01","run_03"),("run_02","run_03")):
        for r in _values_for_pair(root,cfg,left,right): r.update({"left":left,"right":right}); calibration.append(r)
    thresholds,idx=_thresholds(calibration); selection=_medoid(root,cfg,calibration,idx); selected=selection["selected_reference_run"]
    reference=out/"reference/pilot12"; 
    if reference.exists(): shutil.rmtree(reference)
    shutil.copytree(_paths(root,selected),reference)
    _write(out/"metadata/checksums.json",{str(p.relative_to(reference)):sha(p) for p in reference.rglob("*") if p.is_file()})
    manifest=[]
    for r in rows:
        diag=_json(_paths(root,"run_01")/"cases"/r["case_id"] / "diagnostics.json")
        manifest.append({"case_id":r["case_id"],"p0b_audit_id":r["p0b_audit_id"],"input_sha256":r["input_sha256"],"decoded_rgb_sha256":diag["decoded_rgb_uint8"]["array_sha256"],"deca_input_tensor_sha256":diag["deca_input_tensor"]["array_sha256"],"run_01_path":str(_paths(root,"run_01")),"run_02_path":str(_paths(root,"run_02")),"run_03_path":str(_paths(root,"run_03")),"fixed_input_mode":"direct_p0_aligned"})
    _csv(out/"metadata/baseline_manifest.csv",manifest); _csv(out/"metadata/thresholds.csv",thresholds); _csv(out/"comparisons/calibration_pair_metrics.csv",calibration)
    _write(out/"metadata/empirical_envelope.json",{"contract_version":VERSION,"calibration_runs":list(RUN_NAMES),"thresholds":thresholds}); _write(out/"metadata/reference_selection.json",selection)
    env=_json(_paths(root,selected)/"metadata/environment.json"); implementation={name:sha(root/name) for name in ("p0b_deca/deca_runtime.py","p0b_deca/p1_generation.py","p0b_deca/p1_reproducibility_audit.py")}; fingerprint={"environment":env,"inference_implementation_fingerprint":implementation,"deca_output_schema":_json(root/cfg["output_root"] / "metadata/deca_output_inventory.json")}
    _write(out/"metadata/environment_fingerprint.json",fingerprint); _write(out/"metadata/asset_hashes.json",env["asset_hashes"])
    contract={"contract_version":VERSION,"status":"DRAFT","reference_run":selected,"calibration_runs":list(RUN_NAMES),"holdout_excluded":True,"alpha_threshold":ALPHA_THRESHOLD,"threshold_formula":"versioned empirical formula in thresholds.csv","thresholds_sha256":sha(out/"metadata/thresholds.csv"),"environment_fingerprint_sha256":sha(out/"metadata/environment_fingerprint.json"),"fixed_input_mode":"direct_p0_aligned","presets":list(PRESETS),"strict_v1_preserved":True,"full_500_started":False}
    _write(out/"metadata/contract.json",contract)
    return contract


def run_holdout(root:Path,cfg:dict[str,Any]) -> dict[str,Any]:
    out=base(root); contract=_json(out/"metadata/contract.json")
    if contract["status"]!="DRAFT": raise ValueError("holdout requires draft v2 contract")
    dest=out/"holdout_validation/run_04"; return worker(root,cfg,"run_04",dest)


def _exact(root:Path,cfg:dict[str,Any],reference:Path,holdout:Path) -> list[str]:
    errors=[]; ref_env=_json(reference/"metadata/environment.json"); hold_env=_json(holdout/"metadata/environment.json")
    for k in ("asset_hashes","effective_config_sha256","fixed_input_mode","model_training","all_parameters_requires_grad_false"):
        if ref_env.get(k)!=hold_env.get(k): errors.append(f"environment:{k}")
    for row in _case_rows(root,cfg):
        cid=row["case_id"]; rd=_json(reference/"cases"/cid/"diagnostics.json"); hd=_json(holdout/"cases"/cid/"diagnostics.json")
        for key in ("decoded_rgb_uint8","decoded_rgb_float","deca_input_tensor"):
            if rd[key]["array_sha256"]!=hd[key]["array_sha256"]: errors.append(f"{cid}:{key}")
        rl,rm,rr=_load_run(reference.parent,reference.name,cid); hl,hm,hr=_load_run(holdout.parent,holdout.name,cid)
        if set(rl.files)!=set(hl.files) or set(rm.files)!=set(hm.files) or list(rr["preset_names"])!=list(hr["preset_names"]): errors.append(f"{cid}:schema_or_presets")
        for z in (hl,hm,hr):
            for k in z.files:
                if np.issubdtype(z[k].dtype,np.number) and not np.isfinite(z[k]).all(): errors.append(f"{cid}:nonfinite:{k}")
    return errors


def validate_holdout(root:Path,cfg:dict[str,Any]) -> dict[str,Any]:
    out=base(root); contract=_json(out/"metadata/contract.json"); reference=out/"reference/pilot12"; holdout=out/"holdout_validation/run_04"; threshold_rows=list(csv.DictReader((out/"metadata/thresholds.csv").open())); thresholds={"|".join((x["category"],x["output"],x["metric"])):x for x in threshold_rows}
    exact=_exact(root,cfg,reference,holdout); values=[]
    # Reuse the exact same comparison implementation by presenting reference/run_04 as audit children.
    temp=out/"calibration_runs"; temp.mkdir(exist_ok=True)
    # no copying or modification of observations: direct metric calculation is kept here.
    for row in _case_rows(root,cfg):
        cid=row["case_id"]; la,ma,ra=_load_run(reference.parent,reference.name,cid); lb,mb,rb=_load_run(holdout.parent,holdout.name,cid)
        for key in LATENTS:
            p=_pair_metrics(la[key],lb[key]);
            for metric in ("max_abs","MAE","RMSE","relative_L2"): values.append({"case_id":cid,"category":"latent","output":key,"metric":metric,"value":p[metric]})
            values.append({"case_id":cid,"category":"latent","output":key,"metric":"cosine_distance","value":1-p["cosine_similarity"]})
        face=_read_any(Path(row["face_valid_path"]))>0; physics=_read_any(Path(row["physics_core_skin_path"]))>0; regions=_regions(ma["alpha"],mb["alpha"],face,physics)
        for output in ("albedo_like","normal_coarse"):
            for region,prefix in (("physics_core_skin","physics_core"),("alpha_intersection_erode_3","eroded_alpha_3")):
                m=_map_metrics(ma[output],mb[output],regions[region],normal=output=="normal_coarse")
                for metric in ("MAE","p99_abs","SSIM"): values.append({"case_id":cid,"category":output,"output":output,"metric":f"{prefix}_{metric if metric!='SSIM' else 'SSIM_difference'}","value":1-m[metric] if metric=="SSIM" else m[metric]})
                if output=="normal_coarse":
                    for metric in ("mean_angular_error_deg","p99_angular_error_deg"): values.append({"case_id":cid,"category":output,"output":output,"metric":f"{prefix}_{metric}","value":m[metric]})
        for i,preset in enumerate(PRESETS):
            for region,mask in (("full_image",np.ones(ma["alpha"].shape,bool)),("physics_core",physics),("eroded_alpha_3",regions["alpha_intersection_erode_3"])):
                m=_map_metrics(ra["relighted_images"][i],rb["relighted_images"][i],mask)
                for metric in ("MAE","SSIM"): values.append({"case_id":cid,"category":"relighting","output":preset,"metric":f"{region}_{metric if metric!='SSIM' else 'SSIM_difference'}","value":1-m[metric] if metric=="SSIM" else m[metric]})
            d=np.abs(_q(ra["relighted_images"][i]).astype(np.int16)-_q(rb["relighted_images"][i]).astype(np.int16))
            for level in (1,2,3): values.append({"case_id":cid,"category":"relighting","output":preset,"metric":f"fraction_gt_{level}_gray_levels","value":float((d>level).mean())})
    failures=[]; warnings=[]; failed_case_ids:set[str]=set(); categories={"latent":[],"albedo_like":[],"normal_coarse":[],"relighting":[]}
    for v in values:
        t=thresholds["|".join((v["category"],v["output"],v["metric"]))]; value=float(v["value"]); warning=float(t["warning_limit"]); hard=float(t["hard_limit"]); state="PASS" if value<=warning else "PASS_WITH_MARGIN" if value<=hard else "FAIL"; v.update({"warning_limit":warning,"hard_limit":hard,"state":state}); categories[v["category"]].append(v)
    for category,items in categories.items():
        for metric in sorted({(x["output"],x["metric"]) for x in items}):
            group=[x for x in items if (x["output"],x["metric"])==metric]; t=thresholds["|".join((category,metric[0],metric[1]))]
            failed = any(x["state"]=="FAIL" for x in group) or sum(x["state"]=="PASS_WITH_MARGIN" for x in group)>2 or np.median([x["value"] for x in group])>float(t["cohort_median_limit"]) or np.quantile([x["value"] for x in group],.95)>float(t["cohort_p95_limit"])
            if failed:
                failures.append(f"{category}:{metric[0]}:{metric[1]}")
                failed_case_ids.update(x["case_id"] for x in group if x["state"] != "PASS")
            warnings.extend([x for x in group if x["state"]=="PASS_WITH_MARGIN"])
    _csv(out/"comparisons/run04_vs_reference_metrics.csv",values); _csv(out/"comparisons/run04_warning_cases.csv",warnings)
    status="VALIDATED" if not exact and not failures else "REJECTED"; decision="READY_FOR_FULL500" if status=="VALIDATED" else "KEEP_BLOCKED"
    result={"contract_version":VERSION,"contract_status":status,"reference_run":contract["reference_run"],"environment_match":not bool(exact),"exact_gate_status":"pass" if not exact else "fail","latent_gate_status":"pass" if not any(x.startswith("latent:") for x in failures) else "fail","albedo_gate_status":"pass" if not any(x.startswith("albedo_like:") for x in failures) else "fail","normal_gate_status":"pass" if not any(x.startswith("normal_coarse:") for x in failures) else "fail","relighting_gate_status":"pass" if not any(x.startswith("relighting:") for x in failures) else "fail","warning_count":len(warnings),"failure_count":len(failures),"failure_metrics":failures,"failed_case_ids":sorted(failed_case_ids),"pilot12_pass_count":12-len(failed_case_ids),"decision":decision,"full_500_started":False}
    _write(out/"metadata/validation_decision.json",result); legacy=_json(root/cfg["output_root"] / "metadata/pilot12_regression.json"); _write(out/"metadata/legacy_compatibility.json",{"v1_strict_contract_result":legacy["status"],"v2_cross_process_contract_result":status,"reason_for_difference":"v1 requires historical strict replay; v2 is calibrated only from three independent current-process observations","v1_scope":"strict repeat-inference audit","v2_scope":"independent-process historical regression"})
    _write(out/"metadata/checksums.json",{str(p.relative_to(out)):sha(p) for p in (out/"reference").rglob("*") if p.is_file()})
    return result


def freeze(root:Path) -> dict[str,Any]:
    out=base(root); decision=_json(out/"metadata/validation_decision.json")
    if decision["contract_status"]!="VALIDATED": raise ValueError("cannot freeze rejected or draft v2 contract")
    target=out/"metadata/FROZEN.json"
    if target.exists(): raise FileExistsError("frozen contracts are immutable")
    checks={str(p.relative_to(out)):sha(p) for p in (out/"reference").rglob("*") if p.is_file()}; _write(out/"metadata/checksums.json",checks)
    frozen={"contract_version":VERSION,"contract_sha256":sha(out/"metadata/contract.json"),"reference_manifest_sha256":sha(out/"metadata/baseline_manifest.csv"),"thresholds_sha256":sha(out/"metadata/thresholds.csv"),"environment_fingerprint_sha256":sha(out/"metadata/environment_fingerprint.json"),"reference_output_hashes":checks,"validation_run_id":"run_04","validation_decision_sha256":sha(out/"metadata/validation_decision.json"),"frozen_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"immutable":True}; _write(target,frozen); return frozen
