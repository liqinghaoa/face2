"""SO-R1-C: read-only frozen Baseline/Proposed synthetic ID/OOD evaluation."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import Dataset

from src.skin_optics_so_r1 import baseline_training as bt
from src.skin_optics_so_r1 import proposed_training as pt

B1_RUN = Path("runs/so_r1_b1_baseline_v6")
B2_RUN = Path("runs/so_r1_b2_proposed_v3")
C_DATA = Path("data/processed/SO_R1_C_SyntheticEvaluation_v1")
C_REPORT = Path("reports/so_r1_c_synthetic_evaluation")
PROTOCOL = Path("config/so_r1/so_r1_c_synthetic_evaluation_protocol_v1.yaml")
LOCK = Path("config/so_r1/SO_R1_C_EVALUATION_LOCK.json")
TEST_SPLITS = ("ID Test", "Camera-OOD", "Light-OOD", "Joint-OOD")
ROLES = ("A0", "A1", "A2", "A3", "A4")
PAIR_TYPES = {"camera_only": ("camera", 1), "light_only": ("light", 2), "appearance_only": ("appearance", 3), "joint": ("joint", 4)}
PRIMARY = {"Camera-OOD": "camera", "Light-OOD": "light", "Joint-OOD": "joint"}


def _sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()


def _hash(value: Any) -> str: return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()
def _dump(path: Path, value: Any) -> None: path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,indent=2,sort_keys=True,default=str)+"\n",encoding="utf-8")
def _finite(array: np.ndarray) -> bool: return bool(np.isfinite(array).all())


def masked_map_metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, mean_target: np.ndarray) -> dict[str, np.ndarray]:
    """Per-latent, per-channel metrics; inputs are [B,2,H,W], mask [B,1,H,W]."""
    valid=np.broadcast_to(mask,pred.shape); denom=valid.sum((2,3)).clip(1); error=pred-target; absolute=np.abs(error); smooth=np.where(absolute<.1,.5*error**2/.1,absolute-.05)
    pred_mean=(pred*valid).sum((2,3))/denom; target_mean=(target*valid).sum((2,3))/denom
    pred_var=np.maximum((pred**2*valid).sum((2,3))/denom-pred_mean**2,0); target_var=np.maximum((target**2*valid).sum((2,3))/denom-target_mean**2,0)
    result={"masked_smooth_l1":(smooth*valid).sum((2,3))/denom,"mae":(absolute*valid).sum((2,3))/denom,"rmse":np.sqrt((error**2*valid).sum((2,3))/denom),"mean_predictor_mae":(np.abs(mean_target[None]-target)*valid).sum((2,3))/denom,"prediction_mean":pred_mean,"target_mean":target_mean,"prediction_variance":pred_var,"target_variance":target_var,"variance_ratio":pred_var/np.maximum(target_var,1e-20),"zero_saturation_fraction":((pred<=1e-6)*valid).sum((2,3))/denom,"one_saturation_fraction":((pred>=1-1e-6)*valid).sum((2,3))/denom}
    pearson=np.empty_like(pred_mean); spearman=np.empty_like(pred_mean)
    for i in range(pred.shape[0]):
        for c in range(2):
            use=valid[i,c].astype(bool); x=pred[i,c][use]; y=target[i,c][use]
            pearson[i,c]=pearsonr(x,y).statistic if x.size>1 and np.std(x)>0 and np.std(y)>0 else 0.0
            spearman[i,c]=spearmanr(x,y).statistic if x.size>1 and np.std(x)>0 and np.std(y)>0 else 0.0
    result["pearson"]=pearson; result["spearman"]=spearman
    return result


def paired_bootstrap(proposed: np.ndarray, baseline: np.ndarray, *, draws: int, seed: int, one_sided: bool = False) -> dict[str, Any]:
    proposed=np.asarray(proposed,dtype=np.float64); baseline=np.asarray(baseline,dtype=np.float64)
    if proposed.shape != baseline.shape or proposed.ndim != 1 or not len(proposed): raise ValueError("paired one-dimensional latent vectors required")
    delta=proposed-baseline; rng=np.random.default_rng(seed); means=np.empty(draws,dtype=np.float64)
    for i in range(draws): means[i]=delta[rng.integers(0,len(delta),size=len(delta))].mean()
    return {"n":int(len(delta)),"mean_delta":float(delta.mean()),"ci":([float(np.quantile(means,.05)),float(np.quantile(means,.95))] if one_sided else [float(np.quantile(means,.025)),float(np.quantile(means,.975))]),"bootstrap_draws":draws}


def stratified_bootstrap(deltas: dict[str,np.ndarray], *, draws: int, seed: int, one_sided: bool = False) -> dict[str, Any]:
    if not deltas or any(np.asarray(v).ndim != 1 or not len(v) for v in deltas.values()): raise ValueError("nonempty one-dimensional strata required")
    keys=tuple(sorted(deltas)); arrays={k:np.asarray(deltas[k],dtype=np.float64) for k in keys}; rng=np.random.default_rng(seed); values=np.empty(draws,dtype=np.float64)
    for i in range(draws): values[i]=np.mean([array[rng.integers(0,len(array),size=len(array))].mean() for array in arrays.values()])
    ci=([float(np.quantile(values,.05)),float(np.quantile(values,.95))] if one_sided else [float(np.quantile(values,.025)),float(np.quantile(values,.975))])
    return {"strata":{k:int(len(v)) for k,v in arrays.items()},"mean_delta":float(np.mean([v.mean() for v in arrays.values()])),"ci":ci,"bootstrap_draws":draws,"one_sided":one_sided}


def summarize(values: np.ndarray, *, draws: int = 1000, seed: int = 0) -> dict[str, Any]:
    values=np.asarray(values,dtype=np.float64); rng=np.random.default_rng(seed); means=np.empty(draws,dtype=np.float64)
    for i in range(draws): means[i]=values[rng.integers(0,len(values),size=len(values))].mean()
    return {"n":int(len(values)),"mean":float(values.mean()),"median":float(np.median(values)),"p25":float(np.quantile(values,.25)),"p75":float(np.quantile(values,.75)),"p95":float(np.quantile(values,.95)),"bootstrap95ci":[float(np.quantile(means,.025)),float(np.quantile(means,.975))]}


class TestLatentDataset(Dataset):
    def __init__(self, rows: list[dict[str,Any]]) -> None: self.rows=sorted(rows,key=lambda r:r["latent_id"])
    def __len__(self) -> int: return len(self.rows)
    def __getitem__(self,index: int) -> dict[str,Any]:
        row=self.rows[index]
        rgb,target,mask=bt._load_npz(str(row["file_path"]),[0,1,2,3,4])
        return {"images":torch.from_numpy(rgb),"target":torch.from_numpy(target),"mask":torch.from_numpy(mask),"latent_id":row["latent_id"],"split":row["split"]}


def _mean_predictor_targets_only(rows: list[dict[str,Any]]) -> tuple[np.ndarray,int]:
    sums=np.zeros((2,256,256),dtype=np.float64); counts=np.zeros((1,256,256),dtype=np.float64)
    for row in rows:
        if row["split"] != "Train": continue
        with np.load(row["file_path"],allow_pickle=False) as z:
            target=np.stack((z["m"],z["h"])).astype(np.float32); mask=z["mask"].astype(np.float32)[None]
        sums += target*mask; counts += mask
    return (sums/np.maximum(counts,1)).astype(np.float32), int(sum(row["split"]=="Train" for row in rows))


def _authority(root: Path, rows: list[dict[str,Any]]) -> tuple[dict[str,Any],dict[str,Any],dict[str,Any],dict[str,Path]]:
    root=Path(root); b1=json.loads((root/B1_RUN/"B1_ACCEPTANCE.json").read_text()); b2=json.loads((root/B2_RUN/"B2_ACCEPTANCE.json").read_text()); selected=json.loads((root/B2_RUN/"PROPOSED_SELECTED_CHECKPOINT.json").read_text())
    expected_b1=root/B1_RUN/"checkpoints/best_val_masked_smoothl1.pt"; expected_b2=root/B2_RUN/"lambda_0p50/checkpoints/best_val_masked_smoothl1.pt"
    required_b1={"status":"PASS","baseline_checkpoint_authoritative":True}; required_b2={"status":"PASS","proposed_checkpoint_authoritative":True,"selected_lambda":.5,"next_stage":"SO-R1-C","next_stage_authorized":True}
    if any(b1.get(k)!=v for k,v in required_b1.items()) or any(b2.get(k)!=v for k,v in required_b2.items()): raise RuntimeError("FAIL_INPUT_PROVENANCE_GATE:ACCEPTANCE")
    paths={"baseline":Path(b1["best_checkpoint"]),"proposed":Path(selected["selected_checkpoint_path"]),"b1_contract":root/"config/so_r1/so_r1_b1_training_contract_v6.yaml","b2_lock":root/pt.LAMBDA_LOCK,"d0_lock":root/"config/so_r1/SO_R1_A2_D0_AM1_PROTOCOL_LOCK.json","g1_acceptance":root/bt.DATA_REL/"G1_AM1_ACCEPTANCE.json"}
    if paths["baseline"].resolve()!=expected_b1.resolve() or paths["proposed"].resolve()!=expected_b2.resolve() or not all(path.is_file() for path in paths.values()): raise RuntimeError("FAIL_INPUT_PROVENANCE_GATE:PATH")
    if _sha(paths["baseline"])!=b2["B1_checkpoint_hash"] or _sha(paths["proposed"])!=selected["selected_checkpoint_sha256"] or b2["selected_checkpoint_sha256"]!=selected["selected_checkpoint_sha256"]: raise RuntimeError("FAIL_INPUT_PROVENANCE_GATE:CHECKPOINT_HASH")
    d0=json.loads(paths["d0_lock"].read_text()); g1=json.loads(paths["g1_acceptance"].read_text())
    if b1["contract_hash"]!=b2["B1_training_contract_hash"] or _sha(paths["b2_lock"])!=b2["lambda_selection_lock_hash"] or d0["protocol_hash"]!=b2["D0_AM1_protocol_hash"] or _sha(paths["g1_acceptance"])!=b2["G1_AM1_dataset_acceptance_hash"] or g1["dataset_status"]!="ACCEPTED": raise RuntimeError("FAIL_INPUT_PROVENANCE_GATE:CHAIN_HASH")
    contract=yaml.safe_load(paths["b1_contract"].read_text()); if_arch=contract["model_architecture"]
    if if_arch["id"]!=bt.BaselineMHUNet.architecture_id or if_arch["output_order"]!=bt.BaselineMHUNet.output_order or contract["amp_contract"]["precision"]!="FP32": raise RuntimeError("FAIL_INPUT_PROVENANCE_GATE:CONTRACT")
    expected={"ID Test":1000,"Camera-OOD":500,"Light-OOD":500,"Joint-OOD":500}
    counts={name:sum(row["split"]==name for row in rows) for name in expected}
    if counts!=expected: raise RuntimeError("FAIL_INPUT_PROVENANCE_GATE:TEST_COUNTS")
    return b1,b2,selected,paths


def _pair_integrity(root: Path, rows: list[dict[str,Any]]) -> dict[str,Any]:
    path=root/bt.DATA_REL/"manifests/pair_manifest.csv"; pairs=pd.read_csv(path); test=pairs[pairs["split"].isin(TEST_SPLITS)].copy(); expected={"ID Test":4000,"Camera-OOD":2000,"Light-OOD":2000,"Joint-OOD":2000}
    counts=test.groupby("split").size().to_dict(); semantic=True
    expected_changes={"camera_only":(True,False,False),"light_only":(False,True,False),"appearance_only":(False,False,True),"joint":(True,True,True)}
    for name,(label,index) in PAIR_TYPES.items():
        subset=test[test["pair_type"]==name]; semantic &= bool(((subset["target_acquisition"].str[-1].astype(int)==index)&subset["m_hash_match"]&subset["h_hash_match"]&subset["mask_hash_match"]).all())
        semantic &= bool((subset[["camera_changed","light_changed","appearance_changed"]].astype(bool).to_numpy()==np.asarray(expected_changes[name])).all())
    refs=set(test["reference_acquisition"].str[:-3]); row_ids={r["latent_id"] for r in rows if r["split"] in TEST_SPLITS}
    if counts!=expected or not semantic or refs!=row_ids: raise RuntimeError("FAIL_PAIR_MANIFEST_INTEGRITY")
    return {"status":"PASS","pair_counts":counts,"pair_total":int(len(test)),"latent_reference_count":len(refs),"manifest_sha256":_sha(path)}


def _protocol(b1: dict[str,Any], b2: dict[str,Any], selected: dict[str,Any], paths: dict[str,Path]) -> dict[str,Any]:
    return {"stage_id":"SO-R1-C","version":"v1","read_only":True,"splits":list(TEST_SPLITS),"roles":list(ROLES),"pairs":PAIR_TYPES,"primary_ood":PRIMARY,"precision":"FP32","bootstrap_draws":10000,"quality_bootstrap_draws":1000,"selected_lambda":.5,"baseline_checkpoint_sha256":_sha(paths["baseline"]),"proposed_checkpoint_sha256":_sha(paths["proposed"]),"b1_contract_hash":b1["contract_hash"],"b2_lock_hash":_sha(paths["b2_lock"]),"d0_protocol_hash":json.loads(paths["d0_lock"].read_text())["protocol_hash"],"g1_acceptance_hash":_sha(paths["g1_acceptance"]),"within_between_min_fraction":.5}


def _model(path: Path, device: torch.device) -> torch.nn.Module:
    payload=torch.load(path,map_location=device,weights_only=False); model=bt.BaselineMHUNet().to(device); model.load_state_dict(payload["model_state"]); model.eval(); return model


def _quality_and_drift(root: Path, rows: list[dict[str,Any]], models: dict[str,torch.nn.Module], mean_target: np.ndarray, device: torch.device) -> tuple[pd.DataFrame,pd.DataFrame,dict[str,int]]:
    quality: list[dict[str,Any]]=[]; drift: list[dict[str,Any]]=[]; access={"Train RGB access":0,"Validation RGB access":0,**{f"{name} access":0 for name in TEST_SPLITS}}
    for split in TEST_SPLITS:
        subset=[r for r in rows if r["split"]==split]; loader=bt._loader(TestLatentDataset(subset),4,shuffle=False,seed=0,workers=2,groups=False,timeout_seconds=120)
        for batch in loader:
            images=batch["images"].to(device,non_blocking=True); target=batch["target"].numpy(); mask=batch["mask"].numpy(); ids=list(batch["latent_id"]); access[f"{split} access"] += len(ids)*5
            for model_name,model in models.items():
                with torch.no_grad(): output=model(images.flatten(0,1)).reshape(images.shape[0],5,2,256,256)
                prediction=output.detach().cpu().numpy()
                if not _finite(prediction): raise RuntimeError("FAIL_PIPELINE_INVALID:NONFINITE_OUTPUT")
                for role_index,role in enumerate(ROLES):
                    values=masked_map_metrics(prediction[:,role_index],target,mask,mean_target)
                    for i,latent_id in enumerate(ids):
                        for channel,label in enumerate(("M","H")):
                            quality.append({"model":model_name,"split":split,"latent_id":latent_id,"role":role,"channel":label,**{key:float(value[i,channel]) for key,value in values.items()}})
                for manifest_name,(pair,index) in PAIR_TYPES.items():
                    valid=np.broadcast_to(mask,prediction[:,0].shape); denom=valid.sum((2,3)).clip(1); delta=prediction[:,0]-prediction[:,index]
                    mae=(np.abs(delta)*valid).sum((2,3))/denom; rmse=np.sqrt((delta**2*valid).sum((2,3))/denom)
                    for i,latent_id in enumerate(ids):
                        for channel,label in enumerate(("M","H")): drift.append({"model":model_name,"split":split,"latent_id":latent_id,"pair_type":pair,"channel":label,"masked_mae_drift":float(mae[i,channel]),"masked_rmse_drift":float(rmse[i,channel])})
        print(json.dumps({"phase":"test_inference","split":split,"latents":len(subset)},sort_keys=True),flush=True)
    return pd.DataFrame(quality),pd.DataFrame(drift),access


def _within_between(rows: list[dict[str,Any]], models: dict[str,torch.nn.Module], device: torch.device, drift: pd.DataFrame) -> pd.DataFrame:
    """Cyclic different-latent A0 audit, batched without altering either model."""
    records=[]
    lookup=drift.set_index(["model","split","latent_id","pair_type","channel"])["masked_mae_drift"].to_dict()
    for split in TEST_SPLITS:
        subset=sorted([r for r in rows if r["split"]==split],key=lambda r:r["latent_id"])
        for start in range(0,len(subset),16):
            chosen=subset[start:start+16]; partners=[subset[(start+i+1)%len(subset)] for i in range(len(chosen))]
            rgb=[]; masks=[]
            for row,other in zip(chosen,partners):
                rgb_a,_,mask_a=bt._load_npz(str(row["file_path"]),[0]); rgb_b,_,_=bt._load_npz(str(other["file_path"]),[0]); rgb.append(np.concatenate((rgb_a,rgb_b))); masks.append(mask_a)
            inputs=torch.from_numpy(np.concatenate(rgb)).to(device); mask=np.stack(masks)
            for name,model in models.items():
                with torch.no_grad(): prediction=model(inputs).detach().cpu().numpy().reshape(len(chosen),2,2,256,256)
                if not _finite(prediction): raise RuntimeError("FAIL_PIPELINE_INVALID:NONFINITE_WITHIN_BETWEEN_OUTPUT")
                valid=np.broadcast_to(mask,prediction[:,0].shape); denom=valid.sum((2,3)).clip(1); between=(np.abs(prediction[:,0]-prediction[:,1])*valid).sum((2,3))/denom
                for i,row in enumerate(chosen):
                    for _,(pair_type,_) in PAIR_TYPES.items():
                        for c,label in enumerate(("M","H")):
                            within=float(lookup[(name,split,row["latent_id"],pair_type,label)])
                            records.append({"model":name,"split":split,"latent_id":row["latent_id"],"pair_type":pair_type,"channel":label,"within_drift":within,"between_drift":float(between[i,c]),"within_between_ratio":float(within/max(between[i,c],1e-20)),"within_lt_between":bool(within<between[i,c])})
    return pd.DataFrame(records)


def _summary_frame(frame: pd.DataFrame, group: list[str], metrics: list[str], seed: int) -> pd.DataFrame:
    rows=[]
    for keys,item in frame.groupby(group,sort=True):
        base=dict(zip(group,keys if isinstance(keys,tuple) else (keys,)))
        for n,metric in enumerate(metrics): rows.append({**base,"metric":metric,**summarize(item[metric].to_numpy(),seed=seed+n)})
    return pd.DataFrame(rows)


def _comparisons(drift: pd.DataFrame, quality: pd.DataFrame) -> tuple[dict[str,Any],dict[str,Any],dict[str,Any],dict[str,Any]]:
    paired={}; primary={}; ni={}; mean={}
    for (split,pair,channel), subset in drift.groupby(["split", "pair_type", "channel"], sort=True):
        item = subset.pivot(index="latent_id", columns="model", values="masked_mae_drift")
        if not {"Baseline","Proposed"}.issubset(item.columns): continue
        b=item["Baseline"].to_numpy(); p=item["Proposed"].to_numpy(); boot=paired_bootstrap(p,b,draws=10000,seed=100+len(paired)); paired["|".join((split,pair,channel))]={**boot,"baseline_mean":float(b.mean()),"proposed_mean":float(p.mean()),"relative_reduction":float(1-p.mean()/b.mean()),"reduced_proportion":float((p<b).mean())}
    for channel in ("M","H"):
        strata={}
        for split,pair in PRIMARY.items():
            item=drift[(drift.split==split)&(drift.pair_type==pair)&(drift.channel==channel)].pivot(index="latent_id",columns="model",values="masked_mae_drift"); strata[split]=1-item["Proposed"].to_numpy()/item["Baseline"].to_numpy()
        primary[channel]=stratified_bootstrap(strata,draws=10000,seed=700+(channel=="H")); primary[channel]["relative_reduction_mean"]=float(np.mean([v.mean() for v in strata.values()]))
        strata_quality={}
        for split in TEST_SPLITS:
            item=quality[(quality.split==split)&(quality.channel==channel)].groupby(["latent_id","model"],as_index=False).mae.mean().pivot(index="latent_id",columns="model",values="mae"); strata_quality[split]=item["Proposed"].to_numpy()-item["Baseline"].to_numpy()
        baseline_mean=float(np.mean([quality[(quality.split==split)&(quality.channel==channel)&(quality.model=="Baseline")].groupby("latent_id").mae.mean().mean() for split in TEST_SPLITS])); record=stratified_bootstrap(strata_quality,draws=10000,seed=800+(channel=="H"),one_sided=True); margin=.05*baseline_mean; record.update({"baseline_mae":baseline_mean,"margin":margin,"upper95ci":record["ci"][1],"pass":record["ci"][1] <= margin}); ni[channel]=record
    for (model,split,channel),item in quality.groupby(["model","split","channel"]): mean["|".join((model,split,channel))]={"model_mae":float(item.mae.mean()),"mean_predictor_mae":float(item.mean_predictor_mae.mean()),"pass":bool(item.mae.mean()<item.mean_predictor_mae.mean())}
    return paired,primary,ni,mean


def run(root: Path) -> dict[str,Any]:
    root=Path(root); out=root/C_DATA; report=root/C_REPORT
    restarting=False
    if out.exists():
        previous=out/"evaluation_run_manifest.json"
        existing=json.loads((out/"C_ACCEPTANCE.json").read_text()) if (out/"C_ACCEPTANCE.json").is_file() else {}
        restarting=previous.is_file() and json.loads(previous.read_text()).get("status")=="RUNNING" and (not existing or existing.get("error_type") in {"FileExistsError","ValueError","KeyError"})
        if not restarting: raise RuntimeError("REFUSE_OVERWRITE_EXISTING_C_EVALUATION")
    _,_,rows=bt.validate_authoritative_input(root); b1,b2,selected,paths=_authority(root,rows); pair_audit=_pair_integrity(root,rows); protocol=_protocol(b1,b2,selected,paths)
    if (root/PROTOCOL).is_file() and _hash(yaml.safe_load((root/PROTOCOL).read_text()))!=_hash(protocol): raise RuntimeError("FAIL_INPUT_PROVENANCE_GATE:PROTOCOL")
    (root/PROTOCOL).parent.mkdir(parents=True,exist_ok=True); (root/PROTOCOL).write_text(yaml.safe_dump(protocol,sort_keys=False),encoding="utf-8")
    lock={"status":"FROZEN_BEFORE_TEST","evaluation_protocol_hash":_hash(protocol),"baseline_checkpoint_sha256":protocol["baseline_checkpoint_sha256"],"proposed_checkpoint_sha256":protocol["proposed_checkpoint_sha256"],"selected_lambda":.5,"test_splits":list(TEST_SPLITS)}
    if (root/LOCK).is_file() and json.loads((root/LOCK).read_text())!=lock: raise RuntimeError("FAIL_INPUT_PROVENANCE_GATE:LOCK")
    _dump(root/LOCK,lock); before=pt.audit_protected_assets(root,verify_payloads=True)
    if before["status"]!="PASS": raise RuntimeError("FAIL_PIPELINE_INVALID:PROTECTED_ASSET_AUDIT")
    out.mkdir(parents=True,exist_ok=restarting); report.mkdir(parents=True,exist_ok=True); _dump(report/"protected_asset_hash_audit_before.json",before); _dump(out/"evaluation_run_manifest.json",{"status":"RUNNING","read_only":True,"protocol_hash":_hash(protocol),"lock_hash":_sha(root/LOCK),"pair_audit":pair_audit})
    fixed=pd.DataFrame([{"split":split,"latent_id":row["latent_id"],"selection_rule":"lexicographically first 3 latent_id; QUALITATIVE_AUDIT_ONLY"} for split in TEST_SPLITS for row in sorted([r for r in rows if r["split"]==split],key=lambda r:r["latent_id"])[:3]])
    fixed.to_csv(out/"fixed_qualitative_latent_manifest.csv",index=False); (report/"fixed_qualitative_examples").mkdir(exist_ok=True)
    if not torch.cuda.is_available(): raise RuntimeError("FAIL_PIPELINE_INVALID:CUDA_UNAVAILABLE")
    device=torch.device("cuda:0"); mean_target,train_target_mask_access=_mean_predictor_targets_only(rows); models={"Baseline":_model(paths["baseline"],device),"Proposed":_model(paths["proposed"],device)}
    quality,drift,access=_quality_and_drift(root,rows,models,mean_target,device); within=_within_between(rows,models,device,drift)
    access["Train target/mask access"] = train_target_mask_access
    quality.to_csv(report/"prediction_quality_latent_metrics.csv",index=False); drift.to_csv(report/"pair_drift_latent_metrics.csv",index=False); within.to_csv(report/"within_between_audit.csv",index=False)
    quality_summary=_summary_frame(quality,["model","split","role","channel"],["masked_smooth_l1","mae","rmse","pearson","spearman","prediction_mean","target_mean","prediction_variance","target_variance","variance_ratio","zero_saturation_fraction","one_saturation_fraction"],10); drift_summary=_summary_frame(drift,["model","split","pair_type","channel"],["masked_mae_drift","masked_rmse_drift"],100); quality_summary.to_csv(report/"prediction_quality_summary.csv",index=False); drift_summary.to_csv(report/"pair_drift_summary.csv",index=False)
    paired,primary,noninferiority,mean_audit=_comparisons(drift,quality); _dump(report/"baseline_vs_proposed_paired_bootstrap.json",paired); _dump(report/"ood_primary_endpoint_bootstrap.json",primary); _dump(report/"ood_noninferiority_bootstrap.json",noninferiority); _dump(report/"mean_predictor_audit.json",mean_audit)
    collapse={}; within_audit={}
    for model in ("Baseline","Proposed"):
        for split in TEST_SPLITS:
            for channel in ("M","H"):
                q=quality[(quality.model==model)&(quality.split==split)&(quality.channel==channel)]; collapse[f"{model}|{split}|{channel}"]={"min_variance_ratio":float(q.variance_ratio.min()),"max_zero_saturation":float(q.zero_saturation_fraction.max()),"max_one_saturation":float(q.one_saturation_fraction.max()),"pass":bool(q.variance_ratio.min()>=.01 and q.zero_saturation_fraction.max()<.99 and q.one_saturation_fraction.max()<.99)}
                w=within[(within.model==model)&(within.split==split)&(within.channel==channel)]; within_audit[f"{model}|{split}|{channel}"]={"mean_within_between_ratio":float(w.within_between_ratio.mean()),"within_lt_between_proportion":float(w.within_lt_between.mean()),"pass":bool(w.within_lt_between.mean()>.5)}
    _dump(report/"collapse_audit.json",collapse); _dump(report/"within_between_audit.json",within_audit); _dump(report/"split_access_audit.json",access)
    after=pt.audit_protected_assets(root,verify_payloads=True); unchanged=before==after; _dump(report/"protected_asset_hash_audit.json",{"before":before,"after":after,"changed":0 if unchanged else 1,"missing":0})
    channel={}
    for label in ("M","H"):
        channel[label]=primary[label]["ci"][0]>0 and noninferiority[label]["pass"] and all(item["pass"] for key,item in mean_audit.items() if key.endswith("|"+label)) and all(item["pass"] for key,item in collapse.items() if key.startswith("Proposed|") and key.endswith("|"+label)) and all(item["pass"] for key,item in within_audit.items() if key.startswith("Proposed|") and key.endswith("|"+label))
    if not unchanged: status="FAIL_PROTECTED_ASSET_MUTATION"; gate="INVALID"
    elif channel["M"] and channel["H"]: status="PASS"; gate="PASS"
    elif channel["M"] or channel["H"]: status="PARTIAL"; gate="PARTIAL"
    else: status="FAIL"; gate="FAIL"
    acceptance={"status":status,"pipeline_validity":"PASS" if unchanged else "FAIL","synthetic_gate":gate,"M_channel_status":"CHANNEL_PASS" if channel["M"] else "CHANNEL_NOT_PASS","H_channel_status":"CHANNEL_PASS" if channel["H"] else "CHANNEL_NOT_PASS","baseline_checkpoint_hash":protocol["baseline_checkpoint_sha256"],"proposed_checkpoint_hash":protocol["proposed_checkpoint_sha256"],"lambda_pair":.5,"B1_training_contract_hash":protocol["b1_contract_hash"],"B2_lambda_selection_lock_hash":protocol["b2_lock_hash"],"D0_AM1_protocol_hash":protocol["d0_protocol_hash"],"G1_AM1_dataset_acceptance_hash":protocol["g1_acceptance_hash"],"test_split_counts":{s:int(sum(r["split"]==s for r in rows)) for s in TEST_SPLITS},"pair_count":pair_audit["pair_total"],"primary_ood_endpoint":primary,"ood_noninferiority_status":{c:"PASS" if noninferiority[c]["pass"] else "FAIL" for c in ("M","H")},"collapse_status":collapse,"protected_assets_unchanged":unchanged,"next_stage":"SO-R2" if status=="PASS" else None,"next_stage_authorized":status=="PASS","SO_R3_authorized":False,"test_ood_access":access}
    _dump(out/"C_ACCEPTANCE.json",acceptance); _dump(out/"evaluation_run_manifest.json",acceptance); (report/"SO_R1_C_Synthetic_ID_OOD_Evaluation_Report.md").write_text("# SO-R1-C Synthetic ID/OOD Evaluation\n\nConclusion boundary: evidence applies only to the frozen synthetic camera spectral response, light SPD, exposure, shading/specular conditions. It does not establish real-device or real-environment generalization.\n\n```json\n"+json.dumps(acceptance,indent=2,default=str)+"\n```\n",encoding="utf-8")
    return acceptance


def write_failure(root: Path, error: BaseException) -> None:
    root=Path(root); out=root/C_DATA; report=root/C_REPORT; out.mkdir(parents=True,exist_ok=True); report.mkdir(parents=True,exist_ok=True); result={"status":"FAIL_PIPELINE_INVALID","pipeline_validity":"FAIL","synthetic_gate":"INVALID","next_stage_authorized":False,"error_type":type(error).__name__,"error":repr(error)}; _dump(out/"C_ACCEPTANCE.json",result); _dump(report/"failure_evidence.json",result)
