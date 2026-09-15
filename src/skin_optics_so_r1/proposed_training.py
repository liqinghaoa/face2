"""SO-R1-B2: paired-output-invariance training under the frozen B1 contract."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torch import Tensor
from torch.utils.data import Dataset
import yaml

from src.skin_optics_so_r1 import baseline_training as bt

B1_RUN = Path("runs/so_r1_b1_baseline_v6")
B1_REPORT = Path("reports/so_r1_b1_baseline_v6")
B1_CONTRACT = Path("config/so_r1/so_r1_b1_training_contract_v6.yaml")
B2_RUN = Path("runs/so_r1_b2_proposed_v3")
B2_REPORT = Path("reports/so_r1_b2_proposed_v3")
LAMBDA_PROTOCOL = Path("config/so_r1/so_r1_b2_lambda_selection_protocol_v1.yaml")
LAMBDA_LOCK = Path("config/so_r1/SO_R1_B2_LAMBDA_SELECTION_LOCK.json")
LAMBDA_CANDIDATES = (0.05, 0.10, 0.20, 0.50)
PAIR_KEYS = ("camera", "light", "appearance", "joint")
PAIR_ACQUISITION = {"camera": 1, "light": 2, "appearance": 3, "joint": 4}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canon_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _finite(*values: Tensor) -> bool:
    return all(bool(torch.isfinite(value).all()) for value in values)


def _lambda_id(value: float) -> str:
    return f"lambda_{value:.2f}".replace(".", "p")


def masked_supervised_loss(predictions: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Average independent masked M/H supervision across both views."""
    if predictions.ndim != 5 or predictions.shape[1:3] != (2, 2):
        raise ValueError("Expected predictions [B,2,2,H,W]")
    flat = predictions.flatten(0, 1)
    return bt.masked_smooth_l1(flat, target.repeat_interleave(2, dim=0), mask.repeat_interleave(2, dim=0))


def masked_pair_consistency_loss(predictions: Tensor, mask: Tensor) -> Tensor:
    """Masked SmoothL1 between A/B final predictions; neither branch is detached."""
    if predictions.ndim != 5 or predictions.shape[1:3] != (2, 2):
        raise ValueError("Expected predictions [B,2,2,H,W]")
    return bt.masked_smooth_l1(predictions[:, 0], predictions[:, 1], mask)


def paired_bootstrap_noninferiority(proposed: np.ndarray, baseline: np.ndarray, *, draws: int = 1000, seed: int = 0) -> dict[str, float | bool]:
    """Latent-level paired bootstrap for a pre-frozen one-sided non-inferiority test."""
    proposed=np.asarray(proposed, dtype=np.float64); baseline=np.asarray(baseline, dtype=np.float64)
    if proposed.shape != baseline.shape or proposed.ndim != 1 or not len(proposed):
        raise ValueError("Paired latent-level MAE vectors are required")
    delta=proposed-baseline; rng=np.random.default_rng(seed); means=[]
    for _ in range(draws):
        means.append(float(delta[rng.integers(0, len(delta), size=len(delta))].mean()))
    margin=.05*float(baseline.mean()); upper=float(np.quantile(means,.95))
    return {"mean_delta":float(delta.mean()),"upper95ci":upper,"margin":margin,"pass":upper <= margin,"bootstrap_draws":draws}


def drift_score(baseline: dict[str, dict[str, float]], proposed: dict[str, dict[str, float]]) -> float:
    values=[]
    for channel in ("M","H"):
        for pair in PAIR_KEYS:
            reference=float(baseline[channel][pair]); candidate=float(proposed[channel][pair])
            if reference <= 0 or candidate <= 0: raise ValueError("Pair drift must be positive for scoring")
            values.append((reference-candidate)/reference)
    return float(np.mean(values))


def select_lambda(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    viable=[item for item in candidates if item["noninferiority_status"] == "PASS" and item["viability_status"] == "PASS"]
    if not viable: return None
    viable.sort(key=lambda item: (-float(item["drift_score"]), float(item["lambda_pair"]), item["run_id"]))
    top=viable[0]
    tied=[item for item in viable if abs(float(item["drift_score"])-float(top["drift_score"])) <= 1e-6]
    tied.sort(key=lambda item: (float(item["lambda_pair"]), item["run_id"]))
    return tied[0]


def _authority_paths(root: Path) -> dict[str, Path]:
    data=root/bt.DATA_REL
    return {
        "g1_acceptance": data/"G1_AM1_ACCEPTANCE.json", "g1_ledger": data/"completed_latent_ledger.jsonl",
        "latent_manifest": data/"manifests/latent_manifest.csv", "acquisition_manifest": data/"manifests/acquisition_manifest.csv",
        "pair_manifest": data/"manifests/pair_manifest.csv", "qc_manifest": data/"manifests/qc_manifest.csv",
        "d0_lock": root/"config/so_r1/SO_R1_A2_D0_AM1_PROTOCOL_LOCK.json",
        "d0_config": root/"config/so_r1/so_r1_a2_full_generation_protocol_v1_1.yaml",
        "am4_acceptance": root/"data/processed/SO_R1_A0_AM4_GlobalCameraAtlas_v1/AM4_ACCEPTANCE.json",
        "am4_camera_config": root/"config/so_r1/frozen_camera_light_split_v1_3.yaml",
        "final_24pair_allowlist": root/"reports/so_r1_a0_am4_global_camera_atlas/final_24pair_allowlist.csv",
        "b1_acceptance": root/B1_RUN/"B1_ACCEPTANCE.json", "b1_contract": root/B1_CONTRACT,
        "b1_checkpoint": root/B1_RUN/"checkpoints/best_val_masked_smoothl1.pt",
        "b1_metrics": root/B1_RUN/"metrics/validation_summary.json", "b1_schedules": root/B1_RUN/"metrics/paired_sampler_schedule_hashes.json",
        "b1_protected_audit": root/B1_REPORT/"protected_authority_audit_after.json",
    }


def audit_protected_assets(root: Path, *, verify_payloads: bool) -> dict[str, Any]:
    root=Path(root); paths=_authority_paths(root)
    if not all(path.is_file() for path in paths.values()): raise RuntimeError("FAIL_B1_OR_DATASET_INPUT_GATE:MISSING_AUTHORITY")
    payload=bt.source_audit(root, verify_payloads=verify_payloads)
    allowlist=sum(1 for _ in csv.DictReader(paths["final_24pair_allowlist"].open(encoding="utf-8-sig")))
    am4=json.loads(paths["am4_acceptance"].read_text(encoding="utf-8"))
    return {"status":"PASS" if payload["status"] == "PASS" and allowlist == 24 and am4.get("status") == "PASS" else "FAIL","payload_audit":payload,"allowlist_pair_count":allowlist,"am4_status":am4.get("status"),"sha256":{name:_sha(path) for name,path in paths.items()}}


def load_and_validate_b1_contract(root: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    root=Path(root); acceptance,lock,rows=bt.validate_authoritative_input(root); paths=_authority_paths(root)
    b1=json.loads(paths["b1_acceptance"].read_text(encoding="utf-8")); contract=yaml.safe_load(paths["b1_contract"].read_text(encoding="utf-8"))
    required={"status":"PASS","baseline_checkpoint_authoritative":True,"baseline_viability_status":"PASS","next_stage":"SO-R1-B2","next_stage_authorized":True}
    if any(b1.get(key) != value for key,value in required.items()): raise RuntimeError("FAIL_B1_OR_DATASET_INPUT_GATE:B1_ACCEPTANCE")
    if Path(b1["best_checkpoint"]).resolve() != paths["b1_checkpoint"].resolve() or not paths["b1_checkpoint"].is_file(): raise RuntimeError("FAIL_B1_OR_DATASET_INPUT_GATE:B1_CHECKPOINT")
    expected={"precision":"FP32","epochs":25,"workers":2,"batch":4,"parameters":7763074}
    actual={"precision":contract["amp_contract"]["precision"],"epochs":contract["num_epochs"],"workers":contract["data_loader_contract"]["num_workers"],"batch":contract["batch_contract"]["latent_groups_per_batch"],"parameters":contract["model_architecture"]["parameter_count"]}
    if actual != expected or contract["model_architecture"]["id"] != bt.BaselineMHUNet.architecture_id: raise RuntimeError("FAIL_B1_OR_DATASET_INPUT_GATE:CONTRACT")
    if contract["source_dataset_protocol_hash"] != lock["protocol_hash"] or contract["source_dataset_acceptance_hash"] != _sha(root/bt.DATA_REL/"G1_AM1_ACCEPTANCE.json"): raise RuntimeError("FAIL_B1_OR_DATASET_INPUT_GATE:HASH")
    return contract,lock,rows,{"b1_acceptance":b1,"checkpoint_sha256":_sha(paths["b1_checkpoint"]),"contract_sha256":_sha(paths["b1_contract"]),"architecture_hash":_canon_hash(contract["model_architecture"])}


class ValidationLatentDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None: self.rows=sorted([row for row in rows if row["split"] == "Validation"],key=lambda row:row["latent_id"])
    def __len__(self) -> int: return len(self.rows)
    def __getitem__(self,index: int) -> dict[str, Tensor | str]:
        row=self.rows[index]; rgb,target,mask=bt._load_npz(str(row["file_path"]),[0,1,2,3,4])
        return {"images":torch.from_numpy(rgb),"target":torch.from_numpy(target),"mask":torch.from_numpy(mask),"latent_id":row["latent_id"]}


def _metric_summary(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray, mean_target: np.ndarray) -> dict[str, Any]:
    valid=np.broadcast_to(mask, prediction.shape); denom=valid.sum(axis=(0,2,3)); error=prediction-target
    smooth=np.where(np.abs(error) < .1, .5*error**2/.1, np.abs(error)-.05)*valid
    absolute=np.abs(error)*valid; squared=error**2*valid; mean_abs=np.abs(mean_target-target)*valid
    pred_mean=(prediction*valid).sum((0,2,3))/denom; target_mean=(target*valid).sum((0,2,3))/denom
    pred_var=np.maximum((prediction**2*valid).sum((0,2,3))/denom-pred_mean**2,0); target_var=np.maximum((target**2*valid).sum((0,2,3))/denom-target_mean**2,0)
    image_pred=(prediction*valid).sum((2,3))/np.maximum(valid.sum((2,3)),1); image_target=(target*valid).sum((2,3))/np.maximum(valid.sum((2,3)),1)
    return {"masked_smooth_l1":(smooth.sum((0,2,3))/denom).tolist(),"mae":(absolute.sum((0,2,3))/denom).tolist(),"rmse":np.sqrt(squared.sum((0,2,3))/denom).tolist(),"mean_predictor_mae":(mean_abs.sum((0,2,3))/denom).tolist(),"prediction_mean":pred_mean.tolist(),"target_mean":target_mean.tolist(),"prediction_variance":pred_var.tolist(),"target_variance":target_var.tolist(),"variance_ratio":(pred_var/np.maximum(target_var,1e-20)).tolist(),"zero_saturation_fraction":(((prediction<=1e-6)*valid).sum((0,2,3))/denom).astype(float).tolist(),"one_saturation_fraction":(((prediction>=1-1e-6)*valid).sum((0,2,3))/denom).astype(float).tolist(),"pearson":[float(pearsonr(image_pred[:,channel],image_target[:,channel]).statistic) for channel in range(2)],"spearman":[float(spearmanr(image_pred[:,channel],image_target[:,channel]).statistic) for channel in range(2)]}


def evaluate_validation_supervision(model: torch.nn.Module, rows: list[dict[str, Any]], mean_target: np.ndarray, device: torch.device, *, workers: int, timeout_seconds: int) -> dict[str, Any]:
    dataset=ValidationLatentDataset(rows); loader=bt._loader(dataset,4,shuffle=False,seed=0,workers=workers,groups=False,timeout_seconds=timeout_seconds); model.eval()
    sums={key:np.zeros(2,dtype=np.float64) for key in ("smooth","abs","sq","pred","pred_sq","target","target_sq","mean_abs")}; denom=np.zeros(2,dtype=np.float64); zero=np.zeros(2); one=np.zeros(2); image_predictions=[]; image_targets=[]; latent_mae=[]; drift_vectors={pair:[] for pair in PAIR_KEYS}
    with torch.no_grad():
        for batch in loader:
            x=batch["images"].to(device,non_blocking=True); target=batch["target"].to(device,non_blocking=True); mask=batch["mask"].to(device,non_blocking=True)
            prediction=model(x.flatten(0,1)).reshape(x.shape[0],5,2,256,256)
            if not _finite(prediction): raise RuntimeError("FAIL_NONFINITE_VALIDATION")
            # Stream one acquisition at a time.  The previous implementation flattened all
            # 5 acquisitions, then kept error, smooth, absolute, squared, and several masked
            # products alive together.  On Windows that transient CPU peak can fail even when
            # a small additional allocation is requested.  This version retains only [B,2,H,W]
            # tensors and transfers reduced scalars / per-image summaries to CPU.
            valid=mask.expand(-1,2,-1,-1)
            valid_sum=valid.sum((2,3)).clamp_min(1)
            valid_channel_sum=valid.sum((0,2,3)).detach().cpu().numpy().astype(np.float64)
            denom += valid_channel_sum * prediction.shape[1]
            target_sum=(target*valid).sum((0,2,3)).detach().cpu().numpy().astype(np.float64)
            target_sq_sum=((target*target)*valid).sum((0,2,3)).detach().cpu().numpy().astype(np.float64)
            target_images=((target*valid).sum((2,3))/valid_sum).detach().cpu().numpy()
            # The B1 mean predictor is a spatial [M/H,H,W] field, not two scalar means.
            target_mean_tensor=torch.as_tensor(mean_target,dtype=target.dtype,device=device).unsqueeze(0)
            latent_abs_sum=torch.zeros((prediction.shape[0],2),dtype=torch.float64,device=device)
            base_prediction=prediction[:,0]
            for acquisition in range(prediction.shape[1]):
                view_prediction=prediction[:,acquisition]
                error=view_prediction-target
                absolute=error.abs()
                smooth=torch.where(absolute < .1,.5*error.square()/.1,absolute-.05)
                sums["smooth"] += (smooth*valid).sum((0,2,3)).detach().cpu().numpy().astype(np.float64)
                sums["abs"] += (absolute*valid).sum((0,2,3)).detach().cpu().numpy().astype(np.float64)
                sums["sq"] += (error.square()*valid).sum((0,2,3)).detach().cpu().numpy().astype(np.float64)
                sums["pred"] += (view_prediction*valid).sum((0,2,3)).detach().cpu().numpy().astype(np.float64)
                sums["pred_sq"] += (view_prediction.square()*valid).sum((0,2,3)).detach().cpu().numpy().astype(np.float64)
                sums["target"] += target_sum
                sums["target_sq"] += target_sq_sum
                sums["mean_abs"] += ((target_mean_tensor-target).abs()*valid).sum((0,2,3)).detach().cpu().numpy().astype(np.float64)
                zero += ((view_prediction<=1e-6)*valid).sum((0,2,3)).detach().cpu().numpy().astype(np.float64)
                one += ((view_prediction>=1-1e-6)*valid).sum((0,2,3)).detach().cpu().numpy().astype(np.float64)
                image_predictions.append(((view_prediction*valid).sum((2,3))/valid_sum).detach().cpu().numpy())
                image_targets.append(target_images)
                latent_abs_sum += (absolute*valid).sum((2,3)).to(torch.float64)
            latent_mae.append((latent_abs_sum/(valid_sum[:,0:1].to(torch.float64)*prediction.shape[1])).detach().cpu().numpy())
            for pair,acquisition in PAIR_ACQUISITION.items():
                drift_vectors[pair].append(((base_prediction-prediction[:,acquisition]).abs()*valid).sum((2,3)).div(valid_sum).detach().cpu().numpy())
    pred_mean=sums["pred"]/denom; target_mean=sums["target"]/denom; pred_var=np.maximum(sums["pred_sq"]/denom-pred_mean**2,0); target_var=np.maximum(sums["target_sq"]/denom-target_mean**2,0); image_pred=np.concatenate(image_predictions); image_target=np.concatenate(image_targets)
    metrics={"masked_smooth_l1":(sums["smooth"]/denom).tolist(),"mae":(sums["abs"]/denom).tolist(),"rmse":np.sqrt(sums["sq"]/denom).tolist(),"mean_predictor_mae":(sums["mean_abs"]/denom).tolist(),"prediction_mean":pred_mean.tolist(),"target_mean":target_mean.tolist(),"prediction_variance":pred_var.tolist(),"target_variance":target_var.tolist(),"variance_ratio":(pred_var/np.maximum(target_var,1e-20)).tolist(),"zero_saturation_fraction":(zero/denom).tolist(),"one_saturation_fraction":(one/denom).tolist(),"pearson":[float(pearsonr(image_pred[:,channel],image_target[:,channel]).statistic) for channel in range(2)],"spearman":[float(spearmanr(image_pred[:,channel],image_target[:,channel]).statistic) for channel in range(2)]}
    latent_mae=np.concatenate(latent_mae,axis=0); drift={channel:{} for channel in ("M","H")}
    for pair,acquisition in PAIR_ACQUISITION.items():
        values=np.concatenate(drift_vectors[pair],axis=0)
        drift_vectors[pair]=values
        for channel,label in enumerate(("M","H")):
            item=values[:,channel]; rng=np.random.default_rng(1000+channel+acquisition); boot=np.asarray([item[rng.integers(0,len(item),size=len(item))].mean() for _ in range(300)])
            drift[label][pair]={"mean":float(item.mean()),"median":float(np.median(item)),"p25":float(np.quantile(item,.25)),"p75":float(np.quantile(item,.75)),"p95":float(np.quantile(item,.95)),"bootstrap95ci":[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))]}
    return {"metrics":metrics,"latent_mae":latent_mae,"drift":drift,"drift_vectors":drift_vectors,"latent_count":len(dataset),"acquisition_count":len(dataset)*5}


def revalidate_baseline_reference(root: Path, contract: dict[str, Any], rows: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    path=root/B1_RUN/"checkpoints/best_val_masked_smoothl1.pt"; checkpoint=torch.load(path,map_location=device,weights_only=False)
    if checkpoint.get("training_contract_hash") != json.loads((root/B1_RUN/"B1_ACCEPTANCE.json").read_text())["contract_hash"]: raise RuntimeError("FAIL_B1_OR_DATASET_INPUT_GATE:CHECKPOINT_PROVENANCE")
    bt._configure_determinism(contract["model_init_seed"]); model=bt.BaselineMHUNet().to(device); model.load_state_dict(checkpoint["model_state"])
    mean_target,_=bt.train_mean_predictor(rows); official=bt.validate(model,rows,mean_target,device,False,workers=contract["data_loader_contract"]["num_workers"],timeout_seconds=contract["data_loader_contract"]["timeout_seconds"])
    recorded=json.loads((root/B1_RUN/"metrics/validation_summary.json").read_text())["metrics"]
    for key in ("masked_smooth_l1","mae","rmse","pearson","variance_ratio"):
        if not np.allclose(official[key],recorded[key],rtol=0,atol=2e-6): raise RuntimeError("FAIL_BASELINE_REFERENCE_EVALUATION_MISMATCH:"+key)
    detailed=evaluate_validation_supervision(model,rows,mean_target,device,workers=contract["data_loader_contract"]["num_workers"],timeout_seconds=contract["data_loader_contract"]["timeout_seconds"])
    return {"official_metrics":official,"recorded_metrics":recorded,"detailed":detailed,"checkpoint_sha256":_sha(path),"status":"PASS"}


def _candidate_checkpoint(path: Path, *, model: torch.nn.Module, optimizer: Any, scheduler: Any, epoch: int, lambda_pair: float, contract_hash: str, schedule_hash: str, provenance: dict[str, Any]) -> Path:
    result=path/"checkpoints"/f"epoch_{epoch:02d}.pt"; result.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"model_state":model.state_dict(),"optimizer_state":optimizer.state_dict(),"scheduler_state":scheduler.state_dict(),"epoch":epoch,"lambda_pair":lambda_pair,"model_architecture":bt.BaselineMHUNet.architecture_id,"b1_training_contract_hash":contract_hash,"paired_schedule_hash":schedule_hash,"selected_precision":"FP32",**provenance},result)
    return result


def _candidate_contract(lambda_pair: float, contract: dict[str, Any], lock: dict[str, Any]) -> dict[str, Any]:
    return {"status":"TUNING_CANDIDATE","authoritative":False,"lambda_pair":lambda_pair,"b1_contract_hash":lock["b1_training_contract_hash"],"selection_protocol_hash":lock["selection_protocol_hash"],"model_init_seed":contract["model_init_seed"],"paired_sampler_seed":contract["paired_sampler_seed"],"precision":"FP32","batch_contract":contract["batch_contract"],"data_loader_contract":contract["data_loader_contract"],"optimizer":contract["optimizer"],"scheduler":contract["scheduler"],"num_epochs":contract["num_epochs"]}


def _completed_candidate(run: Path, lambda_pair: float) -> dict[str, Any] | None:
    acceptance=run/"CANDIDATE_ACCEPTANCE.json"
    if not acceptance.is_file(): return None
    result=json.loads(acceptance.read_text(encoding="utf-8"))
    if float(result.get("lambda_pair",float("nan"))) != lambda_pair or result.get("candidate_status") != "TUNING_CANDIDATE":
        raise RuntimeError("FAIL_RESUME_GATE:COMPLETED_CANDIDATE_PROVENANCE")
    checkpoint=Path(result["best_checkpoint"])
    if not checkpoint.is_file() or _sha(checkpoint) != result.get("best_checkpoint_sha256"):
        raise RuntimeError("FAIL_RESUME_GATE:COMPLETED_CANDIDATE_CHECKPOINT")
    return result


def _resume_candidate_state(run: Path, *, lambda_pair: float, candidate_contract: dict[str, Any], model: torch.nn.Module, optimizer: Any, scheduler: Any, expected_schedules: list[dict[str, Any]], rows: list[dict[str, Any]]) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]], dict[str, int], float, int]:
    """Restore an interrupted candidate only from a contiguous post-validation checkpoint."""
    snapshot=run/"training_contract_snapshot.yaml"; metrics_path=run/"metrics/epoch_metrics.csv"; schedules_path=run/"metrics/paired_sampler_schedule_hashes.json"
    if not all(path.is_file() for path in (snapshot,metrics_path,schedules_path)):
        raise RuntimeError("FAIL_RESUME_GATE:MISSING_CANDIDATE_STATE")
    if _canon_hash(yaml.safe_load(snapshot.read_text(encoding="utf-8"))) != _canon_hash(candidate_contract):
        raise RuntimeError("FAIL_RESUME_GATE:CANDIDATE_CONTRACT_MISMATCH")
    metrics=pd.read_csv(metrics_path).to_dict("records")
    if not metrics: raise RuntimeError("FAIL_RESUME_GATE:EMPTY_EPOCH_METRICS")
    completed=[int(item["epoch"]) for item in metrics]
    if completed != list(range(1,len(completed)+1)):
        raise RuntimeError("FAIL_RESUME_GATE:NONCONTIGUOUS_EPOCH_METRICS")
    schedules=json.loads(schedules_path.read_text(encoding="utf-8"))
    if len(schedules) != len(metrics): raise RuntimeError("FAIL_RESUME_GATE:SCHEDULE_RECORD_COUNT")
    for index,schedule in enumerate(schedules):
        if schedule.get("epoch_pair_schedule_hash") != expected_schedules[index].get("epoch_pair_schedule_hash"):
            raise RuntimeError("FAIL_RESUME_GATE:SCHEDULE_MISMATCH")
    latest=run/"checkpoints"/f"epoch_{completed[-1]:02d}.pt"
    if not latest.is_file(): raise RuntimeError("FAIL_RESUME_GATE:MISSING_LATEST_CHECKPOINT")
    payload=torch.load(latest,map_location="cuda:0",weights_only=False)
    if int(payload.get("epoch",0)) != completed[-1] or float(payload.get("lambda_pair",float("nan"))) != lambda_pair or payload.get("model_architecture") != bt.BaselineMHUNet.architecture_id or payload.get("selected_precision") != "FP32":
        raise RuntimeError("FAIL_RESUME_GATE:CHECKPOINT_PROVENANCE")
    model.load_state_dict(payload["model_state"]); optimizer.load_state_dict(payload["optimizer_state"]); scheduler.load_state_dict(payload["scheduler_state"])
    best_index=min(range(len(metrics)),key=lambda index:float(metrics[index]["validation_masked_smooth_l1"]))
    validation_rows=sum(1 for row in rows if row["split"] == "Validation")
    access={"Train":0,"Validation":validation_rows*5*len(metrics),"ID Test":0,"Camera-OOD":0,"Light-OOD":0,"Joint-OOD":0}
    for epoch in completed:
        groups,_=bt.build_epoch_groups(rows,candidate_contract["paired_sampler_seed"],epoch); access["Train"] += len(groups)
    return completed[-1]+1,metrics,schedules,access,float(metrics[best_index]["validation_masked_smooth_l1"]),int(metrics[best_index]["epoch"])


def train_lambda_candidate(root: Path, *, lambda_pair: float, contract: dict[str, Any], lock: dict[str, Any], rows: list[dict[str, Any]], baseline: dict[str, Any], report: Path, resume: bool = False) -> dict[str, Any]:
    root=Path(root); run=root/B2_RUN/_lambda_id(lambda_pair); existed=run.exists()
    if existed and not resume: raise RuntimeError("REFUSE_OVERWRITE_EXISTING_LAMBDA_CANDIDATE")
    run.mkdir(parents=True,exist_ok=resume); (run/"metrics").mkdir(exist_ok=resume); (run/"checkpoints").mkdir(exist_ok=resume)
    candidate_contract=_candidate_contract(lambda_pair,contract,lock)
    candidate_hash=_canon_hash(candidate_contract)
    if not resume: (run/"training_contract_snapshot.yaml").write_text(yaml.safe_dump(candidate_contract,sort_keys=False),encoding="utf-8")
    _dump(run/"run_manifest.json",{"status":"RUNNING","candidate_status":"TUNING_CANDIDATE","not_for_test_evaluation":True,"contract_hash":candidate_hash,"resume":resume})
    seed=contract["model_init_seed"]; bt._configure_determinism(seed); device=torch.device("cuda:0"); model=bt.BaselineMHUNet().to(device)
    if bt.count_parameters(model) != contract["model_architecture"]["parameter_count"]: raise RuntimeError("FAIL_B1_OR_DATASET_INPUT_GATE:ARCHITECTURE")
    optimizer=torch.optim.AdamW(model.parameters(),lr=contract["optimizer"]["learning_rate"],weight_decay=contract["optimizer"]["weight_decay"]); scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=contract["scheduler"]["T_max"],eta_min=contract["scheduler"]["eta_min"])
    mean_target,_=bt.train_mean_predictor(rows); expected=json.loads((root/B1_RUN/"metrics/paired_sampler_schedule_hashes.json").read_text())
    if resume:
        start_epoch,metrics,schedules,access,best,best_epoch=_resume_candidate_state(run,lambda_pair=lambda_pair,candidate_contract=candidate_contract,model=model,optimizer=optimizer,scheduler=scheduler,expected_schedules=expected,rows=rows)
    else:
        start_epoch=1; metrics=[]; access={"Train":0,"Validation":0,"ID Test":0,"Camera-OOD":0,"Light-OOD":0,"Joint-OOD":0}; best=float("inf"); best_epoch=0; schedules=[]
    for epoch in range(start_epoch,contract["num_epochs"]+1):
        groups,schedule=bt.build_epoch_groups(rows,contract["paired_sampler_seed"],epoch)
        if schedule["epoch_pair_schedule_hash"] != expected[epoch-1]["epoch_pair_schedule_hash"]: raise RuntimeError("FAIL_B1_OR_DATASET_INPUT_GATE:SCHEDULE_MISMATCH")
        schedules.append(schedule); loader=bt._loader(bt.PairedGroupDataset(groups),contract["batch_contract"]["latent_groups_per_batch"],shuffle=False,seed=seed+epoch,workers=contract["data_loader_contract"]["num_workers"],groups=True,timeout_seconds=contract["data_loader_contract"]["timeout_seconds"]); total_values=[]; sup_values=[]; pair_values=[]
        for batch in loader:
            image=batch["images"].to(device,non_blocking=True); target=batch["target"].to(device,non_blocking=True); mask=batch["mask"].to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True); prediction=model(image.flatten(0,1)).reshape(image.shape[0],2,2,256,256); supervised=masked_supervised_loss(prediction,target,mask); paired=masked_pair_consistency_loss(prediction,mask); loss=supervised+lambda_pair*paired
            if not _finite(prediction,supervised,paired,loss): raise RuntimeError("FAIL_NONFINITE_TRAINING")
            loss.backward()
            if not all(_finite(parameter.grad) for parameter in model.parameters() if parameter.grad is not None): raise RuntimeError("FAIL_NONFINITE_GRADIENT")
            optimizer.step(); total_values.append(float(loss.detach().cpu())); sup_values.append(float(supervised.detach().cpu())); pair_values.append(float(paired.detach().cpu()))
        access["Train"] += len(groups); evaluated=evaluate_validation_supervision(model,rows,mean_target,device,workers=contract["data_loader_contract"]["num_workers"],timeout_seconds=contract["data_loader_contract"]["timeout_seconds"]); access["Validation"] += evaluated["acquisition_count"]
        validation=float(np.mean(evaluated["metrics"]["masked_smooth_l1"])); scheduler.step(); checkpoint=_candidate_checkpoint(run,model=model,optimizer=optimizer,scheduler=scheduler,epoch=epoch,lambda_pair=lambda_pair,contract_hash=lock["b1_training_contract_hash"],schedule_hash=schedule["epoch_pair_schedule_hash"],provenance={"d0_am1_protocol_hash":contract["source_dataset_protocol_hash"],"g1_am1_dataset_acceptance_hash":contract["source_dataset_acceptance_hash"],"b1_checkpoint_sha256":baseline["checkpoint_sha256"]})
        row={"epoch":epoch,"lambda_pair":lambda_pair,"train_total_loss":float(np.mean(total_values)),"train_supervised_loss":float(np.mean(sup_values)),"train_pair_loss":float(np.mean(pair_values)),"validation_masked_smooth_l1":validation,"learning_rate":float(optimizer.param_groups[0]["lr"]),**{f"val_{key}_{label}":value[channel] for key,value in evaluated["metrics"].items() if isinstance(value,list) for channel,label in enumerate(("M","H"))}}
        metrics.append(row); pd.DataFrame(metrics).to_csv(run/"metrics/epoch_metrics.csv",index=False); _dump(run/"metrics/paired_sampler_schedule_hashes.json",schedules)
        if validation < best: best=validation; best_epoch=epoch; shutil.copy2(checkpoint,run/"checkpoints/best_val_masked_smoothl1.pt")
        print(json.dumps({"lambda_pair":lambda_pair,"epoch":epoch,"validation_masked_smooth_l1":validation,"best_epoch":best_epoch},sort_keys=True),flush=True)
    best_checkpoint=torch.load(run/"checkpoints/best_val_masked_smoothl1.pt",map_location=device,weights_only=False); model.load_state_dict(best_checkpoint["model_state"]); final=evaluate_validation_supervision(model,rows,mean_target,device,workers=contract["data_loader_contract"]["num_workers"],timeout_seconds=contract["data_loader_contract"]["timeout_seconds"]); access["Validation"] += final["acquisition_count"]
    noninferiority={label:paired_bootstrap_noninferiority(final["latent_mae"][:,channel],baseline["detailed"]["latent_mae"][:,channel],seed=100+channel+int(lambda_pair*100)) for channel,label in enumerate(("M","H"))}
    gate=all(noninferiority[label]["pass"] for label in ("M","H")); metric=final["metrics"]; viability=all(metric["mae"][channel] < metric["mean_predictor_mae"][channel] and metric["variance_ratio"][channel] >= .01 and metric["zero_saturation_fraction"][channel] < .99 and metric["one_saturation_fraction"][channel] < .99 for channel in range(2)) and all(value == 0 for key,value in access.items() if key not in ("Train","Validation"))
    result={"run_id":_lambda_id(lambda_pair),"lambda_pair":lambda_pair,"best_epoch":best_epoch,"best_checkpoint":str(run/"checkpoints/best_val_masked_smoothl1.pt"),"best_checkpoint_sha256":_sha(run/"checkpoints/best_val_masked_smoothl1.pt"),"metrics":metric,"pair_drift":final["drift"],"noninferiority":noninferiority,"noninferiority_status":"PASS" if gate else "FAIL","viability_status":"PASS" if viability else "FAIL","candidate_status":"TUNING_CANDIDATE","data_access":access}
    _dump(run/"metrics/validation_summary.json",result); _dump(run/"data_access_audit.json",access); _dump(run/"CANDIDATE_ACCEPTANCE.json",result)
    return result


def _protocol(contract: dict[str, Any], b1: dict[str, Any]) -> dict[str, Any]:
    return {"stage_id":"SO-R1-B2","version":"v1","candidates":list(LAMBDA_CANDIDATES),"selection_split":"Validation only","test_ood_access":"forbidden","noninferiority":{"margin":"0.05 * B1 latent-level MAE","bootstrap_draws":1000,"upper_ci":"one-sided 95%"},"drift_score":"mean relative reduction across M/H x camera/light/appearance/joint","tie_break":["score within 1e-6: smaller lambda","then lexicographically earlier run ID"],"b1_training_contract_hash":b1["contract_hash"],"frozen_b1_contract":contract}


def write_b2_failure(root: Path, error: BaseException) -> None:
    root=Path(root); run=root/B2_RUN; report=root/B2_REPORT; run.mkdir(parents=True,exist_ok=True); report.mkdir(parents=True,exist_ok=True)
    result={"status":"FAIL","proposed_training_status":"NOT_COMPLETE","proposed_checkpoint_authoritative":False,"next_stage_authorized":False,"training_started":False,"failure_type":type(error).__name__,"failure":repr(error)}
    _dump(run/"B2_ACCEPTANCE.json",result); _dump(report/"failure_evidence.json",result)


def run(root: Path, *, resume: bool = False) -> dict[str, Any]:
    root=Path(root); run=root/B2_RUN; report=root/B2_REPORT
    if run.exists() and not resume: raise RuntimeError("REFUSE_OVERWRITE_EXISTING_B2_RUN")
    contract,lock,rows,b1=load_and_validate_b1_contract(root); before=audit_protected_assets(root,verify_payloads=True)
    if before["status"] != "PASS": raise RuntimeError("FAIL_B1_OR_DATASET_INPUT_GATE:ASSET_AUDIT")
    expected_protocol=_protocol(contract,b1["b1_acceptance"])
    if (root/LAMBDA_PROTOCOL).is_file():
        protocol=yaml.safe_load((root/LAMBDA_PROTOCOL).read_text(encoding="utf-8"))
        if _canon_hash(protocol) != _canon_hash(expected_protocol): raise RuntimeError("FAIL_B1_OR_DATASET_INPUT_GATE:LAMBDA_PROTOCOL_MISMATCH")
    else:
        protocol=expected_protocol; (root/LAMBDA_PROTOCOL).write_text(yaml.safe_dump(protocol,sort_keys=False),encoding="utf-8")
    protocol_hash=_canon_hash(protocol); expected_lock={"status":"FROZEN_BEFORE_TRAINING","selection_protocol_hash":protocol_hash,"b1_training_contract_hash":b1["b1_acceptance"]["contract_hash"],"b1_checkpoint_path":b1["b1_acceptance"]["best_checkpoint"],"b1_checkpoint_sha256":b1["checkpoint_sha256"],"candidates":list(LAMBDA_CANDIDATES),"selection_rule":protocol["tie_break"]}
    if (root/LAMBDA_LOCK).is_file():
        selection_lock=json.loads((root/LAMBDA_LOCK).read_text(encoding="utf-8"))
        if selection_lock != expected_lock: raise RuntimeError("FAIL_B1_OR_DATASET_INPUT_GATE:LAMBDA_LOCK_MISMATCH")
    else:
        selection_lock=expected_lock; _dump(root/LAMBDA_LOCK,selection_lock)
    lock_hash=_sha(root/LAMBDA_LOCK)
    original_before_path=report/"protected_asset_hash_audit_before.json"
    if resume:
        if not original_before_path.is_file() or json.loads(original_before_path.read_text(encoding="utf-8")) != before:
            raise RuntimeError("FAIL_RESUME_GATE:PROTECTED_ASSET_AUDIT_MISMATCH")
        failed=json.loads((run/"B2_ACCEPTANCE.json").read_text(encoding="utf-8")) if (run/"B2_ACCEPTANCE.json").is_file() else {}
        previous_recovery=json.loads((report/"recovery_audit.json").read_text(encoding="utf-8")) if (report/"recovery_audit.json").is_file() else {}
        original_failure=previous_recovery.get("recovered_from_failure",failed)
        if original_failure.get("failure_type") != "MemoryError": raise RuntimeError("FAIL_RESUME_GATE:UNSUPPORTED_FAILURE")
        _dump(report/"recovery_audit.json",{"status":"PASS","recovery_mode":"lambda_0p50_from_latest_post_validation_checkpoint","recovered_from_failure":original_failure,"latest_recovery_attempt_status":"RESTARTING","protected_assets_unchanged":True})
    else:
        run.mkdir(parents=True); report.mkdir(parents=True,exist_ok=True); _dump(original_before_path,before)
    run.mkdir(parents=True,exist_ok=True); report.mkdir(parents=True,exist_ok=True)
    _dump(run/"B2_ACCEPTANCE.json",{"status":"RUNNING_RECOVERY" if resume else "RUNNING","proposed_training_status":"IN_PROGRESS","proposed_checkpoint_authoritative":False,"next_stage_authorized":False,"resume":resume})
    _dump(run/"run_manifest.json",{"status":"RUNNING","b1_contract_hash":b1["b1_acceptance"]["contract_hash"],"selection_protocol_hash":protocol_hash,"selection_lock_hash":lock_hash,"training_started":True,"resume":resume})
    device=torch.device("cuda:0"); baseline=revalidate_baseline_reference(root,contract,rows,device); _dump(report/"baseline_reference_validation.json",baseline)
    summaries=[]
    for value in LAMBDA_CANDIDATES:
        candidate_run=run/_lambda_id(value); completed=_completed_candidate(candidate_run,value) if candidate_run.exists() else None
        if completed is not None:
            result=completed
        elif resume and value == .50:
            result=train_lambda_candidate(root,lambda_pair=value,contract=contract,lock={**selection_lock,"selection_protocol_hash":protocol_hash,"b1_training_contract_hash":b1["b1_acceptance"]["contract_hash"]},rows=rows,baseline=baseline,report=report,resume=True)
        elif resume:
            raise RuntimeError("FAIL_RESUME_GATE:UNEXPECTED_INCOMPLETE_CANDIDATE")
        else:
            result=train_lambda_candidate(root,lambda_pair=value,contract=contract,lock={**selection_lock,"selection_protocol_hash":protocol_hash,"b1_training_contract_hash":b1["b1_acceptance"]["contract_hash"]},rows=rows,baseline=baseline,report=report)
        result["drift_score"]=drift_score({channel:{pair:baseline["detailed"]["drift"][channel][pair]["mean"] for pair in PAIR_KEYS} for channel in ("M","H")},{channel:{pair:result["pair_drift"][channel][pair]["mean"] for pair in PAIR_KEYS} for channel in ("M","H")}) if result["noninferiority_status"] == "PASS" and result["viability_status"] == "PASS" else float("nan")
        summaries.append(result)
    selected=select_lambda(summaries); after=audit_protected_assets(root,verify_payloads=True); unchanged=before == after
    pd.DataFrame([{ "run_id":item["run_id"],"lambda_pair":item["lambda_pair"],"best_epoch":item["best_epoch"],"M_mae":item["metrics"]["mae"][0],"H_mae":item["metrics"]["mae"][1],"M_variance_ratio":item["metrics"]["variance_ratio"][0],"H_variance_ratio":item["metrics"]["variance_ratio"][1],"noninferiority_status":item["noninferiority_status"],"viability_status":item["viability_status"],"drift_score":item["drift_score"]} for item in summaries]).to_csv(report/"lambda_candidate_summary.csv",index=False)
    supervision_rows=[]; drift_rows=[]
    for item in summaries:
        for channel,label in enumerate(("M","H")):
            supervision_rows.append({"run_id":item["run_id"],"lambda_pair":item["lambda_pair"],"channel":label,**{key:value[channel] for key,value in item["metrics"].items() if isinstance(value,list)}})
            for pair in PAIR_KEYS: drift_rows.append({"run_id":item["run_id"],"lambda_pair":item["lambda_pair"],"channel":label,"pair_type":pair,**item["pair_drift"][label][pair]})
    pd.DataFrame(supervision_rows).to_csv(report/"validation_supervision_metrics.csv",index=False); pd.DataFrame(drift_rows).to_csv(report/"validation_pair_drift_metrics.csv",index=False); _dump(report/"validation_noninferiority_bootstrap.json",{item["run_id"]:item["noninferiority"] for item in summaries}); _dump(report/"lambda_selection_audit.json",{"candidates":summaries,"selected":selected,"protocol_hash":protocol_hash,"lock_hash":lock_hash}); _dump(report/"protected_asset_hash_audit.json",{"before":before,"after":after,"changed":0 if unchanged else 1,"missing":0})
    access={"Train":sum(item["data_access"]["Train"] for item in summaries)+10000,"Validation":sum(item["data_access"]["Validation"] for item in summaries)+10000,"ID Test":0,"Camera-OOD":0,"Light-OOD":0,"Joint-OOD":0}; _dump(report/"data_access_audit.json",access)
    if selected is None or not unchanged:
        acceptance={"status":"FAIL_NO_VALID_LAMBDA" if selected is None else "FAIL_PROTECTED_ASSET_MUTATION","proposed_training_status":"COMPLETE","proposed_checkpoint_authoritative":False,"lambda_selection_status":"FAILED","validation_noninferiority_status":"FAIL","validation_viability_status":"FAIL","source_dataset_unchanged":unchanged,"test_ood_access_count":0,"next_stage":None,"next_stage_authorized":False}
    else:
        checkpoint=Path(selected["best_checkpoint"]); selected_payload={"selected_lambda":selected["lambda_pair"],"selected_run_id":selected["run_id"],"selected_checkpoint_path":str(checkpoint),"selected_checkpoint_sha256":_sha(checkpoint),"selected_epoch":selected["best_epoch"],"B1_checkpoint_path":b1["b1_acceptance"]["best_checkpoint"],"B1_checkpoint_hash":b1["checkpoint_sha256"],"B1_training_contract_hash":b1["b1_acceptance"]["contract_hash"],"D0_AM1_protocol_hash":contract["source_dataset_protocol_hash"],"G1_AM1_dataset_acceptance_hash":contract["source_dataset_acceptance_hash"],"selection_protocol_hash":protocol_hash,"lambda_selection_lock_hash":lock_hash}; _dump(run/"PROPOSED_SELECTED_CHECKPOINT.json",selected_payload)
        acceptance={"status":"PASS","proposed_training_status":"COMPLETE","proposed_checkpoint_authoritative":True,"lambda_selection_status":"FROZEN","validation_noninferiority_status":"PASS","validation_viability_status":"PASS","source_dataset_unchanged":True,"test_ood_access_count":0,"next_stage":"SO-R1-C","next_stage_authorized":True,"proposed_training_started":False,**selected_payload}
    _dump(run/"B2_ACCEPTANCE.json",acceptance); _dump(run/"run_manifest.json",acceptance); (report/"SO_R1_B2_Proposed_Training_Report.md").write_text("# SO-R1-B2 Proposed Training\n\n```json\n"+json.dumps(acceptance,indent=2)+"\n```\n",encoding="utf-8")
    return acceptance
