"""SO-R1-B1: isolated supervised Baseline training on accepted G1-AM1 data."""
from __future__ import annotations

import csv, hashlib, json, os, platform, random, shutil, subprocess, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
import yaml

ROOT_SEED_SALT = "SO-R1-B1 paired comparison seed v1"
DATA_REL = Path("data/processed/SO_R1_A2_G1_AM1_FullGeneration_v1")
RUN_REL = Path("runs/so_r1_b1_baseline_v6")
REPORT_REL = Path("reports/so_r1_b1_baseline_v6")
STABILITY_REPORT_REL = Path("reports/so_r1_b1_baseline_v3")
WORKERS2_PREFLIGHT_RUN_REL = Path("runs/so_r1_b1_workers2_preflight_v2")
WORKERS2_PREFLIGHT_REPORT_REL = Path("reports/so_r1_b1_workers2_preflight_v2")
DATA_LOADER_TIMEOUT_SECONDS = 120


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canon_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _seed_from_protocol(protocol_hash: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{ROOT_SEED_SALT}|{protocol_hash}".encode()).digest()[:8], "big") % (2**31 - 1)


def _configure_determinism(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _worker_seed(worker_id: int) -> None:
    value = (torch.initial_seed() + worker_id) % (2**32)
    random.seed(value); np.random.seed(value)


class ConvNormAct(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        groups = min(8, cout)
        self.layers = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(groups, cout), nn.LeakyReLU(0.01, inplace=False),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(groups, cout), nn.LeakyReLU(0.01, inplace=False),
        )
    def forward(self, x: Tensor) -> Tensor: return self.layers(x)


class Down(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__(); self.layers = nn.Sequential(nn.MaxPool2d(2), ConvNormAct(cin, cout))
    def forward(self, x: Tensor) -> Tensor: return self.layers(x)


class Up(nn.Module):
    def __init__(self, cin: int, skip: int, cout: int) -> None:
        super().__init__(); self.up = nn.ConvTranspose2d(cin, cout, 2, stride=2); self.conv = ConvNormAct(cout + skip, cout)
    def forward(self, x: Tensor, skip: Tensor) -> Tensor: return self.conv(torch.cat((skip, self.up(x)), dim=1))


class BaselineMHUNet(nn.Module):
    """Randomly initialized standard 2D GroupNorm U-Net, 3 -> [M,H]."""
    architecture_id = "SO_R1_B1_UNet_GN_32_64_128_256_512_3to2_sigmoid_v1"
    output_order = ["M-sensitive", "H-sensitive"]
    def __init__(self) -> None:
        super().__init__()
        self.inc = ConvNormAct(3, 32); self.d1 = Down(32, 64); self.d2 = Down(64, 128); self.d3 = Down(128, 256); self.d4 = Down(256, 512)
        self.u1 = Up(512, 256, 256); self.u2 = Up(256, 128, 128); self.u3 = Up(128, 64, 64); self.u4 = Up(64, 32, 32)
        self.head = nn.Conv2d(32, 2, 1); self.activation = nn.Sigmoid(); self.reset_parameters()
    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, a=.01, mode="fan_out", nonlinearity="leaky_relu")
                if module.bias is not None: nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm): nn.init.ones_(module.weight); nn.init.zeros_(module.bias)
    def forward(self, x: Tensor) -> Tensor:
        a=self.inc(x); b=self.d1(a); c=self.d2(b); d=self.d3(c); e=self.d4(d)
        return self.activation(self.head(self.u4(self.u3(self.u2(self.u1(e,d),c),b),a)))


def count_parameters(model: nn.Module) -> int: return int(sum(p.numel() for p in model.parameters()))


@dataclass(frozen=True)
class Group:
    latent_id: str
    split: str
    file_path: str
    target_index: int
    pair_type: str


PAIR_INDEX = {"camera_only": 1, "light_only": 2, "appearance_only": 3, "joint": 4}
PAIR_TYPES = tuple(PAIR_INDEX)


def _load_npz(path: str, indices: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        rgb = z["rgb"][indices].astype(np.float32)
        target = np.stack((z["m"], z["h"])).astype(np.float32)
        mask = z["mask"].astype(np.float32)[None]
    return rgb, target, mask


class PairedGroupDataset(Dataset):
    def __init__(self, groups: list[Group]) -> None: self.groups = groups
    def __len__(self) -> int: return len(self.groups)
    def __getitem__(self, index: int) -> dict[str, Tensor]:
        group = self.groups[index]; rgb, target, mask = _load_npz(group.file_path, [0, group.target_index])
        return {"images": torch.from_numpy(rgb), "target": torch.from_numpy(target), "mask": torch.from_numpy(mask), "latent_id": group.latent_id, "pair_type": group.pair_type}


class ValidationDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None: self.rows = rows
    def __len__(self) -> int: return len(self.rows) * 5
    def __getitem__(self, index: int) -> dict[str, Tensor]:
        row, acq = self.rows[index // 5], index % 5; rgb, target, mask = _load_npz(str(row["file_path"]), [acq])
        return {"image": torch.from_numpy(rgb[0]), "target": torch.from_numpy(target), "mask": torch.from_numpy(mask), "latent_id": row["latent_id"]}


def masked_smooth_l1(pred: Tensor, target: Tensor, mask: Tensor, beta: float = .1) -> Tensor:
    valid = mask.expand_as(pred); loss = torch.nn.functional.smooth_l1_loss(pred, target, beta=beta, reduction="none")
    by_channel = (loss * valid).sum(dim=(0, 2, 3)) / valid.sum(dim=(0, 2, 3)).clamp_min(1.0)
    return by_channel.mean()


def _manifest_hashes(root: Path) -> dict[str, str]:
    m = root / DATA_REL / "manifests"
    return {name: _sha(m / name) for name in ("latent_manifest.csv", "acquisition_manifest.csv", "pair_manifest.csv", "qc_manifest.csv")}


def source_audit(root: Path, *, verify_payloads: bool) -> dict[str, Any]:
    data = root / DATA_REL; lock = root / "config/so_r1/SO_R1_A2_D0_AM1_PROTOCOL_LOCK.json"
    acceptance = data / "G1_AM1_ACCEPTANCE.json"; ledger = data / "completed_latent_ledger.jsonl"
    records = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line]
    bad: list[str] = []; low=np.full(2, np.inf); high=np.full(2, -np.inf)
    if verify_payloads:
        for record in records:
            path = Path(record["file_path"])
            if not path.is_file() or _sha(path) != record["file_sha256"]: bad.append(record["latent_id"])
            else:
                with np.load(path, allow_pickle=False) as z:
                    low=np.minimum(low, [float(z["m"].min()), float(z["h"].min())]); high=np.maximum(high, [float(z["m"].max()), float(z["h"].max())])
    target_range = {"M": [float(low[0]), float(high[0])], "H": [float(low[1]), float(high[1])]} if verify_payloads else None
    range_ok = not verify_payloads or (bool(np.all(low >= 0.0)) and bool(np.all(high <= 1.0)))
    result = {"status": "PASS" if len(records) == 13500 and not bad and range_ok else "FAIL", "acceptance_sha256": _sha(acceptance), "protocol_lock_sha256": _sha(lock), "manifest_hashes": _manifest_hashes(root), "ledger_sha256": _sha(ledger), "ledger_records": len(records), "payloads_verified": verify_payloads, "payload_failure_count": len(bad), "first_payload_failure": bad[:1], "target_range": target_range, "target_range_is_unit_interval": range_ok}
    return result


def validate_authoritative_input(root: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    data = root / DATA_REL; acceptance = json.loads((data / "G1_AM1_ACCEPTANCE.json").read_text()); lock = json.loads((root / "config/so_r1/SO_R1_A2_D0_AM1_PROTOCOL_LOCK.json").read_text())
    required = {"status": "PASS", "dataset_status": "ACCEPTED", "formal_generation_completed": True, "training_authorized": True, "next_stage": "SO-R1-B1"}
    if any(acceptance.get(k) != v for k, v in required.items()): raise RuntimeError("FAIL_INPUT_GATE:G1_ACCEPTANCE")
    config=root/"config/so_r1/so_r1_a2_full_generation_protocol_v1_1.yaml"
    am4=root/"data/processed/SO_R1_A0_AM4_GlobalCameraAtlas_v1/AM4_ACCEPTANCE.json"
    camera_config=root/"config/so_r1/frozen_camera_light_split_v1_3.yaml"
    allowlist=root/"reports/so_r1_a0_am4_global_camera_atlas/final_24pair_allowlist.csv"
    if not all(path.is_file() for path in (config,am4,camera_config,allowlist)): raise RuntimeError("FAIL_INPUT_GATE:MISSING_AUTHORITY")
    am4_doc=json.loads(am4.read_text())
    if am4_doc.get("status") != "PASS" or len(pd.read_csv(allowlist)) != 24: raise RuntimeError("FAIL_INPUT_GATE:AM4")
    qc=json.loads((root/REPORT_REL.parent/"so_r1_a2_g1_am1_full_generation"/"full_generation_qc_summary.json").read_text())
    coverage=json.loads((root/REPORT_REL.parent/"so_r1_a2_g1_am1_full_generation"/"camera_light_coverage_audit.json").read_text())
    replay=json.loads((root/REPORT_REL.parent/"so_r1_a2_g1_am1_full_generation"/"independent_replay_audit.json").read_text())
    if qc.get("high_clip_violations") != 0 or qc.get("low_clip_violations") != 0 or qc.get("nonfinite") != 0 or coverage.get("unique_camera_light_pairs") != 24 or replay.get("status") != "PASS": raise RuntimeError("FAIL_INPUT_GATE:DATASET_AUDIT")
    rows = pd.read_csv(data / "manifests/latent_manifest.csv").to_dict("records")
    if len(rows) != 13500 or sum(r["split"] == "Train" for r in rows) != 10000 or sum(r["split"] == "Validation" for r in rows) != 1000: raise RuntimeError("FAIL_INPUT_GATE:MANIFEST_COUNTS")
    return acceptance, lock, rows


def build_epoch_groups(rows: list[dict[str, Any]], seed: int, epoch: int) -> tuple[list[Group], dict[str, Any]]:
    train = [r for r in rows if r["split"] == "Train"]
    ordered = sorted(train, key=lambda r: hashlib.sha256(f"{seed}|{epoch}|{r['latent_id']}".encode()).hexdigest())
    offset = int.from_bytes(hashlib.sha256(f"{seed}|{epoch}|pair-offset".encode()).digest()[:4], "big") % 4
    groups = [Group(r["latent_id"], r["split"], r["file_path"], PAIR_INDEX[PAIR_TYPES[(rank + offset) % 4]], PAIR_TYPES[(rank + offset) % 4]) for rank, r in enumerate(ordered)]
    ids = [g.latent_id for g in groups]
    schedule = {"epoch": epoch, "epoch_pair_schedule_hash": _canon_hash([(g.latent_id, g.pair_type) for g in groups]), "pair_type_count": {kind: sum(g.pair_type == kind for g in groups) for kind in PAIR_TYPES}, "unique_latent_count": len(set(ids)), "duplicate_latent_count": len(ids) - len(set(ids))}
    if schedule["unique_latent_count"] != 10000 or schedule["duplicate_latent_count"]: raise RuntimeError("FAIL_TRAIN_SAMPLER")
    return groups, schedule


def _collate_groups(items: list[dict[str, Tensor]]) -> dict[str, Any]:
    return {"images": torch.stack([x["images"] for x in items]), "target": torch.stack([x["target"] for x in items]), "mask": torch.stack([x["mask"] for x in items]), "latent_id": [x["latent_id"] for x in items], "pair_type": [x["pair_type"] for x in items]}


def _loader(dataset: Dataset, batch_size: int, *, shuffle: bool, seed: int, workers: int, groups: bool, timeout_seconds: int = 0) -> DataLoader:
    generator = torch.Generator(); generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=True, persistent_workers=workers > 0, worker_init_fn=_worker_seed, generator=generator, collate_fn=_collate_groups if groups else None, timeout=timeout_seconds if workers else 0)


def _flatten(batch: dict[str, Any], device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    images = batch["images"].to(device, non_blocking=True).flatten(0, 1)
    target = batch["target"].to(device, non_blocking=True).repeat_interleave(2, dim=0)
    mask = batch["mask"].to(device, non_blocking=True).repeat_interleave(2, dim=0)
    return images, target, mask


def _finite(*items: Tensor) -> bool: return all(bool(torch.isfinite(x).all()) for x in items)


def _one_step(model: nn.Module, optimizer: torch.optim.Optimizer, batch: dict[str, Any], device: torch.device, amp: bool, scaler: torch.amp.GradScaler) -> tuple[float, bool]:
    model.train(); optimizer.zero_grad(set_to_none=True); x,y,m = _flatten(batch, device)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp): out = model(x); loss = masked_smooth_l1(out, y, m)
    if not _finite(out, loss): raise RuntimeError("FAIL_NONFINITE_FORWARD")
    scale_before=float(scaler.get_scale()); scaler.scale(loss).backward(); scaler.unscale_(optimizer)
    if not all(_finite(p.grad) for p in model.parameters() if p.grad is not None): raise RuntimeError("FAIL_NONFINITE_GRADIENT")
    scaler.step(optimizer); scaler.update()
    return float(loss.detach().cpu()), float(scaler.get_scale()) < scale_before


def gpu_environment() -> dict[str, Any]:
    if not torch.cuda.is_available(): raise RuntimeError("FAIL_GPU_ENVIRONMENT_NOT_READY")
    props = torch.cuda.get_device_properties(0)
    driver="unavailable"
    try: driver=subprocess.check_output(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"],text=True,timeout=10).strip().splitlines()[0]
    except Exception: pass
    return {"os": str(platform.platform()), "python": str(sys.version), "torch": str(torch.__version__), "cuda_runtime": str(torch.version.cuda), "gpu_name": str(props.name), "gpu_memory_bytes": int(props.total_memory), "nvidia_driver": str(driver), "cuda_available": True, "selected_device": "cuda:0"}


def preflight(rows: list[dict[str, Any]], seed: int, run: Path, preferred_precision: str) -> tuple[bool, int, dict[str, Any], dict[str, Any]]:
    device = torch.device("cuda:0"); groups, _ = build_epoch_groups(rows, seed, 1); dataset = PairedGroupDataset(groups)
    final_workers=2
    batch = next(iter(_loader(dataset, 4, shuffle=False, seed=seed, workers=final_workers, groups=True, timeout_seconds=DATA_LOADER_TIMEOUT_SECONDS)))
    result: dict[str, Any] = {}
    for amp in (False, True):
        _configure_determinism(seed); torch.cuda.init(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); model=BaselineMHUNet().to(device); optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4); scaler=torch.amp.GradScaler("cuda", enabled=amp)
        try:
            loss, overflow=_one_step(model, optimizer, batch, device, amp, scaler); torch.cuda.synchronize(); result["AMP" if amp else "FP32"]={"status":"PASS" if not overflow else "FAIL", "loss":loss, "overflow_detected":overflow, "peak_memory_bytes":int(torch.cuda.max_memory_allocated()), "data_loader_workers":final_workers}
        except Exception as exc: result["AMP" if amp else "FP32"]={"status":"FAIL", "error":repr(exc)}
        finally: del model, optimizer; torch.cuda.empty_cache()
    if result["FP32"]["status"] != "PASS": raise RuntimeError("FAIL_FP32_PREFLIGHT")
    if preferred_precision not in ("AMP", "FP32"): raise RuntimeError("FAIL_PRECISION_POLICY")
    if result[preferred_precision]["status"] != "PASS": raise RuntimeError("FAIL_PRECISION_PREFLIGHT")
    amp = preferred_precision == "AMP"
    capacity: dict[str, Any] = {"precision": "AMP" if amp else "FP32", "candidates": {}}
    chosen = 0
    for size in (4,3,2,1):
        _configure_determinism(seed); torch.cuda.init(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); model=BaselineMHUNet().to(device); optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4); scaler=torch.amp.GradScaler("cuda", enabled=amp); loader=_loader(dataset, size, shuffle=False, seed=seed, workers=final_workers, groups=True, timeout_seconds=DATA_LOADER_TIMEOUT_SECONDS)
        try:
            values=[]; overflows=[]
            for index, item in zip(range(10), loader):
                value, overflow=_one_step(model, optimizer, item, device, amp, scaler); values.append(value); overflows.append(overflow)
            torch.cuda.synchronize()
            capacity["candidates"][str(size)]={"status":"PASS" if not any(overflows) else "FAIL", "steps":len(values), "overflow_detected":any(overflows), "peak_memory_bytes":int(torch.cuda.max_memory_allocated()), "last_loss":values[-1], "data_loader_workers":final_workers}
            if any(overflows): continue
            chosen=size; break
        except torch.cuda.OutOfMemoryError as exc: capacity["candidates"][str(size)]={"status":"OOM", "error":repr(exc)}; torch.cuda.empty_cache()
        except Exception as exc: capacity["candidates"][str(size)]={"status":"FAIL", "error":repr(exc)}
        finally: del model, optimizer, loader; torch.cuda.empty_cache()
    if not chosen: raise RuntimeError("FAIL_BATCH_CAPACITY_PREFLIGHT")
    capacity["selected_latent_groups_per_batch"] = chosen; capacity["images_per_batch"] = chosen * 2; capacity["data_loader_workers"] = final_workers
    _dump_json(run / "environment/amp_preflight.json", result); _dump_json(run / "environment/batch_capacity_preflight.json", capacity)
    return amp, chosen, result, capacity


def choose_precision_from_stability_diagnosis(root: Path) -> dict[str, Any]:
    path=root/STABILITY_REPORT_REL/"gradient_stability_diagnosis.json"
    if not path.is_file(): raise RuntimeError("FAIL_MISSING_STABILITY_DIAGNOSIS")
    diagnosis=json.loads(path.read_text())
    amp=diagnosis.get("modes",{}).get("AMP",{}); fp32=diagnosis.get("modes",{}).get("FP32",{})
    if amp.get("status") == "PASS": return {"selected_precision":"AMP", "diagnosis":diagnosis}
    if fp32.get("status") == "PASS": return {"selected_precision":"FP32", "diagnosis":diagnosis}
    raise RuntimeError("FAIL_NUMERICAL_STABILITY")


def train_mean_predictor(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    sums=np.zeros((2,256,256),dtype=np.float64); counts=np.zeros((1,256,256),dtype=np.float64)
    for row in rows:
        if row["split"] != "Train": continue
        _, target, mask = _load_npz(str(row["file_path"]), [0]); sums += target * mask; counts += mask
    return (sums / np.maximum(counts, 1)).astype(np.float32), (counts > 0).astype(np.float32)


def validate(model: nn.Module, rows: list[dict[str, Any]], mean_target: np.ndarray, device: torch.device, amp: bool, workers: int, timeout_seconds: int = 0) -> dict[str, Any]:
    data=ValidationDataset([r for r in rows if r["split"] == "Validation"]); loader=_loader(data, 8, shuffle=False, seed=0, workers=workers, groups=False, timeout_seconds=timeout_seconds); model.eval()
    sums={key:np.zeros(2,dtype=np.float64) for key in ("smooth","abs","sq","pred","pred_sq","target","target_sq","mean_abs")}; denom=np.zeros(2,dtype=np.float64); zero=np.zeros(2); one=np.zeros(2); sample_pred=[]; sample_true=[]
    with torch.no_grad():
        for batch in loader:
            x=batch["image"].to(device,non_blocking=True); y=batch["target"].to(device,non_blocking=True); m=batch["mask"].to(device,non_blocking=True); valid=m.expand_as(y)
            with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=amp): p=model(x)
            if not _finite(p): raise RuntimeError("FAIL_NONFINITE_VALIDATION")
            e=torch.nn.functional.smooth_l1_loss(p,y,beta=.1,reduction="none")*valid; ab=(p-y).abs()*valid; sq=(p-y).square()*valid; mean=torch.from_numpy(mean_target).to(device).unsqueeze(0); mab=(mean-y).abs()*valid
            for key,val in (("smooth",e),("abs",ab),("sq",sq),("pred",p*valid),("pred_sq",p.square()*valid),("target",y*valid),("target_sq",y.square()*valid),("mean_abs",mab)):
                sums[key]+=val.sum((0,2,3)).detach().cpu().numpy()
            denom+=valid.sum((0,2,3)).detach().cpu().numpy(); zero += ((p<=1e-6)*valid).sum((0,2,3)).cpu().numpy(); one += ((p>=1-1e-6)*valid).sum((0,2,3)).cpu().numpy()
            sample_pred.extend(((p*valid).sum((2,3))/valid.sum((2,3)).clamp_min(1)).cpu().numpy().tolist()); sample_true.extend(((y*valid).sum((2,3))/valid.sum((2,3)).clamp_min(1)).cpu().numpy().tolist())
    pred_mean=sums["pred"]/denom; target_mean=sums["target"]/denom; pred_var=np.maximum(sums["pred_sq"]/denom-pred_mean**2,0); target_var=np.maximum(sums["target_sq"]/denom-target_mean**2,0)
    out={"masked_smooth_l1":(sums["smooth"]/denom).tolist(),"mae":(sums["abs"]/denom).tolist(),"rmse":np.sqrt(sums["sq"]/denom).tolist(),"mean_predictor_mae":(sums["mean_abs"]/denom).tolist(),"prediction_mean":pred_mean.tolist(),"target_mean":target_mean.tolist(),"prediction_variance":pred_var.tolist(),"target_variance":target_var.tolist(),"variance_ratio":(pred_var/np.maximum(target_var,1e-20)).tolist(),"zero_saturation_fraction":(zero/denom).tolist(),"one_saturation_fraction":(one/denom).tolist()}
    # Image-level map means are a compact training diagnostic, not inferential statistics.
    from scipy.stats import pearsonr, spearmanr
    p=np.asarray(sample_pred); t=np.asarray(sample_true)
    out["pearson"]=[float(pearsonr(p[:,i],t[:,i]).statistic) for i in range(2)]; out["spearman"]=[float(spearmanr(p[:,i],t[:,i]).statistic) for i in range(2)]
    return out


def save_fixed_validation_examples(model: nn.Module, rows: list[dict[str, Any]], device: torch.device, amp: bool, destination: Path) -> list[str]:
    """Persist four deterministic validation predictions as training artifacts only."""
    destination.mkdir(parents=True, exist_ok=True); model.eval(); written=[]
    with torch.no_grad():
        for row in sorted((r for r in rows if r["split"] == "Validation"), key=lambda r: r["latent_id"])[:4]:
            rgb,target,mask=_load_npz(str(row["file_path"]),[0]); x=torch.from_numpy(rgb).to(device)
            with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=amp): prediction=model(x).detach().cpu().numpy().astype(np.float16)
            path=destination/f"{row['latent_id']}_A0.npz"; np.savez_compressed(path,prediction=prediction,target=target.astype(np.float16),mask=mask.astype(np.uint8),latent_id=row["latent_id"]); written.append(path.name)
    return written


def _checkpoint(run: Path, epoch: int, model: nn.Module, optimizer: Any, scheduler: Any, contract_hash: str, lock: dict[str, Any], acceptance_hash: str, manifest_hashes: dict[str,str], schedule_hash: str, precision: str) -> Path:
    payload={"model_state":model.state_dict(),"optimizer_state":optimizer.state_dict(),"scheduler_state":scheduler.state_dict(),"epoch":epoch,"model_architecture":BaselineMHUNet.architecture_id,"training_contract_hash":contract_hash,"d0_am1_protocol_hash":lock["protocol_hash"],"g1_am1_dataset_acceptance_hash":acceptance_hash,"dataset_manifest_hashes":manifest_hashes,"paired_schedule_hash":schedule_hash,"selected_precision":precision}
    path=run/"checkpoints"/f"epoch_{epoch:02d}.pt"; path.parent.mkdir(parents=True,exist_ok=True); torch.save(payload,path); return path


def write_failure(root: Path, error: BaseException) -> None:
    """Persist failure evidence in the isolated v2 output without touching inputs."""
    run_dir=root/RUN_REL; report=root/REPORT_REL
    run_dir.mkdir(parents=True, exist_ok=True); report.mkdir(parents=True, exist_ok=True)
    acceptance={"status":"FAIL","baseline_training_status":"NOT_COMPLETE","baseline_checkpoint_authoritative":False,"baseline_viability_status":"NOT_RUN","source_dataset_unchanged":None,"training_authorized":False,"next_stage":None,"next_stage_authorized":False,"proposed_training_started":False,"failure_type":type(error).__name__,"failure":repr(error)}
    _dump_json(run_dir/"B1_ACCEPTANCE.json",acceptance); _dump_json(report/"failure_evidence.json",acceptance)


def write_workers2_preflight_failure(root: Path, error: BaseException) -> None:
    """Persist an isolated workers=2 verification failure without touching B1 outputs."""
    root=Path(root); run_dir=root/WORKERS2_PREFLIGHT_RUN_REL; report=root/WORKERS2_PREFLIGHT_REPORT_REL
    run_dir.mkdir(parents=True, exist_ok=True); report.mkdir(parents=True, exist_ok=True)
    result={"status":"FAIL","verification":"workers=2 FP32 preflight plus one complete epoch","formal_b1_training_started":False,"workers":2,"precision":"FP32","failure_type":type(error).__name__,"failure":repr(error)}
    _dump_json(run_dir/"WORKERS2_PREFLIGHT_ACCEPTANCE.json",result); _dump_json(report/"failure_evidence.json",result)


def verify_workers2_one_epoch(root: Path) -> dict[str, Any]:
    """Verify Windows workers=2 through preflight and one full FP32 epoch; never creates a formal B1 run."""
    root=Path(root); run_dir=root/WORKERS2_PREFLIGHT_RUN_REL; report=root/WORKERS2_PREFLIGHT_REPORT_REL
    if run_dir.exists(): raise RuntimeError("REFUSE_OVERWRITE_EXISTING_WORKERS2_PREFLIGHT")
    _, lock, rows=validate_authoritative_input(root)
    before=source_audit(root,verify_payloads=True)
    if before["status"] != "PASS": raise RuntimeError("FAIL_INPUT_HASH_AUDIT")
    run_dir.mkdir(parents=True); report.mkdir(parents=True, exist_ok=True)
    (run_dir/"environment").mkdir(parents=True); (run_dir/"metrics").mkdir(parents=True); (run_dir/"checkpoints").mkdir(parents=True)
    _dump_json(report/"source_dataset_hash_audit_before.json",before)
    _dump_json(run_dir/"environment/gpu_environment.json",gpu_environment())
    seed=_seed_from_protocol(lock["protocol_hash"])
    precision_evidence=choose_precision_from_stability_diagnosis(root)
    if precision_evidence["selected_precision"] != "FP32": raise RuntimeError("FAIL_PRECISION_CONTRACT_EXPECTED_FP32")
    amp,batch_size,amp_result,capacity=preflight(rows,seed,run_dir,"FP32")
    if amp: raise RuntimeError("FAIL_PRECISION_CONTRACT_EXPECTED_FP32")
    contract={"verification_id":"SO-R1-B1-workers2-preflight-v1","purpose":"Windows workers=2 preflight and one complete epoch; not a formal B1 training run","formal_b1_training_started":False,"source_dataset_protocol_hash":lock["protocol_hash"],"source_dataset_hashes":before,"seed":seed,"precision":"FP32","num_workers":2,"pin_memory":True,"persistent_workers":True,"data_loader_timeout_seconds":120,"latent_groups_per_batch":batch_size,"images_per_batch":batch_size*2,"epochs":1,"optimizer":{"type":"AdamW","learning_rate":.001,"weight_decay":.0001},"scheduler":{"type":"CosineAnnealingLR","T_max":25,"eta_min":1e-6},"amp_preflight":amp_result,"batch_capacity_preflight":capacity}
    contract_hash=_canon_hash(contract)
    (run_dir/"workers2_preflight_contract.yaml").write_text(yaml.safe_dump(contract,sort_keys=False),encoding="utf-8")
    _dump_json(run_dir/"run_manifest.json",{"status":"RUNNING","formal_b1_training_started":False,"contract_hash":contract_hash,"source_hash_audit":before})
    _configure_determinism(seed); device=torch.device("cuda:0"); model=BaselineMHUNet().to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001); scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=25,eta_min=1e-6); scaler=torch.amp.GradScaler("cuda",enabled=False)
    mean_target,_=train_mean_predictor(rows); groups,schedule=build_epoch_groups(rows,seed,1)
    loader=_loader(PairedGroupDataset(groups),batch_size,shuffle=False,seed=seed+1,workers=2,groups=True,timeout_seconds=120)
    values=[]; started=time.monotonic()
    for index,batch in enumerate(loader,1):
        loss,overflow=_one_step(model,optimizer,batch,device,False,scaler)
        if overflow: raise RuntimeError("FAIL_UNEXPECTED_FP32_SCALER_OVERFLOW")
        values.append(loss)
        if index % 100 == 0:
            progress={"status":"RUNNING","phase":"epoch_1_train","completed_batches":index,"total_batches":len(loader),"last_loss":loss,"elapsed_seconds":time.monotonic()-started,"workers":2,"precision":"FP32"}
            _dump_json(run_dir/"metrics/progress.json",progress); print(json.dumps(progress,sort_keys=True),flush=True)
    torch.cuda.synchronize(); validation=validate(model,rows,mean_target,device,False,workers=2); scheduler.step()
    checkpoint=_checkpoint(run_dir,1,model,optimizer,scheduler,contract_hash,lock,before["acceptance_sha256"],before["manifest_hashes"],schedule["epoch_pair_schedule_hash"],"FP32")
    elapsed=time.monotonic()-started
    after=source_audit(root,verify_payloads=True); unchanged=before==after
    result={"status":"PASS" if unchanged else "FAIL_PROTECTED_DATASET_MUTATION","verification":"workers=2 FP32 preflight plus one complete epoch","formal_b1_training_started":False,"workers":2,"precision":"FP32","epoch":1,"train_batches":len(values),"train_masked_smooth_l1":float(np.mean(values)),"validation":validation,"elapsed_seconds":elapsed,"source_dataset_unchanged":unchanged,"checkpoint":str(checkpoint),"checkpoint_is_authoritative":False,"paired_schedule":schedule,"contract_hash":contract_hash}
    _dump_json(run_dir/"metrics/epoch_1_verification.json",result); _dump_json(report/"workers2_preflight_report.json",result); _dump_json(report/"source_dataset_hash_audit_after.json",after); _dump_json(run_dir/"WORKERS2_PREFLIGHT_ACCEPTANCE.json",result); _dump_json(run_dir/"run_manifest.json",result)
    return result


def diagnose_gradient_stability(root: Path, steps: int = 250) -> dict[str, Any]:
    """Read-only-data diagnostic; compare actual AMP and FP32 update stability."""
    root=Path(root); _, lock, rows=validate_authoritative_input(root); seed=_seed_from_protocol(lock["protocol_hash"])
    if not torch.cuda.is_available(): raise RuntimeError("FAIL_GPU_ENVIRONMENT_NOT_READY")
    groups,_=build_epoch_groups(rows,seed,1); device=torch.device("cuda:0"); outcome: dict[str, Any]={"steps_requested":steps,"workers":2,"batch_contract":{"latent_groups":4,"images":8},"modes":{}}
    for label, amp in (("AMP",True),("FP32",False)):
        _configure_determinism(seed); torch.cuda.init(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        model=BaselineMHUNet().to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001); scaler=torch.amp.GradScaler("cuda",enabled=amp); loader=_loader(PairedGroupDataset(groups),4,shuffle=False,seed=seed+1,workers=2,groups=True)
        losses=[]; failure=None
        try:
            for index,batch in zip(range(steps),loader):
                try:
                    loss,overflow=_one_step(model,optimizer,batch,device,amp,scaler)
                    if overflow: raise RuntimeError("AMP_SCALER_OVERFLOW")
                    losses.append(loss)
                    if (index + 1) % 250 == 0:
                        _dump_json(root/"reports/so_r1_b1_baseline_v3/gradient_stability_progress.json",{"mode":label,"completed_steps":index+1,"requested_steps":steps,"last_loss":loss,"status":"RUNNING"})
                        print(json.dumps({"mode":label,"completed_steps":index+1,"requested_steps":steps,"last_loss":loss}),flush=True)
                except Exception as error:
                    bad=[name for name,p in model.named_parameters() if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
                    failure={"step":index+1,"error":repr(error),"nonfinite_gradient_parameters":bad[:8],"losses_before_failure":len(losses)}; break
            torch.cuda.synchronize()
            outcome["modes"][label]={"status":"PASS" if failure is None else "FAIL","completed_steps":len(losses),"first_loss":losses[:1],"last_loss":losses[-1:] ,"failure":failure,"peak_memory_bytes":int(torch.cuda.max_memory_allocated())}
        finally:
            del model,optimizer,loader; torch.cuda.empty_cache()
    _dump_json(root/"reports/so_r1_b1_baseline_v3/gradient_stability_diagnosis.json",outcome)
    return outcome


def run(root: Path) -> dict[str, Any]:
    root=Path(root); run_dir=root/RUN_REL; report=root/REPORT_REL
    if run_dir.exists(): raise RuntimeError("REFUSE_OVERWRITE_EXISTING_B1_RUN")
    acceptance,lock,rows=validate_authoritative_input(root); before=source_audit(root,verify_payloads=True); report.mkdir(parents=True,exist_ok=True); _dump_json(report/"source_dataset_hash_audit_before.json",before)
    if before["status"]!="PASS": raise RuntimeError("FAIL_INPUT_HASH_AUDIT")
    seed=_seed_from_protocol(lock["protocol_hash"]); env=gpu_environment(); run_dir.mkdir(parents=True); (run_dir/"metrics").mkdir(parents=True,exist_ok=True); (run_dir/"checkpoints").mkdir(parents=True,exist_ok=True); (run_dir/"artifacts/fixed_validation_prediction_examples").mkdir(parents=True,exist_ok=True); _dump_json(run_dir/"environment/gpu_environment.json",env)
    precision_evidence=choose_precision_from_stability_diagnosis(root); precision=precision_evidence["selected_precision"]
    _dump_json(run_dir/"environment/precision_stability_diagnosis.json",precision_evidence)
    amp,batch_size,amp_result,capacity=preflight(rows,seed,run_dir,precision); manifests=before["manifest_hashes"]; acceptance_hash=before["acceptance_sha256"]
    parameter_count=count_parameters(BaselineMHUNet())
    contract={"stage_id":"SO-R1-B1","contract_version":"v6","source_dataset_protocol_hash":lock["protocol_hash"],"source_dataset_acceptance_hash":acceptance_hash,"dataset_root":str(root/DATA_REL),"dataset_manifest_hashes":manifests,"target_range_audit":before["target_range"],"model_architecture":{"id":BaselineMHUNet.architecture_id,"parameter_count":parameter_count,"input_channels":3,"output_channels":2,"output_order":BaselineMHUNet.output_order,"encoder_channels":[32,64,128,256,512],"normalization":"GroupNorm(8)","activation":"LeakyReLU(0.01)","output_activation":"Sigmoid"},"paired_comparison_seed":{"derivation":f"int(sha256('{ROOT_SEED_SALT}|' + D0 protocol hash)[:8]) mod (2^31-1)","value":seed},"model_init_seed":seed,"paired_sampler_seed":seed,"training_seed":seed,"num_epochs":25,"optimizer":{"type":"AdamW","learning_rate":.001,"weight_decay":.0001},"scheduler":{"type":"CosineAnnealingLR","T_max":25,"eta_min":1e-6},"batch_contract":{"latent_groups_per_batch":batch_size,"images_per_batch":batch_size*2,"pair_flattening":"[B,2,3,H,W] -> [2B,3,H,W]; independent supervised samples only"},"amp_contract":{"precision":precision,"amp_preflight":amp_result,"stability_diagnosis":precision_evidence["diagnosis"]},"data_loader_contract":{"num_workers":2,"pin_memory":True,"persistent_workers":True,"timeout_seconds":DATA_LOADER_TIMEOUT_SECONDS,"worker_seed":"torch initial seed + worker id","augmentation":"geometric/photometric/color-jitter/exposure/gamma/crop all disabled"},"loss_contract":{"name":"masked SmoothL1","beta":.1,"channel_weights":[.5,.5],"lambda_pair":0,"pair_loss":"absent","feature_loss":"absent","adversarial_loss":"absent","reconstruction_loss":"absent"},"validation_contract":{"split":"Validation","latent":1000,"acquisition":5000},"checkpoint_selection_rule":"lowest validation masked SmoothL1; earliest epoch on exact tie","software_environment":env,"batch_capacity_preflight":capacity}
    contract_hash=_canon_hash(contract); (root/"config/so_r1").mkdir(parents=True,exist_ok=True); (root/"config/so_r1/so_r1_b1_training_contract_v6.yaml").write_text(yaml.safe_dump(contract,sort_keys=False),encoding="utf-8"); (run_dir/"training_contract_snapshot.yaml").write_text(yaml.safe_dump(contract,sort_keys=False),encoding="utf-8"); _dump_json(run_dir/"run_manifest.json",{"status":"RUNNING","contract_hash":contract_hash,"source_hash_audit":before,"training_authorized":False,"proposed_training_started":False})
    _configure_determinism(seed); device=torch.device("cuda:0"); model=BaselineMHUNet().to(device); params=count_parameters(model); optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001); scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=25,eta_min=1e-6); scaler=torch.amp.GradScaler("cuda",enabled=amp); mean_target,_=train_mean_predictor(rows)
    schedules=[]; metrics=[]; best=float("inf"); best_epoch=0; best_path=None
    for epoch in range(1,26):
        groups,schedule=build_epoch_groups(rows,seed,epoch); schedules.append(schedule); loader=_loader(PairedGroupDataset(groups),batch_size,shuffle=False,seed=seed+epoch,workers=2,groups=True,timeout_seconds=DATA_LOADER_TIMEOUT_SECONDS); values=[]
        for batch in loader:
            loss, overflow=_one_step(model,optimizer,batch,device,amp,scaler)
            if overflow: raise RuntimeError("FAIL_AMP_OVERFLOW")
            values.append(loss)
        validation=validate(model,rows,mean_target,device,amp,workers=2,timeout_seconds=DATA_LOADER_TIMEOUT_SECONDS); val=float(np.mean(validation["masked_smooth_l1"])); scheduler.step(); row={"epoch":epoch,"train_masked_smooth_l1":float(np.mean(values)),"validation_masked_smooth_l1":val,"learning_rate":float(optimizer.param_groups[0]["lr"]),**{f"val_{key}_M":value[0] for key,value in validation.items() if isinstance(value,list)},**{f"val_{key}_H":value[1] for key,value in validation.items() if isinstance(value,list)}}; metrics.append(row); path=_checkpoint(run_dir,epoch,model,optimizer,scheduler,contract_hash,lock,acceptance_hash,manifests,schedule["epoch_pair_schedule_hash"],precision)
        if val < best: best=val; best_epoch=epoch; best_path=path; shutil.copy2(path,run_dir/"checkpoints"/"best_val_masked_smoothl1.pt")
        pd.DataFrame(metrics).to_csv(run_dir/"metrics/epoch_metrics.csv",index=False); pd.DataFrame(metrics).to_csv(report/"training_curves.csv",index=False); _dump_json(run_dir/"metrics/paired_sampler_schedule_hashes.json",schedules)
        print(json.dumps({"epoch":epoch,"train_masked_smooth_l1":row["train_masked_smooth_l1"],"validation_masked_smooth_l1":val,"best_epoch":best_epoch},sort_keys=True),flush=True)
    pd.DataFrame(metrics).to_csv(run_dir/"metrics/epoch_metrics.csv",index=False); pd.DataFrame(metrics).to_csv(report/"training_curves.csv",index=False); _dump_json(run_dir/"metrics/paired_sampler_schedule_hashes.json",schedules)
    best_ckpt=torch.load(run_dir/"checkpoints"/"best_val_masked_smoothl1.pt",map_location=device,weights_only=False); model.load_state_dict(best_ckpt["model_state"]); final=validate(model,rows,mean_target,device,amp,workers=2,timeout_seconds=DATA_LOADER_TIMEOUT_SECONDS); examples=save_fixed_validation_examples(model,rows,device,amp,run_dir/"artifacts/fixed_validation_prediction_examples"); _dump_json(run_dir/"metrics/validation_summary.json",{"best_epoch":best_epoch,"metrics":final,"fixed_validation_examples":examples}); viable=all(final["mae"][i]<final["mean_predictor_mae"][i] and final["prediction_variance"][i]>0 and final["variance_ratio"][i]>=.01 and final["zero_saturation_fraction"][i]<.99 and final["one_saturation_fraction"][i]<.99 for i in range(2)); viability={"status":"PASS" if viable else "FAIL_BASELINE_VIABILITY","best_epoch":best_epoch,"metrics":final,"mean_predictor_train_only":True,"checkpoint_reloadable":True,"parameter_count":params,"fixed_validation_examples":examples}; _dump_json(report/"baseline_viability_audit.json",viability)
    after=source_audit(root,verify_payloads=True); _dump_json(report/"source_dataset_hash_audit_after.json",after); unchanged=before==after
    accepted=viable and unchanged; final_acc={"status":"PASS" if accepted else "FAIL","baseline_training_status":"COMPLETE","baseline_checkpoint_authoritative":accepted,"baseline_viability_status":viability["status"],"source_dataset_unchanged":unchanged,"training_authorized":accepted,"next_stage":"SO-R1-B2" if accepted else None,"next_stage_authorized":accepted,"proposed_training_started":False,"best_checkpoint":str(run_dir/"checkpoints"/"best_val_masked_smoothl1.pt"),"best_epoch":best_epoch,"contract_hash":contract_hash}; _dump_json(run_dir/"B1_ACCEPTANCE.json",final_acc); _dump_json(run_dir/"run_manifest.json",{**final_acc,"parameter_count":params,"precision":precision}); (report/"SO_R1_B1_Baseline_Training_Report.md").write_text("# SO-R1-B1 Baseline Training\n\n```json\n"+json.dumps(final_acc,indent=2)+"\n```\n",encoding="utf-8")
    return final_acc
