"""Run frozen SO-1D v1.1 formal training from epoch 1."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from skin_optics_so1.decomposition.amp_recovery import (  # noqa: E402
    adam_state_sha256,
    grad_scaler_found_inf,
    optimizer_group_sha256,
    parameter_state_sha256,
    scaler_step_and_update,
)
from skin_optics_so1.decomposition.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from skin_optics_so1.decomposition.formal_protocol import (  # noqa: E402
    build_formal_identity,
    split_access_payload,
    validate_dataset_contract,
)
from skin_optics_so1.decomposition.formal_protocol_v1_1 import (  # noqa: E402
    NUMERICAL_POLICY,
    PROTOCOL_VERSION,
    RUN_NAME,
    OverflowFrequencyGate,
    audit_resume,
    build_identity,
    validate_config,
)
from skin_optics_so1.decomposition.losses import CHANNEL_NAMES, masked_smooth_l1_loss  # noqa: E402
from skin_optics_so1.decomposition.metrics import (  # noqa: E402
    MaskedMAEAggregator,
    PredictionMonitoringAggregator,
)
from skin_optics_so1.decomposition.numerical_diagnostics import (  # noqa: E402
    batchnorm_health,
    gradient_health,
    state_finite,
    tensor_health,
    write_csv,
    write_json,
)
from skin_optics_so1.decomposition.synthetic_dataset import SO1DecompositionDataset  # noqa: E402
from skin_optics_so1.decomposition.unet_decomposer import SO1UNetDecomposer, count_parameters  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "config/train/skin_optics_so1/so1d_formal_train_v1_1.yaml"
DATA_ROOT = PROJECT_ROOT / "data/processed/SO1_Synthetic_Generator_v1_localfull"
ROOT = PROJECT_ROOT / "experiments/skin_optics_so1/SO1_Decomposition_UNet_v1"
OUTPUT_DIR = ROOT / "formal_train_v1_1"
V10_DIR = ROOT / "formal_train"
SO1C_CONTRACT = ROOT / "smoke/dataset_contract.json"
SO1C_CHECKPOINT = ROOT / "smoke/checkpoints/last.pt"
R3_STATUS = ROOT / "diagnostics/so1d_r3_gradscaler_recoverability/final_status.json"
V10_HASHES = {
    "last.pt": "85e559ac4cb9a3da5e889b57db7fc45974cdc14f8f8d1312479c1a980e755349",
    "best.pt": "e8c3be6d5f818a049160a97b151cf2a14c82275c7c00f7577523b56123846704",
}
V10_HISTORICAL_FILES = [
    "config_frozen.yaml", "runtime.json", "SO1D_formal_training_report.md",
    "training_history.csv", "checkpoints/last.pt", "checkpoints/best.pt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--initial-pytest-passed", type=int)
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--final-pytest-passed", type=int)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def v10_historical_hashes() -> dict[str, str]:
    return {name: file_sha256(V10_DIR / name) for name in V10_HISTORICAL_FILES}


def git_value(*args: str) -> str | None:
    result = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def memory_payload() -> dict[str, Any]:
    try:
        import ctypes
        class Status(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        status = Status(); status.dwLength = ctypes.sizeof(Status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return {}
        return {
            "system_ram_total_bytes": int(status.ullTotalPhys),
            "system_ram_available_bytes": int(status.ullAvailPhys),
            "page_file_total_bytes": int(status.ullTotalPageFile),
            "page_file_available_bytes": int(status.ullAvailPageFile),
        }
    except Exception:
        return {}


def environment_payload() -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "os": platform.platform(), "python": sys.version,
        "python_executable": sys.executable, "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "gpu": {"name": torch.cuda.get_device_name(0), "total_memory_bytes": properties.total_memory},
        "git_commit": git_value("rev-parse", "HEAD"),
        **memory_payload(),
    }


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def worker_init_fn(worker_id: int) -> None:
    seed = (torch.initial_seed() + worker_id) % (2**32)
    random.seed(seed); np.random.seed(seed)


def make_loader(dataset: Any, config: dict[str, Any], *, shuffle: bool, seed: int) -> DataLoader:
    loader = config["dataloader"]
    return DataLoader(
        dataset, batch_size=8, shuffle=shuffle, num_workers=loader["num_workers"],
        pin_memory=loader["pin_memory"], persistent_workers=loader["persistent_workers"],
        prefetch_factor=loader["prefetch_factor"], worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(seed),
    )


def freeze_config() -> tuple[dict[str, Any], str]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    validate_config(config)
    digest = file_sha256(CONFIG_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frozen = OUTPUT_DIR / "config_frozen.yaml"
    if frozen.is_file() and file_sha256(frozen) != digest:
        raise RuntimeError("REFUSE_RUN: frozen v1.1 config differs")
    if not frozen.is_file():
        frozen.write_bytes(CONFIG_PATH.read_bytes())
    (OUTPUT_DIR / "config_sha256.txt").write_text(digest + "\n", encoding="ascii")
    return config, digest


def bn_summary(model: Any) -> dict[str, Any]:
    rows = batchnorm_health(model)
    means = [row["max_abs"] for row in rows if row["tensor"] == "running_mean"]
    variances = [row for row in rows if row["tensor"] == "running_var"]
    result = {
        "BN_finite": all(row["finite"] for row in rows),
        "max_abs_BN_running_mean": max(means, default=0.0),
        "max_BN_running_var": max((row["max"] for row in variances), default=0.0),
        "min_BN_running_var": min((row["min"] for row in variances), default=0.0),
    }
    for group in ("encoder", "bottleneck", "decoder"):
        selected = [row for row in rows if row["module_group"] == group]
        result[f"{group}_max_abs_running_mean"] = max(
            (row["max_abs"] for row in selected if row["tensor"] == "running_mean"), default=0.0
        )
        result[f"{group}_max_running_var"] = max(
            (row["max"] for row in selected if row["tensor"] == "running_var"), default=0.0
        )
    return result


def bn_buffers_finite(model: Any) -> bool:
    checks = []
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            checks.extend((
                torch.isfinite(module.running_mean).all(),
                torch.isfinite(module.running_var).all(),
            ))
    return bool(torch.stack(checks).all().item()) if checks else True


def tensors_finite(values: Any) -> bool:
    checks = [torch.isfinite(value).all() for value in values]
    return bool(torch.stack(checks).all().item())


def save_prediction_previews(payload: dict[str, torch.Tensor], tag: str) -> None:
    import matplotlib.pyplot as plt

    rgb = payload["rgb"].numpy(); mask = payload["mask"].numpy()
    target = payload["target"].numpy(); prediction = payload["prediction"].numpy()
    for sample in range(rgb.shape[0]):
        images: list[tuple[str, Any, str]] = [
            ("linear RGB", np.moveaxis(rgb[sample], 0, -1).clip(0.0, 1.0), "rgb"),
            ("mask", mask[sample, 0], "gray"),
        ]
        for channel, name in enumerate(CHANNEL_NAMES):
            images.extend((
                (f"{name} true", target[sample, channel], "viridis"),
                (f"{name} pred", prediction[sample, channel], "viridis"),
                (f"{name} error", np.abs(target[sample, channel] - prediction[sample, channel]), "magma"),
            ))
        figure, axes = plt.subplots(2, 7, figsize=(16, 5), constrained_layout=True)
        for axis, (title, image, cmap) in zip(axes.flat, images):
            axis.imshow(image, cmap=None if cmap == "rgb" else cmap, vmin=0.0, vmax=1.0)
            axis.set_title(title, fontsize=8); axis.axis("off")
        figure.savefig(OUTPUT_DIR / "prediction_previews" / f"{tag}_sample_{sample + 1}.png", dpi=120)
        plt.close(figure)


def state_passes(model: Any, optimizer: Any) -> tuple[dict[str, Any], bool]:
    state = state_finite(model, optimizer)
    return state, all(state[key] for key in (
        "parameters_finite", "optimizer_state_finite", "batchnorm_buffers_finite"
    ))


def preflight(
    config: dict[str, Any], contract: dict[str, Any], fingerprint: dict[str, Any],
    identity: dict[str, Any], config_sha: str, environment: dict[str, Any],
) -> tuple[SO1DecompositionDataset, SO1DecompositionDataset]:
    checks: dict[str, Any] = {}
    train = SO1DecompositionDataset(DATA_ROOT, "train")
    validation = SO1DecompositionDataset(DATA_ROOT, "validation")
    checks["dataset_counts"] = len(train) == 20_000 and len(validation) == 2_000
    set_seed(config["seed"])
    model = SO1UNetDecomposer(numerical_precision=NUMERICAL_POLICY).cuda().train()
    checks["parameter_count"] = count_parameters(model)["parameter_count"] == 31_037_828
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    batch = next(iter(make_loader(train, config, shuffle=True, seed=config["seed"] + 11)))
    dtypes: dict[str, str] = {}
    handles = [
        model.up1.conv.register_forward_pre_hook(
            lambda _m, i: dtypes.__setitem__("up1_input", str(i[0].dtype))
        ),
        model.up1.conv.register_forward_hook(
            lambda _m, _i, o: dtypes.__setitem__("up1_output", str(o.dtype))
        ),
        model.up2.up.register_forward_hook(
            lambda _m, _i, o: dtypes.__setitem__("other_amp", str(o.dtype))
        ),
    ]
    x = batch["linear_rgb"].cuda(); target = batch["target_mhsp"].cuda(); mask = batch["valid_mask"].cuda()
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", enabled=True):
        prediction = model(x); losses = masked_smooth_l1_loss(prediction, target, mask)
    scaler.scale(losses["total"]).backward(); scaler.step(optimizer); scaler.update()
    for handle in handles: handle.remove()
    checks["global_amp"] = True
    checks["up1_fp32"] = dtypes.get("up1_input") == dtypes.get("up1_output") == "torch.float32"
    checks["other_amp_preserved"] = dtypes.get("other_amp") == "torch.float16"
    checks["forward_loss_backward_optimizer"] = all((
        tensor_health(prediction)["finite"], tensor_health(losses["total"])["finite"]
    ))
    temp = OUTPUT_DIR / "preflight_checkpoint.tmp.pt"
    save_checkpoint(
        temp, model=model, optimizer=optimizer, scheduler=scheduler, grad_scaler=scaler,
        epoch=0, global_step=1, best_selection_metric=float("inf"), early_stopping_counter=0,
        identity=identity, resolved_config=config, dataset_contract=contract,
        dataset_fingerprint=fingerprint, seed=config["seed"], config_sha256=config_sha,
        environment=environment,
    )
    reloaded = SO1UNetDecomposer(numerical_precision=NUMERICAL_POLICY).cuda()
    reloaded.load_state_dict(torch.load(temp, map_location="cuda", weights_only=False)["model_state_dict"], strict=True)
    temp.unlink()
    checks["checkpoint_roundtrip"] = True
    r3 = json.loads(R3_STATUS.read_text(encoding="utf-8"))
    checks["R3_regression_guard"] = (
        r3["SO-1D-R3"] == "PASS" and r3["primary_diagnosis"] == "RECOVERABLE_AMP_GRADIENT_OVERFLOW"
    )
    payload = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "dtypes": dtypes}
    write_json(OUTPUT_DIR / "resource_preflight.json", payload)
    if payload["status"] != "PASS":
        raise RuntimeError("SO-1D v1.1 PRECHECK FAIL")
    del model, optimizer, scheduler, scaler, reloaded, prediction, losses
    torch.cuda.empty_cache()
    return train, validation


class FormalTrainerV11:
    def __init__(
        self, *, config: dict[str, Any], contract: dict[str, Any], fingerprint: dict[str, Any],
        identity: dict[str, Any], config_sha: str, environment: dict[str, Any],
        resume_from: Path | None,
    ) -> None:
        self.config = config; self.contract = contract; self.fingerprint = fingerprint
        self.identity = identity; self.config_sha = config_sha; self.environment = environment
        self.model = SO1UNetDecomposer(numerical_precision=NUMERICAL_POLICY).cuda()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.0001)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=30, eta_min=1e-6)
        self.scaler = torch.amp.GradScaler("cuda", enabled=True)
        self.start_epoch = 1; self.global_step = 0; self.optimizer_updates = 0
        self.best_selection = float("inf"); self.best_epoch: int | None = None; self.patience = 0
        self.overflow_events: list[dict[str, Any]] = []
        self.min_scale = float(self.scaler.get_scale()); self.max_scale = float(self.scaler.get_scale())
        self.gate = OverflowFrequencyGate(max_consecutive=5, max_per_100=10)
        self.validation_preview: dict[str, torch.Tensor] = {}
        self.resume_rng: dict[str, Any] = {}
        if resume_from is not None:
            checkpoint = load_checkpoint(
                resume_from, model=self.model, optimizer=self.optimizer, scheduler=self.scheduler,
                grad_scaler=self.scaler, expected_identity=identity, map_location="cuda", restore_rng=True,
            )
            self.start_epoch = int(checkpoint["epoch"]) + 1
            self.global_step = int(checkpoint["global_step"])
            self.optimizer_updates = int(checkpoint["optimizer_update_count"])
            self.best_selection = float(checkpoint["best_selection_metric"])
            self.best_epoch = checkpoint["best_epoch"]
            self.patience = int(checkpoint["early_stopping_counter"])
            self.overflow_events = list(checkpoint.get("amp_overflow_events", []))
            self.min_scale = float(checkpoint.get("min_gradscaler_scale", self.scaler.get_scale()))
            self.max_scale = float(checkpoint.get("max_gradscaler_scale", self.scaler.get_scale()))
            gate_state = checkpoint.get("overflow_frequency_gate_state") or {}
            self.gate.consecutive = int(gate_state.get("consecutive", 0))
            self.gate.window.extend(bool(value) for value in gate_state.get("window", []))
            self.gate.maximum_consecutive = int(gate_state.get("maximum_consecutive", 0))
            self.gate.maximum_per_100 = int(gate_state.get("maximum_per_100", 0))
            self.resume_rng = checkpoint.get("dataloader_rng_state") or {}

    def _save(self, path: Path, *, epoch: int, train_gen: Any, val_gen: Any) -> None:
        save_checkpoint(
            path, model=self.model, optimizer=self.optimizer, scheduler=self.scheduler,
            grad_scaler=self.scaler, epoch=epoch, global_step=self.global_step,
            best_selection_metric=self.best_selection, best_epoch=self.best_epoch,
            early_stopping_counter=self.patience, identity=self.identity,
            resolved_config=self.config, dataset_contract=self.contract,
            dataset_fingerprint=self.fingerprint, seed=self.config["seed"],
            config_sha256=self.config_sha, git_commit=self.environment.get("git_commit"),
            environment=self.environment,
            dataloader_rng_state={"train": train_gen.get_state(), "validation": val_gen.get_state()},
            extra_payload={
                "amp_enabled": True, "localized_fp32_blocks": ["up1.conv"],
                "amp_overflow_count": len(self.overflow_events),
                "optimizer_skipped_step_count": len(self.overflow_events),
                "optimizer_update_count": self.optimizer_updates,
                "min_gradscaler_scale": self.min_scale, "max_gradscaler_scale": self.max_scale,
                "amp_overflow_events": self.overflow_events,
                "overflow_frequency_gate_state": {
                    "consecutive": self.gate.consecutive,
                    "window": list(self.gate.window),
                    "maximum_consecutive": self.gate.maximum_consecutive,
                    "maximum_per_100": self.gate.maximum_per_100,
                },
            },
        )

    def train_epoch(self, loader: DataLoader, epoch: int) -> dict[str, Any]:
        self.model.train(); totals = {key: 0.0 for key in ("total", *CHANNEL_NAMES)}
        samples = 0; epoch_overflows = 0; started = time.perf_counter()
        monitor = {"conv0": 0.0, "block": 0.0, "active": False}
        handles = [
            self.model.up1.conv.net[0].register_forward_hook(
                lambda _m, _i, o: monitor.__setitem__(
                    "conv0", max(monitor["conv0"], float(o.detach().abs().max().cpu()))
                ) if monitor["active"] else None
            ),
            self.model.up1.conv.register_forward_hook(
                lambda _m, _i, o: monitor.__setitem__(
                    "block", max(monitor["block"], float(o.detach().abs().max().cpu()))
                ) if monitor["active"] else None
            ),
        ]
        for batch_index, batch in enumerate(loader):
            self.global_step += 1
            monitor["active"] = batch_index % 100 == 0
            x = batch["linear_rgb"].cuda(non_blocking=True)
            target = batch["target_mhsp"].cuda(non_blocking=True)
            mask = batch["valid_mask"].cuda(non_blocking=True)
            if not tensors_finite((x, target, mask)):
                raise FloatingPointError("DATA_NONFINITE")
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=True):
                prediction = self.model(x); losses = masked_smooth_l1_loss(prediction, target, mask)
            if not tensors_finite((prediction,)): raise FloatingPointError("FORWARD_NONFINITE")
            if not tensors_finite(losses.values()):
                raise FloatingPointError("LOSS_NONFINITE")
            if not bn_buffers_finite(self.model): raise FloatingPointError("BATCHNORM_STATE_CONTAMINATION")
            scale_before = float(self.scaler.get_scale())
            self.scaler.scale(losses["total"]).backward(); self.scaler.unscale_(self.optimizer)
            overflow = grad_scaler_found_inf(self.scaler, self.optimizer)
            gradients = {"first_nonfinite_parameter": None}
            if overflow:
                _, gradients = gradient_health(self.model)
            before = None
            if overflow:
                before = (
                    parameter_state_sha256(self.model), adam_state_sha256(self.optimizer, self.model),
                    optimizer_group_sha256(self.optimizer),
                )
            step = scaler_step_and_update(self.scaler, self.optimizer)
            self.min_scale = min(self.min_scale, step["scale_after"])
            self.max_scale = max(self.max_scale, step["scale_after"])
            if overflow:
                after = (
                    parameter_state_sha256(self.model), adam_state_sha256(self.optimizer, self.model),
                    optimizer_group_sha256(self.optimizer),
                )
                state, finite = state_passes(self.model, self.optimizer)
                if not step["optimizer_step_skipped"] or before != after or not finite:
                    raise RuntimeError("RECOVERABLE_OVERFLOW_HANDLER_FAILURE")
                event = {
                    "epoch": epoch, "batch_index": batch_index, "global_step": self.global_step,
                    "first_nonfinite_gradient": gradients["first_nonfinite_parameter"],
                    "scale_before": scale_before, "scale_after": step["scale_after"],
                    "optimizer_step_skipped": True,
                    "parameters_finite": state["parameters_finite"],
                    "BN_finite": state["batchnorm_buffers_finite"],
                    "Adam_finite": state["optimizer_state_finite"],
                }
                self.overflow_events.append(event); epoch_overflows += 1
            else:
                if not step["optimizer_step_executed"]: raise RuntimeError("RECOVERABLE_OVERFLOW_HANDLER_FAILURE")
                self.optimizer_updates += 1
            if self.gate.update(overflow):
                raise RuntimeError("EXCESSIVE_AMP_OVERFLOW_FREQUENCY")
            count = x.shape[0]; samples += count
            for key, value in losses.items(): totals[key] += float(value.detach().cpu()) * count
            self.optimizer.zero_grad(set_to_none=True)
        for handle in handles: handle.remove()
        state, finite = state_passes(self.model, self.optimizer)
        if not finite:
            raise FloatingPointError(f"MODEL_OR_OPTIMIZER_STATE_NONFINITE: {state}")
        return {
            **{f"train_{key}_loss": totals[key] / samples for key in totals},
            "samples_per_second": samples / (time.perf_counter() - started),
            "epoch_amp_overflows": epoch_overflows,
            "up1_conv0_pre_bn_max_abs": monitor["conv0"],
            "up1_conv_block_output_max_abs": monitor["block"],
        }

    @torch.inference_mode()
    def validate(self, loader: DataLoader) -> dict[str, Any]:
        self.model.eval(); metrics = MaskedMAEAggregator(); monitoring = PredictionMonitoringAggregator()
        loss_sums = {key: 0.0 for key in ("total", *CHANNEL_NAMES)}; samples = 0
        self.validation_preview = {}
        for batch_index, batch in enumerate(loader):
            x = batch["linear_rgb"].cuda(non_blocking=True)
            target = batch["target_mhsp"].cuda(non_blocking=True)
            mask = batch["valid_mask"].cuda(non_blocking=True)
            with torch.amp.autocast("cuda", enabled=True):
                prediction = self.model(x); losses = masked_smooth_l1_loss(prediction, target, mask)
            if not tensors_finite((prediction,)): raise FloatingPointError("VALIDATION_FORWARD_NONFINITE")
            if batch_index == 0:
                self.validation_preview = {
                    "rgb": x[:4].float().cpu(), "mask": mask[:4].float().cpu(),
                    "target": target[:4].float().cpu(), "prediction": prediction[:4].float().cpu(),
                }
            metrics.update(prediction.float(), target.float(), mask.float())
            monitoring.update(prediction.float(), target.float(), mask.float())
            count = x.shape[0]; samples += count
            for key, value in losses.items(): loss_sums[key] += float(value.cpu()) * count
        return {
            **{f"val_{key}_loss": loss_sums[key] / samples for key in loss_sums},
            **{f"val_{key}": value for key, value in metrics.compute().items()},
            **monitoring.compute(),
        }

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> dict[str, Any]:
        if self.resume_rng:
            train_loader.generator.set_state(self.resume_rng["train"].cpu())
            val_loader.generator.set_state(self.resume_rng["validation"].cpu())
        history_path = OUTPUT_DIR / "training_history.csv"
        if self.start_epoch == 1 and history_path.exists():
            raise RuntimeError("REFUSE_RUN: v1.1 history exists; use --resume-from")
        started = time.perf_counter(); stop_reason = "MAX_EPOCHS_REACHED"
        for epoch in range(self.start_epoch, 31):
            epoch_started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
            lr = self.optimizer.param_groups[0]["lr"]
            train_started = time.perf_counter(); train = self.train_epoch(train_loader, epoch)
            train_seconds = time.perf_counter() - train_started
            val_started = time.perf_counter(); val = self.validate(val_loader)
            val_seconds = time.perf_counter() - val_started
            self.scheduler.step()
            selection = val["val_selection_metric"]
            improved = selection < self.best_selection
            if improved:
                self.best_selection = selection; self.best_epoch = epoch; self.patience = 0
            else: self.patience += 1
            bn = bn_summary(self.model); memory = memory_payload()
            record = {
                "epoch": epoch, "global_step": self.global_step, **train, **val,
                "selection_metric": selection, "best_selection_metric": self.best_selection,
                "best_epoch": self.best_epoch, "early_stopping_counter": self.patience,
                "learning_rate": lr, "gradscaler_scale": float(self.scaler.get_scale()),
                "cumulative_amp_overflow_count": len(self.overflow_events),
                "cumulative_optimizer_skipped_steps": len(self.overflow_events),
                "maximum_consecutive_overflows": self.gate.maximum_consecutive,
                "maximum_overflows_per_100_batches": self.gate.maximum_per_100,
                "optimizer_update_count": self.optimizer_updates,
                "epoch_train_seconds": train_seconds, "epoch_validation_seconds": val_seconds,
                "epoch_total_seconds": time.perf_counter() - epoch_started,
                "gpu_peak_allocated": torch.cuda.max_memory_allocated(),
                "gpu_peak_reserved": torch.cuda.max_memory_reserved(),
                "system_RAM_available": memory.get("system_ram_available_bytes"),
                "page_file_available": memory.get("page_file_available_bytes"), **bn,
            }
            append_csv(history_path, record); append_csv(OUTPUT_DIR / "epoch_metrics.csv", record)
            self._save(OUTPUT_DIR / "checkpoints/last.pt", epoch=epoch,
                       train_gen=train_loader.generator, val_gen=val_loader.generator)
            if improved:
                self._save(OUTPUT_DIR / "checkpoints/best.pt", epoch=epoch,
                           train_gen=train_loader.generator, val_gen=val_loader.generator)
            if epoch in self.config["checkpoint"]["milestone_epochs"]:
                self._save(OUTPUT_DIR / f"checkpoints/epoch_{epoch:02d}.pt", epoch=epoch,
                           train_gen=train_loader.generator, val_gen=val_loader.generator)
            save_prediction_previews(self.validation_preview, "final")
            if epoch == 1:
                save_prediction_previews(self.validation_preview, "epoch_01")
            if improved:
                save_prediction_previews(self.validation_preview, f"best_epoch_{epoch:02d}")
            write_csv(OUTPUT_DIR / "amp_overflow_events.csv", self.overflow_events, [
                "epoch", "batch_index", "global_step", "first_nonfinite_gradient",
                "scale_before", "scale_after", "optimizer_step_skipped",
                "parameters_finite", "BN_finite", "Adam_finite",
            ])
            write_json(OUTPUT_DIR / "progress.json", {
                "epoch_completed": epoch, "best_epoch": self.best_epoch,
                "best_selection_metric": self.best_selection, "early_stopping_counter": self.patience,
            })
            print(
                f"epoch={epoch} selection={selection:.8f} best={self.best_selection:.8f} "
                f"patience={self.patience} overflows={len(self.overflow_events)} "
                f"seconds={record['epoch_total_seconds']:.1f}",
                flush=True,
            )
            if self.patience >= 5:
                stop_reason = "EARLY_STOPPING"; break
        runtime = {
            "status": "COMPLETED", "stop_reason": stop_reason,
            "actual_epochs": epoch, "global_step": self.global_step,
            "optimizer_update_count": self.optimizer_updates,
            "best_epoch": self.best_epoch, "best_selection_metric": self.best_selection,
            "wall_seconds": time.perf_counter() - started,
        }
        write_json(OUTPUT_DIR / "runtime.json", runtime)
        return runtime


def append_csv(path: Path, row: dict[str, Any]) -> None:
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row));
        if not exists: writer.writeheader()
        writer.writerow(row); handle.flush()


def finalize(final_pytest: int | None = None) -> dict[str, Any]:
    runtime = json.loads((OUTPUT_DIR / "runtime.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((OUTPUT_DIR / "training_history.csv").open(encoding="utf-8-sig")))
    best = next(row for row in rows if int(float(row["epoch"])) == runtime["best_epoch"])
    last = rows[-1]
    checkpoint = torch.load(OUTPUT_DIR / "checkpoints/best.pt", map_location="cpu", weights_only=False)
    last_checkpoint = torch.load(
        OUTPUT_DIR / "checkpoints/last.pt", map_location="cpu", weights_only=False
    )
    model = SO1UNetDecomposer(numerical_precision=NUMERICAL_POLICY)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True); optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    state, finite = state_passes(model, optimizer)
    last_model = SO1UNetDecomposer(numerical_precision=NUMERICAL_POLICY)
    last_optimizer = torch.optim.AdamW(last_model.parameters(), lr=0.001, weight_decay=0.0001)
    last_model.load_state_dict(last_checkpoint["model_state_dict"], strict=True)
    last_optimizer.load_state_dict(last_checkpoint["optimizer_state_dict"])
    last_state, last_finite = state_passes(last_model, last_optimizer)
    sha = file_sha256(OUTPUT_DIR / "checkpoints/best.pt")
    (OUTPUT_DIR / "best_checkpoint_sha256.txt").write_text(sha + "\n", encoding="ascii")
    overflow_summary = {
        "total_overflow_events": last_checkpoint["amp_overflow_count"],
        "total_skipped_optimizer_steps": last_checkpoint["optimizer_skipped_step_count"],
        "min_gradscaler_scale": last_checkpoint["min_gradscaler_scale"],
        "max_gradscaler_scale": last_checkpoint["max_gradscaler_scale"],
        "final_gradscaler_scale": last_checkpoint["grad_scaler_state_dict"]["scale"],
        "maximum_consecutive_overflows": int(last["maximum_consecutive_overflows"]),
        "maximum_overflows_per_100_batches": int(last["maximum_overflows_per_100_batches"]),
        "excessive_overflow_gate": "PASS",
    }
    write_json(OUTPUT_DIR / "amp_overflow_summary.json", overflow_summary)
    access = split_access_payload(train_accessed=True, validation_accessed=True)
    write_json(OUTPUT_DIR / "split_access_audit.json", access)
    v10_current = v10_historical_hashes()
    v10_before = json.loads(
        (OUTPUT_DIR / "v1_0_historical_hashes_before.json").read_text(encoding="utf-8")
    )
    pass_status = all((
        runtime["stop_reason"] in ("EARLY_STOPPING", "MAX_EPOCHS_REACHED"), finite, last_finite,
        (OUTPUT_DIR / "checkpoints/last.pt").is_file(), access["status"] == "PASS",
        v10_current == v10_before, final_pytest is not None,
    ))
    final = {
        "SO-1D_v1.1": "PASS" if pass_status else "FAIL",
        "formal_best_checkpoint_frozen": pass_status,
        "best_checkpoint_sha256": sha if pass_status else None,
        "best_state_health": state, "last_state_health": last_state,
        "v1_0_historical_hashes_unchanged": v10_current == v10_before,
        "initial_pytest": json.loads((OUTPUT_DIR / "initial_pytest.json").read_text(encoding="utf-8")),
        "final_pytest": ({"passed": final_pytest, "failed": 0} if final_pytest is not None else None),
        "allow_automatic_SO1E": False, "allow_SO2": False, "allow_classification": False,
    }
    write_json(OUTPUT_DIR / "final_status.json", final)
    report = [
        "# SO-1D v1.1 Formal Training Report", "",
        "## 1. Background", "", "SO-1D v1.0 remains a historical FAIL due to FP16 forward overflow and BatchNorm contamination.", "",
        "## 2. R1-R3 Evidence", "", "R1 localized the overflow, R2 validated `up1.conv` FP32, and R3 established standard GradScaler recovery.", "",
        "## 3. Frozen Protocol", "", "Fresh seed-20260801 training from epoch1 with global AMP and only `up1.conv` in FP32.", "",
        "## 4. Dataset Contract", "", "PASS: Train=20,000; Validation=2,000; no test/OOD access.", "",
        "## 5. Training Result", "", json.dumps(runtime, ensure_ascii=False), "",
        "## 6. Best Metrics", "", f"Epoch {runtime['best_epoch']}; M/H/S/P={best['val_M_MAE']}/{best['val_H_MAE']}/{best['val_S_MAE']}/{best['val_P_MAE']}; selection={best['selection_metric']}.", "",
        "## 7. Last Metrics", "", f"M/H/S/P={last['val_M_MAE']}/{last['val_H_MAE']}/{last['val_S_MAE']}/{last['val_P_MAE']}.", "",
        "## 8. AMP Safety", "", json.dumps(overflow_summary, ensure_ascii=False), "",
        "## 9. BN And Output Monitoring", "", "Per-epoch BN scales and prediction distribution/saturation fields are recorded in `training_history.csv`.", "",
        "## 10. Checkpoint", "", f"Formal best SHA256: `{sha}`", "",
        "## 11. Regression", "", f"Initial={final['initial_pytest']}; final={final['final_pytest']}.", "",
        "## 12. Final", "", json.dumps(final, ensure_ascii=False), "",
        "SO-1E, SO-2, classification, and all test/OOD evaluation were not executed.", "",
    ]
    (OUTPUT_DIR / "SO1D_v1_1_formal_training_report.md").write_text("\n".join(report), encoding="utf-8")
    return final


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("SO-1D v1.1 requires CUDA")
    if args.finalize_only:
        if args.final_pytest_passed is None: raise ValueError("--final-pytest-passed required")
        finalize(args.final_pytest_passed); return
    (OUTPUT_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "prediction_previews").mkdir(parents=True, exist_ok=True)
    config, config_sha = freeze_config()
    historical_hash_path = OUTPUT_DIR / "v1_0_historical_hashes_before.json"
    current_v10_hashes = v10_historical_hashes()
    if historical_hash_path.is_file():
        if json.loads(historical_hash_path.read_text(encoding="utf-8")) != current_v10_hashes:
            raise RuntimeError("SO-1D v1.0 historical artifacts changed")
    else:
        write_json(historical_hash_path, current_v10_hashes)
    if args.initial_pytest_passed is not None:
        write_json(OUTPUT_DIR / "initial_pytest.json", {"passed": args.initial_pytest_passed, "failed": 0})
    environment = environment_payload(); write_json(OUTPUT_DIR / "environment.json", environment)
    contract, fingerprint, audit = validate_dataset_contract(
        data_root=DATA_ROOT, so1c_contract_path=SO1C_CONTRACT, so1c_checkpoint_path=SO1C_CHECKPOINT,
    )
    write_json(OUTPUT_DIR / "dataset_contract.json", contract)
    write_json(OUTPUT_DIR / "dataset_fingerprint.json", fingerprint)
    write_json(OUTPUT_DIR / "dataset_contract_audit.json", audit)
    base_identity = build_formal_identity(contract_hash=fingerprint["contract_hash"], config={
        **config, "protocol_version": "SO1D_Formal_Train_v1.0",
        "training": {"batch_size": 8, "amp": True, "max_epochs": 30},
    })
    identity = build_identity(base_identity, config)
    write_json(OUTPUT_DIR / "split_access_audit.json", split_access_payload(train_accessed=False, validation_accessed=False))
    train, validation = preflight(config, contract, fingerprint, identity, config_sha, environment)
    if args.preflight_only: return
    resume = args.resume_from.resolve() if args.resume_from else None
    if resume:
        result = audit_resume(
            resume, expected_path=OUTPUT_DIR / "checkpoints/last.pt", config_sha256=config_sha,
            identity=identity, history_path=OUTPUT_DIR / "training_history.csv",
        ); write_json(OUTPUT_DIR / "resume_audit.json", result)
    set_seed(config["seed"])
    trainer = FormalTrainerV11(
        config=config, contract=contract, fingerprint=fingerprint, identity=identity,
        config_sha=config_sha, environment=environment, resume_from=resume,
    )
    train_loader = make_loader(train, config, shuffle=True, seed=config["seed"] + 11)
    val_loader = make_loader(validation, config, shuffle=False, seed=config["seed"] + 17)
    write_json(OUTPUT_DIR / "split_access_audit.json", split_access_payload(train_accessed=True, validation_accessed=True))
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
