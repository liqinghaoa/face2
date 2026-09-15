"""Execute the single frozen SO-1D formal training run on native Windows."""

from __future__ import annotations

import argparse
import gc
import json
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

from skin_optics_so1.decomposition.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from skin_optics_so1.decomposition.formal_protocol import (  # noqa: E402
    FORMAL_RUN_NAME,
    audit_formal_resume,
    build_formal_identity,
    file_sha256,
    split_access_payload,
    validate_dataset_contract,
    validate_formal_config,
)
from skin_optics_so1.decomposition.losses import masked_smooth_l1_loss  # noqa: E402
from skin_optics_so1.decomposition.synthetic_dataset import SO1DecompositionDataset  # noqa: E402
from skin_optics_so1.decomposition.trainer import SO1DecompositionTrainer  # noqa: E402
from skin_optics_so1.decomposition.unet_decomposer import (  # noqa: E402
    SO1UNetDecomposer,
    count_parameters,
)


CONFIG_PATH = PROJECT_ROOT / "config/train/skin_optics_so1/so1d_formal_train_v1.yaml"
DATA_ROOT = PROJECT_ROOT / "data/processed/SO1_Synthetic_Generator_v1_localfull"
OUTPUT_DIR = PROJECT_ROOT / "experiments/skin_optics_so1/SO1_Decomposition_UNet_v1/formal_train"
SO1C_CONTRACT_PATH = (
    PROJECT_ROOT
    / "experiments/skin_optics_so1/SO1_Decomposition_UNet_v1/smoke/dataset_contract.json"
)
SO1C_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "experiments/skin_optics_so1/SO1_Decomposition_UNet_v1/smoke/checkpoints/last.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--initial-pytest-passed", type=int)
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--final-pytest-passed", type=int)
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def git_value(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
    )
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

        status = Status()
        status.dwLength = ctypes.sizeof(Status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        return {
            "system_ram_total_bytes": int(status.ullTotalPhys),
            "system_ram_available_bytes": int(status.ullAvailPhys),
            "page_file_total_bytes": int(status.ullTotalPageFile),
            "page_file_available_bytes": int(status.ullAvailPageFile),
        }
    except Exception as exc:
        return {"memory_query_error": str(exc)}


def environment_payload() -> dict[str, Any]:
    gpu: dict[str, Any] | None = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "total_memory_bytes": int(properties.total_memory),
            "allocated_bytes": int(torch.cuda.memory_allocated(0)),
            "reserved_bytes": int(torch.cuda.memory_reserved(0)),
        }
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "os": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": gpu,
        **memory_payload(),
        "git_commit": git_value("rev-parse", "HEAD"),
        "working_tree_status": git_value("status", "--short"),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def worker_init_fn(worker_id: int) -> None:
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loader(
    dataset: SO1DecompositionDataset, config: dict[str, Any], *, shuffle: bool, seed: int
) -> DataLoader:
    loader = config["dataloader"]
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(loader["num_workers"]),
        pin_memory=bool(loader["pin_memory"]),
        persistent_workers=bool(loader["persistent_workers"]),
        prefetch_factor=int(loader["prefetch_factor"]),
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(seed),
    )


def freeze_config(config_path: Path, output_dir: Path) -> tuple[dict[str, Any], str]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_formal_config(config)
    digest = file_sha256(config_path)
    frozen_path = output_dir / "config_frozen.yaml"
    if frozen_path.is_file() and file_sha256(frozen_path) != digest:
        raise RuntimeError("REFUSE_RUN: existing frozen config differs from SO-1D v1.0 config")
    if not frozen_path.is_file():
        frozen_path.write_bytes(config_path.read_bytes())
    (output_dir / "config_sha256.txt").write_text(digest + "\n", encoding="ascii")
    return config, digest


def run_resource_preflight(
    *, config: dict[str, Any], contract: dict[str, Any], fingerprint: dict[str, Any],
    identity: dict[str, Any], config_sha256: str, environment: dict[str, Any],
) -> tuple[SO1DecompositionDataset, SO1DecompositionDataset, dict[str, Any]]:
    started = time.perf_counter()
    checks: dict[str, Any] = {}
    if not torch.cuda.is_available():
        raise RuntimeError("SO-1D PRECHECK FAIL: CUDA unavailable in Face2 environment")
    device = torch.device("cuda")
    checks["cuda_import"] = "PASS"
    temp_path = OUTPUT_DIR / "resource_preflight_checkpoint.tmp.pt"
    train_dataset: SO1DecompositionDataset | None = None
    validation_dataset: SO1DecompositionDataset | None = None
    try:
        temp_model = SO1UNetDecomposer().to(device)
        temp_optimizer = torch.optim.AdamW(
            temp_model.parameters(), lr=float(config["optimizer"]["lr"]),
            weight_decay=float(config["optimizer"]["weight_decay"]),
        )
        temp_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            temp_optimizer, T_max=int(config["scheduler"]["T_max"]),
            eta_min=float(config["scheduler"]["eta_min"]),
        )
        temp_scaler = torch.amp.GradScaler("cuda", enabled=True)
        checks["model_gpu_allocation"] = "PASS"

        train_dataset = SO1DecompositionDataset(DATA_ROOT, "train")
        validation_dataset = SO1DecompositionDataset(DATA_ROOT, "validation")
        if len(train_dataset) != 20_000 or len(validation_dataset) != 2_000:
            raise RuntimeError("Formal Dataset counts changed after contract validation")
        checks["train_dataset_open"] = "PASS"
        checks["validation_dataset_open"] = "PASS"
        train_loader = make_loader(train_dataset, config, shuffle=True, seed=int(config["seed"]) + 11)
        validation_loader = make_loader(
            validation_dataset, config, shuffle=False, seed=int(config["seed"]) + 17
        )
        train_batch = next(iter(train_loader))
        checks["train_dataloader_batch"] = list(train_batch["linear_rgb"].shape)
        x = train_batch["linear_rgb"].to(device, non_blocking=True)
        target = train_batch["target_mhsp"].to(device, non_blocking=True)
        mask = train_batch["valid_mask"].to(device, non_blocking=True)
        temp_optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=True):
            prediction = temp_model(x)
            losses = masked_smooth_l1_loss(prediction, target, mask, beta=0.1)
        temp_scaler.scale(losses["total"]).backward()
        temp_scaler.step(temp_optimizer)
        temp_scaler.update()
        checks["forward_loss_backward_optimizer"] = "PASS"
        validation_batch = next(iter(validation_loader))
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True):
            validation_prediction = temp_model(
                validation_batch["linear_rgb"].to(device, non_blocking=True)
            )
        if tuple(validation_prediction.shape) != (8, 4, 256, 256):
            raise RuntimeError(f"Validation output shape mismatch: {validation_prediction.shape}")
        checks["validation_forward"] = "PASS"
        save_checkpoint(
            temp_path, model=temp_model, optimizer=temp_optimizer, scheduler=temp_scheduler,
            grad_scaler=temp_scaler, epoch=0, global_step=1,
            best_selection_metric=float("inf"), early_stopping_counter=0,
            identity=identity, resolved_config=config, dataset_contract=contract,
            dataset_fingerprint=fingerprint, seed=int(config["seed"]), best_epoch=None,
            config_sha256=config_sha256, git_commit=environment.get("git_commit"),
            environment=environment,
        )
        reloaded_model = SO1UNetDecomposer().to(device)
        reloaded_optimizer = torch.optim.AdamW(reloaded_model.parameters(), lr=0.001, weight_decay=0.0001)
        reloaded_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            reloaded_optimizer, T_max=30, eta_min=0.000001
        )
        reloaded_scaler = torch.amp.GradScaler("cuda", enabled=True)
        load_checkpoint(
            temp_path, model=reloaded_model, optimizer=reloaded_optimizer,
            scheduler=reloaded_scheduler, grad_scaler=reloaded_scaler,
            expected_identity=identity, map_location=device, restore_rng=False,
        )
        checks["checkpoint_write_read"] = "PASS"
    except Exception as exc:
        payload = {
            "status": "FAIL", "error_type": type(exc).__name__, "error": str(exc),
            "checks": checks, "environment": environment,
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(OUTPUT_DIR / "resource_preflight.json", payload)
        raise
    finally:
        if temp_path.is_file():
            temp_path.unlink()
        for name in (
            "temp_model", "temp_optimizer", "temp_scheduler", "temp_scaler",
            "reloaded_model", "reloaded_optimizer", "reloaded_scheduler", "reloaded_scaler",
            "train_loader", "validation_loader", "train_batch", "validation_batch",
            "x", "target", "mask", "prediction", "validation_prediction", "losses",
        ):
            if name in locals():
                del locals()[name]
        gc.collect()
        torch.cuda.empty_cache()
    assert train_dataset is not None and validation_dataset is not None
    payload = {
        "status": "PASS", "checks": checks, "environment": environment,
        "preflight_model_isolated_from_formal_model": True,
        "formal_seed_reset_required_after_preflight": True,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(OUTPUT_DIR / "resource_preflight.json", payload)
    return train_dataset, validation_dataset, payload


def write_report(*, initial_pytest_passed: int | None, final_pytest_passed: int | None = None) -> None:
    runtime_path = OUTPUT_DIR / "runtime.json"
    history_path = OUTPUT_DIR / "training_history.csv"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.is_file() else {}
    rows: list[dict[str, str]] = []
    if history_path.is_file():
        import csv
        with history_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    best = min(rows, key=lambda row: float(row["selection_metric"])) if rows else None
    last = rows[-1] if rows else None
    best_hash_path = OUTPUT_DIR / "best_checkpoint_sha256.txt"
    lines = [
        "# SO-1D Formal Training Report", "",
        "## Scope", "",
        "Single frozen SO-1D engineering training run. Scientific ID/OOD evaluation was not executed.", "",
        "## Protocol", "",
        f"- Run: {FORMAL_RUN_NAME}",
        f"- Config SHA256: {(OUTPUT_DIR / 'config_sha256.txt').read_text(encoding='ascii').strip() if (OUTPUT_DIR / 'config_sha256.txt').is_file() else 'N/A'}",
        "- Train/validation: 20,000 / 2,000",
        "- Batch/AMP/workers: 8 / true / 2",
        "- Max epochs / patience: 30 / 5",
        "- Selection metric: val_M_MAE + val_H_MAE",
        "",
        "## Preflight", "",
        f"- Dataset contract: {'PASS' if (OUTPUT_DIR / 'dataset_contract_audit.json').is_file() else 'NOT_RUN'}",
        f"- Resource preflight: {'PASS' if (OUTPUT_DIR / 'resource_preflight.json').is_file() and json.loads((OUTPUT_DIR / 'resource_preflight.json').read_text(encoding='utf-8')).get('status') == 'PASS' else 'NOT_RUN/FAIL'}",
        f"- Initial pytest: {initial_pytest_passed if initial_pytest_passed is not None else 'N/A'} passed, 0 failed",
        "",
        "## Training", "",
        f"- Actual epochs: {runtime.get('completed_epoch', 0)}",
        f"- Total optimizer steps: {runtime.get('global_step', 0)}",
        f"- Stop reason: {runtime.get('stop_reason', 'IN_PROGRESS')}",
        f"- Training wall seconds: {runtime.get('run_wall_seconds', 0)}",
        f"- Best epoch: {runtime.get('best_epoch')}",
    ]
    if best:
        lines.extend([
            f"- Best M/H/S/P MAE: {best['val_M_MAE']} / {best['val_H_MAE']} / {best['val_S_MAE']} / {best['val_P_MAE']}",
            f"- Best selection metric: {best['selection_metric']}",
        ])
    if last:
        lines.extend([
            f"- Last M/H/S/P MAE: {last['val_M_MAE']} / {last['val_H_MAE']} / {last['val_S_MAE']} / {last['val_P_MAE']}",
            f"- Last learning rate: {last['learning_rate']}",
            f"- Last GPU peak allocated/reserved: {last['gpu_peak_allocated']} / {last['gpu_peak_reserved']}",
            f"- Last RAM/page-file available: {last['system_RAM_available']} / {last['page_file_available']}",
        ])
    lines.extend([
        "", "## Checkpoints", "",
        f"- best.pt SHA256: {best_hash_path.read_text(encoding='ascii').strip() if best_hash_path.is_file() else 'NOT_FROZEN'}",
        f"- Resume used: {(OUTPUT_DIR / 'resume_audit.json').is_file()}",
        "", "## Data Access", "",
        "Train and validation were accessed. ID test and all OOD splits were not accessed.",
        "", "## Regression", "",
        f"- Final pytest: {final_pytest_passed if final_pytest_passed is not None else 'PENDING'} passed, 0 failed",
        "", "## Final Status", "",
        f"SO-1D: {'PASS' if final_pytest_passed is not None and best_hash_path.is_file() else 'IN_PROGRESS'}",
        "", "Automatic SO-1E: NO", "", "SO-2: NO", "", "Classification: NO", "",
        "## Not Executed", "",
        "ID/OOD scientific metrics, correlations, baselines, cross-talk, RGB reconstruction, real-face inference, SO-2 and SO-3.",
    ])
    (OUTPUT_DIR / "SO1D_formal_training_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize(*, final_pytest_passed: int | None) -> None:
    if final_pytest_passed is None:
        raise ValueError("--final-pytest-passed is required with --finalize-only")
    best_path = OUTPUT_DIR / "checkpoints/best.pt"
    last_path = OUTPUT_DIR / "checkpoints/last.pt"
    if not best_path.is_file() or not last_path.is_file():
        raise RuntimeError("Cannot finalize SO-1D without best.pt and last.pt")
    digest = file_sha256(best_path)
    (OUTPUT_DIR / "best_checkpoint_sha256.txt").write_text(digest + "\n", encoding="ascii")
    access = split_access_payload(train_accessed=True, validation_accessed=True)
    write_json(OUTPUT_DIR / "split_access_audit.json", access)
    initial_path = OUTPUT_DIR / "initial_pytest.json"
    initial = json.loads(initial_path.read_text(encoding="utf-8")) if initial_path.is_file() else {}
    write_json(OUTPUT_DIR / "final_pytest.json", {"passed": final_pytest_passed, "failed": 0})
    write_report(initial_pytest_passed=initial.get("passed"), final_pytest_passed=final_pytest_passed)
    write_json(OUTPUT_DIR / "final_status.json", {
        "SO-1D": "PASS", "formal_best_checkpoint_frozen": True,
        "best_checkpoint": str(best_path), "best_checkpoint_sha256": digest,
        "allow_automatic_SO-1E": False, "allow_SO-2": False, "allow_classification": False,
    })


def main() -> None:
    args = parse_args()
    if args.finalize_only:
        finalize(final_pytest_passed=args.final_pytest_passed)
        return
    if platform.system() != "Windows":
        raise RuntimeError("SO-1D must run on native Windows")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "checkpoints").mkdir(exist_ok=True)
    (OUTPUT_DIR / "prediction_previews").mkdir(exist_ok=True)
    config, config_sha256 = freeze_config(CONFIG_PATH, OUTPUT_DIR)
    if args.initial_pytest_passed is not None:
        write_json(OUTPUT_DIR / "initial_pytest.json", {"passed": args.initial_pytest_passed, "failed": 0})
    environment = environment_payload()
    write_json(OUTPUT_DIR / "environment.json", environment)
    contract, fingerprint, contract_audit = validate_dataset_contract(
        data_root=DATA_ROOT, so1c_contract_path=SO1C_CONTRACT_PATH,
        so1c_checkpoint_path=SO1C_CHECKPOINT_PATH,
    )
    write_json(OUTPUT_DIR / "dataset_contract.json", contract)
    write_json(OUTPUT_DIR / "dataset_fingerprint.json", fingerprint)
    write_json(OUTPUT_DIR / "dataset_contract_audit.json", contract_audit)
    identity = build_formal_identity(contract_hash=fingerprint["contract_hash"], config=config)
    write_json(OUTPUT_DIR / "split_access_audit.json", split_access_payload(
        train_accessed=False, validation_accessed=False
    ))
    train_dataset, validation_dataset, _ = run_resource_preflight(
        config=config, contract=contract, fingerprint=fingerprint, identity=identity,
        config_sha256=config_sha256, environment=environment,
    )
    write_json(OUTPUT_DIR / "split_access_audit.json", split_access_payload(
        train_accessed=True, validation_accessed=True
    ))
    if args.preflight_only:
        write_report(initial_pytest_passed=args.initial_pytest_passed)
        return

    history_path = OUTPUT_DIR / "training_history.csv"
    if history_path.is_file() and args.resume_from is None:
        raise RuntimeError("REFUSE_RUN: formal history exists; use --resume-from formal_train/checkpoints/last.pt")
    resume_path: Path | None = None
    if args.resume_from is not None:
        resume_path = args.resume_from.resolve()
        expected = (OUTPUT_DIR / "checkpoints/last.pt").resolve()
        if resume_path != expected:
            raise RuntimeError(f"REFUSE_RESUME: only the formal run last.pt is allowed: {expected}")

    # Preflight used separate objects. Reset every RNG before constructing the formal model.
    set_seed(int(config["seed"]))
    model = SO1UNetDecomposer()
    if resume_path is not None:
        resume_audit = audit_formal_resume(
            checkpoint_path=resume_path, model=model, identity=identity,
            config_sha256=config_sha256, history_path=history_path,
        )
        write_json(OUTPUT_DIR / "resume_audit.json", resume_audit)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["optimizer"]["lr"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config["scheduler"]["T_max"]),
        eta_min=float(config["scheduler"]["eta_min"]),
    )
    resolved_config = json.loads(json.dumps(config))
    resolved_config["resolved"] = {
        "run_name": FORMAL_RUN_NAME, "config_path": str(CONFIG_PATH),
        "output_dir": str(OUTPUT_DIR), "data_root": str(DATA_ROOT),
        "config_sha256": config_sha256, "identity": identity,
        "model": count_parameters(model), "device": "cuda",
    }
    trainer = SO1DecompositionTrainer(
        model=model, optimizer=optimizer, scheduler=scheduler, device=torch.device("cuda"),
        output_dir=OUTPUT_DIR, resolved_config=resolved_config,
        dataset_contract=contract, dataset_fingerprint=fingerprint, identity=identity,
        seed=int(config["seed"]), amp_enabled=True,
        early_stopping_patience=int(config["early_stopping"]["patience"]),
        resume_from=resume_path, milestone_epochs=config["checkpoint"]["milestone_epochs"],
        config_sha256=config_sha256, git_commit=environment.get("git_commit"),
        environment=environment, formal_mode=True, save_root_checkpoints=False,
    )
    train_loader = make_loader(train_dataset, config, shuffle=True, seed=int(config["seed"]) + 11)
    validation_loader = make_loader(
        validation_dataset, config, shuffle=False, seed=int(config["seed"]) + 17
    )
    trainer.fit(train_loader, validation_loader, max_epochs=int(config["training"]["max_epochs"]))
    write_report(initial_pytest_passed=args.initial_pytest_passed)


if __name__ == "__main__":
    main()
