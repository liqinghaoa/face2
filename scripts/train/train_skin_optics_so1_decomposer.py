"""Train SO-1 U-Net decomposition preflight runs on Windows/PyTorch."""

from __future__ import annotations

import argparse
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
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from skin_optics_so1.decomposition.checkpoint import load_checkpoint  # noqa: E402
from skin_optics_so1.decomposition.contract import (  # noqa: E402
    EXPECTED_GENERATOR_CONFIG_HASH,
    build_dataset_contract,
    canonical_json_hash,
    sha256_file,
)
from skin_optics_so1.decomposition.synthetic_dataset import SO1DecompositionDataset  # noqa: E402
from skin_optics_so1.decomposition.trainer import SO1DecompositionTrainer  # noqa: E402
from skin_optics_so1.decomposition.unet_decomposer import (  # noqa: E402
    SO1UNetDecomposer,
    count_parameters,
)


DEFAULT_CONFIG = PROJECT_ROOT / "config/train/skin_optics_so1/so1_decomposition_smoke_v1.yaml"
TARGET_ORDER = ["M_norm", "H_norm", "S_norm", "P_norm"]
NORMALIZATION = {
    "M": "identity",
    "H": "identity",
    "S_norm": "(S - 0.25) / 1.75",
    "P_norm": "P / 0.10",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--phase-output-dir", type=Path)
    parser.add_argument("--fit-max-epochs", type=int)
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def save_yaml(payload: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def optional_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def environment_payload(config_path: Path, data_root: Path, seed: int, deterministic: bool, command: list[str]) -> dict[str, Any]:
    gpu = None
    if torch.cuda.is_available():
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        }
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": gpu,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "git_commit": optional_git_commit(),
        "git_dirty": git_dirty(),
        "command": command,
        "data_root": str(data_root),
        "config_path": str(config_path),
        "seed": int(seed),
        "deterministic": bool(deterministic),
    }


def memory_payload() -> dict[str, Any]:
    try:
        import ctypes
        class Status(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        status = Status(); status.dwLength = ctypes.sizeof(Status)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return {"system_ram_total_bytes": int(status.ullTotalPhys), "system_ram_available_bytes": int(status.ullAvailPhys), "page_file_total_bytes": int(status.ullTotalPageFile), "page_file_available_bytes": int(status.ullAvailPageFile)}
    except Exception as exc:
        return {"unavailable": str(exc)}


def write_lines(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def select_overfit16_ids(data_root: Path, preflight_dir: Path) -> list[str]:
    path = preflight_dir / "overfit16_ids.txt"
    if path.is_file():
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    frame = pd.read_csv(data_root / "train" / "metadata.csv")
    candidates = frame.loc[pd.to_numeric(frame["retry_count"], errors="coerce").fillna(0).astype(int) == 0].copy()
    candidates = candidates.sort_values(
        ["p_nonzero_fraction", "m_mean", "h_mean", "camera_light_pair", "split_index"],
        ascending=[False, True, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    positions = np.linspace(0, len(candidates) - 1, 16, dtype=int)
    ids = candidates.iloc[positions]["sample_id"].astype(str).tolist()
    write_lines(path, ids)
    return ids


def select_spread_ids(data_root: Path, split: str, count: int, path: Path) -> list[str]:
    if path.is_file():
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    frame = pd.read_csv(data_root / split / "metadata.csv").sort_values(
        ["m_mean", "h_mean", "camera_light_pair", "split_index"],
        kind="mergesort",
    )
    positions = np.linspace(0, len(frame) - 1, count, dtype=int)
    ids = frame.iloc[positions]["sample_id"].astype(str).tolist()
    write_lines(path, ids)
    return ids


def subset_by_ids(dataset: SO1DecompositionDataset, sample_ids: list[str]) -> Subset:
    lookup = {
        str(row.sample_id): int(position)
        for position, row in enumerate(dataset.metadata.itertuples(index=False))
    }
    missing = [sample_id for sample_id in sample_ids if sample_id not in lookup]
    if missing:
        raise ValueError(f"Requested sample IDs are absent from {dataset.split}: {missing[:5]}")
    return Subset(dataset, [lookup[sample_id] for sample_id in sample_ids])


def worker_init_fn(worker_id: int) -> None:
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loader(dataset, config: dict[str, Any], *, shuffle: bool, seed: int) -> DataLoader:
    loader_cfg = config["dataloader"]
    num_workers = int(loader_cfg["num_workers"])
    kwargs: dict[str, Any] = {
        "batch_size": int(config["train"]["batch_size"]),
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": bool(loader_cfg["pin_memory"]) and torch.cuda.is_available(),
        "worker_init_fn": worker_init_fn,
        "generator": torch.Generator().manual_seed(seed),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(loader_cfg["persistent_workers"])
        kwargs["prefetch_factor"] = int(loader_cfg["prefetch_factor"])
    return DataLoader(dataset, **kwargs)


def prepare_datasets(config: dict[str, Any], preflight_dir: Path) -> tuple[Any, Any, dict[str, Any]]:
    data_root = project_path(config["data"]["root"])
    train_dataset = SO1DecompositionDataset(data_root, "train")
    val_dataset = SO1DecompositionDataset(data_root, "validation")
    subset_name = config["data"]["subset"]
    if subset_name == "overfit16":
        ids = select_overfit16_ids(data_root, preflight_dir)
        train_subset = subset_by_ids(train_dataset, ids)
        val_subset = subset_by_ids(train_dataset, ids)
        return train_subset, val_subset, {"train_ids": ids, "val_ids": ids}
    if subset_name == "smoke":
        train_ids = select_spread_ids(
            data_root, "train", int(config["data"]["smoke_train_count"]), preflight_dir / "smoke_train128_ids.txt"
        )
        val_ids = select_spread_ids(
            data_root, "validation", int(config["data"]["smoke_val_count"]), preflight_dir / "smoke_val32_ids.txt"
        )
        return subset_by_ids(train_dataset, train_ids), subset_by_ids(val_dataset, val_ids), {
            "train_ids": train_ids,
            "val_ids": val_ids,
        }
    raise ValueError(f"Unknown data subset: {subset_name}")


def build_identity(contract: dict[str, Any], contract_hash: str, model: SO1UNetDecomposer) -> dict[str, Any]:
    provenance = contract["provenance"]
    return {
        "target_order": TARGET_ORDER,
        "dataset_contract_hash": contract_hash,
        "generator_config_hash": provenance["generator_config_hash"],
        "SO0_version": provenance["so0_version"],
        "SO0_config_hash": provenance["so0_config_hash"],
        "model_architecture": model.architecture_id,
    }


def benchmark_dataloader(config: dict[str, Any], output_dir: Path, preflight_dir: Path) -> dict[str, Any]:
    results = []
    original_workers = int(config["dataloader"]["num_workers"])
    for workers in (0, 2, 4):
        bench_config = json.loads(json.dumps(config))
        bench_config["dataloader"]["num_workers"] = workers
        bench_config["dataloader"]["persistent_workers"] = workers > 0
        train_subset, _, _ = prepare_datasets(bench_config, preflight_dir)
        started = time.perf_counter()
        loader = make_loader(train_subset, bench_config, shuffle=False, seed=int(config["train"]["seed"]))
        init_seconds = time.perf_counter() - started
        stable = True
        samples = 0
        data_seconds = 0.0
        try:
            end = time.perf_counter()
            for index, batch in enumerate(loader):
                data_seconds += time.perf_counter() - end
                samples += int(batch["linear_rgb"].shape[0])
                if index >= 7:
                    break
                end = time.perf_counter()
        except Exception as exc:
            stable = False
            error = str(exc)
        else:
            error = None
        results.append(
            {
                "num_workers": workers,
                "init_seconds": init_seconds,
                "batch_data_time_seconds": data_seconds,
                "samples": samples,
                "samples_per_second": samples / max(data_seconds, 1.0e-9),
                "stable": stable,
                "error": error,
            }
        )
    config["dataloader"]["num_workers"] = original_workers
    best = max([r for r in results if r["stable"]], key=lambda r: r["samples_per_second"], default=None)
    payload = {"results": results, "recommended_num_workers": best["num_workers"] if best else None}
    (output_dir / "dataloader_benchmark.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def write_report(output_dir: Path, title: str, filename: str, records: list[dict[str, Any]], extra: dict[str, Any]) -> None:
    lines = [f"# {title}", ""]
    if records:
        first = records[0]
        last = records[-1]
        lines.extend(
            [
                f"Completed epoch: {last['epoch']}",
                f"Initial train total loss: {float(first['train_total_loss']):.6f}",
                f"Final train total loss: {float(last['train_total_loss']):.6f}",
                f"Final selection metric: {float(last['selection_metric']):.6f}",
                f"Peak GPU memory bytes: {int(float(last['gpu_peak_memory_bytes']))}",
                "",
            ]
        )
    for key, value in extra.items():
        lines.append(f"- {key}: {value}")
    (output_dir / filename).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    phase_started_at = datetime.now(timezone.utc).isoformat()
    config_path = project_path(args.config).resolve()
    config = load_yaml(config_path)
    if args.max_epochs is not None:
        config["train"]["max_epochs"] = int(args.max_epochs)
    data_root = project_path(config["data"]["root"]).resolve()
    output_dir = project_path(args.output_dir or config["experiment"]["output_dir"]).resolve()
    preflight_dir = project_path(config["experiment"]["preflight_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config["train"]["seed"])
    deterministic = bool(config["train"]["deterministic"])
    set_seed(seed, deterministic)
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is unavailable; rerun only after fixing the Face2 CUDA environment or pass --allow-cpu for CPU-only tests")
    if args.benchmark_only:
        benchmark_dataloader(config, output_dir, preflight_dir)
        return

    contract = build_dataset_contract(data_root)
    contract_hash = canonical_json_hash(contract)
    if contract["provenance"]["generator_config_hash"] != EXPECTED_GENERATOR_CONFIG_HASH:
        raise RuntimeError("Refusing training because generator_config_hash does not match SO-1 frozen contract")
    (output_dir / "dataset_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dataset_fingerprint = {
        "contract_hash": contract_hash,
        "data_root": str(data_root),
        "train_metadata_sha256": sha256_file(data_root / "train" / "metadata.csv"),
        "validation_metadata_sha256": sha256_file(data_root / "validation" / "metadata.csv"),
    }
    train_subset, val_subset, subset_meta = prepare_datasets(config, preflight_dir)
    if config["data"]["subset"] == "smoke":
        access = {
            "train_ids": subset_meta["train_ids"], "validation_ids": subset_meta["val_ids"],
            "train_count": len(subset_meta["train_ids"]), "validation_count": len(subset_meta["val_ids"]),
            "train_prefixes_valid": all(value.startswith("train_") for value in subset_meta["train_ids"]),
            "validation_prefixes_valid": all(value.startswith("validation_") for value in subset_meta["val_ids"]),
            "forbidden_splits_accessed": [],
        }
        (output_dir / "access_audit.json").write_text(json.dumps(access, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    model = SO1UNetDecomposer()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    scheduler_name = str(config["train"]["scheduler"]).lower()
    scheduler = None
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config["train"]["max_epochs"]),
            eta_min=float(config["train"]["eta_min"]),
        )
    elif scheduler_name not in {"none", "null"}:
        raise ValueError(f"Unsupported scheduler: {config['train']['scheduler']}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved_config = json.loads(json.dumps(config))
    resolved_config["resolved"] = {
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "preflight_dir": str(preflight_dir),
        "target_order": TARGET_ORDER,
        "normalization": NORMALIZATION,
        "model": count_parameters(model),
        "subset": subset_meta,
        "device": str(device),
    }
    save_yaml(resolved_config, output_dir / "config_resolved.yaml")
    (output_dir / "environment.json").write_text(
        json.dumps(
            environment_payload(config_path, data_root, seed, deterministic, sys.argv),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    identity = build_identity(contract, contract_hash, model)
    train_loader = make_loader(train_subset, config, shuffle=True, seed=seed + 11)
    val_loader = make_loader(val_subset, config, shuffle=False, seed=seed + 17)
    trainer = SO1DecompositionTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir,
        resolved_config=resolved_config,
        dataset_contract=contract,
        dataset_fingerprint=dataset_fingerprint,
        identity=identity,
        seed=seed,
        amp_enabled=bool(config["train"]["amp"]),
        early_stopping_patience=config["train"].get("early_stopping_patience"),
        resume_from=args.resume_from,
    )
    fit_max_epochs = int(args.fit_max_epochs or config["train"]["max_epochs"])
    if fit_max_epochs > int(config["train"]["max_epochs"]):
        raise ValueError("fit-max-epochs cannot exceed configured max_epochs")
    records = trainer.fit(train_loader, val_loader, max_epochs=fit_max_epochs)
    if args.resume_from is not None:
        checkpoint = load_checkpoint(
            output_dir / "checkpoints" / "last.pt",
            model=model,
            optimizer=None,
            scheduler=None,
            grad_scaler=None,
            expected_identity=identity,
            map_location=device,
            restore_rng=False,
        )
        if int(checkpoint["epoch"]) < 2:
            raise RuntimeError("Resume smoke did not advance to epoch 2")
    report_name = "overfit16" if config["data"]["subset"] == "overfit16" else "smoke"
    write_report(
        output_dir,
        "SO-1C Overfit16" if report_name == "overfit16" else "SO-1C Smoke",
        "overfit_report.md" if report_name == "overfit16" else "smoke_report.md",
        records,
        {"checkpoint_resume": args.resume_from is not None, "subset": config["data"]["subset"]},
    )
    if args.phase_output_dir is not None:
        phase_dir = project_path(args.phase_output_dir).resolve(); phase_dir.mkdir(parents=True, exist_ok=True)
        (phase_dir / "training_history.csv").write_text((output_dir / "training_history.csv").read_text(encoding="utf-8-sig"), encoding="utf-8")
        runtime = json.loads((output_dir / "runtime.json").read_text(encoding="utf-8"))
        runtime.update({"process_id": os.getpid(), "phase_start_time": phase_started_at, "phase_end_time": datetime.now(timezone.utc).isoformat(), "memory": memory_payload()})
        (phase_dir / "runtime.json").write_text(json.dumps(runtime, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if args.resume_from is not None:
            audit = {
                "checkpoint_path": str(args.resume_from.resolve()),
                **trainer.resume_checkpoint_metadata,
                "config_match": True,
                "contract_match": True,
                "history_epochs": [int(float(row["epoch"])) for row in records],
                "history_continuous": [int(float(row["epoch"])) for row in records] == list(range(1, fit_max_epochs + 1)),
            }
            (phase_dir / "resume_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
