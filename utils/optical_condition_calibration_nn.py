"""Deterministic utilities for Stage 2B nonlinear acquisition calibration.

This module accepts only acquisition conditions, identifiers, availability, and
the six fixed optical observations. It never loads images or clinical labels.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import torch
from torch import nn

from models.exif_conditioned_response_mlp import (
    ARCHITECTURE,
    EXPECTED_PARAMETER_COUNT,
    EXIFConditionedResponseMLP,
)
from utils.optical_condition_calibration import (
    ALL_TARGETS,
    CHEEK_TARGETS,
    CONDITION_NAMES,
    DESIGN_FEATURE_NAMES,
    EXPECTED_CAMERAS,
    FOREHEAD_CHEEK_TARGETS,
    build_design_matrix,
    camera_difference_metrics,
    fit_condition_scaler,
    fit_error_metrics,
    spearman_rho,
    transform_conditions,
    validate_camera_values,
)

TARGET_STD_EPSILON = 1.0e-8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_ids(ids: Sequence[str]) -> str:
    content = "\n".join(sorted(str(value) for value in ids)) + "\n"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def combined_file_sha256(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: value.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\n")
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(dict(payload)), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path, index=False, encoding="utf-8-sig", na_rep="",
        float_format="%.17g", lineterminator="\n",
    )


def set_cpu_determinism(seed: int, torch_threads: int = 1) -> dict[str, Any]:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(int(torch_threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return {
        "python_random_seed": int(seed),
        "numpy_seed": int(seed),
        "torch_seed": int(seed),
        "torch_use_deterministic_algorithms": True,
        "torch_threads": int(torch.get_num_threads()),
        "device": "cpu",
        "dtype": "float32",
    }


def fit_target_scaler(
    targets: pd.DataFrame | np.ndarray,
    target_names: Sequence[str],
    std_epsilon: float = TARGET_STD_EPSILON,
) -> dict[str, Any]:
    values = np.asarray(targets, dtype=np.float64)
    names = tuple(str(value) for value in target_names)
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError("Target matrix and target names do not align")
    if not np.isfinite(values).all():
        raise ValueError("Target scaler input contains nonfinite values")
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0, ddof=0)
    degenerate = [names[index] for index, value in enumerate(std) if value < float(std_epsilon)]
    if degenerate:
        raise ValueError(f"Target population std below {std_epsilon}: {degenerate}")
    return {
        "method": "training-subset target-wise population standardization",
        "target_names": list(names),
        "mean": mean.tolist(),
        "population_std": std.tolist(),
        "std_epsilon": float(std_epsilon),
        "ddof": 0,
        "train_n": int(len(values)),
    }


def standardize_targets(values: np.ndarray, scaler: Mapping[str, Any]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    std = np.asarray(scaler["population_std"], dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != len(mean) or not np.isfinite(array).all():
        raise ValueError("Invalid values for target standardization")
    return (array - mean) / std


def restore_target_scale(values: np.ndarray, scaler: Mapping[str, Any]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    std = np.asarray(scaler["population_std"], dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != len(mean) or not np.isfinite(array).all():
        raise ValueError("Invalid values for target scale restoration")
    return array * std + mean


def deterministic_inner_split(
    frame: pd.DataFrame,
    seed: int,
    val_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    validate_camera_values(frame["camera_id"], require_both=True)
    if frame["ID"].astype(str).duplicated().any():
        raise ValueError("Inner split source contains duplicate IDs")
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    camera_counts: dict[str, Any] = {}
    for camera in EXPECTED_CAMERAS:
        group = frame.loc[frame["camera_id"].astype(str).eq(camera)].copy()
        group["_inner_hash"] = group["ID"].astype(str).map(
            lambda identifier: hashlib.sha256(f"{int(seed)}|{identifier}".encode("utf-8")).hexdigest()
        )
        group = group.sort_values(["_inner_hash", "ID"], kind="stable")
        n_val = max(1, round(len(group) * float(val_fraction)))
        if n_val >= len(group):
            raise ValueError(f"Insufficient rows for inner train/val in camera {camera}")
        val_parts.append(group.iloc[:n_val].drop(columns="_inner_hash"))
        train_parts.append(group.iloc[n_val:].drop(columns="_inner_hash"))
        camera_counts[camera] = {
            "source_n": int(len(group)), "inner_train_n": int(len(group) - n_val),
            "inner_val_n": int(n_val),
        }
    train = pd.concat(train_parts, ignore_index=True).sort_values("ID", kind="stable").reset_index(drop=True)
    val = pd.concat(val_parts, ignore_index=True).sort_values("ID", kind="stable").reset_index(drop=True)
    if set(train["ID"]) & set(val["ID"]) or set(train["ID"]) | set(val["ID"]) != set(frame["ID"]):
        raise ValueError("Inner split is not an exact disjoint partition")
    manifest = {
        "seed": int(seed), "val_fraction": float(val_fraction),
        "stratification_fields": ["camera_id"], "hash_rule": 'SHA256(f"{seed}|{ID}")',
        "camera_counts": camera_counts,
        "train_n": int(len(train)), "val_n": int(len(val)),
        "train_id_sha256": sha256_ids(train["ID"].tolist()),
        "val_id_sha256": sha256_ids(val["ID"].tolist()),
    }
    return train, val, manifest


def is_improvement(best_loss: float, candidate_loss: float, minimum_improvement: float) -> bool:
    return math.isfinite(candidate_loss) and (
        not math.isfinite(best_loss) or best_loss - candidate_loss >= float(minimum_improvement)
    )


def should_early_stop(
    epoch: int,
    epochs_without_improvement: int,
    minimum_epochs: int,
    patience: int,
) -> bool:
    return int(epoch) >= int(minimum_epochs) and int(epochs_without_improvement) >= int(patience)


def _tensor(values: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(values), dtype=torch.float32, device="cpu")


def _loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        return float(torch.mean((model(x) - y) ** 2).cpu())


def save_selection_curve(log: pd.DataFrame, path: Path, selected_epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    axis.plot(log["epoch"], log["train_loss"], label="inner train", color="#2563EB", linewidth=1.5)
    axis.plot(log["epoch"], log["inner_val_loss"], label="inner val", color="#DC2626", linewidth=1.5)
    axis.axvline(selected_epoch, color="#111827", linestyle="--", linewidth=1.0, label=f"selected={selected_epoch}")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Standardized-target MSE")
    axis.set_title("Inner epoch selection")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, metadata={"Software": "face2-stage2b"})
    plt.close(figure)


def train_epoch_selection(
    inner_train: pd.DataFrame,
    inner_val: pd.DataFrame,
    target_names: Sequence[str],
    fold: int,
    network_type: str,
    seed: int,
    output_dir: Path,
    config: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_names = tuple(target_names)
    condition_scaler = {"fold": int(fold), "network_type": network_type, **fit_condition_scaler(inner_train)}
    condition_scaler_path = output_dir / "inner_condition_scaler.json"
    write_json(condition_scaler, condition_scaler_path)
    target_scaler = fit_target_scaler(inner_train.loc[:, list(target_names)], target_names)
    target_scaler_path = output_dir / "inner_target_scaler.json"
    write_json(target_scaler, target_scaler_path)
    train_scaled = transform_conditions(inner_train, condition_scaler)
    val_scaled = transform_conditions(inner_val, condition_scaler)
    x_train = _tensor(build_design_matrix(train_scaled))
    x_val = _tensor(build_design_matrix(val_scaled))
    y_train_raw = inner_train.loc[:, list(target_names)].to_numpy(float)
    y_val_raw = inner_val.loc[:, list(target_names)].to_numpy(float)
    y_train = _tensor(standardize_targets(y_train_raw, target_scaler))
    y_val = _tensor(standardize_targets(y_val_raw, target_scaler))
    determinism = set_cpu_determinism(seed, int(config["torch_threads"]))
    model = EXIFConditionedResponseMLP().to("cpu", dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    best_loss = math.inf
    best_epoch = 0
    stale = 0
    records: list[dict[str, Any]] = []
    checkpoint_path = output_dir / "best_checkpoint.pth"
    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x_train)
        loss = torch.mean((prediction - y_train) ** 2)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["gradient_clip_max_norm"])
        ))
        optimizer.step()
        train_loss = _loss(model, x_train, y_train)
        val_loss = _loss(model, x_val, y_val)
        improved = is_improvement(best_loss, val_loss, float(config["minimum_improvement"]))
        if improved:
            best_loss = val_loss
            best_epoch = epoch
            stale = 0
            torch.save({
                "model_state_dict": model.state_dict(), "architecture": ARCHITECTURE,
                "parameter_count": EXPECTED_PARAMETER_COUNT, "fold": int(fold),
                "network_type": network_type, "epoch": int(epoch), "seed": int(seed),
                "target_names": list(target_names), "condition_feature_names": list(DESIGN_FEATURE_NAMES),
                "inner_condition_scaler": condition_scaler, "inner_target_scaler": target_scaler,
            }, checkpoint_path)
        else:
            stale += 1
        records.append({
            "epoch": int(epoch), "train_loss": train_loss, "inner_val_loss": val_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "gradient_norm_before_clip": gradient_norm,
            "improved_by_minimum_threshold": int(improved),
            "epochs_without_improvement": int(stale),
        })
        if should_early_stop(
            epoch, stale, int(config["minimum_epochs"]), int(config["early_stopping_patience"])
        ):
            break
    if best_epoch <= 0 or not checkpoint_path.is_file():
        raise RuntimeError("Epoch selection did not produce a best checkpoint")
    log = pd.DataFrame(records)
    log_path = output_dir / "training_log.csv"
    write_csv(log, log_path)
    curve_path = output_dir / "selection_curve.png"
    save_selection_curve(log, curve_path, best_epoch)
    selected = {
        "fold": int(fold), "network_type": network_type,
        "selected_epoch": int(best_epoch), "best_inner_val_loss": float(best_loss),
        "best_inner_train_loss": float(log.loc[log["epoch"].eq(best_epoch), "train_loss"].iloc[0]),
        "tie_breaking": "minimum improvement of 1e-6; ties retain earlier epoch",
        "train_n": int(len(inner_train)), "val_n": int(len(inner_val)), "seed": int(seed),
        "inner_split_sha256": context["inner_split_sha256"],
        "inner_condition_scaler_sha256": sha256_file(condition_scaler_path),
        "inner_target_scaler_sha256": sha256_file(target_scaler_path),
        "config_sha256": context["config_sha256"],
        "stopped_epoch": int(log["epoch"].max()),
        "stop_reason": "early_stopping_patience" if int(log["epoch"].max()) < int(config["max_epochs"]) else "max_epochs",
        "outer_validation_loaded": False,
        "loss": "mean standardized-target MSE; three outputs equally weighted",
        "determinism": determinism,
    }
    write_json(selected, output_dir / "selected_epoch.json")
    return {
        "selected": selected, "condition_scaler": condition_scaler,
        "target_scaler": target_scaler, "training_log": log,
        "checkpoint_path": checkpoint_path, "curve_path": curve_path,
    }


def train_final_model(
    train: pd.DataFrame,
    target_names: Sequence[str],
    condition_scaler: Mapping[str, Any],
    fold: int,
    network_type: str,
    selected_epoch: int,
    seed: int,
    checkpoint_path: Path,
    target_scaler_path: Path,
    manifest_path: Path,
    config: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    target_names = tuple(target_names)
    target_scaler = fit_target_scaler(train.loc[:, list(target_names)], target_names)
    write_json(target_scaler, target_scaler_path)
    transformed = transform_conditions(train, condition_scaler)
    x = _tensor(build_design_matrix(transformed))
    y = _tensor(standardize_targets(train.loc[:, list(target_names)].to_numpy(float), target_scaler))
    determinism = set_cpu_determinism(seed, int(config["torch_threads"]))
    model = EXIFConditionedResponseMLP().to("cpu", dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    for _epoch in range(1, int(selected_epoch) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(x) - y) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_max_norm"]))
        optimizer.step()
    final_loss = _loss(model, x, y)
    payload = {
        "model_state_dict": model.state_dict(), "architecture": ARCHITECTURE,
        "input_dim": 5, "hidden_dims": [8, 8], "output_dim": 3,
        "activation": "Tanh", "parameter_count": EXPECTED_PARAMETER_COUNT,
        "fold": int(fold), "network_type": network_type,
        "selected_epoch": int(selected_epoch), "trained_epoch_count": int(selected_epoch),
        "seed": int(seed),
        "optimizer": {
            "name": "AdamW", "learning_rate": float(config["learning_rate"]),
            "weight_decay": float(config["weight_decay"]),
            "scheduler": "none", "batch_mode": "full_batch",
            "gradient_clip_max_norm": float(config["gradient_clip_max_norm"]),
        },
        "target_names": list(target_names),
        "condition_feature_names": list(DESIGN_FEATURE_NAMES),
        "target_scaler": target_scaler,
        "stage2a_condition_scaler_path": context["stage2a_condition_scaler_path"],
        "stage2a_condition_scaler_sha256": context["stage2a_condition_scaler_sha256"],
        "training_id_sha256": sha256_ids(train["ID"].tolist()),
        "training_n": int(len(train)), "split_sha256": context["split_sha256"],
        "config_sha256": context["config_sha256"],
        "first_stage_input_sha256": context["first_stage_sha256"],
        "stage2a_run_manifest_sha256": context["stage2a_manifest_sha256"],
        "pytorch_version": torch.__version__, "git_commit": context["git_commit"],
        "fresh_initialization_after_epoch_selection": True,
        "validation_loader_used": False, "final_standardized_train_mse": final_loss,
        "determinism": determinism,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)
    manifest = {key: value for key, value in payload.items() if key != "model_state_dict"}
    manifest.update({
        "status": "PASS", "checkpoint_path": context["checkpoint_relative_path"],
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "target_scaler_path": context["target_scaler_relative_path"],
        "target_scaler_sha256": sha256_file(target_scaler_path),
    })
    write_json(manifest, manifest_path)
    return {"model": model.eval(), "payload": payload, "manifest": manifest, "target_scaler": target_scaler}


def predict_original_scale(
    model: nn.Module,
    frame: pd.DataFrame,
    condition_scaler: Mapping[str, Any],
    target_scaler: Mapping[str, Any],
) -> np.ndarray:
    transformed = transform_conditions(frame, condition_scaler)
    x = _tensor(build_design_matrix(transformed))
    model.eval()
    with torch.no_grad():
        predicted_z = model(x).cpu().numpy().astype(np.float64)
    return restore_target_scale(predicted_z, target_scaler)


def transform_feature_frame(
    frame: pd.DataFrame,
    fold: int,
    role: str,
    condition_scaler: Mapping[str, Any],
    cheek_model: nn.Module,
    forehead_model: nn.Module,
    cheek_target_scaler: Mapping[str, Any],
    forehead_target_scaler: Mapping[str, Any],
) -> pd.DataFrame:
    transformed = transform_conditions(frame, condition_scaler)
    output = transformed.loc[:, [
        "ID", "camera_id", "forehead_available", *CONDITION_NAMES,
        "z_relative_optical_exposure", "z_log2_iso_condition",
    ]].copy()
    output.insert(1, "fold", int(fold))
    output.insert(2, "split_role", role)
    cheek_prediction = predict_original_scale(
        cheek_model, transformed, condition_scaler, cheek_target_scaler
    )
    cheek_mean = np.asarray(cheek_target_scaler["mean"], dtype=np.float64)
    for index, target in enumerate(CHEEK_TARGETS):
        raw = transformed[target].to_numpy(float)
        predicted = cheek_prediction[:, index]
        residual = raw - predicted
        output[f"raw_{target}"] = raw
        output[f"predicted_condition_nn_{target}"] = predicted
        output[f"residual_nn_{target}"] = residual
        output[f"calibrated_nn_{target}"] = residual + cheek_mean[index]
    available = transformed["forehead_available"].astype(int).eq(1).to_numpy()
    forehead_prediction = predict_original_scale(
        forehead_model, transformed.loc[available], condition_scaler, forehead_target_scaler
    )
    forehead_mean = np.asarray(forehead_target_scaler["mean"], dtype=np.float64)
    for index, target in enumerate(FOREHEAD_CHEEK_TARGETS):
        raw = transformed[target].to_numpy(float)
        predicted = np.full(len(transformed), np.nan, dtype=np.float64)
        predicted[available] = forehead_prediction[:, index]
        residual = np.full(len(transformed), np.nan, dtype=np.float64)
        calibrated = np.full(len(transformed), np.nan, dtype=np.float64)
        residual[available] = raw[available] - predicted[available]
        calibrated[available] = residual[available] + forehead_mean[index]
        output[f"raw_{target}"] = raw
        output[f"predicted_condition_nn_{target}"] = predicted
        output[f"residual_nn_{target}"] = residual
        output[f"calibrated_nn_{target}"] = calibrated
    return output.sort_values("ID", kind="stable").reset_index(drop=True)


def build_diagnostics(frame: pd.DataFrame, fold: int, role: str) -> dict[str, pd.DataFrame]:
    fit_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    camera_rows: list[dict[str, Any]] = []
    variance_rows: list[dict[str, Any]] = []
    scopes = [("overall", "ALL", frame)]
    scopes.extend(("camera_id", camera, frame.loc[frame["camera_id"].eq(camera)]) for camera in EXPECTED_CAMERAS)
    for target in ALL_TARGETS:
        fit_rows.append({
            "fold": fold, "split_role": role, "target": target,
            **fit_error_metrics(frame[f"raw_{target}"], frame[f"predicted_condition_nn_{target}"]),
        })
        for representation in ("raw", "residual_nn", "calibrated_nn"):
            column = f"{representation}_{target}"
            for scope, camera, subset in scopes:
                for condition in CONDITION_NAMES:
                    valid_n, rho = spearman_rho(subset[column], subset[condition])
                    correlation_rows.append({
                        "fold": fold, "split_role": role, "target": target,
                        "representation": representation, "scope": scope,
                        "camera_id": camera, "condition": condition,
                        "valid_n": valid_n, "spearman_rho": rho,
                    })
        for representation in ("raw", "calibrated_nn"):
            camera_rows.append({
                "fold": fold, "split_role": role, "target": target,
                "representation": representation,
                **camera_difference_metrics(frame, f"{representation}_{target}"),
            })
        raw = frame[f"raw_{target}"].dropna().to_numpy(float)
        calibrated = frame[f"calibrated_nn_{target}"].dropna().to_numpy(float)
        raw_variance = float(np.var(raw, ddof=0)) if len(raw) else math.nan
        calibrated_variance = float(np.var(calibrated, ddof=0)) if len(calibrated) else math.nan
        variance_rows.append({
            "fold": fold, "split_role": role, "target": target, "valid_n": len(raw),
            "raw_variance": raw_variance, "calibrated_nn_variance": calibrated_variance,
            "variance_retention_nn": calibrated_variance / raw_variance if raw_variance > 0 else math.nan,
        })
    return {
        "fit": pd.DataFrame(fit_rows), "correlation": pd.DataFrame(correlation_rows),
        "camera": pd.DataFrame(camera_rows), "variance": pd.DataFrame(variance_rows),
    }


def combined_diagnostics(diagnostics: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for family, frame in diagnostics.items():
        item = frame.copy()
        item.insert(0, "metric_family", family)
        frames.append(item)
    return pd.concat(frames, ignore_index=True, sort=False)


def condition_range_audit(
    train: pd.DataFrame,
    val: pd.DataFrame,
    scaler: Mapping[str, Any],
    fold: int,
) -> pd.DataFrame:
    train_z = transform_conditions(train, scaler)
    val_z = transform_conditions(val, scaler)
    rows: list[dict[str, Any]] = []
    for camera in EXPECTED_CAMERAS:
        train_camera = train_z.loc[train_z["camera_id"].eq(camera)]
        val_camera = val_z.loc[val_z["camera_id"].eq(camera)]
        ranges = {
            column: (float(train_camera[column].min()), float(train_camera[column].max()))
            for column in (*CONDITION_NAMES, "z_relative_optical_exposure", "z_log2_iso_condition")
        }
        for row in val_camera.sort_values("ID", kind="stable").itertuples(index=False):
            exposure_low = float(row.relative_optical_exposure) < ranges["relative_optical_exposure"][0]
            exposure_high = float(row.relative_optical_exposure) > ranges["relative_optical_exposure"][1]
            iso_low = float(row.log2_iso_condition) < ranges["log2_iso_condition"][0]
            iso_high = float(row.log2_iso_condition) > ranges["log2_iso_condition"][1]
            exposure_out = exposure_low or exposure_high
            iso_out = iso_low or iso_high
            rows.append({
                "ID": str(row.ID), "fold": int(fold), "camera_id": camera,
                "train_exposure_min": ranges["relative_optical_exposure"][0],
                "train_exposure_max": ranges["relative_optical_exposure"][1],
                "train_iso_min": ranges["log2_iso_condition"][0],
                "train_iso_max": ranges["log2_iso_condition"][1],
                "train_z_exposure_min": ranges["z_relative_optical_exposure"][0],
                "train_z_exposure_max": ranges["z_relative_optical_exposure"][1],
                "train_z_iso_min": ranges["z_log2_iso_condition"][0],
                "train_z_iso_max": ranges["z_log2_iso_condition"][1],
                "relative_optical_exposure": float(row.relative_optical_exposure),
                "log2_iso_condition": float(row.log2_iso_condition),
                "z_relative_optical_exposure": float(row.z_relative_optical_exposure),
                "z_log2_iso_condition": float(row.z_log2_iso_condition),
                "exposure_below_train_min": int(exposure_low),
                "exposure_above_train_max": int(exposure_high),
                "iso_below_train_min": int(iso_low), "iso_above_train_max": int(iso_high),
                "any_condition_outside_train_range": int(exposure_out or iso_out),
                "both_conditions_outside_train_range": int(exposure_out and iso_out),
            })
    return pd.DataFrame(rows).sort_values("ID", kind="stable").reset_index(drop=True)
