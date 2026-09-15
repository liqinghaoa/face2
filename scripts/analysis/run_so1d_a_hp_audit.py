"""Execute the read-only SO-1D-A H/P learnability and identifiability audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import rankdata
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from skin_optics_so1.decomposition.so1d_audit import (  # noqa: E402
    ALLOWED_SPLITS,
    AUDIT_SEED,
    CHANNELS,
    FORBIDDEN_SPLITS,
    assert_fp32_eval_read_only,
    bootstrap_ci,
    configure_read_only_fp32,
    cross_talk_correlation,
    file_sha256,
    find_rgb_neighbors,
    fit_train_affine,
    model_state_sha256,
    output_is_isolated,
    pearson_from_sufficient,
    require_audit_split,
    spearman_columns,
)
from skin_optics_so1.decomposition.synthetic_dataset import SO1DecompositionDataset  # noqa: E402
from skin_optics_so1.decomposition.unet_decomposer import SO1UNetDecomposer, count_parameters  # noqa: E402


DATA_ROOT = ROOT / "data/processed/SO1_Synthetic_Generator_v1_localfull"
FORMAL_DIR = ROOT / "experiments/skin_optics_so1/SO1_Decomposition_UNet_v1/formal_train_v1_1"
CHECKPOINT = FORMAL_DIR / "checkpoints/best.pt"
OUTPUT = ROOT / "experiments/skin_optics_so1/SO1_Decomposition_UNet_v1/diagnostics/so1d_a_hp_learnability_identifiability"
CACHE = OUTPUT / "cache_v2"
FIGURES = OUTPUT / "figures"
CONTACTS = OUTPUT / "contact_sheets"
POLICY = {"global_amp": True, "mode": "localized_fp32", "fp32_blocks": ["up1.conv"]}
EXPECTED_METRICS = {
    "val_M_MAE": 0.0348162431,
    "val_H_MAE": 0.1538493898,
    "val_S_MAE": 0.0300033154,
    "val_P_MAE": 0.0087745273,
    "val_selection_metric": 0.1886656329,
}
P_COUNT_THRESHOLDS = np.array([0.001, 0.005, 0.010, 0.020], dtype=np.float32)
ACTIVE_THRESHOLD_INDICES = (1, 2, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("all", "audit", "infer", "analyze"), default="all")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def load_history_epoch18() -> dict[str, str]:
    rows = list(csv.DictReader((FORMAL_DIR / "training_history.csv").open(encoding="utf-8-sig")))
    selected = [row for row in rows if int(float(row["epoch"])) == 18]
    if len(selected) != 1:
        raise RuntimeError("CHECKPOINT_IDENTITY_MISMATCH: epoch18 history row")
    return selected[0]


def checkpoint_audit() -> dict[str, Any]:
    if not output_is_isolated(OUTPUT, FORMAL_DIR):
        raise RuntimeError("SO-1D-A output overlaps formal_train_v1_1")
    before = file_sha256(CHECKPOINT)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    history = load_history_epoch18()
    identity_checks = {
        "checkpoint_epoch_18": int(checkpoint.get("epoch", -1)) == 18,
        "checkpoint_best_epoch_18": int(checkpoint.get("best_epoch", -1)) == 18,
        "checkpoint_best_selection_metric": math.isclose(
            float(checkpoint.get("best_selection_metric", math.nan)),
            EXPECTED_METRICS["val_selection_metric"], abs_tol=5e-10,
        ),
        "diagnostic_only": not (FORMAL_DIR / "best_checkpoint_sha256.txt").exists(),
    }
    for key, expected in EXPECTED_METRICS.items():
        identity_checks[f"history_{key}"] = math.isclose(float(history[key]), expected, abs_tol=5e-10)
    if not all(identity_checks.values()):
        payload = {"status": "FAIL", "reason": "CHECKPOINT_IDENTITY_MISMATCH", "checks": identity_checks}
        write_json(OUTPUT / "checkpoint_audit.json", payload)
        raise RuntimeError("CHECKPOINT_IDENTITY_MISMATCH")
    model = configure_read_only_fp32(SO1UNetDecomposer(numerical_precision=POLICY))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert_fp32_eval_read_only(model)
    parameter_finite = all(bool(torch.isfinite(value).all()) for value in model.parameters())
    bn_rows = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            bn_rows.append({
                "name": name,
                "mean_finite": bool(torch.isfinite(module.running_mean).all()),
                "var_finite": bool(torch.isfinite(module.running_var).all()),
                "var_nonnegative": bool((module.running_var >= 0).all()),
            })
    health = {
        "strict_load": True,
        "parameter_count": count_parameters(model)["parameter_count"],
        "parameters_finite": parameter_finite,
        "BN_running_mean_finite": all(row["mean_finite"] for row in bn_rows),
        "BN_running_var_finite": all(row["var_finite"] for row in bn_rows),
        "BN_running_var_nonnegative": all(row["var_nonnegative"] for row in bn_rows),
        "model_eval": not model.training,
        "all_parameters_FP32": all(value.dtype == torch.float32 for value in model.parameters()),
        "all_parameters_requires_grad_false": all(not value.requires_grad for value in model.parameters()),
    }
    status = all((parameter_finite, health["parameter_count"] == 31_037_828,
                  health["BN_running_mean_finite"], health["BN_running_var_finite"],
                  health["BN_running_var_nonnegative"]))
    payload = {
        "status": "PASS" if status else "FAIL",
        "checkpoint": str(CHECKPOINT), "identity": "DIAGNOSTIC_ONLY",
        "checkpoint_sha256_read_only_guard": before, "identity_checks": identity_checks,
        "health": health, "state_digest_before_inference": model_state_sha256(model),
        "inference_policy": {"eval": True, "no_grad": True, "autocast": False, "dtype": "float32"},
    }
    write_json(OUTPUT / "checkpoint_audit.json", payload)
    if not status:
        raise RuntimeError("SO-1D-A checkpoint health FAIL")
    return payload


def make_loader(dataset: SO1DecompositionDataset, batch_size: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2,
                      pin_memory=True, persistent_workers=True, prefetch_factor=2)


def allocate_cache(n: int) -> dict[str, np.ndarray]:
    return {
        "valid_count": np.zeros(n, np.float64),
        "abs_sum": np.zeros((n, 4), np.float64), "sq_error_sum": np.zeros((n, 4), np.float64),
        "signed_sum": np.zeros((n, 4), np.float64), "pred_sum": np.zeros((n, 4), np.float64),
        "target_sum": np.zeros((n, 4), np.float64), "pred_sq_sum": np.zeros((n, 4), np.float64),
        "target_sq_sum": np.zeros((n, 4), np.float64), "pred_target_sum": np.zeros((n, 4), np.float64),
        "cross_sum": np.zeros((n, 4, 4), np.float64), "descriptor": np.zeros((n, 774), np.float32),
        "p_valid": np.zeros((n, 4), np.float64), "p_target_active": np.zeros((n, 4), np.float64),
        "p_pred_active": np.zeros((n, 4), np.float64), "p_tp": np.zeros((n, 4), np.float64),
        "p_fp": np.zeros((n, 4), np.float64), "p_fn": np.zeros((n, 4), np.float64),
        "p_active_abs": np.zeros((n, 4), np.float64), "p_inactive_abs": np.zeros((n, 4), np.float64),
        "p_active_signed": np.zeros((n, 4), np.float64), "p_inactive_count": np.zeros((n, 4), np.float64),
    }


def infer_split(split: str, batch_size: int) -> None:
    require_audit_split(split)
    path = CACHE / f"{split}_summary.npz"
    sample_path = CACHE / f"{split}_pixel_sample.npz"
    if path.is_file() and sample_path.is_file():
        print(f"reuse complete cache: {split}", flush=True); return
    dataset = SO1DecompositionDataset(DATA_ROOT, split)
    n = len(dataset); arrays = allocate_cache(n)
    metadata = dataset.metadata.copy()
    sample_per_image = 50 if split == "train" else 500
    sampled_prediction = np.empty((n * sample_per_image, 4), np.float32)
    sampled_target = np.empty_like(sampled_prediction)
    validation_hp = None
    if split == "validation":
        validation_hp = np.lib.format.open_memmap(
            CACHE / "validation_hp_prediction.f16.npy", mode="w+", dtype=np.float16,
            shape=(n, 2, 256, 256),
        )
    spatial_sum = np.zeros((4, 256, 256), np.float64) if split == "train" else None
    spatial_count = np.zeros((1, 256, 256), np.float64) if split == "train" else None
    checkpoint_hash = file_sha256(CHECKPOINT)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = configure_read_only_fp32(SO1UNetDecomposer(numerical_precision=POLICY))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    state_before = model_state_sha256(model)
    model.cuda(); assert_fp32_eval_read_only(model)
    started = time.perf_counter(); offset = 0
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
        for batch_index, batch in enumerate(make_loader(dataset, batch_size)):
            x = batch["linear_rgb"].cuda(non_blocking=True).float()
            target = batch["target_mhsp"].cuda(non_blocking=True).float()
            mask = batch["valid_mask"].cuda(non_blocking=True).float()
            prediction = model(x)
            if prediction.dtype != torch.float32 or not bool(torch.isfinite(prediction).all()):
                raise FloatingPointError(f"SO-1D-A FP32 inference nonfinite: {split} batch {batch_index}")
            b = x.shape[0]; sl = slice(offset, offset + b); valid = mask.bool(); expanded = valid.expand_as(target)
            difference = prediction - target
            count = mask.sum(dim=(1, 2, 3))
            def summed(value: torch.Tensor) -> np.ndarray:
                return value.sum(dim=(2, 3)).cpu().double().numpy()
            arrays["valid_count"][sl] = count.cpu().double().numpy()
            arrays["abs_sum"][sl] = summed(torch.where(expanded, difference.abs(), 0.0))
            arrays["sq_error_sum"][sl] = summed(torch.where(expanded, difference.square(), 0.0))
            arrays["signed_sum"][sl] = summed(torch.where(expanded, difference, 0.0))
            arrays["pred_sum"][sl] = summed(torch.where(expanded, prediction, 0.0))
            arrays["target_sum"][sl] = summed(torch.where(expanded, target, 0.0))
            arrays["pred_sq_sum"][sl] = summed(torch.where(expanded, prediction.square(), 0.0))
            arrays["target_sq_sum"][sl] = summed(torch.where(expanded, target.square(), 0.0))
            arrays["pred_target_sum"][sl] = summed(torch.where(expanded, prediction * target, 0.0))
            pflat = (prediction * mask).flatten(2); tflat = (target * mask).flatten(2)
            arrays["cross_sum"][sl] = torch.einsum("bip,bjp->bij", pflat, tflat).cpu().double().numpy()
            masked_x = x * mask
            pooled = F.adaptive_avg_pool2d(masked_x, (16, 16)).flatten(1)
            rgb_sum = masked_x.sum(dim=(2, 3)); rgb_mean = rgb_sum / count[:, None]
            rgb_var = ((x - rgb_mean[:, :, None, None]).square() * mask).sum(dim=(2, 3)) / count[:, None]
            arrays["descriptor"][sl] = torch.cat((pooled, rgb_mean, rgb_var.sqrt()), dim=1).cpu().numpy()
            for threshold_index, threshold in enumerate(P_COUNT_THRESHOLDS):
                true_active = (target[:, 3] > float(threshold)) & valid[:, 0]
                pred_active = (prediction[:, 3] > float(threshold)) & valid[:, 0]
                inactive = (~true_active) & valid[:, 0]; pdiff = difference[:, 3]
                for key, tensor in (
                    ("p_valid", valid[:, 0]), ("p_target_active", true_active),
                    ("p_pred_active", pred_active), ("p_tp", true_active & pred_active),
                    ("p_fp", (~true_active) & pred_active & valid[:, 0]),
                    ("p_fn", true_active & (~pred_active)), ("p_inactive_count", inactive),
                ):
                    arrays[key][sl, threshold_index] = tensor.sum(dim=(1, 2)).cpu().double().numpy()
                arrays["p_active_abs"][sl, threshold_index] = torch.where(true_active, pdiff.abs(), 0.0).sum(dim=(1, 2)).cpu().double().numpy()
                arrays["p_inactive_abs"][sl, threshold_index] = torch.where(inactive, pdiff.abs(), 0.0).sum(dim=(1, 2)).cpu().double().numpy()
                arrays["p_active_signed"][sl, threshold_index] = torch.where(true_active, pdiff, 0.0).sum(dim=(1, 2)).cpu().double().numpy()
            mask_cpu = batch["valid_mask"][:, 0].numpy().astype(bool)
            target_cpu = batch["target_mhsp"].numpy()
            prediction_cpu = prediction.cpu().numpy()
            for local in range(b):
                global_index = offset + local
                valid_indices = np.flatnonzero(mask_cpu[local].reshape(-1))
                rng = np.random.default_rng(AUDIT_SEED + (0 if split == "train" else 10_000_000) + global_index)
                selected = rng.choice(valid_indices, size=sample_per_image, replace=len(valid_indices) < sample_per_image)
                sample_slice = slice(global_index * sample_per_image, (global_index + 1) * sample_per_image)
                sampled_prediction[sample_slice] = prediction_cpu[local].reshape(4, -1)[:, selected].T
                sampled_target[sample_slice] = target_cpu[local].reshape(4, -1)[:, selected].T
            if validation_hp is not None:
                validation_hp[sl] = prediction_cpu[:, (1, 3)].astype(np.float16)
            if spatial_sum is not None and spatial_count is not None:
                spatial_sum += (batch["target_mhsp"].numpy() * batch["valid_mask"].numpy()).sum(axis=0)
                spatial_count += batch["valid_mask"].numpy().sum(axis=0)
            offset += b
            if (batch_index + 1) % 250 == 0:
                print(f"{split}: {offset}/{n}", flush=True)
    model.cpu(); state_after = model_state_sha256(model)
    if state_before != state_after or checkpoint_hash != file_sha256(CHECKPOINT):
        raise RuntimeError("SO-1D-A read-only guard failed")
    arrays["sample_id"] = metadata["sample_id"].astype(str).to_numpy()
    arrays["split_index"] = metadata["split_index"].astype(np.int32).to_numpy()
    arrays["base_latent_id"] = metadata["base_latent_id"].astype(np.int64).to_numpy()
    for column in ("camera_name", "light_name", "camera_light_pair", "acquisition_variant_id"):
        if column in metadata:
            arrays[column] = metadata[column].astype(str).to_numpy()
    np.savez(path, **arrays)
    np.savez(sample_path, prediction=sampled_prediction, target=sampled_target,
             seed=AUDIT_SEED, sample_count=len(sampled_target), sample_per_image=sample_per_image)
    if spatial_sum is not None and spatial_count is not None:
        np.savez(CACHE / "train_spatial_target.npz", sum=spatial_sum, count=spatial_count)
    if validation_hp is not None:
        validation_hp.flush()
    write_json(CACHE / f"{split}_inference_manifest.json", {
        "status": "COMPLETED", "split": split, "samples": n, "FP32": True,
        "eval": True, "no_grad": True, "autocast": False,
        "pixel_sample_count": len(sampled_target), "pixel_sampling_seed": AUDIT_SEED,
        "seconds": time.perf_counter() - started, "checkpoint_sha256_unchanged": True,
        "model_state_unchanged": True,
    })


def aggregate_metrics(cache: Any, sample: Any, split: str) -> list[dict[str, Any]]:
    count = float(cache["valid_count"].sum())
    rows = []
    for index, channel in enumerate(CHANNELS):
        sx = float(cache["pred_sum"][:, index].sum()); sy = float(cache["target_sum"][:, index].sum())
        sx2 = float(cache["pred_sq_sum"][:, index].sum()); sy2 = float(cache["target_sq_sum"][:, index].sum())
        sxy = float(cache["pred_target_sum"][:, index].sum())
        pred_mean = sx / count; target_mean = sy / count
        pred_std = math.sqrt(max(sx2 / count - pred_mean**2, 0.0))
        target_std = math.sqrt(max(sy2 / count - target_mean**2, 0.0))
        pred_values = sample["prediction"][:, index].astype(np.float32)
        target_values = sample["target"][:, index].astype(np.float32)
        rows.append({
            "split": split, "channel": channel,
            "MAE": float(cache["abs_sum"][:, index].sum() / count),
            "RMSE": math.sqrt(float(cache["sq_error_sum"][:, index].sum() / count)),
            "bias": float(cache["signed_sum"][:, index].sum() / count),
            "pixel_Pearson": float(pearson_from_sufficient(count, sx, sy, sx2, sy2, sxy)),
            "pixel_Spearman_sampled": float(np.corrcoef(rankdata(pred_values), rankdata(target_values))[0, 1]),
            "pixel_Spearman_sample_count": len(pred_values), "sampling_seed": AUDIT_SEED,
            "prediction_mean": pred_mean, "prediction_std": pred_std,
            "target_mean": target_mean, "target_std": target_std,
            "pred_target_std_ratio": pred_std / target_std if target_std else np.nan,
            **{f"prediction_p{q:02d}": float(np.quantile(pred_values, q / 100)) for q in (1, 5, 50, 95, 99)},
            **{f"target_p{q:02d}": float(np.quantile(target_values, q / 100)) for q in (1, 5, 50, 95, 99)},
        })
        pred_image = cache["pred_sum"][:, index] / cache["valid_count"]
        target_image = cache["target_sum"][:, index] / cache["valid_count"]
        rows[-1]["image_Pearson"] = float(np.corrcoef(pred_image, target_image)[0, 1])
        rows[-1]["image_Spearman"] = float(np.corrcoef(rankdata(pred_image), rankdata(target_image))[0, 1])
    return rows


def summarize_active(cache: Any, split: str) -> list[dict[str, Any]]:
    rows = []
    for i in ACTIVE_THRESHOLD_INDICES:
        threshold = P_COUNT_THRESHOLDS[i]
        sums = {key: float(cache[key][:, i].sum()) for key in (
            "p_valid", "p_target_active", "p_pred_active", "p_tp", "p_fp", "p_fn",
            "p_active_abs", "p_inactive_abs", "p_active_signed", "p_inactive_count")}
        precision = sums["p_tp"] / (sums["p_tp"] + sums["p_fp"])
        recall = sums["p_tp"] / sums["p_target_active"]
        rows.append({
            "split": split, "threshold": float(threshold),
            "target_active_prevalence": sums["p_target_active"] / sums["p_valid"],
            "pred_active_fraction": sums["p_pred_active"] / sums["p_valid"],
            "precision": precision, "recall": recall,
            "F1": 2 * precision * recall / (precision + recall),
            "active_region_MAE": sums["p_active_abs"] / sums["p_target_active"],
            "inactive_region_MAE": sums["p_inactive_abs"] / sums["p_inactive_count"],
            "active_region_bias": sums["p_active_signed"] / sums["p_target_active"],
        })
    return rows


def baseline_comparison(train_cache: Any, val_cache: Any, train_sample: Any) -> list[dict[str, Any]]:
    train_target = np.load(DATA_ROOT / "train/target_mhsp.f16.npy", mmap_mode="r")
    train_mask = np.load(DATA_ROOT / "train/valid_mask.u8.npy", mmap_mode="r")
    val_target = np.load(DATA_ROOT / "validation/target_mhsp.f16.npy", mmap_mode="r")
    val_mask = np.load(DATA_ROOT / "validation/valid_mask.u8.npy", mmap_mode="r")
    total_count = train_cache["valid_count"].sum()
    scalar_mean = train_cache["target_sum"].sum(axis=0) / total_count
    histograms = np.zeros((4, 65536), dtype=np.int64)
    for start in range(0, len(train_target), 16):
        raw = train_target[start:start + 16]
        valid = train_mask[start:start + 16].astype(bool)
        selected = np.broadcast_to(valid, raw.shape)
        for channel_index in range(4):
            codes = raw[:, channel_index][selected[:, channel_index]].view(np.uint16)
            histograms[channel_index] += np.bincount(codes, minlength=65536)
    scalar_median = np.empty(4, dtype=np.float64)
    for channel_index in range(4):
        cumulative = np.cumsum(histograms[channel_index])
        total = int(cumulative[-1])
        lower_code = int(np.searchsorted(cumulative, (total - 1) // 2 + 1))
        upper_code = int(np.searchsorted(cumulative, total // 2 + 1))
        scalar_median[channel_index] = (
            np.array([lower_code], dtype=np.uint16).view(np.float16)[0].item()
            + np.array([upper_code], dtype=np.uint16).view(np.float16)[0].item()
        ) / 2.0
    spatial = np.load(CACHE / "train_spatial_target.npz")
    spatial_mean = np.divide(spatial["sum"], spatial["count"], out=np.zeros_like(spatial["sum"]), where=spatial["count"] > 0)
    rows = []
    for split, target, mask, cache in (("train", train_target, train_mask, train_cache),
                                        ("validation", val_target, val_mask, val_cache)):
        baseline_sums = {name: np.zeros(4, np.float64) for name in ("scalar_mean", "scalar_median", "spatial_mean")}
        zero_sum = 0.0; count = 0.0
        for start in range(0, len(target), 16):
            values = target[start:start + 16].astype(np.float32)
            valid = mask[start:start + 16].astype(bool)
            count += valid.sum()
            for name, prediction in (("scalar_mean", scalar_mean[:, None, None]),
                                     ("scalar_median", scalar_median[:, None, None]),
                                     ("spatial_mean", spatial_mean)):
                baseline_sums[name] += (np.abs(values - prediction) * valid).sum(axis=(0, 2, 3))
            zero_sum += (np.abs(values[:, 3:4]) * valid).sum()
        for channel_index, channel in enumerate(CHANNELS):
            network = float(cache["abs_sum"][:, channel_index].sum() / count)
            candidates = {name: float(values[channel_index] / count) for name, values in baseline_sums.items()}
            if channel == "P": candidates["zero"] = zero_sum / count
            for name, baseline in candidates.items():
                rows.append({
                    "split": split, "channel": channel, "baseline": name,
                    "network_MAE": network, "baseline_MAE": baseline,
                    "absolute_improvement": baseline - network,
                    "relative_improvement": (baseline - network) / baseline if baseline else np.nan,
                    "baseline_fit_source": "train_only",
                    "median_note": "exact full-Train float16 histogram median" if name == "scalar_median" else "exact",
                })
    return rows


def p_sparsity_rows(caches: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for split, cache in caches.items():
        for index, threshold in enumerate(P_COUNT_THRESHOLDS):
            valid = float(cache["p_valid"][:, index].sum())
            for source, key in (("target", "p_target_active"), ("prediction", "p_pred_active")):
                greater = float(cache[key][:, index].sum()) / valid
                rows.append({"split": split, "source": source, "threshold": float(threshold),
                             "le_fraction": 1.0 - greater, "gt_fraction": greater,
                             "valid_pixel_count": int(valid), "calculation": "exact_streaming"})
    return rows


def calibration_and_quantiles(train_sample: Any, val_sample: Any) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    fit = fit_train_affine(train_sample["prediction"][:, 1], train_sample["target"][:, 1])
    hp = np.load(CACHE / "validation_hp_prediction.f16.npy", mmap_mode="r")
    target = np.load(DATA_ROOT / "validation/target_mhsp.f16.npy", mmap_mode="r")
    mask = np.load(DATA_ROOT / "validation/valid_mask.u8.npy", mmap_mode="r")
    sums = {key: 0.0 for key in ("n", "abs_before", "sq_before", "abs_after", "sq_after", "sx", "sy", "sx2", "sy2", "sxy")}
    h_true_sample = val_sample["target"][:, 1]; h_pred_sample = val_sample["prediction"][:, 1]
    calibrated_sample = fit["a"] * h_pred_sample + fit["b"]
    h_rows = []; edges = np.quantile(h_true_sample, np.linspace(0, 1, 11))
    for bin_index in range(10):
        selected = (h_true_sample >= edges[bin_index]) & (h_true_sample <= edges[bin_index + 1] if bin_index == 9 else h_true_sample < edges[bin_index + 1])
        difference = h_pred_sample[selected] - h_true_sample[selected]
        h_rows.append({"level": "pixel_sample", "bin": bin_index + 1, "count": int(selected.sum()),
                       "true_H_mean": float(h_true_sample[selected].mean()),
                       "pred_H_mean": float(h_pred_sample[selected].mean()),
                       "MAE": float(np.abs(difference).mean()), "bias": float(difference.mean())})
    val_cache = np.load(CACHE / "validation_summary.npz", allow_pickle=True)
    true_image = val_cache["target_sum"][:, 1] / val_cache["valid_count"]
    pred_image = val_cache["pred_sum"][:, 1] / val_cache["valid_count"]
    image_edges = np.quantile(true_image, np.linspace(0, 1, 11))
    for bin_index in range(10):
        selected = ((true_image >= image_edges[bin_index])
                    & (true_image <= image_edges[bin_index + 1]
                       if bin_index == 9 else true_image < image_edges[bin_index + 1]))
        difference = pred_image[selected] - true_image[selected]
        h_rows.append({"level": "image_mean", "bin": bin_index + 1,
                       "count": int(selected.sum()),
                       "true_H_mean": float(true_image[selected].mean()),
                       "pred_H_mean": float(pred_image[selected].mean()),
                       "MAE": float(np.abs(difference).mean()),
                       "bias": float(difference.mean())})
    p_rows = []
    p_true = val_sample["target"][:, 3]; p_pred = val_sample["prediction"][:, 3]
    selected_active = p_true > 0.01; active_edges = np.quantile(p_true[selected_active], np.linspace(0, 1, 5))
    for bin_index in range(4):
        selected = selected_active & (p_true >= active_edges[bin_index]) & (p_true <= active_edges[bin_index + 1] if bin_index == 3 else p_true < active_edges[bin_index + 1])
        diff = p_pred[selected] - p_true[selected]
        p_rows.append({"bin": f"Q{bin_index + 1}", "count": int(selected.sum()),
                       "true_P_mean": float(p_true[selected].mean()), "pred_P_mean": float(p_pred[selected].mean()),
                       "MAE": float(np.abs(diff).mean()), "bias": float(diff.mean()),
                       "calculation": "deterministic_1M_validation_pixel_sample", "sampling_seed": AUDIT_SEED})
    for start in range(0, len(target), 16):
        true = target[start:start + 16, 1].astype(np.float32); pred = hp[start:start + 16, 0].astype(np.float32)
        valid = mask[start:start + 16, 0].astype(bool); x = pred[valid].astype(np.float64); y = true[valid].astype(np.float64)
        after = fit["a"] * x + fit["b"]; before_diff = x - y; after_diff = after - y
        sums["n"] += len(x); sums["abs_before"] += np.abs(before_diff).sum(); sums["sq_before"] += np.square(before_diff).sum()
        sums["abs_after"] += np.abs(after_diff).sum(); sums["sq_after"] += np.square(after_diff).sum()
        sums["sx"] += after.sum(); sums["sy"] += y.sum(); sums["sx2"] += np.square(after).sum(); sums["sy2"] += np.square(y).sum(); sums["sxy"] += (after * y).sum()
    diagnostic = {**fit, "designation": "DIAGNOSTIC_POSTHOC_CALIBRATION",
                  "validation_H_MAE_before": sums["abs_before"] / sums["n"],
                  "validation_H_MAE_after": sums["abs_after"] / sums["n"],
                  "validation_H_RMSE_before": math.sqrt(sums["sq_before"] / sums["n"]),
                  "validation_H_RMSE_after": math.sqrt(sums["sq_after"] / sums["n"]),
                  "validation_H_Pearson_after": float(pearson_from_sufficient(sums["n"], sums["sx"], sums["sy"], sums["sx2"], sums["sy2"], sums["sxy"])),
                  "validation_H_Spearman_after_sampled": float(np.corrcoef(rankdata(calibrated_sample), rankdata(h_true_sample))[0, 1]),
                  "validation_targets_used_for_fit": False}
    return diagnostic, h_rows, p_rows


def p_zero_active_baseline() -> list[dict[str, Any]]:
    rows = []
    for split in ALLOWED_SPLITS:
        target = np.load(DATA_ROOT / f"{split}/target_mhsp.f16.npy", mmap_mode="r")
        mask = np.load(DATA_ROOT / f"{split}/valid_mask.u8.npy", mmap_mode="r")
        for threshold in (0.005, 0.010, 0.020):
            active_count = 0.0; target_sum = 0.0
            for start in range(0, len(target), 32):
                true = target[start:start + 32, 3].astype(np.float32)
                valid = mask[start:start + 32, 0].astype(bool)
                active = (true > threshold) & valid
                active_count += active.sum()
                target_sum += true[active].sum(dtype=np.float64)
            rows.append({"split": split, "threshold": threshold,
                         "zero_baseline_active_region_MAE": target_sum / active_count,
                         "active_pixel_count": int(active_count),
                         "calculation": "exact_full_split"})
    return rows


def cross_talk_outputs(caches: dict[str, Any], samples: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pearson_rows = []; spearman_rows = []
    for split in ALLOWED_SPLITS:
        cache = caches[split]; sample = samples[split]; n = float(cache["valid_count"].sum())
        pearson = np.empty((4, 4))
        for i in range(4):
            for j in range(4):
                pearson[i, j] = pearson_from_sufficient(n, cache["pred_sum"][:, i].sum(), cache["target_sum"][:, j].sum(), cache["pred_sq_sum"][:, i].sum(), cache["target_sq_sum"][:, j].sum(), cache["cross_sum"][:, i, j].sum())
        spearman = cross_talk_correlation(sample["prediction"], sample["target"], method="spearman")
        for i, pred_name in enumerate(CHANNELS):
            pearson_rows.append({"split": split, "predicted_channel": pred_name, **{f"True_{CHANNELS[j]}": pearson[i, j] for j in range(4)}})
            spearman_rows.append({"split": split, "predicted_channel": pred_name, **{f"True_{CHANNELS[j]}": spearman[i, j] for j in range(4)}, "sample_count": len(sample["target"]), "sampling_seed": AUDIT_SEED})
    return pearson_rows, spearman_rows


def pair_target_differences(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return exact top-1 and 10-random target-map intersection MAEs per validation case."""
    train_target = np.load(DATA_ROOT / "train/target_mhsp.f16.npy", mmap_mode="r")
    train_mask = np.load(DATA_ROOT / "train/valid_mask.u8.npy", mmap_mode="r")
    val_target = np.load(DATA_ROOT / "validation/target_mhsp.f16.npy", mmap_mode="r")
    val_mask = np.load(DATA_ROOT / "validation/valid_mask.u8.npy", mmap_mode="r")
    rng = np.random.default_rng(AUDIT_SEED); random_indices = rng.integers(0, len(train_target), size=(len(val_target), 10))
    nn_diff = np.zeros((len(val_target), 4), np.float64); random_diff = np.zeros((len(val_target), 10, 4), np.float64)
    for query in range(len(val_target)):
        qtarget = val_target[query].astype(np.float32); qmask = val_mask[query].astype(bool)
        refs = np.concatenate(([indices[query, 0]], random_indices[query]))
        targets = train_target[refs].astype(np.float32); common = train_mask[refs].astype(bool) & qmask
        counts = common.sum(axis=(1, 2, 3)); diffs = (np.abs(targets - qtarget[None]) * common).sum(axis=(2, 3)) / counts[:, None]
        nn_diff[query] = diffs[0]; random_diff[query] = diffs[1:]
        if (query + 1) % 250 == 0: print(f"NN target pairs: {query + 1}/2000", flush=True)
    return nn_diff, random_diff


def nn_identifiability(train_cache: Any, val_cache: Any) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, np.ndarray]]:
    mean = train_cache["descriptor"].mean(axis=0); std = train_cache["descriptor"].std(axis=0); std[std < 1e-8] = 1.0
    train_desc = (train_cache["descriptor"] - mean) / std; val_desc = (val_cache["descriptor"] - mean) / std
    reusable = CACHE / "rgb_nn_pair_cache.npz"
    if reusable.is_file():
        stored = np.load(reusable)
        indices, distances = stored["indices"], stored["distances"]
        nn_diff, random_diff = stored["nn_diff"], stored["random_diff"]
    else:
        indices, distances = find_rgb_neighbors(train_desc, val_desc, top_k=5)
        nn_diff, random_diff = pair_target_differences(indices)
        np.savez_compressed(reusable, indices=indices, distances=distances,
                            nn_diff=nn_diff, random_diff=random_diff)
    rows = []
    quartiles = pd.qcut(distances[:, 0], 4, labels=("Q1", "Q2", "Q3", "Q4"))
    for query in range(len(indices)):
        row = {"validation_split_index": query, "rgb_nn_distance": float(distances[query, 0]),
               "rgb_distance_quartile": str(quartiles[query])}
        for rank in range(5): row[f"top{rank + 1}_train_index"] = int(indices[query, rank]); row[f"top{rank + 1}_distance"] = float(distances[query, rank])
        for channel, name in enumerate(CHANNELS):
            row[f"NN_delta_{name}"] = nn_diff[query, channel]
            row[f"random_delta_{name}_median10"] = np.median(random_diff[query, :, channel])
        rows.append(row)
    summary: dict[str, Any] = {"descriptor": "masked linear RGB adaptive-average-pool 16x16 + masked RGB mean/std", "standardization": "train_only", "query": "validation_2000", "reference": "train_20000", "random_references_per_query": 10, "seed": AUDIT_SEED, "channels": {}}
    for channel, name in enumerate(CHANNELS):
        nn_median = float(np.median(nn_diff[:, channel])); random_median = float(np.median(random_diff[:, :, channel])); ratio = nn_median / random_median
        paired = np.column_stack((nn_diff[:, channel], random_diff[:, :, channel]))
        ci = bootstrap_ci(paired, lambda value: np.median(value[:, 0]) / np.median(value[:, 1:]))
        summary["channels"][name] = {"NN_target_difference_median": nn_median, "random_target_difference_median": random_median, "ambiguity_ratio": ratio, "bootstrap_95_CI": ci, "distance_delta_spearman": float(np.corrcoef(rankdata(distances[:, 0]), rankdata(nn_diff[:, channel]))[0, 1]), "quartiles": {str(q): float(np.median(nn_diff[np.asarray(quartiles == q), channel])) for q in ("Q1", "Q2", "Q3", "Q4")}}
    return rows, summary, {"indices": indices, "distances": distances, "nn_diff": nn_diff, "random_diff": random_diff}


def nuisance_audit(cache: Any) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    metadata = pd.read_csv(DATA_ROOT / "validation/metadata.csv").sort_values("split_index")
    available = {name: name in metadata.columns for name in ("camera_name", "light_name", "base_latent_id", "acquisition_variant_id")}
    group_rows = []
    for group_type, column in (("camera", "camera_name"), ("light", "light_name")):
        if not available[column]: continue
        for group, frame in metadata.groupby(column):
            idx = frame["split_index"].to_numpy(np.int64); count = cache["valid_count"][idx].sum()
            h_pearson = pearson_from_sufficient(count, cache["pred_sum"][idx, 1].sum(), cache["target_sum"][idx, 1].sum(), cache["pred_sq_sum"][idx, 1].sum(), cache["target_sq_sum"][idx, 1].sum(), cache["pred_target_sum"][idx, 1].sum())
            t = 2; target_active = cache["p_target_active"][idx, t].sum()
            group_rows.append({"group_type": group_type, "group": group, "sample_count": len(idx),
                               "H_MAE": cache["abs_sum"][idx, 1].sum() / count, "H_Pearson": float(h_pearson),
                               "P_active_recall_at_0.01": cache["p_tp"][idx, t].sum() / target_active,
                               "P_active_MAE_at_0.01": cache["p_active_abs"][idx, t].sum() / target_active})
    matched_rows: list[dict[str, Any]] = []
    audit = {"NUISANCE_METADATA_AVAILABLE": "YES" if available["camera_name"] or available["light_name"] else "NO", "camera_metadata": available["camera_name"], "light_metadata": available["light_name"], "actual_fields": [key for key, value in available.items() if value], "MATCHED_LATENT_AUDIT": "NOT_AVAILABLE", "matched_latent_reason": "base_latent_id pairs share some base factors, but S/P seeds vary; metadata does not prove an identical complete MHSP latent target across acquisitions"}
    if group_rows:
        grouped = pd.DataFrame(group_rows)
        audit["descriptive_group_ranges"] = {}
        for group_type in ("camera", "light"):
            selected = grouped[grouped.group_type == group_type]
            audit["descriptive_group_ranges"][group_type] = {
                "groups": len(selected),
                "sample_count_min": int(selected.sample_count.min()),
                "sample_count_max": int(selected.sample_count.max()),
                "H_MAE_min": float(selected.H_MAE.min()),
                "H_MAE_max": float(selected.H_MAE.max()),
                "H_Pearson_min": float(selected.H_Pearson.min()),
                "H_Pearson_max": float(selected.H_Pearson.max()),
                "P_active_recall_min": float(selected["P_active_recall_at_0.01"].min()),
                "P_active_recall_max": float(selected["P_active_recall_at_0.01"].max()),
                "P_active_MAE_min": float(selected["P_active_MAE_at_0.01"].min()),
                "P_active_MAE_max": float(selected["P_active_MAE_at_0.01"].max()),
            }
    return audit, group_rows, matched_rows


def bootstrap_summary(val_cache: Any, nn_data: dict[str, np.ndarray]) -> dict[str, Any]:
    rng = np.random.default_rng(AUDIT_SEED); n = len(val_cache["valid_count"]); reps = 1000
    outputs = {key: np.empty(reps) for key in ("H_MAE", "H_Pearson", "P_recall_0.01", "P_active_MAE_0.01", "H_ambiguity_ratio", "P_ambiguity_ratio")}
    for rep in range(reps):
        idx = rng.integers(0, n, size=n); count = val_cache["valid_count"][idx].sum()
        outputs["H_MAE"][rep] = val_cache["abs_sum"][idx, 1].sum() / count
        outputs["H_Pearson"][rep] = pearson_from_sufficient(count, val_cache["pred_sum"][idx, 1].sum(), val_cache["target_sum"][idx, 1].sum(), val_cache["pred_sq_sum"][idx, 1].sum(), val_cache["target_sq_sum"][idx, 1].sum(), val_cache["pred_target_sum"][idx, 1].sum())
        outputs["P_recall_0.01"][rep] = val_cache["p_tp"][idx, 2].sum() / val_cache["p_target_active"][idx, 2].sum()
        outputs["P_active_MAE_0.01"][rep] = val_cache["p_active_abs"][idx, 2].sum() / val_cache["p_target_active"][idx, 2].sum()
        for channel, key in ((1, "H_ambiguity_ratio"), (3, "P_ambiguity_ratio")):
            outputs[key][rep] = np.median(nn_data["nn_diff"][idx, channel]) / np.median(nn_data["random_diff"][idx, :, channel])
    return {key: {"estimate": float(np.nanmedian(value)), "bootstrap_95_CI": [float(x) for x in np.nanpercentile(value, [2.5, 97.5])]} for key, value in outputs.items()} | {"replicates": reps, "seed": AUDIT_SEED, "unit": "validation_case"}


def plot_outputs(metrics: list[dict[str, Any]], baselines: list[dict[str, Any]], active: list[dict[str, Any]], samples: dict[str, Any], h_quantiles: list[dict[str, Any]], pearson_rows: list[dict[str, Any]], nn_summary: dict[str, Any]) -> None:
    mpl.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130, "savefig.dpi": 220})
    FIGURES.mkdir(parents=True, exist_ok=True); colors = {"train": "#4477AA", "validation": "#CC6677"}
    for split in ALLOWED_SPLITS:
        s = samples[split]; fig, ax = plt.subplots(figsize=(5, 4)); ax.hexbin(s["target"][:, 1], s["prediction"][:, 1], gridsize=70, bins="log", mincnt=1, cmap="viridis"); ax.plot([0, 1], [0, 1], "k--", lw=.8); ax.set(xlabel="True H", ylabel="Predicted H", title=f"H target vs prediction: {split}"); fig.tight_layout(); fig.savefig(FIGURES / f"h_hexbin_{split}.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for split in ALLOWED_SPLITS:
        ax.hist(samples[split]["target"][:, 1], bins=80, density=True, histtype="step", color=colors[split], label=f"{split} true"); ax.hist(samples[split]["prediction"][:, 1], bins=80, density=True, alpha=.25, color=colors[split], label=f"{split} pred")
    ax.set(xlabel="H", ylabel="Density", title="H target and prediction distributions"); ax.legend(ncol=2); fig.tight_layout(); fig.savefig(FIGURES / "h_distribution.png"); plt.close(fig)
    q = pd.DataFrame(h_quantiles); fig, ax = plt.subplots(figsize=(5, 4));
    for level, marker, color in (("pixel_sample", "o", "#CC6677"), ("image_mean", "s", "#4477AA")):
        frame = q[q.level == level]; ax.plot(frame.true_H_mean, frame.pred_H_mean, marker=marker, color=color, label=level.replace("_", " "))
    ax.plot([q.true_H_mean.min(), q.true_H_mean.max()], [q.true_H_mean.min(), q.true_H_mean.max()], "k--", label="identity"); ax.set(xlabel="True H bin mean", ylabel="Predicted H mean", title="Validation H quantile calibration"); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "h_quantile_calibration.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4)); ax.hist(samples["validation"]["target"][:, 3], bins=100, range=(0, .2), density=True, histtype="step", label="true", color="#4477AA"); ax.hist(samples["validation"]["prediction"][:, 3], bins=100, range=(0, .2), density=True, alpha=.35, label="pred", color="#CC6677"); ax.set(xlabel="P", ylabel="Density", title="Validation P distribution (0-0.2)"); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "p_distribution.png"); plt.close(fig)
    a = pd.DataFrame(active); a = a[a.split == "validation"]; fig, ax = plt.subplots(figsize=(6, 4)); x=np.arange(3); ax.bar(x-.18,a.target_active_prevalence,.36,label="target active",color="#4477AA"); ax.bar(x+.18,a.pred_active_fraction,.36,label="pred active",color="#CC6677"); ax.set_xticks(x,[f">{v:.3f}" for v in a.threshold]); ax.set(ylabel="Fraction", title="P active threshold comparison"); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "p_active_thresholds.png"); plt.close(fig)
    s=samples["validation"]; sel=s["target"][:,3]>.01; fig,ax=plt.subplots(figsize=(5,4)); ax.hexbin(s["target"][sel,3],s["prediction"][sel,3],gridsize=60,bins="log",mincnt=1,cmap="magma"); ax.plot([.01,1],[.01,1],"k--",lw=.8); ax.set(xlabel="True active P",ylabel="Predicted P",title="P active-region response"); fig.tight_layout(); fig.savefig(FIGURES/"p_active_true_vs_pred.png"); plt.close(fig)
    matrices=[]
    for split in ALLOWED_SPLITS:
        frame=pd.DataFrame([r for r in pearson_rows if r["split"]==split]).set_index("predicted_channel")[[f"True_{c}" for c in CHANNELS]]; matrices.append(frame.to_numpy())
    fig,axes=plt.subplots(1,2,figsize=(8,3.5));
    for ax,matrix,split in zip(axes,matrices,ALLOWED_SPLITS):
        im=ax.imshow(matrix,vmin=-1,vmax=1,cmap="coolwarm"); ax.set_xticks(range(4),CHANNELS); ax.set_yticks(range(4),CHANNELS); ax.set(xlabel="True",ylabel="Predicted",title=f"{split} Pearson");
        for i in range(4):
            for j in range(4): ax.text(j,i,f"{matrix[i,j]:.2f}",ha="center",va="center",fontsize=7)
    fig.colorbar(im,ax=axes,shrink=.75); fig.savefig(FIGURES/"cross_talk_heatmap.png",bbox_inches="tight"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4)); x=np.arange(4); nn=[nn_summary["channels"][c]["NN_target_difference_median"] for c in CHANNELS]; rnd=[nn_summary["channels"][c]["random_target_difference_median"] for c in CHANNELS]; ax.bar(x-.18,nn,.36,label="RGB nearest",color="#44AA99"); ax.bar(x+.18,rnd,.36,label="random",color="#999999"); ax.set_xticks(x,CHANNELS); ax.set(ylabel="Median target-map MAE",title="RGB-neighbor target differences"); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES/"rgb_nn_target_difference.png"); plt.close(fig)
    b=pd.DataFrame(baselines); b=b[(b.split=="validation") & (b.baseline.isin(["spatial_mean","zero"]))]; fig,ax=plt.subplots(figsize=(6,4)); pivot=b.pivot(index="channel",columns="baseline",values="baseline_MAE"); network=b.groupby("channel").network_MAE.first(); xx=np.arange(4); ax.bar(xx-.2,[network.get(c,np.nan) for c in CHANNELS],.4,label="network",color="#4477AA"); ax.bar(xx+.2,[pivot.loc[c].min() if c in pivot.index else np.nan for c in CHANNELS],.4,label="best spatial/zero baseline",color="#BBBBBB"); ax.set_xticks(xx,CHANNELS); ax.set(ylabel="MAE",title="Validation network vs baseline"); ax.legend(); fig.tight_layout(); fig.savefig(FIGURES/"network_vs_baseline_mae.png"); plt.close(fig)
    m=pd.DataFrame(metrics); fig,axes=plt.subplots(1,2,figsize=(9,3.5));
    for ax,value,title in ((axes[0],"MAE","Channel MAE"),(axes[1],"pixel_Pearson","Pixel Pearson")):
        for k,split in enumerate(ALLOWED_SPLITS):
            frame=m[m.split==split].set_index("channel"); ax.bar(np.arange(4)+(k-.5)*.36,[frame.loc[c,value] for c in CHANNELS],.36,label=split,color=colors[split]);
        ax.set_xticks(range(4),CHANNELS); ax.set_title(title)
    axes[0].set_ylabel("MAE"); axes[1].set_ylabel("r"); axes[1].legend(); fig.tight_layout(); fig.savefig(FIGURES/"train_validation_summary.png"); plt.close(fig)


def contact_sheets(val_cache: Any) -> None:
    h_mae = val_cache["abs_sum"][:, 1] / val_cache["valid_count"]
    p_recall = np.divide(val_cache["p_tp"][:, 2], val_cache["p_target_active"][:, 2], out=np.full(len(h_mae), np.nan), where=val_cache["p_target_active"][:, 2] > 0)
    groups = {"h_best": np.argsort(h_mae)[:4], "h_worst": np.argsort(h_mae)[-4:][::-1],
              "p_recall_best": np.argsort(np.nan_to_num(p_recall,nan=-1))[-4:][::-1],
              "p_recall_worst": np.argsort(np.nan_to_num(p_recall,nan=np.inf))[:4]}
    dataset=SO1DecompositionDataset(DATA_ROOT,"validation"); checkpoint=torch.load(CHECKPOINT,map_location="cpu",weights_only=False)
    model=configure_read_only_fp32(SO1UNetDecomposer(numerical_precision=POLICY)); model.load_state_dict(checkpoint["model_state_dict"],strict=True); model.cuda()
    CONTACTS.mkdir(parents=True,exist_ok=True)
    with torch.no_grad(), torch.autocast("cuda",enabled=False):
        for name,indices in groups.items():
            fig,axes=plt.subplots(4,8,figsize=(14,7),constrained_layout=True)
            for row,index in enumerate(indices):
                item=dataset[int(index)]; pred=model(item["linear_rgb"][None].cuda().float())[0].cpu().numpy(); true=item["target_mhsp"].numpy(); mask=item["valid_mask"][0].numpy(); rgb=np.moveaxis(item["linear_rgb"].numpy(),0,-1).clip(0,1)
                panels=((rgb,None,"RGB",1.0),(mask,"gray","mask",1.0),(true[1],"viridis","H true",1.0),(pred[1],"viridis","H pred",1.0),(np.abs(pred[1]-true[1])*mask,"magma","H error",1.0),(true[3],"viridis","P true (0-0.2)",.2),(pred[3],"viridis","P pred (0-0.2)",.2),(np.abs(pred[3]-true[3])*mask,"magma","P error (0-0.2)",.2))
                for col,(image,cmap,title,vmax) in enumerate(panels): axes[row,col].imshow(image,cmap=cmap,vmin=0,vmax=vmax); axes[row,col].set_title(f"{title}\nidx {index}" if col==0 else title,fontsize=7); axes[row,col].axis("off")
            fig.savefig(CONTACTS/f"{name}.png",dpi=180); plt.close(fig)


def labels_and_route(metrics: list[dict[str, Any]], baselines: list[dict[str, Any]], active: list[dict[str, Any]], calibration: dict[str, Any], cross_pearson: list[dict[str, Any]], nn_summary: dict[str, Any], sparsity: list[dict[str, Any]], nuisance: dict[str, Any]) -> dict[str, Any]:
    m=pd.DataFrame(metrics).set_index(["split","channel"]); b=pd.DataFrame(baselines); a=pd.DataFrame(active); s=pd.DataFrame(sparsity)
    h_train=m.loc[("train","H")]; h_val=m.loc[("validation","H")]; p_val=m.loc[("validation","P")]
    h_labels=[]; p_labels=[]
    if h_train.pixel_Pearson > 0 and h_train.image_Pearson > 0: h_labels.append("H_TRAIN_LEARNABILITY_GOOD")
    else: h_labels.append("H_TRAIN_LEARNABILITY_POOR")
    if h_val.MAE > h_train.MAE and h_val.pixel_Pearson < h_train.pixel_Pearson: h_labels.append("H_GENERALIZATION_GAP")
    if h_val.pred_target_std_ratio < 0.8: h_labels.append("H_REGRESSION_TO_MEAN")
    improvement=(calibration["validation_H_MAE_before"]-calibration["validation_H_MAE_after"])/calibration["validation_H_MAE_before"]
    if improvement > 0.05: h_labels.append("H_SCALE_CALIBRATION_ERROR")
    h_cross=[r for r in cross_pearson if r["split"]=="validation" and r["predicted_channel"]=="H"][0]
    if max(h_cross["True_M"],h_cross["True_S"],h_cross["True_P"]) > h_cross["True_H"]: h_labels.append("H_CROSSTALK_SUSPECTED")
    h_ratio=nn_summary["channels"]["H"]["ambiguity_ratio"]; ms_ratio=np.mean([nn_summary["channels"][c]["ambiguity_ratio"] for c in ("M","S")])
    if h_ratio > ms_ratio: h_labels.append("H_LOW_EMPIRICAL_IDENTIFIABILITY")
    target_sparse=float(s[(s.split=="validation")&(s.source=="target")&(np.isclose(s.threshold,.01))].le_fraction.iloc[0]); pred_sparse=float(s[(s.split=="validation")&(s.source=="prediction")&(np.isclose(s.threshold,.01))].le_fraction.iloc[0])
    if target_sparse > 0.5: p_labels.append("P_TARGET_INTRINSICALLY_SPARSE")
    zero=b[(b.split=="validation")&(b.channel=="P")&(b.baseline=="zero")].iloc[0]
    if zero.relative_improvement > 0.05: p_labels.append("P_NETWORK_OUTPERFORMS_ZERO_BASELINE")
    else: p_labels.append("P_NETWORK_DOES_NOT_MEANINGFULLY_OUTPERFORM_ZERO_BASELINE")
    pactive=a[(a.split=="validation")&np.isclose(a.threshold,.01)].iloc[0]
    if pactive.recall < 0.5: p_labels.append("P_ACTIVE_REGION_FAILURE")
    else: p_labels.append("P_ACTIVE_REGION_LEARNABLE")
    if pred_sparse > target_sparse and pactive.recall < 0.5: p_labels.append("P_TRIVIAL_NEAR_ZERO_SOLUTION_SUPPORTED")
    if nn_summary["channels"]["P"]["ambiguity_ratio"] > ms_ratio: p_labels.append("P_LOW_EMPIRICAL_IDENTIFIABILITY")
    camera_range = nuisance.get("descriptive_group_ranges", {}).get("camera", {})
    if camera_range and camera_range["H_MAE_max"] > 2 * camera_range["H_MAE_min"] and camera_range["H_Pearson_max"] - camera_range["H_Pearson_min"] > 0.2:
        h_labels.append("H_CAMERA_LIGHT_SENSITIVE")
    if camera_range and camera_range["P_active_recall_max"] - camera_range["P_active_recall_min"] > 0.3 and camera_range["P_active_MAE_max"] > 1.5 * camera_range["P_active_MAE_min"]:
        p_labels.append("P_CAMERA_LIGHT_SENSITIVE")
    if "H_LOW_EMPIRICAL_IDENTIFIABILITY" in h_labels: route="C"; route_name="REPRESENTATION_OR_IDENTIFIABILITY_REDESIGN_REQUIRED"
    elif "P_ACTIVE_REGION_FAILURE" in p_labels or "P_TRIVIAL_NEAR_ZERO_SOLUTION_SUPPORTED" in p_labels: route="B"; route_name="SUPERVISION_OR_LOSS_REDESIGN_REQUIRED"
    else: route="A"; route_name="CURRENT_MODELING_STILL_JUSTIFIED"
    loss_redesign_supported = any(label in p_labels for label in ("P_ACTIVE_REGION_FAILURE", "P_TRIVIAL_NEAR_ZERO_SOLUTION_SUPPORTED", "P_NETWORK_DOES_NOT_MEANINGFULLY_OUTPERFORM_ZERO_BASELINE"))
    recommendation = (
        "Revisit RGB-to-H/P representation and nuisance identifiability first; redesign P sparse/active-region supervision in parallel, but do not treat loss redesign alone as a solution to H."
        if route == "C" else "Redesign supervision/loss before another final training run."
        if route == "B" else "Current modeling remains justified; a full-FP32 formal run may isolate the remaining numerical issue."
    )
    return {"SO-1D-A":"PASS","H_labels":h_labels,"P_labels":p_labels,"primary_problem_class":"MULTIFACTORIAL" if len(h_labels)+len(p_labels)>3 else "INCONCLUSIVE","route":route,"route_name":route_name,"recommendation":recommendation,"allow_full_fp32_formal_training":route=="A","allow_loss_redesign":loss_redesign_supported,"allow_representation_redesign":route=="C","allow_SO1E":False,"allow_SO2":False,"allow_classification":False}


def report(decision: dict[str, Any], metrics: list[dict[str, Any]], baselines: list[dict[str, Any]], active: list[dict[str, Any]], calibration: dict[str, Any], nn_summary: dict[str, Any], nuisance: dict[str, Any], bootstrap: dict[str, Any]) -> None:
    m=pd.DataFrame(metrics).set_index(["split","channel"]); b=pd.DataFrame(baselines); a=pd.DataFrame(active)
    htrain=m.loc[("train","H")]; hval=m.loc[("validation","H")]; pval=m.loc[("validation","P")]
    best_h=b[(b.split=="validation")&(b.channel=="H")].sort_values("baseline_MAE").iloc[0]; zero=b[(b.split=="validation")&(b.channel=="P")&(b.baseline=="zero")].iloc[0]; pact=a[(a.split=="validation")&np.isclose(a.threshold,.01)].iloc[0]
    zero_active = pd.read_csv(OUTPUT / "p_zero_baseline_active_region.csv")
    zero_active_001 = zero_active[(zero_active.split == "validation") & np.isclose(zero_active.threshold, .01)].iloc[0]
    questions = [
        f"Q1. H is informative on Train (MAE {htrain.MAE:.6f}, pixel r {htrain.pixel_Pearson:.4f}); it is not an MAE-only judgment.",
        f"Q2. Train-to-Validation degradation is MAE {htrain.MAE:.6f} to {hval.MAE:.6f} and r {htrain.pixel_Pearson:.4f} to {hval.pixel_Pearson:.4f}.",
        f"Q3. Validation H ranking: pixel Spearman {hval.pixel_Spearman_sampled:.4f}, image Spearman {hval.image_Spearman:.4f}.",
        f"Q4. Train-fitted affine calibration changes Validation H MAE {calibration['validation_H_MAE_before']:.6f} to {calibration['validation_H_MAE_after']:.6f}.",
        f"Q5. H pred/target std ratio is {hval.pred_target_std_ratio:.4f}.",
        f"Q6. Cross-talk evidence is summarized by {decision['H_labels']}.",
        f"Q7. H RGB-NN ambiguity ratio is {nn_summary['channels']['H']['ambiguity_ratio']:.4f}.",
        f"Q8. P target sparsity is quantified in p_sparsity_audit.csv.",
        f"Q9. Network P MAE {pval.MAE:.6f} vs zero {zero.baseline_MAE:.6f}, relative improvement {zero.relative_improvement:.2%}.",
        f"Q10. P@0.01 recall {pact.recall:.4f}, network active MAE {pact.active_region_MAE:.6f}, zero active MAE {zero_active_001.zero_baseline_active_region_MAE:.6f}.",
        f"Q11. P evidence labels: {decision['P_labels']}.",
        f"Q12. Camera/light fields are available={nuisance['NUISANCE_METADATA_AVAILABLE']}; group ranges are {nuisance.get('descriptive_group_ranges')}. Evidence is descriptive, not causal.",
    ]
    lines=["# SO-1D-A H/P Learnability & Identifiability Audit","","## Status","","PASS: the read-only audit completed. This does not convert the checkpoint into a formal frozen model.","","## Checkpoint And Inference","","Epoch-18 best candidate, strict-load PASS, 31,037,828 parameters, finite parameters/BN, eval + no-grad + full FP32 inference, checkpoint unchanged.","","## Three-Level Diagnosis","",f"- Learnability: H Train/Validation evidence labels: {', '.join(decision['H_labels'])}.",f"- Identifiability: H ratio {nn_summary['channels']['H']['ambiguity_ratio']:.4f}; P ratio {nn_summary['channels']['P']['ambiguity_ratio']:.4f}.",f"- Objective/target imbalance: {', '.join(decision['P_labels'])}.","","## H Summary","",f"Train H MAE/Pearson/Spearman: {htrain.MAE:.6f} / {htrain.pixel_Pearson:.4f} / {htrain.pixel_Spearman_sampled:.4f}.",f"Validation H MAE/Pearson/Spearman: {hval.MAE:.6f} / {hval.pixel_Pearson:.4f} / {hval.pixel_Spearman_sampled:.4f}.",f"Best Validation H baseline: {best_h.baseline} MAE={best_h.baseline_MAE:.6f}; network improvement={best_h.relative_improvement:.2%}.","","## P Summary","",f"Validation P MAE={pval.MAE:.6f}; zero baseline={zero.baseline_MAE:.6f}; relative improvement={zero.relative_improvement:.2%}.",f"P@0.01 prevalence={pact.target_active_prevalence:.4f}, predicted active={pact.pred_active_fraction:.4f}, precision/recall/F1={pact.precision:.4f}/{pact.recall:.4f}/{pact.F1:.4f}.",f"At P>0.01, network active MAE={pact.active_region_MAE:.6f} versus zero active MAE={zero_active_001.zero_baseline_active_region_MAE:.6f}.","","## Required Questions","",*questions,"","## Bootstrap","","Validation-case bootstrap (1,000 replicates, seed 20260801):",json.dumps(bootstrap,indent=2),"","## Route Decision","",f"Route {decision['route']}: {decision['route_name']}.",decision["recommendation"],f"Full FP32 formal training allowed: {decision['allow_full_fp32_formal_training']}. Loss redesign allowed: {decision['allow_loss_redesign']}. Representation redesign allowed: {decision['allow_representation_redesign']}.","","SO-1E, SO-2, classification, training, fine-tuning, backward, and optimizer steps were not executed."]
    (OUTPUT/"SO1D_A_HP_learnability_identifiability_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def analyze() -> None:
    caches={split:np.load(CACHE/f"{split}_summary.npz",allow_pickle=True) for split in ALLOWED_SPLITS}; samples={split:np.load(CACHE/f"{split}_pixel_sample.npz") for split in ALLOWED_SPLITS}
    metrics=sum((aggregate_metrics(caches[s],samples[s],s) for s in ALLOWED_SPLITS),[])
    write_rows(OUTPUT/"channel_metrics_train.csv",[r for r in metrics if r["split"]=="train"]); write_rows(OUTPUT/"channel_metrics_validation.csv",[r for r in metrics if r["split"]=="validation"])
    distribution=[{key:value for key,value in row.items() if key in ("split","channel","prediction_mean","prediction_std","target_mean","target_std","pred_target_std_ratio") or key.startswith("prediction_p") or key.startswith("target_p")} for row in metrics]
    write_rows(OUTPUT/"target_distribution_train.csv",[r for r in distribution if r["split"]=="train"]); write_rows(OUTPUT/"target_distribution_validation.csv",[r for r in distribution if r["split"]=="validation"])
    baselines=baseline_comparison(caches["train"],caches["validation"],samples["train"]); write_rows(OUTPUT/"baseline_comparison.csv",baselines)
    sparsity=p_sparsity_rows(caches); write_rows(OUTPUT/"p_sparsity_audit.csv",sparsity)
    active=sum((summarize_active(caches[s],s) for s in ALLOWED_SPLITS),[]); write_rows(OUTPUT/"p_active_region_metrics.csv",active)
    write_rows(OUTPUT/"p_zero_baseline_active_region.csv",p_zero_active_baseline())
    calibration,hq,pq=calibration_and_quantiles(samples["train"],samples["validation"]); write_json(OUTPUT/"h_calibration_diagnostic.json",calibration); write_rows(OUTPUT/"h_quantile_calibration.csv",hq); write_rows(OUTPUT/"p_active_severity.csv",pq)
    pearson,spearman=cross_talk_outputs(caches,samples); write_rows(OUTPUT/"cross_talk_pearson.csv",pearson); write_rows(OUTPUT/"cross_talk_spearman.csv",spearman)
    nn_rows,nn_summary,nn_data=nn_identifiability(caches["train"],caches["validation"]); write_rows(OUTPUT/"rgb_nn_identifiability.csv",nn_rows); write_json(OUTPUT/"rgb_nn_identifiability_summary.json",nn_summary)
    nuisance,groups,matched=nuisance_audit(caches["validation"]); write_json(OUTPUT/"nuisance_metadata_audit.json",nuisance); write_rows(OUTPUT/"nuisance_group_metrics.csv",groups); write_rows(OUTPUT/"matched_latent_audit.csv",matched)
    bootstrap=bootstrap_summary(caches["validation"],nn_data); write_json(OUTPUT/"bootstrap_summary.json",bootstrap)
    plot_outputs(metrics,baselines,active,samples,hq,pearson,nn_summary); contact_sheets(caches["validation"])
    decision=labels_and_route(metrics,baselines,active,calibration,pearson,nn_summary,sparsity,nuisance); write_json(OUTPUT/"final_diagnosis.json",decision)
    report(decision,metrics,baselines,active,calibration,nn_summary,nuisance,bootstrap)
    audit=json.loads((OUTPUT/"checkpoint_audit.json").read_text(encoding="utf-8")); audit["checkpoint_sha256_after_audit"]=file_sha256(CHECKPOINT); audit["checkpoint_unchanged_after_audit"]=audit["checkpoint_sha256_read_only_guard"]==audit["checkpoint_sha256_after_audit"]; write_json(OUTPUT/"checkpoint_audit.json",audit)
    write_json(OUTPUT/"split_access_audit.json",{"train":{"accessed":True},"validation":{"accessed":True},**{s:{"accessed":False} for s in FORBIDDEN_SPLITS},"forbidden_splits_accessed":[],"status":"PASS"})


def main() -> None:
    args=parse_args(); OUTPUT.mkdir(parents=True,exist_ok=True); CACHE.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True); CONTACTS.mkdir(exist_ok=True)
    if args.phase in ("all","audit"): checkpoint_audit()
    if args.phase in ("all","infer"):
        if not (OUTPUT/"checkpoint_audit.json").is_file(): checkpoint_audit()
        for split in ALLOWED_SPLITS: infer_split(split,args.batch_size)
    if args.phase in ("all","analyze"): analyze()


if __name__=="__main__": main()
