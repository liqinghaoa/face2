from __future__ import annotations

import json
import math
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import torch
import torchvision
import yaml
from PIL import Image
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)

from datasets.control_patient_binary_dataset import ControlPatientFaceDataset
from datasets.nyha_3class_face_dataset import build_transforms
from models.nyha_backbone_factory import build_nyha_classification_model
from scripts.train.train_e0b_global_resnet18_control_patient_binary_5fold import evaluate, loader
from utils.experiment_utils import load_yaml
from utils.resnet18_anti_overfit import configure_trainability, normalize_strategy

from .checkpoint_resolver import sha256_file
from .color_transform import apply_exposure_ev_uint8, apply_gamma_uint8
from .config import Stage3A0Config, ensure_dirs


FACE2_PYTHON = Path(r"E:\resarch\Anaconda3\envs\face2\python.exe")
CONDITIONS = [
    {"condition": "original", "transform_type": "original", "transform_level": "original", "delta_ev": 0.0, "gamma": np.nan},
    {"condition": "ev_m1", "transform_type": "exposure", "transform_level": "-1.0", "delta_ev": -1.0, "gamma": np.nan},
    {"condition": "ev_m05", "transform_type": "exposure", "transform_level": "-0.5", "delta_ev": -0.5, "gamma": np.nan},
    {"condition": "ev_p05", "transform_type": "exposure", "transform_level": "+0.5", "delta_ev": 0.5, "gamma": np.nan},
    {"condition": "ev_p1", "transform_type": "exposure", "transform_level": "+1.0", "delta_ev": 1.0, "gamma": np.nan},
    {"condition": "gamma_08", "transform_type": "gamma", "transform_level": "0.8", "delta_ev": np.nan, "gamma": 0.8},
    {"condition": "gamma_12", "transform_type": "gamma", "transform_level": "1.2", "delta_ev": np.nan, "gamma": 1.2},
]
EV_CONDITION_BY_LEVEL = {-1.0: "ev_m1", -0.5: "ev_m05", 0.0: "original", 0.5: "ev_p05", 1.0: "ev_p1"}


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_logit(prob: float) -> float:
    p = min(max(float(prob), 1e-6), 1.0 - 1e-6)
    return float(math.log(p / (1.0 - p)))


def _runtime_text() -> tuple[str, dict[str, Any]]:
    payload = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "tf32_matmul": getattr(torch.backends.cuda.matmul, "allow_tf32", None),
        "tf32_cudnn": getattr(torch.backends.cudnn, "allow_tf32", None),
    }
    return "\n".join(f"{key}: {value}" for key, value in payload.items()) + "\n", payload


def _enforce_face2() -> None:
    actual = Path(sys.executable).resolve()
    expected = FACE2_PYTHON.resolve()
    if actual != expected:
        raise RuntimeError(f"Stage3-A0 resume must run with {expected}, got {actual}")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage3-A0 resume requires CUDA in the face2 environment")


def _read_resume(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "resume_from": "counterfactual_generation",
        "selected_inference_mode": "original_validation_batch16_cuda",
        "paired_baseline_source": "reproduced_original_probabilities",
        "rerun_original_baseline": True,
        "run_counterfactual_inference": True,
        "run_training": False,
        "modify_checkpoints": False,
        "classification_threshold": 0.5,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(f"Resume config {key} must be {expected!r}, got {payload.get(key)!r}")
    return payload


def _load_rgb_config(config: Stage3A0Config) -> dict[str, Any]:
    snapshot = config.rgb_experiment_root / "config_snapshot.yaml"
    fallback = config.project_root / "config/train/e0b_global_resnet18_control_patient_binary_realface_256x320_blackbg_from_raw_v1_anti_overfit_v3_bn_eval_full.yaml"
    return load_yaml(snapshot if snapshot.is_file() else fallback)


def _load_model(checkpoint_path: Path, rgb_config: dict[str, Any], device: torch.device) -> torch.nn.Module:
    model = build_nyha_classification_model(
        "resnet18",
        num_classes=2,
        pretrained=False,
        freeze_backbone=False,
        dropout=rgb_config["model"].get("dropout"),
    ).to(device)
    configure_trainability(model, normalize_strategy(rgb_config["model"].get("trainability_strategy", "full_finetune")))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model


def _run_condition_inference(
    config: Stage3A0Config,
    rgb_config: dict[str, Any],
    checkpoints: pd.DataFrame,
    image_root: Path,
    split_root: Path,
    condition: str,
) -> pd.DataFrame:
    device = torch.device("cuda")
    rows = []
    for fold in range(5):
        data, train_cfg = rgb_config["data"], rgb_config["train"]
        val_tf = build_transforms("val", data["image_size"], rgb_config["normalize"]["mean"], rgb_config["normalize"]["std"], False)
        val_csv = split_root / data["val_csv_pattern"].format(fold=fold)
        val_frame = pd.read_csv(val_csv, dtype={"ID": "string", "patient_group_id": "string"})
        if val_frame.empty:
            continue
        val_set = ControlPatientFaceDataset(val_csv, val_tf, image_root, data.get("image_filename_template", "{ID}.png"))
        val_loader = loader(val_set, int(train_cfg["batch_size"]), False, int(train_cfg["num_workers"]), int(train_cfg["random_seed"]) + 1000 + fold, bool(train_cfg["pin_memory"]))
        ckpt = Path(checkpoints.loc[checkpoints["fold"].astype(int) == fold, "checkpoint_path"].iloc[0])
        model = _load_model(ckpt, rgb_config, device)
        frame, _ = evaluate(model, val_loader, device, fold, int(torch.load(ckpt, map_location="cpu", weights_only=False)["epoch"]), ckpt)
        frame["condition"] = condition
        rows.append(frame)
    out = pd.concat(rows, ignore_index=True)
    return out.rename(columns={"prob_patient": "probability", "pred_class": "prediction", "logit_patient": "model_logit_patient"})


def _y_values(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    img = np.asarray(rgb, dtype=np.float32) / 255.0
    y = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    return y[np.asarray(mask) > 0]


def _qc_metrics(sample_id: str, condition: str, rgb: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    face = np.asarray(mask) > 0
    values = _y_values(rgb, mask)
    face_rgb = np.asarray(rgb)[face]
    outside_nonzero = int(np.count_nonzero(np.asarray(rgb)[~face]))
    any255 = np.any(face_rgb == 255, axis=1) if len(face_rgb) else np.array([])
    all255 = np.all(face_rgb == 255, axis=1) if len(face_rgb) else np.array([])
    any0 = np.any(face_rgb == 0, axis=1) if len(face_rgb) else np.array([])
    all0 = np.all(face_rgb == 0, axis=1) if len(face_rgb) else np.array([])
    any255_frac = float(any255.mean()) if len(any255) else float("nan")
    all255_frac = float(all255.mean()) if len(all255) else float("nan")
    any0_frac = float(any0.mean()) if len(any0) else float("nan")
    all0_frac = float(all0.mean()) if len(all0) else float("nan")
    high = any255_frac
    low = any0_frac
    substantial = bool(high > 0.01 or low > 0.01 or all255_frac > 0.001 or all0_frac > 0.001)
    overstress = bool(high > 0.05 or low > 0.05 or outside_nonzero > 0)
    return {
        "sample_id": sample_id,
        "condition": condition,
        "skin_y_median": float(np.median(values)),
        "skin_y_p05": float(np.quantile(values, 0.05)),
        "skin_y_p99": float(np.quantile(values, 0.99)),
        "face_high_end_clipping_fraction": high,
        "face_low_end_clipping_fraction": low,
        "any_channel_eq_255_fraction": any255_frac,
        "all_channels_eq_255_fraction": all255_frac,
        "any_channel_eq_0_fraction": any0_frac,
        "all_channels_eq_0_fraction": all0_frac,
        "mask_outside_nonzero_pixel_count": outside_nonzero,
        "image_shape": list(np.asarray(rgb).shape),
        "mild_clipping_flag": bool(high > 0.001 or low > 0.001),
        "substantial_clipping_flag": substantial,
        "overstress_flag": overstress,
    }


def _transform_image(rgb: np.ndarray, mask: np.ndarray, condition: dict[str, Any]) -> np.ndarray:
    if condition["condition"] == "original":
        return np.asarray(rgb, dtype=np.uint8).copy()
    if condition["transform_type"] == "exposure":
        return apply_exposure_ev_uint8(rgb, mask, float(condition["delta_ev"]))
    return apply_gamma_uint8(rgb, mask, float(condition["gamma"]))


def _generate_condition_images(
    sample_ids: list[str],
    source_dir: Path,
    mask_dir: Path,
    target_root: Path,
    condition: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, bool]]:
    root = target_root / condition["condition"]
    root.mkdir(parents=True, exist_ok=True)
    qc_rows = []
    smoke_checks = {"zero_ev_noop": True, "mask_outside_zero": True, "shape_unchanged": True, "deterministic": True}
    for sample_id in sample_ids:
        with Image.open(source_dir / f"{sample_id}.png") as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        with Image.open(mask_dir / f"{sample_id}.png") as image:
            mask = np.asarray(image.convert("L")) > 0
        if condition["condition"] == "original":
            out = rgb.copy()
        else:
            out = _transform_image(rgb, mask, condition)
            out2 = _transform_image(rgb, mask, condition)
            smoke_checks["deterministic"] = smoke_checks["deterministic"] and np.array_equal(out, out2)
        if not np.array_equal(apply_exposure_ev_uint8(rgb, mask, 0.0), rgb):
            smoke_checks["zero_ev_noop"] = False
        smoke_checks["shape_unchanged"] = smoke_checks["shape_unchanged"] and out.shape == rgb.shape
        smoke_checks["mask_outside_zero"] = smoke_checks["mask_outside_zero"] and int(np.count_nonzero(out[~mask])) == 0
        Image.fromarray(out).save(root / f"{sample_id}.png")
        qc_rows.append(_qc_metrics(sample_id, condition["condition"], out, mask))
    return pd.DataFrame(qc_rows), smoke_checks


def _make_split_subset(rgb_config: dict[str, Any], source_split_root: Path, target_split_root: Path, sample_ids: set[str]) -> None:
    target_split_root.mkdir(parents=True, exist_ok=True)
    data = rgb_config["data"]
    for fold in range(5):
        frame = pd.read_csv(source_split_root / data["val_csv_pattern"].format(fold=fold), dtype={"ID": "string", "patient_group_id": "string"})
        subset = frame.loc[frame["ID"].astype(str).isin(sample_ids)].copy()
        subset.to_csv(target_split_root / data["val_csv_pattern"].format(fold=fold), index=False, encoding="utf-8-sig")


def _select_smoke_samples(meta: pd.DataFrame, stage2: pd.DataFrame, baseline: pd.DataFrame) -> list[str]:
    data = meta.merge(stage2.loc[:, ["sample_id", "skin_y_median"]], on="sample_id", how="left")
    data = data.merge(baseline.loc[:, ["sample_id", "reproduced_probability", "reproduced_prediction"]], on="sample_id", how="left")
    data["confidence_distance"] = (data["reproduced_probability"] - 0.5).abs()
    picks: list[str] = []

    def add(frame: pd.DataFrame, n: int = 1) -> None:
        for sid in frame["sample_id"].astype(str).tolist():
            if sid not in picks:
                picks.append(sid)
            if len(picks) >= 16:
                return

    for fold in range(5):
        add(data.loc[data["fold"].astype(int).eq(fold)].sort_values("confidence_distance"), 1)
    add(data.loc[(data["camera_model"].eq("M2006J10C")) & (data["binary_label"].eq(0))].sort_values("confidence_distance"), 2)
    add(data.loc[(data["camera_model"].eq("M2006J10C")) & (data["binary_label"].eq(1))].sort_values("confidence_distance"), 2)
    add(data.loc[(data["camera_model"].eq("BVL-AN00")) & (data["binary_label"].eq(1))].sort_values("confidence_distance"), 2)
    add(data.sort_values("skin_y_median"), 2)
    add(data.sort_values("skin_y_median", ascending=False), 2)
    add(data.sort_values("confidence_distance"), 2)
    add(data.loc[data["binary_label"].astype(int).eq(data["reproduced_prediction"].astype(int))].sort_values("confidence_distance"), 2)
    add(data.loc[~data["binary_label"].astype(int).eq(data["reproduced_prediction"].astype(int))].sort_values("confidence_distance"), 2)
    return picks[:16]


def _predictions_to_long(predictions: dict[str, pd.DataFrame], meta: pd.DataFrame, qc: pd.DataFrame) -> pd.DataFrame:
    base = predictions["original"].loc[:, ["sample_id", "probability", "prediction"]].rename(columns={"probability": "original_probability", "prediction": "original_prediction"})
    rows = []
    cond_map = {c["condition"]: c for c in CONDITIONS}
    qc_index = qc.set_index(["sample_id", "condition"])
    for cond, frame in predictions.items():
        merged = frame.merge(base, on="sample_id", validate="one_to_one")
        merged = merged.merge(meta.loc[:, ["sample_id", "camera_model"]], on="sample_id", validate="one_to_one")
        c = cond_map[cond]
        for row in merged.itertuples(index=False):
            orig_p = float(row.original_probability)
            cf_p = float(row.probability)
            orig_logit = _safe_logit(orig_p)
            cf_logit = _safe_logit(cf_p)
            y = int(row.binary_label)
            orig_pred = int(row.original_prediction)
            cf_pred = int(row.prediction)
            q = qc_index.loc[(str(row.sample_id), cond)].to_dict()
            original_q = qc_index.loc[(str(row.sample_id), "original")].to_dict()
            rows.append(
                {
                    "sample_id": str(row.sample_id),
                    "patient_group_id": str(row.patient_group_id),
                    "fold": int(row.fold),
                    "binary_label": y,
                    "camera_model": row.camera_model,
                    "condition": cond,
                    "transform_type": c["transform_type"],
                    "transform_level": c["transform_level"],
                    "delta_ev": c["delta_ev"],
                    "gamma": c["gamma"],
                    "original_probability": orig_p,
                    "counterfactual_probability": cf_p,
                    "delta_probability": cf_p - orig_p,
                    "absolute_delta_probability": abs(cf_p - orig_p),
                    "original_logit": orig_logit,
                    "counterfactual_logit": cf_logit,
                    "delta_logit": cf_logit - orig_logit,
                    "original_prediction": orig_pred,
                    "counterfactual_prediction": cf_pred,
                    "prediction_flip": bool(orig_pred != cf_pred),
                    "original_correct": bool(orig_pred == y),
                    "counterfactual_correct": bool(cf_pred == y),
                    "correct_to_wrong": bool(orig_pred == y and cf_pred != y),
                    "wrong_to_correct": bool(orig_pred != y and cf_pred == y),
                    "original_true_margin": (2 * y - 1) * orig_logit,
                    "counterfactual_true_margin": (2 * y - 1) * cf_logit,
                    "delta_true_margin": ((2 * y - 1) * cf_logit) - ((2 * y - 1) * orig_logit),
                    "original_skin_y_median": original_q["skin_y_median"],
                    "counterfactual_skin_y_median": q["skin_y_median"],
                    "delta_skin_y_median": q["skin_y_median"] - original_q["skin_y_median"],
                    **{k: v for k, v in q.items() if k not in {"sample_id", "condition", "skin_y_median"}},
                }
            )
    return pd.DataFrame(rows)


def _classification_metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    pred = (probs >= 0.5).astype(int)
    if len(np.unique(labels)) < 2:
        auc = np.nan
    else:
        auc = float(roc_auc_score(labels, probs))
    cm = confusion_matrix(labels, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "auc": auc,
        "accuracy": float(accuracy_score(labels, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)) if len(np.unique(labels)) == 2 else np.nan,
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "brier_score": float(brier_score_loss(labels, probs)),
    }


def _performance_tables(long: pd.DataFrame, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (scope, group), subset in _scopes(long):
        for condition, cdf in subset.groupby("condition"):
            metrics = _classification_metrics(cdf["binary_label"].to_numpy(int), cdf["counterfactual_probability"].to_numpy(float))
            rows.append({"scope": scope, "group": group, "condition": condition, "n": len(cdf), **metrics})
    perf = pd.DataFrame(rows)
    diffs = []
    for (scope, group), sub in perf.groupby(["scope", "group"], dropna=False):
        original = sub.loc[sub["condition"].eq("original")]
        if original.empty:
            continue
        base = original.iloc[0]
        for row in sub.itertuples(index=False):
            item = {"scope": scope, "group": group, "condition": row.condition}
            for metric in ["auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "macro_f1", "brier_score"]:
                item[f"delta_{metric}"] = getattr(row, metric) - base[metric] if pd.notna(getattr(row, metric)) and pd.notna(base[metric]) else np.nan
            diffs.append(item)
    diff = pd.DataFrame(diffs)
    perf.to_csv(output / "performance_by_transform.csv", index=False, encoding="utf-8-sig")
    diff.to_csv(output / "performance_differences_vs_original.csv", index=False, encoding="utf-8-sig")
    return perf, diff


def _scopes(long: pd.DataFrame):
    yield ("all", "all500"), long
    yield ("camera", "Xiaomi_all",), long.loc[long["camera_model"].eq("M2006J10C")]
    yield ("camera_label", "Xiaomi_Control"), long.loc[(long["camera_model"].eq("M2006J10C")) & (long["binary_label"].eq(0))]
    yield ("camera_label", "Xiaomi_Patient"), long.loc[(long["camera_model"].eq("M2006J10C")) & (long["binary_label"].eq(1))]
    yield ("camera_label", "HONOR_Patient"), long.loc[(long["camera_model"].eq("BVL-AN00")) & (long["binary_label"].eq(1))]


def _cluster_bootstrap(values: pd.DataFrame, value_col: str, group_col: str = "patient_group_id", iterations: int = 2000, seed: int = 2026) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    groups = values[group_col].astype(str).unique()
    if len(values) == 0:
        return {"mean": np.nan, "median": np.nan, "ci_low": np.nan, "ci_high": np.nan, "positive_proportion": np.nan, "valid_bootstrap": 0, "invalid_bootstrap": iterations}
    observed = values[value_col].to_numpy(float)
    boot = []
    invalid = 0
    grouped = {g: values.loc[values[group_col].astype(str).eq(g), value_col].to_numpy(float) for g in groups}
    for _ in range(iterations):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        arr = np.concatenate([grouped[g] for g in sampled])
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            invalid += 1
            continue
        boot.append(float(np.mean(arr)))
    return {
        "mean": float(np.mean(observed)),
        "median": float(np.median(observed)),
        "ci_low": float(np.quantile(boot, 0.025)) if boot else np.nan,
        "ci_high": float(np.quantile(boot, 0.975)) if boot else np.nan,
        "positive_proportion": float(np.mean(observed > 0)),
        "valid_bootstrap": len(boot),
        "invalid_bootstrap": invalid,
    }


def _sample_effects(long: pd.DataFrame) -> pd.DataFrame:
    wide = long.pivot(index="sample_id", columns="condition", values="counterfactual_logit")
    probs = long.pivot(index="sample_id", columns="condition", values="counterfactual_probability")
    meta = long.drop_duplicates("sample_id").set_index("sample_id")
    rows = []
    for sid, row in wide.iterrows():
        ev = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        logits = np.array([row[EV_CONDITION_BY_LEVEL[v]] for v in ev], dtype=float)
        slope, intercept = np.polyfit(ev, logits, 1)
        fitted = intercept + slope * ev
        ss_res = float(np.sum((logits - fitted) ** 2))
        ss_tot = float(np.sum((logits - logits.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        rows.append(
            {
                "sample_id": sid,
                "patient_group_id": meta.loc[sid, "patient_group_id"],
                "fold": int(meta.loc[sid, "fold"]),
                "binary_label": int(meta.loc[sid, "binary_label"]),
                "camera_model": meta.loc[sid, "camera_model"],
                "D_EV_0.5": float(row["ev_p05"] - row["ev_m05"]),
                "D_EV_1.0": float(row["ev_p1"] - row["ev_m1"]),
                "D_gamma": float(row["gamma_08"] - row["gamma_12"]),
                "exposure_slope": float(slope),
                "slope_r_squared": float(r2),
                "slope_positive": bool(slope > 0),
                "slope_negative": bool(slope < 0),
                "exposure_monotonic_increasing": bool(np.all(np.diff(logits) >= -1e-8)),
                "exposure_monotonic_decreasing": bool(np.all(np.diff(logits) <= 1e-8)),
                "ev_logit_spearman": float(spearmanr(ev, logits).statistic),
                "original_probability": float(probs.loc[sid, "original"]),
            }
        )
    return pd.DataFrame(rows)


def _effects_by_scope(effects: pd.DataFrame, iterations: int, seed: int) -> pd.DataFrame:
    rows = []
    scopes = [
        ("all500", effects),
        ("Xiaomi_all", effects.loc[effects["camera_model"].eq("M2006J10C")]),
        ("Xiaomi_Control", effects.loc[(effects["camera_model"].eq("M2006J10C")) & (effects["binary_label"].eq(0))]),
        ("Xiaomi_Patient", effects.loc[(effects["camera_model"].eq("M2006J10C")) & (effects["binary_label"].eq(1))]),
        ("HONOR_Patient", effects.loc[(effects["camera_model"].eq("BVL-AN00")) & (effects["binary_label"].eq(1))]),
    ]
    for scope, sub in scopes:
        for metric in ["D_EV_0.5", "D_EV_1.0", "D_gamma", "exposure_slope"]:
            stats = _cluster_bootstrap(sub.loc[:, ["patient_group_id", metric]].rename(columns={metric: "value"}), "value", iterations=iterations, seed=seed)
            rows.append({"scope": scope, "metric": metric, "n": len(sub), **stats})
    return pd.DataFrame(rows)


def _flip_rates(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, cond, filt in [
        ("Xiaomi_Patient_darkening_false_negative", "ev_m05", (long["camera_model"].eq("M2006J10C")) & (long["binary_label"].eq(1))),
        ("Xiaomi_Control_brightening_false_positive", "ev_p05", (long["camera_model"].eq("M2006J10C")) & (long["binary_label"].eq(0))),
        ("HONOR_Patient_darkening_false_negative", "ev_m05", (long["camera_model"].eq("BVL-AN00")) & (long["binary_label"].eq(1))),
    ]:
        sub = long.loc[filt & long["condition"].eq(cond)]
        denom = sub.loc[sub["original_correct"]]
        correct_to_wrong = float(denom["correct_to_wrong"].mean()) if len(denom) else np.nan
        wrong = sub.loc[~sub["original_correct"]]
        wrong_to_correct = float(wrong["wrong_to_correct"].mean()) if len(wrong) else np.nan
        rows.append({"endpoint": name, "condition": cond, "n": len(sub), "denominator_original_correct": len(denom), "correct_to_wrong_rate": correct_to_wrong, "wrong_to_correct_rate": wrong_to_correct, "net_harmful_flip_rate": correct_to_wrong - (0 if np.isnan(wrong_to_correct) else wrong_to_correct)})
    return pd.DataFrame(rows)


def _write_basic_figures(long: pd.DataFrame, effects: pd.DataFrame, figures: Path, perf: pd.DataFrame) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    ev_long = long.loc[long["condition"].isin(["ev_m1", "ev_m05", "original", "ev_p05", "ev_p1"])].copy()
    ev_long["ev"] = ev_long["condition"].map({"ev_m1": -1.0, "ev_m05": -0.5, "original": 0.0, "ev_p05": 0.5, "ev_p1": 1.0})
    summary = ev_long.groupby("ev")["counterfactual_probability"].agg(["mean", "median"]).reset_index()
    plt.figure(figsize=(6, 4)); plt.plot(summary["ev"], summary["mean"], marker="o"); plt.xlabel("EV transform"); plt.ylabel("Mean patient probability"); plt.tight_layout(); plt.savefig(figures / "probability_by_exposure_level.png", dpi=180); plt.close()
    summary = ev_long.groupby("ev")["counterfactual_logit"].agg(["mean", "median"]).reset_index()
    plt.figure(figsize=(6, 4)); plt.plot(summary["ev"], summary["mean"], marker="o"); plt.xlabel("EV transform"); plt.ylabel("Mean logit(probability)"); plt.tight_layout(); plt.savefig(figures / "logit_by_exposure_level.png", dpi=180); plt.close()
    plt.figure(figsize=(6, 4)); plt.hist(effects["exposure_slope"], bins=40); plt.xlabel("Exposure slope"); plt.ylabel("N"); plt.tight_layout(); plt.savefig(figures / "exposure_slope_distribution.png", dpi=180); plt.close()
    tmp = effects.assign(group=effects["camera_model"] + "_" + effects["binary_label"].map({0: "Control", 1: "Patient"}))
    grp = tmp.groupby("group")["D_EV_0.5"].mean().reset_index()
    plt.figure(figsize=(8, 4)); plt.bar(grp["group"], grp["D_EV_0.5"]); plt.xticks(rotation=30, ha="right"); plt.ylabel("Mean D_EV_0.5"); plt.tight_layout(); plt.savefig(figures / "d_ev_05_by_camera_and_label.png", dpi=180); plt.close()
    gamma = long.loc[long["condition"].isin(["gamma_08", "gamma_12"])].groupby("condition")["counterfactual_logit"].mean().reset_index()
    plt.figure(figsize=(5, 4)); plt.bar(gamma["condition"], gamma["counterfactual_logit"]); plt.ylabel("Mean logit(probability)"); plt.tight_layout(); plt.savefig(figures / "gamma_effects.png", dpi=180); plt.close()
    xperf = perf.loc[perf["group"].eq("Xiaomi_all")]
    plt.figure(figsize=(7, 4)); plt.bar(xperf["condition"], xperf["balanced_accuracy"]); plt.xticks(rotation=45, ha="right"); plt.ylabel("Balanced accuracy"); plt.tight_layout(); plt.savefig(figures / "performance_by_transform_xiaomi.png", dpi=180); plt.close()


def _write_smoke_qc_panels(sample_ids: list[str], image_root: Path, panel_dir: Path) -> None:
    panel_dir.mkdir(parents=True, exist_ok=True)
    for sample_id in sample_ids:
        fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(14, 3))
        for ax, condition in zip(axes, CONDITIONS):
            path = image_root / condition["condition"] / f"{sample_id}.png"
            with Image.open(path) as image:
                ax.imshow(image.convert("RGB"))
            ax.set_title(condition["condition"], fontsize=8)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(panel_dir / f"{sample_id}_stage3_a0_smoke.png", dpi=140)
        plt.close(fig)


def _evidence_classification(primary: pd.DataFrame, flips: pd.DataFrame, effects: pd.DataFrame, sensitivity: pd.DataFrame) -> dict[str, Any]:
    all_d = primary.loc[(primary["scope"].eq("all500")) & (primary["metric"].eq("D_EV_0.5"))].iloc[0]
    x_d = primary.loc[(primary["scope"].eq("Xiaomi_all")) & (primary["metric"].eq("D_EV_0.5"))].iloc[0]
    slope_positive = float(effects["slope_positive"].mean())
    monotonic = float(effects["exposure_monotonic_increasing"].mean())
    x_patient_flip = flips.loc[flips["endpoint"].eq("Xiaomi_Patient_darkening_false_negative"), "correct_to_wrong_rate"].iloc[0]
    x_control_flip = flips.loc[flips["endpoint"].eq("Xiaomi_Control_brightening_false_positive"), "correct_to_wrong_rate"].iloc[0]
    x_ci_positive = pd.notna(x_d["ci_low"]) and x_d["ci_low"] > 0
    all_ci_positive = pd.notna(all_d["ci_low"]) and all_d["ci_low"] > 0
    if x_ci_positive and all_ci_positive and slope_positive > 0.6 and monotonic > 0.5:
        lum = "strong" if max(x_patient_flip if pd.notna(x_patient_flip) else 0, x_control_flip if pd.notna(x_control_flip) else 0) > 0.02 else "moderate"
        within = "strong" if x_d["positive_proportion"] > 0.7 else "moderate"
        shortcut = "supported" if lum == "moderate" else "strongly_supported"
    elif x_d["mean"] > 0 and slope_positive > 0.55:
        lum, within, shortcut = "weak", "weak", "partially_supported"
    else:
        lum, within, shortcut = "none", "none", "not_supported"
    return {
        "counterfactual_luminance_sensitivity_level": lum,
        "within_camera_evidence_level": within,
        "brightness_shortcut_support": shortcut,
        "basis": {
            "all_D_EV_0.5_mean": float(all_d["mean"]),
            "all_D_EV_0.5_ci": [float(all_d["ci_low"]), float(all_d["ci_high"])],
            "xiaomi_D_EV_0.5_mean": float(x_d["mean"]),
            "xiaomi_D_EV_0.5_ci": [float(x_d["ci_low"]), float(x_d["ci_high"])],
            "slope_positive_proportion": slope_positive,
            "monotonic_increasing_proportion": monotonic,
            "xiaomi_patient_darkening_correct_to_wrong": None if pd.isna(x_patient_flip) else float(x_patient_flip),
            "xiaomi_control_brightening_correct_to_wrong": None if pd.isna(x_control_flip) else float(x_control_flip),
        },
    }


def run_stage3_a0_resume(base_config_path: Path, resume_config_path: Path) -> dict[str, Any]:
    _enforce_face2()
    config = Stage3A0Config.from_yaml(base_config_path)
    resume_payload = _read_resume(resume_config_path)
    dirs = ensure_dirs(config.output_dir)
    smoke_dir = config.output_dir / "smoke"
    smoke_dirs = {name: smoke_dir / name for name in ["qc_panels", "tmp_images", "splits"]}
    for path in smoke_dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    for extra in ["smoke", "tmp_counterfactual_images"]:
        (config.output_dir / extra).mkdir(parents=True, exist_ok=True)
    runtime_text, runtime_payload = _runtime_text()
    (dirs["logs"] / "resume_runtime_environment.txt").write_text(runtime_text, encoding="utf-8")

    rgb_config = _load_rgb_config(config)
    checkpoints = pd.read_csv(config.output_dir / "preflight/checkpoint_inventory.csv")
    r0_root = config.project_root / "experiments/lighting_confounding/Stage3_A0_R0_OOF_Reproduction_Discrepancy_Audit_v1"
    r0_gate = json.loads((r0_root / "decision/stage3_a0_resume_gate.json").read_text(encoding="utf-8"))
    r0_ckpt = pd.read_csv(r0_root / "preflight/r0_checkpoint_hash_reaudit.csv")
    r0_images = pd.read_csv(r0_root / "preflight/r0_image_hash_reaudit.csv", dtype={"sample_id": str})
    current_ckpt_ok = all(sha256_file(Path(p)) == h for p, h in zip(r0_ckpt["checkpoint_path"], r0_ckpt["current_sha256"]))
    current_image = pd.DataFrame({"sample_id": r0_images["sample_id"], "current_sha256": [sha256_file(config.input_image_dir / f"{sid}.png") for sid in r0_images["sample_id"].astype(str)]})
    current_image_ok = bool((current_image["current_sha256"].values == r0_images["sha256"].values).all())
    reproduction_summary = json.loads((config.output_dir / "reproduction/original_oof_reproduction_summary.json").read_text(encoding="utf-8"))
    integrity = {
        "r0_status_strict_pass": r0_gate["reproduction_status"] == "strict_pass",
        "resume_allowed": bool(r0_gate["resume_allowed"]),
        "selected_mode_original_validation_batch16_cuda": r0_gate["selected_inference_mode"]["selected_reproduction_mode"] == "original_validation_batch16_cuda",
        "checkpoint_hash_unchanged_vs_r0": current_ckpt_ok,
        "input_hash_unchanged_vs_r0": current_image_ok,
        "sample_count": int(len(r0_images)),
        "oof_reproduction_pass": bool(reproduction_summary["pass"]),
        "oof_max_diff": float(reproduction_summary["max_probability_abs_diff"]),
        "prediction_match_count": int(reproduction_summary["prediction_match_count"]),
        "threshold_crossing_count": int(r0_gate["observed_discrepancy_statistics"]["threshold_crossing_count"]),
        "pass": False,
    }
    integrity["pass"] = all(
        v
        for k, v in integrity.items()
        if k not in {"sample_count", "oof_max_diff", "prediction_match_count", "threshold_crossing_count", "pass"}
    ) and integrity["sample_count"] == 500 and integrity["oof_max_diff"] <= 1e-5 and integrity["prediction_match_count"] == 500 and integrity["threshold_crossing_count"] == 0
    _write_json(dirs["preflight"] / "resume_integrity_check.json", integrity)
    (dirs["preflight"] / "resume_integrity_report.md").write_text(f"# Resume Integrity\n\nPASS: {integrity['pass']}\n\nSelected mode: original_validation_batch16_cuda\n", encoding="utf-8")
    if not integrity["pass"]:
        raise RuntimeError(f"Resume integrity failed: {integrity}")

    baseline = pd.read_csv(config.output_dir / "reproduction/original_oof_reproduced.csv", dtype={"sample_id": str, "patient_group_id": str})
    baseline = baseline.rename(columns={"reproduced_probability": "probability", "reproduced_prediction": "prediction", "reproduced_logit_patient": "model_logit_patient"})
    stage1 = pd.read_csv(config.stage1_master_csv, dtype={"sample_id": str, "patient_group_id": str})
    meta = stage1.loc[:, ["sample_id", "patient_group_id", "fold", "binary_label", "camera_model"]].copy()
    stage2 = pd.read_csv(config.stage2_feature_csv, dtype={"sample_id": str, "patient_group_id": str})
    smoke_ids = _select_smoke_samples(meta, stage2, baseline.rename(columns={"probability": "reproduced_probability", "prediction": "reproduced_prediction"}))
    _make_split_subset(rgb_config, config.project_root / rgb_config["data"]["split_dir"], smoke_dirs["splits"], set(smoke_ids))
    smoke_predictions = []
    smoke_qc = []
    smoke_checks: dict[str, Any] = {"sample_count": len(smoke_ids)}
    for condition in CONDITIONS:
        qc_df, checks = _generate_condition_images(smoke_ids, config.input_image_dir, config.face_valid_mask_dir, smoke_dirs["tmp_images"], condition)
        smoke_qc.append(qc_df)
        for k, v in checks.items():
            smoke_checks[k] = smoke_checks.get(k, True) and bool(v)
        pred = _run_condition_inference(config, rgb_config, checkpoints, smoke_dirs["tmp_images"] / condition["condition"], smoke_dirs["splits"], condition["condition"])
        smoke_predictions.append(pred)
    smoke_pred = pd.concat(smoke_predictions, ignore_index=True)
    smoke_qc_frame = pd.concat(smoke_qc, ignore_index=True)
    smoke_pred.to_csv(smoke_dir / "stage3_a0_resume_smoke_predictions.csv", index=False, encoding="utf-8-sig")
    smoke_qc_frame.to_csv(smoke_dir / "stage3_a0_resume_smoke_qc.csv", index=False, encoding="utf-8-sig")
    smoke_original = smoke_pred.loc[smoke_pred["condition"].eq("original")].merge(baseline.loc[:, ["sample_id", "probability"]], on="sample_id", suffixes=("_smoke", "_baseline"))
    smoke_checks["original_probability_matches_baseline"] = bool(np.max(np.abs(smoke_original["probability_smoke"] - smoke_original["probability_baseline"])) <= 1e-5)
    smoke_checks["finite_predictions"] = bool(np.isfinite(smoke_pred["probability"]).all())
    _write_smoke_qc_panels(smoke_ids, smoke_dirs["tmp_images"], smoke_dirs["qc_panels"])
    smoke_checks["qc_panel_count"] = len(list(smoke_dirs["qc_panels"].glob("*_stage3_a0_smoke.png")))
    smoke_checks["pass"] = bool(all(v for v in smoke_checks.values() if isinstance(v, bool)))
    (smoke_dir / "stage3_a0_resume_smoke_report.md").write_text(f"# Stage3-A0 Resume Smoke\n\nPASS: {smoke_checks['pass']}\n\nSamples: {len(smoke_ids)}\n", encoding="utf-8")
    if not smoke_checks["pass"]:
        raise RuntimeError(f"Smoke failed: {smoke_checks}")

    temp_root = config.output_dir / "tmp_counterfactual_images"
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True)
    sample_ids = meta["sample_id"].astype(str).tolist()
    predictions: dict[str, pd.DataFrame] = {}
    qc_frames = []
    for condition in CONDITIONS:
        if condition["condition"] == "original":
            predictions["original"] = _run_condition_inference(config, rgb_config, checkpoints, config.input_image_dir, config.project_root / rgb_config["data"]["split_dir"], "original")
            qc_df, _ = _generate_condition_images(sample_ids, config.input_image_dir, config.face_valid_mask_dir, temp_root, condition)
        else:
            qc_df, _ = _generate_condition_images(sample_ids, config.input_image_dir, config.face_valid_mask_dir, temp_root, condition)
            predictions[condition["condition"]] = _run_condition_inference(config, rgb_config, checkpoints, temp_root / condition["condition"], config.project_root / rgb_config["data"]["split_dir"], condition["condition"])
        qc_frames.append(qc_df)

    qc = pd.concat(qc_frames, ignore_index=True)
    long = _predictions_to_long(predictions, meta, qc)
    if len(long) != 3500 or long.groupby("sample_id")["condition"].nunique().min() != 7:
        raise RuntimeError("Formal inference did not produce strict 3500 rows / 7 conditions per sample")
    long.to_csv(dirs["predictions"] / "counterfactual_predictions_long.csv", index=False, encoding="utf-8-sig")
    wide = long.pivot(index="sample_id", columns="condition", values=["counterfactual_probability", "counterfactual_logit", "counterfactual_prediction", "prediction_flip"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide.reset_index().to_csv(dirs["predictions"] / "counterfactual_predictions_wide.csv", index=False, encoding="utf-8-sig")
    qc.to_csv(dirs["qc"] / "counterfactual_qc_metrics.csv", index=False, encoding="utf-8-sig")
    _write_json(dirs["qc"] / "counterfactual_qc_summary.json", {"rows": len(qc), "substantial_clipping_rows": int(qc["substantial_clipping_flag"].sum()), "overstress_rows": int(qc["overstress_flag"].sum())})
    qc.loc[qc["mild_clipping_flag"] | qc["substantial_clipping_flag"] | qc["overstress_flag"]].to_csv(dirs["qc"] / "clipping_diagnostics.csv", index=False, encoding="utf-8-sig")

    effects = _sample_effects(long)
    effects.to_csv(dirs["paired"] / "per_sample_exposure_sensitivity.csv", index=False, encoding="utf-8-sig")
    primary = _effects_by_scope(effects, config.bootstrap_iterations, config.random_seed)
    primary.to_csv(dirs["bootstrap"] / "primary_endpoint_bootstrap.csv", index=False, encoding="utf-8-sig")
    primary.loc[primary["metric"].eq("D_EV_0.5")].to_csv(dirs["metrics"] / "paired_logit_effects.csv", index=False, encoding="utf-8-sig")
    prob_eff = long.loc[~long["condition"].eq("original")].groupby("condition").agg(mean_delta_probability=("delta_probability", "mean"), median_delta_probability=("delta_probability", "median"), flip_rate=("prediction_flip", "mean")).reset_index()
    prob_eff.to_csv(dirs["metrics"] / "paired_probability_effects.csv", index=False, encoding="utf-8-sig")
    perf, perf_diff = _performance_tables(long, dirs["metrics"])
    flips = _flip_rates(long)
    flips.to_csv(dirs["metrics"] / "flip_rate_summary.csv", index=False, encoding="utf-8-sig")
    perf_diff.to_csv(dirs["bootstrap"] / "performance_difference_bootstrap.csv", index=False, encoding="utf-8-sig")
    _write_json(dirs["bootstrap"] / "bootstrap_diagnostics.json", {"iterations": config.bootstrap_iterations, "seed": config.random_seed, "primary_valid_min": int(primary["valid_bootstrap"].min()), "primary_invalid_max": int(primary["invalid_bootstrap"].max())})

    cam_effects = primary.loc[primary["scope"].isin(["Xiaomi_all", "Xiaomi_Control", "Xiaomi_Patient", "HONOR_Patient"])].copy()
    cam_effects.to_csv(dirs["subgroups"] / "camera_stratified_paired_effects.csv", index=False, encoding="utf-8-sig")
    perf.loc[perf["scope"].isin(["camera", "camera_label"])].to_csv(dirs["subgroups"] / "camera_stratified_performance.csv", index=False, encoding="utf-8-sig")
    flips.to_csv(dirs["subgroups"] / "camera_stratified_flip_rates.csv", index=False, encoding="utf-8-sig")
    assignments = pd.read_csv(config.stage2_r2_root / "repair/erosion_foldwise_assignments_repaired.csv", dtype={"sample_id": str})
    brightness = assignments.loc[(assignments["erosion_px"].eq(2)) & (assignments["metric_name"].eq("skin_y_median")), ["sample_id", "assigned_group"]]
    bright_eff = effects.merge(brightness, on="sample_id", how="left").groupby("assigned_group").agg(n=("sample_id", "count"), mean_D_EV_05=("D_EV_0.5", "mean"), mean_exposure_slope=("exposure_slope", "mean"), positive_slope_rate=("slope_positive", "mean")).reset_index()
    bright_eff.to_csv(dirs["subgroups"] / "stage2_brightness_stratified_effects.csv", index=False, encoding="utf-8-sig")
    conf = effects.assign(confidence_group=pd.qcut((effects["original_probability"] - 0.5).abs(), 3, labels=["low", "middle", "high"]))
    conf.groupby("confidence_group", observed=False).agg(n=("sample_id", "count"), mean_D_EV_05=("D_EV_0.5", "mean"), mean_slope=("exposure_slope", "mean")).reset_index().to_csv(dirs["subgroups"] / "baseline_confidence_stratified_effects.csv", index=False, encoding="utf-8-sig")
    low_clip_ids = qc.loc[~qc["substantial_clipping_flag"] & ~qc["overstress_flag"], "sample_id"].drop_duplicates()
    sens_effects = effects.loc[effects["sample_id"].isin(low_clip_ids)]
    sens_primary = _effects_by_scope(sens_effects, config.bootstrap_iterations, config.random_seed)
    sens_effects.to_csv(dirs["sensitivity"] / "clipping_filtered_paired_effects.csv", index=False, encoding="utf-8-sig")
    sens_primary.to_csv(dirs["sensitivity"] / "clipping_filtered_primary_endpoints.csv", index=False, encoding="utf-8-sig")

    _write_basic_figures(long, effects, dirs["figures"], perf)
    evidence = _evidence_classification(primary, flips, effects, sens_primary)
    _write_json(dirs["reports"] / "stage3_a0_evidence_classification.json", evidence)
    decision_text = "# Stage3-A0 to A1 Decision\n\n"
    if evidence["brightness_shortcut_support"] in {"supported", "strongly_supported"} and evidence["within_camera_evidence_level"] in {"moderate", "strong"}:
        decision_text += "Recommendation: proceed to Stage3-A1 Exposure/Gamma robustness training design. Do not start Stage3-A1 in this run.\n"
    elif evidence["counterfactual_luminance_sensitivity_level"] in {"weak", "moderate", "strong"}:
        decision_text += "Recommendation: brightness sensitivity is present, but classification shortcut strength should be interpreted with flip and within-camera evidence before Stage3-A1.\n"
    else:
        decision_text += "Recommendation: do not proceed blindly to Stage3-A1; current evidence does not establish a stable within-camera brightness shortcut.\n"
    (dirs["reports"] / "stage3_a0_to_a1_decision.md").write_text(decision_text, encoding="utf-8")
    machine = {
        "runtime": runtime_payload,
        "resume_config": resume_payload,
        "resume_integrity": integrity,
        "smoke": smoke_checks,
        "formal_inference_rows": len(long),
        "qc_summary": json.loads((dirs["qc"] / "counterfactual_qc_summary.json").read_text(encoding="utf-8")),
        "primary_endpoints": primary.to_dict("records"),
        "flip_rates": flips.to_dict("records"),
        "evidence": evidence,
        "training_executed": False,
        "checkpoint_modified": False,
        "stage3_a1_executed": False,
    }
    _write_json(dirs["reports"] / "stage3_a0_machine_summary.json", machine)
    report = (
        "# Stage3-A0 Frozen RGB Paired Exposure/Gamma Stress\n\n"
        f"Runtime: {runtime_payload['python_executable']} / torch {runtime_payload['torch']} / GPU {runtime_payload['gpu']}.\n\n"
        f"Formal inference rows: {len(long)}. Smoke pass: {smoke_checks['pass']}.\n\n"
        f"Evidence: {evidence['counterfactual_luminance_sensitivity_level']}, within-camera: {evidence['within_camera_evidence_level']}, shortcut: {evidence['brightness_shortcut_support']}.\n\n"
        "No model training, checkpoint modification, threshold optimization, or Stage3-A1 execution was performed.\n"
    )
    (dirs["reports"] / "stage3_a0_report.md").write_text(report, encoding="utf-8")
    inventory = {str(path.relative_to(config.output_dir)): path.stat().st_size for path in config.output_dir.rglob("*") if path.is_file()}
    _write_json(dirs["reports"] / "stage3_a0_output_inventory.json", inventory)
    (dirs["logs"] / "run.log").write_text("Stage3-A0 resume completed; stopped before Stage3-A1.\n", encoding="utf-8")
    pd.DataFrame({"warning": []}).to_csv(dirs["logs"] / "warnings.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"failure": []}).to_csv(dirs["logs"] / "failures.csv", index=False, encoding="utf-8-sig")
    if temp_root.exists():
        shutil.rmtree(temp_root)
    if smoke_dirs["tmp_images"].exists():
        shutil.rmtree(smoke_dirs["tmp_images"])
    return machine
