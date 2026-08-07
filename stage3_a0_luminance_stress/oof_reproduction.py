from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, recall_score, roc_auc_score

from datasets.control_patient_binary_dataset import ControlPatientFaceDataset
from datasets.nyha_3class_face_dataset import build_transforms
from models.nyha_backbone_factory import build_nyha_classification_model
from scripts.train.train_e0b_global_resnet18_control_patient_binary_5fold import evaluate, loader
from utils.experiment_utils import load_yaml
from utils.resnet18_anti_overfit import configure_trainability, normalize_strategy

from .config import Stage3A0Config
from .frozen_model_loader import cuda_or_cpu


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    pred = (p >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "brier_score": float(brier_score_loss(y, p)),
    }


def reproduce_oof(config: Stage3A0Config, checkpoints: pd.DataFrame, dirs: dict[str, Path]) -> dict[str, Any]:
    oof = pd.read_csv(config.rgb_oof_csv, dtype={"sample_id": str, "patient_group_id": str})
    rgb_config_path = config.rgb_experiment_root / "config_snapshot.yaml"
    rgb_config = load_yaml(rgb_config_path if rgb_config_path.is_file() else config.project_root / "config/train/e0b_global_resnet18_control_patient_binary_realface_256x320_blackbg_from_raw_v1_anti_overfit_v3_bn_eval_full.yaml")
    device = cuda_or_cpu()
    frames = []
    for fold in sorted(oof["fold"].astype(int).unique()):
        fold = int(fold)
        data, train_cfg = rgb_config["data"], rgb_config["train"]
        val_tf = build_transforms("val", data["image_size"], rgb_config["normalize"]["mean"], rgb_config["normalize"]["std"], False)
        val_set = ControlPatientFaceDataset(
            config.project_root / data["split_dir"] / data["val_csv_pattern"].format(fold=fold),
            val_tf,
            config.project_root / data["image_root"],
            data.get("image_filename_template", "{ID}.png"),
        )
        val_loader = loader(
            val_set,
            int(train_cfg["batch_size"]),
            False,
            int(train_cfg["num_workers"]),
            int(train_cfg["random_seed"]) + 1000 + fold,
            bool(train_cfg["pin_memory"]),
        )
        ckpt = Path(checkpoints.loc[checkpoints["fold"] == int(fold), "checkpoint_path"].iloc[0])
        model = build_nyha_classification_model(
            "resnet18",
            num_classes=2,
            pretrained=False,
            freeze_backbone=False,
            dropout=rgb_config["model"].get("dropout"),
        ).to(device)
        configure_trainability(model, normalize_strategy(rgb_config["model"].get("trainability_strategy", "full_finetune")))
        checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        frame, _ = evaluate(model, val_loader, device, fold, int(checkpoint["epoch"]), ckpt)
        frames.append(frame)
    raw_repro = pd.concat(frames, ignore_index=True)
    repro = raw_repro.rename(
        columns={
            "logit_normal": "reproduced_logit_normal",
            "logit_patient": "reproduced_logit_patient",
            "prob_patient": "reproduced_probability",
            "pred_class": "reproduced_prediction",
        }
    ).loc[
        :,
        [
            "sample_id",
            "patient_group_id",
            "fold",
            "binary_label",
            "reproduced_logit_normal",
            "reproduced_logit_patient",
            "reproduced_probability",
            "reproduced_prediction",
            "checkpoint_path",
        ],
    ]
    merged = oof.loc[:, ["sample_id", "fold", "prob_patient", "pred_class", "logit_patient"]].merge(
        repro.loc[:, ["sample_id", "reproduced_probability", "reproduced_prediction", "reproduced_logit_patient"]],
        on="sample_id",
        validate="one_to_one",
    )
    rows = []
    for row in merged.itertuples(index=False):
        rows.append({
            "sample_id": row.sample_id,
            "fold": int(row.fold),
            "stored_probability": float(row.prob_patient),
            "reproduced_probability": float(row.reproduced_probability),
            "absolute_difference": abs(float(row.prob_patient) - float(row.reproduced_probability)),
            "stored_prediction": int(row.pred_class),
            "reproduced_prediction": int(row.reproduced_prediction),
            "prediction_match": int(int(row.pred_class) == int(row.reproduced_prediction)),
            "stored_logit_patient": float(row.logit_patient),
            "reproduced_logit_patient": float(row.reproduced_logit_patient),
            "logit_patient_abs_difference": abs(float(row.logit_patient) - float(row.reproduced_logit_patient)),
        })
    comp = pd.DataFrame(rows)
    repro.to_csv(dirs["reproduction"] / "original_oof_reproduced.csv", index=False, encoding="utf-8-sig")
    comp.to_csv(dirs["reproduction"] / "original_oof_comparison.csv", index=False, encoding="utf-8-sig")
    y = oof.sort_values("sample_id")["binary_label"].astype(int).to_numpy()
    stored_p = oof.sort_values("sample_id")["prob_patient"].astype(float).to_numpy()
    rep_sorted = repro.sort_values("sample_id")
    rep_p = rep_sorted["reproduced_probability"].astype(float).to_numpy()
    stored_metrics = _metrics(y, stored_p)
    reproduced_metrics = _metrics(y, rep_p)
    summary = {
        "device": str(device),
        "coverage": int(len(repro)),
        "max_probability_abs_diff": float(comp["absolute_difference"].max()),
        "mean_probability_abs_diff": float(comp["absolute_difference"].mean()),
        "max_logit_patient_abs_diff": float(comp["logit_patient_abs_difference"].max()),
        "prediction_match_count": int(comp["prediction_match"].sum()),
        "prediction_match_rate": float(comp["prediction_match"].mean()),
        "stored_metrics": stored_metrics,
        "reproduced_metrics": reproduced_metrics,
        "metric_abs_differences": {k: abs(stored_metrics[k] - reproduced_metrics[k]) for k in stored_metrics},
        "csv_min_observed_probability_step": _min_step(oof["prob_patient"].astype(float)),
        "probability_tolerance": 1e-5,
    }
    summary["pass"] = bool(
        summary["coverage"] == 500
        and summary["max_probability_abs_diff"] <= 1e-5
        and summary["prediction_match_count"] == 500
        and all(v <= 1e-8 for v in summary["metric_abs_differences"].values())
    )
    (dirs["reproduction"] / "original_oof_reproduction_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _min_step(values: pd.Series) -> float | None:
    vals = np.sort(values.dropna().unique())
    diffs = np.diff(vals)
    diffs = diffs[diffs > 0]
    return float(diffs.min()) if len(diffs) else None
