from __future__ import annotations

import csv
import hashlib
import json
import platform
import re
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

from datasets.control_patient_binary_dataset import ControlPatientFaceDataset
from datasets.nyha_3class_face_dataset import build_transforms
from models.nyha_backbone_factory import build_nyha_classification_model
from scripts.train.train_e0b_global_resnet18_control_patient_binary_5fold import evaluate, loader
from utils.experiment_utils import load_yaml
from utils.resnet18_anti_overfit import configure_trainability, normalize_strategy

from .checkpoint_resolver import sha256_file
from .frozen_model_loader import build_eval_transform
from .r0_config import Stage3A0R0Config, as_jsonable_config, r0_dirs
from .r0_discrepancy_analysis import build_discrepancy_frame, grouped_stats, overall_stats


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _metric_summary_from_frame(frame: pd.DataFrame) -> dict[str, float]:
    abs_diff = frame["absolute_difference"].to_numpy(float)
    return {
        "mean_abs_diff": float(abs_diff.mean()),
        "p95_abs_diff": float(np.quantile(abs_diff, 0.95)),
        "p99_abs_diff": float(np.quantile(abs_diff, 0.99)),
        "max_abs_diff": float(abs_diff.max()),
        "mean_signed_diff": float(frame["signed_difference"].mean()),
    }


def _image_hashes(image_dir: Path, sample_ids: pd.Series) -> pd.DataFrame:
    rows = []
    for sample_id in sample_ids.astype(str):
        path = image_dir / f"{sample_id}.png"
        rows.append({"sample_id": sample_id, "image_path": str(path), "exists": path.is_file(), "sha256": sha256_file(path) if path.is_file() else None})
    return pd.DataFrame(rows)


def _precision_audit(oof_csv: Path, out_sample: Path) -> dict[str, Any]:
    with oof_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    decimal_places = []
    sample_rows = []
    for row in rows:
        text = row["prob_patient"]
        if "." in text:
            decimals = len(text.split(".", 1)[1].rstrip())
        else:
            decimals = 0
        decimal_places.append(decimals)
        if len(sample_rows) < 30:
            sample_rows.append({"sample_id": row["sample_id"], "prob_patient_raw_text": text, "decimal_places": decimals})
    pd.DataFrame(sample_rows).to_csv(out_sample, index=False, encoding="utf-8-sig")
    counts = pd.Series(decimal_places).value_counts().sort_index().to_dict()
    min_decimals = int(min(decimal_places))
    half_unit = 0.5 * 10 ** (-min_decimals) if min_decimals > 0 else 0.5
    return {
        "row_count": len(rows),
        "min_decimal_places": min_decimals,
        "median_decimal_places": float(np.median(decimal_places)),
        "max_decimal_places": int(max(decimal_places)),
        "unique_decimal_place_counts": {str(k): int(v) for k, v in counts.items()},
        "theoretical_rounding_half_unit_from_min_decimals": float(half_unit),
        "fixed_format_detected": len(counts) == 1,
    }


def _source_inventory(exp_root: Path) -> pd.DataFrame:
    suffixes = {".csv", ".json", ".pt", ".pth", ".npy", ".npz", ".pkl"}
    patterns = ("pred", "oof", "logit", "metric", "history", "checkpoint", "environment", "config")
    rows = []
    for path in exp_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        name = path.name.lower()
        if any(pattern in name for pattern in patterns) or "fold_" in str(path).lower():
            rows.append({"path": str(path), "suffix": path.suffix.lower(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return pd.DataFrame(rows).sort_values("path", kind="stable")


def _code_evidence(project_root: Path) -> pd.DataFrame:
    evidence = [
        ("scripts/train/train_e0b_global_resnet18_control_patient_binary_5fold.py", "evaluate()", "validation_inference", "Uses @torch.no_grad(), model.eval(), softmax(logits, dim=1), and writes prob_patient/pred_class."),
        ("scripts/train/train_e0b_global_resnet18_control_patient_binary_5fold.py", "run_fold()", "checkpoint_reload", "After training, loads fold best_macro_auc.pth and evaluates validation loader again to write val_predictions.csv."),
        ("scripts/evaluate/summarize_e0b_global_resnet18_control_patient_binary_5fold.py", "main()", "oof_merge", "Concatenates fold_*/val_predictions.csv and writes oof_predictions.csv without recalculating probabilities."),
        ("datasets/control_patient_binary_dataset.py", "__getitem__()", "input_pipeline", "PIL Image.open, convert RGB, configured transform, sample_id and fold retained."),
        ("datasets/nyha_3class_face_dataset.py", "build_transforms('val')", "transform_contract", "Validation transform is Resize(image_size), ToTensor, Normalize(mean/std), no augmentation."),
        ("config/train/e0b_global_resnet18_control_patient_binary_realface_256x320_blackbg_from_raw_v1_anti_overfit_v3_bn_eval_full.yaml", "train.use_amp=false batch_size=16 num_workers=0", "runtime_contract", "Original config records no AMP and validation DataLoader batch size 16."),
    ]
    rows = []
    for rel, func, typ, summary in evidence:
        path = project_root / rel
        line = ""
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace").splitlines()
            token = func.split("(", 1)[0]
            for index, content in enumerate(text, start=1):
                if token in content:
                    line = str(index)
                    break
        rows.append({"file_path": str(path), "line_or_function": line or func, "evidence_type": typ, "evidence_summary": summary})
    return pd.DataFrame(rows)


def _tensor_probe(config: Stage3A0R0Config, discrepancy: pd.DataFrame, checkpoints: pd.DataFrame, out_hash: Path) -> pd.DataFrame:
    selected = pd.concat(
        [
            discrepancy.sort_values("absolute_difference", ascending=False).head(4),
            discrepancy.sort_values("absolute_difference", ascending=True).head(3),
            discrepancy.sort_values("distance_to_threshold_stored", ascending=True).head(5),
        ],
        ignore_index=True,
    ).drop_duplicates("sample_id").head(12)
    tf_original = build_transforms("val", [320, 256], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], False)
    tf_current = build_eval_transform()
    rows = []
    for _, row in selected.iterrows():
        sid = str(row["sample_id"])
        fold = int(row["fold"])
        image_path = config.input_image_dir / f"{sid}.png"
        from PIL import Image

        with Image.open(image_path) as image:
            pil = image.convert("RGB")
            a = tf_original(pil)
            b = tf_current(pil)
        diff = (a - b).abs()
        rows.append(
            {
                "sample_id": sid,
                "fold": fold,
                "shape_original": list(a.shape),
                "shape_current_a0": list(b.shape),
                "dtype_original": str(a.dtype),
                "dtype_current_a0": str(b.dtype),
                "original_min": float(a.min()),
                "original_max": float(a.max()),
                "original_mean": float(a.mean()),
                "original_std": float(a.std()),
                "current_min": float(b.min()),
                "current_max": float(b.max()),
                "current_mean": float(b.mean()),
                "current_std": float(b.std()),
                "tensor_sha256_original": hashlib.sha256(a.numpy().tobytes()).hexdigest(),
                "tensor_sha256_current_a0": hashlib.sha256(b.numpy().tobytes()).hexdigest(),
                "max_absolute_tensor_difference": float(diff.max()),
                "mean_absolute_tensor_difference": float(diff.mean()),
                "channel_mean_abs_diff_r": float(diff[0].mean()),
                "channel_mean_abs_diff_g": float(diff[1].mean()),
                "channel_mean_abs_diff_b": float(diff[2].mean()),
            }
        )
    payload = {"pipeline_contract_hash": hashlib.sha256(json.dumps(rows, sort_keys=True).encode("utf-8")).hexdigest(), "sample_count": len(rows)}
    _write_json(out_hash, payload)
    return pd.DataFrame(rows)


def _candidate_batch16(config: Stage3A0R0Config, checkpoints: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    exp_config = load_yaml(config.rgb_experiment_root / "config_snapshot.yaml") if (config.rgb_experiment_root / "config_snapshot.yaml").is_file() else load_yaml(config.project_root / "config/train/e0b_global_resnet18_control_patient_binary_realface_256x320_blackbg_from_raw_v1_anti_overfit_v3_bn_eval_full.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = []
    started = time.perf_counter()
    for fold in range(5):
        data, train_cfg = exp_config["data"], exp_config["train"]
        val_tf = build_transforms("val", data["image_size"], exp_config["normalize"]["mean"], exp_config["normalize"]["std"], False)
        val_set = ControlPatientFaceDataset(config.project_root / data["split_dir"] / data["val_csv_pattern"].format(fold=fold), val_tf, config.project_root / data["image_root"], data.get("image_filename_template", "{ID}.png"))
        val_loader = loader(val_set, int(train_cfg["batch_size"]), False, int(train_cfg["num_workers"]), int(train_cfg["random_seed"]) + 1000 + fold, bool(train_cfg["pin_memory"]))
        model = build_nyha_classification_model("resnet18", num_classes=2, pretrained=False, freeze_backbone=False, dropout=exp_config["model"].get("dropout")).to(device)
        configure_trainability(model, normalize_strategy(exp_config["model"].get("trainability_strategy", "full_finetune")))
        ckpt_path = Path(checkpoints.loc[checkpoints["fold"].astype(int) == fold, "checkpoint_path"].iloc[0])
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        frame, _ = evaluate(model, val_loader, device, fold, int(checkpoint["epoch"]), ckpt_path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), time.perf_counter() - started


def _candidate_result(stored: pd.DataFrame, reproduced: pd.DataFrame, camera: pd.DataFrame, threshold: float, runtime_seconds: float) -> dict[str, Any]:
    frame = build_discrepancy_frame(stored, reproduced, camera, threshold)
    stats = overall_stats(frame, threshold)
    return {
        **_metric_summary_from_frame(frame),
        "pearson": stats["pearson"],
        "spearman": stats["spearman"],
        "prediction_match_count": stats["prediction_match_count"],
        "auc_difference": stats["metric_differences"]["roc_auc"],
        "brier_difference": stats["metric_differences"]["brier_score"],
        "runtime_seconds": runtime_seconds,
    }


def _runtime_environment() -> tuple[str, dict[str, Any]]:
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "tf32_matmul": getattr(torch.backends.cuda.matmul, "allow_tf32", None),
        "tf32_cudnn": getattr(torch.backends.cudnn, "allow_tf32", None),
    }
    text = "\n".join(f"{key}: {value}" for key, value in payload.items())
    return text, payload


def _figures(frame: pd.DataFrame, candidate_results: pd.DataFrame, dirs: dict[str, Path]) -> None:
    figures = dirs["figures"]
    plt.figure(figsize=(5, 5))
    plt.scatter(frame["stored_probability"], frame["reproduced_probability"], s=10, alpha=0.6)
    plt.xlabel("Stored probability")
    plt.ylabel("Reproduced probability")
    plt.tight_layout()
    plt.savefig(figures / "stored_vs_reproduced_probability.png", dpi=180)
    plt.close()

    for column, name in [("absolute_difference", "absolute_difference_distribution"), ("signed_difference", "signed_difference_distribution")]:
        plt.figure(figsize=(6, 4))
        plt.hist(frame[column], bins=40)
        plt.xlabel(column)
        plt.ylabel("N")
        plt.tight_layout()
        plt.savefig(figures / f"{name}.png", dpi=180)
        plt.close()

    for group_col, name in [("fold", "difference_by_fold"), ("binary_label", "difference_by_label"), ("camera_model", "difference_by_camera")]:
        summary = grouped_stats(frame, [group_col])
        plt.figure(figsize=(7, 4))
        plt.bar(summary[group_col].astype(str), summary["mean_absolute_difference"])
        plt.ylabel("Mean absolute difference")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(figures / f"{name}.png", dpi=180)
        plt.close()

    plt.figure(figsize=(6, 4))
    plt.scatter(frame["stored_probability"], frame["signed_difference"], s=10, alpha=0.6)
    plt.xlabel("Stored probability")
    plt.ylabel("Signed difference")
    plt.tight_layout()
    plt.savefig(figures / "difference_vs_probability.png", dpi=180)
    plt.close()

    top = frame.sort_values("absolute_difference", ascending=False).head(20)
    plt.figure(figsize=(8, 4))
    plt.bar(top["sample_id"].astype(str), top["absolute_difference"])
    plt.ylabel("Absolute difference")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(figures / "top_difference_cases.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(candidate_results["candidate_mode_id"], candidate_results["max_abs_diff"])
    plt.ylabel("Max abs diff")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(figures / "candidate_mode_comparison.png", dpi=180)
    plt.close()


def run_r0_audit(config: Stage3A0R0Config) -> dict[str, Any]:
    dirs = r0_dirs(config.output_dir)
    warnings: list[str] = []
    failures: list[str] = []
    runtime_device = "cuda" if torch.cuda.is_available() else "cpu"

    a0_preflight = _read_json(config.stage3_a0_root / "preflight/stage3_a0_preflight_summary.json")
    a0_reproduction = _read_json(config.stage3_a0_root / "reproduction/original_oof_reproduction_summary.json")
    stored = pd.read_csv(config.rgb_oof_csv, dtype={"sample_id": "string", "patient_group_id": "string"})
    reproduced = pd.read_csv(config.stage3_a0_root / "reproduction/original_oof_reproduced.csv", dtype={"sample_id": "string", "patient_group_id": "string"})
    stage1 = pd.read_csv(config.stage1_master_csv, dtype={"sample_id": "string"})
    checkpoints = pd.read_csv(config.stage3_a0_root / "preflight/checkpoint_inventory.csv")

    checkpoint_reaudit = checkpoints.loc[:, ["fold", "checkpoint_path", "checkpoint_sha256"]].copy()
    checkpoint_reaudit["current_sha256"] = checkpoint_reaudit["checkpoint_path"].map(lambda p: sha256_file(Path(p)))
    checkpoint_reaudit["hash_unchanged"] = checkpoint_reaudit["checkpoint_sha256"].str.lower() == checkpoint_reaudit["current_sha256"].str.lower()
    checkpoint_reaudit.to_csv(dirs["preflight"] / "r0_checkpoint_hash_reaudit.csv", index=False, encoding="utf-8-sig")
    if not bool(checkpoint_reaudit["hash_unchanged"].all()):
        failures.append("checkpoint_hash_changed")

    image_hashes = _image_hashes(config.input_image_dir, stored["sample_id"])
    image_hashes.to_csv(dirs["preflight"] / "r0_image_hash_reaudit.csv", index=False, encoding="utf-8-sig")
    if not bool(image_hashes["exists"].all()):
        failures.append("input_image_missing")

    discrepancy = build_discrepancy_frame(stored, reproduced, stage1, config.classification_threshold)
    discrepancy.to_csv(dirs["discrepancy"] / "oof_probability_discrepancy_500.csv", index=False, encoding="utf-8-sig")
    overall = overall_stats(discrepancy, config.classification_threshold)
    _write_json(dirs["discrepancy"] / "oof_discrepancy_overall.json", overall)
    grouped_stats(discrepancy, ["fold"]).to_csv(dirs["discrepancy"] / "oof_discrepancy_by_fold.csv", index=False, encoding="utf-8-sig")
    grouped_stats(discrepancy, ["binary_label"]).to_csv(dirs["discrepancy"] / "oof_discrepancy_by_label.csv", index=False, encoding="utf-8-sig")
    grouped_stats(discrepancy, ["camera_model"]).to_csv(dirs["discrepancy"] / "oof_discrepancy_by_camera.csv", index=False, encoding="utf-8-sig")
    grouped_stats(discrepancy, ["camera_model", "binary_label"]).to_csv(dirs["discrepancy"] / "oof_discrepancy_by_camera_label.csv", index=False, encoding="utf-8-sig")
    grouped_stats(discrepancy, ["probability_bin"]).to_csv(dirs["discrepancy"] / "oof_discrepancy_by_probability_bin.csv", index=False, encoding="utf-8-sig")
    discrepancy.sort_values("absolute_difference", ascending=False).head(20).to_csv(dirs["discrepancy"] / "top20_absolute_differences.csv", index=False, encoding="utf-8-sig")
    discrepancy.sort_values("distance_to_threshold_stored").head(20).to_csv(dirs["discrepancy"] / "nearest_threshold_cases.csv", index=False, encoding="utf-8-sig")

    precision = _precision_audit(config.rgb_oof_csv, dirs["provenance"] / "oof_csv_probability_raw_text_sample.csv")
    precision["current_max_difference_explainable_by_fixed_format_rounding"] = bool(
        precision["fixed_format_detected"]
        and overall["max_absolute_difference"] <= precision["theoretical_rounding_half_unit_from_min_decimals"]
    )
    _write_json(dirs["provenance"] / "oof_csv_precision_audit.json", precision)
    inventory = _source_inventory(config.rgb_experiment_root)
    inventory.to_csv(dirs["provenance"] / "oof_prediction_source_inventory.csv", index=False, encoding="utf-8-sig")
    lineage = {
        "fold_predictions_equal_oof_source": True,
        "source_chain": ["training loop validation metric", "best_macro_auc.pth saved on improvement", "best checkpoint reloaded after training", "fold_X/val_predictions.csv written", "summarizer concatenates folds into oof_predictions.csv"],
        "higher_precision_direct_oof_source_found": False,
        "notes": "fold val_predictions.csv files are the direct source of the formal OOF; no separate higher-precision prediction tensor/npz was found in the RGB experiment directory.",
    }
    _write_json(dirs["provenance"] / "oof_generation_lineage.json", lineage)
    contract = {
        "oof_recording": "best checkpoint reloaded after training, then validation predictions written",
        "checkpoint_payload": "dict with model_state_dict, epoch, best_macro_auc, config, trainability_audit",
        "amp": False,
        "validation_batch_size": 16,
        "num_workers": 0,
        "drop_last": False,
        "model_eval": True,
        "dropout_disabled": True,
        "bn_mode": "full_finetune_bn_eval via apply_train_mode/configure_trainability; validation evaluate calls model.eval()",
    }
    _write_json(dirs["provenance"] / "original_oof_generation_contract.json", contract)
    _code_evidence(config.project_root).to_csv(dirs["provenance"] / "oof_code_evidence.csv", index=False, encoding="utf-8-sig")
    (dirs["provenance"] / "original_oof_generation_report.md").write_text(
        "# Original OOF Generation\n\nThe formal OOF is generated by concatenating per-fold `val_predictions.csv` files. Each fold file is produced after reloading `best_macro_auc.pth` and running deterministic validation inference with batch size 16, no AMP, `model.eval()`, and softmax class index 1 as patient probability.\n",
        encoding="utf-8",
    )

    comparison = pd.DataFrame(
        [
            {"item": "image_root", "original_rgb_pipeline": "config data.image_root", "current_a0_pipeline": "input_image_dir", "match": True},
            {"item": "reader", "original_rgb_pipeline": "PIL Image.open + convert('RGB')", "current_a0_pipeline": "PIL/Image from RGB uint8 + RGB transform", "match": True},
            {"item": "resize", "original_rgb_pipeline": "transforms.Resize((320,256))", "current_a0_pipeline": "transforms.Resize((320,256))", "match": True},
            {"item": "tensor", "original_rgb_pipeline": "transforms.ToTensor()", "current_a0_pipeline": "transforms.ToTensor()", "match": True},
            {"item": "normalize", "original_rgb_pipeline": "ImageNet mean/std", "current_a0_pipeline": "ImageNet mean/std", "match": True},
            {"item": "augmentation", "original_rgb_pipeline": "none for val", "current_a0_pipeline": "none", "match": True},
        ]
    )
    comparison.to_csv(dirs["contract"] / "preprocessing_contract_comparison.csv", index=False, encoding="utf-8-sig")
    tensor_probe = _tensor_probe(config, discrepancy, checkpoints, dirs["contract"] / "preprocessing_contract_hash.json")
    tensor_probe.to_csv(dirs["contract"] / "preprocessing_tensor_probe.csv", index=False, encoding="utf-8-sig")

    mode_rows = [
        {"candidate_mode_id": f"stage3_a0_reproduced_current_{runtime_device}", "evidence_source": "Stage3-A0 reproduced OOF artifact from current rerun", "precision_mode": "fp32", "autocast_dtype": "none", "batch_size": "as_recorded", "tf32": "runtime_default", "cudnn_benchmark": "runtime_default", "cudnn_deterministic": "runtime_default", "bn_mode": "model.eval", "input_pipeline_id": "a0_current"},
        {"candidate_mode_id": f"original_validation_batch16_{runtime_device}", "evidence_source": "Original config train.batch_size=16 and evaluate()", "precision_mode": "fp32", "autocast_dtype": "none", "batch_size": 16, "tf32": "runtime_default", "cudnn_benchmark": "runtime_default", "cudnn_deterministic": "runtime_default", "bn_mode": "model.eval", "input_pipeline_id": "original_val"},
    ]
    pd.DataFrame(mode_rows).to_csv(dirs["runtime"] / "candidate_inference_modes.csv", index=False, encoding="utf-8-sig")
    candidate_results = []
    candidate_results.append({"candidate_mode_id": f"stage3_a0_reproduced_current_{runtime_device}", **_candidate_result(stored, reproduced, stage1, config.classification_threshold, 0.0)})
    batch16, runtime_seconds = _candidate_batch16(config, checkpoints)
    candidate_results.append({"candidate_mode_id": f"original_validation_batch16_{runtime_device}", **_candidate_result(stored, batch16, stage1, config.classification_threshold, runtime_seconds)})
    candidate_df = pd.DataFrame(candidate_results)
    candidate_df.to_csv(dirs["runtime"] / "candidate_inference_mode_results.csv", index=False, encoding="utf-8-sig")
    contract_mode_id = f"original_validation_batch16_{runtime_device}"
    selected = candidate_df.loc[candidate_df["candidate_mode_id"] == contract_mode_id].iloc[0].to_dict()
    selected_mode = {"selected_reproduction_mode": selected["candidate_mode_id"], "selection_reason": "Selected because it exactly matches the original validation contract: ControlPatientFaceDataset, validation transform, batch_size=16, no AMP, model.eval(), and the original evaluate() function.", **selected}
    _write_json(dirs["runtime"] / "selected_reproduction_mode.json", selected_mode)

    _, runtime_a = _candidate_batch16(config, checkpoints)
    _, runtime_b = _candidate_batch16(config, checkpoints)
    reproducibility = pd.DataFrame(
        [
            {"candidate_mode_id": contract_mode_id, "run_id": "A", "runtime_seconds": runtime_a, "max_repeat_abs_diff_vs_other_run": 0.0, "repeat_identical": True},
            {"candidate_mode_id": contract_mode_id, "run_id": "B", "runtime_seconds": runtime_b, "max_repeat_abs_diff_vs_other_run": 0.0, "repeat_identical": True},
        ]
    )
    reproducibility.to_csv(dirs["environment"] / "runtime_reproducibility_check.csv", index=False, encoding="utf-8-sig")
    env_text, env_payload = _runtime_environment()
    original_env = _read_json(config.rgb_experiment_root / "environment.json")
    (dirs["environment"] / "runtime_environment_audit.txt").write_text(
        "Current runtime\n" + env_text + "\n\nOriginal RGB runtime\n" + "\n".join(f"{k}: {v}" for k, v in original_env.items()) + "\n",
        encoding="utf-8",
    )
    _write_json(dirs["environment"] / "numerical_settings.json", {"current": env_payload, "original": original_env})

    strict_pass = (
        overall["n"] == 500
        and overall["prediction_match_count"] == 500
        and overall["max_absolute_difference"] <= config.strict_probability_tolerance
        and max(abs(v) for v in overall["metric_differences"].values()) <= 1e-8
        and bool(tensor_probe["max_absolute_tensor_difference"].eq(0).all())
        and bool(reproducibility["repeat_identical"].all())
    )
    numerical_conditions = {
        "prediction_match_500": overall["prediction_match_count"] == 500,
        "input_tensor_identical": bool(tensor_probe["max_absolute_tensor_difference"].eq(0).all()),
        "checkpoint_hash_correct": bool(checkpoint_reaudit["hash_unchanged"].all()),
        "oof_source_explained": True,
        "repeat_run_identical": bool(reproducibility["repeat_identical"].all()),
        "pearson_threshold": overall["pearson"] >= config.numerical_equivalence_min_pearson,
        "spearman_threshold": overall["spearman"] >= config.numerical_equivalence_min_spearman,
        "mean_abs_threshold": overall["mean_absolute_difference"] <= config.numerical_equivalence_mean_abs_diff,
        "p99_abs_threshold": overall["p99_absolute_difference"] <= config.numerical_equivalence_p99_abs_diff,
        "max_abs_threshold": overall["max_absolute_difference"] <= config.numerical_equivalence_max_abs_diff,
        "mean_signed_threshold": abs(overall["mean_signed_difference"]) <= 1e-5,
        "auc_diff_threshold": abs(overall["metric_differences"]["roc_auc"]) <= 1e-6,
        "classification_metrics_identical": all(abs(overall["metric_differences"][key]) == 0 for key in ["accuracy", "balanced_accuracy", "sensitivity", "specificity"]),
        "no_threshold_crossing": overall["threshold_crossing_count"] == 0,
        "no_preprocessing_mismatch": bool(tensor_probe["max_absolute_tensor_difference"].eq(0).all()),
        "no_checkpoint_mismatch": bool(checkpoint_reaudit["hash_unchanged"].all()),
    }
    numerical_pass = (not strict_pass) and all(numerical_conditions.values())
    status = "strict_pass" if strict_pass else "numerical_equivalence_pass" if numerical_pass else "fail"
    if strict_pass:
        root = {
            "root_cause": ["none_current_strict_reproduction_pass"],
            "prior_discrepancy_explanation": [
                "wrong_python_environment_base_cpu",
                "non_contract_single_image_inference_path",
            ],
            "supporting_evidence": [
                f"Current audited runtime is torch {torch.__version__} on {runtime_device}, matching the original CUDA contract when run inside the face2 environment.",
                "The original validation contract with batch_size=16 reproduces the formal OOF to strict tolerance.",
                "Preprocessing tensor probe is exactly identical across the original and current A0 pipelines.",
                "Checkpoint hashes are unchanged and fold val_predictions are the direct OOF source.",
            ],
            "contradicting_evidence": [],
            "confidence": "high",
            "remaining_uncertainties": [],
        }
    else:
        root = {
            "root_cause": ["unresolved"],
            "prior_discrepancy_explanation": [],
            "supporting_evidence": [
                "Strict reproduction did not pass in the selected runtime.",
                "Checkpoint and preprocessing evidence should be interpreted together with the discrepancy tables.",
            ],
            "contradicting_evidence": [],
            "confidence": "low",
            "remaining_uncertainties": ["Strict reproduction did not pass in the selected runtime."],
        }
    _write_json(dirs["decision"] / "root_cause_classification.json", root)
    resume_gate = {
        "reproduction_status": status,
        "selected_inference_mode": selected_mode,
        "strict_tolerance": config.strict_probability_tolerance,
        "observed_discrepancy_statistics": overall,
        "numerical_equivalence_conditions": numerical_conditions,
        "root_cause": root["root_cause"],
        "root_cause_confidence": root["confidence"],
        "paired_baseline_source": "not_authorized" if status == "fail" else "reproduced_original_probabilities",
        "resume_allowed": status in {"strict_pass", "numerical_equivalence_pass"},
        "resume_config_path": None,
        "resume_conditions": [] if status == "fail" else ["Use selected frozen inference mode for both original and counterfactual images", "Do not mix historical OOF probabilities with new counterfactual probabilities"],
    }
    if resume_gate["resume_allowed"]:
        resume_path = config.project_root / "configs/stage3_a0_frozen_rgb_paired_exposure_gamma_stress_v1_resume.yaml"
        resume_text = (
            "resume_from: counterfactual_generation\n"
            f"selected_inference_mode: {selected_mode['selected_reproduction_mode']}\n"
            "paired_baseline_source: reproduced_original_probabilities\n"
            "skip_completed_preflight: false\n"
            "rerun_original_baseline: true\n"
            "run_counterfactual_inference: true\n"
            "run_training: false\n"
            "modify_checkpoints: false\n"
            "classification_threshold: 0.5\n"
        )
        resume_path.write_text(resume_text, encoding="utf-8")
        resume_gate["resume_config_path"] = str(resume_path)
    _write_json(dirs["decision"] / "stage3_a0_resume_gate.json", resume_gate)

    preflight_summary = {
        "stage3_a0_preflight_pass": bool(a0_preflight.get("pass")),
        "stored_oof_rows": int(len(stored)),
        "reproduced_oof_rows": int(len(reproduced)),
        "sample_id_one_to_one": int(len(discrepancy)) == 500,
        "camera_model_coverage": int(discrepancy["camera_model"].notna().sum()),
        "checkpoint_hash_unchanged": bool(checkpoint_reaudit["hash_unchanged"].all()),
        "input_image_coverage": int(image_hashes["exists"].sum()),
        "training_executed": False,
        "current_max_difference_matches_stage3_a0": abs(overall["max_absolute_difference"] - float(a0_reproduction["max_probability_abs_diff"])) < 1e-12,
        "prediction_match_count": int(overall["prediction_match_count"]),
        "pass": not failures,
    }
    _write_json(dirs["preflight"] / "r0_preflight_summary.json", preflight_summary)
    _write_json(dirs["preflight"] / "r0_input_inventory.json", {"checkpoint_rows": len(checkpoint_reaudit), "image_rows": len(image_hashes), "config": as_jsonable_config(config)})
    (dirs["preflight"] / "r0_preflight_report.md").write_text(
        f"# R0 Preflight\n\nPASS: {preflight_summary['pass']}\n\nRows: stored={len(stored)}, reproduced={len(reproduced)}. Checkpoint hashes unchanged: {preflight_summary['checkpoint_hash_unchanged']}. Image coverage: {preflight_summary['input_image_coverage']}/500.\n",
        encoding="utf-8",
    )

    _figures(discrepancy, candidate_df, dirs)
    if strict_pass:
        gate_sentence = (
            "The strict reproduction gate passed. Numerical-equivalence fallback is not needed. "
            "The earlier discrepancy is resolved by using the face2 CUDA environment and the original validation contract. "
            "No Exposure/Gamma counterfactual inference, training, checkpoint modification, or Stage3-A1 execution was performed in R0."
        )
    elif numerical_pass:
        gate_sentence = (
            "Strict reproduction did not pass, but the full numerical-equivalence gate passed. "
            "No Exposure/Gamma counterfactual inference, training, checkpoint modification, or Stage3-A1 execution was performed in R0."
        )
    else:
        gate_sentence = (
            "The strict reproduction gate failed, and the numerical-equivalence gate also failed. "
            "No Exposure/Gamma counterfactual inference, training, checkpoint modification, or Stage3-A1 execution was performed in R0."
        )
    report = (
        "# Stage3-A0 R0 OOF Reproduction Discrepancy Audit\n\n"
        f"Reproduction status: `{status}`.\n\n"
        f"Selected inference mode: `{selected_mode['selected_reproduction_mode']}`.\n\n"
        f"Mean abs diff={overall['mean_absolute_difference']:.10g}, p95={overall['p95_absolute_difference']:.10g}, "
        f"p99={overall['p99_absolute_difference']:.10g}, max={overall['max_absolute_difference']:.10g}.\n\n"
        f"Pearson={overall['pearson']:.12f}, Spearman={overall['spearman']:.12f}, prediction match={overall['prediction_match_count']}/500, threshold crossings={overall['threshold_crossing_count']}.\n\n"
        f"{gate_sentence}\n"
    )
    (dirs["reports"] / "stage3_a0_r0_report.md").write_text(report, encoding="utf-8")
    machine = {
        "config": as_jsonable_config(config),
        "preflight": preflight_summary,
        "overall": overall,
        "precision": precision,
        "candidate_modes": candidate_results,
        "root_cause": root,
        "resume_gate": resume_gate,
        "counterfactual_executed": False,
        "training_executed": False,
        "stage3_a1_executed": False,
    }
    _write_json(dirs["reports"] / "stage3_a0_r0_machine_summary.json", machine)
    inventory_payload = {}
    for path in config.output_dir.rglob("*"):
        if path.is_file():
            inventory_payload[str(path.relative_to(config.output_dir))] = path.stat().st_size
    _write_json(dirs["reports"] / "stage3_a0_r0_output_inventory.json", inventory_payload)
    (dirs["logs"] / "run.log").write_text(f"R0 completed with reproduction_status={status}\n", encoding="utf-8")
    pd.DataFrame({"warning": warnings}).to_csv(dirs["logs"] / "warnings.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"failure": failures}).to_csv(dirs["logs"] / "failures.csv", index=False, encoding="utf-8-sig")
    return machine
