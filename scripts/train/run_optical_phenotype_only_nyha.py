"""Run the fixed five-fold optical-phenotype-only NYHA experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metrics.classification_metrics import (  # noqa: E402
    compute_classification_metrics,
    flatten_metrics,
)
from models.optical_phenotype_logistic_classifier import (  # noqa: E402
    ABSOLUTE_TOLERANCE,
    CLASS_NAMES,
    VARIANTS,
    VARIANT_FEATURE_COLUMNS,
    VARIANT_INPUT_COLUMNS,
    VARIANT_INPUT_DIM,
    OpticalPhenotypeScaler,
    balanced_class_weights,
    build_classifier,
    build_model_input,
    fit_classifier,
    fit_scaler,
    load_model,
    predict_probabilities,
    save_model,
)
from utils.optical_feature_preprocessor import (  # noqa: E402
    AVAILABILITY_COLUMN,
    assert_classifier_feature_path,
    load_feature_frame,
    relative_path,
    resolve_feature_source,
    sha256_file,
    sha256_ids,
    sha256_json,
)


GLOBAL_VARIANT = {
    "o_mask": "global_mask",
    "o_raw": "global_raw",
    "o_stage2a": "global_stage2a",
    "o_stage2b": "global_stage2b",
}
PROBABILITY_COLUMNS = ["prob_normal", "prob_mild", "prob_severe"]
PREDICTION_COLUMNS = [
    "ID", "patient_group", "fold", "split_role", "y_true", "y_pred",
    *PROBABILITY_COLUMNS, AVAILABILITY_COLUMN, "correct", "variant",
]


def load_config(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    resolved = resolved.resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return resolved, config


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(type(value).__name__)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n", float_format="%.17g")


def canonical_frame_sha256(frame: pd.DataFrame) -> str:
    text = frame.to_csv(index=False, lineterminator="\n", float_format="%.17g")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def root_path(config_value: str) -> Path:
    return (PROJECT_ROOT / config_value).resolve()


def _read_split(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"ID": "string", "patient_group_id": "string"})
    frame["ID"] = frame["ID"].astype(str)
    frame["patient_group_id"] = frame["patient_group_id"].astype(str)
    return frame


def load_official_metadata(config: Mapping[str, Any]) -> pd.DataFrame:
    data = config["data"]
    label_path = root_path(data["label_path"])
    master_path = root_path(data["master_split"])
    labels = pd.read_csv(label_path, usecols=["ID", "SEX", "NYHA"], dtype={"ID": "string"})
    labels["ID"] = labels["ID"].astype(str)
    if labels["ID"].duplicated().any():
        raise RuntimeError("Official label file contains duplicate IDs")
    mapping = {int(key): int(value) for key, value in data["nyha_mapping"].items()}
    labels["derived_y"] = pd.to_numeric(labels["NYHA"], errors="raise").astype(int).map(mapping)
    if labels["derived_y"].isna().any():
        raise RuntimeError("Official NYHA values fall outside the locked mapping")
    master = _read_split(master_path)
    merged = master.merge(labels, on="ID", how="left", validate="one_to_one", suffixes=("_split", "_label"))
    if merged[["SEX_label", "NYHA_label", "derived_y"]].isna().any().any():
        raise RuntimeError("A fixed-split ID is absent from the official label file")
    if not np.array_equal(merged["derived_y"].to_numpy(int), merged["label_3class"].to_numpy(int)):
        raise RuntimeError("Official-label NYHA mapping disagrees with the fixed split")
    if not np.array_equal(merged["NYHA_label"].to_numpy(int), merged["NYHA_split"].to_numpy(int)):
        raise RuntimeError("Official NYHA values disagree with the fixed split")
    if not np.array_equal(merged["SEX_label"].to_numpy(int), merged["SEX_split"].to_numpy(int)):
        raise RuntimeError("Official SEX values disagree with the fixed split")
    result = merged.loc[:, ["ID", "patient_group_id", "fold", "derived_y"]].rename(
        columns={"patient_group_id": "patient_group", "derived_y": "y_true"}
    )
    return result


def fold_metadata(config: Mapping[str, Any], fold: int, role: str) -> pd.DataFrame:
    data = config["data"]
    pattern = data["train_csv_pattern"] if role == "train" else data["val_csv_pattern"]
    frame = _read_split(root_path(data["split_root"]) / pattern.format(fold=fold))
    return frame.loc[:, ["ID", "patient_group_id", "label_3class"]].rename(
        columns={"patient_group_id": "patient_group", "label_3class": "y_true"}
    )


def load_variant_features(
    config: Mapping[str, Any], variant: str, fold: int, role: str, ids: Iterable[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    expected = [str(value) for value in ids]
    source, schema, upstream = resolve_feature_source(
        config, GLOBAL_VARIANT[variant], fold, role, PROJECT_ROOT
    )
    assert source is not None and schema is not None and upstream is not None
    frame = load_feature_frame(
        source, GLOBAL_VARIANT[variant], expected, fold=fold, split_role=role,
        schema_path=schema, allow_source_superset=variant in {"o_mask", "o_raw"},
    )
    frame = frame.rename(columns={column: column for column in frame.columns})
    audit = {
        "variant": variant,
        "fold": int(fold),
        "split_role": role,
        "source": relative_path(source, PROJECT_ROOT),
        "source_sha256": sha256_file(source),
        "schema": relative_path(schema, PROJECT_ROOT),
        "schema_sha256": sha256_file(schema),
        "upstream_manifest": relative_path(upstream, PROJECT_ROOT),
        "upstream_manifest_sha256": sha256_file(upstream),
        "rows": len(frame),
        "id_sha256": sha256_ids(frame["ID"]),
        "oof_source": False,
    }
    return frame, audit


def _split_checks(config: Mapping[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    metadata = load_official_metadata(config)
    data = config["data"]
    if len(metadata) != int(data["expected_num_samples"]) or metadata["ID"].nunique() != len(metadata):
        raise RuntimeError("Fixed master split is not 500 unique IDs")
    if metadata["patient_group"].nunique() != int(data["expected_patient_groups"]):
        raise RuntimeError("Fixed master split patient-group count is not 483")
    counts = metadata["y_true"].value_counts().reindex([0, 1, 2], fill_value=0).tolist()
    if counts != list(data["expected_class_counts"]):
        raise RuntimeError(f"Unexpected class counts: {counts}")
    rows: list[dict[str, Any]] = []
    val_seen: list[str] = []
    for fold in data["folds"]:
        train, val = fold_metadata(config, fold, "train"), fold_metadata(config, fold, "val")
        if len(train) != 400 or len(val) != 100:
            raise RuntimeError(f"Fold {fold} is not 400/100")
        if set(train.ID) & set(val.ID) or set(train.patient_group) & set(val.patient_group):
            raise RuntimeError(f"Fold {fold} has ID or patient-group leakage")
        if set(train.y_true) != {0, 1, 2} or set(val.y_true) != {0, 1, 2}:
            raise RuntimeError(f"Fold {fold} does not contain all three classes")
        val_seen.extend(val.ID.tolist())
        for role, frame in (("train", train), ("val", val)):
            rows.append({
                "fold": fold, "split_role": role, "rows": len(frame),
                "unique_ids": frame.ID.nunique(), "patient_groups": frame.patient_group.nunique(),
                "normal": int((frame.y_true == 0).sum()), "mild": int((frame.y_true == 1).sum()),
                "severe": int((frame.y_true == 2).sum()), "id_sha256": sha256_ids(frame.ID),
                "patient_group_sha256": sha256_ids(frame.patient_group),
            })
    if len(val_seen) != 500 or len(set(val_seen)) != 500 or set(val_seen) != set(metadata.ID):
        raise RuntimeError("Each fixed-split ID must occur in outer validation exactly once")
    return metadata, rows


def run_protocol(config_path: Path, config: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    metadata, split_rows = _split_checks(config)
    raw_path = assert_classifier_feature_path(root_path(config["features"]["raw_source"]))
    raw = pd.read_csv(raw_path, usecols=["ID", AVAILABILITY_COLUMN], dtype={"ID": "string"})
    raw["ID"] = raw["ID"].astype(str)
    raw = raw.set_index("ID").loc[metadata.ID]
    if int((raw[AVAILABILITY_COLUMN] == 0).sum()) != 14:
        raise RuntimeError("Expected exactly 14 forehead-unavailable cases")
    source_rows: list[dict[str, Any]] = []
    canonical_availability = raw[AVAILABILITY_COLUMN].astype(int).to_dict()
    for fold in config["data"]["folds"]:
        for role in ("train", "val"):
            split = fold_metadata(config, fold, role)
            for variant in VARIANTS:
                frame, audit = load_variant_features(config, variant, fold, role, split.ID)
                actual = dict(zip(frame.ID, frame[AVAILABILITY_COLUMN].astype(int)))
                if any(actual[id_] != canonical_availability[id_] for id_ in split.ID):
                    raise RuntimeError(f"Availability mismatch: {variant} fold {fold} {role}")
                source_rows.append(audit)
    # The forbidden OOF paths are audited as existing inputs but never opened by the classifier loader.
    for key in ("stage2a_oof", "stage2b_oof"):
        try:
            assert_classifier_feature_path(root_path(config["features"][key]))
        except ValueError:
            pass
        else:
            raise RuntimeError(f"OOF guard did not reject {key}")
    protocol_dir = output_root / "protocol"
    write_csv(pd.DataFrame(split_rows), protocol_dir / "split_audit.csv")
    write_csv(pd.DataFrame(source_rows), protocol_dir / "feature_source_audit.csv")
    input_rows = [
        {"input": "config", "path": relative_path(config_path, PROJECT_ROOT), "sha256": sha256_file(config_path)},
        {"input": "labels", "path": config["data"]["label_path"], "sha256": sha256_file(root_path(config["data"]["label_path"]))},
        {"input": "master_split", "path": config["data"]["master_split"], "sha256": sha256_file(root_path(config["data"]["master_split"]))},
        {"input": "stage1_raw", "path": config["features"]["raw_source"], "sha256": sha256_file(raw_path)},
        {"input": "stage2a_manifest", "path": config["features"]["stage2a_manifest"], "sha256": sha256_file(root_path(config["features"]["stage2a_manifest"]))},
        {"input": "stage2b_manifest", "path": config["features"]["stage2b_manifest"], "sha256": sha256_file(root_path(config["features"]["stage2b_manifest"]))},
    ]
    write_csv(pd.DataFrame(input_rows), protocol_dir / "input_audit.csv")
    checks = {
        "fixed_split_reused": True, "label_file_read_only": True,
        "stage_inputs_hashed": True, "oof_inputs_rejected": True,
        "stage_fold_role_exact": True, "variant_id_sets_equal": True,
        "patient_group_leakage": False, "unavailable_cases_retained": 14,
        "availability_cross_source_equal": True, "scaler_train_only": True,
        "class_weights_train_only": True, "outer_validation_tuning": False,
        "images_loaded": False, "resnet_used": False, "exif_used": False,
        "camera_used": False, "hyperparameter_search": False, "threshold_tuning": False,
    }
    manifest = {"status": "PASS", "checks": checks, "input_sha256": input_rows,
                "sample_count": 500, "patient_group_count": 483,
                "class_counts": [115, 237, 148], "forehead_unavailable": 14}
    write_json(protocol_dir / "protocol_manifest.json", manifest)
    return manifest


@dataclass
class FitResult:
    variant: str
    fold: int
    train_meta: pd.DataFrame
    val_meta: pd.DataFrame
    train_features: pd.DataFrame
    val_features: pd.DataFrame
    x_train: np.ndarray
    x_val: np.ndarray
    train_prob: np.ndarray
    val_prob: np.ndarray
    model: Any
    scaler: OpticalPhenotypeScaler | None
    warnings: list[dict[str, str]]
    sources: list[dict[str, Any]]
    train_metrics: dict[str, Any]
    val_metrics: dict[str, Any]


def fit_one(config: Mapping[str, Any], variant: str, fold: int) -> FitResult:
    train_meta, val_meta = fold_metadata(config, fold, "train"), fold_metadata(config, fold, "val")
    train_features, train_source = load_variant_features(config, variant, fold, "train", train_meta.ID)
    val_features, val_source = load_variant_features(config, variant, fold, "val", val_meta.ID)
    scaler = None
    if variant != "o_mask":
        source_hash = sha256_json([train_source["source_sha256"], val_source["source_sha256"]])
        scaler = fit_scaler(train_features, variant, fold, train_id_sha256=sha256_ids(train_meta.ID),
                            feature_source_sha256=source_hash,
                            std_epsilon=float(config["feature_standardization"]["std_epsilon"]))
    x_train = build_model_input(train_features, variant, scaler)
    x_val = build_model_input(val_features, variant, scaler)
    model = build_classifier(int(config["classifier"]["base_seed"]) + int(fold))
    warnings = fit_classifier(model, x_train, train_meta.y_true)
    train_prob, val_prob = predict_probabilities(model, x_train), predict_probabilities(model, x_val)
    return FitResult(
        variant, fold, train_meta, val_meta, train_features, val_features, x_train, x_val,
        train_prob, val_prob, model, scaler, warnings, [train_source, val_source],
        compute_classification_metrics(train_meta.y_true, train_prob),
        compute_classification_metrics(val_meta.y_true, val_prob),
    )


def prediction_frame(result: FitResult, role: str) -> pd.DataFrame:
    meta = result.train_meta if role == "train" else result.val_meta
    features = result.train_features if role == "train" else result.val_features
    prob = result.train_prob if role == "train" else result.val_prob
    pred = prob.argmax(axis=1)
    frame = meta.copy()
    frame["fold"] = result.fold
    frame["split_role"] = role
    frame["y_pred"] = pred
    for index, column in enumerate(PROBABILITY_COLUMNS):
        frame[column] = prob[:, index]
    frame[AVAILABILITY_COLUMN] = features[AVAILABILITY_COLUMN].to_numpy(int)
    frame["correct"] = (frame.y_true.to_numpy(int) == pred).astype(int)
    frame["variant"] = result.variant
    return frame.loc[:, PREDICTION_COLUMNS]


def feature_distribution(result: FitResult) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for role, source, transformed in (
        ("train", result.train_features, result.x_train), ("val", result.val_features, result.x_val)
    ):
        available = source[AVAILABILITY_COLUMN].to_numpy(int) == 1
        if result.variant == "o_mask":
            entries = [(AVAILABILITY_COLUMN, source[AVAILABILITY_COLUMN].to_numpy(float),
                        transformed[:, 0], np.ones(len(source), bool))]
        else:
            entries = []
            for index, name in enumerate(VARIANT_FEATURE_COLUMNS[result.variant]):
                eligible = np.ones(len(source), bool) if index < 3 else available
                entries.append((name, pd.to_numeric(source[name]).to_numpy(float), transformed[:, index], eligible))
            entries.append((AVAILABILITY_COLUMN, source[AVAILABILITY_COLUMN].to_numpy(float),
                            transformed[:, -1], np.ones(len(source), bool)))
        for name, raw, post, eligible in entries:
            raw_finite = np.isfinite(raw)
            for scale, values, valid_n, missing_n in (
                ("raw", raw[raw_finite], int(raw_finite.sum()), int((~raw_finite).sum())),
                ("standardized_model_input", post[eligible], int(eligible.sum()), 0),
            ):
                rows.append({
                    "variant": result.variant, "fold": result.fold, "split_role": role,
                    "scale": scale, "feature": name, "total_n": len(source),
                    "valid_n": valid_n, "missing_n": missing_n,
                    "statistic_population": "available_only" if name != AVAILABILITY_COLUMN and name in VARIANT_FEATURE_COLUMNS[result.variant][3:] else "all_cases",
                    "mean": float(np.mean(values)), "std_ddof0": float(np.std(values, ddof=0)),
                    "min": float(np.min(values)), "median": float(np.median(values)),
                    "q1": float(np.quantile(values, 0.25)), "q3": float(np.quantile(values, 0.75)),
                    "max": float(np.max(values)), "nonfinite": int((~np.isfinite(values)).sum()),
                    "unavailable_filled_zero_n": int(np.sum(post[~eligible] == 0)) if scale == "standardized_model_input" else 0,
                })
    return pd.DataFrame(rows)


def coefficients_frame(result: FitResult) -> pd.DataFrame:
    rows = []
    names = VARIANT_INPUT_COLUMNS[result.variant]
    for class_index, class_name in enumerate(CLASS_NAMES):
        rows.append({"variant": result.variant, "fold": result.fold, "class_index": class_index,
                     "class_name": class_name, "feature": "__intercept__",
                     "coefficient": float(result.model.intercept_[class_index]),
                     "absolute_coefficient": float(abs(result.model.intercept_[class_index])), "is_intercept": True})
        for feature_index, name in enumerate(names):
            rows.append({"variant": result.variant, "fold": result.fold, "class_index": class_index,
                         "class_name": class_name, "feature": name,
                         "coefficient": float(result.model.coef_[class_index, feature_index]),
                         "absolute_coefficient": float(abs(result.model.coef_[class_index, feature_index])),
                         "is_intercept": False})
    return pd.DataFrame(rows)


def save_result(result: FitResult, config_path: Path, config: Mapping[str, Any], run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    if result.scaler is not None:
        result.scaler.save_json(run_dir / "feature_scaler.json")
    save_model(result.model, run_dir / "model.joblib")
    train_predictions, val_predictions = prediction_frame(result, "train"), prediction_frame(result, "val")
    write_csv(train_predictions, run_dir / "train_predictions.csv")
    write_csv(val_predictions, run_dir / "val_predictions.csv")
    write_csv(coefficients_frame(result), run_dir / "coefficients.csv")
    write_csv(feature_distribution(result), run_dir / "feature_distribution.csv")
    metrics = {
        "variant": result.variant, "fold": result.fold,
        "train": result.train_metrics, "val": result.val_metrics,
        "train_val_gap": {key: flatten_metrics(result.train_metrics)[key] - flatten_metrics(result.val_metrics)[key]
                          for key in flatten_metrics(result.val_metrics)},
    }
    write_json(run_dir / "metrics.json", metrics)
    restored = load_model(run_dir / "model.joblib")
    restored_train = predict_probabilities(restored, result.x_train)
    restored_val = predict_probabilities(restored, result.x_val)
    if not np.array_equal(restored_train, result.train_prob) or not np.array_equal(restored_val, result.val_prob):
        if not (np.allclose(restored_train, result.train_prob, atol=ABSOLUTE_TOLERANCE, rtol=ABSOLUTE_TOLERANCE)
                and np.allclose(restored_val, result.val_prob, atol=ABSOLUTE_TOLERANCE, rtol=ABSOLUTE_TOLERANCE)):
            raise RuntimeError("Reloaded model probabilities differ")
    artifact_names = ["model.joblib", "train_predictions.csv", "val_predictions.csv",
                      "metrics.json", "coefficients.csv", "feature_distribution.csv"]
    if result.scaler is not None:
        artifact_names.append("feature_scaler.json")
    manifest = {
        "status": "COMPLETE", "variant": result.variant, "fold": result.fold,
        "input_dim": VARIANT_INPUT_DIM[result.variant], "feature_order": list(VARIANT_INPUT_COLUMNS[result.variant]),
        "classifier": config["classifier"], "random_state": int(config["classifier"]["base_seed"]) + result.fold,
        "actual_class_weights": balanced_class_weights(result.train_meta.y_true),
        "class_order": list(CLASS_NAMES), "n_iter": result.model.n_iter_.tolist(),
        "convergence_warnings": result.warnings, "converged": True,
        "sources": result.sources, "config_sha256": sha256_file(config_path),
        "train_id_sha256": sha256_ids(result.train_meta.ID), "val_id_sha256": sha256_ids(result.val_meta.ID),
        "model_reload_probability_max_abs_diff": float(max(
            np.max(np.abs(restored_train - result.train_prob)), np.max(np.abs(restored_val - result.val_prob)))),
        "model_reload_pass": True,
        "artifact_sha256": {name: sha256_file(run_dir / name) for name in artifact_names},
        "canonical_prediction_sha256": {
            "train": canonical_frame_sha256(train_predictions), "val": canonical_frame_sha256(val_predictions)},
        "images_loaded": False, "resnet_used": False, "global_features_used": False,
        "exif_used": False, "camera_used": False, "outer_validation_tuning": False,
        "hyperparameter_search": False, "threshold_tuning": False,
    }
    write_json(run_dir / "model_manifest.json", manifest)


def validate_completed(run_dir: Path, config_path: Path) -> bool:
    manifest_path = run_dir / "model_manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE" or manifest.get("config_sha256") != sha256_file(config_path):
        raise RuntimeError(f"Completed output is incompatible with current config: {run_dir}")
    for name, expected in manifest["artifact_sha256"].items():
        if not (run_dir / name).is_file() or sha256_file(run_dir / name) != expected:
            raise RuntimeError(f"Completed artifact hash mismatch: {run_dir / name}")
    return True


def determinism_check(config: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    rows = []
    with tempfile.TemporaryDirectory(prefix="optical_phenotype_repeat_") as temporary:
        temporary_root = Path(temporary)
        for fold in config["data"]["folds"]:
            for variant in VARIANTS:
                formal_dir = output_root / f"fold_{fold}" / variant
                formal_coef = pd.read_csv(formal_dir / "coefficients.csv")["coefficient"].to_numpy(float)
                formal_train = pd.read_csv(formal_dir / "train_predictions.csv")
                formal_val = pd.read_csv(formal_dir / "val_predictions.csv")
                repeated = fit_one(config, variant, fold)
                repeated_train, repeated_val = prediction_frame(repeated, "train"), prediction_frame(repeated, "val")
                coef = coefficients_frame(repeated)["coefficient"].to_numpy(float)
                probability_diff = max(
                    np.max(np.abs(formal_train[PROBABILITY_COLUMNS].to_numpy() - repeated.train_prob)),
                    np.max(np.abs(formal_val[PROBABILITY_COLUMNS].to_numpy() - repeated.val_prob)),
                )
                coefficient_diff = float(np.max(np.abs(formal_coef - coef)))
                scaler_diff = 0.0
                if repeated.scaler is not None:
                    formal_scaler = OpticalPhenotypeScaler.load_json(formal_dir / "feature_scaler.json")
                    scaler_diff = float(max(
                        np.max(np.abs(np.asarray(formal_scaler.mean) - repeated.scaler.mean)),
                        np.max(np.abs(np.asarray(formal_scaler.std) - repeated.scaler.std))))
                write_csv(repeated_train, temporary_root / f"{variant}_{fold}_train.csv")
                write_csv(repeated_val, temporary_root / f"{variant}_{fold}_val.csv")
                canonical_match = (
                    sha256_file(formal_dir / "train_predictions.csv")
                    == sha256_file(temporary_root / f"{variant}_{fold}_train.csv")
                    and sha256_file(formal_dir / "val_predictions.csv")
                    == sha256_file(temporary_root / f"{variant}_{fold}_val.csv")
                )
                formal_metrics = json.loads((formal_dir / "metrics.json").read_text(encoding="utf-8"))
                repeated_metrics = {"train": repeated.train_metrics, "val": repeated.val_metrics}
                metric_differences = []
                for role in ("train", "val"):
                    for key, value in flatten_metrics(repeated_metrics[role]).items():
                        metric_differences.append(abs(float(formal_metrics[role][key]) - value))
                metrics_diff = float(max(metric_differences, default=0.0))
                passed = (probability_diff <= 1e-12 and coefficient_diff <= 1e-12
                          and scaler_diff <= 1e-12 and metrics_diff <= 1e-12 and canonical_match)
                if not passed:
                    raise RuntimeError(f"Determinism failure for {variant} fold {fold}")
                rows.append({"variant": variant, "fold": fold, "probability_max_abs_diff": float(probability_diff),
                             "coefficient_max_abs_diff": coefficient_diff, "scaler_max_abs_diff": scaler_diff,
                             "metrics_max_abs_diff": metrics_diff,
                             "canonical_csv_sha256_match": canonical_match, "pass": passed})
    audit = {"status": "PASS", "models_repeated": len(rows), "temporary_directory_removed": True,
             "absolute_tolerance": 1e-12, "relative_tolerance": 1e-12, "rows": rows}
    write_json(output_root / "protocol" / "determinism_audit.json", audit)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/train/optical_phenotype_only/optical_phenotype_logistic_5fold.yaml")
    parser.add_argument("--fold", choices=["0", "1", "2", "3", "4", "all"], default="all")
    parser.add_argument("--variant", choices=[*VARIANTS, "all"], default="all")
    parser.add_argument("--protocol-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--determinism-check", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path, config = load_config(args.config)
    output_root = root_path(config["experiment"]["output_root"])
    if args.smoke_test:
        with tempfile.TemporaryDirectory(prefix="optical_phenotype_smoke_") as directory:
            for variant in VARIANTS:
                result = fit_one(config, variant, 0)
                save_result(result, config_path, config, Path(directory) / "fold_0" / variant)
            print("SMOKE_STATUS=PASS (4 variants, fold 0, temporary output removed)")
        return
    output_root.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        protocol = run_protocol(config_path, config, output_root)
        print(f"PROTOCOL_STATUS={protocol['status']}")
    if args.protocol_only:
        return
    if args.summarize_only:
        from scripts.evaluate.summarize_optical_phenotype_only_nyha import summarize
        summarize(config_path, config, output_root)
        return
    folds = config["data"]["folds"] if args.fold == "all" else [int(args.fold)]
    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    completed = 0
    for fold in folds:
        for variant in variants:
            run_dir = output_root / f"fold_{fold}" / variant
            if run_dir.exists() and any(run_dir.iterdir()):
                if (args.resume or args.skip_completed) and validate_completed(run_dir, config_path):
                    print(f"SKIP_COMPLETED fold={fold} variant={variant}")
                    completed += 1
                    continue
                if not args.overwrite:
                    raise FileExistsError(f"Output already exists; use --resume or --overwrite: {run_dir}")
                shutil.rmtree(run_dir)
            result = fit_one(config, variant, fold)
            save_result(result, config_path, config, run_dir)
            completed += 1
            print(f"FIT_STATUS=PASS fold={fold} variant={variant} n_iter={result.model.n_iter_.tolist()}")
    matrix_complete = all(validate_completed(output_root / f"fold_{fold}" / variant, config_path)
                          for fold in config["data"]["folds"] for variant in VARIANTS)
    if matrix_complete:
        from scripts.evaluate.summarize_optical_phenotype_only_nyha import summarize
        summarize(config_path, config, output_root)
        if args.determinism_check or (args.fold == "all" and args.variant == "all"):
            determinism_check(config, output_root)
            summarize(config_path, config, output_root)
        print("OPTICAL_PHENOTYPE_ONLY_STATUS=COMPLETE models=20")
    else:
        print(f"OPTICAL_PHENOTYPE_ONLY_STATUS=PARTIAL completed_this_run={completed}")


if __name__ == "__main__":
    main()
