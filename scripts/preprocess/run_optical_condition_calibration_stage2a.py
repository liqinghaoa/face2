"""Run Stage 2A label-free EXIF/device-conditioned optical calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.optical_condition_calibration import (  # noqa: E402
    ALL_TARGETS,
    CHEEK_TARGETS,
    CONDITION_NAMES,
    DESIGN_FEATURE_NAMES,
    EXPECTED_CAMERAS,
    FOREHEAD_CHEEK_TARGETS,
    REFERENCE_CAMERA,
    RidgeModel,
    build_design_matrix,
    calibrate_values,
    camera_difference_metrics,
    coefficient_records,
    coefficient_stability,
    fit_condition_scaler,
    fit_error_metrics,
    fit_ridge,
    spearman_rho,
    transform_conditions,
    validate_camera_values,
)

TASK = "optical_condition_calibration_stage2a"
TASK_TITLE = "第二阶段A：EXIF与设备条件化的区域光学表型校准"
REPORT_MARKER = f"<!-- task: {TASK} -->"
DEFAULT_CONFIG = PROJECT_ROOT / "config/preprocess/optical_condition_calibration_stage2a.yaml"
INPUT_COLUMNS = (
    "ID", "camera_id", "ExposureTime", "FNumber", "ISOSpeedRatings",
    "relative_optical_exposure", "log2_iso_condition", "forehead_available",
    *ALL_TARGETS,
)
IMPLEMENTATION_FILES = (
    "utils/optical_condition_calibration.py",
    "scripts/preprocess/run_optical_condition_calibration_stage2a.py",
    "config/preprocess/optical_condition_calibration_stage2a.yaml",
    "tests/test_optical_condition_calibration_stage2a.py",
    "tests/test_optical_calibration_fivefold_protocol.py",
)


class CalibrationFailure(RuntimeError):
    def __init__(self, stage: str, errors: Sequence[str]):
        self.stage = str(stage)
        self.errors = [str(value) for value in errors]
        super().__init__(f"{self.stage}: {' | '.join(self.errors)}")


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


def project_relative(path: Path, root: Path = PROJECT_ROOT) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def resolve_under_project(value: str | Path, root: Path) -> Path:
    candidate = Path(value).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Path outside project root: {resolved}") from exc
    return resolved


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
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
        path, index=False, encoding="utf-8-sig", na_rep="", float_format="%.17g", lineterminator="\n"
    )


def get_git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unavailable"


def load_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=TASK_TITLE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--fold", default="all", choices=["all", "0", "1", "2", "3", "4"])
    parser.add_argument("--protocol-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true", default=None)
    cli = parser.parse_args(argv)
    config_path = cli.config.resolve() if cli.config.is_absolute() else (PROJECT_ROOT / cli.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    root_value = config.get("project_root", ".")
    root = PROJECT_ROOT if root_value in (None, "", ".") else Path(root_value).resolve()
    values = dict(config)
    values.update({"project_root": root, "config_path": config_path})
    for key in (
        "first_stage_csv", "first_stage_schema", "first_stage_manifest", "split_dir",
        "experiment_output_dir", "report_output_dir",
    ):
        values[key] = resolve_under_project(values[key], root)
    values["fold"] = cli.fold
    values["protocol_only"] = bool(cli.protocol_only)
    values["summarize_only"] = bool(cli.summarize_only)
    if cli.overwrite is not None:
        values["overwrite"] = bool(cli.overwrite)
    args = argparse.Namespace(**values)
    validate_config(args)
    return args


def validate_config(args: argparse.Namespace) -> None:
    errors: list[str] = []
    if str(args.task) != TASK:
        errors.append("task name differs from fixed Stage 2A task")
    if int(args.n_folds) != 5:
        errors.append("n_folds must be 5")
    if float(args.alpha) != 1.0 or not bool(args.fit_intercept) or bool(args.penalize_intercept):
        errors.append("Ridge must use alpha=1, unpenalized intercept")
    if float(args.std_epsilon) != 1.0e-8:
        errors.append("std_epsilon must be 1e-8")
    if str(args.reference_camera) != REFERENCE_CAMERA:
        errors.append("reference camera differs from HONOR/BVL-AN00")
    if dict(args.camera_mapping) != {EXPECTED_CAMERAS[0]: 0, EXPECTED_CAMERAS[1]: 1}:
        errors.append("camera mapping differs from fixed mapping")
    if tuple(args.condition_names) != CONDITION_NAMES:
        errors.append("condition field order differs from fixed order")
    if tuple(args.design_feature_names) != DESIGN_FEATURE_NAMES:
        errors.append("design matrix field order differs from fixed order")
    if tuple(args.cheek_targets) != CHEEK_TARGETS or tuple(args.forehead_cheek_targets) != FOREHEAD_CHEEK_TARGETS:
        errors.append("target field order differs from fixed order")
    expected_exp = (args.project_root / "experiments/optical_condition_calibration_stage2a").resolve()
    expected_report = (args.project_root / "reports/optical_condition_calibration_stage2a").resolve()
    if args.experiment_output_dir != expected_exp or args.report_output_dir != expected_report:
        errors.append("output directories differ from fixed Stage 2A directories")
    if args.protocol_only and args.summarize_only:
        errors.append("--protocol-only and --summarize-only are mutually exclusive")
    if args.summarize_only and args.fold != "all":
        errors.append("--summarize-only requires --fold all")
    if errors:
        raise CalibrationFailure("config", errors)


def load_first_stage(args: argparse.Namespace) -> pd.DataFrame:
    for path in (args.first_stage_csv, args.first_stage_schema, args.first_stage_manifest):
        if not path.is_file():
            raise CalibrationFailure("preflight", [f"Missing first-stage input: {path}"])
    schema = json.loads(args.first_stage_schema.read_text(encoding="utf-8"))
    manifest = json.loads(args.first_stage_manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE" or manifest.get("clinical_labels_read") is not False:
        raise CalibrationFailure("preflight", ["First-stage manifest is not a completed label-free extraction"])
    if manifest.get("output_files", {}).get("main", {}).get("sha256") != sha256_file(args.first_stage_csv):
        raise CalibrationFailure("preflight", ["First-stage CSV SHA256 does not match extraction manifest"])
    required_schema_targets = set(schema.get("derived_observation_columns", []))
    if not set(ALL_TARGETS).issubset(required_schema_targets):
        raise CalibrationFailure("preflight", ["First-stage feature schema does not contain all six fixed targets"])
    header = pd.read_csv(args.first_stage_csv, nrows=0, encoding="utf-8-sig").columns.tolist()
    missing = sorted(set(INPUT_COLUMNS).difference(header))
    if missing:
        raise CalibrationFailure("preflight", [f"Missing first-stage columns: {missing}"])
    frame = pd.read_csv(
        args.first_stage_csv,
        usecols=list(INPUT_COLUMNS),
        dtype={"ID": str, "camera_id": str},
        encoding="utf-8-sig",
    )
    frame["ID"] = frame["ID"].astype(str).str.strip()
    errors: list[str] = []
    if len(frame) != 500 or frame["ID"].nunique() != 500 or frame["ID"].str.casefold().nunique() != 500:
        errors.append("First-stage table must contain 500 exact unique IDs")
    try:
        validate_camera_values(frame["camera_id"], require_both=True)
    except ValueError as exc:
        errors.append(str(exc))
    numeric = frame[["ExposureTime", "FNumber", "ISOSpeedRatings", *CONDITION_NAMES]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(float)).all() or (numeric[["ExposureTime", "FNumber", "ISOSpeedRatings"]] <= 0).any().any():
        errors.append("Raw EXIF and derived conditions must be finite; raw EXIF must be positive")
    expected_exposure = np.log2(numeric["ExposureTime"] / numeric["FNumber"] ** 2)
    expected_iso = np.log2(numeric["ISOSpeedRatings"] / 100.0)
    if not np.allclose(expected_exposure, numeric["relative_optical_exposure"], rtol=0, atol=1e-12):
        errors.append("relative_optical_exposure does not match raw EXIF")
    if not np.allclose(expected_iso, numeric["log2_iso_condition"], rtol=0, atol=1e-12):
        errors.append("log2_iso_condition does not match raw EXIF")
    available = pd.to_numeric(frame["forehead_available"], errors="coerce")
    if not available.isin([0, 1]).all():
        errors.append("forehead_available must contain only 0/1")
    if not np.isfinite(frame[list(CHEEK_TARGETS)].to_numpy(float)).all():
        errors.append("All cheek targets must be finite")
    forehead = frame[list(FOREHEAD_CHEEK_TARGETS)]
    if not np.isfinite(forehead.loc[available == 1].to_numpy(float)).all():
        errors.append("Available forehead targets must be finite")
    if not forehead.loc[available == 0].isna().all().all():
        errors.append("Unavailable forehead targets must all be NaN")
    if errors:
        raise CalibrationFailure("preflight", errors)
    return frame.sort_values("ID", kind="stable").reset_index(drop=True)


def audit_splits(args: argparse.Namespace, first_stage: pd.DataFrame) -> dict[str, Any]:
    split_files: list[Path] = []
    splits: dict[int, dict[str, pd.DataFrame]] = {}
    audit_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    first_ids = set(first_stage["ID"].astype(str))
    all_validation_ids: list[str] = []
    per_fold_hashes: dict[str, Any] = {}
    for fold in range(5):
        fold_frames: dict[str, pd.DataFrame] = {}
        fold_hashes: dict[str, str] = {}
        for role, pattern in (("train", args.train_csv_pattern), ("val", args.val_csv_pattern)):
            path = args.split_dir / str(pattern).format(fold=fold)
            split_files.append(path)
            if not path.is_file():
                errors.append(f"Missing split file: {path}")
                continue
            try:
                frame = pd.read_csv(
                    path, usecols=["ID", "fold"], dtype={"ID": str}, encoding="utf-8-sig"
                )
            except ValueError as exc:
                errors.append(f"Split missing or invalid ID/fold columns: {path}: {exc}")
                continue
            frame["ID"] = frame["ID"].astype(str).str.strip()
            frame["fold"] = pd.to_numeric(frame["fold"], errors="coerce")
            fold_frames[role] = frame
            fold_hashes[role] = sha256_file(path)
            audit_rows.append({
                "fold": fold, "split_role": role, "n_rows": len(frame),
                "unique_id_n": frame["ID"].nunique(), "file_sha256": fold_hashes[role],
                "id_sha256": sha256_ids(frame["ID"].tolist()),
            })
        if set(fold_frames) != {"train", "val"}:
            continue
        train, val = fold_frames["train"], fold_frames["val"]
        train_ids, val_ids = set(train["ID"]), set(val["ID"])
        if len(train) != 400 or train["ID"].nunique() != 400:
            errors.append(f"Fold {fold} train is not 400 unique IDs")
        if len(val) != 100 or val["ID"].nunique() != 100:
            errors.append(f"Fold {fold} val is not 100 unique IDs")
        if train_ids & val_ids:
            errors.append(f"Fold {fold} train/val ID overlap")
        if train_ids | val_ids != first_ids:
            errors.append(f"Fold {fold} train/val union differs from first-stage IDs")
        if not (val["fold"] == fold).all() or (train["fold"] == fold).any():
            errors.append(f"Fold {fold} fold assignments are inconsistent")
        train_camera = first_stage.set_index("ID").loc[sorted(train_ids), "camera_id"]
        try:
            validate_camera_values(train_camera, require_both=True)
        except ValueError as exc:
            errors.append(f"Fold {fold}: {exc}")
        forehead_n = int(first_stage.set_index("ID").loc[sorted(train_ids), "forehead_available"].sum())
        if forehead_n <= len(DESIGN_FEATURE_NAMES) + 1:
            errors.append(f"Fold {fold} has insufficient available forehead training rows: {forehead_n}")
        all_validation_ids.extend(val["ID"].tolist())
        per_fold_hashes[str(fold)] = {
            "train_file_sha256": fold_hashes["train"],
            "val_file_sha256": fold_hashes["val"],
            "train_id_sha256": sha256_ids(train["ID"].tolist()),
            "val_id_sha256": sha256_ids(val["ID"].tolist()),
        }
        splits[fold] = {"train": train, "val": val}
    if len(all_validation_ids) != 500 or len(set(all_validation_ids)) != 500 or set(all_validation_ids) != first_ids:
        errors.append("Five validation folds do not form an exact one-time 500-ID OOF partition")
    if errors:
        raise CalibrationFailure("split_protocol", errors)
    return {
        "splits": splits,
        "audit": pd.DataFrame(audit_rows).sort_values(["fold", "split_role"], kind="stable"),
        "split_files": split_files,
        "split_sha256": combined_file_sha256(split_files),
        "per_fold_hashes": per_fold_hashes,
        "status": "PASS",
    }


def verify_owned_output(path: Path, report: bool = False) -> bool:
    if not path.exists() or not any(path.iterdir()):
        return True
    owner = path / ".stage2a_owner.json"
    if owner.is_file():
        try:
            return json.loads(owner.read_text(encoding="utf-8")).get("task") == TASK
        except Exception:
            return False
    if report:
        report_path = path / "optical_condition_calibration_stage2a_report.md"
        return report_path.is_file() and REPORT_MARKER in report_path.read_text(encoding="utf-8")
    manifest = path / "summary/run_manifest.json"
    try:
        return manifest.is_file() and json.loads(manifest.read_text(encoding="utf-8")).get("task") == TASK
    except Exception:
        return False


def prepare_outputs(args: argparse.Namespace) -> None:
    roots = (args.experiment_output_dir, args.report_output_dir)
    nonempty = [path for path in roots if path.exists() and any(path.iterdir())]
    if nonempty and not bool(args.overwrite) and not args.summarize_only:
        raise CalibrationFailure("output_safety", [f"Non-empty output directory: {path}" for path in nonempty])
    if nonempty and not args.summarize_only:
        ownership = [
            verify_owned_output(path, report=(path == args.report_output_dir))
            for path in nonempty
        ]
        if not all(ownership):
            raise CalibrationFailure("output_safety", ["Existing outputs are not verified as Stage 2A artifacts"])
        for path in roots:
            if path.exists():
                shutil.rmtree(path)
    for path in roots:
        path.mkdir(parents=True, exist_ok=True)
        write_json({"task": TASK}, path / ".stage2a_owner.json")


def select_rows(first_stage: pd.DataFrame, ids: Sequence[str]) -> pd.DataFrame:
    indexed = first_stage.set_index("ID", drop=False)
    return indexed.loc[sorted(str(value) for value in ids)].reset_index(drop=True)


def model_payload(
    model: RidgeModel,
    fold: int,
    group: str,
    reference_mean: np.ndarray,
    train_ids: Sequence[str],
    fitting_ids: Sequence[str],
    target_available_n: int,
    scaler: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "method": "fixed multi-output Ridge acquisition-conditioned optical calibration",
        "fold": int(fold),
        "target_group": group,
        **model.to_dict(),
        "reference_camera": REFERENCE_CAMERA,
        "camera_mapping": {EXPECTED_CAMERAS[0]: 0, EXPECTED_CAMERAS[1]: 1},
        "reference_mean_by_target": {
            target: float(reference_mean[index]) for index, target in enumerate(model.target_names)
        },
        "train_n": int(len(train_ids)),
        "target_available_n": int(target_available_n),
        "condition_scaler_path": f"fold_{fold}/condition_scaler.json",
        "condition_scaler": scaler,
        "training_id_sha256": sha256_ids(train_ids),
        "fitting_id_sha256": sha256_ids(fitting_ids),
        "split_sha256": context["split_sha256"],
        "first_stage_input_sha256": context["first_stage_sha256"],
        "config_sha256": context["config_sha256"],
        "implementation": "NumPy closed-form Ridge",
        "numpy_version": np.__version__,
        "git_commit": context["git_commit"],
    }


def transform_feature_frame(
    frame: pd.DataFrame,
    fold: int,
    role: str,
    scaler: Mapping[str, Any],
    cheek_model: RidgeModel,
    forehead_model: RidgeModel,
    cheek_reference: np.ndarray,
    forehead_reference: np.ndarray,
) -> pd.DataFrame:
    transformed = transform_conditions(frame, scaler)
    x = build_design_matrix(transformed)
    output = transformed.loc[:, [
        "ID", "camera_id", "forehead_available", *CONDITION_NAMES,
        "z_relative_optical_exposure", "z_log2_iso_condition",
    ]].copy()
    output.insert(1, "fold", int(fold))
    output.insert(2, "split_role", role)
    cheek_predictions = cheek_model.predict(x)
    for index, target in enumerate(CHEEK_TARGETS):
        raw = pd.to_numeric(transformed[target], errors="coerce").to_numpy(float)
        predicted = cheek_predictions[:, index]
        residual, calibrated = calibrate_values(raw, predicted, cheek_reference[index])
        output[f"raw_{target}"] = raw
        output[f"predicted_acquisition_{target}"] = predicted
        output[f"residual_{target}"] = residual
        output[f"calibrated_{target}"] = calibrated
    available = transformed["forehead_available"].astype(int).to_numpy() == 1
    forehead_predictions = forehead_model.predict(x[available])
    for index, target in enumerate(FOREHEAD_CHEEK_TARGETS):
        raw = pd.to_numeric(transformed[target], errors="coerce").to_numpy(float)
        predicted = np.full(len(transformed), np.nan, dtype=np.float64)
        predicted[available] = forehead_predictions[:, index]
        residual = np.full(len(transformed), np.nan, dtype=np.float64)
        calibrated = np.full(len(transformed), np.nan, dtype=np.float64)
        residual[available], calibrated[available] = calibrate_values(
            raw[available], predicted[available], forehead_reference[index]
        )
        output[f"raw_{target}"] = raw
        output[f"predicted_acquisition_{target}"] = predicted
        output[f"residual_{target}"] = residual
        output[f"calibrated_{target}"] = calibrated
    return output.sort_values("ID", kind="stable").reset_index(drop=True)


def build_diagnostics(frame: pd.DataFrame, fold: int, role: str) -> dict[str, pd.DataFrame]:
    fit_rows: list[dict[str, Any]] = []
    corr_rows: list[dict[str, Any]] = []
    camera_rows: list[dict[str, Any]] = []
    variance_rows: list[dict[str, Any]] = []
    scopes = [("overall", "ALL", frame)]
    scopes.extend(
        ("camera_id", camera, frame.loc[frame["camera_id"] == camera]) for camera in EXPECTED_CAMERAS
    )
    for target in ALL_TARGETS:
        raw_name = f"raw_{target}"
        predicted_name = f"predicted_acquisition_{target}"
        calibrated_name = f"calibrated_{target}"
        fit_rows.append({
            "fold": fold, "split_role": role, "target": target,
            **fit_error_metrics(frame[raw_name], frame[predicted_name]),
        })
        for representation in ("raw", "residual", "calibrated"):
            column = f"{representation}_{target}"
            for scope, camera, subset in scopes:
                for condition in CONDITION_NAMES:
                    valid_n, rho = spearman_rho(subset[column], subset[condition])
                    corr_rows.append({
                        "fold": fold, "split_role": role, "target": target,
                        "representation": representation, "scope": scope, "camera_id": camera,
                        "condition": condition, "valid_n": valid_n, "spearman_rho": rho,
                    })
        for representation in ("raw", "calibrated"):
            camera_rows.append({
                "fold": fold, "split_role": role, "target": target,
                "representation": representation,
                **camera_difference_metrics(frame, f"{representation}_{target}"),
            })
        raw = pd.to_numeric(frame[raw_name], errors="coerce").dropna().to_numpy(float)
        calibrated = pd.to_numeric(frame[calibrated_name], errors="coerce").dropna().to_numpy(float)
        raw_variance = float(np.var(raw, ddof=0)) if len(raw) else math.nan
        calibrated_variance = float(np.var(calibrated, ddof=0)) if len(calibrated) else math.nan
        variance_rows.append({
            "fold": fold, "split_role": role, "target": target,
            "valid_n": int(len(raw)), "raw_variance": raw_variance,
            "calibrated_variance": calibrated_variance,
            "variance_retention": calibrated_variance / raw_variance if raw_variance > 0 else math.nan,
        })
    return {
        "fit": pd.DataFrame(fit_rows),
        "correlation": pd.DataFrame(corr_rows),
        "camera": pd.DataFrame(camera_rows),
        "variance": pd.DataFrame(variance_rows),
    }


def combined_diagnostics(diagnostics: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for family, frame in diagnostics.items():
        item = frame.copy()
        item.insert(0, "metric_family", family)
        frames.append(item)
    return pd.concat(frames, ignore_index=True, sort=False)


def run_fold(
    fold: int,
    first_stage: pd.DataFrame,
    split_info: Mapping[str, Any],
    output_root: Path,
    context: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    fold_dir = output_root / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    train_split = split_info["splits"][fold]["train"]
    val_split = split_info["splits"][fold]["val"]
    train = select_rows(first_stage, train_split["ID"].tolist())
    val = select_rows(first_stage, val_split["ID"].tolist())
    scaler = fit_condition_scaler(train, float(args.std_epsilon))
    scaler = {"fold": fold, **scaler}
    write_json(scaler, fold_dir / "condition_scaler.json")
    train_scaled = transform_conditions(train, scaler)
    x_train = build_design_matrix(train_scaled)
    cheek_y = train_scaled.loc[:, list(CHEEK_TARGETS)].to_numpy(float)
    cheek_model = fit_ridge(x_train, cheek_y, CHEEK_TARGETS, alpha=1.0)
    available = train_scaled["forehead_available"].astype(int).to_numpy() == 1
    forehead_y = train_scaled.loc[available, list(FOREHEAD_CHEEK_TARGETS)].to_numpy(float)
    forehead_model = fit_ridge(x_train[available], forehead_y, FOREHEAD_CHEEK_TARGETS, alpha=1.0)
    cheek_reference = np.mean(cheek_y, axis=0)
    forehead_reference = np.mean(forehead_y, axis=0)
    cheek_payload = model_payload(
        cheek_model, fold, "cheek", cheek_reference, train["ID"].tolist(),
        train["ID"].tolist(), len(train), scaler, context
    )
    forehead_payload = model_payload(
        forehead_model, fold, "forehead_cheek", forehead_reference,
        train["ID"].tolist(), train.loc[available, "ID"].tolist(),
        int(available.sum()), scaler, context,
    )
    write_json(cheek_payload, fold_dir / "cheek_calibrator.json")
    write_json(forehead_payload, fold_dir / "forehead_cheek_calibrator.json")
    coefficient_frame = pd.DataFrame(
        coefficient_records(fold, cheek_model) + coefficient_records(fold, forehead_model)
    )
    write_csv(coefficient_frame, fold_dir / "coefficient_table.csv")
    train_features = transform_feature_frame(
        train, fold, "train", scaler, cheek_model, forehead_model, cheek_reference, forehead_reference
    )
    val_features = transform_feature_frame(
        val, fold, "val", scaler, cheek_model, forehead_model, cheek_reference, forehead_reference
    )
    write_csv(train_features, fold_dir / "train_calibrated_features.csv")
    write_csv(val_features, fold_dir / "val_calibrated_features.csv")
    train_diag = build_diagnostics(train_features, fold, "train")
    val_diag = build_diagnostics(val_features, fold, "val")
    diagnostics = {
        family: pd.concat([train_diag[family], val_diag[family]], ignore_index=True)
        for family in train_diag
    }
    write_csv(combined_diagnostics(diagnostics), fold_dir / "calibration_diagnostics.csv")
    degenerate = [
        f"{camera}/{condition}"
        for camera in EXPECTED_CAMERAS
        for condition in CONDITION_NAMES
        if scaler["camera_parameters"][camera]["conditions"][condition]["degenerate_std"]
    ]
    summary = {
        "fold": fold, "status": "PASS", "train_n": len(train), "val_n": len(val),
        "train_val_overlap_n": len(set(train["ID"]) & set(val["ID"])),
        "cheek_calibrator_train_n": len(train),
        "forehead_cheek_calibrator_train_n": int(available.sum()),
        "train_forehead_available_n": int(available.sum()),
        "val_forehead_available_n": int(val["forehead_available"].sum()),
        "degenerate_condition_count": len(degenerate),
        "degenerate_conditions": degenerate,
        "train_id_sha256": sha256_ids(train["ID"].tolist()),
        "val_id_sha256": sha256_ids(val["ID"].tolist()),
    }
    write_json(summary, fold_dir / "fold_summary.json")
    return {
        "summary": summary, "val_features": val_features, "coefficients": coefficient_frame,
        "diagnostics": diagnostics,
    }


def build_feature_schema() -> dict[str, Any]:
    raw = [f"raw_{target}" for target in ALL_TARGETS]
    predicted = [f"predicted_acquisition_{target}" for target in ALL_TARGETS]
    residual = [f"residual_{target}" for target in ALL_TARGETS]
    calibrated = [f"calibrated_{target}" for target in ALL_TARGETS]
    forbidden = [
        "camera_id", "ExposureTime", "FNumber", "ISOSpeedRatings", *CONDITION_NAMES,
        "z_relative_optical_exposure", "z_log2_iso_condition", *predicted,
        "valid_skin_fraction", "valid_skin_pixel_count", "bbox_area", "cheek_abs_diff",
        "IQR", "channel_clipping_fractions",
    ]
    return {
        "schema_name": "optical_condition_calibration_stage2a",
        "identifier_columns": ["ID", "fold", "split_role"],
        "availability_columns": ["forehead_available"],
        "raw_observation_columns": raw,
        "predicted_acquisition_columns": predicted,
        "residual_columns": residual,
        "calibrated_optical_feature_columns": calibrated,
        "diagnostic_condition_columns": [*CONDITION_NAMES, "z_relative_optical_exposure", "z_log2_iso_condition"],
        "device_condition_columns": ["camera_id"],
        "qc_only_columns": ["valid_skin_fraction", "valid_skin_pixel_count", "bbox_area", "cheek_abs_diff", "IQR", "channel_clipping_fractions"],
        "forbidden_direct_nyha_classifier_columns": forbidden,
        "usage_note": "Do not combine all raw/residual/calibrated representations in one classifier; camera and EXIF fields are diagnostic only.",
        "clinical_or_label_columns_loaded": [],
    }


def validate_oof(oof: pd.DataFrame, first_stage: pd.DataFrame, split_info: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if len(oof) != 500 or oof["ID"].nunique() != 500:
        errors.append("OOF is not 500 rows / 500 unique IDs")
    if set(oof["fold"].astype(int)) != set(range(5)):
        errors.append("OOF fold values are not exactly 0-4")
    expected_fold = {}
    for fold in range(5):
        expected_fold.update({str(value): fold for value in split_info["splits"][fold]["val"]["ID"]})
    actual_fold = dict(zip(oof["ID"].astype(str), oof["fold"].astype(int)))
    if actual_fold != expected_fold:
        errors.append("OOF ID-to-fold mapping differs from the fixed split")
    source_available = first_stage.set_index("ID")["forehead_available"].astype(int)
    oof_available = oof.set_index("ID")["forehead_available"].astype(int)
    if not source_available.sort_index().equals(oof_available.sort_index()):
        errors.append("OOF forehead_available differs from first stage")
    cheek_output = [
        f"{representation}_{target}"
        for target in CHEEK_TARGETS
        for representation in ("raw", "predicted_acquisition", "residual", "calibrated")
    ]
    forehead_output = [
        f"{representation}_{target}"
        for target in FOREHEAD_CHEEK_TARGETS
        for representation in ("raw", "predicted_acquisition", "residual", "calibrated")
    ]
    if not np.isfinite(oof[cheek_output].to_numpy(float)).all():
        errors.append("OOF contains nonfinite cheek outputs")
    available = oof["forehead_available"].astype(int) == 1
    if not np.isfinite(oof.loc[available, forehead_output].to_numpy(float)).all():
        errors.append("OOF contains nonfinite available-forehead outputs")
    if not oof.loc[~available, forehead_output].isna().all().all():
        errors.append("OOF contains illegal unavailable-forehead outputs")
    forbidden_tokens = ("nyha", "label", "sex", "bnp", "brightnessvalue")
    illegal_columns = [column for column in oof.columns if any(token in column.casefold() for token in forbidden_tokens)]
    if illegal_columns:
        errors.append(f"OOF contains forbidden columns: {illegal_columns}")
    return {
        "status": "PASS" if not errors else "FAIL", "errors": errors,
        "oof_rows": len(oof), "oof_unique_ids": int(oof["ID"].nunique()),
        "forehead_available_n": int(available.sum()),
        "forehead_unavailable_n": int((~available).sum()),
        "nonfinite_cheek_output_n": int((~np.isfinite(oof[cheek_output].to_numpy(float))).sum()),
        "illegal_forehead_output_n": int(oof.loc[~available, forehead_output].notna().sum().sum()),
    }


def coefficient_stability_interpretation(stability: pd.DataFrame) -> dict[str, Any]:
    groups = {
        "exposure_slopes": ("honor_exposure_slope", "xiaomi_exposure_slope"),
        "iso_slopes": ("honor_iso_slope", "xiaomi_iso_slope"),
        "camera_intercept_differences": ("device_intercept_difference",),
    }
    interpretation: dict[str, Any] = {}
    for label, coefficient_names in groups.items():
        view = stability.loc[stability["coefficient_name"].isin(coefficient_names)].copy()
        consistent = view["sign_consistent_fold_n"].astype(int).eq(view["fold_valid_n"].astype(int))
        unstable = [
            {
                "target": str(row.target),
                "coefficient_name": str(row.coefficient_name),
                "sign_consistent_fold_n": int(row.sign_consistent_fold_n),
                "fold_valid_n": int(row.fold_valid_n),
            }
            for row in view.loc[~consistent].itertuples(index=False)
        ]
        interpretation[label] = {
            "full_direction_consistency_n": int(consistent.sum()),
            "total_dimension_n": int(len(view)),
            "fully_consistent_positive_n": int((consistent & view["mean"].astype(float).gt(0)).sum()),
            "fully_consistent_negative_n": int((consistent & view["mean"].astype(float).lt(0)).sum()),
            "unstable_dimensions": unstable,
        }
    return interpretation


def summarize_fold_results(
    first_stage: pd.DataFrame,
    split_info: Mapping[str, Any],
    experiment_root: Path,
    report_root: Path,
    fold_results: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    summary_dir = experiment_root / "summary"
    oof = pd.concat([fold_results[fold]["val_features"] for fold in range(5)], ignore_index=True)
    oof = oof.sort_values("ID", kind="stable").reset_index(drop=True)
    validation = validate_oof(oof, first_stage, split_info)
    if validation["status"] != "PASS":
        raise CalibrationFailure("oof_validation", validation["errors"])
    write_csv(oof, summary_dir / "oof_calibrated_features.csv")
    coefficients = pd.concat([fold_results[fold]["coefficients"] for fold in range(5)], ignore_index=True)
    stability = coefficient_stability(coefficients)
    write_csv(stability, summary_dir / "coefficient_stability.csv")
    diagnostics_by_family = {
        family: pd.concat(
            [fold_results[fold]["diagnostics"][family] for fold in range(5)], ignore_index=True
        )
        for family in ("fit", "correlation", "camera", "variance")
    }
    all_diagnostics = combined_diagnostics(diagnostics_by_family)
    write_csv(all_diagnostics, summary_dir / "fold_diagnostics_all.csv")
    schema = build_feature_schema()
    write_json(schema, summary_dir / "calibration_feature_schema.json")
    fold_summaries = [fold_results[fold]["summary"] for fold in range(5)]
    calibration_summary = {
        "task": TASK, "status": "PASS", "fold_count": 5,
        "fold_summaries": fold_summaries, "oof_validation": validation,
        "coefficient_all_finite": bool(np.isfinite(coefficients["value"].to_numpy(float)).all()),
        "coefficient_stability_interpretation": coefficient_stability_interpretation(stability),
        "clinical_labels_loaded": False, "nyha_used": False,
        "validation_tuning": False, "full_cohort_normalization": False,
    }
    write_json(calibration_summary, summary_dir / "calibration_summary.json")
    report_root.mkdir(parents=True, exist_ok=True)
    report_frames = {
        "fold_calibration_metrics.csv": diagnostics_by_family["fit"],
        "raw_vs_calibrated_exif_correlations.csv": diagnostics_by_family["correlation"],
        "raw_vs_calibrated_camera_differences.csv": diagnostics_by_family["camera"],
        "variance_retention.csv": diagnostics_by_family["variance"],
        "coefficient_stability.csv": stability,
    }
    for filename, frame in report_frames.items():
        write_csv(frame, report_root / filename)
    return {
        "fold_results": dict(fold_results), "oof": oof, "validation": validation,
        "coefficients": coefficients, "stability": stability,
        "diagnostics": diagnostics_by_family, "calibration_summary": calibration_summary,
    }


def load_existing_fold_results(experiment_root: Path) -> dict[int, dict[str, Any]]:
    required_files = (
        "condition_scaler.json", "cheek_calibrator.json", "forehead_cheek_calibrator.json",
        "fold_summary.json", "val_calibrated_features.csv", "coefficient_table.csv",
        "calibration_diagnostics.csv",
    )
    fold_results: dict[int, dict[str, Any]] = {}
    errors: list[str] = []
    for fold in range(5):
        fold_dir = experiment_root / f"fold_{fold}"
        missing = [name for name in required_files if not (fold_dir / name).is_file()]
        if missing:
            errors.append(f"Fold {fold} missing existing artifacts: {missing}")
            continue
        try:
            summary = json.loads((fold_dir / "fold_summary.json").read_text(encoding="utf-8"))
            val_features = pd.read_csv(
                fold_dir / "val_calibrated_features.csv", dtype={"ID": str}, encoding="utf-8-sig"
            )
            coefficients = pd.read_csv(fold_dir / "coefficient_table.csv", encoding="utf-8-sig")
            diagnostic_union = pd.read_csv(fold_dir / "calibration_diagnostics.csv", encoding="utf-8-sig")
            if set(diagnostic_union["metric_family"].dropna().astype(str)) != {
                "fit", "correlation", "camera", "variance"
            }:
                raise ValueError("metric_family does not contain the four required families")
            diagnostics = {
                family: diagnostic_union.loc[
                    diagnostic_union["metric_family"].eq(family)
                ].drop(columns="metric_family").dropna(axis=1, how="all").reset_index(drop=True)
                for family in ("fit", "correlation", "camera", "variance")
            }
        except Exception as exc:
            errors.append(f"Fold {fold} existing artifacts cannot be read: {type(exc).__name__}: {exc}")
            continue
        if summary.get("status") != "PASS" or int(summary.get("fold", -1)) != fold:
            errors.append(f"Fold {fold} summary is not a matching PASS artifact")
            continue
        fold_results[fold] = {
            "summary": summary, "val_features": val_features,
            "coefficients": coefficients, "diagnostics": diagnostics,
        }
    if errors or set(fold_results) != set(range(5)):
        raise CalibrationFailure("summarize_only", errors or ["Existing five-fold artifacts are incomplete"])
    return fold_results


def summarize_existing(
    first_stage: pd.DataFrame,
    split_info: Mapping[str, Any],
    experiment_root: Path,
    report_root: Path,
) -> dict[str, Any]:
    if not verify_owned_output(experiment_root) or not verify_owned_output(report_root, report=True):
        raise CalibrationFailure(
            "output_safety", ["Summarize-only outputs are not verified as Stage 2A artifacts"]
        )
    return summarize_fold_results(
        first_stage, split_info, experiment_root, report_root,
        load_existing_fold_results(experiment_root),
    )


def execute_core(
    args: argparse.Namespace,
    first_stage: pd.DataFrame,
    split_info: Mapping[str, Any],
    experiment_root: Path,
    report_root: Path,
    context: Mapping[str, Any],
    selected_folds: Sequence[int] = tuple(range(5)),
) -> dict[str, Any]:
    protocol_dir = experiment_root / "protocol"
    summary_dir = experiment_root / "summary"
    write_csv(split_info["audit"], protocol_dir / "split_audit.csv")
    write_json({
        "task": TASK, "status": "PASS", "split_source": project_relative(args.split_dir),
        "split_sha256": split_info["split_sha256"], "n_folds": 5,
        "clinical_columns_loaded": [], "per_fold_hashes": split_info["per_fold_hashes"],
        "split_regenerated": False,
    }, protocol_dir / "split_manifest.json")
    if args.protocol_only:
        return {"fold_results": {}, "protocol_only": True}
    fold_results: dict[int, dict[str, Any]] = {}
    for fold in selected_folds:
        fold_results[fold] = run_fold(
            fold, first_stage, split_info, experiment_root, context, args
        )
    if set(selected_folds) != set(range(5)):
        return {"fold_results": fold_results, "protocol_only": False, "partial": True}
    return summarize_fold_results(
        first_stage, split_info, experiment_root, report_root, fold_results
    )


def run_test_file(root: Path, filename: str) -> tuple[str, str]:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", filename], cwd=root, text=True, capture_output=True
    )
    output = (result.stdout + "\n" + result.stderr).strip()
    return ("PASS" if result.returncode == 0 else "FAIL", output)


def deterministic_files(experiment_root: Path, report_root: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    for prefix, root in (("experiment", experiment_root), ("report", report_root)):
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name.startswith("."):
                continue
            if path.name in ("run_manifest.json", "optical_condition_calibration_stage2a_report.md", "run.log"):
                continue
            if path.suffix.lower() not in (".csv", ".json"):
                continue
            output[f"{prefix}/{path.relative_to(root).as_posix()}"] = sha256_file(path)
    return output


def markdown_table(frame: pd.DataFrame, columns: Sequence[str], max_rows: int = 20) -> str:
    view = frame.loc[:, list(columns)].head(max_rows).copy()
    for column in view.select_dtypes(include=[np.number]).columns:
        view[column] = view[column].map(lambda value: round(float(value), 6) if pd.notna(value) else "NaN")
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |" for row in view.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *rows])


def stability_report_lines(stability: pd.DataFrame) -> list[str]:
    interpretation = coefficient_stability_interpretation(stability)
    labels = (
        ("exposure_slopes", "Exposure斜率"),
        ("iso_slopes", "ISO斜率"),
        ("camera_intercept_differences", "设备截距差"),
    )
    lines: list[str] = []
    unstable_all: list[dict[str, Any]] = []
    for key, display in labels:
        item = interpretation[key]
        lines.append(
            f"- {display}：{item['full_direction_consistency_n']}/{item['total_dimension_n']}个维度在全部有效fold中方向一致；"
            f"其中稳定正向{item['fully_consistent_positive_n']}个、稳定负向{item['fully_consistent_negative_n']}个。"
        )
        unstable_all.extend(item["unstable_dimensions"])
    if unstable_all:
        details = "；".join(
            f"`{row['target']} / {row['coefficient_name']}`="
            f"{row['sign_consistent_fold_n']}/{row['fold_valid_n']}"
            for row in unstable_all
        )
        lines.append(f"- 未达到全fold方向一致的维度：{details}。")
    else:
        lines.append("- 上述Exposure、ISO与设备截距差维度均达到全fold方向一致。")
    lines.append("- 这些结果仅描述五折稳定性；未据此改变alpha、筛选维度或重拟合模型。")
    return lines


def write_report(
    args: argparse.Namespace,
    result: Mapping[str, Any],
    split_info: Mapping[str, Any],
    unit_status: str,
    unit_output: str,
    protocol_test_status: str,
    protocol_test_output: str,
    deterministic_match: bool,
) -> Path:
    report_path = args.report_output_dir / "optical_condition_calibration_stage2a_report.md"
    fold_summary = pd.DataFrame([result["fold_results"][fold]["summary"] for fold in range(5)])
    fit = result["diagnostics"]["fit"]
    corr = result["diagnostics"]["correlation"]
    camera = result["diagnostics"]["camera"]
    variance = result["diagnostics"]["variance"]
    stability = result["stability"]
    corr_view = corr.loc[
        (corr["split_role"] == "val") & (corr["scope"] == "overall")
        & (corr["target"].isin((CHEEK_TARGETS[0], FOREHEAD_CHEEK_TARGETS[0])))
        & (corr["representation"].isin(("raw", "calibrated")))
    ]
    camera_view = camera.loc[
        (camera["split_role"] == "val")
        & (camera["target"].isin((CHEEK_TARGETS[0], FOREHEAD_CHEEK_TARGETS[0])))
    ]
    variance_view = variance.loc[variance["split_role"] == "val"]
    stability_view = stability.loc[stability["coefficient_name"].isin((
        "honor_exposure_slope", "xiaomi_exposure_slope", "honor_iso_slope",
        "xiaomi_iso_slope", "device_intercept_difference",
    ))]
    files = [*IMPLEMENTATION_FILES]
    files.extend(
        project_relative(path) for root in (args.experiment_output_dir, args.report_output_dir)
        for path in sorted(root.rglob("*")) if path.is_file() and not path.name.startswith(".")
    )
    files.extend((
        "experiments/optical_condition_calibration_stage2a/summary/run_manifest.json",
        "reports/optical_condition_calibration_stage2a/optical_condition_calibration_stage2a_report.md",
        "reports/optical_condition_calibration_stage2a/run.log",
    ))
    files = list(dict.fromkeys(files))
    lines = [
        REPORT_MARKER,
        f"# {TASK_TITLE}", "",
        "## 1. 完成状态", "",
        "- `OPTICAL_CALIBRATION_STAGE2A_STATUS=COMPLETE`",
        "- 本产物是 acquisition-conditioned optical calibration（采集条件校准后的区域光学表型），不是皮肤真实反射率、传感器线性RGB、生理参数或完整物理反演。", "",
        "## 2. 新增/修改文件", "",
        *[f"- `{path}`" for path in files], "",
        "## 3. 第一阶段输入来源", "",
        f"- `{project_relative(args.first_stage_csv)}`（SHA256 `{sha256_file(args.first_stage_csv)}`）",
        f"- `{project_relative(args.first_stage_schema)}`与`{project_relative(args.first_stage_manifest)}`。", "",
        "## 4. 普通五折split来源和审计", "",
        f"固定使用`{project_relative(args.split_dir)}/fold_{{fold}}_{{train|val}}.csv`，组合SHA256为`{split_info['split_sha256']}`；未重新生成或改变任何ID所属fold。", "",
        markdown_table(fold_summary, ["fold", "train_n", "val_n", "train_val_overlap_n", "cheek_calibrator_train_n", "forehead_cheek_calibrator_train_n"]), "",
        "## 5. 未读取NYHA声明", "",
        "校准代码对白名单读取第一阶段字段；split仅以`usecols=['ID','fold']`读取。`clinical_labels_loaded=false`，`nyha_used=false`。", "",
        "## 6. 六维观测定义", "",
        "脸颊三维为`cheek_mean_log2_y/log2_rg/log2_bg`；额部—脸颊三维为`forehead_minus_cheek_log2_y/log2_rg/log2_bg`。", "",
        "## 7. EXIF条件定义", "",
        "仅使用`relative_optical_exposure`、`log2_iso_condition`和`camera_id`。FNumber只通过`log2(ExposureTime/FNumber^2)`进入条件；BrightnessValue未读取。", "",
        "## 8. 设备内标准化方法", "",
        "每fold仅在训练集内、分别按两设备计算均值和population std（ddof=0）；验证集复用对应训练参数。若std<1e-8则z统一为0并记录。", "",
        "## 9. 设计矩阵", "",
        "固定列顺序：`camera_xiaomi, z_relative_optical_exposure, z_log2_iso_condition, camera_xiaomi_x_z_exposure, camera_xiaomi_x_z_iso`，另含独立截距。", "",
        "## 10. Ridge公式和alpha", "",
        "使用NumPy闭式多输出Ridge：`sum((Y-Y_hat)^2)+1.0*sum(beta^2)`；截距不惩罚，alpha不扫描，求解失败时才显式使用pinv回退。", "",
        "## 11. 两个校准器的训练子集", "",
        "Cheek校准器使用每fold全部400个训练病例；额部—脸颊校准器只使用`forehead_available==1`且三目标有限的训练病例。", "",
        "## 12. 额部缺失处理", "",
        "额部不可用病例被保留，额部—脸颊的raw、predicted、residual、calibrated均保持NaN；未插补、填0、替代或重算Mask。", "",
        "## 13. residual和calibrated定义", "",
        "`residual=raw-predicted_acquisition`；`calibrated=residual+current_fold_training_reference_mean`。训练和验证使用同一训练参考均值。", "",
        "## 14. 每fold训练及验证数量", "",
        markdown_table(fold_summary, ["fold", "train_n", "val_n", "train_forehead_available_n", "val_forehead_available_n", "degenerate_condition_count"]), "",
        "## 15. 每fold条件拟合误差", "",
        markdown_table(fit.loc[fit["split_role"] == "val"], ["fold", "target", "valid_n", "mae", "rmse", "r2"], max_rows=30), "",
        "## 16. 校准前后EXIF相关性", "",
        markdown_table(corr_view, ["fold", "target", "representation", "condition", "valid_n", "spearman_rho"], max_rows=40), "",
        "这些相关性仅用于描述校准行为，不做显著性筛选或调参。", "",
        "## 17. 校准前后设备差异", "",
        markdown_table(camera_view, ["fold", "target", "representation", "mean_difference_honor_minus_xiaomi", "median_difference_honor_minus_xiaomi", "standardized_mean_difference"], max_rows=30), "",
        "设备差异可能降低，也可能不降低；不声称完全消除设备差异。", "",
        "## 18. 方差保留", "",
        markdown_table(variance_view, ["fold", "target", "valid_n", "raw_variance", "calibrated_variance", "variance_retention"], max_rows=30), "",
        "方差保留不是越小越好，仅用于描述校准后的观测变异。", "",
        "## 19. 五折系数稳定性", "",
        markdown_table(stability_view, ["target", "coefficient_name", "fold_valid_n", "mean", "std", "min", "max", "sign_consistent_fold_n"], max_rows=40), "",
        *stability_report_lines(stability), "",
        "## 20. OOF完整性", "",
        f"OOF={result['validation']['oof_rows']}行、唯一ID={result['validation']['oof_unique_ids']}；额部可用={result['validation']['forehead_available_n']}、不可用={result['validation']['forehead_unavailable_n']}；非有限cheek输出={result['validation']['nonfinite_cheek_output_n']}；非法额部输出={result['validation']['illegal_forehead_output_n']}。", "",
        "## 21. 单元测试及协议测试", "",
        f"单元测试：`{unit_status}`（{unit_output.replace(chr(10), ' | ')}）；五折协议测试：`{protocol_test_status}`（{protocol_test_output.replace(chr(10), ' | ')}）。", "",
        "## 22. 确定性验证", "",
        f"在实验输出目录内的临时目录重复完整核心五折，全部CSV/模型JSON SHA256一致：`{deterministic_match}`。", "",
        "## 23. 历史输入未修改声明", "",
        "第一阶段CSV、schema、manifest及十份固定split在运行前后SHA256一致；`historical_inputs_modified=false`。", "",
        "## 24. 局限性", "",
        "1. 这是低容量线性采集条件校准基线，不是因果分解。",
        "2. 模型不能恢复手机ISP处理前的传感器信号，也不能得到真实皮肤反射率。",
        "3. 仅有两种设备；对未知设备不做外推编码。",
        "4. 校准诊断不使用NYHA，不能证明校准特征与NYHA相关或提高分类性能。", "",
        "## 25. 下一阶段可使用的六个calibrated字段", "",
        *[f"- `calibrated_{target}`" for target in ALL_TARGETS], "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def final_lines(
    args: argparse.Namespace,
    result: Mapping[str, Any],
    split_info: Mapping[str, Any],
    unit_status: str,
    protocol_test_status: str,
    deterministic_match: bool,
    historical_modified: bool,
) -> list[str]:
    fold_summaries = [result["fold_results"][fold]["summary"] for fold in range(5)]
    files = [*IMPLEMENTATION_FILES]
    files.extend(
        project_relative(path) for root in (args.experiment_output_dir, args.report_output_dir)
        for path in sorted(root.rglob("*")) if path.is_file() and not path.name.startswith(".")
    )
    files.extend((
        "experiments/optical_condition_calibration_stage2a/summary/run_manifest.json",
        "reports/optical_condition_calibration_stage2a/optical_condition_calibration_stage2a_report.md",
        "reports/optical_condition_calibration_stage2a/run.log",
    ))
    files = list(dict.fromkeys(files))
    return [
        "OPTICAL_CALIBRATION_STAGE2A_STATUS=COMPLETE",
        "ADDED_OR_MODIFIED_FILES=" + ",".join(files),
        f"SPLIT_SOURCE={project_relative(args.split_dir)}/fold_{{fold}}_{{train|val}}.csv",
        f"SPLIT_SHA256={split_info['split_sha256']}",
        "SPLIT_PROTOCOL_STATUS=PASS",
        f"UNIT_TEST_STATUS={unit_status}",
        f"FIVEFOLD_PROTOCOL_TEST_STATUS={protocol_test_status}",
        "COMPLETED_FOLD_COUNT=5",
        "FOLD_TRAIN_VAL_COUNTS=" + ",".join(f"{row['fold']}:{row['train_n']}/{row['val_n']}" for row in fold_summaries),
        "CHEEK_CALIBRATOR_TRAIN_COUNTS=" + ",".join(f"{row['fold']}:{row['cheek_calibrator_train_n']}" for row in fold_summaries),
        "FOREHEAD_CHEEK_CALIBRATOR_TRAIN_COUNTS=" + ",".join(f"{row['fold']}:{row['forehead_cheek_calibrator_train_n']}" for row in fold_summaries),
        f"OOF_ROWS={result['validation']['oof_rows']}",
        f"OOF_UNIQUE_IDS={result['validation']['oof_unique_ids']}",
        f"FOREHEAD_AVAILABLE_COUNT={result['validation']['forehead_available_n']}",
        f"FOREHEAD_UNAVAILABLE_COUNT={result['validation']['forehead_unavailable_n']}",
        f"NONFINITE_CHEEK_OUTPUT_COUNT={result['validation']['nonfinite_cheek_output_n']}",
        f"ILLEGAL_FOREHEAD_OUTPUT_COUNT={result['validation']['illegal_forehead_output_n']}",
        f"ALL_COEFFICIENTS_FINITE={'YES' if result['calibration_summary']['coefficient_all_finite'] else 'NO'}",
        f"DETERMINISTIC_REPEAT_MATCH={'YES' if deterministic_match else 'NO'}",
        f"OOF_OUTPUT_PATH={args.experiment_output_dir / 'summary/oof_calibrated_features.csv'}",
        f"SCHEMA_PATH={args.experiment_output_dir / 'summary/calibration_feature_schema.json'}",
        f"REPORT_PATH={args.report_output_dir / 'optical_condition_calibration_stage2a_report.md'}",
        "CLINICAL_LABELS_LOADED=false", "NYHA_USED=false", "VALIDATION_TUNING=false",
        "FULL_COHORT_NORMALIZATION=false",
        f"HISTORICAL_INPUTS_MODIFIED={str(historical_modified).lower()}",
        "ALL_ACCEPTANCE_CRITERIA_MET=YES",
    ]


def write_run_manifest(
    args: argparse.Namespace,
    result: Mapping[str, Any],
    split_info: Mapping[str, Any],
    unit_status: str,
    protocol_test_status: str,
    deterministic_match: bool,
    input_hashes: Mapping[str, str],
) -> None:
    model_hashes = {}
    for fold in range(5):
        model_hashes[str(fold)] = {
            name: sha256_file(args.experiment_output_dir / f"fold_{fold}/{name}")
            for name in ("condition_scaler.json", "cheek_calibrator.json", "forehead_cheek_calibrator.json")
        }
    manifest = {
        "task": TASK, "status": "COMPLETE",
        "method": "fixed multi-output Ridge acquisition-conditioned optical calibration",
        "alpha": 1.0, "fit_intercept": True, "penalize_intercept": False,
        "design_matrix_fields": list(DESIGN_FEATURE_NAMES),
        "target_fields": list(ALL_TARGETS),
        "reference_camera": REFERENCE_CAMERA,
        "camera_mapping": {EXPECTED_CAMERAS[0]: 0, EXPECTED_CAMERAS[1]: 1},
        "std_epsilon": 1.0e-8,
        "first_stage_csv": {"path": project_relative(args.first_stage_csv), "sha256": input_hashes["first_stage_csv"]},
        "first_stage_feature_schema_sha256": input_hashes["first_stage_schema"],
        "first_stage_extraction_manifest_sha256": input_hashes["first_stage_manifest"],
        "split": {"path": project_relative(args.split_dir), "sha256": split_info["split_sha256"]},
        "per_fold_split_hashes": split_info["per_fold_hashes"],
        "per_fold_model_file_sha256": model_hashes,
        "oof_output_sha256": sha256_file(args.experiment_output_dir / "summary/oof_calibrated_features.csv"),
        "config": {"path": project_relative(args.config_path), "sha256": sha256_file(args.config_path)},
        "script": {"path": project_relative(Path(__file__)), "sha256": sha256_file(Path(__file__))},
        "tests": {"unit": unit_status, "fivefold_protocol": protocol_test_status},
        "deterministic_repeat_match": deterministic_match,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__, "pandas_version": pd.__version__, "pyyaml_version": yaml.__version__,
        "git_commit": get_git_commit(args.project_root),
        "clinical_labels_loaded": False, "nyha_used": False,
        "full_cohort_normalization": False, "validation_tuning": False,
        "historical_inputs_modified": False,
    }
    write_json(manifest, args.experiment_output_dir / "summary/run_manifest.json")


def existing_determinism_status(experiment_root: Path) -> bool:
    manifest_path = experiment_root / "summary/run_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CalibrationFailure(
            "summarize_only",
            [f"Completed run manifest cannot be read: {type(exc).__name__}: {exc}"],
        ) from exc
    if manifest.get("task") != TASK or manifest.get("status") != "COMPLETE":
        raise CalibrationFailure("summarize_only", ["Existing run manifest is not a completed Stage 2A run"])
    return bool(manifest.get("deterministic_repeat_match", False))


def write_failure_log_safely(
    args: argparse.Namespace | None, failure: CalibrationFailure
) -> Path | None:
    if args is None:
        return None
    report_root = args.report_output_dir
    try:
        if report_root.exists() and any(report_root.iterdir()):
            if not verify_owned_output(report_root, report=True):
                return None
        report_root.mkdir(parents=True, exist_ok=True)
        write_json({"task": TASK}, report_root / ".stage2a_owner.json")
        failure_path = report_root / "run_failure.log"
        failure_path.write_text(
            f"TASK={TASK}\nSTATUS=FAILED\nSTAGE={failure.stage}\n"
            + "\n".join(failure.errors) + "\n",
            encoding="utf-8",
        )
        return failure_path
    except Exception:
        return None


def main(argv: Sequence[str] | None = None) -> int:
    args: argparse.Namespace | None = None
    try:
        args = load_args(argv)
        input_paths = [args.first_stage_csv, args.first_stage_schema, args.first_stage_manifest]
        first_stage = load_first_stage(args)
        input_hashes = {
            "first_stage_csv": sha256_file(args.first_stage_csv),
            "first_stage_schema": sha256_file(args.first_stage_schema),
            "first_stage_manifest": sha256_file(args.first_stage_manifest),
        }
        split_info = audit_splits(args, first_stage)
        input_paths.extend(split_info["split_files"])
        historical_before = {project_relative(path): sha256_file(path) for path in input_paths}
        print(f"PREFLIGHT_STATUS=PASS FIRST_STAGE_ROWS={len(first_stage)} UNIQUE_IDS={first_stage['ID'].nunique()}")
        print(f"SPLIT_PROTOCOL_STATUS=PASS SPLIT_SHA256={split_info['split_sha256']}")
        unit_status, unit_output = run_test_file(
            args.project_root, "tests/test_optical_condition_calibration_stage2a.py"
        )
        if unit_status != "PASS":
            raise CalibrationFailure("unit_tests", [unit_output])
        protocol_test_status, protocol_test_output = run_test_file(
            args.project_root, "tests/test_optical_calibration_fivefold_protocol.py"
        )
        if protocol_test_status != "PASS":
            raise CalibrationFailure("fivefold_protocol_tests", [protocol_test_output])
        if args.summarize_only:
            deterministic_match = existing_determinism_status(args.experiment_output_dir)
            result = summarize_existing(
                first_stage, split_info, args.experiment_output_dir, args.report_output_dir
            )
        else:
            prepare_outputs(args)
            context = {
                "split_sha256": split_info["split_sha256"],
                "first_stage_sha256": input_hashes["first_stage_csv"],
                "config_sha256": sha256_file(args.config_path),
                "git_commit": get_git_commit(args.project_root),
            }
            selected = tuple(range(5)) if args.fold == "all" else (int(args.fold),)
            result = execute_core(
                args, first_stage, split_info, args.experiment_output_dir, args.report_output_dir,
                context, selected_folds=selected,
            )
            if args.protocol_only:
                print("OPTICAL_CALIBRATION_STAGE2A_STATUS=PROTOCOL_ONLY_COMPLETE")
                print(f"SPLIT_SHA256={split_info['split_sha256']}")
                return 0
            if set(selected) != set(range(5)):
                print("OPTICAL_CALIBRATION_STAGE2A_STATUS=PARTIAL")
                print(f"COMPLETED_FOLD_COUNT={len(selected)}")
                print("ALL_ACCEPTANCE_CRITERIA_MET=NO")
                return 0
            official_hashes = deterministic_files(args.experiment_output_dir, args.report_output_dir)
            with tempfile.TemporaryDirectory(prefix=".determinism_", dir=args.experiment_output_dir) as temp_name:
                temp_root = Path(temp_name)
                repeat_result = execute_core(
                    args, first_stage, split_info, temp_root / "experiment", temp_root / "report",
                    context, selected_folds=tuple(range(5)),
                )
                repeat_hashes = deterministic_files(temp_root / "experiment", temp_root / "report")
                deterministic_match = official_hashes == repeat_hashes
                if not deterministic_match:
                    missing = sorted(set(official_hashes).symmetric_difference(repeat_hashes))
                    changed = sorted(key for key in set(official_hashes) & set(repeat_hashes) if official_hashes[key] != repeat_hashes[key])
                    raise CalibrationFailure("determinism", [f"missing={missing}", f"changed={changed}"])
                if not result["oof"].equals(repeat_result["oof"]):
                    raise CalibrationFailure("determinism", ["Repeated OOF DataFrame differs"])
        historical_after = {project_relative(path): sha256_file(path) for path in input_paths}
        historical_modified = historical_before != historical_after
        if historical_modified:
            raise CalibrationFailure("historical_input_integrity", ["Input SHA256 changed during run"])
        write_report(
            args, result, split_info, unit_status, unit_output,
            protocol_test_status, protocol_test_output, deterministic_match,
        )
        lines = final_lines(
            args, result, split_info, unit_status, protocol_test_status,
            deterministic_match, historical_modified,
        )
        (args.report_output_dir / "run.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
        write_run_manifest(
            args, result, split_info, unit_status, protocol_test_status,
            deterministic_match, input_hashes,
        )
        print("\n".join(lines))
        return 0
    except Exception as exc:
        failure = exc if isinstance(exc, CalibrationFailure) else CalibrationFailure(
            "unexpected_error", [f"{type(exc).__name__}: {exc}"]
        )
        failure_path = write_failure_log_safely(args, failure)
        print("OPTICAL_CALIBRATION_STAGE2A_STATUS=FAILED")
        print(f"FAILED_STAGE={failure.stage}")
        print("ERRORS=" + " | ".join(failure.errors))
        print(f"ERROR_LOG_PATH={failure_path if failure_path else 'unavailable'}")
        print("EXACT_RESUME_COMMAND=python scripts/preprocess/run_optical_condition_calibration_stage2a.py --config config/preprocess/optical_condition_calibration_stage2a.yaml --fold all --overwrite")
        print("ALL_ACCEPTANCE_CRITERIA_MET=NO")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
