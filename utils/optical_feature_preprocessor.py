"""Strict optical-feature loading and outer-train-only standardization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


VARIANTS = (
    "global_only",
    "global_mask",
    "global_raw",
    "global_stage2a",
    "global_stage2b",
)
RAW_FEATURE_COLUMNS = (
    "cheek_mean_log2_y",
    "cheek_mean_log2_rg",
    "cheek_mean_log2_bg",
    "forehead_minus_cheek_log2_y",
    "forehead_minus_cheek_log2_rg",
    "forehead_minus_cheek_log2_bg",
)
STAGE2A_FEATURE_COLUMNS = (
    "calibrated_cheek_mean_log2_y",
    "calibrated_cheek_mean_log2_rg",
    "calibrated_cheek_mean_log2_bg",
    "calibrated_forehead_minus_cheek_log2_y",
    "calibrated_forehead_minus_cheek_log2_rg",
    "calibrated_forehead_minus_cheek_log2_bg",
)
STAGE2B_FEATURE_COLUMNS = (
    "calibrated_nn_cheek_mean_log2_y",
    "calibrated_nn_cheek_mean_log2_rg",
    "calibrated_nn_cheek_mean_log2_bg",
    "calibrated_nn_forehead_minus_cheek_log2_y",
    "calibrated_nn_forehead_minus_cheek_log2_rg",
    "calibrated_nn_forehead_minus_cheek_log2_bg",
)
VARIANT_FEATURE_COLUMNS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "global_only": (),
        "global_mask": (),
        "global_raw": RAW_FEATURE_COLUMNS,
        "global_stage2a": STAGE2A_FEATURE_COLUMNS,
        "global_stage2b": STAGE2B_FEATURE_COLUMNS,
    }
)
VARIANT_AUX_DIM: Mapping[str, int] = MappingProxyType(
    {
        "global_only": 0,
        "global_mask": 1,
        "global_raw": 7,
        "global_stage2a": 7,
        "global_stage2b": 7,
    }
)
SCHEMA_COLUMN_KEYS: Mapping[str, str | None] = MappingProxyType(
    {
        "global_only": None,
        "global_mask": None,
        "global_raw": "derived_observation_columns",
        "global_stage2a": "calibrated_optical_feature_columns",
        "global_stage2b": "calibrated_nn_optical_feature_columns",
    }
)
AVAILABILITY_COLUMN = "forehead_available"
STD_EPSILON = 1e-8
OOF_FILENAMES = frozenset(
    {"oof_calibrated_features.csv", "oof_nn_calibrated_features.csv"}
)


def validate_variant(variant: str) -> str:
    normalized = str(variant).strip().lower()
    if normalized not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {VARIANTS}")
    return normalized


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_ids(ids: Iterable[str]) -> str:
    return sha256_json([str(value) for value in ids])


def relative_path(path: str | Path, project_root: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path(project_root).resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def code_sha256(code_paths: Sequence[str | Path], project_root: str | Path) -> str:
    root = Path(project_root).resolve()
    hashes = {
        relative_path(path, root): sha256_file(path)
        for path in code_paths
    }
    return sha256_json(hashes)


def assert_classifier_feature_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    lowered_parts = tuple(part.lower() for part in resolved.parts)
    if resolved.name.lower() in OOF_FILENAMES or (
        "summary" in lowered_parts and resolved.name.lower().startswith("oof_")
    ):
        raise ValueError(f"OOF features are forbidden as classifier input: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"Feature source does not exist: {resolved}")
    return resolved


def validate_feature_schema(schema_path: str | Path, variant: str) -> dict[str, Any]:
    variant = validate_variant(variant)
    path = Path(schema_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Feature schema does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    key = SCHEMA_COLUMN_KEYS[variant]
    if key is not None:
        actual = tuple(schema.get(key, ()))
        expected = VARIANT_FEATURE_COLUMNS[variant]
        if actual != expected:
            raise ValueError(
                f"Schema order mismatch for {variant}: expected {expected}, got {actual}"
            )
    if variant == "global_mask" and AVAILABILITY_COLUMN not in tuple(
        schema.get("availability_columns", ())
    ):
        raise ValueError(f"Schema does not declare {AVAILABILITY_COLUMN}")
    return schema


def resolve_feature_source(
    config: Mapping[str, Any], variant: str, fold: int, split_role: str,
    project_root: str | Path,
) -> tuple[Path | None, Path | None, Path | None]:
    """Resolve source, schema and upstream manifest using explicit paths only."""
    variant = validate_variant(variant)
    role = str(split_role).lower()
    if role not in {"train", "val"}:
        raise ValueError(f"split_role must be train or val, got {split_role!r}")
    root = Path(project_root).resolve()
    features = config["features"]
    if variant == "global_only":
        return None, None, None
    if variant in {"global_mask", "global_raw"}:
        return (
            assert_classifier_feature_path(root / features["raw_source"]),
            (root / features["raw_schema"]).resolve(),
            (root / features["raw_manifest"]).resolve(),
        )
    if variant == "global_stage2a":
        source = root / features["stage2a_root"] / f"fold_{fold}" / (
            "train_calibrated_features.csv" if role == "train" else "val_calibrated_features.csv"
        )
        return (
            assert_classifier_feature_path(source),
            (root / features["stage2a_schema"]).resolve(),
            (root / features["stage2a_manifest"]).resolve(),
        )
    source = root / features["stage2b_root"] / f"fold_{fold}" / (
        "train_nn_calibrated_features.csv" if role == "train" else "val_nn_calibrated_features.csv"
    )
    return (
        assert_classifier_feature_path(source),
        (root / features["stage2b_schema"]).resolve(),
        (root / features["stage2b_manifest"]).resolve(),
    )


def _coerce_and_validate_availability(frame: pd.DataFrame, source: Path) -> np.ndarray:
    availability = pd.to_numeric(frame[AVAILABILITY_COLUMN], errors="coerce").to_numpy()
    if not np.isfinite(availability).all() or not np.isin(availability, [0, 1]).all():
        raise ValueError(f"{AVAILABILITY_COLUMN} must contain only finite 0/1 in {source}")
    return availability.astype(np.float64)


def validate_feature_values(frame: pd.DataFrame, variant: str, source: str | Path) -> None:
    variant = validate_variant(variant)
    source_path = Path(source)
    availability = _coerce_and_validate_availability(frame, source_path)
    columns = VARIANT_FEATURE_COLUMNS[variant]
    if not columns:
        return
    values = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    if not np.isfinite(values[:, :3]).all():
        raise ValueError(f"Cheek features must be finite for every sample in {source_path}")
    available = availability == 1
    if not np.isfinite(values[available, 3:]).all():
        raise ValueError(f"Available forehead features must be finite in {source_path}")
    if np.isfinite(values[~available, 3:]).any():
        raise ValueError(
            f"Unavailable forehead features must be NaN, not finite, in {source_path}"
        )
    if (~np.isnan(values[~available, 3:])).any():
        raise ValueError(f"Unavailable forehead features must all be NaN in {source_path}")


def load_feature_frame(
    path: str | Path,
    variant: str,
    expected_ids: Sequence[str],
    *,
    fold: int | None = None,
    split_role: str | None = None,
    schema_path: str | Path | None = None,
    allow_source_superset: bool = False,
) -> pd.DataFrame:
    """Read only the positive allowlist and return rows in split-CSV order."""
    variant = validate_variant(variant)
    source = assert_classifier_feature_path(path)
    if variant == "global_only":
        raise ValueError("global_only must not read an optical feature source")
    if schema_path is not None:
        validate_feature_schema(schema_path, variant)
    meta = ["ID"]
    if variant in {"global_stage2a", "global_stage2b"}:
        meta.extend(["fold", "split_role"])
    usecols = meta + [AVAILABILITY_COLUMN] + list(VARIANT_FEATURE_COLUMNS[variant])
    try:
        frame = pd.read_csv(
            source, usecols=usecols, dtype={"ID": "string"}, encoding="utf-8-sig"
        )
    except ValueError as exc:
        raise ValueError(f"Feature source is missing an allowlisted column: {source}") from exc
    frame["ID"] = frame["ID"].astype(str)
    if frame["ID"].duplicated().any():
        duplicates = frame.loc[frame["ID"].duplicated(keep=False), "ID"].unique()[:20]
        raise ValueError(f"Duplicate feature IDs in {source}: {duplicates.tolist()}")
    expected = [str(value) for value in expected_ids]
    if len(expected) != len(set(expected)):
        raise ValueError("Expected split IDs contain duplicates")
    source_ids, expected_set = set(frame["ID"]), set(expected)
    missing = sorted(expected_set - source_ids)
    extra = sorted(source_ids - expected_set)
    if missing or (extra and not allow_source_superset):
        raise ValueError(
            f"Feature IDs do not exactly match split IDs in {source}; "
            f"missing={missing[:20]}, extra={extra[:20]}"
        )
    frame = frame.set_index("ID", drop=False).loc[expected].reset_index(drop=True)
    if variant in {"global_stage2a", "global_stage2b"}:
        if fold is None or split_role is None:
            raise ValueError("Stage 2 feature loading requires fold and split_role")
        actual_fold = pd.to_numeric(frame["fold"], errors="coerce")
        if actual_fold.isna().any() or not (actual_fold.astype(int) == int(fold)).all():
            raise ValueError(f"Stage 2 feature fold does not equal classification fold {fold}")
        if not (frame["split_role"].astype(str).str.lower() == str(split_role).lower()).all():
            raise ValueError(
                f"Stage 2 split_role does not equal requested role {split_role!r}"
            )
    validate_feature_values(frame, variant, source)
    return frame.loc[:, usecols].copy()


@dataclass(frozen=True)
class FeatureScaler:
    schema_version: str
    variant: str
    fold: int
    feature_names: list[str]
    mean: list[float]
    std: list[float]
    valid_n: list[int]
    forehead_available_train_n: int
    forehead_unavailable_train_n: int
    ddof: int
    std_epsilon: float
    missing_fill_after_standardization: float
    availability_position: int
    source_relative_path: str
    source_sha256: str
    schema_relative_path: str
    schema_sha256: str
    upstream_manifest_relative_path: str
    upstream_manifest_sha256: str
    train_id_sha256: str
    split_sha256: str
    fit_timestamp: str
    code_sha256: str
    config_sha256: str

    def __post_init__(self) -> None:
        variant = validate_variant(self.variant)
        expected_names = VARIANT_FEATURE_COLUMNS[variant]
        if variant not in {"global_raw", "global_stage2a", "global_stage2b"}:
            raise ValueError(f"FeatureScaler is not valid for {variant}")
        if tuple(self.feature_names) != expected_names:
            raise ValueError("Scaler feature_names do not match the immutable schema order")
        if not (len(self.mean) == len(self.std) == len(self.valid_n) == 6):
            raise ValueError("Scaler mean/std/valid_n must each contain six values")
        if self.ddof != 0 or self.availability_position != 6:
            raise ValueError("Scaler must use ddof=0 and availability_position=6")
        if self.schema_version != "global_optical_feature_scaler_v1":
            raise ValueError("Unsupported feature scaler schema_version")
        if float(self.std_epsilon) <= 0:
            raise ValueError("std_epsilon must be positive")
        if self.missing_fill_after_standardization != 0:
            raise ValueError("Missing forehead values must be filled with zero after standardization")
        mean = np.asarray(self.mean, dtype=np.float64)
        std = np.asarray(self.std, dtype=np.float64)
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise ValueError("Scaler statistics must be finite")
        if (std < float(self.std_epsilon)).any():
            raise ValueError("Scaler contains a standard deviation below std_epsilon")
        if any(int(value) <= 0 for value in self.valid_n):
            raise ValueError("Scaler valid_n values must be positive")
        train_total = int(self.forehead_available_train_n) + int(
            self.forehead_unavailable_train_n
        )
        if self.valid_n[:3] != [train_total] * 3:
            raise ValueError("Cheek valid_n must equal the entire outer-train size")
        if self.valid_n[3:] != [int(self.forehead_available_train_n)] * 3:
            raise ValueError("Forehead valid_n must equal available outer-train cases")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def payload_sha256(self) -> str:
        return sha256_json(self.to_dict())

    def save_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)

    @classmethod
    def load_json(cls, path: str | Path) -> "FeatureScaler":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls(**json.load(handle))

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if tuple(self.feature_names) != VARIANT_FEATURE_COLUMNS[self.variant]:
            raise ValueError("Scaler feature order does not match the immutable variant schema")
        validate_feature_values(frame, self.variant, "transform frame")
        values = frame.loc[:, self.feature_names].to_numpy(dtype=np.float64, copy=True)
        availability = pd.to_numeric(frame[AVAILABILITY_COLUMN]).to_numpy(np.float64)
        standardized = (values - np.asarray(self.mean)) / np.asarray(self.std)
        standardized[availability == 0, 3:] = self.missing_fill_after_standardization
        aux = np.concatenate([standardized, availability[:, None]], axis=1).astype(np.float32)
        if aux.shape[1] != VARIANT_AUX_DIM[self.variant] or not np.isfinite(aux).all():
            raise ValueError("Transformed auxiliary features are invalid")
        return aux


def validate_feature_scaler_provenance(
    scaler: FeatureScaler,
    frame: pd.DataFrame,
    variant: str,
    fold: int,
    *,
    source_path: str | Path,
    schema_path: str | Path,
    upstream_manifest_path: str | Path,
    split_path: str | Path,
    code_paths: Sequence[str | Path],
    config_path: str | Path,
    project_root: str | Path,
) -> None:
    """Validate a persisted scaler without refitting it or changing its hash."""
    variant = validate_variant(variant)
    root = Path(project_root).resolve()
    source = Path(source_path).resolve()
    schema = Path(schema_path).resolve()
    manifest = Path(upstream_manifest_path).resolve()
    split = Path(split_path).resolve()
    config = Path(config_path).resolve()
    expected: dict[str, Any] = {
        "variant": variant,
        "fold": int(fold),
        "feature_names": list(VARIANT_FEATURE_COLUMNS[variant]),
        "source_relative_path": relative_path(source, root),
        "source_sha256": sha256_file(source),
        "schema_relative_path": relative_path(schema, root),
        "schema_sha256": sha256_file(schema),
        "upstream_manifest_relative_path": relative_path(manifest, root),
        "upstream_manifest_sha256": sha256_file(manifest),
        "train_id_sha256": sha256_ids(frame["ID"].astype(str).tolist()),
        "split_sha256": sha256_file(split),
        "code_sha256": code_sha256(code_paths, root),
        "config_sha256": sha256_file(config),
    }
    payload = scaler.to_dict()
    mismatches = {
        key: {"saved": payload.get(key), "current": value}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "Persisted feature scaler provenance does not match the resume run: "
            f"{mismatches}"
        )
    validate_feature_values(frame, variant, source)


def fit_feature_scaler(
    frame: pd.DataFrame,
    variant: str,
    fold: int,
    *,
    source_path: str | Path,
    schema_path: str | Path,
    upstream_manifest_path: str | Path,
    split_path: str | Path,
    code_paths: Sequence[str | Path],
    config_path: str | Path,
    project_root: str | Path,
    std_epsilon: float = STD_EPSILON,
) -> FeatureScaler:
    variant = validate_variant(variant)
    if variant not in {"global_raw", "global_stage2a", "global_stage2b"}:
        raise ValueError(f"{variant} does not fit a six-dimensional scaler")
    validate_feature_values(frame, variant, source_path)
    names = list(VARIANT_FEATURE_COLUMNS[variant])
    values = frame.loc[:, names].to_numpy(np.float64, copy=True)
    availability = pd.to_numeric(frame[AVAILABILITY_COLUMN]).to_numpy(np.float64)
    mean = np.empty(6, dtype=np.float64)
    std = np.empty(6, dtype=np.float64)
    valid_n: list[int] = []
    for index in range(6):
        eligible = np.ones(len(frame), dtype=bool) if index < 3 else availability == 1
        selected = values[eligible, index]
        mean[index] = np.mean(selected, dtype=np.float64)
        std[index] = np.std(selected, ddof=0, dtype=np.float64)
        valid_n.append(int(selected.size))
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Scaler statistics contain NaN or infinity")
    if (std < float(std_epsilon)).any():
        bad = [names[i] for i in np.flatnonzero(std < float(std_epsilon))]
        raise ValueError(f"Feature std is below {std_epsilon}: {bad}")
    root = Path(project_root).resolve()
    source, schema, manifest, split, config = map(
        lambda value: Path(value).resolve(),
        (source_path, schema_path, upstream_manifest_path, split_path, config_path),
    )
    return FeatureScaler(
        schema_version="global_optical_feature_scaler_v1",
        variant=variant,
        fold=int(fold),
        feature_names=names,
        mean=mean.tolist(),
        std=std.tolist(),
        valid_n=valid_n,
        forehead_available_train_n=int((availability == 1).sum()),
        forehead_unavailable_train_n=int((availability == 0).sum()),
        ddof=0,
        std_epsilon=float(std_epsilon),
        missing_fill_after_standardization=0.0,
        availability_position=6,
        source_relative_path=relative_path(source, root),
        source_sha256=sha256_file(source),
        schema_relative_path=relative_path(schema, root),
        schema_sha256=sha256_file(schema),
        upstream_manifest_relative_path=relative_path(manifest, root),
        upstream_manifest_sha256=sha256_file(manifest),
        train_id_sha256=sha256_ids(frame["ID"].astype(str).tolist()),
        split_sha256=sha256_file(split),
        fit_timestamp=datetime.now(timezone.utc).isoformat(),
        code_sha256=code_sha256(code_paths, root),
        config_sha256=sha256_file(config),
    )


def build_aux_features(
    frame: pd.DataFrame | None, variant: str, scaler: FeatureScaler | None = None
) -> np.ndarray:
    variant = validate_variant(variant)
    if variant == "global_only":
        if frame is not None:
            raise ValueError("global_only must not receive a feature frame")
        return np.empty((0, 0), dtype=np.float32)
    if frame is None:
        raise ValueError(f"{variant} requires a feature frame")
    availability = _coerce_and_validate_availability(frame, Path("feature frame"))
    if variant == "global_mask":
        if scaler is not None:
            raise ValueError("global_mask must not use a six-dimensional scaler")
        return availability[:, None].astype(np.float32)
    if scaler is None:
        raise ValueError(f"{variant} requires a fitted scaler")
    if scaler.variant != variant:
        raise ValueError(f"Scaler variant {scaler.variant} does not match {variant}")
    return scaler.transform(frame)


def feature_distribution_rows(
    train_frame: pd.DataFrame, val_frame: pd.DataFrame, variant: str, fold: int
) -> list[dict[str, Any]]:
    """Return raw train/validation distribution diagnostics without mutation."""
    variant = validate_variant(variant)
    names = VARIANT_FEATURE_COLUMNS[variant]
    if not names:
        return []
    train_av = pd.to_numeric(train_frame[AVAILABILITY_COLUMN]).to_numpy(int)
    val_av = pd.to_numeric(val_frame[AVAILABILITY_COLUMN]).to_numpy(int)
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        train_values = pd.to_numeric(train_frame[name], errors="coerce").to_numpy(float)
        val_values = pd.to_numeric(val_frame[name], errors="coerce").to_numpy(float)
        train_valid = train_values[np.isfinite(train_values)]
        val_valid = val_values[np.isfinite(val_values)]
        train_std = float(np.std(train_valid, ddof=0))
        val_mean = float(np.mean(val_valid))
        train_mean = float(np.mean(train_valid))
        rows.append({
            "variant": variant, "fold": int(fold), "feature": name,
            "train_valid_n": int(train_valid.size), "val_valid_n": int(val_valid.size),
            "train_mean": train_mean, "train_std": train_std,
            "train_median": float(np.median(train_valid)), "train_min": float(np.min(train_valid)),
            "train_max": float(np.max(train_valid)), "val_mean": val_mean,
            "val_std": float(np.std(val_valid, ddof=0)), "val_median": float(np.median(val_valid)),
            "val_min": float(np.min(val_valid)), "val_max": float(np.max(val_valid)),
            "standardized_mean_difference": (val_mean - train_mean) / train_std,
            "train_unavailable_n": int((train_av == 0).sum()),
            "val_unavailable_n": int((val_av == 0).sum()),
            "train_availability_ratio": float(train_av.mean()),
            "val_availability_ratio": float(val_av.mean()),
            "forehead_feature": bool(index >= 3),
        })
    return rows
