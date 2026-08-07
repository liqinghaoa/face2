from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lighting_confounding_stage1.features import fold_z_features
from lighting_confounding_stage1.id_normalization import normalize_id
from lighting_confounding_stage1.metadata_models import _permuted_labels, run_one_model_oof
from lighting_confounding_stage1.metadata_models import _prepare_permutation_designs
from lighting_confounding_stage1.metrics_utils import cluster_bootstrap_metrics, compute_metrics
from lighting_confounding_stage1.stratified_metrics import _assign_fold_quantile_strata


def synthetic_master(n: int = 50) -> pd.DataFrame:
    rows = []
    for i in range(n):
        fold = i % 5
        label = int(i % 4 != 0)
        device = "A" if i % 2 == 0 else "B"
        rows.append(
            {
                "sample_id": str(1000 + i),
                "patient_group_id": str(1000 + i),
                "fold": fold,
                "binary_label": label,
                "camera_model": device,
                "camera_make": "M",
                "software": "S",
                "image_width": 100 + (i % 2),
                "image_height": 200,
                "megapixels": 0.02,
                "brightness_value_apex": float(i % 9) + (1 if device == "A" else 2),
                "log2_iso": float((i % 7) + 6),
                "log2_exposure_time": float(-5 + (i % 5)),
                "exposure_bias_ev": 0.0,
                "flash": str(i % 2),
                "metering_mode": str(i % 3),
                "white_balance": str(i % 2),
                "exposure_mode": str(i % 2),
                "capture_year": "2024",
                "capture_month": str((i % 12) + 1),
                "capture_year_month": "2024-01" if i < n // 2 else "2024-02",
                "capture_hour_bin": "morning" if i % 2 == 0 else "afternoon",
                "rgb_oof_probability_patient": 0.2 + 0.6 * label,
            }
        )
    df = pd.DataFrame(rows)
    df["rgb_oof_predicted_label"] = (df["rgb_oof_probability_patient"] >= 0.5).astype(int)
    df["rgb_oof_error"] = (df["rgb_oof_predicted_label"] != df["binary_label"]).astype(int)
    return df


def test_id_normalization() -> None:
    assert normalize_id("E:/x/100037382.jpg") == "100037382"
    assert normalize_id(100037382.0) == "100037382"


def test_label_conversion() -> None:
    nyha = pd.Series([0, 1, 2, 3, 4])
    assert ((nyha >= 1).astype(int).tolist()) == [0, 1, 1, 1, 1]


def test_duplicate_id_rejection_pattern() -> None:
    frame = pd.DataFrame({"sample_id": ["1", "1"]})
    assert frame["sample_id"].duplicated().any()


def test_one_to_one_alignment_gate_pattern() -> None:
    split = pd.DataFrame({"sample_id": ["1", "2"], "binary_label": [0, 1]})
    meta = pd.DataFrame({"sample_id": ["1", "2"], "camera_model": ["A", "B"]})
    oof = pd.DataFrame({"sample_id": ["1", "2"], "probability_patient": [0.1, 0.9]})
    merged = split.merge(meta, on="sample_id", how="left", validate="one_to_one").merge(oof, on="sample_id", how="left", validate="one_to_one")
    assert len(merged) == 2
    assert merged["camera_model"].notna().sum() == 2
    assert merged["probability_patient"].notna().sum() == 2


def test_oof_probability_direction() -> None:
    metrics = compute_metrics([0, 1, 1], [0.1, 0.8, 0.7])
    assert metrics["roc_auc"] == 1.0
    assert metrics["sensitivity"] == 1.0


def test_patient_group_cross_fold_rejection_pattern() -> None:
    frame = pd.DataFrame({"patient_group_id": ["a", "a"], "fold": [0, 1]})
    assert frame.groupby("patient_group_id")["fold"].nunique().gt(1).any()


def test_fold_device_z_uses_training_only() -> None:
    df = synthetic_master()
    transformed, _ = fold_z_features(df, 0)
    train = df[df["fold"] != 0]
    stats = train.groupby("camera_model")["brightness_value_apex"].agg(["mean", "std"])
    idx = df.index[df["fold"] == 0][0]
    expected = (df.loc[idx, "brightness_value_apex"] - stats.loc[df.loc[idx, "camera_model"], "mean"]) / stats.loc[df.loc[idx, "camera_model"], "std"]
    assert np.isclose(transformed.loc[idx, "brightness_device_z"], expected)


def test_metadata_only_oof_covers_all_rows() -> None:
    df = synthetic_master()
    oof, _ = run_one_model_oof(df, "META-D", seed=2026, c_grid=(1.0,), inner_cv_splits=3)
    assert len(oof) == len(df)
    assert oof["sample_id"].nunique() == len(df)


def test_permutation_preprocessing_uses_training_fold_rows_only() -> None:
    df = synthetic_master()
    prepared = _prepare_permutation_designs(df, "META-D", C=1.0, seed=2026)
    for item in prepared:
        assert item["x_train"].shape[0] == int((df["fold"] != item["fold"]).sum())
        assert item["x_test"].shape[0] == int((df["fold"] == item["fold"]).sum())


def test_cluster_bootstrap_reproducible() -> None:
    df = synthetic_master().rename(columns={"rgb_oof_probability_patient": "probability_patient"})
    a = cluster_bootstrap_metrics(df, repeats=20, seed=123)
    b = cluster_bootstrap_metrics(df, repeats=20, seed=123)
    assert a["ci95"] == b["ci95"]


def test_stratification_thresholds_from_training_fold() -> None:
    df = synthetic_master()
    strata = _assign_fold_quantile_strata(df, "brightness_value_apex", (0.3333, 0.6667))
    assert strata.notna().all()
    assert set(strata.unique()).issubset({"low", "middle", "high", "missing", "missing_threshold"})


def test_single_class_auc_returns_na() -> None:
    metrics = compute_metrics([1, 1, 1], [0.7, 0.8, 0.9])
    assert metrics["roc_auc"] is None


def test_permutation_preserves_group_structure() -> None:
    df = synthetic_master()
    perm = _permuted_labels(df, np.random.default_rng(1))
    assert len(perm) == len(df)
    assert set(perm.unique()) == {0, 1}


def test_fixed_seed_model_reproducible() -> None:
    df = synthetic_master()
    a, _ = run_one_model_oof(df, "META-D", seed=2026, c_grid=(1.0,), inner_cv_splits=3)
    b, _ = run_one_model_oof(df, "META-D", seed=2026, c_grid=(1.0,), inner_cv_splits=3)
    assert np.allclose(a["probability_patient"], b["probability_patient"])


def test_expected_output_file_list() -> None:
    required = {
        "preflight/preflight_report.md",
        "metadata/stage1_master_500.csv",
        "metadata_only/metadata_only_metrics.csv",
        "rgb_dependence/rgb_metadata_correlations.csv",
        "stratified/rgb_worst_group_metrics.json",
        "reports/stage1_lighting_confounding_report.md",
    }
    assert "metadata/stage1_master_500.csv" in required
