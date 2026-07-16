from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.evaluate.summarize_global_optical_fusion_5fold import (
    PAIRWISE_COMPARISONS,
    _bootstrap_indices_by_patient_group,
    paired_cluster_bootstrap,
    print_completion_summary,
    scalar_metrics,
    validate_formal_fold_artifacts,
    validate_oof_alignment,
    validate_oof_frame,
)
from scripts.run.run_global_optical_fusion_5fold import training_code_signature
from utils.optical_feature_preprocessor import VARIANTS


def _oof(better: bool = False) -> pd.DataFrame:
    rows = []
    index = 0
    for label in (0, 1, 2):
        for group_number in range(4):
            group = f"g{label}_{group_number}"
            repeats = 2 if label == 1 and group_number == 0 else 1
            for _ in range(repeats):
                base = np.full(3, 0.2 if better else 0.3)
                base[label] = 0.6 if better else 0.4
                base /= base.sum()
                rows.append({
                    "ID": f"id_{index}", "patient_group_id": group,
                    "fold": index % 5, "true_label": label,
                    "prob_normal": base[0], "prob_mild": base[1], "prob_severe": base[2],
                })
                index += 1
    return pd.DataFrame(rows)


def test_validate_oof_and_alignment():
    frame = _oof()
    assert len(validate_oof_frame(frame, expected_rows=len(frame))) == len(frame)
    validate_oof_alignment(frame, frame.copy())
    duplicate = pd.concat([frame.iloc[:-1], frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="unique"):
        validate_oof_frame(duplicate, expected_rows=len(frame))
    bad_probability = frame.copy()
    bad_probability.loc[0, "prob_normal"] = 0.99
    with pytest.raises(ValueError, match="sum"):
        validate_oof_frame(bad_probability, expected_rows=len(frame))


@pytest.mark.parametrize("column", ["fold", "true_label", "patient_group_id", "ID"])
def test_pair_alignment_mismatch_fails(column):
    left, right = _oof(), _oof()
    right.loc[0, column] = "changed" if column in {"ID", "patient_group_id"} else 2
    if right.loc[0, column] == left.loc[0, column]:
        right.loc[0, column] = 1
    with pytest.raises(ValueError, match="alignment"):
        validate_oof_alignment(left, right)


def test_metric_delta_direction_and_bootstrap_reproducibility():
    reference, candidate = _oof(False), _oof(True)
    assert scalar_metrics(candidate)["macro_auc"] >= scalar_metrics(reference)["macro_auc"]
    first, first_audit = paired_cluster_bootstrap(
        candidate, reference, repetitions=40, seed=2026, minimum_valid_repetitions=40
    )
    second, second_audit = paired_cluster_bootstrap(
        candidate, reference, repetitions=40, seed=2026, minimum_valid_repetitions=40
    )
    pd.testing.assert_frame_equal(first, second)
    assert first_audit == second_audit
    assert (first["delta_mean"] >= 0).all()
    assert first_audit["valid_repetitions"] == 40


def test_bootstrap_keeps_all_images_of_sampled_patient_group():
    frame = _oof()
    indices = _bootstrap_indices_by_patient_group(frame, np.random.default_rng(8))
    selected = frame.iloc[indices]
    source_counts = frame.groupby("patient_group_id").size()
    selected_counts = selected.groupby("patient_group_id").size()
    for group, count in selected_counts.items():
        assert count % source_counts[group] == 0


def test_bootstrap_supports_mixed_label_patient_groups():
    reference, candidate = _oof(False), _oof(True)
    mixed_rows = reference.index[reference["patient_group_id"].eq("g1_0")].tolist()
    reference.loc[mixed_rows[0], "true_label"] = 2
    candidate.loc[mixed_rows[0], "true_label"] = 2

    indices = _bootstrap_indices_by_patient_group(reference, np.random.default_rng(9))
    sampled = reference.iloc[indices]
    assert sampled["true_label"].value_counts().sort_index().to_dict() == (
        reference["true_label"].value_counts().sort_index().to_dict()
    )
    result, audit = paired_cluster_bootstrap(
        candidate, reference, repetitions=20, seed=2026,
        minimum_valid_repetitions=20,
    )
    assert (result["valid_repetitions"] == 20).all()
    assert audit["mixed_true_label_patient_group_count"] == 1
    assert audit["stratification_rule"] == "patient_group_true_label_count_composition"
    assert audit["preserves_exact_image_class_counts"] is True


def test_complete_manifest_cannot_hide_missing_fold_artifacts(tmp_path):
    fold_dir = tmp_path / "global_only/fold_0"
    fold_dir.mkdir(parents=True)
    manifest = {
        "status": "COMPLETE", "formal_result": True,
        "variant": "global_only", "fold": 0,
        "implementation_signature": training_code_signature(),
    }
    with pytest.raises(FileNotFoundError, match="incomplete"):
        validate_formal_fold_artifacts(fold_dir, "global_only", 0, manifest)


def test_recorded_fold_signature_can_be_validated_after_summary_only_fix(tmp_path):
    fold_dir = tmp_path / "global_only/fold_0"
    fold_dir.mkdir(parents=True)
    manifest = {
        "status": "COMPLETE", "formal_result": True,
        "variant": "global_only", "fold": 0,
        "implementation_signature": "recorded-training-signature",
    }
    with pytest.raises(FileNotFoundError, match="incomplete"):
        validate_formal_fold_artifacts(
            fold_dir, "global_only", 0, manifest,
            expected_implementation_signature="recorded-training-signature",
        )
    with pytest.raises(ValueError, match="stale implementation"):
        validate_formal_fold_artifacts(
            fold_dir, "global_only", 0, manifest,
            expected_implementation_signature="different-signature",
        )


def test_formal_terminal_output_contains_required_fields(tmp_path, capsys):
    summary = tmp_path / "summary"
    summary.mkdir()
    pd.DataFrame([
        {
            "variant": variant, "macro_auc": 0.7, "accuracy": 0.6,
            "balanced_accuracy": 0.59, "macro_f1": 0.58,
        }
        for variant in VARIANTS
    ]).to_csv(summary / "oof_metrics_all_variants.csv", index=False)
    for variant in VARIANTS:
        output = summary / variant
        output.mkdir()
        pd.DataFrame({"ID": ["a", "b"]}).to_csv(output / "oof_predictions.csv", index=False)
        pd.DataFrame({"fold": range(5), "macro_auc": [0.7] * 5}).to_csv(
            output / "fold_metrics.csv", index=False
        )
    pd.DataFrame([
        {
            "candidate": candidate, "reference": reference, "metric": "macro_auc",
            "oof_delta_candidate_minus_reference": 0.01,
            "candidate_better_folds": 3, "reference_better_folds": 2,
            "equal_folds": 0,
        }
        for candidate, reference in PAIRWISE_COMPARISONS
    ]).to_csv(summary / "pairwise_comparison.csv", index=False)
    pd.DataFrame([
        {
            "candidate": candidate, "reference": reference, "metric": "macro_auc",
            "valid_repetitions": 2000, "ci_lower_2_5": -0.01,
            "ci_upper_97_5": 0.03,
        }
        for candidate, reference in PAIRWISE_COMPARISONS
    ]).to_csv(summary / "pairwise_bootstrap_deltas.csv", index=False)
    pd.DataFrame({
        "variant": ["global_stage2b", "global_stage2b"],
        "standardized_mean_difference": [0.1, -0.2],
    }).to_csv(summary / "feature_distribution_audit.csv", index=False)
    (summary / "run_manifest.json").write_text(
        '{"completed_variants":[1,2,3,4,5],"tests":{"status":"PASS","passed_count":124}}',
        encoding="utf-8",
    )
    (summary / "experiment_summary.json").write_text(
        '{"status":"COMPLETE"}', encoding="utf-8"
    )
    print_completion_summary(summary)
    output = capsys.readouterr().out
    assert "GLOBAL_OPTICAL_FUSION_EXPERIMENT_STATUS=COMPLETE" in output
    assert "COMPLETED_FOLD_RUNS=25" in output
    assert "GLOBAL_ONLY_OOF_ROWS=2;UNIQUE_IDS=2" in output
    assert "GLOBAL_MASK_MINUS_GLOBAL_ONLY=" in output
    assert "bootstrap_valid_repetitions=2000" in output
    assert "STAGE2B_DISTRIBUTION_SHIFT=" in output
    assert "TEST_STATUS=PASS;passed=124" in output
    assert "ALL_ACCEPTANCE_CONDITIONS_MET=true" in output
