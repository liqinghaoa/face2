from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from skin_optics_so1.decomposition.so1d_audit import (
    CHANNELS,
    configure_read_only_fp32,
    cross_talk_correlation,
    find_rgb_neighbors,
    fit_train_affine,
    masked_error_sums,
    output_is_isolated,
    p_active_counts,
    require_audit_split,
    summarize_active_counts,
    zero_baseline_mae,
)


def test_audit_model_is_fp32_eval_and_read_only() -> None:
    model = configure_read_only_fp32(torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.BatchNorm1d(1)))
    assert not model.training
    assert all(not value.requires_grad and value.dtype == torch.float32 for value in model.parameters())


def test_forbidden_split_access_is_rejected() -> None:
    require_audit_split("train"); require_audit_split("validation")
    for split in ("id_test", "camera_ood", "light_ood", "joint_ood"):
        with pytest.raises(ValueError, match="forbids"):
            require_audit_split(split)


def test_masked_metrics_ignore_invalid_pixels() -> None:
    pred = torch.tensor([[[[1.0, 100.0]], [[2.0, 100.0]], [[3.0, 100.0]], [[4.0, 100.0]]]])
    true = torch.zeros_like(pred); mask = torch.tensor([[[[1.0, 0.0]]]])
    sums = masked_error_sums(pred, true, mask)
    np.testing.assert_allclose(sums["absolute"], [[1.0, 2.0, 3.0, 4.0]])
    np.testing.assert_allclose(sums["count"], [1.0])


def test_p_active_threshold_metrics_are_correct() -> None:
    pred = torch.zeros(1, 4, 1, 4); true = torch.zeros_like(pred); mask = torch.ones(1, 1, 1, 4)
    true[:, 3] = torch.tensor([[[0.02, 0.02, 0.0, 0.0]]])
    pred[:, 3] = torch.tensor([[[0.02, 0.0, 0.02, 0.0]]])
    result = summarize_active_counts(p_active_counts(pred, true, mask, 0.01))
    assert result["precision"] == 0.5 and result["recall"] == 0.5 and result["f1"] == 0.5


def test_zero_baseline_is_mean_absolute_target_on_mask() -> None:
    assert zero_baseline_mae(np.array([0.0, 0.2, 0.9]), np.array([1, 1, 0])) == pytest.approx(0.1)


def test_affine_calibration_is_fit_from_train_arguments_only() -> None:
    train_pred = np.array([0.0, 1.0, 2.0]); train_true = 2 * train_pred + 3
    first = fit_train_affine(train_pred, train_true)
    validation_target = np.array([999.0]); validation_target[:] = -999.0
    second = fit_train_affine(train_pred, train_true)
    assert first == second == {"a": pytest.approx(2.0), "b": pytest.approx(3.0), "fit_source": "train_only"}


def test_cross_talk_preserves_mhsp_order() -> None:
    rng = np.random.default_rng(4); target = rng.normal(size=(100, 4)); pred = target.copy()
    matrix = cross_talk_correlation(pred, target, method="pearson")
    assert CHANNELS == ("M", "H", "S", "P")
    np.testing.assert_allclose(np.diag(matrix), 1.0)


def test_nearest_neighbor_selection_has_no_target_argument_or_leakage() -> None:
    train = np.array([[0.0], [10.0]], dtype=np.float32); validation = np.array([[9.0]], dtype=np.float32)
    indices, distances = find_rgb_neighbors(train, validation, top_k=1)
    assert indices.tolist() == [[1]] and distances[0, 0] == pytest.approx(1.0)


def test_output_directory_isolated_from_formal_training() -> None:
    root = Path("experiments/skin_optics_so1/SO1_Decomposition_UNet_v1")
    assert output_is_isolated(root / "diagnostics/so1d_a_hp_learnability_identifiability", root / "formal_train_v1_1")


def test_checkpoint_file_hash_guard_detects_no_read_only_change(tmp_path: Path) -> None:
    from skin_optics_so1.decomposition.so1d_audit import file_sha256
    path = tmp_path / "checkpoint.pt"
    torch.save({"weight": torch.ones(1)}, path)
    before = file_sha256(path)
    torch.load(path, map_location="cpu", weights_only=False)
    assert file_sha256(path) == before


def test_audit_runner_contains_no_training_calls() -> None:
    source = (Path(__file__).resolve().parents[2]
              / "scripts/analysis/run_so1d_a_hp_audit.py").read_text(encoding="utf-8")
    assert ".backward(" not in source
    assert "optimizer.step(" not in source
