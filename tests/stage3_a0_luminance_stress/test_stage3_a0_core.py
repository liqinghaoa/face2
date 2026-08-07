from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from stage3_a0_luminance_stress.checkpoint_resolver import resolve_checkpoints
from stage3_a0_luminance_stress.color_transform import apply_exposure_ev_uint8, apply_gamma_uint8, linear_to_srgb, srgb_to_linear
from stage3_a0_luminance_stress.config import Stage3A0Config
from stage3_a0_luminance_stress.r0_config import Stage3A0R0Config
from stage3_a0_luminance_stress.r0_discrepancy_analysis import build_discrepancy_frame, grouped_stats, overall_stats


ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "configs/stage3_a0_frozen_rgb_paired_exposure_gamma_stress_v1.yaml"
R0_CFG = ROOT / "configs/stage3_a0_r0_oof_reproduction_discrepancy_audit_v1.yaml"


def test_config_fixed_gates() -> None:
    cfg = Stage3A0Config.from_yaml(CFG)
    assert cfg.run_training is False
    assert cfg.stop_after_stress_test is True
    assert cfg.classification_threshold == 0.5
    assert cfg.exposure_ev_levels == [-1.0, -0.5, 0.0, 0.5, 1.0]
    assert cfg.gamma_levels == [0.8, 1.2]


def test_checkpoint_resolution_unique_when_available() -> None:
    cfg = Stage3A0Config.from_yaml(CFG)
    if cfg.rgb_oof_csv.is_file():
        ckpts = resolve_checkpoints(cfg.rgb_oof_csv)
        assert ckpts["fold"].tolist() == [0, 1, 2, 3, 4]
        assert ckpts["checkpoint_sha256"].str.len().eq(64).all()
        assert ckpts["checkpoint_path"].is_unique


def test_srgb_roundtrip() -> None:
    x = np.linspace(0, 1, 17, dtype=np.float32)
    assert np.allclose(linear_to_srgb(srgb_to_linear(x)), x, atol=1e-6)


def test_exposure_ev_scaling_direction_and_noop() -> None:
    rgb = np.full((4, 4, 3), 100, dtype=np.uint8)
    mask = np.full((4, 4), 255, dtype=np.uint8)
    assert np.array_equal(apply_exposure_ev_uint8(rgb, mask, 0.0), rgb)
    assert apply_exposure_ev_uint8(rgb, mask, -0.5).mean() < rgb.mean()
    assert apply_exposure_ev_uint8(rgb, mask, 0.5).mean() > rgb.mean()


def test_gamma_direction() -> None:
    rgb = np.full((4, 4, 3), 100, dtype=np.uint8)
    mask = np.full((4, 4), 255, dtype=np.uint8)
    assert apply_gamma_uint8(rgb, mask, 0.8).mean() > rgb.mean()
    assert apply_gamma_uint8(rgb, mask, 1.2).mean() < rgb.mean()


def test_mask_background_remains_zero() -> None:
    rgb = np.full((4, 4, 3), 100, dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[:2, :2] = 255
    out = apply_exposure_ev_uint8(rgb, mask, 1.0)
    assert np.count_nonzero(out[mask == 0]) == 0
    out2 = apply_gamma_uint8(rgb, mask, 0.8)
    assert np.count_nonzero(out2[mask == 0]) == 0


def test_no_training_tokens_in_stage3_package() -> None:
    src = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "stage3_a0_luminance_stress").glob("*.py"))
    assert "torch.optim" not in src
    assert ".train(" not in src


def test_reproduction_outputs_when_present() -> None:
    root = ROOT / "experiments/lighting_confounding/Stage3_A0_FrozenRGB_Paired_ExposureGamma_Stress_v1"
    path = root / "reproduction/original_oof_comparison.csv"
    if path.is_file():
        df = pd.read_csv(path)
        assert len(df) == 500
        assert {"stored_probability", "reproduced_probability", "absolute_difference"}.issubset(df.columns)


def test_r0_config_fixed_audit_only_gates() -> None:
    cfg = Stage3A0R0Config.from_yaml(R0_CFG)
    assert cfg.mode == "stage3_a0_r0"
    assert cfg.run_counterfactual_inference is False
    assert cfg.run_training is False
    assert cfg.modify_checkpoints is False
    assert cfg.classification_threshold == 0.5
    assert cfg.stop_after_r0 is True


def test_r0_difference_formulas_and_overall_stats() -> None:
    stored = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "patient_group_id": ["p1", "p2", "p3"],
            "fold": [0, 0, 1],
            "binary_label": [0, 1, 1],
            "prob_patient": [0.2, 0.7, 0.8],
            "pred_class": [0, 1, 1],
        }
    )
    reproduced = pd.DataFrame({"sample_id": ["a", "b", "c"], "prob_patient": [0.25, 0.65, 0.82], "pred_class": [0, 1, 1]})
    camera = pd.DataFrame({"sample_id": ["a", "b", "c"], "camera_model": ["X", "X", "Y"]})
    frame = build_discrepancy_frame(stored, reproduced, camera, 0.5)
    assert np.allclose(frame["signed_difference"], [0.05, -0.05, 0.02])
    assert np.allclose(frame["absolute_difference"], [0.05, 0.05, 0.02])
    stats = overall_stats(frame)
    assert stats["prediction_match_count"] == 3
    assert np.isclose(stats["max_absolute_difference"], 0.05)
    assert np.isclose(stats["p95_absolute_difference"], np.quantile([0.05, 0.05, 0.02], 0.95))


def test_r0_grouped_stats_by_fold() -> None:
    stored = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "patient_group_id": ["p1", "p2", "p3", "p4"],
            "fold": [0, 0, 1, 1],
            "binary_label": [0, 1, 0, 1],
            "prob_patient": [0.2, 0.7, 0.4, 0.9],
            "pred_class": [0, 1, 0, 1],
        }
    )
    reproduced = pd.DataFrame({"sample_id": ["a", "b", "c", "d"], "prob_patient": [0.3, 0.6, 0.45, 0.91], "pred_class": [0, 1, 0, 1]})
    camera = pd.DataFrame({"sample_id": ["a", "b", "c", "d"], "camera_model": ["X", "X", "Y", "Y"]})
    frame = build_discrepancy_frame(stored, reproduced, camera, 0.5)
    by_fold = grouped_stats(frame, ["fold"]).sort_values("fold")
    assert by_fold["n"].tolist() == [2, 2]
    assert np.allclose(by_fold["mean_absolute_difference"], [0.1, 0.03])


def test_r0_outputs_when_present() -> None:
    root = ROOT / "experiments/lighting_confounding/Stage3_A0_R0_OOF_Reproduction_Discrepancy_Audit_v1"
    gate = root / "decision/stage3_a0_resume_gate.json"
    if gate.is_file():
        payload = json.loads(gate.read_text(encoding="utf-8"))
        assert payload["reproduction_status"] in {"strict_pass", "numerical_equivalence_pass", "fail"}
        assert "resume_allowed" in payload
        assert "numerical_equivalence_conditions" in payload
        discrepancy = pd.read_csv(root / "discrepancy/oof_probability_discrepancy_500.csv")
        assert len(discrepancy) == 500
        assert {"signed_difference", "absolute_difference", "camera_model"}.issubset(discrepancy.columns)
