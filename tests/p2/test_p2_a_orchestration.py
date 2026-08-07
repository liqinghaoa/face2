from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

import p2_counterfactual.orchestration as orchestration
from p2_counterfactual.orchestration import dry_run_plan, summarize_pipeline


def test_dry_run_lists_first_stage_five_fold_without_training() -> None:
    plan = dry_run_plan()
    assert plan["training_started"] is False
    assert len(plan["commands"]) == 10
    assert plan["first_stage_experiment_order"] == ["p2_a1_colorjitter", "p2_a2_relighting"]
    assert "gate" in plan["gate_position"]


def test_manifest_hash_is_frozen_for_orchestration_tests() -> None:
    manifest = Path("data/processed/P2_Counterfactual_Relighting500_v1/manifests/p2_training_manifest.csv")
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == "f181449d248ffc078dc8e2743df10e0764c4c0ab747220c9ffcfb0c8f6f2406c"


def test_summarize_writes_bootstrap_alias_and_preserves_formal_gate_on_force_a3(tmp_path: Path, monkeypatch) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("case_id,patient_group_id,fold,binary_label\n", encoding="utf-8")
    configs = {
        exp: {"manifest_path": str(manifest), "gate": orchestration.DEFAULT_GATE_CONFIG, "output_root": str(tmp_path / "rgb_benchmark"), "experiment_id": exp}
        for exp in orchestration.FIRST_STAGE_EXPERIMENTS
    }
    model_df = pd.DataFrame(
        [
            {"experiment_id": "p2_a1_colorjitter", "macro_auc": 0.85, "accuracy": 0.75, "macro_precision": 0.75, "macro_recall": 0.75, "macro_f1": 0.75, "balanced_accuracy": 0.75, "patient_sensitivity": 0.85, "control_specificity": 0.65},
            {"experiment_id": "p2_a2_relighting", "macro_auc": 0.80, "accuracy": 0.72, "macro_precision": 0.72, "macro_recall": 0.72, "macro_f1": 0.71, "balanced_accuracy": 0.72, "patient_sensitivity": 0.82, "control_specificity": 0.62},
        ]
    )
    stability_df = pd.DataFrame(
        [
            {"experiment_id": "p2_a1_colorjitter", "mean_prediction_std": 0.2, "median_prediction_std": 0.2, "case_level_flip_rate": 0.3, "worst_light_auc": 0.55, "mean_relighted_auc": 0.56, "auc_range": 0.02, "mean_feature_cosine": 0.8, "median_feature_cosine": 0.8},
            {"experiment_id": "p2_a2_relighting", "mean_prediction_std": 0.1, "median_prediction_std": 0.1, "case_level_flip_rate": 0.2, "worst_light_auc": 0.58, "mean_relighted_auc": 0.59, "auc_range": 0.01, "mean_feature_cosine": 0.75, "median_feature_cosine": 0.75},
        ]
    )
    paired = pd.DataFrame(
        [
            {
                "comparison": "p2_a2_relighting vs p2_a1_colorjitter",
                "metric": "macro_auc",
                "model_a": "p2_a1_colorjitter",
                "model_b": "p2_a2_relighting",
                "value_a": 0.85,
                "value_b": 0.80,
                "difference_b_minus_a": -0.05,
                "ci_lower": -0.1,
                "ci_upper": -0.02,
                "bootstrap_iterations": 2000,
            }
        ]
    )
    monkeypatch.setattr(orchestration, "_model_and_stability_rows", lambda *args, **kwargs: (model_df, stability_df))
    monkeypatch.setattr(orchestration, "_run_comparisons", lambda *args, **kwargs: paired)

    result = summarize_pipeline(
        experiment_ids=list(orchestration.FIRST_STAGE_EXPERIMENTS),
        configs=configs,
        summary_dir=tmp_path,
        pipeline_smoke=False,
        bootstrap_iterations=2000,
        bootstrap_seed=2026,
        force_a3=True,
    )

    gate = json.loads((tmp_path / "p2_a_gate_decision.json").read_text(encoding="utf-8"))
    assert gate["gate_pass"] is False
    assert gate["a3_should_run"] is False
    assert "force_a3_requested" not in gate
    override = json.loads((tmp_path / "p2_a_force_a3_override.json").read_text(encoding="utf-8"))
    assert override["status"] == "NON_FORMAL_FORCE_A3_REQUESTED"
    assert override["effective_a3_run_requested"] is True
    assert result["force_a3_override"] == override


def test_run_comparisons_writes_required_bootstrap_alias(tmp_path: Path, monkeypatch) -> None:
    output_root = tmp_path / "rgb_benchmark"
    configs = {}
    for exp in ["p2_a1_colorjitter", "p2_a2_relighting"]:
        configs[exp] = {"output_root": str(output_root), "experiment_id": exp}
        exp_dir = output_root / exp
        exp_dir.mkdir(parents=True)
        (exp_dir / "_EXPERIMENT_SUCCESS.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(orchestration, "load_experiment_frames", lambda exp_dir, exp: object())
    monkeypatch.setattr(
        orchestration,
        "paired_cluster_bootstrap_comparison",
        lambda *args, **kwargs: [
            {
                "comparison": "p2_a2_relighting vs p2_a1_colorjitter",
                "metric": "macro_auc",
                "model_a": "p2_a1_colorjitter",
                "model_b": "p2_a2_relighting",
                "value_a": 0.8,
                "value_b": 0.81,
                "difference_b_minus_a": 0.01,
                "ci_lower": -0.01,
                "ci_upper": 0.03,
                "bootstrap_iterations": 2000,
            }
        ],
    )

    frame = orchestration._run_comparisons(
        experiment_ids=["p2_a1_colorjitter", "p2_a2_relighting"],
        configs=configs,
        summary_dir=tmp_path,
        pipeline_smoke=False,
        iterations=2000,
        seed=2026,
    )

    assert len(frame) == 1
    assert (tmp_path / "p2_a_cluster_bootstrap.csv").is_file()
    assert (tmp_path / "cluster_bootstrap_comparisons.csv").is_file()
