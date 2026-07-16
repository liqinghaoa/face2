from __future__ import annotations

import json
import copy
from pathlib import Path

import numpy as np  # Preload NumPy before torch for the local Anaconda/MKL runtime.
import pytest
import torch

import scripts.run.run_global_optical_fusion_5fold as fusion_runner
from scripts.run.run_global_optical_fusion_5fold import run_preflight, validate_locked_config
from trainers.global_optical_fusion_trainer import (
    epoch_augmentation_seed,
    make_data_generator,
    seed_payload,
)
from utils.experiment_utils import load_yaml, set_random_seed
from utils.optical_feature_preprocessor import VARIANT_AUX_DIM, VARIANTS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "config/train/global_optical_fusion/global_resnet18_optical_fusion.yaml"

assert np.__version__


def test_locked_config_and_variant_dimensions():
    config = load_yaml(CONFIG)
    validate_locked_config(config)
    assert tuple(config["experiment"]["variants"]) == VARIANTS
    assert dict(VARIANT_AUX_DIM) == {
        "global_only": 0, "global_mask": 1, "global_raw": 7,
        "global_stage2a": 7, "global_stage2b": 7,
    }


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("transforms", "mean", [0.0, 0.0, 0.0]),
        ("transforms", "std", [1.0, 1.0, 1.0]),
        ("experiment", "output_root", "experiments/optical_condition_calibration_stage2a"),
        ("data", "image_root", "data/processed/another_500_images"),
        ("data", "expected_class_counts", [114, 238, 148]),
        ("features", "stage2a_root", "experiments/other_stage2a"),
        ("summary", "bootstrap_repetitions", 100),
    ],
)
def test_locked_config_rejects_protocol_drift(section, key, value):
    config = copy.deepcopy(load_yaml(CONFIG))
    config[section][key] = value
    with pytest.raises(ValueError, match="Locked config mismatch"):
        validate_locked_config(config)


def test_real_metadata_preflight(tmp_path):
    manifest = run_preflight(CONFIG, tmp_path / "experiment")
    assert manifest["status"] == "PASS"
    assert manifest["cohort_size"] == 500
    assert manifest["class_counts"] == [115, 237, 148]
    assert manifest["patient_group_count"] == 483
    assert manifest["multi_image_patient_group_count"] == 17
    assert manifest["availability_counts"] == {"available": 486, "unavailable": 14}
    assert manifest["oof_used_as_classifier_train"] is False
    protocol = tmp_path / "experiment/protocol"
    assert {path.name for path in protocol.iterdir()} == {
        "environment_audit.json", "input_audit.csv", "feature_source_audit.csv",
        "fold_alignment_audit.csv", "schema_audit.json", "protocol_manifest.json",
    }
    saved = json.loads((protocol / "protocol_manifest.json").read_text(encoding="utf-8"))
    assert saved["critical_checks"] == 27
    assert saved["full_training_executed"] is False


def test_git_metadata_is_optional_when_git_is_not_installed(monkeypatch):
    def missing_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(fusion_runner.subprocess, "run", missing_git)
    metadata = fusion_runner._collect_git_metadata()
    assert metadata["git_available"] is False
    assert metadata["git_repository"] is False
    assert metadata["git_branch"] is None
    assert metadata["git_commit"] is None
    assert metadata["git_error"] == "Git executable was not found on PATH"


def test_rng_streams_are_variant_order_independent():
    orders = []
    flips = []
    for variant in reversed(VARIANTS):
        seeds = seed_payload(2026, 2)
        orders.append(torch.randperm(40, generator=make_data_generator(seeds["shuffle_seed"])).tolist())
        set_random_seed(epoch_augmentation_seed(seeds, 3))
        flips.append((torch.rand(40) < 0.5).tolist())
    assert all(order == orders[0] for order in orders)
    assert all(decisions == flips[0] for decisions in flips)
    assert seed_payload(2026, 1) != seed_payload(2026, 2)
    assert seed_payload(2026, 2) == seed_payload(2026, 2)
