from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts" / "p1"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_p1_extension_abc as abc  # noqa: E402
from models.p1_independent_dual_resnet18 import IndependentDualResNet18  # noqa: E402


def test_json_safe_handles_numpy_scalars() -> None:
    payload = abc._json_safe({"flag": np.bool_(True), "value": np.float32(0.5), "items": [np.int64(3)]})
    encoded = json.dumps(payload)
    assert '"flag": true' in encoded
    assert payload["value"] == 0.5
    assert payload["items"] == [3]


def test_independent_dual_resnet18_has_independent_encoders_and_bn() -> None:
    model = IndependentDualResNet18(pretrained=False)
    assert model.encoder_primary is not model.encoder_secondary
    assert model.classifier.in_features == 1024
    primary_bn = [module for module in model.encoder_primary.modules() if isinstance(module, torch.nn.BatchNorm2d)]
    secondary_bn = [module for module in model.encoder_secondary.modules() if isinstance(module, torch.nn.BatchNorm2d)]
    assert primary_bn
    assert len(primary_bn) == len(secondary_bn)
    assert all(a is not b for a, b in zip(primary_bn, secondary_bn))
    assert all(a.weight is not b.weight for a, b in zip(primary_bn, secondary_bn))


def test_rgb_rgb_collate_uses_identical_augmented_tensors_for_both_branches() -> None:
    collate = abc.build_dual_collate("rgb_rgb_capacity_control", normalization_state=None, training=True)
    sample = {
        "case_id": "case_1",
        "patient_group_id": "patient_1",
        "fold": 0,
        "label_original": 0,
        "label_3class": 0,
        "label_binary": 0,
        "rgb": np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5) / 100.0,
        "secondary_sample": None,
    }
    batch = collate([sample])
    assert torch.equal(batch["primary"], batch["secondary"])


def test_complementarity_counts_correction_and_harm_by_stratum() -> None:
    base = pd.DataFrame(
        {
            "case_id": ["1", "2", "3", "4"],
            "label_binary": [0, 0, 1, 1],
            "pred_binary": [0, 1, 0, 1],
            "prob_patient": [0.1, 0.8, 0.2, 0.9],
        }
    )
    other = pd.DataFrame(
        {
            "case_id": ["1", "2", "3", "4"],
            "pred_binary": [1, 0, 1, 1],
            "prob_patient": [0.7, 0.2, 0.8, 0.9],
        }
    )
    rows = pd.DataFrame(abc._complementarity_rows("a", base, "b", other))
    all_row = rows[rows["stratum"] == "all"].iloc[0]
    assert all_row["correction_count"] == 2
    assert all_row["harm_count"] == 1
    assert all_row["net_correction_count"] == 1
    control = rows[rows["stratum"] == "control"].iloc[0]
    patient = rows[rows["stratum"] == "patient"].iloc[0]
    assert control["correction_count"] == 1
    assert control["harm_count"] == 1
    assert patient["correction_count"] == 1
    assert patient["harm_count"] == 0


def test_stratified_auc_only_compares_within_stratum_pairs() -> None:
    frame = pd.DataFrame(
        {
            "label_binary": [1, 0, 1, 0],
            "prob_patient": [0.9, 0.8, 0.2, 0.1],
            "stratum": ["a", "a", "b", "b"],
        }
    )
    result = abc._stratified_auc(frame, "stratum")
    assert result["valid_strata"] == 2
    assert result["valid_pairs"] == 2
    assert result["within_stratum_weighted_auc"] == 1.0


def test_protocol_source_blocks_disallowed_search_and_stage_d_autostart() -> None:
    source = (ROOT / "scripts" / "p1" / "run_p1_extension_abc.py").read_text(encoding="utf-8")
    assert '"stage_d_auto_start": False' in source
    assert '"threshold_optimization": False' in source
    assert '"hyperparameter_search": False' in source
    assert 'LogisticRegression(penalty="l2", C=1.0, class_weight="balanced", solver="liblinear", max_iter=5000, random_state=SEED)' in source
    assert 'mode="a"' not in source
