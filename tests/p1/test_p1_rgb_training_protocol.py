import json
from pathlib import Path

import torch

from losses.classification_losses import compute_class_weights
from scripts.train.train_p1_rgb_p0aligned_resnet18_5fold import make_model

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments/500Data/P1_RGB_P0Aligned_ResNet18_5fold_v1"


def test_resnet18_full_finetune_and_fold_class_weight_formula():
    model = make_model()
    assert model.fc.out_features == 2
    assert all(p.requires_grad for p in model.parameters())
    weights = compute_class_weights([0] * 92 + [1] * 308, 2)
    assert torch.allclose(weights, torch.tensor([400 / (2 * 92), 400 / (2 * 308)]))


def test_corrected_protocol_keeps_training_fields_but_replaces_evaluation_unit():
    old_protocol = json.loads((EXP / "metadata/p1_component_training_protocol_v1.json").read_text())
    protocol = json.loads((EXP / "metadata/p1_component_training_protocol_v1_1.json").read_text())
    assert old_protocol["evaluation_protocol_deprecated"] is True
    assert protocol["checkpoint_selection"] == "case_level_macro_auc max"
    assert protocol["amp"] is False
    assert protocol["optimizer"] == "AdamW"
    assert protocol["learning_rate"] == 1e-4
    assert protocol["primary_evaluation_level"] == "visit_case"
    assert protocol["split_group_unit"] == "patient_group_id"
    assert protocol["bootstrap_cluster_unit"] == "patient_group_id"
    assert protocol["within_cluster_prediction_aggregation"] == "none"
    assert protocol["longitudinal_labels_preserved"] is True
    assert protocol["same_patient_visits_may_have_different_labels"] is True
    assert protocol["patient_group_id_used_only_for_split_and_resampling"] is True
