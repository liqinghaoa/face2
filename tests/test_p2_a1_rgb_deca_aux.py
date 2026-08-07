from pathlib import Path

import numpy as np
import torch

from datasets.control_patient_binary_dataset import map_three_class_to_binary
from datasets.nyha_3class_face_dataset import build_transforms
from datasets.p2_a1_rgb_deca_aux_dataset import (
    FEATURE_ORDER,
    P2A1Dataset,
    read_frozen_auxiliary_vector,
    split_fold_zero,
    train_stats,
)
from losses.classification_losses import compute_class_weights
from models.p2_a1_rgb_deca_residual_fusion import P2A1Fusion
from utils.experiment_utils import load_yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_yaml(ROOT / "config/p2/p2_a1_binary_rgb_deca_aux_fold0_v1.yaml")


def make_dataset(train: bool, stats=None) -> P2A1Dataset:
    return P2A1Dataset(
        ROOT / CONFIG["data"]["split_table"], ROOT / CONFIG["data"]["image_root"],
        ROOT / CONFIG["assets"]["p1_root"], CONFIG["p1"], 0, train,
        build_transforms("train" if train else "val", horizontal_flip=train), stats,
    )


def test_fixed_protocol_and_real_fold_zero_split_contract():
    assert CONFIG["task"]["type"] == "binary_control_vs_patient"
    assert CONFIG["task"]["num_classes"] == 2 and CONFIG["data"]["fold"] == 0
    assert CONFIG["data"]["image_root"] == "data/processed/P0_Physics_Audit_v1/images/e0b_meanbg_224"
    train, validation = split_fold_zero(ROOT / CONFIG["data"]["split_table"], True), split_fold_zero(ROOT / CONFIG["data"]["split_table"], False)
    assert len(train) == 400 and len(validation) == 100
    assert (train.label_3class.map(map_three_class_to_binary) == 0).sum() == 92
    assert (validation.label_3class.map(map_three_class_to_binary) == 0).sum() == 23
    assert not (set(train.patient_group_id.astype(str)) & set(validation.patient_group_id.astype(str)))


def test_exact_frozen_latent_interface_and_dataset_sample_contract():
    dataset = make_dataset(True)
    case_id = str(dataset.frame.ID.iloc[0])
    raw = read_frozen_auxiliary_vector(ROOT / CONFIG["assets"]["p1_root"], CONFIG["p1"], case_id)
    assert CONFIG["auxiliary"]["feature_order"] == list(FEATURE_ORDER)
    assert raw.shape == (278,) and raw.dtype == np.float32 and np.isfinite(raw).all()
    assert np.array_equal(raw, dataset.vectors[0])
    sample = dataset[0]
    assert sample["image"].shape == (3, 224, 224)
    assert sample["aux_vector"].shape == (278,) and sample["aux_vector"].dtype == torch.float32
    assert sample["label"].item() in (0, 1)
    source = (ROOT / "datasets/p2_a1_rgb_deca_aux_dataset.py").read_text(encoding="utf-8")
    source = source[source.index("def read_frozen_auxiliary_vector"):source.index("def audit_p0a_p1_assets")]
    for forbidden in ("light_code", "pose", "camera", "expression", "relighting", "albedo", "normal", "quality", "EXIF"):
        assert forbidden not in source


def test_train_only_normalization_and_dynamic_weighting():
    train, validation = make_dataset(True), make_dataset(False)
    statistics = train_stats(train, CONFIG["auxiliary"]["epsilon"])
    assert statistics["train_case_count"] == 400
    assert statistics["feature_dim"] == 278 and statistics["tex_dim"] == 50 and statistics["shape_dim"] == 100 and statistics["detail_dim"] == 128
    assert np.all(statistics["std"] >= 1e-6)
    validation.set_stats(statistics)
    expected = (validation.vectors[0] - statistics["mean"]) / statistics["std"]
    assert torch.allclose(validation[0]["aux_vector"], torch.tensor(expected, dtype=torch.float32))
    weights = compute_class_weights([int(item["label"]) for item in train], 2)
    assert torch.allclose(weights.cpu(), torch.tensor([400 / 184, 400 / 616], dtype=torch.float32), atol=1e-6)


def test_fusion_shapes_alpha_single_scalar_and_gradients():
    model = P2A1Fusion()
    outputs = model(torch.zeros(2, 3, 224, 224), torch.zeros(2, 278))
    assert outputs["z_rgb"].shape == outputs["z_aux"].shape == outputs["z_fused"].shape == (2, 512)
    assert outputs["logits"].shape == (2, 2) and abs(outputs["alpha"].item() - 0.1) < 1e-5
    assert model.alpha_logit.numel() == 1 and isinstance(model.alpha_logit, torch.nn.Parameter)
    loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([2.173913, 0.649351]))(outputs["logits"], torch.tensor([0, 1]))
    assert torch.isfinite(loss)
    loss.backward()
    assert model.rgb.conv1.weight.grad is not None
    assert model.aux[0].weight.grad is not None
    assert model.alpha_logit.grad is not None


def test_training_and_runner_guardrails_are_protocol_locked():
    trainer = (ROOT / "scripts/train/train_p2_a1_binary_rgb_deca_aux_fold0.py").read_text(encoding="utf-8")
    runner = (ROOT / "scripts/run/run_p2_a1_binary_rgb_deca_aux_fold0.py").read_text(encoding="utf-8")
    assert "torch.cuda.is_available" in trainer and "refusing CPU fallback" in trainer
    assert 'validation_metrics["macro_auc"] > best_auc' in trainer
    assert "threshold_search" in trainer and "AdamW" in trainer and "CrossEntropyLoss" in trainer
    assert "P2_B0_BinaryRGB_Fold0_v1" not in trainer
    assert "audit_p0a_p1_assets" in runner and "P2-B0 is frozen" in runner
