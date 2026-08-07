from pathlib import Path

import numpy as np
import pandas as pd
import torch

from datasets.nyha_3class_face_dataset import build_transforms
from datasets.p2_a1_rgb_deca_aux_dataset import P2A1Dataset, FEATURE_ORDER, split_fold, train_stats
from losses.classification_losses import compute_class_weights
from models.p2_a1_rgb_deca_residual_fusion import P2A1Fusion
from scripts.train.train_p2_a1_binary_rgb_deca_aux import required_fold_outputs, validate_protocol
from utils.experiment_utils import load_yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_yaml(ROOT / "config/p2/p2_a1_binary_rgb_deca_aux_5fold_v1.yaml")


def dataset(fold: int, train: bool) -> P2A1Dataset:
    return P2A1Dataset(
        ROOT / CONFIG["data"]["split_csv"], ROOT / CONFIG["data"]["image_root"], ROOT / CONFIG["assets"]["p1_root"],
        CONFIG["p1"], fold, train, build_transforms("train" if train else "val", horizontal_flip=train),
    )


def test_fixed_fivefold_split_and_group_contract():
    assert CONFIG["data"]["folds"] == [0, 1, 2, 3, 4]
    validation_ids = set()
    for fold in CONFIG["data"]["folds"]:
        train, validation = split_fold(ROOT / CONFIG["data"]["split_csv"], fold, True), split_fold(ROOT / CONFIG["data"]["split_csv"], fold, False)
        assert len(train) == 400 and len(validation) == 100
        assert not (set(train.patient_group_id.astype(str)) & set(validation.patient_group_id.astype(str)))
        assert not (validation_ids & set(validation.ID.astype(str)))
        validation_ids.update(validation.ID.astype(str))
    assert len(validation_ids) == 500


def test_each_fold_uses_own_train_only_latent_statistics():
    train0, validation0, train1 = dataset(0, True), dataset(0, False), dataset(1, True)
    stats0, stats1 = train_stats(train0), train_stats(train1)
    assert stats0["fold"] == 0 and stats1["fold"] == 1
    assert stats0["train_case_count"] == stats1["train_case_count"] == 400
    assert stats0["validation_case_count"] == stats1["validation_case_count"] == 100
    assert stats0["feature_dim"] == 278 and tuple(stats0["feature_order"].tolist()) == FEATURE_ORDER
    assert not np.array_equal(stats0["mean"], stats1["mean"])
    validation0.set_stats(stats0)
    expected = (validation0.vectors[0] - stats0["mean"]) / stats0["std"]
    assert torch.allclose(validation0[0]["aux_vector"], torch.tensor(expected, dtype=torch.float32))


def test_fixed_model_loss_and_global_alpha_contract():
    validate_protocol(CONFIG, 3)
    model = P2A1Fusion()
    outputs = model(torch.zeros(2, 3, 224, 224), torch.zeros(2, 278))
    assert outputs["logits"].shape == (2, 2)
    assert outputs["z_rgb"].shape == outputs["z_aux"].shape == outputs["z_fused"].shape == (2, 512)
    assert model.alpha_logit.numel() == 1 and abs(outputs["alpha"].item() - .1) < 1e-5
    weights = compute_class_weights([0] * 92 + [1] * 308, 2)
    loss = torch.nn.CrossEntropyLoss(weight=weights)(outputs["logits"], torch.tensor([0, 1]))
    assert torch.isfinite(loss)
    loss.backward()
    assert model.rgb.conv1.weight.grad is not None and model.aux[0].weight.grad is not None and model.alpha_logit.grad is not None


def test_b0_reference_is_complete_and_exactly_paired_to_p0a_split():
    b0 = pd.read_csv(ROOT / CONFIG["evaluation"]["reference_b0_5fold_dir"] / "oof_predictions.csv", dtype={"sample_id": "string", "patient_group_id": "string"})
    split = pd.read_csv(ROOT / CONFIG["data"]["split_csv"], dtype={"ID": "string", "patient_group_id": "string"})
    paired = b0.merge(split, left_on="sample_id", right_on="ID", suffixes=("_b0", "_p0"), validate="one_to_one")
    assert len(paired) == 500 and paired.sample_id.nunique() == 500
    assert (paired.patient_group_id_b0.astype(str) == paired.patient_group_id_p0.astype(str)).all()
    assert (paired.fold_b0 == paired.fold_p0).all() and (paired.binary_label_b0 == paired.binary_label_p0).all()


def test_fivefold_runner_and_trainer_prohibit_protocol_drift():
    trainer = (ROOT / "scripts/train/train_p2_a1_binary_rgb_deca_aux.py").read_text(encoding="utf-8")
    runner = (ROOT / "scripts/run/run_p2_a1_binary_rgb_deca_aux_5fold.py").read_text(encoding="utf-8")
    assert "best_auc" in trainer and 'validation_metrics["macro_auc"] > best_auc' in trainer
    assert "torch.cuda.is_available" in trainer and "CPU fallback is prohibited" in trainer
    assert "P2_B0" not in trainer and "p0b_deca" not in trainer
    assert "_SUCCESS.json" in runner and "--overwrite-fold" in runner
    names = {path.name for path in required_fold_outputs(Path("fold_2"))}
    assert {"aux_normalization_stats.npz", "training_history.csv", "val_predictions.csv", "metrics.json", "aux_behavior.json", "_SUCCESS.json"} <= names
