from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from datasets.R3DPR.binary_face_dataset import R3DPRBinaryFaceDataset, build_r3dpr_transforms
from losses.classification_losses import build_criterion
from metrics.R3DPR.binary_classification_metrics import compute_binary_metrics
from models.R3DPR.resnet18_binary import build_r3dpr_resnet18_binary, build_r3dpr_resnet_binary
from scripts.evaluate.R3DPR.plot_r3dpr_training_diagnostics import plot_experiment
from scripts.train.R3DPR.train_r3dpr_resnet18_binary_5fold import apply_training_mode, build_lr_scheduler, configure_trainability
from utils.R3DPR.binary_data_audit import audit_r3dpr_binary_data


def make_table_and_config(tmp_path: Path) -> tuple[pd.DataFrame, dict]:
    image_root = tmp_path / "images"
    image_root.mkdir()
    rows = []
    for fold in range(5):
        for label in (0, 1):
            sample_id = f"sample_{fold}_{label}"
            Image.new("RGB", (256, 320), color=(fold * 20, label * 100, 50)).save(image_root / f"{sample_id}.png")
            rows.append({"ID": sample_id, "patient_group_id": sample_id, "SEX": label, "fold": fold, "binary_label": label})
    table_path = tmp_path / "splits.csv"
    table = pd.DataFrame(rows)
    table.to_csv(table_path, index=False)
    config = {
        "data": {
            "table_csv": str(table_path),
            "image_root": str(image_root),
            "image_filename_template": "{ID}.png",
            "id_column": "ID",
            "group_id_column": "patient_group_id",
            "fold_column": "fold",
            "label_column": "binary_label",
            "sex_column": "SEX",
            "n_folds": 5,
            "source_image_height": 320,
            "source_image_width": 256,
            "image_height": 320,
            "image_width": 256,
        }
    }
    return table, config


def test_direct_binary_labels_and_rectangular_transform(tmp_path: Path):
    table, config = make_table_and_config(tmp_path)
    transform = build_r3dpr_transforms("val", 320, 256, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], False)
    dataset = R3DPRBinaryFaceDataset(table, config["data"]["image_root"], "{ID}.png", transform, "ID", "patient_group_id", "fold", "binary_label", "SEX")
    sample = dataset[0]
    assert tuple(sample["image"].shape) == (3, 320, 256)
    assert sample["label"].item() == int(table.loc[0, "binary_label"])


def test_audit_requires_complete_images_and_group_isolation(tmp_path: Path):
    table, config = make_table_and_config(tmp_path)
    audited = audit_r3dpr_binary_data(config, tmp_path / "audit", tmp_path)
    assert len(audited) == 10
    assert "PASSED" in (tmp_path / "audit" / "data_audit_report.md").read_text(encoding="utf-8")
    (Path(config["data"]["image_root"]) / "sample_0_0.png").unlink()
    with pytest.raises(ValueError, match="no image file"):
        audit_r3dpr_binary_data(config, tmp_path / "audit_missing", tmp_path)


def test_metrics_and_model_head_are_binary():
    metrics = compute_binary_metrics([0, 1, 0, 1], [[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
    assert metrics["macro_auc"] == 1.0
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["confusion_matrix"].tolist() == [[2, 0], [0, 2]]
    asymmetric_metrics = compute_binary_metrics(
        [0, 0, 1, 1],
        [[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]],
    )
    assert asymmetric_metrics["sensitivity"] == pytest.approx(0.5)
    assert asymmetric_metrics["specificity"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        compute_binary_metrics([0, 0], [[0.8, 0.2], [0.7, 0.3]])
    model = build_r3dpr_resnet18_binary(pretrained="none")
    assert model.fc.in_features == 512
    assert model.fc.out_features == 2
    assert np.isfinite(sum(parameter.detach().float().mean().item() for parameter in model.parameters()))
    dropout_model = build_r3dpr_resnet18_binary(pretrained="none", dropout=0.3)
    assert isinstance(dropout_model.fc, torch.nn.Sequential)
    assert isinstance(dropout_model.fc[0], torch.nn.Dropout)
    assert dropout_model.fc[0].p == pytest.approx(0.3)
    assert dropout_model.fc[-1].in_features == 512
    assert dropout_model.fc[-1].out_features == 2


@pytest.mark.parametrize(
    ("backbone", "features"),
    [("resnet18", 512), ("resnet34", 512), ("resnet50", 2048)],
)
def test_supported_r3dpr_resnets_have_two_logit_heads(backbone: str, features: int):
    model = build_r3dpr_resnet_binary(backbone, pretrained="none")
    assert model.fc.in_features == features
    assert model.fc.out_features == 2
    with torch.no_grad():
        assert tuple(model(torch.zeros(1, 3, 64, 64)).shape) == (1, 2)


def test_head_only_freezes_backbone_and_batchnorm_statistics():
    model = build_r3dpr_resnet18_binary(pretrained="none")
    configure_trainability(model, "head_only")
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable_names == ["fc.weight", "fc.bias"]
    assert sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) == 1026
    apply_training_mode(model, "head_only")
    assert not any(module.training for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d))


def test_full_finetune_can_keep_dropout_training_and_batchnorm_eval():
    model = build_r3dpr_resnet18_binary(pretrained="none", dropout=0.3)
    configure_trainability(model, "full_finetune")
    apply_training_mode(model, "full_finetune", batchnorm_mode="eval")
    assert model.fc[0].training
    assert not any(module.training for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d))


def test_cosine_lr_scheduler_anneals_and_is_optional():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4)
    scheduler = build_lr_scheduler(optimizer, {"lr_scheduler": "cosine", "lr_scheduler_t_max": 3, "lr_scheduler_eta_min": 0.0, "epochs": 3})
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    initial = float(optimizer.param_groups[0]["lr"])
    optimizer.step()
    scheduler.step()
    assert float(optimizer.param_groups[0]["lr"]) < initial
    assert build_lr_scheduler(optimizer, {"lr_scheduler": "none", "epochs": 3}) is None


def test_binary_weighted_label_smoothing_matches_manual_formula():
    weights = torch.tensor([2.0, 0.5])
    criterion = build_criterion(
        "weighted_ce_label_smoothing",
        class_weights=weights,
        num_classes=2,
        smoothing=0.1,
    )
    logits = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    target = torch.tensor([0, 1])
    loss = criterion(logits, target)
    probs = torch.softmax(logits, dim=1)
    manual = (
        -weights[0] * (0.9 * torch.log(probs[0, 0]) + 0.1 * torch.log(probs[0, 1]))
        - weights[1] * (0.1 * torch.log(probs[1, 0]) + 0.9 * torch.log(probs[1, 1]))
    ) / 2.0
    assert loss.item() == pytest.approx(manual.item())


def test_staged_layer4_unfreezes_only_layer4_and_fc():
    model = build_r3dpr_resnet18_binary(pretrained="none")
    configure_trainability(model, "staged_layer4")
    assert [name for name, parameter in model.named_parameters() if parameter.requires_grad] == ["fc.weight", "fc.bias"]
    configure_trainability(model, "staged_layer4", stage="layer4_plus_fc")
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable_names[0].startswith("layer4.")
    assert trainable_names[-2:] == ["fc.weight", "fc.bias"]
    assert all(name.startswith("layer4.") or name.startswith("fc.") for name in trainable_names)


def test_training_diagnostics_create_two_panel_exports(tmp_path: Path):
    experiment_dir = tmp_path / "experiment"
    experiment_dir.mkdir()
    fold_metrics = []
    for fold in range(5):
        fold_dir = experiment_dir / f"fold_{fold}"
        fold_dir.mkdir()
        history = pd.DataFrame(
            {
                "epoch": [1, 2, 3],
                "train_loss": [0.70, 0.55, 0.40],
                "val_loss": [0.68, 0.60, 0.66],
                "train_macro_auc": [0.70, 0.86, 0.95],
                "val_macro_auc": [0.72, 0.84, 0.79],
            }
        )
        history.to_csv(fold_dir / "training_history.csv", index=False)
        fold_metrics.append({"fold": fold, "best_epoch": 2, "macro_auc": 0.84})
    pd.DataFrame(fold_metrics).to_csv(experiment_dir / "fold_metrics.csv", index=False)

    output_dir = plot_experiment(experiment_dir)
    assert len(list(output_dir.glob("fold_*_training_diagnostics.png"))) == 5
    assert not list(output_dir.glob("fold_*_training_diagnostics.svg"))
    assert not list(output_dir.glob("fold_*_training_diagnostics.pdf"))
    summary = pd.read_csv(output_dir / "training_diagnostic_summary.csv")
    assert summary["auc_drop_after_best"].tolist() == pytest.approx([0.05] * 5)
