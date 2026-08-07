from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts" / "p1"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_p1_stage_d_v1 as stage_d  # noqa: E402
from models.p1_independent_triple_resnet18 import IndependentTripleResNet18, count_parameters  # noqa: E402
from utils.p1_representation_normalization import ComponentNormalizationState  # noqa: E402


def _state(key: str) -> ComponentNormalizationState:
    if key == "p1_s":
        return ComponentNormalizationState(
            experiment_key="p1_s",
            representation="shading_like",
            strategy="clip_p01_p99_then_imagenet_normalize",
            fold=0,
            fitted_from="train_fold",
            p01=(0.0, 0.0, 0.0),
            p99=(1.0, 1.0, 1.0),
            source_case_count=400,
        )
    return ComponentNormalizationState(
        experiment_key="p1_r",
        representation="signed_residual",
        strategy="signed_residual_scale_then_imagenet_normalize",
        fold=0,
        fitted_from="train_fold",
        residual_scale=(1.0, 1.0, 1.0),
        source_case_count=400,
    )


def _sample() -> dict:
    rgb = torch.linspace(0.0, 1.0, steps=18, dtype=torch.float32).reshape(3, 2, 3)
    shading = rgb.clone()
    residual = torch.linspace(-1.0, 1.0, steps=18, dtype=torch.float32).reshape(3, 2, 3)
    mask = torch.ones(1, 2, 3, dtype=torch.float32)
    return {
        "case_id": "case_1",
        "patient_group_id": "patient_1",
        "fold": 0,
        "label_original": 0,
        "label_3class": 0,
        "label_binary": 0,
        "rgb": rgb,
        "shading_sample": {"representation": shading, "valid_mask": mask},
        "residual_sample": {"representation": residual, "valid_mask": mask},
    }


def test_triple_model_has_three_independent_encoders_and_batchnorm() -> None:
    model = IndependentTripleResNet18(pretrained=False)
    encoders = [model.encoder_rgb, model.encoder_shading, model.encoder_residual]
    assert len({id(encoder) for encoder in encoders}) == 3
    first_params = [next(encoder.parameters()) for encoder in encoders]
    assert len({id(param) for param in first_params}) == 3
    assert len({int(param.data_ptr()) for param in first_params}) == 3
    bn_sets = [[m for m in encoder.modules() if isinstance(m, torch.nn.BatchNorm2d)] for encoder in encoders]
    assert all(bn_sets)
    for i in range(len(bn_sets[0])):
        assert len({id(bn_sets[j][i]) for j in range(3)}) == 3
        assert len({id(bn_sets[j][i].running_mean) for j in range(3)}) == 3


def test_triple_model_dimensions_classifier_and_forward() -> None:
    model = IndependentTripleResNet18(pretrained=False)
    assert model.feature_dim == 512
    assert model.fused_dim == 1536
    assert model.classifier.in_features == 1536
    assert model.classifier.out_features == 2
    assert count_parameters(model)["classifier_params"] == 3074
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 224, 224), torch.zeros(2, 3, 224, 224), torch.zeros(2, 3, 224, 224))
    assert tuple(out.shape) == (2, 2)


def test_no_bottleneck_dropout_attention_or_gating_modules() -> None:
    model = IndependentTripleResNet18(pretrained=False)
    assert not any(isinstance(module, torch.nn.Dropout) for module in model.modules())
    assert model.classifier.__class__ is torch.nn.Linear
    assert model.classifier.in_features == 1536
    names = " ".join(dict(model.named_modules()).keys()).lower()
    assert "attention" not in names
    assert "gate" not in names
    assert "gating" not in names
    assert "bottleneck" not in names


def test_triple_collate_synchronously_flips_three_branches(monkeypatch) -> None:
    monkeypatch.setattr(stage_d.torch, "rand", lambda *args, **kwargs: torch.tensor([0.0]))
    collate = stage_d.build_triple_collate(_state("p1_s"), _state("p1_r"), training=True)
    batch = collate([_sample()])
    assert tuple(batch["rgb"].shape) == (1, 3, 2, 3)
    assert tuple(batch["shading"].shape) == (1, 3, 2, 3)
    assert tuple(batch["residual"].shape) == (1, 3, 2, 3)
    rgb_input = _sample()["rgb"]
    expected_rgb = stage_d._imagenet(rgb_input.flip(-1))
    assert torch.allclose(batch["rgb"][0], expected_rgb)
    assert torch.allclose(batch["shading"][0], expected_rgb)
    residual = _sample()["residual_sample"]["representation"].flip(-1)
    expected_residual = stage_d._imagenet((residual.clamp(-1, 1) * 0.5 + 0.5))
    assert torch.allclose(batch["residual"][0], expected_residual)


def test_residual_contract_preserves_signed_information() -> None:
    collate = stage_d.build_triple_collate(_state("p1_s"), _state("p1_r"), training=False)
    batch = collate([_sample()])
    residual_out = batch["residual"][0]
    assert not torch.allclose(residual_out[..., 0], residual_out[..., -1])
    assert torch.all(residual_out[..., -1] > residual_out[..., 0])


def test_stage_d_source_does_not_connect_camera_exif_checkpoints_or_group_oof() -> None:
    source = (ROOT / "scripts" / "p1" / "run_p1_stage_d_v1.py").read_text(encoding="utf-8")
    assert '"camera_features_enabled": False' in source
    assert '"exif_features_enabled": False' in source
    assert "camera_model" not in source
    assert "exposure_time" not in source
    assert "oof_predictions_group.csv" in source
    assert 'torch.load(fold_dir / "checkpoints/best_macro_auc.pth"' in source
    assert "P1-RGB checkpoint" not in source
    assert '"stage_d_v2_approved": False' in source
    assert '"allow_threshold_search": False' in source
    assert '"allow_hyperparameter_search": False' in source


def test_patient_group_is_only_metadata_split_bootstrap_not_model_input() -> None:
    source = (ROOT / "scripts" / "p1" / "run_p1_stage_d_v1.py").read_text(encoding="utf-8")
    assert '"patient_group_id"' in source
    assert 'cluster_unit": "patient_group_id"' in source
    assert "model(rgb, shading, residual)" in source
    assert "patient_group_id).to(device" not in source
