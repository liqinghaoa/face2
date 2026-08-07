from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from p2_counterfactual.config import resolve_p2_a_config
from p2_counterfactual.evaluator import P2AEvaluator


class TinyP2Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, images: torch.Tensor):
        pooled = images.mean(dim=(1, 2, 3)) * self.scale
        logits = torch.stack([-pooled, pooled], dim=1)
        features = pooled[:, None].repeat(1, 512)
        return logits, features


def test_evaluator_writes_original_and_six_relighted_outputs(tmp_path: Path) -> None:
    config = resolve_p2_a_config("config/p2/p2_a/p2_a3_full_consistency.yaml")
    config["output_root"] = str(tmp_path)
    config["smoke_output_root"] = str(tmp_path / "smoke")
    config["training"]["batch_size"] = 4
    model = TinyP2Model()
    out_dir = tmp_path / "eval"
    evaluator = P2AEvaluator(model, config, fold=0, device=torch.device("cpu"), output_dir=out_dir)
    original = evaluator.evaluate_original(max_cases=6, num_workers=0)
    relighted = evaluator.evaluate_relighted(max_cases=6, num_workers=0)
    assert original["rows"] == 6
    assert relighted["rows"] == 36
    assert relighted["feature_shape"] == [6, 6, 512]
    assert (out_dir / "val_predictions_original.csv").is_file()
    assert (out_dir / "val_predictions_relighted.csv").is_file()
    assert (out_dir / "val_features_original.npz").is_file()
    assert (out_dir / "val_features_relighted.npz").is_file()
