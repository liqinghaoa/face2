"""Small label-free EXIF-conditioned acquisition-response network."""

from __future__ import annotations

from os import PathLike
from typing import Any, Mapping

import torch
from torch import nn


ARCHITECTURE = {
    "name": "EXIFConditionedResponseMLP",
    "input_dim": 5,
    "hidden_dims": [8, 8],
    "output_dim": 3,
    "activation": "Tanh",
    "output_activation": None,
}
EXPECTED_PARAMETER_COUNT = 147


class EXIFConditionedResponseMLP(nn.Module):
    """Predict three standardized observations from five acquisition conditions."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(5, 8),
            nn.Tanh(),
            nn.Linear(8, 8),
            nn.Tanh(),
            nn.Linear(8, 3),
        )
        if self.parameter_count != EXPECTED_PARAMETER_COUNT:
            raise RuntimeError(f"Unexpected parameter count: {self.parameter_count}")

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, condition_input: torch.Tensor) -> torch.Tensor:
        if condition_input.ndim != 2 or condition_input.shape[1] != 5:
            raise ValueError("condition_input must have shape [batch, 5]")
        return self.network(condition_input)


def restore_model_from_checkpoint(
    checkpoint: str | PathLike[str] | Mapping[str, Any],
    map_location: str | torch.device = "cpu",
) -> tuple[EXIFConditionedResponseMLP, Mapping[str, Any]]:
    payload = (
        torch.load(checkpoint, map_location=map_location, weights_only=False)
        if isinstance(checkpoint, (str, PathLike))
        else checkpoint
    )
    if payload.get("architecture") != ARCHITECTURE:
        raise ValueError("Checkpoint architecture does not match EXIFConditionedResponseMLP")
    if int(payload.get("parameter_count", -1)) != EXPECTED_PARAMETER_COUNT:
        raise ValueError("Checkpoint parameter count is not 147")
    model = EXIFConditionedResponseMLP().to(map_location)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload
