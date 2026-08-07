"""Single-layer light-code probe for P1-L."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class P1LightLinearProbe(nn.Module):
    """Linear probe for flattened 9x3 light code."""

    def __init__(self, input_dim: int = 27, num_classes: int = 2) -> None:
        super().__init__()
        if int(input_dim) != 27:
            raise ValueError("P1-L expects flattened 27-dimensional light input")
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.classifier = nn.Linear(self.input_dim, self.num_classes)

    def forward(self, inputs: torch.Tensor | dict[str, Any]) -> torch.Tensor:
        if isinstance(inputs, dict):
            inputs = inputs.get("representation", inputs.get("light", inputs))
        tensor = torch.as_tensor(inputs, dtype=torch.float32)
        return self.classifier(tensor.reshape(tensor.shape[0], -1))


def build_light_linear_probe(num_classes: int = 2) -> P1LightLinearProbe:
    return P1LightLinearProbe(input_dim=27, num_classes=num_classes)

