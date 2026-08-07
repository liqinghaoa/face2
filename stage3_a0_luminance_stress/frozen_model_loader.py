from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models.nyha_backbone_factory import build_nyha_classification_model


def build_eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((320, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def load_frozen_model(checkpoint_path: str | Path, device: torch.device) -> torch.nn.Module:
    model = build_nyha_classification_model(
        "resnet18", num_classes=2, pretrained=False, freeze_backbone=False, dropout=0.3
    ).to(device)
    state = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    return model


@torch.no_grad()
def infer_uint8(model: torch.nn.Module, rgb: np.ndarray, device: torch.device) -> tuple[float, float, float]:
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    x = build_eval_transform()(image).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    return float(logits[0, 0].detach().cpu()), float(logits[0, 1].detach().cpu()), float(probs[1].detach().cpu())


def cuda_or_cpu() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
