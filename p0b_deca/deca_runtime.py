"""Audited, FAN-free access to the local official DECA core."""
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

import numpy as np


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_deca_config(deca_root: Path) -> Any:
    """Clone official config and explicitly bind every protected-asset path."""
    root = deca_root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from decalib.utils.config import get_cfg_defaults

    cfg = get_cfg_defaults()
    data = root / "data"
    cfg.device = "cuda"
    cfg.rasterizer_type = "pytorch3d"
    cfg.pretrained_modelpath = str(data / "deca_model.tar")
    cfg.model.topology_path = str(data / "head_template.obj")
    cfg.model.dense_template_path = str(data / "texture_data_256.npy")
    cfg.model.fixed_displacement_path = str(data / "fixed_displacement_256.npy")
    cfg.model.flame_model_path = str(data / "generic_model.pkl")
    cfg.model.flame_lmk_embedding_path = str(data / "landmark_embedding.npy")
    cfg.model.face_mask_path = str(data / "uv_face_mask.png")
    cfg.model.face_eye_mask_path = str(data / "uv_face_eye_mask.png")
    cfg.model.mean_tex_path = str(data / "mean_texture.jpg")
    cfg.model.tex_path = str(data / "FLAME_albedo_from_BFM.npz")
    cfg.model.use_tex = True
    cfg.model.tex_type = "BFM"
    cfg.model.extract_tex = False
    return cfg


def load_deca(deca_root: Path, device: str, seed: int) -> tuple[Any, Any]:
    """Load DECA without importing FAN, face_alignment, or TestData."""
    seed_everything(seed)
    cfg = build_deca_config(deca_root)
    from decalib.deca import DECA

    model = DECA(config=cfg, device=device)
    model.eval()
    return model, cfg


def image_to_tensor(rgb: np.ndarray, device: str) -> Any:
    import torch

    value = np.asarray(rgb)
    if value.shape != (224, 224, 3):
        raise ValueError(f"DECA input must be 224x224 RGB, got {value.shape}")
    return torch.from_numpy(value.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)


def forward_deca(model: Any, image: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    import torch

    model.eval()
    with torch.inference_mode():
        code = model.encode(image, use_detail=True)
        output, visual = model.decode(code, rendering=True, return_vis=True, use_detail=True)
    return code, output, visual

