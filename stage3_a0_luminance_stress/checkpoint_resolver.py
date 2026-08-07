from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_checkpoints(rgb_oof_csv: Path) -> pd.DataFrame:
    oof = pd.read_csv(rgb_oof_csv, dtype={"sample_id": str, "patient_group_id": str})
    rows: list[dict[str, Any]] = []
    for fold, sub in oof.groupby("fold", sort=True):
        paths = sorted(sub["checkpoint_path"].astype(str).unique().tolist())
        epochs = sorted(pd.to_numeric(sub["selected_epoch"], errors="raise").astype(int).unique().tolist())
        if len(paths) != 1 or len(epochs) != 1:
            raise ValueError(f"fold {fold} does not resolve to one checkpoint and one epoch")
        ckpt = Path(paths[0])
        if not ckpt.is_file():
            raise FileNotFoundError(f"checkpoint missing for fold {fold}: {ckpt}")
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model_keys = list(state["model_state_dict"].keys())
        cfg = state.get("config", {})
        audit = state.get("trainability_audit", {})
        rows.append(
            {
                "fold": int(fold),
                "checkpoint_path": str(ckpt),
                "checkpoint_sha256": sha256_file(ckpt),
                "selected_epoch": int(epochs[0]),
                "checkpoint_epoch": int(state.get("epoch", -1)),
                "architecture": str(cfg.get("model", {}).get("backbone", "resnet18")),
                "model_state_key_count": len(model_keys),
                "model_state_keys_preview": ";".join(model_keys[:12]),
                "normalization": json.dumps(cfg.get("normalize", {}), ensure_ascii=False),
                "input_size": json.dumps(cfg.get("data", {}).get("image_size", [320, 256])),
                "color_order": "RGB",
                "bn_mode": str(audit.get("strategy", cfg.get("model", {}).get("trainability_strategy", "unknown"))),
                "inference_dtype": "float32",
                "device": "cuda_if_available_else_cpu",
            }
        )
    out = pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)
    if out["fold"].tolist() != [0, 1, 2, 3, 4]:
        raise ValueError(f"expected fold checkpoints 0-4, got {out['fold'].tolist()}")
    return out


def inference_contract(checkpoints: pd.DataFrame) -> dict[str, Any]:
    first = checkpoints.iloc[0]
    return {
        "architecture": first["architecture"],
        "input_size_hwc": [320, 256, 3],
        "resize_size_hw": [320, 256],
        "color_order": "RGB",
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "output": "softmax(logits) with patient probability at class index 1",
        "classification_threshold": 0.5,
        "bn_mode": "model.eval() during inference; BatchNorm statistics are not updated",
        "augmentation": "none during validation/reproduction",
        "fold_checkpoint_rule": "each sample uses only its original test-fold checkpoint",
    }
