"""Smoke-test the NYHA backbone factory with synthetic 224x224 RGB inputs."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.nyha_backbone_factory import (  # noqa: E402
    build_nyha_classification_model,
    count_parameters,
    get_supported_backbones,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated backbone names to smoke-test.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device; default is cuda when available, otherwise cpu.",
    )
    return parser.parse_args()


def _selected_backbones(value: str | None) -> list[str]:
    supported = get_supported_backbones()
    if not value:
        return supported
    requested = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(requested).difference(supported))
    if unknown:
        raise ValueError(f"Unsupported --only backbone(s): {unknown}")
    return requested


def main() -> int:
    args = parse_args()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    backbones = _selected_backbones(args.only)
    exit_code = 0
    print("backbone,status,output_shape,total_params,trainable_params,error_message")
    for backbone in backbones:
        model = None
        try:
            model = build_nyha_classification_model(
                backbone,
                num_classes=3,
                pretrained=False,
            ).to(device)
            model.eval()
            with torch.no_grad():
                inputs = torch.randn(2, 3, 224, 224, device=device)
                outputs = model(inputs)
            if list(outputs.shape) != [2, 3]:
                raise ValueError(f"Unexpected output shape: {list(outputs.shape)}")
            counts = count_parameters(model)
            print(
                f"{backbone},OK,{list(outputs.shape)},"
                f"{counts['total_params']},{counts['trainable_params']},"
            )
        except Exception as exc:
            exit_code = 1
            print(f"{backbone},FAILED,,0,0,{str(exc).replace(',', ';')}")
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
