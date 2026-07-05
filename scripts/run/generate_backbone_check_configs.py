"""Generate ResNet34/50 configs for preprocessing backbone checks."""

from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path
from typing import Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "config" / "train"
OUTPUT_DIR = CONFIG_ROOT / "preprocess_ablation_backbone_check"
MANIFEST_PATH = OUTPUT_DIR / "backbone_check_config_manifest.csv"
EXPERIMENT_OUTPUT_ROOT = "experiments/preprocess_ablation_500Data/backbone_check"
EXPECTED_SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits_500"
TEMPLATE_BY_BACKBONE = {
    "resnet34": CONFIG_ROOT / "nyha_3class_global224_imagenet_resnet34.yaml",
    "resnet50": CONFIG_ROOT / "nyha_3class_global224_imagenet_resnet50.yaml",
}
JOBS = (
    ("B1", "resnet34", "hybrid_black_baseline"),
    ("B2", "resnet34", "hybrid_imagenet_meanbg"),
    ("B3", "resnet50", "hybrid_black_baseline"),
    ("B4", "resnet50", "hybrid_imagenet_meanbg"),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    return parser.parse_args(argv)


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def _load_yaml(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Template config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Template config must be a mapping: {path}")
    return config


def _check_template(config: dict, template_path: Path) -> None:
    for key in ("experiment", "data", "model", "train"):
        if key not in config or not isinstance(config[key], dict):
            raise ValueError(f"Template missing mapping `{key}`: {template_path}")
    split_dir = str(config["data"].get("split_dir", ""))
    if not split_dir:
        raise ValueError(f"Template missing data.split_dir: {template_path}")
    split_path = _resolve(split_dir)
    if not split_path.is_dir():
        raise FileNotFoundError(f"Template split_dir does not exist: {split_path}")
    if split_path != EXPECTED_SPLIT_DIR.resolve():
        print(
            "[warning] Template data.split_dir differs from prompt current split dir; "
            f"preserving template value per instruction. template={split_dir}, "
            f"expected=data/processed/splits_500"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = _resolve(args.output_dir)
    manifest_path = _resolve(args.manifest)

    if not EXPECTED_SPLIT_DIR.is_dir():
        raise FileNotFoundError(f"Required split directory not found: {EXPECTED_SPLIT_DIR}")

    output_dir.mkdir(parents=True, exist_ok=True)
    configs_by_backbone = {
        backbone: _load_yaml(path) for backbone, path in TEMPLATE_BY_BACKBONE.items()
    }
    for backbone, config in configs_by_backbone.items():
        _check_template(config, TEMPLATE_BY_BACKBONE[backbone])

    manifest_rows: list[dict[str, str]] = []
    for job_id, backbone, variant_name in JOBS:
        base_config = configs_by_backbone[backbone]
        config = copy.deepcopy(base_config)
        image_root = f"data/processed/global_face/preprocess_ablation/{variant_name}/images"
        image_root_path = _resolve(image_root)
        if not image_root_path.is_dir():
            raise FileNotFoundError(f"Image root for {job_id} does not exist: {image_root_path}")

        display_backbone = "ResNet34" if backbone == "resnet34" else "ResNet50"
        experiment_name = f"BackboneCheck_{display_backbone}_{variant_name}"
        config["experiment"]["name"] = experiment_name
        config["experiment"]["output_dir"] = EXPERIMENT_OUTPUT_ROOT
        config["data"]["image_root"] = image_root

        config_path = output_dir / f"nyha_3class_{backbone}_preproc_{variant_name}.yaml"
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

        manifest_rows.append(
            {
                "job_id": job_id,
                "backbone": backbone,
                "variant_name": variant_name,
                "config_path": _relative(config_path),
                "image_root": image_root,
                "experiment_name": experiment_name,
                "output_root": EXPERIMENT_OUTPUT_ROOT,
            }
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "job_id",
                "backbone",
                "variant_name",
                "config_path",
                "image_root",
                "experiment_name",
                "output_root",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Generated {len(manifest_rows)} backbone check configs: {output_dir}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
