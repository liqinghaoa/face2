"""Generate ResNet18 training configs for preprocessing ablation variants."""

from __future__ import annotations

import copy
import csv
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = (
    PROJECT_ROOT / "config" / "train" / "nyha_3class_global224_imagenet_resnet18.yaml"
)
OUTPUT_DIR = PROJECT_ROOT / "config" / "train" / "preprocess_ablation_resnet18"
EXPERIMENT_OUTPUT_DIR = "experiments/preprocess_ablation_500Data"
VARIANTS = (
    "hybrid_black_baseline",
    "hybrid_imagenet_meanbg",
    "hybrid_black_labl_norm",
    "hybrid_imagenet_meanbg_labl_norm",
    "hybrid_black_clahe_l",
    "hybrid_black_gray3ch",
    "hybrid_black_masked_grayworld_wb",
    "hybrid_black_retinex_msr",
)


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def main() -> int:
    with BASE_CONFIG.open("r", encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle)
    if not isinstance(base_config, dict):
        raise ValueError(f"Base config must be a mapping: {BASE_CONFIG}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str]] = []
    for variant in VARIANTS:
        config = copy.deepcopy(base_config)
        experiment_name = f"PreprocAblation_ResNet18_NYHA3Class_{variant}"
        image_root = (
            f"data/processed/global_face/preprocess_ablation/{variant}/images"
        )
        config["experiment"]["name"] = experiment_name
        config["experiment"]["output_dir"] = EXPERIMENT_OUTPUT_DIR
        config["data"]["image_root"] = image_root

        config_path = OUTPUT_DIR / f"nyha_3class_resnet18_preproc_{variant}.yaml"
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
        manifest_rows.append(
            {
                "variant_name": variant,
                "config_path": _relative(config_path),
                "image_root": image_root,
                "experiment_name": experiment_name,
                "output_dir": EXPERIMENT_OUTPUT_DIR,
            }
        )

    manifest_path = OUTPUT_DIR / "config_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "variant_name",
                "config_path",
                "image_root",
                "experiment_name",
                "output_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Generated {len(manifest_rows)} configs: {OUTPUT_DIR}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
