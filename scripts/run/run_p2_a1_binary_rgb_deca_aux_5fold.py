"""Sequential, resumable P2-A1 full five-fold runner."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.p2_a1_rgb_deca_aux_dataset import audit_p0a_p1_assets, split_fold
from scripts.evaluate.summarize_p2_a1_5fold import fold0_reproduction
from scripts.train.train_p2_a1_binary_rgb_deca_aux import required_fold_outputs
from scripts.train.train_e0b_global_resnet18_control_patient_binary_5fold import save_environment
from utils.experiment_utils import load_yaml, save_yaml


def fold_is_complete(output_dir: Path, fold: int) -> bool:
    fold_dir = output_dir / f"fold_{fold}"
    return (fold_dir / "_SUCCESS.json").is_file() and all(path.is_file() for path in required_fold_outputs(fold_dir))


def audit_full_assets(config: dict, output_dir: Path) -> None:
    compatible = dict(config)
    compatible["data"] = {**config["data"], "split_table": config["data"]["split_csv"]}
    record = audit_p0a_p1_assets(compatible, output_dir)
    per_fold = {}
    validation_ids = set()
    for fold in range(5):
        train, validation = split_fold(config["data"]["split_csv"], fold, True), split_fold(config["data"]["split_csv"], fold, False)
        overlap = set(train.patient_group_id.astype(str)) & set(validation.patient_group_id.astype(str))
        if overlap or validation_ids & set(validation.ID.astype(str)):
            raise ValueError(f"fold {fold}: leakage or duplicate validation samples")
        validation_ids.update(validation.ID.astype(str))
        per_fold[str(fold)] = {"train_cases": len(train), "validation_cases": len(validation), "patient_group_overlap_count": len(overlap)}
    if len(validation_ids) != 500:
        raise ValueError("five validation folds do not cover 500 unique samples")
    record["per_fold"] = per_fold
    record["fivefold_validation_unique_cases"] = len(validation_ids)
    (output_dir / "asset_record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "p0a_p1_asset_record.json").unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite-fold", action="append", type=int, default=[])
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    config = load_yaml(config_path)
    if list(config["data"]["folds"]) != [0, 1, 2, 3, 4]:
        raise ValueError("runner requires exactly folds [0,1,2,3,4]")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not (output_dir / "config_snapshot.yaml").exists():
        save_yaml(config, output_dir / "config_snapshot.yaml")
        save_environment(output_dir)
    audit_full_assets(config, output_dir)
    trainer = ROOT / "scripts" / "train" / "train_p2_a1_binary_rgb_deca_aux.py"
    for fold in config["data"]["folds"]:
        if fold_is_complete(output_dir, fold) and fold not in args.overwrite_fold:
            print(f"P2_A1_5FOLD_SKIP_COMPLETED_FOLD={fold}")
            if fold == 0:
                reproduction = fold0_reproduction(output_dir)
                if not reproduction["matches_within_1e-6"]:
                    raise RuntimeError("existing fold 0 does not reproduce frozen P2-A1 fold 0; stopping before folds 1-4")
            continue
        command = [sys.executable, str(trainer), "--config", str(config_path), "--fold", str(fold), "--output-dir", str(output_dir)]
        if fold in args.overwrite_fold:
            command.append("--overwrite-fold")
        subprocess.run(command, cwd=ROOT, check=True)
        if fold == 0:
            reproduction = fold0_reproduction(output_dir)
            if not reproduction["matches_within_1e-6"]:
                raise RuntimeError("new fold 0 does not reproduce frozen P2-A1 fold 0; stopping before folds 1-4")
    summarize = ROOT / "scripts" / "evaluate" / "summarize_p2_a1_5fold.py"
    subprocess.run([sys.executable, str(summarize), "--experiment-dir", str(output_dir)], cwd=ROOT, check=True)
    print(f"P2_A1_5FOLD_COMPLETED={output_dir}")


if __name__ == "__main__":
    main()
