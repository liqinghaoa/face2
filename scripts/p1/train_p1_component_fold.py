from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ROOT, load_yaml_config, resolve_output_dir
from trainers.p1_component_trainer import P1ComponentTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config/p1/p1_component_sweep_v1.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/500Data/P1_Component_Sweep_v1")
    parser.add_argument("--experiment", type=str, default="")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--execution-mode", type=str, default="validate_only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    config["root"] = str(ROOT)
    config["config_path"] = str(args.config)
    if args.experiment:
        config["experiment_key"] = args.experiment
    trainer = P1ComponentTrainer()
    result = trainer.fit_fold(config, args.fold, resolve_output_dir(args.output_dir), args.execution_mode)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
