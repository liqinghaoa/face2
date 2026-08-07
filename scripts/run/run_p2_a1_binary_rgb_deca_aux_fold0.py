"""One-shot P2-A1 runner: immutable asset audit -> train -> B0 paired comparison."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.p2_a1_rgb_deca_aux_dataset import audit_p0a_p1_assets
from utils.experiment_utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--skip-final-comparison", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    smoke = args.epochs is not None
    if smoke != output_dir.name.endswith("_smoke") or (args.skip_final_comparison and not smoke):
        raise ValueError("--epochs is smoke-only and smoke output directories must end in _smoke")
    if not smoke and output_dir.resolve() == (ROOT / "experiments/500Data/P2_B0_BinaryRGB_Fold0_v1").resolve():
        raise ValueError("P2-B0 is frozen and cannot be used as a P2-A1 output directory")

    resolved_output = output_dir if output_dir.is_absolute() else ROOT / output_dir
    resolved_output.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config) if Path(args.config).is_absolute() else ROOT / args.config
    audit_p0a_p1_assets(load_yaml(config_path), resolved_output)

    command = [sys.executable, str(ROOT / "scripts/train/train_p2_a1_binary_rgb_deca_aux_fold0.py"), "--config", str(config_path), "--output-dir", str(resolved_output)]
    if smoke:
        command.extend(["--epochs", str(args.epochs)])
    subprocess.run(command, cwd=ROOT, check=True)
    if not args.skip_final_comparison:
        subprocess.run([sys.executable, str(ROOT / "scripts/evaluate/summarize_p2_a1_vs_b0.py"), "--experiment-dir", str(resolved_output)], cwd=ROOT, check=True)
    print(f"P2_A1_COMPLETED={resolved_output}")


if __name__ == "__main__":
    main()
