from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .config import Stage3A0Config, ensure_dirs
from .oof_reproduction import reproduce_oof
from .preflight import run_preflight
from .r0_audit import run_r0_audit
from .r0_config import Stage3A0R0Config
from .reporting import write_failure_reports
from .stage3_a0_resume import run_stage3_a0_resume


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_text = args.config.read_text(encoding="utf-8")
    if "resume_from: counterfactual_generation" in config_text:
        base_config = PROJECT_ROOT / "configs/stage3_a0_frozen_rgb_paired_exposure_gamma_stress_v1.yaml"
        result = run_stage3_a0_resume(base_config, args.config)
        print(
            "Stage3-A0 resume completed: "
            f"rows={result['formal_inference_rows']} "
            f"evidence={result['evidence']['brightness_shortcut_support']}",
            file=sys.stderr,
        )
        return
    if "mode: stage3_a0_r0" in config_text:
        config = Stage3A0R0Config.from_yaml(args.config)
        result = run_r0_audit(config)
        status = result["resume_gate"]["reproduction_status"]
        print(f"R0 discrepancy audit completed: reproduction_status={status}", file=sys.stderr)
        return
    config = Stage3A0Config.from_yaml(args.config)
    dirs = ensure_dirs(config.output_dir)
    preflight_payload = run_preflight(config, dirs)
    preflight = preflight_payload["summary"]
    reproduction = reproduce_oof(config, preflight_payload["checkpoints"], dirs)
    if not reproduction["pass"]:
        reason = (
            "Original OOF reproduction gate failed: "
            f"max_probability_abs_diff={reproduction['max_probability_abs_diff']} "
            f"> tolerance={reproduction['probability_tolerance']}; "
            f"prediction_match_count={reproduction['prediction_match_count']}."
        )
        write_failure_reports(config.output_dir, preflight, reproduction, reason)
        print(reason, file=sys.stderr)
        raise SystemExit(2)
    write_failure_reports(
        config.output_dir,
        preflight,
        reproduction,
        "OOF reproduction passed, but counterfactual module has not been released in this guarded run.",
    )
    print("OOF reproduction passed; guarded run stopped before counterfactual stress.", file=sys.stderr)
    raise SystemExit(3)


if __name__ == "__main__":
    main()
