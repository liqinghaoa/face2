from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p2_counterfactual.io_utils import json_safe
from p2_counterfactual.orchestration import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run P2-A five-fold formal pipeline.")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--only-experiment", choices=["p2_a1_colorjitter", "p2_a2_relighting", "p2_a3_full_consistency"])
    parser.add_argument("--only-fold", type=int)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-a3", action="store_true")
    parser.add_argument("--pipeline-smoke", action="store_true")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.formal and not args.dry_run and not args.pipeline_smoke and not args.summarize_only and not args.validate_only:
        raise SystemExit("Use --formal for formal execution, --dry-run for preview, or --pipeline-smoke for non-formal integration smoke.")
    result = run_pipeline(
        formal=args.formal,
        resume=args.resume,
        device=args.device,
        num_workers=args.num_workers,
        only_experiment=args.only_experiment,
        only_fold=args.only_fold,
        summarize_only=args.summarize_only,
        validate_only=args.validate_only,
        dry_run=args.dry_run,
        force_a3=args.force_a3,
        pipeline_smoke=args.pipeline_smoke,
        max_batches=args.max_batches,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
