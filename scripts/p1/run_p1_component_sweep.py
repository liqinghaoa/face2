from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _bootstrap import ROOT, load_yaml_config, resolve_output_dir
from trainers.p1_component_trainer import Phase2ExecutionNotApprovedError
from utils.p1_component_phase2 import run_experiment_suite
from utils.p1_component_preflight import build_execution_plan, validate_framework
from utils.p1_sweep_state import save_state, state_from_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config/p1/p1_component_sweep_v1.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/500Data/P1_Component_Sweep_v1")
    parser.add_argument("--validate-framework", action="store_true")
    parser.add_argument("--validate-data", action="store_true")
    parser.add_argument("--build-plan", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only", type=str, default="")
    parser.add_argument("--from", dest="from_experiment", type=str, default="")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def _approval_path(output_dir: Path) -> Path:
    return output_dir / "metadata" / "PHASE2_EXECUTION_APPROVED.json"


def _phase2_requested(args: argparse.Namespace) -> bool:
    return bool(args.smoke or args.formal or args.all)


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=repr)]
    if isinstance(value, Path):
        return str(value)
    return value


def _require_phase2_approval(output_dir: Path) -> None:
    if not _approval_path(output_dir).is_file():
        raise Phase2ExecutionNotApprovedError("PHASE2_EXECUTION_NOT_APPROVED")


def _write_plan(output_dir: Path, plan: dict[str, object]) -> Path:
    path = output_dir / "metadata" / "execution_plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(plan), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _summarize_only(output_dir: Path) -> dict[str, object]:
    metadata = output_dir / "metadata"
    summary = {
        "status": "available" if (metadata / "framework_manifest.json").is_file() else "missing",
        "framework_manifest": str(metadata / "framework_manifest.json"),
        "execution_plan": str(metadata / "execution_plan.json"),
        "preflight_validation": str(metadata / "preflight_validation.json"),
        "experiment_status": str(metadata / "experiment_status.json"),
    }
    (metadata / "sweep_summary.json").write_text(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    config["root"] = str(ROOT)
    config["config_path"] = str(args.config)
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if _phase2_requested(args):
        result = run_experiment_suite(
            config,
            output_dir,
            resume=bool(args.resume),
            only=[item.strip() for item in args.only.split(",") if item.strip()] if args.only else None,
            from_experiment=args.from_experiment or None,
        )
        print(json.dumps(_json_safe(result), ensure_ascii=False, indent=2))
        return

    manifest = None
    if args.validate_framework or args.validate_data or args.dry_run:
        manifest = validate_framework(config, output_dir)
        save_state(
            state_from_status(
                experiment_key="p1_component_sweep",
                status=manifest["status"] if manifest else "FRAMEWORK_VALIDATED",
                contract_version="p1_component_stage1",
                note=manifest["blockers"][0] if manifest and manifest["blockers"] else "",
            ),
            output_dir / "metadata" / "experiment_status.json",
        )

    if args.build_plan or args.dry_run:
        plan = build_execution_plan(config)
        if args.only:
            wanted = {item.strip() for item in args.only.split(",") if item.strip()}
            plan["experiments"] = [item for item in plan["experiments"] if item["experiment_key"] in wanted]
        if args.from_experiment:
            order = plan["experiment_order"]
            if args.from_experiment in order:
                start = order.index(args.from_experiment)
                allowed = set(order[start:])
                plan["experiments"] = [item for item in plan["experiments"] if item["experiment_key"] in allowed]
        if args.fold is not None:
            for item in plan["experiments"]:
                item["folds"] = [int(args.fold)]
                item["estimated_training_jobs"] = 1 if item["availability"] == "runnable" and item["experiment_key"] != "p1_spec" else 0
            plan["estimated_training_jobs"] = sum(item["estimated_training_jobs"] for item in plan["experiments"])
        _write_plan(output_dir, plan)

    if args.summarize_only:
        summary = _summarize_only(output_dir)
        print(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2))
        return

    if manifest is not None:
        print(json.dumps(_json_safe(manifest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
