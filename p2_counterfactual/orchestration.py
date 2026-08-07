from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from p2_counterfactual.cluster_bootstrap import P2AExperimentFrames, load_experiment_frames, paired_cluster_bootstrap_comparison
from p2_counterfactual.config import resolve_p2_a_config
from p2_counterfactual.fold_validation import fold_success_valid, validate_p2_a_fold
from p2_counterfactual.gate_decision import DEFAULT_GATE_CONFIG, decide_a3_increment, decide_p2_a2_gate
from p2_counterfactual.io_utils import Tee, json_safe, sha256_file, utc_now, write_json_atomic
from p2_counterfactual.oof_aggregation import aggregate_p2_a_experiment
from p2_counterfactual.reporting import write_final_report
from p2_counterfactual.trainer import P2AFoldTrainer


PIPELINE_VERSION = "p2_a_pipeline_v1"
FIRST_STAGE_EXPERIMENTS = ("p2_a1_colorjitter", "p2_a2_relighting")
A3_EXPERIMENT = "p2_a3_full_consistency"
CONFIG_PATHS = {
    "p2_a1_colorjitter": Path("config/p2/p2_a/p2_a1_colorjitter.yaml"),
    "p2_a2_relighting": Path("config/p2/p2_a/p2_a2_relighting.yaml"),
    "p2_a3_full_consistency": Path("config/p2/p2_a/p2_a3_full_consistency.yaml"),
}


def choose_device(requested: str) -> torch.device:
    requested = str(requested).lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for formal P2-A, but torch.cuda.is_available() is false")
    return torch.device(requested)


def _experiment_dir(config: dict[str, Any], *, pipeline_smoke: bool = False, smoke_root: Path | None = None) -> Path:
    if pipeline_smoke:
        root = smoke_root or Path(config["project_root"]) / "experiments/500Data/P2_Physics_Relighting_v2/_framework_pipeline_smoke"
        return root / str(config["experiment_id"])
    return Path(config["output_root"]) / str(config["experiment_id"])


def _summary_root(config: dict[str, Any], *, pipeline_smoke: bool = False, smoke_root: Path | None = None) -> Path:
    if pipeline_smoke:
        root = smoke_root or Path(config["project_root"]) / "experiments/500Data/P2_Physics_Relighting_v2/_framework_pipeline_smoke"
        return root / "summary"
    return Path(config["output_root"]).parents[0] / "summary"


def _load_config(experiment_id: str, *, pipeline_smoke: bool = False, smoke_root: Path | None = None) -> dict[str, Any]:
    config = resolve_p2_a_config(CONFIG_PATHS[experiment_id])
    config.setdefault("gate", DEFAULT_GATE_CONFIG)
    config.setdefault("bootstrap", {"iterations": 2000, "seed": 2026})
    if pipeline_smoke:
        root = smoke_root or Path(config["project_root"]) / "experiments/500Data/P2_Physics_Relighting_v2/_framework_pipeline_smoke"
        config["smoke_output_root"] = str(root)
    return config


def _write_state(summary_dir: Path, state: dict[str, Any]) -> None:
    state["last_updated"] = utc_now()
    write_json_atomic(summary_dir / "p2_a_pipeline_state.json", state)


def _initial_state(manifest_sha256: str, experiment_order: list[str]) -> dict[str, Any]:
    return {
        "pipeline_version": PIPELINE_VERSION,
        "manifest_sha256": manifest_sha256,
        "experiment_order": experiment_order,
        "current_experiment": None,
        "current_fold": None,
        "completed_folds": {},
        "failed_folds": {},
        "experiment_statuses": {},
        "gate_status": "not_evaluated",
        "a3_triggered": False,
        "pipeline_completed": False,
        "last_updated": utc_now(),
    }


def _populate_state_from_artifacts(
    state: dict[str, Any],
    *,
    experiment_ids: list[str],
    configs: dict[str, dict[str, Any]],
    folds: list[int],
    pipeline_smoke: bool,
) -> None:
    for experiment_id in experiment_ids:
        config = configs[experiment_id]
        exp_dir = _experiment_dir(config, pipeline_smoke=pipeline_smoke)
        completed = [
            int(fold)
            for fold in folds
            if fold_success_valid(exp_dir / f"fold_{int(fold)}", config=config, fold=int(fold))
        ]
        state["completed_folds"][experiment_id] = completed
        state["failed_folds"].setdefault(experiment_id, [])
        if (exp_dir / "_EXPERIMENT_SUCCESS.json").is_file():
            state["experiment_statuses"][experiment_id] = "success"
        elif completed:
            state["experiment_statuses"][experiment_id] = "folds_complete_no_experiment_success"
        else:
            state["experiment_statuses"][experiment_id] = "incomplete"


def dry_run_plan(*, only_experiment: str | None = None, only_fold: int | None = None) -> dict[str, Any]:
    experiments = [only_experiment] if only_experiment else list(FIRST_STAGE_EXPERIMENTS)
    folds = [int(only_fold)] if only_fold is not None else list(range(5))
    commands = []
    for experiment_id in experiments:
        for fold in folds:
            commands.append(
                {
                    "experiment_id": experiment_id,
                    "fold": fold,
                    "config": str(CONFIG_PATHS[experiment_id]),
                    "output_dir": str(Path("experiments/500Data/P2_Physics_Relighting_v2/rgb_benchmark") / experiment_id / f"fold_{fold}"),
                    "command": f"python scripts/p2/run_p2_a_fold.py --config {CONFIG_PATHS[experiment_id]} --fold {fold} --mode train-evaluate --resume --device cuda",
                }
            )
    return {
        "pipeline_version": PIPELINE_VERSION,
        "first_stage_experiment_order": list(FIRST_STAGE_EXPERIMENTS),
        "fold_order": folds,
        "commands": commands,
        "gate_position": "gate after p2_a1_colorjitter and p2_a2_relighting are complete and summarized",
        "a3_default_behavior": "run only if gate_pass=true",
        "training_started": False,
    }


def _run_one_fold(
    *,
    config: dict[str, Any],
    fold: int,
    device: torch.device,
    resume: bool,
    num_workers: int,
    max_batches: int | None,
    pipeline_smoke: bool,
    expected_full_fold: bool,
) -> dict[str, Any]:
    trainer = P2AFoldTrainer(
        config,
        fold=int(fold),
        device=device,
        smoke=pipeline_smoke,
        num_workers=num_workers,
        max_batches=max_batches,
    )
    fold_dir = trainer.output_dir
    if resume and fold_success_valid(fold_dir, config=config, fold=fold):
        return json.loads((fold_dir / "_FOLD_SUCCESS.json").read_text(encoding="utf-8"))
    log_path = fold_dir / "logs" / "pipeline_fold_run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        tee = Tee(sys.stdout, log_handle)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            print(f"[{utc_now()}] START experiment={config['experiment_id']} fold={fold} smoke={pipeline_smoke}", flush=True)
            trainer.train(resume=resume)
            trainer.evaluate()
            print(f"[{utc_now()}] END experiment={config['experiment_id']} fold={fold}", flush=True)
    return validate_p2_a_fold(
        fold_dir=fold_dir,
        config=config,
        fold=fold,
        manifest_path=config["manifest_path"],
        write_success=True,
        expected_full_fold=expected_full_fold,
    )


def _aggregate_if_possible(
    *,
    experiment_id: str,
    config: dict[str, Any],
    folds: list[int],
    pipeline_smoke: bool,
    expected_full_folds: bool,
) -> dict[str, Any] | None:
    experiment_dir = _experiment_dir(config, pipeline_smoke=pipeline_smoke)
    if expected_full_folds and sorted(folds) != [0, 1, 2, 3, 4]:
        return None
    for fold in folds:
        if not fold_success_valid(experiment_dir / f"fold_{fold}", config=config, fold=fold):
            return None
    return aggregate_p2_a_experiment(
        experiment_dir=experiment_dir,
        config=config,
        manifest_path=config["manifest_path"],
        folds=folds,
        expected_case_count=500,
        expected_full_folds=expected_full_folds,
    )


def _model_and_stability_rows(experiment_ids: list[str], configs: dict[str, dict[str, Any]], *, pipeline_smoke: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_rows = []
    stability_rows = []
    for experiment_id in experiment_ids:
        exp_dir = _experiment_dir(configs[experiment_id], pipeline_smoke=pipeline_smoke)
        success_path = exp_dir / "_EXPERIMENT_SUCCESS.json"
        if not success_path.is_file():
            continue
        success = json.loads(success_path.read_text(encoding="utf-8"))
        metrics = success["summary_metrics"]
        stability = success["stability_metrics"]
        model_rows.append(
            {
                "experiment_id": experiment_id,
                "macro_auc": float(metrics["macro_auc"]),
                "accuracy": float(metrics["accuracy"]),
                "macro_precision": float(metrics["macro_precision"]),
                "macro_recall": float(metrics["macro_recall"]),
                "macro_f1": float(metrics["macro_f1"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "patient_sensitivity": float(metrics["patient_sensitivity"]),
                "control_specificity": float(metrics["control_specificity"]),
            }
        )
        stability_rows.append(
            {
                "experiment_id": experiment_id,
                "mean_prediction_std": float(stability["prediction_std"]["mean"]),
                "median_prediction_std": float(stability["prediction_std"]["median"]),
                "case_level_flip_rate": float(stability["case_level_label_flip_rate"]),
                "worst_light_auc": float(stability["worst_light_auc"]),
                "mean_relighted_auc": float(stability["mean_relighted_auc"]),
                "auc_range": float(stability["auc_range"]),
                "mean_feature_cosine": float(stability["mean_feature_cosine"]),
                "median_feature_cosine": float(stability["median_feature_cosine"]),
            }
        )
    return pd.DataFrame(model_rows), pd.DataFrame(stability_rows)


def _run_comparisons(
    *,
    experiment_ids: list[str],
    configs: dict[str, dict[str, Any]],
    summary_dir: Path,
    pipeline_smoke: bool,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    pairs = [("p2_a1_colorjitter", "p2_a2_relighting")]
    if A3_EXPERIMENT in experiment_ids:
        pairs += [("p2_a2_relighting", A3_EXPERIMENT), ("p2_a1_colorjitter", A3_EXPERIMENT)]
    rows = []
    for a, b in pairs:
        if a not in experiment_ids or b not in experiment_ids:
            continue
        a_dir = _experiment_dir(configs[a], pipeline_smoke=pipeline_smoke)
        b_dir = _experiment_dir(configs[b], pipeline_smoke=pipeline_smoke)
        if not (a_dir / "_EXPERIMENT_SUCCESS.json").is_file() or not (b_dir / "_EXPERIMENT_SUCCESS.json").is_file():
            continue
        rows.extend(
            paired_cluster_bootstrap_comparison(
                load_experiment_frames(str(a_dir), a),
                load_experiment_frames(str(b_dir), b),
                iterations=iterations,
                seed=seed,
            )
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame.to_csv(summary_dir / "p2_a_paired_comparisons.csv", index=False, encoding="utf-8-sig")
        frame.to_csv(summary_dir / "p2_a_cluster_bootstrap.csv", index=False, encoding="utf-8-sig")
        frame.to_csv(summary_dir / "cluster_bootstrap_comparisons.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=["comparison", "metric"]).to_csv(summary_dir / "p2_a_paired_comparisons.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(columns=["comparison", "metric"]).to_csv(summary_dir / "p2_a_cluster_bootstrap.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(columns=["comparison", "metric"]).to_csv(summary_dir / "cluster_bootstrap_comparisons.csv", index=False, encoding="utf-8-sig")
    return frame


def summarize_pipeline(
    *,
    experiment_ids: list[str],
    configs: dict[str, dict[str, Any]],
    summary_dir: Path,
    pipeline_smoke: bool,
    bootstrap_iterations: int,
    bootstrap_seed: int,
    force_a3: bool = False,
) -> dict[str, Any]:
    summary_dir.mkdir(parents=True, exist_ok=True)
    model_df, stability_df = _model_and_stability_rows(experiment_ids, configs, pipeline_smoke=pipeline_smoke)
    model_df.to_csv(summary_dir / "p2_a_model_comparison.csv", index=False, encoding="utf-8-sig")
    stability_df.to_csv(summary_dir / "p2_a_stability_comparison.csv", index=False, encoding="utf-8-sig")
    model_metrics = {row["experiment_id"]: row for row in model_df.to_dict("records")}
    stability_metrics = {row["experiment_id"]: row for row in stability_df.to_dict("records")}
    paired = _run_comparisons(
        experiment_ids=experiment_ids,
        configs=configs,
        summary_dir=summary_dir,
        pipeline_smoke=pipeline_smoke,
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
    )
    gate = None
    force_a3_override = None
    if all(exp in model_metrics for exp in FIRST_STAGE_EXPERIMENTS):
        gate = decide_p2_a2_gate(
            model_metrics=model_metrics,
            stability_metrics=stability_metrics,
            bootstrap_rows=paired.to_dict("records"),
            gate_config=DEFAULT_GATE_CONFIG,
        )
        if force_a3 and not gate["gate_pass"]:
            force_a3_override = {
                "status": "NON_FORMAL_FORCE_A3_REQUESTED",
                "formal_gate_pass": bool(gate["gate_pass"]),
                "formal_a3_should_run": bool(gate["a3_should_run"]),
                "effective_a3_run_requested": True,
                "reason": "--force-a3 requested; formal gate decision is preserved unchanged in p2_a_gate_decision.json.",
                "created_at": utc_now(),
            }
            write_json_atomic(summary_dir / "p2_a_force_a3_override.json", force_a3_override)
        write_json_atomic(summary_dir / "p2_a_gate_decision.json", gate)
    if A3_EXPERIMENT in model_metrics:
        a3_increment = decide_a3_increment(model_metrics=model_metrics, stability_metrics=stability_metrics)
        write_json_atomic(summary_dir / "p2_a3_increment_decision.json", a3_increment)
    else:
        a3_increment = None
    manifest_sha = sha256_file(configs[experiment_ids[0]]["manifest_path"]) if experiment_ids else ""
    statuses = {}
    for experiment_id in experiment_ids:
        exp_dir = _experiment_dir(configs[experiment_id], pipeline_smoke=pipeline_smoke)
        statuses[experiment_id] = "success" if (exp_dir / "_EXPERIMENT_SUCCESS.json").is_file() else "incomplete"
    write_final_report(
        summary_dir=summary_dir,
        model_comparison=model_df,
        stability_comparison=stability_df,
        paired_comparisons=paired,
        gate_decision=gate or {"a3_should_run": False, "gate_pass": None, "decision_reason": "Gate not evaluated."},
        a3_increment=a3_increment,
        manifest_sha256=manifest_sha,
        experiment_statuses=statuses,
    )
    return {
        "model_comparison": model_df.to_dict("records"),
        "stability_comparison": stability_df.to_dict("records"),
        "paired_rows": int(len(paired)),
        "gate": gate,
        "a3_increment": a3_increment,
        "force_a3_override": force_a3_override,
    }


def run_pipeline(
    *,
    formal: bool = False,
    resume: bool = True,
    device: str = "cuda",
    num_workers: int = 0,
    only_experiment: str | None = None,
    only_fold: int | None = None,
    summarize_only: bool = False,
    validate_only: bool = False,
    dry_run: bool = False,
    force_a3: bool = False,
    pipeline_smoke: bool = False,
    max_batches: int | None = None,
    bootstrap_iterations: int = 2000,
) -> dict[str, Any]:
    if dry_run:
        return dry_run_plan(only_experiment=only_experiment, only_fold=only_fold)
    selected = [only_experiment] if only_experiment else list(FIRST_STAGE_EXPERIMENTS)
    folds = [int(only_fold)] if only_fold is not None else ([0] if pipeline_smoke else [0, 1, 2, 3, 4])
    smoke_root = None
    configs = {experiment_id: _load_config(experiment_id, pipeline_smoke=pipeline_smoke, smoke_root=smoke_root) for experiment_id in [*FIRST_STAGE_EXPERIMENTS, A3_EXPERIMENT]}
    summary_dir = _summary_root(configs[selected[0]], pipeline_smoke=pipeline_smoke, smoke_root=smoke_root)
    manifest_sha = sha256_file(configs[selected[0]]["manifest_path"])
    state = _initial_state(manifest_sha, selected)
    _write_state(summary_dir, state)
    dev = choose_device(device)
    expected_full = bool(formal and not pipeline_smoke and only_fold is None)

    if not summarize_only:
        for experiment_id in selected:
            config = configs[experiment_id]
            state["current_experiment"] = experiment_id
            state["completed_folds"].setdefault(experiment_id, [])
            state["failed_folds"].setdefault(experiment_id, [])
            for fold in folds:
                state["current_fold"] = int(fold)
                _write_state(summary_dir, state)
                fold_dir = _experiment_dir(config, pipeline_smoke=pipeline_smoke) / f"fold_{int(fold)}"
                try:
                    if validate_only:
                        result = validate_p2_a_fold(
                            fold_dir=fold_dir,
                            config=config,
                            fold=int(fold),
                            manifest_path=config["manifest_path"],
                            write_success=True,
                            expected_full_fold=expected_full,
                        )
                    else:
                        result = _run_one_fold(
                            config=config,
                            fold=int(fold),
                            device=dev,
                            resume=resume,
                            num_workers=num_workers,
                            max_batches=max_batches,
                            pipeline_smoke=pipeline_smoke,
                            expected_full_fold=expected_full,
                        )
                    if int(fold) not in state["completed_folds"][experiment_id]:
                        state["completed_folds"][experiment_id].append(int(fold))
                    state["experiment_statuses"][experiment_id] = "folds_in_progress"
                    _write_state(summary_dir, state)
                except Exception as exc:
                    state["failed_folds"][experiment_id].append({"fold": int(fold), "error": f"{type(exc).__name__}: {exc}"})
                    state["experiment_statuses"][experiment_id] = "failed"
                    _write_state(summary_dir, state)
                    raise
            success = _aggregate_if_possible(
                experiment_id=experiment_id,
                config=config,
                folds=folds,
                pipeline_smoke=pipeline_smoke,
                expected_full_folds=expected_full,
            )
            if success:
                state["experiment_statuses"][experiment_id] = "success"
                _write_state(summary_dir, state)

    completed_for_summary = [exp for exp in [*FIRST_STAGE_EXPERIMENTS, A3_EXPERIMENT] if (_experiment_dir(configs[exp], pipeline_smoke=pipeline_smoke) / "_EXPERIMENT_SUCCESS.json").is_file()]
    if summarize_only:
        _populate_state_from_artifacts(
            state,
            experiment_ids=completed_for_summary,
            configs=configs,
            folds=folds,
            pipeline_smoke=pipeline_smoke,
        )
        _write_state(summary_dir, state)
    summary = summarize_pipeline(
        experiment_ids=completed_for_summary,
        configs=configs,
        summary_dir=summary_dir,
        pipeline_smoke=pipeline_smoke,
        bootstrap_iterations=int(bootstrap_iterations),
        bootstrap_seed=2026,
        force_a3=force_a3,
    )
    gate = summary.get("gate") or {}
    if not only_experiment and not summarize_only and not validate_only and (gate.get("a3_should_run") or force_a3) and A3_EXPERIMENT not in completed_for_summary:
        state["gate_status"] = "pass" if gate.get("gate_pass") else "force_a3"
        state["a3_triggered"] = True
        _write_state(summary_dir, state)
        a3_config = configs[A3_EXPERIMENT]
        state["current_experiment"] = A3_EXPERIMENT
        state["completed_folds"].setdefault(A3_EXPERIMENT, [])
        state["failed_folds"].setdefault(A3_EXPERIMENT, [])
        for fold in folds:
            state["current_fold"] = int(fold)
            _write_state(summary_dir, state)
            try:
                _run_one_fold(
                    config=a3_config,
                    fold=int(fold),
                    device=dev,
                    resume=resume,
                    num_workers=num_workers,
                    max_batches=max_batches,
                    pipeline_smoke=pipeline_smoke,
                    expected_full_fold=expected_full,
                )
                if int(fold) not in state["completed_folds"][A3_EXPERIMENT]:
                    state["completed_folds"][A3_EXPERIMENT].append(int(fold))
                _write_state(summary_dir, state)
            except Exception as exc:
                state["failed_folds"][A3_EXPERIMENT].append({"fold": int(fold), "error": f"{type(exc).__name__}: {exc}"})
                state["experiment_statuses"][A3_EXPERIMENT] = "failed"
                _write_state(summary_dir, state)
                raise
        success = _aggregate_if_possible(
            experiment_id=A3_EXPERIMENT,
            config=a3_config,
            folds=folds,
            pipeline_smoke=pipeline_smoke,
            expected_full_folds=expected_full,
        )
        if success:
            state["experiment_statuses"][A3_EXPERIMENT] = "success"
            _write_state(summary_dir, state)
        completed_for_summary = [exp for exp in [*FIRST_STAGE_EXPERIMENTS, A3_EXPERIMENT] if (_experiment_dir(configs[exp], pipeline_smoke=pipeline_smoke) / "_EXPERIMENT_SUCCESS.json").is_file()]
        summary = summarize_pipeline(
            experiment_ids=completed_for_summary,
            configs=configs,
            summary_dir=summary_dir,
            pipeline_smoke=pipeline_smoke,
            bootstrap_iterations=int(bootstrap_iterations),
            bootstrap_seed=2026,
            force_a3=force_a3,
        )
    state["current_experiment"] = None
    state["current_fold"] = None
    formal_gate_pass = bool((summary.get("gate") or {}).get("gate_pass"))
    if force_a3 and A3_EXPERIMENT in completed_for_summary and not formal_gate_pass:
        state["gate_status"] = "force_a3_non_formal"
    else:
        state["gate_status"] = "pass" if formal_gate_pass else "fail_or_not_evaluated"
    state["a3_triggered"] = bool(A3_EXPERIMENT in completed_for_summary)
    state["a3_triggered_by_gate"] = bool(formal_gate_pass and A3_EXPERIMENT in completed_for_summary)
    state["pipeline_completed"] = bool(all(exp in completed_for_summary for exp in FIRST_STAGE_EXPERIMENTS))
    _write_state(summary_dir, state)
    return {
        "status": "completed" if state["pipeline_completed"] else "partial",
        "pipeline_smoke": bool(pipeline_smoke),
        "formal": bool(formal),
        "summary_dir": str(summary_dir),
        "completed_experiments": completed_for_summary,
        "manifest_sha256": manifest_sha,
        "summary": json_safe(summary),
    }
