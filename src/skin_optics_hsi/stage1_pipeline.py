"""End-to-end stage-one runner and immutable artifact writer."""

from __future__ import annotations

import json
import platform
import shutil
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
import yaml

from .config import Stage1Config, load_stage1_config
from .data_contracts import (
    HyperSkinSample,
    audit_data_contract,
    discover_hyperskin_vis,
    extract_mean_skin_spectrum,
    load_binary_mask,
    load_hyperskin_cube,
)
from .hsi_inverse_fit import SpectrumFitResult, fit_spectrum
from .skin_forward import SkinForwardModel


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _require_new_output_directory(path: Path) -> None:
    if path.exists():
        if any(path.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty stage output: {path}")
    else:
        path.mkdir(parents=True)


def _gate_values(config: Stage1Config) -> dict[str, float | None]:
    gates = config.raw["gate_thresholds"]
    keys = (
        "median_log_rmse_max",
        "median_sam_rad_max",
        "boundary_fraction_max",
        "multistart_relative_range_max",
        "jacobian_condition_number_max",
        "noise_relative_sd_max",
    )
    return {key: None if gates.get(key) is None else float(gates[key]) for key in keys}


def _aggregate_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"n_spectra": 0}
    multistart_columns = ["multistart_relative_range_M", "multistart_relative_range_Hb"]
    noise_columns = ["noise_relative_sd_M", "noise_relative_sd_Hb"]
    return {
        "n_spectra": int(len(frame)),
        "n_subjects": int(frame["subject_id"].nunique()),
        "median_log_rmse": float(frame["log_rmse"].median()),
        "median_sam_rad": float(frame["sam_rad"].median()),
        "boundary_fraction": float(frame[["boundary_M", "boundary_Hb"]].to_numpy(dtype=float).mean()),
        "multistart_relative_range_p95": float(
            np.nanpercentile(frame[multistart_columns].to_numpy(dtype=float), 95)
        ),
        "jacobian_condition_number_p95": float(np.nanpercentile(frame["jacobian_condition_number"], 95)),
        "noise_relative_sd_p95": float(np.nanpercentile(frame[noise_columns].to_numpy(dtype=float), 95)),
    }


def _evaluate_gate(
    contract: dict[str, Any],
    fit_frame: pd.DataFrame,
    mode: str,
    config: Stage1Config,
) -> dict[str, Any]:
    if contract["status"] != "PASS":
        return {"decision": "STOP", "reason": "critical_data_contract_failure", "checks": []}
    thresholds = _gate_values(config)
    if mode == "development":
        summary = _aggregate_metrics(fit_frame[fit_frame["split"] == "valid"])
        return {
            "decision": "NOT_EVALUATED",
            "reason": "development_run_must_lock_thresholds_before_formal_test",
            "validation_summary": summary,
            "thresholds": thresholds,
            "checks": [],
        }
    if any(value is None for value in thresholds.values()):
        return {
            "decision": "STOP",
            "reason": "formal_test_requires_all_gate_thresholds_to_be_predefined",
            "thresholds": thresholds,
            "checks": [],
        }
    summary = _aggregate_metrics(fit_frame[fit_frame["split"] == "test"])
    if int(summary.get("n_spectra", 0)) == 0:
        return {"decision": "STOP", "reason": "formal_test_has_no_test_spectra", "checks": []}
    comparisons = {
        "median_log_rmse": (summary["median_log_rmse"], thresholds["median_log_rmse_max"]),
        "median_sam_rad": (summary["median_sam_rad"], thresholds["median_sam_rad_max"]),
        "boundary_fraction": (summary["boundary_fraction"], thresholds["boundary_fraction_max"]),
        "multistart_relative_range_p95": (
            summary["multistart_relative_range_p95"],
            thresholds["multistart_relative_range_max"],
        ),
        "jacobian_condition_number_p95": (
            summary["jacobian_condition_number_p95"],
            thresholds["jacobian_condition_number_max"],
        ),
        "noise_relative_sd_p95": (summary["noise_relative_sd_p95"], thresholds["noise_relative_sd_max"]),
    }
    checks = [
        {"metric": key, "value": float(value), "maximum": float(maximum), "pass": bool(value <= maximum)}
        for key, (value, maximum) in comparisons.items()
    ]
    return {
        "decision": "PASS" if all(check["pass"] for check in checks) else "REVISE",
        "reason": "all_predefined_test_checks_passed" if all(check["pass"] for check in checks) else "one_or_more_test_checks_failed",
        "test_summary": summary,
        "thresholds": thresholds,
        "checks": checks,
    }


def _plot_diagnostics(fit_frame: pd.DataFrame, residual_frame: pd.DataFrame, figure_dir: Path) -> None:
    """Create a claim-led diagnostic grid; these plots never select the model."""

    # Figure contract: show cohort-level agreement, structured residuals, and
    # parameter support/boundary behavior without selecting convenient cases.
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    figure_dir.mkdir(parents=True, exist_ok=True)
    width_in = 183.0 / 25.4
    height_in = 62.0 / 25.4
    fig, axes = plt.subplots(1, 3, figsize=(width_in, height_in), constrained_layout=True)
    grouped = residual_frame.groupby("wavelength_nm", sort=True)
    wavelength = np.asarray(sorted(residual_frame["wavelength_nm"].unique()), dtype=float)

    def quantile(column: str, q: float) -> np.ndarray:
        return grouped[column].quantile(q).reindex(wavelength).to_numpy(dtype=float)

    measured_median = quantile("observed_reflectance", 0.5)
    measured_q1 = quantile("observed_reflectance", 0.25)
    measured_q3 = quantile("observed_reflectance", 0.75)
    predicted_median = quantile("predicted_reflectance", 0.5)
    predicted_q1 = quantile("predicted_reflectance", 0.25)
    predicted_q3 = quantile("predicted_reflectance", 0.75)
    axes[0].fill_between(wavelength, measured_q1, measured_q3, color="#9AA5B1", alpha=0.28, linewidth=0)
    axes[0].plot(wavelength, measured_median, color="#53606D", lw=1.2, label="Measured median")
    axes[0].fill_between(wavelength, predicted_q1, predicted_q3, color="#4C78A8", alpha=0.20, linewidth=0)
    axes[0].plot(wavelength, predicted_median, color="#2F5D8A", lw=1.2, label="Fitted median")
    axes[0].set(xlabel="Wavelength (nm)", ylabel="Reflectance", title="Cohort spectral fit")
    axes[0].legend(loc="best")

    residual_median = quantile("residual", 0.5)
    residual_q1 = quantile("residual", 0.25)
    residual_q3 = quantile("residual", 0.75)
    axes[1].axhline(0.0, color="#A7A7A7", lw=0.8)
    axes[1].fill_between(wavelength, residual_q1, residual_q3, color="#E39C5A", alpha=0.25, linewidth=0)
    axes[1].plot(wavelength, residual_median, color="#B56727", lw=1.2)
    axes[1].set(xlabel="Wavelength (nm)", ylabel="Fitted - measured", title="Wavelength residuals")

    boundary = fit_frame["boundary_M"] | fit_frame["boundary_Hb"]
    axes[2].scatter(
        fit_frame.loc[~boundary, "M_absorbance"],
        fit_frame.loc[~boundary, "Hb_absorbance_proxy"],
        s=10,
        color="#4C78A8",
        alpha=0.70,
        linewidths=0,
        label="Interior",
    )
    if bool(boundary.any()):
        axes[2].scatter(
            fit_frame.loc[boundary, "M_absorbance"],
            fit_frame.loc[boundary, "Hb_absorbance_proxy"],
            s=14,
            facecolors="none",
            edgecolors="#B33A3A",
            linewidths=0.8,
            label="Near bound",
        )
    axes[2].set(
        xlabel="M absorbance (dimensionless)",
        ylabel="Hb proxy at 570 nm (mm^-1)",
        title="Parameter support",
    )
    axes[2].legend(loc="best")
    axes[0].text(0.02, 0.98, f"n={len(fit_frame)} spectra", transform=axes[0].transAxes, va="top")
    for label, axis in zip(("a", "b", "c"), axes, strict=True):
        axis.text(-0.18, 1.07, label, transform=axis.transAxes, fontsize=8, fontweight="bold", va="top")
    fig.savefig(figure_dir / "stage1_diagnostics.svg", bbox_inches="tight")
    fig.savefig(figure_dir / "stage1_diagnostics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _decision_markdown(
    decision: dict[str, Any],
    contract: dict[str, Any],
    summaries: dict[str, Any],
    mode: str,
) -> str:
    lines = [
        "# STAGE1 决策",
        "",
        f"- 判定：**{decision['decision']}**",
        f"- 运行模式：`{mode}`",
        f"- 数据合同：`{contract['status']}`",
        f"- 原因：`{decision['reason']}`",
        "",
        "## 分组汇总",
        "",
        "```json",
        json.dumps(summaries, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 解释边界",
        "",
        "本阶段输出是皮肤前向模型拟合参数，不是真实黑色素或血红蛋白浓度。",
        "开发模式不读取 Test 光谱，也不能产生 PASS；只有在 Train/Validation 上预先锁定全部阈值后，正式测试模式才允许评估 PASS/REVISE。",
        "候选 v1 是简化的两参数 Kubelka-Munk 筛查模型，不等同于 Jonasson 三层逆 Monte Carlo 模型。",
        "",
    ]
    return "\n".join(lines)


def run_stage1(
    data_root: str | Path,
    mask_root: str | Path,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    mode: str = "development",
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Run stage one without ever overwriting an existing output directory."""

    if mode not in {"development", "formal-test"}:
        raise ValueError("mode must be 'development' or 'formal-test'")
    if mode == "formal-test" and max_samples is not None:
        raise ValueError("formal-test cannot use max_samples")
    config = load_stage1_config(config_path)
    output = Path(output_dir).resolve()
    _require_new_output_directory(output)
    figure_dir = output / "figures"
    figure_dir.mkdir()
    samples = discover_hyperskin_vis(data_root, mask_root)
    expected_counts = config.raw["data"]["expected_full_release_counts"] if mode == "formal-test" else None
    contract = audit_data_contract(
        samples,
        expected_counts=expected_counts,
        require_masks=bool(config.raw["data"]["require_skin_masks"]),
    )
    _write_json(output / "data_contract.json", contract)
    pd.DataFrame([sample.record() for sample in samples]).to_csv(output / "split_manifest.csv", index=False)
    with (output / "parameter_contract.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config.raw["parameters"], stream, sort_keys=False, allow_unicode=True)
    with (output / "forward_model_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config.raw["forward_model"], stream, sort_keys=False, allow_unicode=True)
    _write_json(output / "global_calibration.json", {"performed": False, "global_reflectance_scale": 1.0})
    _write_json(
        output / "environment.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "config_source": str(config.path),
        },
    )
    if contract["status"] != "PASS":
        decision = _evaluate_gate(contract, pd.DataFrame(), mode, config)
        _write_json(output / "identifiability_report.json", decision)
        (output / "STAGE1_DECISION.md").write_text(
            _decision_markdown(decision, contract, {}, mode), encoding="utf-8"
        )
        return decision

    # A formal test must stop before opening any Test cube when thresholds have
    # not already been frozen in the supplied configuration.
    if mode == "formal-test" and any(value is None for value in _gate_values(config).values()):
        decision = {
            "decision": "STOP",
            "reason": "formal_test_requires_all_gate_thresholds_to_be_predefined",
            "thresholds": _gate_values(config),
            "checks": [],
        }
        _write_json(output / "identifiability_report.json", decision)
        (output / "STAGE1_DECISION.md").write_text(
            _decision_markdown(decision, contract, {}, mode), encoding="utf-8"
        )
        return decision

    selected_splits = {"train", "valid"} if mode == "development" else {"train", "valid", "test"}
    selected = [sample for sample in samples if sample.split in selected_splits]
    if max_samples is not None:
        selected = selected[: int(max_samples)]
    model = SkinForwardModel(config)
    _write_json(output / "forward_model_provenance.json", model.provenance())
    wavelengths = np.asarray(config.wavelength_nm, dtype=np.float64)
    fit_rows: list[dict[str, Any]] = []
    start_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for sample_index, sample in enumerate(selected):
        try:
            cube = load_hyperskin_cube(
                sample.hsi_path,
                dataset_key=str(config.raw["data"]["mat_dataset_key"]),
                expected_bands=int(config.raw["data"]["expected_bands"]),
            )
            if sample.mask_path is None:
                raise ValueError("Missing required skin mask")
            mask = load_binary_mask(sample.mask_path, cube.shape[:2])
            observed, qc = extract_mean_skin_spectrum(
                cube,
                mask,
                reflectance_epsilon=float(config.raw["fit"]["reflectance_epsilon"]),
                reflectance_max=float(config.raw["data"]["reflectance_max"]),
            )
            result: SpectrumFitResult = fit_spectrum(
                observed,
                wavelengths,
                model,
                config,
                seed_offset=sample_index,
                run_noise_audit=True,
            )
            ident = result.identifiability
            fit_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "subject_id": sample.subject_id,
                    "split": sample.split,
                    "region_id": "supplied_skin_mask",
                    "M_absorbance": float(result.theta[0]),
                    "Hb_absorbance_proxy": float(result.theta[1]),
                    "objective": result.objective,
                    **result.metrics,
                    **qc,
                    "boundary_M": bool(ident["boundary_flags"][0]),
                    "boundary_Hb": bool(ident["boundary_flags"][1]),
                    "multistart_relative_range_M": float(ident["multistart_theta_relative_range"][0]),
                    "multistart_relative_range_Hb": float(ident["multistart_theta_relative_range"][1]),
                    "jacobian_condition_number": float(ident["jacobian_condition_number"]),
                    "noise_relative_sd_M": float(ident["noise_theta_relative_sd"][0]),
                    "noise_relative_sd_Hb": float(ident["noise_theta_relative_sd"][1]),
                }
            )
            for start in result.starts:
                start_rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "subject_id": sample.subject_id,
                        "split": sample.split,
                        **start,
                    }
                )
            for band_index, wavelength in enumerate(wavelengths):
                residual_rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "subject_id": sample.subject_id,
                        "split": sample.split,
                        "wavelength_nm": float(wavelength),
                        "observed_reflectance": float(observed[band_index]),
                        "predicted_reflectance": float(result.predicted[band_index]),
                        "residual": float(result.predicted[band_index] - observed[band_index]),
                    }
                )
        except Exception as error:  # Keep a complete, auditable failure ledger.
            failures.append({"sample_id": sample.sample_id, "split": sample.split, "error": repr(error)})

    fit_frame = pd.DataFrame(fit_rows)
    start_frame = pd.DataFrame(start_rows)
    residual_frame = pd.DataFrame(residual_rows)
    fit_frame.to_parquet(output / "per_spectrum_fit.parquet", index=False)
    start_frame.to_parquet(output / "multistart_stability.parquet", index=False)
    residual_frame.to_csv(output / "wavelength_residuals.csv", index=False)
    _write_json(output / "fit_failures.json", failures)
    if failures:
        contract["status"] = "FAIL"
        contract["issues"].append(
            {
                "severity": "critical",
                "code": "SPECTRUM_PROCESSING_FAILURES",
                "message": f"{len(failures)} selected spectra failed processing or fitting",
            }
        )
        _write_json(output / "data_contract.json", contract)
    summaries = {
        split: _aggregate_metrics(fit_frame[fit_frame["split"] == split])
        for split in sorted(selected_splits)
    }
    decision = _evaluate_gate(contract, fit_frame, mode, config)
    identifiability_report = {
        "decision": decision,
        "summaries": summaries,
        "model_provenance": model.provenance(),
        "fit_failure_count": len(failures),
    }
    _write_json(output / "identifiability_report.json", identifiability_report)
    (output / "STAGE1_DECISION.md").write_text(
        _decision_markdown(decision, contract, summaries, mode), encoding="utf-8"
    )
    if not fit_frame.empty and not residual_frame.empty:
        _plot_diagnostics(fit_frame, residual_frame, figure_dir)
    shutil.copy2(config.path, output / "resolved_stage1_config.yaml")
    return decision
