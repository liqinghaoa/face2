"""S1-5R subject-LOO Train-only development for the revised model family."""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import scipy
import torch
import yaml
import pyarrow.parquet as pq
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr

from .metrics import spectral_angle_rad
from .s1_proxy_inverse import fit_group_isolated_pca, fit_proxy_wls, fit_semi_mechanistic_least_squares
from .s1_revised_forward import RevisedOpticalAssets, RevisedSkinForwardModel, file_sha256, load_revised_registry, revised_basis


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WAVELENGTHS = np.arange(400.0, 701.0, 10.0)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _new_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty S1-5R output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _spectrum(row: Mapping[str, Any], statistic: str = "median") -> np.ndarray:
    return np.asarray([float(row[f"reflectance_{statistic}_{int(w)}nm"]) for w in WAVELENGTHS], dtype=np.float64)


def _center(values: np.ndarray) -> np.ndarray:
    return values - np.mean(values, axis=-1, keepdims=True)


def _side(region: str) -> str:
    return "image_left" if region == "left_cheek" else "image_right" if region == "right_cheek" else "other"


def _side_gains(primary: pd.DataFrame, excluded: str | None) -> dict[str, float]:
    selected = primary if excluded is None else primary[primary.subject_id != excluded]
    differences = []
    for _, group in selected.groupby("subject_id", sort=True):
        by_region = {str(r.region): np.log(np.maximum(_spectrum(r._asdict()), 1e-8)) for r in group.itertuples(index=False)}
        if set(by_region) != {"left_cheek", "right_cheek"}:
            raise RuntimeError("Primary Train cheeks are not paired")
        differences.append(float(np.mean(by_region["left_cheek"] - by_region["right_cheek"])))
    d = float(np.median(differences))
    return {"image_left": d / 2.0, "image_right": -d / 2.0, "other": 0.0}


def _subject_log_spectra(primary: pd.DataFrame, gains: Mapping[str, float], excluded: str | None) -> tuple[np.ndarray, np.ndarray]:
    rows, ids = [], []
    selected = primary if excluded is None else primary[primary.subject_id != excluded]
    for subject, group in selected.groupby("subject_id", sort=True):
        logs = [np.log(np.maximum(_spectrum(r._asdict()), 1e-8)) - gains[_side(str(r.region))] for r in group.itertuples(index=False)]
        rows.append(np.mean(np.stack(logs), axis=0))
        ids.append(str(subject))
    return np.stack(rows), np.asarray(ids)


def _metrics(observed: np.ndarray, predicted: np.ndarray, y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    log_residual = np.log(np.maximum(predicted, 1e-8)) - np.log(np.maximum(observed, 1e-8))
    return {
        "raw_log_rmse": float(np.sqrt(np.mean(log_residual ** 2))),
        "raw_sam": float(spectral_angle_rad(predicted, observed, 1e-8)),
        "shape_log_rmse": float(np.sqrt(np.mean((yhat - y) ** 2))),
        "shape_mae": float(np.mean(np.abs(yhat - y))),
    }


def _project(x: np.ndarray, mean: np.ndarray, components: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    score = (x - mean) @ components
    return mean + components @ score, score


def _boundary(theta: np.ndarray, bounds: tuple[tuple[float, float], ...], fraction: float = 0.01) -> bool:
    return any(v <= lo + fraction * (hi - lo) or v >= hi - fraction * (hi - lo) for v, (lo, hi) in zip(theta, bounds, strict=True))


@dataclass
class FoldContext:
    subject_id: str
    gains: dict[str, float]
    ref_log: np.ndarray
    shape_pca_mean: np.ndarray
    shape_pca: np.ndarray
    raw_pca_mean: np.ndarray
    raw_pca: np.ndarray
    k2_gain: float


def _fit_k2_gain(subject_logs: np.ndarray, registry: Any, iterations: int, max_nfev: int) -> float:
    model = RevisedSkinForwardModel("K2-MH-KM", registry)
    g = 1.0
    spectra = np.exp(subject_logs)
    for _ in range(iterations):
        log_offsets = []
        for observed in spectra:
            fit = fit_semi_mechanistic_least_squares(
                "K2-MH-KM", observed, registry=registry, global_params={"g_system": g}, max_nfev=max_nfev,
            )
            base = model.forward_numpy(fit["theta"], WAVELENGTHS, {"g_system": 1.0})
            log_offsets.append(float(np.mean(np.log(observed) - np.log(base))))
        g = float(np.exp(np.mean(log_offsets)))
    return g


def _build_folds(primary: pd.DataFrame, registry: Any, config: Mapping[str, Any], progress: Callable[[str], None]) -> dict[str, FoldContext]:
    result = {}
    subjects = sorted(primary.subject_id.unique())
    for index, subject in enumerate(subjects, 1):
        gains = _side_gains(primary, subject)
        logs, ids = _subject_log_spectra(primary, gains, subject)
        ref_log = logs.mean(axis=0)
        shape_train = _center(logs - ref_log)
        raw_train = logs - ref_log
        shape_pca = fit_group_isolated_pca(shape_train, ids)
        raw_pca = fit_group_isolated_pca(raw_train, ids)
        k2_gain = _fit_k2_gain(
            logs, registry, int(config["fit"]["k2_global_gain_iterations"]),
            int(config["fit"]["k2_max_function_evaluations"]),
        )
        result[subject] = FoldContext(subject, gains, ref_log, shape_pca.mean, shape_pca.components,
                                      raw_pca.mean, raw_pca.components, k2_gain)
        progress(f"LOO calibration: {index}/{len(subjects)}")
    return result


def _fit_primary(primary: pd.DataFrame, folds: Mapping[str, FoldContext], registry: Any, config: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    records, residuals, failures = [], [], []
    d2_model = RevisedSkinForwardModel("D2-MH", registry)
    for row_tuple in primary.sort_values(["subject_id", "region"]).itertuples(index=False):
        row = row_tuple._asdict()
        subject, region = str(row["subject_id"]), str(row["region"])
        ctx = folds[subject]
        gain = float(np.exp(ctx.gains[_side(region)]))
        observed = _spectrum(row)
        corrected = observed / gain
        ref = np.exp(ctx.ref_log)
        d = np.log(corrected) - ctx.ref_log
        a = float(np.mean(d))
        y = _center(d)
        base = {"sample_id": row["sample_id"], "subject_id": subject, "expression": row["expression"],
                "direction": row["direction"], "region": region}
        candidates: list[tuple[str, np.ndarray, np.ndarray, dict[str, float], bool]] = []
        b0s = np.zeros(31)
        candidates.append(("B0-S", ref * np.exp(a + b0s) * gain, b0s, {}, False))
        pca_y, pca_score = _project(y, ctx.shape_pca_mean, ctx.shape_pca)
        candidates.append(("B2-PCA", ref * np.exp(a + pca_y) * gain, pca_y,
                           {"pca_1": pca_score[0], "pca_2": pca_score[1]}, False))
        for model_id in ("D1-M", "D2-MH", "D3-MHG"):
            try:
                fit = fit_proxy_wls(model_id, corrected, ref, registry=registry)
                pred = ref * np.exp(fit.amplitude_log + fit.fitted_centered_log_ratio) * gain
                candidates.append((model_id, pred, fit.fitted_centered_log_ratio,
                                   dict(zip(fit.parameter_names, fit.theta, strict=True)), False))
            except Exception as exc:
                failures.append({**base, "model_id": model_id, "error": repr(exc)})
        b0r = ref * gain
        candidates.append(("B0-R", b0r, np.zeros(31), {}, False))
        rpca_d, rpca_score = _project(d, ctx.raw_pca_mean, ctx.raw_pca)
        rpca_pred = ref * np.exp(rpca_d) * gain
        candidates.append(("B2-RPCA", rpca_pred, _center(rpca_d),
                           {"pca_raw_1": rpca_score[0], "pca_raw_2": rpca_score[1]}, False))
        try:
            kfit = fit_semi_mechanistic_least_squares(
                "K2-MH-KM", corrected, registry=registry, global_params={"g_system": ctx.k2_gain},
                max_nfev=int(config["fit"]["k2_max_function_evaluations"]),
            )
            kpred_corrected = RevisedSkinForwardModel("K2-MH-KM", registry).forward_numpy(
                kfit["theta"], WAVELENGTHS, {"g_system": ctx.k2_gain}
            )
            kpred = kpred_corrected * gain
            kyhat = _center(np.log(kpred_corrected) - ctx.ref_log)
            candidates.append(("K2-MH-KM", kpred, kyhat,
                               dict(zip(RevisedSkinForwardModel("K2-MH-KM", registry).parameter_names, kfit["theta"], strict=True)),
                               _boundary(kfit["theta"], RevisedSkinForwardModel("K2-MH-KM", registry).bounds)))
        except Exception as exc:
            failures.append({**base, "model_id": "K2-MH-KM", "error": repr(exc)})
        for model_id, pred, yhat, params, boundary in candidates:
            rec = {**base, "model_id": model_id, **_metrics(observed, pred, y, yhat), "a_obs": a,
                   "side_log_gain": ctx.gains[_side(region)], "k2_global_gain": ctx.k2_gain,
                   "any_boundary": boundary, **params}
            records.append(rec)
            if model_id == "D2-MH":
                residuals.extend({**base, "wavelength_nm": float(w), "shape_residual": float(yhat[j] - y[j]),
                                  "observed_shape": float(y[j]), "predicted_shape": float(yhat[j])}
                                 for j, w in enumerate(WAVELENGTHS))
    return pd.DataFrame(records), pd.DataFrame(residuals), failures


def _activation_and_stability(fits: pd.DataFrame, residuals: pd.DataFrame, config: Mapping[str, Any], registry: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    d2 = fits[fits.model_id == "D2-MH"].groupby("subject_id")[["delta_M_OD", "delta_Hb_OD"]].mean()
    d3 = fits[fits.model_id == "D3-MHG"].groupby("subject_id")[["delta_M_OD", "delta_Hb_OD"]].mean()
    shifts, correlations = {}, {}
    for name in ("delta_M_OD", "delta_Hb_OD"):
        scale = max(float(np.subtract(*np.percentile(d2[name], [75, 25]))), 1e-8)
        shifts[name] = float(np.median(np.abs(d3[name] - d2[name])) / scale)
        correlations[name] = float(spearmanr(d2[name], d3[name]).statistic)
    stability = {"subject_count": len(d2), "spearman_D2_vs_D3": correlations,
                 "median_absolute_shift_normalized_by_D2_IQR": shifts}
    basis = revised_basis(WAVELENGTHS, RevisedSkinForwardModel("D2-MH", registry).assets)
    phi_o = basis["phi_O"]
    subject_residual = residuals.groupby(["subject_id", "wavelength_nm"]).shape_residual.mean().unstack().loc[:, WAVELENGTHS]
    corrs = np.asarray([np.corrcoef(row, phi_o)[0, 1] for row in subject_residual.to_numpy()])
    gate = config["conditional_activation"]["d3_mho"]
    median_corr = float(np.median(corrs))
    consistent = float(np.mean(np.sign(corrs) == np.sign(median_corr)))
    strong = float(np.mean(np.abs(corrs) >= float(gate["subject_correlation_abs_min"])))
    activation = {"D3-MHO": {"activated": bool(abs(median_corr) >= float(gate["median_subject_residual_correlation_abs_min"])
                                                   and consistent >= float(gate["consistent_subject_fraction_min"])),
                                  "median_subject_residual_correlation": median_corr,
                                  "direction_consistent_subject_fraction": consistent,
                                  "strong_subject_fraction": strong}}
    return activation, stability


def _perturb(primary: pd.DataFrame, folds: Mapping[str, FoldContext], registry: Any, config: Mapping[str, Any], main_fits: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(int(config["fit"]["perturbation_seed"]))
    repeats = int(config["fit"]["perturbation_repeats"])
    main = main_fits[main_fits.model_id == "D2-MH"].set_index(["sample_id", "region"])
    scales = {name: max(float(np.subtract(*np.percentile(main[name], [75, 25]))), 1e-8) for name in ("delta_M_OD", "delta_Hb_OD")}
    records = []
    for row_tuple in primary.itertuples(index=False):
        row = row_tuple._asdict(); subject = str(row["subject_id"]); region = str(row["region"]); ctx = folds[subject]
        observed = _spectrum(row); corrected = observed / np.exp(ctx.gains[_side(region)]); ref = np.exp(ctx.ref_log)
        mad = np.asarray([float(row[f"reflectance_mad_{int(w)}nm"]) for w in WAVELENGTHS])
        trimmed = np.asarray([float(row[f"reflectance_trimmed_mean_{int(w)}nm"]) for w in WAVELENGTHS])
        pixels = min(int(row["effective_pixels"]), 100)
        noise_scale = np.maximum.reduce((1.4826 * mad / np.sqrt(pixels), np.abs(trimmed - observed), 0.005 * observed))
        baseline = main.loc[(row["sample_id"], region)]
        for repeat in range(repeats):
            perturbed = np.maximum(corrected + rng.normal(0.0, noise_scale), 1e-8)
            fit = fit_proxy_wls("D2-MH", perturbed, ref, registry=registry)
            records.append({"sample_id": row["sample_id"], "subject_id": subject, "region": region, "repeat": repeat,
                            **{name: float(value) for name, value in zip(fit.parameter_names, fit.theta, strict=True)},
                            **{f"normalized_shift_{name}": float(abs(value - baseline[name]) / scales[name])
                               for name, value in zip(fit.parameter_names, fit.theta, strict=True)}})
    return pd.DataFrame(records)


def _stress(train: pd.DataFrame, folds: Mapping[str, FoldContext], registry: Any) -> pd.DataFrame:
    records = []
    selected = train[(train.extraction_status == "USABLE") & train.s1_2_usable_flag.astype(bool)]
    for row_tuple in selected.itertuples(index=False):
        row = row_tuple._asdict(); subject = str(row["subject_id"]); region = str(row["region"]); ctx = folds[subject]
        observed = _spectrum(row); corrected = observed / np.exp(ctx.gains[_side(region)]); ref = np.exp(ctx.ref_log)
        fit = fit_proxy_wls("D2-MH", corrected, ref, registry=registry)
        target = _center(np.log(corrected) - ctx.ref_log)
        records.append({"sample_id": row["sample_id"], "subject_id": subject, "expression": row["expression"],
                        "direction": row["direction"], "region": region,
                        "analysis_role": row["s1_2_analysis_role"], "delta_M_OD": fit.theta[0],
                        "delta_Hb_OD": fit.theta[1], "shape_log_rmse": float(np.sqrt(np.mean((fit.fitted_centered_log_ratio-target)**2)))})
    return pd.DataFrame(records)


def _fwhm_assets(registry: Any, fwhm: float) -> RevisedOpticalAssets:
    source = RevisedOpticalAssets.load(registry.srf_sensitivity_path)
    if fwhm == 0:
        return source
    step = float(np.median(np.diff(source.wavelength_nm)))
    sigma = fwhm / 2.354820045 / step
    return RevisedOpticalAssets(source.wavelength_nm, gaussian_filter1d(source.hbo2_mm1, sigma),
                                gaussian_filter1d(source.hb_mm1, sigma), gaussian_filter1d(source.baseline_mm1, sigma),
                                source.source_path, source.source_sha256)


def _sensitivity(primary: pd.DataFrame, folds: Mapping[str, FoldContext], registry: Any, config: Mapping[str, Any]) -> pd.DataFrame:
    records = []
    conditions = []
    for name, value in config["sensitivity"]["band_sets"].items(): conditions.append(("band_set", name, value))
    for value in config["sensitivity"]["melanin_power"]: conditions.append(("melanin_power", str(value), value))
    for value in config["sensitivity"]["vessel_diameter_um"]: conditions.append(("vessel_diameter_um", str(value), value))
    for value in config["sensitivity"]["sO2_ref"]: conditions.append(("sO2_ref", str(value), value))
    for value in config["sensitivity"]["assumed_gaussian_fwhm_nm"]: conditions.append(("fwhm_nm", str(value), value))
    for value in config["sensitivity"]["center_shift_nm"]: conditions.append(("center_shift_nm", str(value), value))
    for subject, group in primary.groupby("subject_id", sort=True):
        ctx = folds[str(subject)]
        logs = [np.log(_spectrum(r._asdict())) - ctx.gains[_side(str(r.region))] for r in group.itertuples(index=False)]
        observed = np.exp(np.mean(np.stack(logs), axis=0)); ref = np.exp(ctx.ref_log)
        for kind, label, value in conditions:
            mask = np.ones(31, dtype=bool); basis_params = {}; assets = None; basis_wave = None
            if kind == "band_set":
                mask = ~np.isin(WAVELENGTHS, np.asarray(value["exclude_nm"], dtype=float))
            elif kind in {"melanin_power", "vessel_diameter_um", "sO2_ref"}:
                basis_params[kind] = float(value)
            elif kind == "fwhm_nm": assets = _fwhm_assets(registry, float(value))
            elif kind == "center_shift_nm":
                # The optical assets cover 400--700 nm; prespecified center
                # shifts therefore use the endpoint-removed 410--690 nm set.
                mask = (WAVELENGTHS > 400) & (WAVELENGTHS < 700)
                basis_wave = WAVELENGTHS + float(value)
            fit = fit_proxy_wls("D2-MH", observed[mask], ref[mask], registry=registry,
                                wavelength_nm=WAVELENGTHS[mask], basis_wavelength_nm=None if basis_wave is None else basis_wave[mask],
                                assets=assets, basis_params=basis_params)
            target = _center(np.log(observed[mask]) - np.log(ref[mask]))
            records.append({"subject_id": subject, "condition_kind": kind, "condition": label,
                            "bands": int(mask.sum()), "delta_M_OD": fit.theta[0], "delta_Hb_OD": fit.theta[1],
                            "shape_log_rmse": float(np.sqrt(np.mean((fit.fitted_centered_log_ratio-target)**2)))})
    return pd.DataFrame(records)


def _rank(value_a: pd.Series, value_b: pd.Series) -> float | None:
    value = float(spearmanr(value_a, value_b).statistic)
    return value if np.isfinite(value) else None


def _sensitivity_summary(frame: pd.DataFrame) -> dict[str, Any]:
    references = {"band_set": "full_31", "melanin_power": "4.3", "vessel_diameter_um": "15.0",
                  "sO2_ref": "0.5", "fwhm_nm": "0.0", "center_shift_nm": "0.0"}
    output = {}
    for kind, reference in references.items():
        group = frame[frame.condition_kind == kind].copy()
        group["condition_text"] = group.condition.astype(str)
        base = group[group.condition_text == reference].set_index("subject_id")
        conditions = {}
        for condition, current in group.groupby("condition_text", sort=True):
            current = current.set_index("subject_id").loc[base.index]
            condition_result = {"median_shape_log_rmse": float(current.shape_log_rmse.median())}
            for name in ("delta_M_OD", "delta_Hb_OD"):
                scale = max(float(np.subtract(*np.percentile(base[name], [75, 25]))), 1e-8)
                condition_result[f"spearman_{name}_vs_reference"] = _rank(base[name], current[name])
                condition_result[f"median_shift_{name}_in_reference_IQR"] = float(np.median(np.abs(current[name]-base[name]))/scale)
                condition_result[f"sign_change_fraction_{name}"] = float(np.mean(np.sign(current[name]) != np.sign(base[name])))
            conditions[condition] = condition_result
        output[kind] = {"reference_condition": reference, "conditions": conditions}
    return output


def _stress_summary(frame: pd.DataFrame) -> dict[str, Any]:
    baseline = frame[(frame.expression == "neutral") & (frame.direction == "front")
                     & frame.region.isin(["left_cheek", "right_cheek"])].groupby("subject_id")[["delta_M_OD", "delta_Hb_OD"]].mean()
    output = {}
    for keys, group in frame.groupby(["expression", "direction", "region"], sort=True):
        current = group.groupby("subject_id")[["delta_M_OD", "delta_Hb_OD"]].mean()
        common = baseline.index.intersection(current.index)
        if len(common) < 10:
            continue
        record = {"subjects": len(common), "median_shape_log_rmse": float(group.shape_log_rmse.median())}
        for name in ("delta_M_OD", "delta_Hb_OD"):
            scale = max(float(np.subtract(*np.percentile(baseline.loc[common, name], [75, 25]))), 1e-8)
            record[f"spearman_{name}_vs_primary"] = _rank(baseline.loc[common, name], current.loc[common, name])
            record[f"median_shift_{name}_in_primary_IQR"] = float(np.median(np.abs(current.loc[common, name]-baseline.loc[common, name]))/scale)
        output["/".join(map(str, keys))] = record
    return output


def _k2_sensitivity(primary: pd.DataFrame, folds: Mapping[str, FoldContext], registry: Any, config: Mapping[str, Any]) -> pd.DataFrame:
    conditions = []
    for value in config["sensitivity"]["vessel_diameter_um"]:
        conditions.append(("vessel_diameter_um", float(value)))
    for value in config["sensitivity"]["sO2_ref"]:
        conditions.append(("sO2_ref", float(value)))
    for value in config["sensitivity"]["k2_epidermis_thickness_mm"]:
        conditions.append(("epidermis_thickness_mm", float(value)))
    for value in config["sensitivity"]["k2_scattering_amplitude_mm1_at_600nm"]:
        conditions.append(("scattering_amplitude_mm1_at_600nm", float(value)))
    records = []
    model = RevisedSkinForwardModel("K2-MH-KM", registry)
    for subject, group in primary.groupby("subject_id", sort=True):
        ctx = folds[str(subject)]
        logs = [np.log(_spectrum(r._asdict())) - ctx.gains[_side(str(r.region))] for r in group.itertuples(index=False)]
        observed = np.exp(np.mean(np.stack(logs), axis=0))
        for kind, value in conditions:
            gp = {"g_system": ctx.k2_gain, kind: value}
            fit = fit_semi_mechanistic_least_squares(
                "K2-MH-KM", observed, registry=registry, global_params=gp,
                max_nfev=int(config["fit"]["k2_max_function_evaluations"]),
            )
            predicted = model.forward_numpy(fit["theta"], WAVELENGTHS, gp)
            records.append({"subject_id": subject, "condition_kind": kind, "condition": value,
                            "M_epi_OD": fit["theta"][0], "f_blood_proxy": fit["theta"][1],
                            "raw_log_rmse": float(np.sqrt(np.mean((np.log(predicted)-np.log(observed))**2))),
                            "any_boundary": _boundary(fit["theta"], model.bounds), "success": fit["success"]})
    return pd.DataFrame(records)


def _append_conditional_models(primary: pd.DataFrame, folds: Mapping[str, FoldContext], registry: Any,
                               config: Mapping[str, Any], fits: pd.DataFrame, activation: Mapping[str, Any],
                               failures: list[dict[str, Any]]) -> pd.DataFrame:
    additions = []
    raw_pivot = fits.pivot_table(index=["sample_id", "region"], columns="model_id", values="raw_log_rmse")
    k2 = fits[fits.model_id == "K2-MH-KM"]
    kcfg = config["conditional_activation"]["k3_mhs_km"]
    activate_k3 = bool((raw_pivot["K2-MH-KM"] < raw_pivot["B0-R"]).mean() >= float(kcfg["k2_improved_spectrum_fraction_min"])
                       and k2.raw_log_rmse.median() < fits[fits.model_id == "B0-R"].raw_log_rmse.median()
                       and k2.any_boundary.mean() <= float(kcfg["k2_boundary_fraction_max"]))
    models = []
    if activation["D3-MHO"]["activated"]: models.append("D3-MHO")
    if activate_k3: models.append("K3-MHS-KM")
    for row_tuple in primary.sort_values(["subject_id", "region"]).itertuples(index=False):
        row = row_tuple._asdict(); subject = str(row["subject_id"]); region = str(row["region"]); ctx = folds[subject]
        side_gain = float(np.exp(ctx.gains[_side(region)])); observed = _spectrum(row); corrected = observed / side_gain
        ref = np.exp(ctx.ref_log); y = _center(np.log(corrected)-ctx.ref_log); base = {
            "sample_id": row["sample_id"], "subject_id": subject, "expression": row["expression"],
            "direction": row["direction"], "region": region}
        for model_id in models:
            try:
                model = RevisedSkinForwardModel(model_id, registry)
                if model_id == "D3-MHO":
                    fit = fit_proxy_wls(model_id, corrected, ref, registry=registry)
                    pred = ref*np.exp(fit.amplitude_log+fit.fitted_centered_log_ratio)*side_gain
                    additions.append({**base, "model_id": model_id, **_metrics(observed, pred, y, fit.fitted_centered_log_ratio),
                                      "a_obs": fit.amplitude_log, "side_log_gain": ctx.gains[_side(region)],
                                      "k2_global_gain": ctx.k2_gain, "any_boundary": False,
                                      **dict(zip(fit.parameter_names, fit.theta, strict=True))})
                else:
                    result = fit_semi_mechanistic_least_squares(model_id, corrected, registry=registry,
                                global_params={"g_system": ctx.k2_gain}, max_nfev=int(config["fit"]["k2_max_function_evaluations"]))
                    pred_corr = model.forward_numpy(result["theta"], WAVELENGTHS, {"g_system": ctx.k2_gain})
                    yhat = _center(np.log(pred_corr)-ctx.ref_log)
                    additions.append({**base, "model_id": model_id, **_metrics(observed, pred_corr*side_gain, y, yhat),
                                      "a_obs": float(np.mean(np.log(corrected)-ctx.ref_log)),
                                      "side_log_gain": ctx.gains[_side(region)], "k2_global_gain": ctx.k2_gain,
                                      "any_boundary": _boundary(result["theta"], model.bounds),
                                      **dict(zip(model.parameter_names, result["theta"], strict=True))})
            except Exception as exc:
                failures.append({**base, "model_id": model_id, "error": repr(exc)})
    return pd.concat([fits, pd.DataFrame(additions)], ignore_index=True) if additions else fits


def _summarize(fits: pd.DataFrame, residuals: pd.DataFrame, perturb: pd.DataFrame, stability: Mapping[str, Any],
               activation: dict[str, Any], config: Mapping[str, Any], registry: Any,
               sensitivity: pd.DataFrame, k2_sensitivity: pd.DataFrame, stress: pd.DataFrame,
               folds: Mapping[str, FoldContext]) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = {}
    for model, group in fits.groupby("model_id", sort=True):
        summary[model] = {"n": len(group), "subjects": int(group.subject_id.nunique()),
                          "median_shape_log_rmse": float(group.shape_log_rmse.median()),
                          "median_raw_log_rmse": float(group.raw_log_rmse.median()),
                          "median_raw_sam": float(group.raw_sam.median()),
                          "boundary_fraction": float(group.any_boundary.mean())}
    pivot = fits.pivot_table(index=["sample_id", "region"], columns="model_id", values="shape_log_rmse")
    raw_pivot = fits.pivot_table(index=["sample_id", "region"], columns="model_id", values="raw_log_rmse")
    improve = float((pivot["D2-MH"] < pivot["B0-S"]).mean())
    ratio = float(summary["D2-MH"]["median_shape_log_rmse"] / summary["B0-S"]["median_shape_log_rmse"])
    denom = pivot["B0-S"] - pivot["B2-PCA"]
    valid = denom > 0
    retention = float(np.median(((pivot.loc[valid, "B0-S"] - pivot.loc[valid, "D2-MH"]) / denom[valid]))) if valid.any() else float("nan")
    k_improve = float((raw_pivot["K2-MH-KM"] < raw_pivot["B0-R"]).mean())
    k_ratio = float(summary["K2-MH-KM"]["median_raw_log_rmse"] / summary["B0-R"]["median_raw_log_rmse"])
    k_shape_ratio = float(summary["K2-MH-KM"]["median_shape_log_rmse"] / summary["D2-MH"]["median_shape_log_rmse"])
    k2_rows = fits[fits.model_id == "K2-MH-KM"]
    k2_parameter_boundaries = {
        "M_epi_OD": float(((k2_rows.M_epi_OD <= 0.01) | (k2_rows.M_epi_OD >= 0.99)).mean()),
        "f_blood_proxy": float(((k2_rows.f_blood_proxy <= 0.0005) | (k2_rows.f_blood_proxy >= 0.0495)).mean()),
    }
    activation["K3-MHS-KM"] = {"activated": bool(
        k_improve >= float(config["conditional_activation"]["k3_mhs_km"]["k2_improved_spectrum_fraction_min"])
        and summary["K2-MH-KM"]["median_raw_log_rmse"] < summary["B0-R"]["median_raw_log_rmse"]
        and summary["K2-MH-KM"]["boundary_fraction"] <= float(config["conditional_activation"]["k3_mhs_km"]["k2_boundary_fraction_max"])),
        "K2_better_fraction": k_improve, "K2_to_B0R_ratio": k_ratio,
        "K2_boundary_fraction": summary["K2-MH-KM"]["boundary_fraction"]}
    basis = revised_basis(WAVELENGTHS, RevisedSkinForwardModel("D2-MH", registry).assets)
    design = np.column_stack((-basis["phi_M"], -basis["phi_H"]))
    condition = float(np.linalg.cond(design))
    max_band_residual = float(residuals.groupby("wavelength_nm").shape_residual.median().abs().max())
    perturb_shift = float(perturb[["normalized_shift_delta_M_OD", "normalized_shift_delta_Hb_OD"]].median().max())
    g = config["development_gates"]
    proxy_checks = {
        "D2_better_fraction": improve >= float(g["proxy"]["d2_better_than_b0s_spectrum_fraction_min"]),
        "D2_to_B0S_ratio": ratio <= float(g["proxy"]["d2_to_b0s_median_shape_log_rmse_ratio_max"]),
        "physical_gain_retention": np.isfinite(retention) and retention >= float(g["proxy"]["median_physical_gain_retention_min"]),
        "D3_MH_spearman": min(stability["spearman_D2_vs_D3"].values()) >= float(g["proxy"]["d3_mh_spearman_min"]),
        "D3_MH_shift": max(stability["median_absolute_shift_normalized_by_D2_IQR"].values()) <= float(g["proxy"]["d3_mh_median_normalized_shift_max"]),
        "design_condition": condition <= float(g["proxy"]["d2_design_condition_max"]),
        "perturbation_shift": perturb_shift <= float(g["proxy"]["perturbation_median_normalized_shift_max"]),
        "band_residual": max_band_residual <= float(g["proxy"]["median_absolute_log_residual_per_band_max"]),
    }
    semi_checks = {
        "K2_better_fraction": k_improve >= float(g["semi_mechanistic"]["k2_better_than_b0r_spectrum_fraction_min"]),
        "K2_to_B0R_ratio": k_ratio <= float(g["semi_mechanistic"]["k2_to_b0r_median_raw_log_rmse_ratio_max"]),
        "K2_boundary": summary["K2-MH-KM"]["boundary_fraction"] <= float(g["semi_mechanistic"]["k2_boundary_fraction_max"]),
        "K2_shape_vs_D2": k_shape_ratio <= float(g["semi_mechanistic"]["k2_to_d2_median_shape_log_rmse_ratio_max"]),
    }
    evidence = {"model_summary": summary, "D2_better_than_B0S_fraction": improve,
                "D2_to_B0S_median_shape_log_rmse_ratio": ratio, "median_physical_gain_retention": retention,
                "K2_better_than_B0R_fraction": k_improve, "K2_to_B0R_median_raw_log_rmse_ratio": k_ratio,
                "K2_to_D2_median_shape_log_rmse_ratio": k_shape_ratio, "D2_design_condition": condition,
                "max_absolute_median_band_shape_residual": max_band_residual,
                "perturbation_median_normalized_shift_max": perturb_shift,
                "D3_stability": stability, "conditional_activation": activation,
                "K2_parameter_boundary_fraction": k2_parameter_boundaries,
                "K2_LOO_global_gain": {"minimum": float(min(v.k2_gain for v in folds.values())),
                                       "median": float(np.median([v.k2_gain for v in folds.values()])),
                                       "maximum": float(max(v.k2_gain for v in folds.values()))},
                "spectral_sensitivity_stability": _sensitivity_summary(sensitivity),
                "K2_physics_sensitivity": {"best_median_raw_log_rmse": float(k2_sensitivity.groupby(["condition_kind", "condition"]).raw_log_rmse.median().min()),
                                           "minimum_boundary_fraction": float(k2_sensitivity.groupby(["condition_kind", "condition"]).any_boundary.mean().min())},
                "condition_stress_stability": _stress_summary(stress),
                "proxy_gate_checks": proxy_checks, "semi_mechanistic_gate_checks": semi_checks}
    proxy_pass = all(proxy_checks.values())
    semi_pass = proxy_pass and all(semi_checks.values())
    decision = {"status": "SEMIMECHANISTIC_READY_FOR_S1_6" if semi_pass else "PROXY_READY_FOR_S1_6" if proxy_pass else "REVISE_OR_STOP",
                "next_stage_allowed": bool(proxy_pass), "authorized_next_stage": "S1-6" if proxy_pass else None,
                "selected_proxy_model": "D2-MH" if proxy_pass else None,
                "selected_semi_mechanistic_model": "K2-MH-KM" if semi_pass else None}
    return evidence, decision


def _assert_physically_train_only(path: Path) -> None:
    parquet = pq.ParquetFile(path)
    split_index = parquet.schema_arrow.names.index("split")
    for index in range(parquet.num_row_groups):
        stats = parquet.metadata.row_group(index).column(split_index).statistics
        if stats is None or stats.min != "train" or stats.max != "train":
            raise RuntimeError(f"S1-5R input row group {index} is not physically Train-only")


def run_s1_5r(config_path: str | Path, output_dir: str | Path,
              supersedes_decision: str | Path | None = None,
              progress: Callable[[str], None] = print) -> dict[str, Any]:
    config_path = _resolve(config_path); output = _resolve(output_dir); _new_output(output)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("stage") != "S1-5R" or config.get("access_policy", {}).get("validation_access_allowed"):
        raise ValueError("Invalid S1-5R Train-only configuration")
    inputs = {key: _resolve(value) for key, value in config["inputs"].items()}
    s13, s14 = _read_json(inputs["s1_3_decision"]), _read_json(inputs["s1_4r_decision"])
    if s13.get("status") != "PASS_FOR_S1_4" or s14.get("status") != "PASS_FOR_REVISED_TRAIN_DEVELOPMENT":
        raise RuntimeError("Upstream decision does not authorize revised Train development")
    if s14.get("authorized_next_stage") != "S1-5R" or s14.get("s1_6_allowed") is not False:
        raise RuntimeError("S1-4R authorization contract mismatch")
    registry = load_revised_registry(inputs["model_registry"])
    _assert_physically_train_only(inputs["region_spectra"])
    columns = pd.read_parquet(inputs["region_spectra"])
    if set(columns.split.unique()) != {"train"}:
        raise RuntimeError("Non-Train rows reached S1-5R")
    roles = config["data_roles"]
    primary = columns[(columns.s1_2_analysis_role == roles["primary_analysis_role"])
                      & (columns.expression == roles["primary_expression"])
                      & (columns.direction == roles["primary_direction"])
                      & columns.region.isin(roles["primary_regions"])
                      & (columns.extraction_status == roles["usable_status"])
                      & columns.s1_2_usable_flag.astype(bool)].copy()
    if primary.subject_id.nunique() != int(roles["expected_subjects"]) or len(primary) != int(roles["expected_primary_spectra"]):
        raise RuntimeError("Primary Train cohort does not match the frozen contract")
    progress("Building 44 subject-LOO calibration folds")
    folds = _build_folds(primary, registry, config, progress)
    progress("Fitting matched primary candidates")
    fits, residuals, failures = _fit_primary(primary, folds, registry, config)
    activation, stability = _activation_and_stability(fits, residuals, config, registry)
    fits = _append_conditional_models(primary, folds, registry, config, fits, activation, failures)
    progress("Running empirical perturbations and condition stress")
    perturb = _perturb(primary, folds, registry, config, fits)
    stress = _stress(columns, folds, registry)
    progress("Running prespecified spectral sensitivities")
    sensitivity = _sensitivity(primary, folds, registry, config)
    k2_sensitivity = _k2_sensitivity(primary, folds, registry, config)
    evidence, decision_core = _summarize(fits, residuals, perturb, stability, activation, config, registry,
                                         sensitivity, k2_sensitivity, stress, folds)
    tables = {"per_spectrum_fits": fits, "d2_wavelength_residuals": residuals, "perturbation_fits": perturb,
              "condition_stress_fits": stress, "spectral_sensitivity": sensitivity,
              "k2_physics_sensitivity": k2_sensitivity}
    for name, frame in tables.items():
        frame.to_parquet(output / f"{name}.parquet", index=False); frame.to_csv(output / f"{name}.csv", index=False)
    _write_json(output / "train_evidence.json", evidence); _write_json(output / "fit_failures.json", {"count": len(failures), "failures": failures})
    audit = {"config": str(config_path), "config_sha256": file_sha256(config_path),
             "inputs": {k: {"path": str(v), "sha256": file_sha256(v)} for k, v in inputs.items()},
             "source_sha256": {p.name: file_sha256(p) for p in [Path(__file__), PROJECT_ROOT / "src/skin_optics_hsi/s1_revised_forward.py",
                                                                  PROJECT_ROOT / "src/skin_optics_hsi/s1_proxy_inverse.py",
                                                                  PROJECT_ROOT / "scripts/skin_optics_hsi/run_s1_5_revised_train.py"]},
             "train_rows_loaded": len(columns), "primary_rows": len(primary), "train_subjects": int(primary.subject_id.nunique()),
             "validation_rows_read": 0, "test_rows_read": 0, "raw_hsi_files_read": 0,
             "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
             "scipy": scipy.__version__, "torch": torch.__version__}
    _write_json(output / "input_audit.json", audit)
    report = f"""# S1-5R Train-only report\n\nStatus: `{decision_core['status']}`  \nNext stage allowed: `{str(decision_core['next_stage_allowed']).lower()}`\n\n- Train subjects: {primary.subject_id.nunique()}\n- Primary spectra: {len(primary)}\n- Validation/Test/raw-HSI content reads: 0/0/0\n- D2/B0-S median shape RMSE ratio: {evidence['D2_to_B0S_median_shape_log_rmse_ratio']:.6f}\n- D2 better than B0-S fraction: {evidence['D2_better_than_B0S_fraction']:.6f}\n- Median physical gain retention versus B2-PCA: {evidence['median_physical_gain_retention']:.6f}\n- K2/B0-R median raw log-RMSE ratio: {evidence['K2_to_B0R_median_raw_log_rmse_ratio']:.6f}\n- K2 boundary fraction: {evidence['model_summary']['K2-MH-KM']['boundary_fraction']:.6f}\n\nThis is a Train development decision, not a formal validation or physiological-concentration claim.\n"""
    (output / "S1_5R_TRAIN_REPORT.md").write_text(report, encoding="utf-8")
    outputs = {p.name: file_sha256(p) for p in output.iterdir() if p.is_file()}
    decision = {"schema_version": 1, "stage": "S1-5R", **decision_core,
                "s1_6_authorization_scope": "Validation model selection and freezing only" if decision_core["next_stage_allowed"] else None,
                "formal_test_allowed": False, "validation_rows_read": 0, "test_rows_read": 0, "raw_hsi_files_read": 0,
                "candidate_activation": activation, "outputs": outputs,
                "created_at_utc": datetime.now(timezone.utc).isoformat()}
    if supersedes_decision is not None:
        previous = _resolve(supersedes_decision)
        if not previous.is_file():
            raise FileNotFoundError(f"Superseded S1-5R decision is missing: {previous}")
        decision["supersedes"] = {"path": str(previous), "sha256": file_sha256(previous)}
    _write_json(output / "s1_5r_decision.json", decision)
    return decision


__all__ = ["run_s1_5r"]
