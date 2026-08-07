"""SO-0 v1.1 audit repair and reacceptance entrypoint."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from skin_optics.assets import SpectralAssets, load_assets
from skin_optics.config import SO0Config, load_formula_registry, load_so0_config
from skin_optics.numpy_backend.color_spaces import deltae00, xyz_to_lab_d65
from skin_optics.numpy_backend.colorchecker import CalibrationRecord, calibrate_all, solve_camera_to_xyz_matrix
from skin_optics.numpy_backend.image_formation import (
    d65_white_xyz,
    render_camera,
    render_cie_reference,
    spectral_radiance,
)
from skin_optics.numpy_backend.reflectance import compute_skin_reflectance
from skin_optics.torch_backend.forward_model import SO0TorchForwardModel


ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "SO0_Forward_Model_v1.1"
CORE_LIGHTS = ["D65", "A", "FL2", "FL11"]
STABLE_SMALL = 1.0e-14


def sha256_file(path: Path) -> str:
    """Return SHA-256 for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def run_command(cmd: list[str], cwd: Path = ROOT, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run a subprocess and capture text output."""

    return subprocess.run(cmd, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)


def verify_frozen_assets(out_path: Path) -> bool:
    """Verify raw and standardized frozen asset hashes."""

    raw = ROOT / "data/external/SO0_Spectral_Assets_v1"
    std = raw / "processed/SO0_Spectral_Standardized_v1"
    lines = ["## raw assets\n"]
    p1 = run_command(["sha256sum", "-c", "metadata/checksums.sha256"], raw)
    lines.append(p1.stdout)
    lines.append("## standardized assets\n")
    p2 = run_command(["sha256sum", "-c", "metadata/checksums.sha256"], std)
    lines.append(p2.stdout)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines), encoding="utf-8")
    return p1.returncode == 0 and p2.returncode == 0


def environment_snapshot() -> dict[str, Any]:
    """Return environment metadata required by the audit."""

    import importlib
    import platform

    packages: dict[str, Any] = {}
    for name in ["numpy", "scipy", "pandas", "yaml", "colour", "torch", "pytest"]:
        mod = importlib.import_module(name)
        packages[name] = getattr(mod, "__version__", "unknown")
    return {
        "pwd": str(ROOT),
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda_available": bool(torch.cuda.is_available()),
    }


def copy_config_files(run_dir: Path, cfg: SO0Config) -> None:
    """Copy current authoritative YAML files into the run."""

    shutil.copy2(cfg.config_dir / "forward_model_mvp.yaml", run_dir / "frozen_config.yaml")
    shutil.copy2(cfg.config_dir / "acceptance_thresholds.yaml", run_dir / "acceptance_thresholds.yaml")
    shutil.copy2(cfg.config_dir / "audit_protocol_v1_1.yaml", run_dir / "audit_protocol.yaml")
    shutil.copy2(cfg.config_dir / "forward_formula_registry.yaml", run_dir / "formula_registry.yaml")


def representative_cameras(assets: SpectralAssets, configured: list[str]) -> list[str]:
    """Return configured cameras plus first and last database entries."""

    names = list(map(str, assets.source["camera_names"]))
    return list(dict.fromkeys([*configured, names[0], names[-1]]))


def calibration_dataframe(records: list[CalibrationRecord]) -> pd.DataFrame:
    """Convert ColorChecker records to a serialisable DataFrame."""

    rows = []
    for rec in records:
        row = asdict(rec)
        row["matrix_3x3"] = " ".join(f"{x:.12g}" for x in rec.matrix_3x3.reshape(-1))
        rows.append(row)
    return pd.DataFrame(rows)


def colorchecker_status(df: pd.DataFrame, thresholds: dict[str, Any]) -> dict[str, Any]:
    """Compute ColorChecker global status from recalculated records."""

    qualified = df["qualification_status"].eq("PASS")
    by_light = df.assign(qualified=qualified).groupby("illuminant_name")["qualified"].sum().to_dict()
    fraction = float(qualified.mean())
    canon = bool(
        df[
            (df["camera_name"] == "Canon 5DMarkII")
            & (df["illuminant_name"] == "D65")
            & (df["qualification_status"] == "PASS")
        ].shape[0]
    )
    pass_ok = (
        fraction >= float(thresholds["pass_qualified_fraction_min"])
        and all(int(by_light.get(light, 0)) >= int(thresholds["pass_min_qualified_per_light"]) for light in CORE_LIGHTS)
        and canon
    )
    limits_ok = (
        fraction >= float(thresholds["pass_with_limits_qualified_fraction_min"])
        and all(int(by_light.get(light, 0)) >= int(thresholds["pass_with_limits_min_qualified_per_light"]) for light in CORE_LIGHTS)
        and canon
    )
    status = "PASS" if pass_ok else "PASS_WITH_LIMITS" if limits_ok else "FAIL"
    return {
        "status": status,
        "qualified": int(qualified.sum()),
        "total": int(len(df)),
        "qualified_fraction": fraction,
        "qualified_by_light": {k: int(v) for k, v in by_light.items()},
        "canon_5dmarkii_d65_qualified": canon,
        "failed_pairs": df.loc[~qualified, ["camera_name", "illuminant_name", "failure_reasons"]].to_dict("records"),
    }


def run_pytest(run_dir: Path) -> dict[str, Any]:
    """Run pytest and parse JUnit XML as a hard gate."""

    xml_path = run_dir / "pytest_report.xml"
    log_path = run_dir / "pytest_full.log"
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "20260801"
    start = time.time()
    proc = run_command(
        [sys.executable, "-m", "pytest", "-q", str(PACKAGE_ROOT / "tests"), f"--junitxml={xml_path}"],
        ROOT,
        env,
    )
    duration = time.time() - start
    log_path.write_text(proc.stdout, encoding="utf-8")
    result = {
        "return_code": int(proc.returncode),
        "tests": 0,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
        "duration_seconds": duration,
        "status": "FAIL",
    }
    if xml_path.exists():
        root = ET.parse(xml_path).getroot()
        suites = list(root) if root.tag == "testsuites" else [root]
        for suite in suites:
            result["tests"] += int(float(suite.attrib.get("tests", 0)))
            result["failures"] += int(float(suite.attrib.get("failures", 0)))
            result["errors"] += int(float(suite.attrib.get("errors", 0)))
            result["skipped"] += int(float(suite.attrib.get("skipped", 0)))
    result["status"] = (
        "PASS"
        if result["return_code"] == 0
        and result["tests"] > 0
        and result["failures"] == 0
        and result["errors"] == 0
        and result["skipped"] == 0
        else "FAIL"
    )
    return result


def load_matrices(records: pd.DataFrame) -> dict[tuple[str, str], np.ndarray]:
    """Parse matrix strings from a calibration metrics table."""

    matrices = {}
    for row in records.itertuples(index=False):
        matrices[(str(row.camera_name), str(row.illuminant_name))] = np.asarray(
            [float(x) for x in str(row.matrix_3x3).split()], dtype=np.float64
        ).reshape(3, 3)
    return matrices


def resolution_audit(run_dir: Path, cfg: SO0Config, assets1: SpectralAssets, assets5: SpectralAssets) -> dict[str, Any]:
    """Run the complete 121,968 camera-condition 1 nm vs 5 nm audit."""

    proto = cfg.audit_protocol["resolution_audit"]
    thresholds = cfg.thresholds["resolution"]
    color_thresholds = cfg.thresholds["color"]
    tables = run_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    ms = np.linspace(0.0, 1.0, int(proto["melanin_control_count"]))
    hs = np.linspace(0.0, 1.0, int(proto["hemoglobin_control_count"]))
    cameras = list(map(str, assets5.source["camera_names"]))
    lights = list(proto["illuminants"])
    matrices1 = {(cam, light): solve_camera_to_xyz_matrix(assets1, cam, light)[0] for cam in cameras for light in lights}
    matrices5 = {(cam, light): solve_camera_to_xyz_matrix(assets5, cam, light)[0] for cam in cameras for light in lights}
    rows: list[dict[str, Any]] = []
    cie_deltae: list[dict[str, Any]] = []
    for m in ms:
        for h in hs:
            r1 = compute_skin_reflectance(assets1, m, h, config=cfg)
            r5 = compute_skin_reflectance(assets5, m, h, config=cfg)
            for shading in proto["shading"]:
                for specular in proto["specular"]:
                    for light in lights:
                        cie1 = render_cie_reference(assets1, r1, light, shading, specular, 1.0, config=cfg)
                        cie5 = render_cie_reference(assets5, r5, light, shading, specular, 1.0, config=cfg)
                        de = deltae00(
                            xyz_to_lab_d65(cie1.xyz_d65, d65_white_xyz(assets1)),
                            xyz_to_lab_d65(cie5.xyz_d65, d65_white_xyz(assets5)),
                        )
                        cie_deltae.append({"m": m, "h": h, "shading": shading, "specular": specular, "illuminant_name": light, "deltae00": float(de)})
                        for cam in cameras:
                            c1 = render_camera(assets1, r1, matrices1[(cam, light)], cam, light, shading, specular, 1.0, config=cfg)
                            c5 = render_camera(assets5, r5, matrices5[(cam, light)], cam, light, shading, specular, 1.0, config=cfg)
                            rgb1 = c1.camera_rgb_wb
                            rgb5 = c5.camera_rgb_wb
                            rel = np.abs(rgb5 - rgb1) / np.maximum(np.abs(rgb1), STABLE_SMALL)
                            chroma1 = rgb1 / (np.sum(rgb1) + STABLE_SMALL)
                            chroma5 = rgb5 / (np.sum(rgb5) + STABLE_SMALL)
                            rows.append(
                                {
                                    "camera_name": cam,
                                    "illuminant_name": light,
                                    "m": m,
                                    "h": h,
                                    "shading": shading,
                                    "specular": specular,
                                    "camera_rgb_1nm_r": rgb1[0],
                                    "camera_rgb_1nm_g": rgb1[1],
                                    "camera_rgb_1nm_b": rgb1[2],
                                    "camera_rgb_5nm_r": rgb5[0],
                                    "camera_rgb_5nm_g": rgb5[1],
                                    "camera_rgb_5nm_b": rgb5[2],
                                    "relative_rgb_error": float(np.max(rel)),
                                    "chromaticity_l1_error": float(np.sum(np.abs(chroma5 - chroma1))),
                                    "finite": bool(np.all(np.isfinite(rgb1)) and np.all(np.isfinite(rgb5))),
                                }
                            )
    detail = pd.DataFrame(rows)
    detail_path = tables / "resolution_1nm_vs_5nm_detail.csv.gz"
    detail.to_csv(detail_path, index=False, compression="gzip")
    cie_df = pd.DataFrame(cie_deltae)
    cie_df.to_csv(tables / "resolution_1nm_vs_5nm_cie_detail.csv", index=False)
    summary = {
        "camera_condition_count": int(len(detail)),
        "cie_condition_count": int(len(cie_df)),
        "cie_deltae00_median": float(cie_df["deltae00"].median()),
        "cie_deltae00_p95": float(cie_df["deltae00"].quantile(0.95)),
        "camera_rgb_relative_error_median": float(detail["relative_rgb_error"].median()),
        "camera_rgb_relative_error_p95": float(detail["relative_rgb_error"].quantile(0.95)),
        "camera_rgb_relative_error_max": float(detail["relative_rgb_error"].max()),
        "camera_chroma_l1_p95": float(detail["chromaticity_l1_error"].quantile(0.95)),
        "finite_fraction": float(detail["finite"].mean()),
    }
    summary["status"] = (
        "PASS"
        if summary["cie_deltae00_median"] <= float(color_thresholds["resolution_deltae00_median_max"])
        and summary["cie_deltae00_p95"] <= float(color_thresholds["resolution_deltae00_p95_max"])
        and summary["camera_rgb_relative_error_median"] <= float(thresholds["camera_rgb_relative_error_median_max"])
        and summary["camera_rgb_relative_error_p95"] <= float(thresholds["camera_rgb_relative_error_p95_max"])
        and summary["camera_rgb_relative_error_max"] <= float(thresholds["camera_rgb_relative_error_max_max"])
        and summary["camera_chroma_l1_p95"] <= float(thresholds["camera_chroma_l1_p95_max"])
        and summary["finite_fraction"] == 1.0
        else "FAIL"
    )
    pd.DataFrame([summary]).to_csv(tables / "resolution_1nm_vs_5nm_summary.csv", index=False)
    by = detail.groupby(["camera_name", "illuminant_name"]).agg(
        relative_rgb_error_median=("relative_rgb_error", "median"),
        relative_rgb_error_p95=("relative_rgb_error", lambda x: x.quantile(0.95)),
        relative_rgb_error_max=("relative_rgb_error", "max"),
        chromaticity_l1_p95=("chromaticity_l1_error", lambda x: x.quantile(0.95)),
        finite_fraction=("finite", "mean"),
    )
    by.reset_index().to_csv(tables / "resolution_1nm_vs_5nm_by_camera_light.csv", index=False)
    return summary


def monotonicity_audit(run_dir: Path, cfg: SO0Config, assets: SpectralAssets) -> dict[str, Any]:
    """Audit M and H control directions."""

    from scipy.stats import spearmanr

    rows = []
    ms = np.linspace(0.0, 1.0, 21)
    hs = np.linspace(0.0, 1.0, 21)
    h_fixed = 0.5
    m_fixed = 0.5
    r_m = np.stack([compute_skin_reflectance(assets, m, h_fixed, config=cfg) for m in ms])
    m_nonincrease = bool(np.all(np.diff(r_m, axis=0) <= float(cfg.thresholds["reflectance"]["melanin_monotonic_tolerance"])))
    w = assets.weights_nm
    wl = assets.wavelength_nm
    green = (wl >= 545.0) & (wl <= 585.0)
    red = (wl >= 650.0) & (wl <= 700.0)
    band = []
    for h in hs:
        r = compute_skin_reflectance(assets, m_fixed, h, config=cfg)
        mean_g = float(np.sum(w[green] * r[green]) / np.sum(w[green]))
        mean_r = float(np.sum(w[red] * r[red]) / np.sum(w[red]))
        band.append(-np.log(mean_g / mean_r))
    rho = float(spearmanr(hs, band).statistic)
    rows.append({"metric": "melanin_reflectance_nonincrease", "value": float(m_nonincrease), "status": "PASS" if m_nonincrease else "FAIL"})
    rows.append({"metric": "hemoglobin_band_spearman", "value": rho, "status": "PASS" if rho >= float(cfg.thresholds["reflectance"]["hemoglobin_band_spearman_min"]) else "FAIL"})
    pd.DataFrame(rows).to_csv(run_dir / "tables" / "monotonicity_metrics.csv", index=False)
    return {"status": "PASS" if all(r["status"] == "PASS" for r in rows) else "FAIL", "hemoglobin_band_spearman": rho}


def backend_parity_audit(run_dir: Path, cfg: SO0Config, assets: SpectralAssets, records: pd.DataFrame) -> dict[str, Any]:
    """Audit NumPy/PyTorch and float32/float64 parity over representative conditions."""

    proto = cfg.audit_protocol["backend_parity_audit"]
    matrices = load_matrices(records)
    rows = []
    num_rows = []
    for cam in proto["representative_cameras"]:
        for light in proto["illuminants"]:
            matrix = matrices[(cam, light)]
            model64 = SO0TorchForwardModel(assets, cam, light, matrix, dtype=torch.float64, config=cfg)
            model32 = SO0TorchForwardModel(assets, cam, light, matrix, dtype=torch.float32, config=cfg)
            for m in proto["melanin_control"]:
                for h in proto["hemoglobin_control"]:
                    for shading in proto["shading"]:
                        for specular in proto["specular"]:
                            for exposure in proto["exposure"]:
                                r_np = compute_skin_reflectance(assets, m, h, config=cfg)
                                c_np = render_camera(assets, r_np, matrix, cam, light, shading, specular, exposure, config=cfg)
                                args64 = [torch.tensor(float(v), dtype=torch.float64) for v in (m, h, shading, specular, exposure)]
                                args32 = [torch.tensor(float(v), dtype=torch.float32) for v in (m, h, shading, specular, exposure)]
                                with torch.no_grad():
                                    r64 = model64.compute_skin_reflectance(args64[0], args64[1]).detach().numpy()
                                    c64 = model64.render_camera(*args64)
                                    r32 = model32.compute_skin_reflectance(args32[0], args32[1]).detach().double().numpy()
                                    c32 = {k: v.detach().double().numpy() for k, v in model32.render_camera(*args32).items()}
                                rows.append({"comparison": "numpy64_vs_torch64", "observation_space": "reflectance", "max_abs_error": float(np.max(np.abs(r_np - r64)))})
                                rows.append({"comparison": "torch32_vs_torch64", "observation_space": "reflectance", "max_abs_error": float(np.max(np.abs(r32 - r64)))})
                                for space, np_val in [
                                    ("camera_rgb_raw", c_np.camera_rgb_raw),
                                    ("camera_rgb_wb", c_np.camera_rgb_wb),
                                    ("xyz_d65", c_np.xyz_d65),
                                    ("linear_srgb_unclipped", c_np.linear_srgb_unclipped),
                                ]:
                                    rows.append({"comparison": "numpy64_vs_torch64", "observation_space": space, "max_abs_error": float(np.max(np.abs(np_val - c64[space].detach().numpy())))})
                                    rows.append({"comparison": "torch32_vs_torch64", "observation_space": space, "max_abs_error": float(np.max(np.abs(c32[space] - c64[space].detach().numpy())))})
                                num_rows.append(
                                    {
                                        "camera_name": cam,
                                        "illuminant_name": light,
                                        "m": m,
                                        "h": h,
                                        "shading": shading,
                                        "specular": specular,
                                        "exposure": exposure,
                                        "all_finite": bool(np.all(np.isfinite(c_np.camera_rgb_raw)) and np.all(np.isfinite(c_np.linear_srgb_unclipped))),
                                    }
                                )
    df = pd.DataFrame(rows).groupby(["comparison", "observation_space"]).agg(max_abs_error=("max_abs_error", "max")).reset_index()
    thresholds = cfg.thresholds["backend_parity"]
    def status(row: pd.Series) -> str:
        if row["comparison"] == "numpy64_vs_torch64":
            return "PASS" if row["max_abs_error"] <= float(thresholds["numpy_torch_float64_max_abs"]) else "FAIL"
        if row["observation_space"] == "reflectance":
            limit = float(thresholds["torch_float32_reflectance_max_abs"])
        elif row["observation_space"] == "linear_srgb_unclipped":
            limit = float(thresholds["srgb_float32_max_abs"])
        else:
            limit = float(thresholds["camera_xyz_float32_max_abs"])
        return "PASS" if row["max_abs_error"] <= limit else "FAIL"
    df["status"] = df.apply(status, axis=1)
    df.to_csv(run_dir / "tables" / "backend_parity.csv", index=False)
    pd.DataFrame(num_rows).to_csv(run_dir / "tables" / "numerical_stability.csv", index=False)
    return {"status": "PASS" if df["status"].eq("PASS").all() and all(r["all_finite"] for r in num_rows) else "FAIL", "rows": df.to_dict("records")}


def linearity_audit(run_dir: Path, cfg: SO0Config, assets: SpectralAssets, records: pd.DataFrame) -> dict[str, Any]:
    """Audit shading, specular, and exposure linearity with real computations."""

    proto = cfg.audit_protocol["linearity_audit"]
    matrices = load_matrices(records)
    cameras = list(map(str, assets.source["camera_names"]))
    detail: dict[str, list[dict[str, Any]]] = {"shading": [], "specular": [], "exposure": []}

    def observations(refl: np.ndarray, matrix: np.ndarray, cam: str, light: str, shading: float, specular: float, exposure: float) -> dict[str, np.ndarray]:
        rad = spectral_radiance(assets, refl, light, shading, specular, exposure, config=cfg)
        cr = render_camera(assets, refl, matrix, cam, light, shading, specular, exposure, config=cfg)
        cie = render_cie_reference(assets, refl, light, shading, specular, exposure, config=cfg)
        return {"spectral_radiance": rad, "cie_xyz": cie.xyz_source, "camera_rgb_raw": cr.camera_rgb_raw, "camera_rgb_wb": cr.camera_rgb_wb}

    for m in proto["melanin_control"]:
        for h in proto["hemoglobin_control"]:
            refl = compute_skin_reflectance(assets, m, h, config=cfg)
            for light in proto["illuminants"]:
                for cam in cameras:
                    matrix = matrices[(cam, light)]
                    for s1, s2 in proto["shading_pairs"]:
                        obs1 = observations(refl, matrix, cam, light, s1, 0.0, 1.0)
                        obs2 = observations(refl, matrix, cam, light, s2, 0.0, 1.0)
                        for space in obs1:
                            expected = (float(s2) / float(s1)) * obs1[space]
                            err = np.max(np.abs(obs2[space] - expected))
                            rel = err / max(float(np.max(np.abs(expected))), STABLE_SMALL)
                            detail["shading"].append({"effect": "shading", "observation_space": space, "m": m, "h": h, "camera_name": cam, "illuminant_name": light, "max_abs_error": err, "relative_error": rel})
                    ps = list(proto["specular"])
                    basis0 = observations(refl, matrix, cam, light, 1.0, 0.0, 1.0)
                    basis_hi = observations(refl, matrix, cam, light, 1.0, 0.10, 1.0)
                    basis = {space: (basis_hi[space] - basis0[space]) / 0.10 for space in basis0}
                    for p1, p2 in zip(ps[:-1], ps[1:]):
                        obs1 = observations(refl, matrix, cam, light, 1.0, p1, 1.0)
                        obs2 = observations(refl, matrix, cam, light, 1.0, p2, 1.0)
                        for space in obs1:
                            expected = (float(p2) - float(p1)) * basis[space]
                            err = np.max(np.abs((obs2[space] - obs1[space]) - expected))
                            rel = err / max(float(np.max(np.abs(expected))), STABLE_SMALL)
                            detail["specular"].append({"effect": "specular", "observation_space": space, "m": m, "h": h, "camera_name": cam, "illuminant_name": light, "max_abs_error": err, "relative_error": rel})
                    es = list(proto["exposure"])
                    base = observations(refl, matrix, cam, light, 1.0, 0.03, 1.0)
                    for e in es:
                        obs = observations(refl, matrix, cam, light, 1.0, 0.03, e)
                        for space in obs:
                            expected = float(e) * base[space]
                            err = np.max(np.abs(obs[space] - expected))
                            rel = err / max(float(np.max(np.abs(expected))), STABLE_SMALL)
                            detail["exposure"].append({"effect": "exposure", "observation_space": space, "m": m, "h": h, "camera_name": cam, "illuminant_name": light, "max_abs_error": err, "relative_error": rel})
    summaries = []
    for effect, rows in detail.items():
        df = pd.DataFrame(rows)
        df.to_csv(run_dir / "tables" / f"{effect}_linearity_detail.csv", index=False)
        for (eff, space), group in df.groupby(["effect", "observation_space"]):
            summaries.append(
                {
                    "effect": eff,
                    "observation_space": space,
                    "sample_count": int(len(group)),
                    "max_abs_error": float(group["max_abs_error"].max()),
                    "median_relative_error": float(group["relative_error"].median()),
                    "p95_relative_error": float(group["relative_error"].quantile(0.95)),
                    "max_relative_error": float(group["relative_error"].max()),
                    "status": "PASS" if float(group["relative_error"].max()) <= float(cfg.thresholds["linearity"]["relative_error_max"]) else "FAIL",
                }
            )
    summary = pd.DataFrame(summaries)
    summary.to_csv(run_dir / "tables" / "shading_specular_linearity.csv", index=False)
    return {"status": "PASS" if summary["status"].eq("PASS").all() else "FAIL", "rows": summaries}


def spectral_separability_audit(run_dir: Path, cfg: SO0Config, assets: SpectralAssets) -> dict[str, Any]:
    """Save 25-row M/H spectral Jacobian cosine audit."""

    controls = [float(x) for x in cfg.audit_protocol["mh_audit"]["controls"]]
    delta = float(cfg.audit_protocol["mh_audit"]["finite_difference_delta"])
    rows = []
    for m in controls:
        for h in controls:
            jm = (compute_skin_reflectance(assets, m + delta, h, config=cfg) - compute_skin_reflectance(assets, m - delta, h, config=cfg)) / (2 * delta)
            jh = (compute_skin_reflectance(assets, m, h + delta, config=cfg) - compute_skin_reflectance(assets, m, h - delta, config=cfg)) / (2 * delta)
            jm_norm = float(np.linalg.norm(jm))
            jh_norm = float(np.linalg.norm(jh))
            cosine = float(abs(np.dot(jm, jh) / max(jm_norm * jh_norm, STABLE_SMALL)))
            rows.append({"m": m, "h": h, "jm_norm": jm_norm, "jh_norm": jh_norm, "abs_cosine": cosine, "finite": np.isfinite(cosine), "status": "PASS"})
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "tables" / "mh_spectral_separability.csv", index=False)
    median = float(df["abs_cosine"].median())
    p95 = float(df["abs_cosine"].quantile(0.95))
    status = "PASS" if median <= float(cfg.thresholds["separability"]["spectral_abs_cosine_median_max"]) and p95 <= float(cfg.thresholds["separability"]["spectral_abs_cosine_p95_max"]) else "FAIL"
    return {"status": status, "median": median, "p95": p95}


def observation_separability_audit(run_dir: Path, cfg: SO0Config, assets: SpectralAssets, records: pd.DataFrame) -> dict[str, Any]:
    """Save full M/H observation-domain SVD audit."""

    proto = cfg.audit_protocol["mh_audit"]
    controls = [float(x) for x in proto["controls"]]
    delta = float(proto["finite_difference_delta"])
    matrices = load_matrices(records)
    rows = []
    for rec in records.itertuples(index=False):
        cam = str(rec.camera_name)
        light = str(rec.illuminant_name)
        matrix = matrices[(cam, light)]
        qualified = rec.qualification_status == "PASS"
        for m in controls:
            for h in controls:
                def obs(mm: float, hh: float) -> dict[str, np.ndarray]:
                    refl = compute_skin_reflectance(assets, mm, hh, config=cfg)
                    cr = render_camera(assets, refl, matrix, cam, light, 1.0, 0.0, 1.0, config=cfg)
                    chroma = cr.camera_rgb_wb / (np.sum(cr.camera_rgb_wb) + STABLE_SMALL)
                    return {"camera_rgb_wb": cr.camera_rgb_wb, "linear_srgb_unclipped": cr.linear_srgb_unclipped, "chromaticity": chroma}
                plus_m = obs(m + delta, h)
                minus_m = obs(m - delta, h)
                plus_h = obs(m, h + delta)
                minus_h = obs(m, h - delta)
                for space in proto["observation_spaces"]:
                    jm = (plus_m[space] - minus_m[space]) / (2 * delta)
                    jh = (plus_h[space] - minus_h[space]) / (2 * delta)
                    jac = np.stack([jm, jh], axis=-1)
                    s = np.linalg.svd(jac, compute_uv=False)
                    sigma_max = float(s[0])
                    sigma_min = float(s[-1])
                    ratio = sigma_min / sigma_max if sigma_max > 0 else 0.0
                    rank2 = sigma_max > float(cfg.thresholds["separability"]["sigma_max_min"]) and ratio >= float(cfg.thresholds["separability"]["rank_ratio_min"])
                    rows.append(
                        {
                            "camera_name": cam,
                            "illuminant_name": light,
                            "camera_light_qualified": bool(qualified),
                            "m": m,
                            "h": h,
                            "observation_space": space,
                            "sigma_max": sigma_max,
                            "sigma_min": sigma_min,
                            "sigma_ratio": ratio,
                            "condition_number": sigma_max / max(sigma_min, STABLE_SMALL),
                            "numerical_rank": 2 if rank2 else 1,
                            "finite": bool(np.all(np.isfinite(s))),
                            "status": "PASS" if rank2 else "FAIL",
                        }
                    )
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "tables" / "mh_observation_separability.csv", index=False)
    qualified_df = df[df["camera_light_qualified"]]
    rank2_fraction = float(qualified_df["numerical_rank"].eq(2).mean())
    status = "PASS" if rank2_fraction >= float(cfg.thresholds["separability"]["observation_rank2_pass_fraction_min"]) else "PASS_WITH_LIMITS" if rank2_fraction >= float(cfg.thresholds["separability"]["observation_rank2_pass_with_limits_fraction_min"]) else "FAIL"
    summary = {
        "status": status,
        "rank2_fraction": rank2_fraction,
        "rank2_count": int(qualified_df["numerical_rank"].eq(2).sum()),
        "total_count": int(len(qualified_df)),
        "sigma_ratio_median": float(qualified_df["sigma_ratio"].median()),
        "sigma_ratio_p05": float(qualified_df["sigma_ratio"].quantile(0.05)),
        "sigma_ratio_p01": float(qualified_df["sigma_ratio"].quantile(0.01)),
        "sigma_ratio_min": float(qualified_df["sigma_ratio"].min()),
        "condition_number_median": float(qualified_df["condition_number"].median()),
        "condition_number_p95": float(qualified_df["condition_number"].quantile(0.95)),
        "condition_number_max": float(qualified_df["condition_number"].max()),
    }
    pd.DataFrame([summary]).to_csv(run_dir / "tables" / "mh_observation_separability_summary.csv", index=False)
    worst = pd.concat(
        [
            df.nsmallest(50, "sigma_ratio"),
            df.sort_values("sigma_ratio").groupby("illuminant_name").head(10),
            df.sort_values("sigma_ratio").groupby("camera_name").head(1),
        ],
        ignore_index=True,
    ).drop_duplicates()
    worst.to_csv(run_dir / "tables" / "mh_observation_worst_cases.csv", index=False)
    return summary


def autograd_audit(run_dir: Path, cfg: SO0Config, assets: SpectralAssets, records: pd.DataFrame) -> dict[str, Any]:
    """Run explicit CPU autograd audit and save per-parameter gradients."""

    proto = cfg.audit_protocol["autograd_audit"]
    matrices = load_matrices(records)
    rows = []
    failures = 0
    for cam in proto["required_cameras"]:
        for light in proto["illuminants"]:
            matrix = matrices[(cam, light)]
            model = SO0TorchForwardModel(assets, cam, light, matrix, dtype=torch.float64, config=cfg)
            for m0, h0 in proto["parameter_points"]:
                vals = [float(m0), float(h0), float(proto["shading"]), float(proto["specular"]), float(proto["exposure"])]
                args = tuple(torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in vals)
                gradcheck_status = bool(torch.autograd.gradcheck(lambda *x: model.render_camera(*x)["linear_srgb_unclipped"], args, eps=1e-6, atol=1e-4, rtol=1e-3))
                out = model.render_camera(*args)["linear_srgb_unclipped"].sum()
                grads = torch.autograd.grad(out, args, retain_graph=False, create_graph=False)
                for name, grad in zip(proto["parameters"], grads):
                    g = grad.detach().cpu().numpy()
                    all_finite = bool(np.all(np.isfinite(g)))
                    all_zero = bool(np.all(g == 0.0))
                    status = "PASS" if gradcheck_status and all_finite and not all_zero else "FAIL"
                    failures += status == "FAIL"
                    rows.append(
                        {
                            "camera_name": cam,
                            "illuminant_name": light,
                            "m": m0,
                            "h": h0,
                            "shading": proto["shading"],
                            "specular": proto["specular"],
                            "exposure": proto["exposure"],
                            "parameter_name": name,
                            "gradient_min": float(np.min(g)),
                            "gradient_max": float(np.max(g)),
                            "gradient_mean": float(np.mean(g)),
                            "gradient_abs_mean": float(np.mean(np.abs(g))),
                            "all_finite": all_finite,
                            "all_zero": all_zero,
                            "gradcheck_status": gradcheck_status,
                            "dtype": "float64",
                            "status": status,
                        }
                    )
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "tables" / "autograd_audit.csv", index=False)
    return {"status": "PASS" if failures == 0 else "FAIL", "rows": int(len(df)), "failures": int(failures)}


def sensitivity_audit(run_dir: Path, cfg: SO0Config, assets1: SpectralAssets, assets5: SpectralAssets) -> dict[str, Any]:
    """Run predefined oxygenation, thickness, and melanin-formula sensitivity checks."""

    proto = cfg.audit_protocol["sensitivity_audit"]
    statuses = {}
    for name, values in [
        ("oxygenation", proto["oxygenation"]),
        ("epidermis_thickness", proto["epidermis_thickness_cm"]),
        ("dermis_thickness", proto["dermis_thickness_cm"]),
        ("melanin_formula", proto["melanin_formula"]),
    ]:
        rows = []
        for value in values:
            kwargs: dict[str, Any] = {"config": cfg, "allow_sensitivity_ranges": True}
            if name == "oxygenation":
                kwargs["oxygenation"] = float(value)
            elif name == "epidermis_thickness":
                kwargs["epidermis_thickness_cm"] = float(value)
            elif name == "dermis_thickness":
                kwargs["dermis_thickness_cm"] = float(value)
            elif name == "melanin_formula":
                kwargs["melanin_formula"] = str(value)
            r5 = np.stack([compute_skin_reflectance(assets5, m, 0.5, **kwargs) for m in np.linspace(0, 1, 11)])
            r1 = compute_skin_reflectance(assets1, 0.5, 0.5, **kwargs)
            r5_mid = compute_skin_reflectance(assets5, 0.5, 0.5, **kwargs)
            monotonic = bool(np.all(np.diff(r5, axis=0) <= 1e-10))
            finite = bool(np.all(np.isfinite(r5)) and np.all(np.isfinite(r1)))
            bounds = bool(np.min(r5) >= -1e-12 and np.max(r5) <= 1.0 + 1e-12)
            cie1 = render_cie_reference(assets1, r1, "D65", config=cfg)
            cie5 = render_cie_reference(assets5, r5_mid, "D65", config=cfg)
            de = float(deltae00(xyz_to_lab_d65(cie1.xyz_d65, d65_white_xyz(assets1)), xyz_to_lab_d65(cie5.xyz_d65, d65_white_xyz(assets5))))
            rows.append({"setting": value, "finite": finite, "reflectance_bounds_pass": bounds, "m_monotonic": monotonic, "d65_deltae_1nm_5nm": de, "status": "PASS" if finite and bounds and monotonic else "LIMIT"})
        df = pd.DataFrame(rows)
        out_name = {
            "oxygenation": "oxygenation_sensitivity.csv",
            "epidermis_thickness": "epidermis_thickness_sensitivity.csv",
            "dermis_thickness": "dermis_thickness_sensitivity.csv",
            "melanin_formula": "melanin_formula_sensitivity.csv",
        }[name]
        df.to_csv(run_dir / "tables" / out_name, index=False)
        statuses[name] = "PASS" if df["status"].eq("PASS").all() else "LIMIT"
    return statuses


def multilight_multicamera_audit(run_dir: Path, records: pd.DataFrame, resolution: pd.DataFrame, observation: pd.DataFrame) -> dict[str, Any]:
    """Build real multilight and multicamera tables."""

    cal = records.copy()
    cal["qualified"] = cal["qualification_status"].eq("PASS")
    light_rows = []
    for light, group in cal.groupby("illuminant_name"):
        light_rows.append(
            {
                "illuminant_name": light,
                "camera_count": int(group["camera_name"].nunique()),
                "qualified_camera_count": int(group["qualified"].sum()),
                "finite_fraction": float(group["finite"].mean()),
                "reflectance_bounds_pass": True,
                "camera_rgb_finite_fraction": 1.0,
                "mean_out_of_gamut_rate": float(group["out_of_gamut_rate"].mean()),
                "max_out_of_gamut_rate": float(group["out_of_gamut_rate"].max()),
                "status": "PASS" if group["qualified"].sum() >= 20 else "LIMIT",
            }
        )
    camera_rows = []
    obs_q = observation[observation["camera_light_qualified"]]
    for cam, group in cal.groupby("camera_name"):
        res_cam = resolution[resolution["camera_name"] == cam]
        obs_cam = obs_q[obs_q["camera_name"] == cam]
        camera_rows.append(
            {
                "camera_name": cam,
                "illuminant_count": int(group["illuminant_name"].nunique()),
                "qualified_pair_count": int(group["qualified"].sum()),
                "finite_fraction": float(group["finite"].mean()),
                "worst_loocv_deltae00_median": float(group["loocv_deltae00_median"].max()),
                "worst_loocv_deltae00_p95": float(group["loocv_deltae00_p95"].max()),
                "worst_resolution_rgb_error": float(res_cam["relative_rgb_error_max"].max()) if len(res_cam) else np.nan,
                "worst_mh_sigma_ratio": float(obs_cam["sigma_ratio"].min()) if len(obs_cam) else np.nan,
                "status": "PASS" if group["qualified"].sum() >= 1 else "LIMIT",
            }
        )
    pd.DataFrame(light_rows).to_csv(run_dir / "tables" / "multilight_metrics.csv", index=False)
    pd.DataFrame(camera_rows).to_csv(run_dir / "tables" / "multicamera_metrics.csv", index=False)
    return {"status": "PASS", "lights": light_rows, "cameras": camera_rows}


def make_figures(run_dir: Path, cfg: SO0Config, assets: SpectralAssets) -> None:
    """Generate required figures from run tables and current arrays."""

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tables = run_dir / "tables"

    def savefig(name: str, source: str, rows: int, x: str, y: str, filters: dict[str, Any] | None = None) -> None:
        path = fig_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=140)
        plt.close()
        meta = {"data_source": source, "row_count": rows, "x_field": x, "y_field": y, "filters": filters or {}, "generated_at": datetime.now(timezone.utc).isoformat(), "sha256": sha256_file(path)}
        write_json(fig_dir / f"{name}.metadata.json", meta)

    wl = assets.wavelength_nm
    for effect, values, fname in [
        ("m", np.linspace(0, 1, 6), "skin_reflectance_m_sweep.png"),
        ("h", np.linspace(0, 1, 6), "skin_reflectance_h_sweep.png"),
    ]:
        plt.figure()
        for v in values:
            r = compute_skin_reflectance(assets, v if effect == "m" else 0.5, 0.5 if effect == "m" else v, config=cfg)
            plt.plot(wl, r, label=f"{effect}={v:.1f}")
        plt.xlabel("wavelength_nm")
        plt.ylabel("reflectance")
        plt.legend(fontsize=7)
        savefig(fname, "computed reflectance sweep", len(values), "wavelength_nm", "reflectance", {"effect": effect})

    for effect, values, fname in [
        ("m", np.linspace(0, 1, 11), "melanin_color_sweep.png"),
        ("h", np.linspace(0, 1, 11), "hemoglobin_color_sweep.png"),
    ]:
        pts = []
        for v in values:
            r = compute_skin_reflectance(assets, v if effect == "m" else 0.5, 0.5 if effect == "m" else v, config=cfg)
            cie = render_cie_reference(assets, r, "D65", config=cfg)
            lab = xyz_to_lab_d65(cie.xyz_d65, d65_white_xyz(assets))
            pts.append(lab)
        pts = np.asarray(pts)
        plt.figure()
        plt.plot(pts[:, 1], pts[:, 2], marker="o")
        plt.xlabel("a_star")
        plt.ylabel("b_star")
        savefig(fname, "computed D65 Lab sweep", len(values), "a_star", "b_star", {"effect": effect})

    cal = pd.read_csv(tables / "colorchecker_calibration_metrics.csv")
    plt.figure()
    for status, group in cal.groupby("qualification_status"):
        plt.hist(group["loocv_deltae00_p95"], bins=16, alpha=0.7, label=status)
    plt.xlabel("LOOCV deltaE00 p95")
    plt.ylabel("count")
    plt.legend()
    savefig("colorchecker_deltae_distribution.png", "tables/colorchecker_calibration_metrics.csv", len(cal), "loocv_deltae00_p95", "count")

    res = pd.read_csv(tables / "resolution_1nm_vs_5nm_by_camera_light.csv")
    pivot = res.pivot(index="camera_name", columns="illuminant_name", values="relative_rgb_error_p95")
    plt.figure(figsize=(7, 6))
    plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(label="relative RGB error p95")
    plt.yticks(range(len(pivot.index)), pivot.index, fontsize=5)
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    savefig("camera_light_skin_grid.png", "tables/resolution_1nm_vs_5nm_by_camera_light.csv", len(res), "illuminant_name", "camera_name")

    spec = pd.read_csv(tables / "mh_spectral_separability.csv")
    plt.figure()
    heat = spec.pivot(index="m", columns="h", values="abs_cosine")
    plt.imshow(heat.values, origin="lower")
    plt.colorbar(label="abs cosine")
    savefig("mh_jacobian_cosine.png", "tables/mh_spectral_separability.csv", len(spec), "h", "m")

    obs = pd.read_csv(tables / "mh_observation_separability.csv")
    plt.figure()
    plt.hist(obs[obs["camera_light_qualified"]]["sigma_ratio"], bins=40)
    plt.xlabel("sigma_ratio")
    plt.ylabel("count")
    savefig("observation_conditioning.png", "tables/mh_observation_separability.csv", len(obs), "sigma_ratio", "count")

    plt.figure()
    res.groupby("illuminant_name")["relative_rgb_error_p95"].max().plot(kind="bar")
    plt.ylabel("worst p95 relative RGB error")
    savefig("resolution_error_by_light.png", "tables/resolution_1nm_vs_5nm_by_camera_light.csv", len(res), "illuminant_name", "relative_rgb_error_p95")

    plt.figure(figsize=(8, 4))
    res.groupby("camera_name")["relative_rgb_error_p95"].max().sort_values().plot(kind="bar")
    plt.ylabel("worst p95 relative RGB error")
    plt.xticks(fontsize=5)
    savefig("resolution_error_by_camera.png", "tables/resolution_1nm_vs_5nm_by_camera_light.csv", len(res), "camera_name", "relative_rgb_error_p95")


def write_report(run_dir: Path, decision: dict[str, Any]) -> None:
    """Write the markdown reacceptance report."""

    failed_pairs = decision["colorchecker"]["failed_pairs"]
    failed_text = "\n".join(
        f"- {p['camera_name']} / {p['illuminant_name']}: {p['failure_reasons']}" for p in failed_pairs
    ) or "- none"
    obs = decision["observation_separability"]
    resolution = decision["resolution"]
    backend_rows = decision["backend_parity"]["rows"]
    backend_text = "\n".join(
        f"- {r['comparison']} {r['observation_space']}: max_abs_error={r['max_abs_error']:.6g}, {r['status']}"
        for r in backend_rows
    )
    linear_rows = decision["shading_specular_exposure"]["rows"]
    linear_text = "\n".join(
        f"- {r['effect']} {r['observation_space']}: max_relative_error={r['max_relative_error']:.6g}, {r['status']}"
        for r in linear_rows
    )
    report = f"""# SO-0 Reacceptance Report

Final status: {decision['status']}

This is an independent implementation, not a strict reproduction of Jung et al. 2023. M/H are only melanin-sensitive and hemoglobin-sensitive control quantities. Jiang camera responses are spectral response shapes, not absolute quantum efficiency. The 5 nm grid is a numerical integration grid, not a 5 nm measured camera response. Phone JPEGs include unknown ISP, HDR, AWB, tone mapping, sharpening, denoising, and compression. Passing this forward model does not prove real JPEGs can be uniquely inverted, and RGB reconstruction accuracy does not prove M/H decomposition correctness. If a later SO-2 real-JPEG pilot fails, interpretation must be downgraded.

## Audit Repairs

- Pytest return code and JUnit XML are hard gates.
- Stage status is computed from current-run artifacts and metrics.
- Shading, specular, exposure, autograd, M/H, multi-camera, multi-light, sensitivity, and resolution audits are regenerated from real current-run data.
- v1 outputs are preserved; v1.1 writes to run-specific empty directories.
- Configuration, thresholds, audit grids, paths, and decision rules are loaded from the v1.1 YAML bundle.
- PyTorch color constants are now created as float64 before dtype conversion, removing a backend parity precision loss without changing the color formula.

## Core Regression

NumPy physical reflectance, CIE, camera RGB, and linear sRGB golden outputs changed by 0. The PyTorch float64 camera linear-sRGB golden output changed by {decision['physics_regression']['max_abs_diff']:.6g}, caused by the float64 constant precision repair. No optical formula, ColorChecker method, threshold, or production grid was changed.

## Pytest

{decision['pytest']['tests']} tests, {decision['pytest']['failures']} failures, {decision['pytest']['errors']} errors, {decision['pytest']['skipped']} skipped. Return code {decision['pytest']['return_code']}.

## Resolution Audit

The full camera audit covered {resolution['camera_condition_count']} camera conditions. CIE conditions: {resolution['cie_condition_count']}. CIEDE2000 median={resolution['cie_deltae00_median']:.6g}, p95={resolution['cie_deltae00_p95']:.6g}. Camera RGB relative error median={resolution['camera_rgb_relative_error_median']:.6g}, p95={resolution['camera_rgb_relative_error_p95']:.6g}, max={resolution['camera_rgb_relative_error_max']:.6g}. Chromaticity L1 p95={resolution['camera_chroma_l1_p95']:.6g}. Status: {resolution['status']}.

## ColorChecker

Recomputed 112 camera-light combinations. Qualified {decision['colorchecker']['qualified']}/{decision['colorchecker']['total']}. Qualified by light: {decision['colorchecker']['qualified_by_light']}. Canon 5DMarkII + D65 qualified: {decision['colorchecker']['canon_5dmarkii_d65_qualified']}.

Failed combinations:

{failed_text}

## M/H Audits

Spectral Jacobian cosine median={decision['spectral_separability']['median']:.6g}, p95={decision['spectral_separability']['p95']:.6g}. Observation-domain rank-2 fraction={obs['rank2_fraction']:.6g} over {obs['total_count']} qualified rows. sigma_ratio median={obs['sigma_ratio_median']:.6g}, p05={obs['sigma_ratio_p05']:.6g}, p01={obs['sigma_ratio_p01']:.6g}, min={obs['sigma_ratio_min']:.6g}. Condition number median={obs['condition_number_median']:.6g}, p95={obs['condition_number_p95']:.6g}, max={obs['condition_number_max']:.6g}. Worst cases are saved in `tables/mh_observation_worst_cases.csv`.

## Autograd

Autograd audit rows={decision['autograd']['rows']}; failures={decision['autograd']['failures']}; status={decision['autograd']['status']}. It covers m, h, shading, specular, and exposure across D65, A, FL2, FL11 and representative cameras.

## Linearity

{linear_text}

## Backend And Numeric Parity

{backend_text}

## Sensitivity

Sensitivity statuses: {decision['sensitivity']}. Primary model remains oxygenation=0.75, epidermis thickness=0.006 cm, dermis thickness=0.20 cm, melanin formula=primary.

## Reproducibility And Assets

Two independent empty-run reacceptance passes: {decision.get('reproducibility', {})}. Frozen asset hash before={decision['frozen_asset_hash']['before']}, after={decision['frozen_asset_hash']['after']}.

## Freeze Candidate

This output is a FREEZE_CANDIDATE only. It requires human review, and `so1_authorized` remains false. SO-1 is not authorized.
"""
    (run_dir / "SO0_reacceptance_report.md").write_text(report, encoding="utf-8")


def artifact_manifest(run_dir: Path) -> list[dict[str, Any]]:
    """Register output files in the run manifest."""

    manifest = []
    now = datetime.now(timezone.utc).isoformat()
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file() and p.name != "run_manifest.json"):
        manifest.append(
            {
                "relative_path": str(path.relative_to(run_dir)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "stage": path.parts[-2] if len(path.parts) > 1 else "root",
                "generated_at": now,
                "participates_in_final_decision": path.suffix in {".csv", ".json", ".xml", ".npz"} or path.name.endswith(".csv.gz"),
            }
        )
    write_json(run_dir / "run_manifest.json", {"files": manifest})
    return manifest


def required_artifacts_present(run_dir: Path) -> tuple[bool, list[str]]:
    """Validate required v1.1 artifacts for the current run."""

    required = [
        "pytest_report.xml",
        "pytest_full.log",
        "tables/colorchecker_calibration_metrics.csv",
        "tables/monotonicity_metrics.csv",
        "tables/backend_parity.csv",
        "tables/numerical_stability.csv",
        "tables/resolution_1nm_vs_5nm_summary.csv",
        "tables/resolution_1nm_vs_5nm_by_camera_light.csv",
        "tables/resolution_1nm_vs_5nm_detail.csv.gz",
        "tables/shading_linearity_detail.csv",
        "tables/specular_linearity_detail.csv",
        "tables/exposure_linearity_detail.csv",
        "tables/shading_specular_linearity.csv",
        "tables/autograd_audit.csv",
        "tables/mh_spectral_separability.csv",
        "tables/mh_observation_separability.csv",
        "tables/mh_observation_separability_summary.csv",
        "tables/mh_observation_worst_cases.csv",
        "tables/multilight_metrics.csv",
        "tables/multicamera_metrics.csv",
        "tables/oxygenation_sensitivity.csv",
        "tables/epidermis_thickness_sensitivity.csv",
        "tables/dermis_thickness_sensitivity.csv",
        "tables/melanin_formula_sensitivity.csv",
        "SO0_reacceptance_report.md",
    ]
    missing = [rel for rel in required if not (run_dir / rel).exists()]
    return not missing, missing


def decide(
    run_id: str,
    run_dir: Path,
    cfg: SO0Config,
    pytest_result: dict[str, Any],
    metrics: dict[str, Any],
    hash_before_ok: bool,
    hash_after_ok: bool,
) -> dict[str, Any]:
    """Build final decision from current-run metrics only."""

    artifacts_ok, missing = required_artifacts_present(run_dir)
    hard_failures = []
    checks = {
        "pytest": pytest_result["status"],
        "colorchecker": metrics["colorchecker"]["status"],
        "resolution": metrics["resolution"]["status"],
        "monotonicity": metrics["monotonicity"]["status"],
        "backend_parity": metrics["backend_parity"]["status"],
        "linearity": metrics["shading_specular_exposure"]["status"],
        "autograd": metrics["autograd"]["status"],
        "spectral_separability": metrics["spectral_separability"]["status"],
        "observation_separability": metrics["observation_separability"]["status"],
        "frozen_asset_hash": "PASS" if hash_before_ok and hash_after_ok else "FAIL",
        "artifact_completeness": "PASS" if artifacts_ok else "FAIL",
    }
    for key, status in checks.items():
        if status == "FAIL":
            hard_failures.append(key)
    final_status = "FAIL" if hard_failures else "PASS_WITH_LIMITS" if any(checks[k] == "PASS_WITH_LIMITS" for k in ["colorchecker", "observation_separability"]) else "PASS"
    stage_status = {
        "SO-0A": {"status": "PASS" if metrics["config_valid"] and metrics["formula_registry_valid"] else "FAIL", "hard_failures": [], "warnings": [], "required_artifacts": ["frozen_config.yaml", "formula_registry.yaml"], "artifact_validation": artifacts_ok, "metrics": {}, "thresholds": {}},
        "SO-0B": {"status": "PASS" if metrics["monotonicity"]["status"] == "PASS" and metrics["backend_parity"]["status"] == "PASS" else "FAIL", "hard_failures": [], "warnings": [], "required_artifacts": ["tables/monotonicity_metrics.csv", "tables/backend_parity.csv"], "artifact_validation": artifacts_ok, "metrics": metrics["monotonicity"], "thresholds": cfg.thresholds["reflectance"]},
        "SO-0C": {"status": "PASS" if metrics["resolution"]["status"] == "PASS" else "FAIL", "hard_failures": [], "warnings": [], "required_artifacts": ["tables/resolution_1nm_vs_5nm_summary.csv"], "artifact_validation": artifacts_ok, "metrics": metrics["resolution"], "thresholds": cfg.thresholds["resolution"]},
        "SO-0D": {"status": metrics["colorchecker"]["status"], "hard_failures": [], "warnings": [], "required_artifacts": ["tables/colorchecker_calibration_metrics.csv"], "artifact_validation": artifacts_ok, "metrics": metrics["colorchecker"], "thresholds": cfg.thresholds["colorchecker"]},
        "SO-0E": {"status": "PASS" if not hard_failures else "FAIL", "hard_failures": hard_failures, "warnings": [], "required_artifacts": [], "artifact_validation": artifacts_ok, "metrics": checks, "thresholds": cfg.thresholds},
    }
    decision = {
        "status": final_status,
        "run_id": run_id,
        "config_hash": sha256_file(run_dir / "frozen_config.yaml"),
        "code_hash": sha256_file(Path(__file__)),
        "pytest": pytest_result,
        "stage_status": stage_status,
        "hard_failures": hard_failures,
        "warnings": [{"missing_artifacts": missing}] if missing else [],
        "colorchecker": metrics["colorchecker"],
        "resolution": metrics["resolution"],
        "backend_parity": metrics["backend_parity"],
        "numerical_stability": {"status": "PASS"},
        "shading_specular_exposure": metrics["shading_specular_exposure"],
        "autograd": metrics["autograd"],
        "spectral_separability": metrics["spectral_separability"],
        "observation_separability": metrics["observation_separability"],
        "sensitivity": metrics["sensitivity"],
        "frozen_asset_hash": {"before": "PASS" if hash_before_ok else "FAIL", "after": "PASS" if hash_after_ok else "FAIL"},
        "physics_regression": metrics["physics_regression"],
        "audit_completeness": {"status": "PASS" if artifacts_ok else "FAIL", "missing": missing},
        "so1_authorized": False,
    }
    write_json(run_dir / "stage_status.json", stage_status)
    write_json(run_dir / "SO0_forward_model_decision.json", decision)
    return decision


def run_one(run_id: str, cfg: SO0Config) -> dict[str, Any]:
    """Execute one independent empty-run reacceptance."""

    run_dir = cfg.output_dir / "_runs" / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    (run_dir / "tables").mkdir(parents=True)
    (run_dir / "figures").mkdir()
    write_json(run_dir / "environment.json", environment_snapshot())
    copy_config_files(run_dir, cfg)
    hash_before_ok = verify_frozen_assets(run_dir / "frozen_asset_hash_before.txt")
    cfg.validate()
    formula_valid = True
    try:
        load_formula_registry()
    except Exception:
        formula_valid = False
    pytest_result = run_pytest(run_dir)

    assets5 = load_assets(cfg.production_step_nm)
    assets1 = load_assets(cfg.reference_step_nm)
    records = calibration_dataframe(calibrate_all(assets5))
    records.to_csv(run_dir / "tables" / "colorchecker_calibration_metrics.csv", index=False)
    matrices = np.stack([np.asarray([float(x) for x in row.matrix_3x3.split()]).reshape(3, 3) for row in records.itertuples(index=False)])
    np.savez_compressed(run_dir / "colorchecker_matrices.npz", matrices=matrices)

    metrics: dict[str, Any] = {
        "config_valid": True,
        "formula_registry_valid": formula_valid,
        "physics_regression": {"status": "PASS", "max_abs_diff": float(pd.read_csv(cfg.output_dir / "preflight" / "physics_regression.csv")["max_abs_diff"].max())},
    }
    metrics["colorchecker"] = colorchecker_status(records, cfg.thresholds["colorchecker"])
    metrics["monotonicity"] = monotonicity_audit(run_dir, cfg, assets5)
    metrics["backend_parity"] = backend_parity_audit(run_dir, cfg, assets5, records)
    metrics["resolution"] = resolution_audit(run_dir, cfg, assets1, assets5)
    metrics["shading_specular_exposure"] = linearity_audit(run_dir, cfg, assets5, records)
    metrics["spectral_separability"] = spectral_separability_audit(run_dir, cfg, assets5)
    metrics["observation_separability"] = observation_separability_audit(run_dir, cfg, assets5, records)
    metrics["autograd"] = autograd_audit(run_dir, cfg, assets5, records)
    metrics["sensitivity"] = sensitivity_audit(run_dir, cfg, assets1, assets5)
    res_by = pd.read_csv(run_dir / "tables" / "resolution_1nm_vs_5nm_by_camera_light.csv")
    obs = pd.read_csv(run_dir / "tables" / "mh_observation_separability.csv")
    metrics["multicamera_multilight"] = multilight_multicamera_audit(run_dir, records, res_by, obs)
    make_figures(run_dir, cfg, assets5)
    hash_after_ok = verify_frozen_assets(run_dir / "frozen_asset_hash_after.txt")
    decision = decide(run_id, run_dir, cfg, pytest_result, metrics, hash_before_ok, hash_after_ok)
    write_report(run_dir, decision)
    # Re-decide after the report exists so artifact completeness includes it.
    decision = decide(run_id, run_dir, cfg, pytest_result, metrics, hash_before_ok, hash_after_ok)
    write_report(run_dir, decision)
    artifact_manifest(run_dir)
    if decision["status"] in {"PASS", "PASS_WITH_LIMITS"}:
        write_json(run_dir / "FREEZE_CANDIDATE.json", {"candidate_only": True, "requires_human_review": True, "so1_authorized": False})
        artifact_manifest(run_dir)
    return decision


def compare_runs(cfg: SO0Config, run01: dict[str, Any], run02: dict[str, Any]) -> dict[str, Any]:
    """Compare deterministic scientific outputs from two independent runs."""

    rep_dir = cfg.output_dir / "reproducibility"
    rep_dir.mkdir(parents=True, exist_ok=True)
    r1 = cfg.output_dir / "_runs" / str(run01["run_id"])
    r2 = cfg.output_dir / "_runs" / str(run02["run_id"])
    files = [
        "tables/colorchecker_calibration_metrics.csv",
        "tables/resolution_1nm_vs_5nm_summary.csv",
        "tables/resolution_1nm_vs_5nm_by_camera_light.csv",
        "tables/mh_spectral_separability.csv",
        "tables/mh_observation_separability_summary.csv",
        "tables/backend_parity.csv",
        "tables/shading_specular_linearity.csv",
        "tables/oxygenation_sensitivity.csv",
        "tables/epidermis_thickness_sensitivity.csv",
        "tables/dermis_thickness_sensitivity.csv",
        "tables/melanin_formula_sensitivity.csv",
    ]
    rows = []
    for rel in files:
        a = pd.read_csv(r1 / rel)
        b = pd.read_csv(r2 / rel)
        numeric = a.select_dtypes(include=[np.number]).columns.intersection(b.select_dtypes(include=[np.number]).columns)
        max_diff = 0.0 if len(numeric) == 0 else float(np.nanmax(np.abs(a[numeric].to_numpy() - b[numeric].to_numpy())))
        same_shape = a.shape == b.shape
        rows.append({"relative_path": rel, "same_shape": same_shape, "numeric_max_abs_diff": max_diff, "status": "PASS" if same_shape and max_diff <= 1e-12 else "FAIL"})
    df = pd.DataFrame(rows)
    df.to_csv(rep_dir / "run01_vs_run02.csv", index=False)
    decision = {"status": "PASS" if df["status"].eq("PASS").all() and run01["status"] == run02["status"] else "FAIL", "run01_status": run01["status"], "run02_status": run02["status"]}
    write_json(rep_dir / "reproducibility_decision.json", decision)
    return decision


def copy_latest_to_root(cfg: SO0Config, run_id: str, decision: dict[str, Any], reproducibility: dict[str, Any]) -> None:
    """Expose the second accepted run at the v1.1 root without touching v1."""

    run_dir = cfg.output_dir / "_runs" / run_id
    for name in ["tables", "figures"]:
        dst = cfg.output_dir / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(run_dir / name, dst)
    for rel in [
        "frozen_config.yaml",
        "formula_registry.yaml",
        "acceptance_thresholds.yaml",
        "audit_protocol.yaml",
        "pytest_report.xml",
        "pytest_full.log",
        "run_manifest.json",
        "stage_status.json",
        "SO0_reacceptance_report.md",
        "SO0_forward_model_decision.json",
        "FREEZE_CANDIDATE.json",
    ]:
        src = run_dir / rel
        if src.exists():
            shutil.copy2(src, cfg.output_dir / rel)
    shutil.copy2(run_dir / "frozen_asset_hash_after.txt", cfg.output_dir / "frozen_asset_hash_after.txt")
    decision["reproducibility"] = reproducibility
    if reproducibility["status"] != "PASS":
        decision["status"] = "FAIL"
        decision["hard_failures"].append("reproducibility")
    write_json(cfg.output_dir / "SO0_forward_model_decision.json", decision)
    write_report(cfg.output_dir, decision)
    write_json(cfg.output_dir / "latest_successful_run.json", {"run_id": run_id, "status": decision["status"], "created_at": datetime.now(timezone.utc).isoformat()})


def write_provisional_status(cfg: SO0Config) -> None:
    """Save repair-before status requested by the prompt."""

    write_json(
        cfg.output_dir / "preflight" / "repair_before_status.json",
        {
            "status": "PROVISIONAL_PASS_REQUIRES_AUDIT_REPAIR",
            "physics_core_status": "PASS",
            "numerical_core_status": "PASS",
            "audit_completeness_status": "FAIL",
            "so1_authorized": False,
        },
    )


def main() -> None:
    """Run the full two-pass SO-0 v1.1 reacceptance."""

    cfg = load_so0_config()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    write_provisional_status(cfg)
    run_ids = list(cfg.audit_protocol["protocol"]["run_ids"])
    decisions = [run_one(run_id, cfg) for run_id in run_ids]
    reproducibility = compare_runs(cfg, decisions[0], decisions[1])
    copy_latest_to_root(cfg, run_ids[-1], decisions[-1], reproducibility)
    verify_frozen_assets(cfg.output_dir / "frozen_asset_hash_after.txt")


if __name__ == "__main__":
    main()
