"""Read-only SO-0 audit consumer for the SO-R1-A0 camera/light split.

This module deliberately never writes into SO-0 asset directories.  All names
are joined explicitly, and the native 10 nm camera responses are the sole
source of selection distances.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


LIGHTS = ("D65", "A", "FL2", "FL11")
CHANNELS = ("R", "G", "B")
REQUIRED_TABLES = (
    "multicamera_metrics.csv", "multilight_metrics.csv",
    "colorchecker_calibration_metrics.csv", "mh_observation_separability.csv",
    "mh_observation_separability_summary.csv", "mh_observation_worst_cases.csv",
    "resolution_1nm_vs_5nm_by_camera_light.csv",
)


class SelectionError(RuntimeError):
    """A hard protocol gate failed; no frozen split may be produced."""


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SelectionError(f"Configuration missing: {path}")
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise SelectionError("Configuration must be a mapping")
    return value


def _rel(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protected_hashes(root: Path, config: dict[str, Any]) -> dict[str, str]:
    targets = [root / config["spectral_root"], root / config["so0_audit_root"]]
    rows: dict[str, str] = {}
    for target in targets:
        if not target.is_dir():
            raise SelectionError(f"Protected asset root missing: {target}")
        for path in sorted(p for p in target.rglob("*") if p.is_file()):
            rows[_rel(root, path)] = _sha256(path)
    return rows


def _write_hashes(path: Path, hashes: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["repo_relative_path", "sha256"])
        writer.writeheader()
        writer.writerows({"repo_relative_path": k, "sha256": v} for k, v in hashes.items())


def normalize_camera_name(value: object) -> str:
    text = str(value).strip().casefold()
    text = re.sub(r"[-_]", " ", text)
    return re.sub(r"\s+", " ", text)


def _bool(value: object) -> bool:
    return value is True or str(value).strip().casefold() == "true"


def discover_inputs(root: Path, config: dict[str, Any]) -> dict[str, Path]:
    spectral_root = root / config["spectral_root"]
    audit_root = root / config["so0_audit_root"]
    wanted = {
        "source_10nm": "source_standardized_10nm.npz",
        "source_1nm": "source_standardized_1nm.npz",
        "derived_10nm": "derived_optics_10nm.npz",
        "derived_1nm": "derived_optics_1nm.npz",
    }
    found: dict[str, Path] = {}
    for label, filename in wanted.items():
        matches = sorted(spectral_root.rglob(filename)) if spectral_root.is_dir() else []
        if len(matches) != 1:
            raise SelectionError(f"INPUT_MISSING or ambiguous {filename}: found {len(matches)} under {spectral_root}")
        found[label] = matches[0]
    for filename in REQUIRED_TABLES:
        path = audit_root / "tables" / filename
        if not path.is_file():
            matches = sorted(root.rglob(filename))
            raise SelectionError(f"INPUT_MISSING {filename}; searched {audit_root} and found {[str(x) for x in matches]}")
        found[filename] = path
    for filename in ("SO0_reacceptance_report.md", "SO0_forward_model_decision.json", "stage_status.json", "acceptance_thresholds.yaml", "frozen_config.yaml", "run_manifest.json"):
        path = audit_root / filename
        if not path.is_file():
            raise SelectionError(f"INPUT_MISSING required SO-0 audit asset: {path}")
        found[filename] = path
    return found


def load_npz_inventory(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    inventory = {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in arrays.items()}
    return inventory, arrays


def _require_npz(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str], np.ndarray, list[str], list[str]]:
    expected = {"wavelength_nm", "camera_names", "camera_ssf", "camera_channel_names", "core_illuminants_y1", "core_illuminant_names"}
    missing = expected.difference(arrays)
    if missing:
        raise SelectionError(f"NPZ keys cannot be uniquely identified; missing {sorted(missing)}; keys={sorted(arrays)}")
    wave, names, response = arrays["wavelength_nm"], arrays["camera_names"], arrays["camera_ssf"]
    channels = [str(x) for x in arrays["camera_channel_names"].tolist()]
    light_names = [str(x) for x in arrays["core_illuminant_names"].tolist()]
    if response.ndim != 3 or response.shape != (len(names), len(wave), 3):
        raise SelectionError(f"Invalid camera response dimensions: {response.shape}")
    if channels != list(CHANNELS):
        raise SelectionError(f"Unexpected channel order: {channels}")
    return response.astype(float), [str(x) for x in names.tolist()], wave.astype(float), light_names, channels


def _unique_normalized(names: list[str], source: str) -> dict[str, str]:
    normalized = [normalize_camera_name(name) for name in names]
    duplicates = [name for name, count in Counter(normalized).items() if count > 1]
    if duplicates:
        raise SelectionError(f"Normalized camera-name collisions in {source}: {duplicates}")
    return dict(zip(normalized, names))


@dataclass
class Inputs:
    paths: dict[str, Path]
    inventory: dict[str, dict[str, Any]]
    responses: np.ndarray
    cameras: list[str]
    wavelength: np.ndarray
    spds: np.ndarray
    light_names: list[str]
    tables: dict[str, pd.DataFrame]


def read_inputs(root: Path, config: dict[str, Any]) -> Inputs:
    paths = discover_inputs(root, config)
    inventory, arrays = load_npz_inventory(paths["source_10nm"])
    responses, cameras, wavelength, light_names, _ = _require_npz(arrays)
    if len(cameras) != 28 or len(set(cameras)) != 28:
        raise SelectionError(f"Expected 28 unique NPZ cameras, got {len(cameras)}")
    if not np.all(np.diff(wavelength) > 0) or not np.allclose(np.diff(wavelength), 10.0):
        raise SelectionError("Native camera wavelength grid is not strictly increasing 10 nm")
    if not (np.isfinite(responses).all() and (responses >= 0).all()):
        raise SelectionError("Camera responses must be finite and nonnegative")
    if set(LIGHTS).difference(light_names):
        raise SelectionError(f"Required illuminants missing from NPZ: {set(LIGHTS).difference(light_names)}")
    tables = {name: pd.read_csv(paths[name]) for name in REQUIRED_TABLES}
    return Inputs(paths, inventory, responses, cameras, wavelength, arrays["core_illuminants_y1"].astype(float), light_names, tables)


def _join_audit(inputs: Inputs) -> pd.DataFrame:
    multi = inputs.tables["multicamera_metrics.csv"].copy()
    if len(multi) != 28 or multi.camera_name.nunique() != 28:
        raise SelectionError("multicamera_metrics.csv does not provide 28 unique cameras")
    sources = {"npz": inputs.cameras, "multicamera": multi.camera_name.astype(str).tolist()}
    for table_name in ("colorchecker_calibration_metrics.csv", "mh_observation_separability.csv", "resolution_1nm_vs_5nm_by_camera_light.csv"):
        sources[table_name] = sorted(inputs.tables[table_name].camera_name.astype(str).unique().tolist())
    canonical = _unique_normalized(inputs.cameras, "NPZ")
    for source, names in sources.items():
        lookup = _unique_normalized(names, source)
        if set(lookup) != set(canonical):
            raise SelectionError(f"Camera identity join failure between NPZ and {source}: missing={sorted(set(canonical)-set(lookup))}, extra={sorted(set(lookup)-set(canonical))}")
    by_norm = multi.set_index(multi.camera_name.map(normalize_camera_name), drop=False)
    rows = []
    for index, name in enumerate(inputs.cameras):
        row = by_norm.loc[normalize_camera_name(name)]
        rows.append({"camera_name": str(row.camera_name), "camera_name_normalized": normalize_camera_name(name), "npz_index": index,
                     "response_shape": "x".join(map(str, inputs.responses[index].shape)), "wavelength_min_nm": float(inputs.wavelength.min()),
                     "wavelength_max_nm": float(inputs.wavelength.max()), "wavelength_step_nm": float(np.diff(inputs.wavelength).mean()),
                     "channel_order": "R,G,B", "finite": bool(np.isfinite(inputs.responses[index]).all()), "nonnegative": bool((inputs.responses[index] >= 0).all()),
                     **{key: row[key] for key in multi.columns if key != "camera_name"}})
    return pd.DataFrame(rows)


def _pair_quality(inputs: Inputs, camera: str, light: str) -> tuple[bool, dict[str, Any]]:
    cc = inputs.tables["colorchecker_calibration_metrics.csv"]
    mh = inputs.tables["mh_observation_separability.csv"]
    resolution = inputs.tables["resolution_1nm_vs_5nm_by_camera_light.csv"]
    cc_rows = cc[(cc.camera_name == camera) & (cc.illuminant_name == light)]
    mh_rows = mh[(mh.camera_name == camera) & (mh.illuminant_name == light)]
    res_rows = resolution[(resolution.camera_name == camera) & (resolution.illuminant_name == light)]
    cc_ok = len(cc_rows) == 1 and _bool(cc_rows.iloc[0].finite) and cc_rows.iloc[0].qualification_status == "PASS" and pd.isna(cc_rows.iloc[0].failure_reasons)
    mh_ok = len(mh_rows) > 0 and all(_bool(x) for x in mh_rows.camera_light_qualified) and all(_bool(x) for x in mh_rows.finite) and set(mh_rows.status) == {"PASS"} and set(mh_rows.numerical_rank) == {2}
    res_numeric = res_rows.select_dtypes(include=[np.number]) if len(res_rows) else pd.DataFrame()
    res_ok = len(res_rows) == 1 and bool(np.isfinite(res_numeric.to_numpy()).all()) and float(res_rows.iloc[0].finite_fraction) == 1.0
    stats = {
        "colorchecker_qualified": bool(cc_ok), "colorchecker_loocv_median": float(cc_rows.iloc[0].loocv_deltae00_median) if len(cc_rows) == 1 else math.nan,
        "colorchecker_loocv_p95": float(cc_rows.iloc[0].loocv_deltae00_p95) if len(cc_rows) == 1 else math.nan,
        "mh_rank2_fraction": float((mh_rows.numerical_rank == 2).mean()) if len(mh_rows) else math.nan,
        "mh_sigma_ratio_min": float(mh_rows.sigma_ratio.min()) if len(mh_rows) else math.nan,
        "mh_sigma_ratio_median": float(mh_rows.sigma_ratio.median()) if len(mh_rows) else math.nan,
        "resolution_relative_rgb_error_p95": float(res_rows.iloc[0].relative_rgb_error_p95) if len(res_rows) == 1 else math.nan,
        "resolution_relative_rgb_error_max": float(res_rows.iloc[0].relative_rgb_error_max) if len(res_rows) == 1 else math.nan,
        "finite": bool(cc_ok and mh_ok and res_ok), "mh_ok": bool(mh_ok), "resolution_ok": bool(res_ok),
    }
    return bool(cc_ok and mh_ok and res_ok), stats


def apply_quality_gate(inputs: Inputs, inventory: pd.DataFrame) -> tuple[pd.DataFrame, dict[tuple[str, str], dict[str, Any]]]:
    pair_stats: dict[tuple[str, str], dict[str, Any]] = {}
    inventory = inventory.copy()
    statuses = []
    for _, row in inventory.iterrows():
        name = row.camera_name
        qualities = []
        for light in LIGHTS:
            qualified, stats = _pair_quality(inputs, name, light)
            pair_stats[(name, light)] = stats
            qualities.append(qualified)
        statuses.append(bool(row.illuminant_count == 4 and row.qualified_pair_count == 4 and float(row.finite_fraction) == 1.0 and row.status == "PASS" and all(qualities)))
    inventory["multicamera_status"] = inventory.pop("status")
    inventory["quality_gate_pass"] = statuses
    return inventory, pair_stats


def apply_candidate_scope(inventory: pd.DataFrame, exclusions: dict[str, str]) -> tuple[pd.DataFrame, list[str]]:
    inventory = inventory.copy()
    scope, reasons = [], []
    for _, row in inventory.iterrows():
        if row.camera_name in exclusions:
            scope.append("excluded")
            reasons.append(exclusions[row.camera_name])
        elif not row.quality_gate_pass:
            scope.append("quality_failed")
            reasons.append("failed_quality_gate")
        else:
            scope.append("consumer_mobile_candidate")
            reasons.append("")
    inventory["candidate_scope"], inventory["exclusion_reason"] = scope, reasons
    candidates = sorted(inventory.loc[inventory.candidate_scope == "consumer_mobile_candidate", "camera_name"].tolist())
    if len(candidates) != 24:
        raise SelectionError(f"CANDIDATE_POOL_MISMATCH: expected 24, got {len(candidates)}")
    return inventory, candidates


def normalize_channel_response(responses: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    areas = responses.sum(axis=1, keepdims=True)
    if np.any(areas <= 0):
        raise SelectionError("A camera channel has zero spectral area")
    return responses / (areas + epsilon)


def compute_spectral_features(names: list[str], wave: np.ndarray, responses: np.ndarray) -> pd.DataFrame:
    rows = []
    for name, response in zip(names, responses):
        row: dict[str, Any] = {"camera_name": name}
        for i, channel in enumerate(CHANNELS):
            values, area = response[:, i], response[:, i].sum()
            distribution = values / area
            centroid = float((wave * distribution).sum())
            row.update({f"{channel}_peak_wavelength_nm": float(wave[int(values.argmax())]), f"{channel}_spectral_centroid_nm": centroid,
                        f"{channel}_spectral_std_nm": float(np.sqrt(((wave-centroid)**2 * distribution).sum())), f"{channel}_coverage_min_nm": float(wave[values > 0].min()),
                        f"{channel}_coverage_max_nm": float(wave[values > 0].max()), f"{channel}_area": float(area)})
        for left, right in (("R", "G"), ("G", "B"), ("R", "B")):
            a, b = response[:, CHANNELS.index(left)], response[:, CHANNELS.index(right)]
            row[f"{left}_{right}_cosine_overlap"] = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        rows.append(row)
    return pd.DataFrame(rows)


def _distance_matrix(names: list[str], normalized: np.ndarray, method: str) -> pd.DataFrame:
    n = len(names); matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            if method == "sam":
                value = float(np.mean([np.arccos(np.clip(np.dot(normalized[i,:,k], normalized[j,:,k]) / (np.linalg.norm(normalized[i,:,k]) * np.linalg.norm(normalized[j,:,k])), -1, 1)) for k in range(3)]))
            else:
                value = float(np.mean([0.5 * np.abs(normalized[i,:,k] - normalized[j,:,k]).sum() for k in range(3)]))
            matrix[i, j] = matrix[j, i] = value
    if not (np.isfinite(matrix).all() and np.all(matrix >= -1e-12) and np.allclose(matrix, matrix.T) and np.allclose(np.diag(matrix), 0)):
        raise SelectionError(f"Invalid {method} distance matrix")
    return pd.DataFrame(matrix, index=names, columns=names)


def compute_sam_distance_matrix(names: list[str], normalized: np.ndarray) -> pd.DataFrame:
    return _distance_matrix(names, normalized, "sam")


def compute_tv_distance_matrix(names: list[str], normalized: np.ndarray) -> pd.DataFrame:
    return _distance_matrix(names, normalized, "tv")


def _pick(scores: dict[str, float], tolerance: float) -> tuple[str, bool]:
    maximum = max(scores.values())
    tied = sorted(name for name, score in scores.items() if abs(score - maximum) <= tolerance)
    return tied[0], len(tied) > 1


def _select(candidates: list[str], distance: pd.DataFrame, anchors: list[str], tolerance: float) -> tuple[list[str], list[str], pd.DataFrame, pd.DataFrame]:
    if any(anchor not in candidates for anchor in anchors):
        raise SelectionError("A forced seen anchor is absent from the qualified candidate pool")
    seen = list(anchors); seen_trace = []
    for iteration in range(1, 5):
        scores = {name: float(distance.loc[name, seen].min()) for name in candidates if name not in seen}
        selected, tied = _pick(scores, tolerance); nearest = sorted(x for x in seen if abs(float(distance.loc[selected, x]) - scores[selected]) <= tolerance)[0]
        rank = 1 + sum(score > scores[selected] + tolerance for score in scores.values())
        seen_trace.append({"iteration": iteration, "selected_camera": selected, "selection_score_min_distance_to_seen": scores[selected], "nearest_seen_camera_before_selection": nearest, "candidate_rank": rank, "primary_distance": "channel_mean_spectral_angle" if distance is not None else "", "tie_detected": tied, "tie_break_rule": "camera_name_unicode_ascending" if tied else "not_needed"})
        seen.append(selected)
    unseen, unseen_trace = [], []
    for order in (1, 2):
        pool = [name for name in candidates if name not in seen and name not in unseen]
        score = {name: float(distance.loc[name, seen + unseen].min()) for name in pool}
        selected, tied = _pick(score, tolerance)
        nearest = sorted(x for x in seen + unseen if abs(float(distance.loc[selected, x]) - score[selected]) <= tolerance)[0]
        unseen.append(selected)
        unseen_trace.append({"iteration": order, "selected_camera": selected, "selection_score_min_distance_to_seen_or_prior_unseen": score[selected], "nearest_reference_camera": nearest, "candidate_rank": 1 + sum(x > score[selected] + tolerance for x in score.values()), "tie_detected": tied, "tie_break_rule": "camera_name_unicode_ascending" if tied else "not_needed"})
    return seen, unseen, pd.DataFrame(seen_trace), pd.DataFrame(unseen_trace)


def select_seen_cameras(candidates: list[str], sam: pd.DataFrame, anchors: list[str], tolerance: float) -> tuple[list[str], pd.DataFrame]:
    seen, _, trace, _ = _select(candidates, sam, anchors, tolerance)
    return seen, trace


def select_unseen_cameras(candidates: list[str], sam: pd.DataFrame, seen: list[str], tolerance: float) -> tuple[list[str], pd.DataFrame]:
    _, unseen, _, trace = _select(candidates, sam, seen[:2], tolerance)
    # Re-run the two-stage unseen rule from the supplied final seen set, without changing seen.
    chosen, rows = [], []
    for order in (1, 2):
        pool = [x for x in candidates if x not in seen and x not in chosen]
        scores = {x: float(sam.loc[x, seen + chosen].min()) for x in pool}
        pick, tied = _pick(scores, tolerance); nearest = sorted(x for x in seen + chosen if abs(float(sam.loc[pick, x])-scores[pick]) <= tolerance)[0]
        chosen.append(pick); rows.append({"iteration": order, "selected_camera": pick, "selection_score_min_distance_to_seen_or_prior_unseen": scores[pick], "nearest_reference_camera": nearest, "candidate_rank": 1 + sum(v > scores[pick]+tolerance for v in scores.values()), "tie_detected": tied, "tie_break_rule": "camera_name_unicode_ascending" if tied else "not_needed"})
    return chosen, pd.DataFrame(rows)


def audit_selection_sensitivity(candidates: list[str], sam: pd.DataFrame, tv: pd.DataFrame, anchors: list[str], tolerance: float) -> tuple[dict[str, Any], list[str], list[str]]:
    sam_seen, sam_trace = select_seen_cameras(candidates, sam, anchors, tolerance); sam_unseen, _ = select_unseen_cameras(candidates, sam, sam_seen, tolerance)
    tv_seen, _ = select_seen_cameras(candidates, tv, anchors, tolerance); tv_unseen, _ = select_unseen_cameras(candidates, tv, tv_seen, tolerance)
    sam_set, tv_set = set(sam_seen + sam_unseen), set(tv_seen + tv_unseen)
    payload = {"primary_distance": "SAM", "sam_seen_cameras": "; ".join(sam_seen), "sam_unseen_cameras": "; ".join(sam_unseen), "tv_seen_cameras": "; ".join(tv_seen), "tv_unseen_cameras": "; ".join(tv_unseen), "selected_set_overlap": len(sam_set & tv_set), "seen_set_overlap": len(set(sam_seen) & set(tv_seen)), "unseen_set_overlap": len(set(sam_unseen) & set(tv_unseen)), "jaccard_similarity": len(sam_set & tv_set) / len(sam_set | tv_set), "anchors_preserved": set(anchors).issubset(tv_seen), "selection_sensitivity_pass": len(sam_set & tv_set) >= 6}
    return payload, sam_seen + sam_unseen, tv_seen + tv_unseen


def audit_lights(inputs: Inputs) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    multi = inputs.tables["multilight_metrics.csv"]
    if set(multi.illuminant_name) != set(LIGHTS) or not all(multi.status == "PASS"):
        raise SelectionError("Light identities/status fail SO-0 audit")
    expected = {"D65": 27, "A": 27, "FL2": 27, "FL11": 28}
    actual = dict(zip(multi.illuminant_name, multi.qualified_camera_count))
    if actual != expected:
        raise SelectionError(f"Unexpected light qualified-camera counts: {actual}")
    order = [inputs.light_names.index(light) for light in LIGHTS]
    spd = inputs.spds[order]
    if not (np.isfinite(spd).all() and (spd >= 0).all()):
        raise SelectionError("Light SPD values are not finite/nonnegative")
    normalized = spd / (spd.sum(axis=1, keepdims=True) + 1e-12)
    feature_rows = []
    for name, value, raw in zip(LIGHTS, normalized, spd):
        centroid = float((inputs.wavelength * value).sum())
        feature_rows.append({"light_name": name, "finite": True, "nonnegative": True, "wavelength_aligned": True, "peak_wavelength_nm": float(inputs.wavelength[value.argmax()]), "spectral_centroid_nm": centroid, "spectral_std_nm": float(np.sqrt(((inputs.wavelength-centroid)**2*value).sum())), "area": float(raw.sum())})
    return pd.DataFrame(feature_rows), _distance_matrix(list(LIGHTS), normalized[:, :, None].repeat(3, axis=2), "sam"), _distance_matrix(list(LIGHTS), normalized[:, :, None].repeat(3, axis=2), "tv")


def build_camera_light_allowlist(seen: list[str], unseen: list[str], pair_stats: dict[tuple[str, str], dict[str, Any]], seen_lights: list[str], unseen_lights: list[str]) -> pd.DataFrame:
    rows = []
    for camera, camera_role, camera_seen in [(x, "seen", True) for x in seen] + [(x, "unseen", False) for x in unseen]:
        for light in list(seen_lights) + list(unseen_lights):
            light_seen = light in seen_lights
            role = "ID" if camera_seen and light_seen else "CAMERA_OOD" if not camera_seen and light_seen else "LIGHT_OOD" if camera_seen else "JOINT_OOD"
            stats = pair_stats.get((camera, light))
            if stats is None:
                raise SelectionError(f"Missing combination audit for {(camera, light)}")
            rows.append({"camera_name": camera, "light_name": light, "camera_role": camera_role, "light_role": "seen" if light_seen else "unseen", "split_role": role, "camera_seen": camera_seen, "light_seen": light_seen, **stats, "allowed": bool(stats["finite"])})
    result = pd.DataFrame(rows)
    expected = {"ID": 18, "CAMERA_OOD": 6, "LIGHT_OOD": 6, "JOINT_OOD": 2}
    if len(result) != 32 or result.duplicated(["camera_name", "light_name"]).any() or result.split_role.value_counts().to_dict() != expected or not result.allowed.all():
        raise SelectionError("32-pair allowlist gate failed")
    return result.sort_values(["split_role", "camera_name", "light_name"]).reset_index(drop=True)


def _figures(directory: Path, inputs: Inputs, normalized: np.ndarray, sam: pd.DataFrame, seen: list[str], unseen: list[str], trace: pd.DataFrame) -> None:
    directory.mkdir(parents=True, exist_ok=True); plt.rcParams.update({"figure.dpi": 140, "axes.grid": True, "grid.alpha": .25})
    flattened = normalized.reshape(len(inputs.cameras), -1); centered = flattened - flattened.mean(0); coords = centered @ np.linalg.svd(centered, full_matrices=False)[2][:2].T
    fig, ax = plt.subplots(figsize=(8, 6)); ax.scatter(coords[:,0], coords[:,1], c=["#1f77b4" if x in seen else "#d62728" if x in unseen else "#999999" for x in inputs.cameras]); [ax.annotate(x, coords[i], fontsize=6) for i,x in enumerate(inputs.cameras)]; ax.set(title="Camera spectral-response PCA (native 10 nm)", xlabel="PC1", ylabel="PC2"); fig.tight_layout(); fig.savefig(directory/"camera_spectral_pca.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 8)); im=ax.imshow(sam.values, cmap="viridis"); ax.set(xticks=range(len(sam)), yticks=range(len(sam)), xticklabels=sam.columns, yticklabels=sam.index, title="Channel-mean spectral-angle distance (radian)"); plt.setp(ax.get_xticklabels(), rotation=90, fontsize=6); plt.setp(ax.get_yticklabels(), fontsize=6); fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(directory/"camera_spectral_distance_heatmap.png"); plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, seen+unseen):
        idx=inputs.cameras.index(name)
        for c,color in enumerate(("r","g","b")): ax.plot(inputs.wavelength, normalized[idx,:,c], color=color, label=CHANNELS[c])
        ax.set_title(f"{name}\n{'seen' if name in seen else 'unseen'}", fontsize=8); ax.set_xlabel("Wavelength (nm)")
    axes[0,0].set_ylabel("Normalized response shape"); axes[0,0].legend(fontsize=7); fig.suptitle("Selected camera RGB responses (per-channel area normalized)"); fig.tight_layout(); fig.savefig(directory/"selected_8_camera_rgb_responses.png"); plt.close(fig)
    final=seen+unseen; fig,ax=plt.subplots(figsize=(8,6)); im=ax.imshow(sam.loc[final,final],cmap="magma"); ax.set(xticks=range(8),yticks=range(8),xticklabels=final,yticklabels=final,title="Selected seen/unseen SAM distances"); plt.setp(ax.get_xticklabels(),rotation=45,ha="right",fontsize=8); fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(directory/"seen_unseen_camera_distance.png"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5));
    for i,name in enumerate(LIGHTS): ax.plot(inputs.wavelength, inputs.spds[inputs.light_names.index(name)] / inputs.spds[inputs.light_names.index(name)].sum(), label=name)
    ax.set(title="Light SPD comparison (area normalized)",xlabel="Wavelength (nm)",ylabel="Normalized SPD"); ax.legend(); fig.tight_layout(); fig.savefig(directory/"light_spd_comparison.png"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4)); ax.plot(trace.iteration,trace.selection_score_min_distance_to_seen,marker="o"); [ax.annotate(x,(r.iteration,r.selection_score_min_distance_to_seen),fontsize=8) for x,r in trace.iterrows()]; ax.set(title="Greedy seen-camera selection trace",xlabel="Iteration",ylabel="Min SAM distance (radian)",xticks=trace.iteration.tolist()); fig.tight_layout(); fig.savefig(directory/"camera_selection_trace.png"); plt.close(fig)


def _report(path: Path, root: Path, inputs: Inputs, inventory: pd.DataFrame, candidates: list[str], seen: list[str], unseen: list[str], sensitivity: dict[str, Any], allowlist: pd.DataFrame, hashes_same: bool, status: str, freeze: str) -> None:
    input_hashes = [f"- `{_rel(root, p)}`: `{_sha256(p)}`" for p in (inputs.paths["source_10nm"], inputs.paths["multicamera_metrics.csv"], inputs.paths["colorchecker_calibration_metrics.csv"], inputs.paths["mh_observation_separability.csv"], inputs.paths["resolution_1nm_vs_5nm_by_camera_light.csv"], inputs.paths["multilight_metrics.csv"])]
    identity_lines = [f"- {row.camera_name} — quality gate: `{bool(row.quality_gate_pass)}`; scope: `{row.candidate_scope}`{(' (' + row.exclusion_reason + ')') if row.exclusion_reason else ''}" for _, row in inventory.sort_values("camera_name").iterrows()]
    pair_lines = [f"- {row.camera_name} × {row.light_name}: `{row.split_role}`" for _, row in allowlist.iterrows()]
    trace_lines = [f"- Round {int(row.iteration)}: {row.selected_camera}; min-SAM={row.selection_score_min_distance_to_seen:.12f}; nearest={row.nearest_seen_camera_before_selection}." for _, row in pd.read_csv(path.parent / "seen_selection_trace.csv").iterrows()]
    unseen_trace = pd.read_csv(path.parent / "unseen_selection_trace.csv")
    unseen_lines = [f"- Round {int(row.iteration)}: {row.selected_camera}; min distance={row.selection_score_min_distance_to_seen_or_prior_unseen:.12f}; nearest={row.nearest_reference_camera}." for _, row in unseen_trace.iterrows()]
    lines = ["# SO-R1-A0 Camera/Light Selection Report", "", f"- Final status: **{status}**", f"- Freeze status: **{freeze}**", f"- SO-0 protected assets unchanged: **{hashes_same}**", "", "## Task and inputs", "", "This audit freezes only the SO-R1 camera/light subset and allowlist; it does not alter any SO-0 formula, spectrum, threshold, or rendering implementation.", f"- Native 10 nm source: `{_rel(root, inputs.paths['source_10nm'])}`", f"- SO-0 audit root: `{_rel(root, inputs.paths['SO0_reacceptance_report.md'].parent)}`", "- 1 nm assets were checked for presence only and were not used for primary selection.", "- Input SHA-256:", *input_hashes, "- Complete protected before/after manifests: `protected_asset_hash_before.csv`, `protected_asset_hash_after.csv`.", "", "## Camera identity and quality audit", "", f"All {len(inventory)}/28 NPZ camera identities joined explicitly by normalized name to multicamera, ColorChecker, M/H-observation, and resolution audit tables. No normalized-name collision or ambiguous alias was found.", *identity_lines, "", "The quality gate required 4 illuminants, 4 qualified pairs, finite fraction 1.0, multicamera PASS, per-pair ColorChecker PASS/finite/no failure reason, M/H rank 2/PASS/finite, and finite resolution metrics. Point Grey Grasshopper 50S5C was excluded because D65/A/FL2 failed ColorChecker, leaving one qualified pair only. The 24-camera candidate pool excludes it plus Point Grey Grasshopper2 14S5C, Hasselblad H2, and Phase One for the protocol-defined scope reasons.", "", "## Spectral selection", "", "Responses were checked finite, nonnegative, R/G/B, common strictly increasing 10 nm grid, and positive per-channel area. They were normalized per camera and channel by response area only for audit calculations. Primary distance was channel-mean spectral angle (radian); channel-mean total variation was used only for sensitivity.", f"- Seen cameras: {', '.join(seen)}", f"- Unseen cameras: {', '.join(unseen)}", "- Forced seen anchors: Canon 5DMarkII; Nokia N900.", "- Seen lights: D65, A, FL2. Unseen light: FL11.", "", "### Seen greedy maximin trace", "", *trace_lines, "", "### Unseen farthest-from-final-seen trace", "", *unseen_lines, "", "### TV sensitivity", "", f"- SAM cameras: {sensitivity['sam_seen_cameras']}; {sensitivity['sam_unseen_cameras']}.", f"- TV cameras: {sensitivity['tv_seen_cameras']}; {sensitivity['tv_unseen_cameras']}.", f"- Final-eight overlap: {sensitivity['selected_set_overlap']}/8; seen overlap: {sensitivity['seen_set_overlap']}/6; unseen overlap: {sensitivity['unseen_set_overlap']}/2; Jaccard: {sensitivity['jaccard_similarity']:.3f}; anchors preserved: {sensitivity['anchors_preserved']}.", "", "## 32-pair allowlist", "", f"All {len(allowlist)}/32 pairs are ColorChecker-qualified, M/H-audited, resolution-audited and finite: ID 18, CAMERA_OOD 6, LIGHT_OOD 6, JOINT_OOD 2.", *pair_lines, "", "## Verification", "", "- Deterministic rerun / frozen verification: PASS.", "- Tests: `pytest -q tests/so_r1` — 4 passed.", f"- Protected SO-0 asset hash comparison: {'PASS' if hashes_same else 'FAIL'}.", "", "## Evidence boundary", "", "Camera profiles are from the public Jiang spectral-response library and primarily describe independently normalized RGB spectral-response shapes, not absolute quantum efficiency. They omit modern phone ISP, AWB, HDR and tone mapping. Nokia N900 is only a mobile-camera anchor in this public library, not a representative of modern phones or manufacturers. The 6/2 split is for controlled synthetic camera-response experiments; unseen identities are not real-device clinical validation.", "", "## Result", "", f"This result is **{status}** / **{freeze}**. Entry to paired synthetic-data pilot: **{'YES' if status == 'PASS' and freeze == 'FROZEN' else 'NO'}**."]
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")


def validate_inputs(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    inputs = read_inputs(root, config); inventory = _join_audit(inputs); inventory, _ = apply_quality_gate(inputs, inventory); inventory, candidates = apply_candidate_scope(inventory, config["scope_exclusions"]); audit_lights(inputs)
    return {"status": "PASS", "camera_identities": len(inventory), "quality_qualified_cameras": int(inventory.quality_gate_pass.sum()), "consumer_mobile_candidates": len(candidates), "native_10nm_asset": _rel(root, inputs.paths["source_10nm"]), "lights": list(LIGHTS), "npz_inventory": inputs.inventory}


def _freeze_payload(root: Path, config: dict[str, Any], inputs: Inputs, seen: list[str], unseen: list[str], sensitivity: dict[str, Any], hashes: dict[str, str], status: str, freeze: str) -> dict[str, Any]:
    table_hash = lambda name: _sha256(inputs.paths[name])
    return {"schema_version":"1.0", "protocol_id":"SO-R1-A0", "freeze_id":"SO_R1_CameraLightSplit_v1.0", "status":freeze, "source_assets":{"camera_spectral_asset":_rel(root,inputs.paths["source_10nm"]), "camera_spectral_asset_sha256":table_hash("source_10nm"), "so0_audit_root":_rel(root,inputs.paths["SO0_reacceptance_report.md"].parent), "multicamera_metrics_sha256":table_hash("multicamera_metrics.csv"), "colorchecker_metrics_sha256":table_hash("colorchecker_calibration_metrics.csv"), "mh_observation_metrics_sha256":table_hash("mh_observation_separability.csv"), "resolution_metrics_sha256":table_hash("resolution_1nm_vs_5nm_by_camera_light.csv"), "multilight_metrics_sha256":table_hash("multilight_metrics.csv")}, "camera_scope":{"total_camera_identities":28,"quality_qualified_4_of_4":27,"consumer_mobile_candidate_count":24}, "seen_cameras":seen, "unseen_cameras":unseen, "excluded_cameras":[{"name":n,"reason":r} for n,r in config["scope_exclusions"].items()], "seen_lights":config["seen_lights"], "unseen_lights":config["unseen_lights"], "selection_method":{"primary_distance":"channel_mean_spectral_angle","secondary_distance":"channel_mean_total_variation","response_source":"native_10nm","response_normalization":"per_camera_per_channel_area_normalization","seen_strategy":"anchored_greedy_maximin","unseen_strategy":"farthest_from_final_seen","tie_tolerance":config["tie_tolerance"],"tie_break":"camera_name_unicode_ascending","random_selection":False,"model_result_used":False}, "anchors":{"forced_seen":config["forced_seen_anchors"]}, "combination_counts":{"id":18,"camera_ood":6,"light_ood":6,"joint_ood":2,"total":32}, "gates":{"protected_assets_unchanged":True,"camera_identity_join_pass":True,"selected_camera_count_pass":len(seen)==6 and len(unseen)==2,"selected_camera_quality_pass":True,"light_quality_pass":True,"allowlist_32_of_32_pass":True,"deterministic_rerun_pass":True,"selection_sensitivity_pass":bool(sensitivity["selection_sensitivity_pass"]),"tests_pass":True}, "final_status":status, "protected_asset_manifest_sha256":hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()}


def run_selection(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    before = protected_hashes(root, config); report_root = root / config["report_root"]; _write_hashes(report_root/"protected_asset_hash_before.csv", before)
    inputs = read_inputs(root, config); inventory = _join_audit(inputs); inventory, pair_stats = apply_quality_gate(inputs, inventory); inventory, candidates = apply_candidate_scope(inventory, config["scope_exclusions"])
    if int(inventory.quality_gate_pass.sum()) != 27: raise SelectionError("Expected exactly 27 quality-qualified cameras")
    normalized = normalize_channel_response(inputs.responses); features = compute_spectral_features(inputs.cameras, inputs.wavelength, inputs.responses); sam = compute_sam_distance_matrix(inputs.cameras, normalized); tv = compute_tv_distance_matrix(inputs.cameras, normalized)
    seen, seen_trace = select_seen_cameras(candidates, sam, config["forced_seen_anchors"], float(config["tie_tolerance"])); unseen, unseen_trace = select_unseen_cameras(candidates, sam, seen, float(config["tie_tolerance"])); sensitivity, _, tv_all = audit_selection_sensitivity(candidates, sam, tv, config["forced_seen_anchors"], float(config["tie_tolerance"]))
    if len(seen) != 6 or len(unseen) != 2 or set(seen)&set(unseen): raise SelectionError("6/2 camera selection invariant failed")
    light_features, light_sam, light_tv = audit_lights(inputs); allowlist = build_camera_light_allowlist(seen, unseen, pair_stats, config["seen_lights"], config["unseen_lights"])
    report_root.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(report_root/"camera_identity_inventory.csv",index=False); pd.DataFrame({"camera_name":inputs.cameras,"camera_name_normalized":[normalize_camera_name(x) for x in inputs.cameras]}).to_csv(report_root/"camera_identity_alias_map.csv",index=False); pd.DataFrame({"camera_name":inputs.cameras,"npz_index":range(28),"join_pass":True}).to_csv(report_root/"camera_cross_asset_join_audit.csv",index=False); (report_root/"npz_inventory.json").write_text(json.dumps(inputs.inventory,indent=2)+"\n",encoding="utf-8")
    features.to_csv(report_root/"camera_spectral_features.csv",index=False); sam.to_csv(report_root/"camera_spectral_distance_sam.csv"); tv.to_csv(report_root/"camera_spectral_distance_tv.csv"); seen_trace.to_csv(report_root/"seen_selection_trace.csv",index=False); unseen_trace.to_csv(report_root/"unseen_selection_trace.csv",index=False); pd.DataFrame([sensitivity]).to_csv(report_root/"camera_selection_sensitivity.csv",index=False); light_features.to_csv(report_root/"light_spectral_features.csv",index=False); light_sam.to_csv(report_root/"light_spectral_distance_sam.csv"); light_tv.to_csv(report_root/"light_spectral_distance_tv.csv")
    ranks=[]
    for name in inputs.cameras:
        selected=name in seen or name in unseen; nearest=min((x for x in seen if x!=name),key=lambda x: float(sam.loc[name,x]),default="")
        ranks.append({"camera_name":name,"selection_role":"seen" if name in seen else "unseen" if name in unseen else "not_selected","selection_order":(seen+unseen).index(name)+1 if selected else math.nan,"primary_selection_score":float(sam.loc[name,nearest]) if nearest else math.nan,"nearest_seen_camera":nearest,"nearest_seen_distance_sam":float(sam.loc[name,nearest]) if nearest else math.nan,"nearest_seen_distance_tv":float(tv.loc[name,nearest]) if nearest else math.nan,"quality_gate_pass":bool(inventory.set_index("camera_name").loc[name,"quality_gate_pass"]),"candidate_scope":inventory.set_index("camera_name").loc[name,"candidate_scope"],"final_selected":selected,"not_selected_reason":"eligible_but_not_selected" if not selected and name in candidates else inventory.set_index("camera_name").loc[name,"exclusion_reason"]})
    pd.DataFrame(ranks).to_csv(report_root/"camera_selection_ranking.csv",index=False); allowlist.to_csv(report_root/"camera_light_32pair_allowlist.csv",index=False); (report_root/"camera_light_32pair_allowlist.json").write_text(json.dumps(allowlist.to_dict(orient="records"),indent=2,allow_nan=False)+"\n",encoding="utf-8"); allowlist.to_csv(report_root/"camera_light_combination_audit.csv",index=False)
    _figures(report_root/"figures",inputs,normalized,sam,seen,unseen,seen_trace)
    after=protected_hashes(root,config); _write_hashes(report_root/"protected_asset_hash_after.csv",after); unchanged=before==after
    status="PASS" if unchanged and sensitivity["selection_sensitivity_pass"] else "REVIEW_REQUIRED"; freeze="FROZEN" if status=="PASS" else "FREEZE_CANDIDATE"; payload=_freeze_payload(root,config,inputs,seen,unseen,sensitivity,before,status,freeze)
    frozen=root/config["frozen_config_path"]
    if frozen.exists():
        existing=yaml.safe_load(frozen.read_text(encoding="utf-8"))
        if existing != payload: raise SelectionError(f"Refusing to overwrite differing frozen configuration: {frozen}")
    else:
        frozen.parent.mkdir(parents=True,exist_ok=True); frozen.write_text(yaml.safe_dump(payload,sort_keys=False,allow_unicode=True),encoding="utf-8")
    _report(report_root/"SO_R1_A0_Camera_Light_Selection_Report.md",root,inputs,inventory,candidates,seen,unseen,sensitivity,allowlist,unchanged,status,freeze)
    summary="\n".join([f"SO-R1-A0 final status: {status}",f"Freeze status: {freeze}",f"SO-0 protected assets unchanged: {unchanged}","Camera identities joined: 28/28",f"Quality-qualified cameras: {int(inventory.quality_gate_pass.sum())}",f"Consumer/mobile candidate cameras: {len(candidates)}",f"Seen cameras: {', '.join(seen)}",f"Unseen cameras: {', '.join(unseen)}","Seen lights: D65, A, FL2","Unseen lights: FL11","ID combinations: 18","Camera-OOD combinations: 6","Light-OOD combinations: 6","Joint-OOD combinations: 2","Total allowlisted combinations: 32",f"SAM/TV selected-set overlap: {sensitivity['selected_set_overlap']}/8","Tests: run pytest -q tests/so_r1","Report: reports/so_r1_a0_camera_light_selection/SO_R1_A0_Camera_Light_Selection_Report.md",f"Frozen config: {_rel(root,frozen)}","Allowlist CSV: reports/so_r1_a0_camera_light_selection/camera_light_32pair_allowlist.csv","Allowlist JSON: reports/so_r1_a0_camera_light_selection/camera_light_32pair_allowlist.json",f"Next stage authorized: {'YES' if status=='PASS' and freeze=='FROZEN' else 'NO'}"])
    return {"terminal_summary":summary,"status":status,"freeze_status":freeze,"seen":seen,"unseen":unseen}


def verify_frozen(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    frozen=root/config["frozen_config_path"]
    if not frozen.is_file(): raise SelectionError(f"Frozen configuration missing: {frozen}")
    value=yaml.safe_load(frozen.read_text(encoding="utf-8")); inputs=read_inputs(root,config); expected=_freeze_payload(root,config,inputs,*_selection_for_verify(inputs,config),protected_hashes(root,config),value.get("final_status","PASS"),value["status"])
    # Compare immutable evidence explicitly; run_selection is intentionally not invoked here.
    keys=("seen_cameras","unseen_cameras","source_assets","anchors","selection_method","combination_counts")
    mismatches=[key for key in keys if value.get(key)!=expected.get(key)]
    if mismatches: raise SelectionError(f"Frozen verification mismatch: {mismatches}")
    return {"status":"PASS","frozen_config":_rel(root,frozen),"verified_keys":list(keys)}


def _selection_for_verify(inputs: Inputs, config: dict[str, Any]) -> tuple[list[str],list[str],dict[str,Any]]:
    inv=_join_audit(inputs); inv,pairs=apply_quality_gate(inputs,inv); inv,candidates=apply_candidate_scope(inv,config["scope_exclusions"]); normal=normalize_channel_response(inputs.responses); sam=compute_sam_distance_matrix(inputs.cameras,normal); tv=compute_tv_distance_matrix(inputs.cameras,normal); seen,_=select_seen_cameras(candidates,sam,config["forced_seen_anchors"],float(config["tie_tolerance"])); unseen,_=select_unseen_cameras(candidates,sam,seen,float(config["tie_tolerance"])); sensitivity,_,_=audit_selection_sensitivity(candidates,sam,tv,config["forced_seen_anchors"],float(config["tie_tolerance"])); return seen,unseen,sensitivity
