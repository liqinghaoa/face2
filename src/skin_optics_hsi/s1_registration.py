"""S1-1: stratified RGB-HSI spatial-transform audit on Train only."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel

from .data_contracts import load_hyperskin_cube


TRANSFORMS = {
    "identity": lambda x: x,
    "rot90_ccw": lambda x: np.rot90(x, 1),
    "rot180": lambda x: np.rot90(x, 2),
    "rot90_cw": lambda x: np.rot90(x, 3),
    "flip_lr": lambda x: np.fliplr(x),
    "flip_ud": lambda x: np.flipud(x),
    "transpose": lambda x: np.transpose(x),
    "anti_transpose": lambda x: np.flipud(np.fliplr(np.transpose(x))),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("Image contains no finite values")
    low, high = np.nanpercentile(values[finite], [1.0, 99.0])
    if high <= low:
        raise ValueError("Image has no usable contrast")
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _resize(image: np.ndarray, size: int) -> np.ndarray:
    uint8 = np.rint(_normalize(image) * 255.0).astype(np.uint8)
    with Image.fromarray(uint8, mode="L") as pil:
        return np.asarray(pil.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    a = left.ravel() - float(left.mean())
    b = right.ravel() - float(right.mean())
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 0 else float("nan")


def _edge_map(image: np.ndarray) -> np.ndarray:
    smooth = gaussian_filter(image, sigma=1.0)
    return np.hypot(sobel(smooth, axis=0), sobel(smooth, axis=1))


def _rgb_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _hsi_proxy(cube: np.ndarray) -> np.ndarray:
    # Broadband mean is used only for spatial registration. It does not depend
    # on the unconfirmed exact center-wavelength array.
    return np.nanmedian(cube, axis=-1)


def select_registration_samples(manifest: pd.DataFrame, samples_per_stratum: int = 3) -> pd.DataFrame:
    train = manifest.loc[manifest["split"] == "train"].copy()
    if train.empty:
        raise ValueError("S1-1 requires Train samples")
    brightness: list[float] = []
    for value in train["rgb_path"]:
        gray = _rgb_gray(Path(value))
        h, w = gray.shape
        center = gray[h // 8 : 7 * h // 8, w // 8 : 7 * w // 8]
        brightness.append(float(np.median(center)))
    train["rgb_central_median_brightness"] = brightness
    selected: list[pd.DataFrame] = []
    required = {(expression, direction) for expression in ("neutral", "smile") for direction in ("front", "left", "right")}
    present = set(zip(train["expression"], train["direction"]))
    if not required.issubset(present):
        raise ValueError(f"Train is missing registration strata: {sorted(required - present)}")
    for expression, direction in sorted(required):
        group = train.loc[(train["expression"] == expression) & (train["direction"] == direction)].sort_values(
            ["rgb_central_median_brightness", "sample_id"]
        )
        if len(group) < samples_per_stratum:
            raise ValueError(f"Not enough samples in {expression}/{direction}")
        positions = np.linspace(0, len(group) - 1, samples_per_stratum).round().astype(int)
        chosen = group.iloc[positions].copy()
        chosen["brightness_stratum"] = ["low", "mid", "high"] if samples_per_stratum == 3 else [f"q{i}" for i in range(samples_per_stratum)]
        selected.append(chosen)
    return pd.concat(selected, ignore_index=True).sort_values(["expression", "direction", "brightness_stratum"])


def _save_panel(path: Path, sample_id: str, rgb: np.ndarray, native: np.ndarray, mapped: np.ndarray, transform: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)
    for axis, image, title in zip(axes, (rgb, native, mapped), ("RGB luminance", "HSI native", f"HSI → RGB: {transform}")):
        axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(sample_id)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _save_contact_sheet(path: Path, previews: list[tuple[str, np.ndarray, np.ndarray]]) -> None:
    columns = 6
    rows = int(np.ceil(len(previews) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(15, 2.6 * rows), squeeze=False, constrained_layout=True)
    for axis in axes.ravel():
        axis.axis("off")
    for axis, (sample_id, rgb, mapped) in zip(axes.ravel(), previews):
        overlay = np.zeros((*rgb.shape, 3), dtype=np.float64)
        overlay[..., 0] = rgb
        overlay[..., 1] = mapped
        axis.imshow(overlay)
        axis.set_title(sample_id, fontsize=8)
        axis.axis("off")
    fig.suptitle("RGB (red) / transformed HSI (green): yellow edges indicate agreement")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_s1_1_registration(
    contract_path: str | Path,
    output_root: str | Path,
    *,
    samples_per_stratum: int = 3,
    working_size: int = 256,
    min_intensity_correlation: float = 0.95,
    min_edge_correlation: float = 0.70,
    min_score_margin: float = 0.05,
) -> dict[str, Any]:
    contract_file = Path(contract_path).resolve()
    contract = json.loads(contract_file.read_text(encoding="utf-8"))
    if contract.get("status") != "PASS" or not contract.get("readiness", {}).get("s1_1_allowed"):
        raise ValueError("S1-0 contract does not allow S1-1")
    manifest_path = Path(contract["manifest"]["path"])
    if _sha256(manifest_path) != contract["manifest"]["sha256"]:
        raise ValueError("S1-0 manifest hash mismatch")
    output = Path(output_root).resolve() / "registration_qc"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty S1-1 output: {output}")
    panels = output / "review_panels"
    panels.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path)
    selected = select_registration_samples(manifest, samples_per_stratum=samples_per_stratum)
    metric_rows: list[dict[str, Any]] = []
    image_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for _, sample in selected.iterrows():
        rgb = _resize(_rgb_gray(Path(sample["rgb_path"])), working_size)
        cube = load_hyperskin_cube(Path(sample["hsi_path"]), dataset_key=contract["hsi_storage"]["dataset_key"], expected_bands=31)
        native_full = _hsi_proxy(cube)
        native = _resize(native_full, working_size)
        image_cache[str(sample["sample_id"])] = (rgb, native)
        scores: list[dict[str, Any]] = []
        for transform_name, operation in TRANSFORMS.items():
            mapped = _resize(operation(native_full), working_size)
            intensity = _corr(rgb, mapped)
            edge = _corr(_edge_map(rgb), _edge_map(mapped))
            score = 0.75 * intensity + 0.25 * edge
            scores.append({"transform": transform_name, "intensity_correlation": intensity, "edge_correlation": edge, "combined_score": score})
        ordered = sorted(scores, key=lambda row: row["combined_score"], reverse=True)
        for row in scores:
            metric_rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "subject_id": sample["subject_id"],
                    "split": "train",
                    "expression": sample["expression"],
                    "direction": sample["direction"],
                    "brightness_stratum": sample["brightness_stratum"],
                    "rgb_central_median_brightness": sample["rgb_central_median_brightness"],
                    **row,
                    "sample_best_transform": ordered[0]["transform"],
                    "sample_best_score_margin": ordered[0]["combined_score"] - ordered[1]["combined_score"],
                }
            )
    metrics = pd.DataFrame(metric_rows)
    summary = (
        metrics.groupby("transform", as_index=False)
        .agg(
            mean_combined_score=("combined_score", "mean"),
            median_combined_score=("combined_score", "median"),
            median_intensity_correlation=("intensity_correlation", "median"),
            minimum_intensity_correlation=("intensity_correlation", "min"),
            median_edge_correlation=("edge_correlation", "median"),
            minimum_edge_correlation=("edge_correlation", "min"),
        )
        .sort_values("mean_combined_score", ascending=False)
    )
    best_transform = str(summary.iloc[0]["transform"])
    best_rows = metrics.loc[metrics["transform"] == best_transform].copy()
    consensus_fraction = float((best_rows["sample_best_transform"] == best_transform).mean())
    median_margin = float(best_rows["sample_best_score_margin"].median())
    checks = {
        "all_samples_same_best_transform": consensus_fraction == 1.0,
        "median_intensity_correlation": float(best_rows["intensity_correlation"].median()) >= min_intensity_correlation,
        "median_edge_correlation": float(best_rows["edge_correlation"].median()) >= min_edge_correlation,
        "median_best_score_margin": median_margin >= min_score_margin,
    }
    automatic_pass = all(checks.values())
    previews: list[tuple[str, np.ndarray, np.ndarray]] = []
    for _, sample in selected.iterrows():
        sample_id = str(sample["sample_id"])
        rgb, native = image_cache[sample_id]
        mapped = _resize(TRANSFORMS[best_transform](native), working_size)
        _save_panel(panels / f"{sample_id}.png", sample_id, rgb, native, mapped, best_transform)
        previews.append((sample_id, rgb, mapped))
    _save_contact_sheet(output / "registration_contact_sheet.png", previews)

    metrics.to_parquet(output / "registration_qc.parquet", index=False)
    metrics.to_csv(output / "registration_qc.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "transform_summary.csv", index=False, encoding="utf-8-sig")
    selected[["sample_id", "subject_id", "expression", "direction", "brightness_stratum", "rgb_central_median_brightness"]].to_csv(
        output / "selected_train_samples.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(
        {
            "sample_id": selected["sample_id"],
            "reviewer_decision": "",
            "anatomical_edge_alignment": "",
            "notes": "",
        }
    ).to_csv(output / "manual_review.csv", index=False, encoding="utf-8-sig")
    decision = {
        "schema_version": 1,
        "stage": "S1-1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_contract": str(contract_file),
        "input_manifest_sha256": contract["manifest"]["sha256"],
        "data_scope": "train_only",
        "selection": {
            "strategy": "per expression/direction stratum: low, median, high RGB central brightness",
            "samples_per_stratum": samples_per_stratum,
            "sample_count": int(len(selected)),
        },
        "metric": {
            "hsi_registration_proxy": "per-pixel median across 31 bands",
            "working_size": working_size,
            "combined_score": "0.75 * Pearson(intensity) + 0.25 * Pearson(Sobel magnitude)",
        },
        "thresholds_predeclared": {
            "consensus_fraction": 1.0,
            "median_intensity_correlation": min_intensity_correlation,
            "median_edge_correlation": min_edge_correlation,
            "median_score_margin": min_score_margin,
        },
        "automatic_candidate_transform": best_transform,
        "automatic_metrics": {
            "consensus_fraction": consensus_fraction,
            "median_intensity_correlation": float(best_rows["intensity_correlation"].median()),
            "minimum_intensity_correlation": float(best_rows["intensity_correlation"].min()),
            "median_edge_correlation": float(best_rows["edge_correlation"].median()),
            "minimum_edge_correlation": float(best_rows["edge_correlation"].min()),
            "median_best_score_margin": median_margin,
        },
        "automatic_checks": checks,
        "automatic_decision": "PASS" if automatic_pass else "REVISE",
        "manual_review": {"status": "pending", "path": str(output / "manual_review.csv")},
        "status": "AUTOMATIC_PASS_MANUAL_REVIEW_PENDING" if automatic_pass else "REVISE",
        "coordinate_mapping_frozen": False,
        "frozen_transform": None,
        "next_stage_allowed": False,
    }
    (output / "registration_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    return decision


def finalize_s1_1_manual_review(
    registration_dir: str | Path,
    *,
    reviewer: str,
    approve: bool,
    notes: str,
) -> dict[str, Any]:
    """Freeze or reject the automatic mapping after an explicit visual review."""

    directory = Path(registration_dir).resolve()
    decision_path = directory / "registration_decision.json"
    review_path = directory / "manual_review.csv"
    if not reviewer.strip():
        raise ValueError("reviewer must be recorded")
    if not notes.strip():
        raise ValueError("manual review notes must be recorded")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("manual_review", {}).get("status") != "pending":
        raise ValueError("S1-1 manual review was already finalized")
    if approve and decision.get("automatic_decision") != "PASS":
        raise ValueError("Cannot approve a registration mapping that failed automatic checks")
    review = pd.read_csv(review_path, keep_default_na=False)
    review["reviewer_decision"] = "PASS" if approve else "FAIL"
    review["anatomical_edge_alignment"] = "aligned" if approve else "not_accepted"
    review["notes"] = notes
    review["reviewer"] = reviewer
    review["reviewed_utc"] = datetime.now(timezone.utc).isoformat()
    review.to_csv(review_path, index=False, encoding="utf-8-sig")
    decision["manual_review"] = {
        "status": "PASS" if approve else "FAIL",
        "path": str(review_path),
        "reviewer": reviewer,
        "notes": notes,
        "reviewed_utc": datetime.now(timezone.utc).isoformat(),
    }
    decision["status"] = "PASS" if approve else "REVISE"
    decision["coordinate_mapping_frozen"] = bool(approve)
    decision["frozen_transform"] = decision["automatic_candidate_transform"] if approve else None
    decision["next_stage_allowed"] = bool(approve)
    decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return decision
