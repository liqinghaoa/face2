"""Reconcile Stage C tolerance sets across every saved profile and multistart solution."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "outputs/skin_optics_hsi_v1/stage1_hsi_physics/km_bio_v1_train_inversion"
OUT = ROOT / "outputs/skin_optics_hsi_v1/stage1_hsi_physics/km_bio_v1_train_inversion_verification"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    target_csv = OUT / "profile_solution_set_reconciliation.csv"
    target_json = OUT / "profile_solution_set_reconciliation.json"
    if target_csv.exists() or target_json.exists():
        raise FileExistsError("Refusing to overwrite profile reconciliation")
    main = pd.read_parquet(RUN / "main_results.parquet").set_index(["capture_id", "roi"])
    grid = pd.read_parquet(RUN / "profile_grid.parquet")
    starts = pd.read_parquet(RUN / "profile_multistart_results.parquet")
    parameters = [("f_mel", 0, 0.03), ("f_blood", 1, 0.01), ("s", 2, 0.20)]
    rows = []
    for key, group in grid.groupby(["capture_id", "roi"]):
        optimum = min(float(main.loc[key, "logrmse"]), float(group.loc[group["success"], "logrmse"].min()))
        start_group = starts.loc[(starts["capture_id"].eq(key[0])) & (starts["roi"].eq(key[1]))]
        for parameter, index, limit in parameters:
            for delta in (0.0025, 0.005, 0.010):
                grid_values = [
                    float(theta[index]) for theta in group.loc[group["success"] & group["logrmse"].le(optimum + delta), "conditional_theta"]
                    if theta is not None
                ]
                start_values = [
                    float(theta[index]) for theta in start_group.loc[start_group["valid"] & start_group["logrmse"].le(optimum + delta), "final_theta"]
                ]
                values = grid_values + start_values
                rows.append({
                    "subject_id": group["subject_id"].iloc[0], "capture_id": key[0], "roi": key[1],
                    "parameter": parameter, "delta_logrmse": delta,
                    "accepted_grid_solution_count": len(grid_values), "accepted_multistart_solution_count": len(start_values),
                    "accepted_min": min(values), "accepted_max": max(values), "total_span": max(values) - min(values),
                    "profile_span_gate": (max(values) - min(values)) <= limit,
                })
    frame = pd.DataFrame(rows)
    frame.to_csv(target_csv, index=False, encoding="utf-8-sig")
    main_delta = frame.loc[frame["delta_logrmse"].eq(0.005)]
    summary = {
        "schema_version": 1,
        "stage": "KM-BIO-v1-Stage-C-profile-reconciliation",
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "min/max across every tolerance-accepted conditional grid solution and all saved conditional multistart solutions",
        "main_delta": 0.005,
        "profile_gate_counts": {
            parameter: int(group["profile_span_gate"].sum()) for parameter, group in main_delta.groupby("parameter")
        },
        "span_statistics": {
            parameter: {
                "median": float(group["total_span"].median()),
                "p90": float(group["total_span"].quantile(0.9)),
                "maximum": float(group["total_span"].max()),
            }
            for parameter, group in main_delta.groupby("parameter")
        },
        "stage_c_status_unchanged": "REVISE_OBSERVATION_OR_MODEL",
        "reason_status_unchanged": "all 88 spectra failed the prerequisite single-spectrum spectral gate",
        "output": {"path": str(target_csv), "sha256": digest(target_csv)},
    }
    target_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

