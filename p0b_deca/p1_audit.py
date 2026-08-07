"""Offline integrity audit for P1 frozen assets.

This audit intentionally performs no classification, label, camera, EXIF, or
counterfactual probe.  It validates only cache completeness and hashes.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Any
import numpy as np
from .p1_generation import CASE_FILES, LATENT_KEYS, MAP_KEYS, sha


def validate_case(case_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"case_id": case_dir.name, "success": False, "errors": []}
    success = case_dir / "_SUCCESS.json"
    if not success.is_file(): result["errors"].append("missing:_SUCCESS.json"); return result
    try:
        marker = json.loads(success.read_text(encoding="utf-8"))
        for name, digest in marker.get("output_file_sha256", {}).items():
            if not (case_dir / name).is_file() or sha(case_dir / name) != digest: result["errors"].append(f"hash_mismatch:{name}")
        for name in CASE_FILES:
            if not (case_dir / name).is_file(): result["errors"].append(f"missing:{name}")
        with np.load(case_dir / "latents.npz") as z, np.load(case_dir / "maps.npz") as maps, np.load(case_dir / "relighting.npz") as relight:
            for key in LATENT_KEYS.values():
                if key not in z.files: result["errors"].append(f"missing_latent:{key}")
            for key in MAP_KEYS:
                if key not in maps.files: result["errors"].append(f"missing_map:{key}")
            numeric = [z[k] for k in z.files] + [maps[k] for k in maps.files] + [relight[k] for k in relight.files if np.issubdtype(relight[k].dtype, np.number)]
            result["finite"] = bool(all(np.isfinite(x).all() for x in numeric))
            if not result["finite"]: result["errors"].append("nonfinite")
            if "preset_names" not in relight.files or "relighted_images" not in relight.files or len(relight["preset_names"]) != 6: result["errors"].append("incomplete_relighting")
        result["success"] = not result["errors"] and marker.get("validation_passed") is True
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}:{exc}")
    return result


def run(root: Path, config_path: Path) -> dict[str, Any]:
    import yaml
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")); out = root / cfg["output_root"]
    manifest = out / "manifests/p1_deca_input_manifest.csv"
    if not manifest.is_file(): raise FileNotFoundError("P1 input manifest is missing; run generation or validate first")
    with manifest.open(newline="", encoding="utf-8") as handle: rows = list(csv.DictReader(handle))
    checks = [validate_case(out / "cases" / row["case_id"]) for row in rows]
    audit = out / "qc" / "offline_integrity_audit.json"; audit.parent.mkdir(parents=True, exist_ok=True)
    summary = {"expected_cases": len(rows), "success_cases": sum(x["success"] for x in checks), "failed_cases": sum(not x["success"] for x in checks), "all_finite_cases": sum(bool(x.get("finite")) for x in checks), "contains_label_or_exif_probe": False, "case_checks": checks}
    audit.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
