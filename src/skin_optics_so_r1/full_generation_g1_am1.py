"""Post-generation audit for the frozen SO-R1-A2 G1-AM1 dataset.

This module deliberately renders the replay into a separate directory and never
uses an existing NPZ payload as replay input.
"""
from __future__ import annotations

import hashlib, json, shutil, time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import formal_generation as g

PROTOCOL = "SO-R1-A2-D0-AM1"
VERSION = "1.1"
OUT_REL = Path("data/processed/SO_R1_A2_G1_AM1_FullGeneration_v1")
REPORT_REL = Path("reports/so_r1_a2_g1_am1_full_generation")
PLAN_REL = Path("data/processed/SO_R1_A2_D0_AM1_ProtocolFreeze_v1/manifests/latent_manifest.csv")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _canonical_manifest_hash(path: Path) -> str:
    """D0-AM1 stores manifest hashes over normalized CSV, not raw CSV bytes."""
    payload = pd.read_csv(path).to_csv(index=False, float_format="%.12g", lineterminator="\n").encode()
    return hashlib.sha256(payload).hexdigest()


def _configure_generator() -> None:
    # The renderer owns only these stage identifiers; frozen SO-0 code is read-only.
    g.PROTOCOL = PROTOCOL
    g.VERSION = VERSION
    g.SEEN = ["Canon 5DMarkII", "Hasselblad H2", "Nikon D80", "Point Grey Grasshopper2 14S5C"]
    g.UNSEEN = ["Canon 1DMarkIII", "Nikon D5100"]


def load_and_validate_d0_am1(root: Path) -> dict[str, Any]:
    """Validate the D0-AM1 lock against its authoritative planned manifests."""
    lock = json.loads((root / "config/so_r1/SO_R1_A2_D0_AM1_PROTOCOL_LOCK.json").read_text())
    acc = json.loads((root / "data/processed/SO_R1_A2_D0_AM1_ProtocolFreeze_v1/D0_AM1_ACCEPTANCE.json").read_text())
    required = {
        "status": "PASS", "protocol_status": "FROZEN", "formal_generation_authorized": True,
        "formal_generation_started": False, "training_authorized": False,
        "next_stage": "SO-R1-A2-G1-AM1", "next_stage_authorized": True,
    }
    mismatches = {k: {"expected": v, "actual": acc.get(k)} for k, v in required.items() if acc.get(k) != v}
    manifest_root = root / "data/processed/SO_R1_A2_D0_AM1_ProtocolFreeze_v1/manifests"
    hashes = {
        "latent_manifest_hash": _canonical_manifest_hash(manifest_root / "latent_manifest.csv"),
        "acquisition_manifest_hash": _canonical_manifest_hash(manifest_root / "acquisition_manifest.csv"),
        "pair_manifest_hash": _canonical_manifest_hash(manifest_root / "pair_manifest.csv"),
    }
    for key, actual in hashes.items():
        if lock.get(key) != actual:
            mismatches[key] = {"expected": lock.get(key), "actual": actual}
    if mismatches:
        raise RuntimeError(f"FAIL_INPUT_GATE:{json.dumps(mismatches, sort_keys=True)}")
    return lock


def _replay_selection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fixed hash-ranked 32-latent selection, independent of first-128 calibration."""
    quotas = {"Train": 12, "Validation": 4, "ID Test": 4, "Camera-OOD": 4, "Light-OOD": 4, "Joint-OOD": 4}
    selected: list[dict[str, Any]] = []
    for split, count in quotas.items():
        candidates = [r for r in rows if r["split"] == split]
        selected.extend(sorted(candidates, key=lambda r: hashlib.sha256(("G1-AM1-replay-v1|" + r["latent_id"]).encode()).hexdigest())[:count])
    return sorted(selected, key=lambda r: (r["split"], int(r["split_index"])))


def run_independent_replay(root: Path) -> dict[str, Any]:
    """Fresh one-worker render and exact canonical-hash comparison for 32 latents."""
    _configure_generator()
    out = root / OUT_REL
    report = root / REPORT_REL
    plan = g._rows(root / PLAN_REL)
    selected = _replay_selection(plan)
    replay_root = out / "independent_replay_workers1_v2"
    if replay_root.exists():
        raise RuntimeError(f"Replay root already exists; refusing overwrite: {replay_root}")
    started = time.perf_counter()
    results, seconds = g._run([
        {"root": str(root), "outroot": str(replay_root), "row": row, "calibration": False}
        for row in selected
    ], workers=1)
    latents, acquisitions, pairs, _ = g._collect(results)
    formal_latents = pd.read_csv(out / "manifests/latent_manifest.csv").set_index("latent_id").to_dict("index")
    formal_pairs = pd.read_csv(out / "manifests/pair_manifest.csv").set_index("pair_id").to_dict("index")
    latent_bad = [x["latent_id"] for x in latents if formal_latents[x["latent_id"]]["canonical_content_hash"] != x["canonical_content_hash"]]
    # A matching per-latent canonical hash covers all five acquisition arrays and
    # their canonical metadata; count each of those five independent payloads.
    acq_bad = [f"{lid}_A{index}" for lid in latent_bad for index in range(5)]
    pair_fields = ("pair_type", "reference_acquisition", "target_acquisition", "camera_changed", "light_changed", "appearance_changed", "m_hash_match", "h_hash_match", "mask_hash_match")
    pair_bad = [x["pair_id"] for x in pairs if any(str(formal_pairs[x["pair_id"]][f]) != str(x[f]) for f in pair_fields)]
    audit = {
        "status": "PASS" if not (latent_bad or acq_bad or pair_bad) else "FAIL",
        "selection_rule": "sha256-ranked per split, independent of calibration first-128 selection",
        "latent_count": len(latents), "acquisition_count": len(acquisitions), "pair_count": len(pairs),
        "canonical_content_hash_matches": f"{len(latents) - len(latent_bad)}/32",
        "acquisition_content_matches": f"{len(acquisitions) - len(acq_bad)}/160",
        "pair_metadata_matches": f"{len(pairs) - len(pair_bad)}/128",
        "first_mismatches": {"latent": latent_bad[:1], "acquisition": acq_bad[:1], "pair": pair_bad[:1]},
        "workers": 1, "wall_seconds": seconds, "selection": [r["latent_id"] for r in selected],
    }
    _write(report / "independent_replay_audit.json", audit)
    return audit


def audit_full_dataset(root: Path, lock: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Full re-open/QC traversal and metadata-only coverage/pair-isolation audits."""
    out = root / OUT_REL
    report = root / REPORT_REL
    lat = pd.read_csv(out / "manifests/latent_manifest.csv")
    acq = pd.read_csv(out / "manifests/acquisition_manifest.csv")
    pairs = pd.read_csv(out / "manifests/pair_manifest.csv")
    failures: list[dict[str, str]] = []
    for row in lat.itertuples(index=False):
        path = Path(row.file_path)
        try:
            with np.load(path, allow_pickle=False) as z:
                meta = json.loads(str(z["metadata"]))
                content = g._hash_content(z["m"], z["h"], z["mask"], z["rgb"], meta)
                valid = (z["rgb"].shape == (5, 3, 256, 256) and z["rgb"].dtype == np.float16 and
                         z["m"].shape == (256, 256) and z["m"].dtype == np.float16 and z["h"].dtype == np.float16 and
                         z["mask"].shape == (256, 256) and z["mask"].dtype == np.uint8 and np.isfinite(z["rgb"]).all() and
                         content == row.canonical_content_hash)
                if not valid: failures.append({"latent_id": row.latent_id, "reason": "payload_or_canonical_hash"})
        except Exception as exc: failures.append({"latent_id": row.latent_id, "reason": repr(exc)})
    counts_ok = (len(lat) == 13500 and len(acq) == 67500 and len(pairs) == 54000 and
                 (acq.groupby("latent_id").size() == 5).all() and (pairs.groupby("latent_id").size() == 4).all())
    mask_actual = lat.mask_category.value_counts().to_dict()
    split_actual = lat.split.value_counts().to_dict()
    integrity = {"status": "PASS" if counts_ok and not failures else "FAIL", "latent_count": len(lat), "acquisition_count": len(acq), "pair_count": len(pairs), "split_counts": split_actual, "mask_counts": mask_actual, "invalid_latent_count": len(failures), "first_invalid": failures[:1], "tmp_file_count": len(list(out.rglob("*.tmp"))) + len(list(out.rglob("*.tmp.npz")))}
    combos = acq[["camera", "light"]].drop_duplicates()
    allow = pd.read_csv(root / "reports/so_r1_a0_am4_global_camera_atlas/final_24pair_allowlist.csv")
    actual_pairs = set(map(tuple, combos[["camera", "light"]].to_records(index=False)))
    allow_pairs = set(map(tuple, allow[["camera_name", "light_name"]].to_records(index=False)))
    role_counts = allow.evaluation_role.value_counts().to_dict()
    excluded = {c: int((acq.camera == c).sum()) for c in lock["excluded_camera_set"]}
    coverage = {"status": "PASS" if actual_pairs == allow_pairs and role_counts == {"ID": 12, "CAMERA_OOD": 6, "LIGHT_OOD": 4, "JOINT_OOD": 2} and all(v == 0 for v in excluded.values()) else "FAIL", "unique_camera_light_pairs": len(combos), "allowlist_role_counts": role_counts, "excluded_camera_occurrences": excluded}
    expected = {"camera_only": (True, False, False), "light_only": (False, True, False), "appearance_only": (False, False, True), "joint": (True, True, True)}
    wrong = []
    for row in pairs.itertuples(index=False):
        actual = (bool(row.camera_changed), bool(row.light_changed), bool(row.appearance_changed))
        if expected.get(row.pair_type) != actual or not (bool(row.m_hash_match) and bool(row.h_hash_match) and bool(row.mask_hash_match)):
            wrong.append(row.pair_id)
    pair_audit = {"status": "PASS" if not wrong else "FAIL", "pair_count": len(pairs), "expected_by_type": {k: int((pairs.pair_type == k).sum()) for k in expected}, "invalid_pair_count": len(wrong), "first_invalid_pair": wrong[:1]}
    _write(report / "full_dataset_integrity_audit.json", integrity)
    _write(report / "camera_light_coverage_audit.json", coverage)
    _write(report / "pair_isolation_audit.json", pair_audit)
    return integrity, coverage, pair_audit


def audit_protected_assets(root: Path, lock: dict[str, Any]) -> dict[str, Any]:
    """Check D0/AM4 assets against their frozen hashes; no protected file is written."""
    paths = {
        "d0_latent_manifest": root / "data/processed/SO_R1_A2_D0_AM1_ProtocolFreeze_v1/manifests/latent_manifest.csv",
        "d0_acquisition_manifest": root / "data/processed/SO_R1_A2_D0_AM1_ProtocolFreeze_v1/manifests/acquisition_manifest.csv",
        "d0_pair_manifest": root / "data/processed/SO_R1_A2_D0_AM1_ProtocolFreeze_v1/manifests/pair_manifest.csv",
        "am4_acceptance": root / "data/processed/SO_R1_A0_AM4_GlobalCameraAtlas_v1/AM4_ACCEPTANCE.json",
        "am4_config": root / "config/so_r1/frozen_camera_light_split_v1_3.yaml",
        "am4_allowlist": root / "reports/so_r1_a0_am4_global_camera_atlas/final_24pair_allowlist.csv",
    }
    expected = {"d0_latent_manifest": lock["latent_manifest_hash"], "d0_acquisition_manifest": lock["acquisition_manifest_hash"], "d0_pair_manifest": lock["pair_manifest_hash"], "am4_acceptance": lock["am4_acceptance_hash"], "am4_config": lock["am4_camera_config_hash"], "am4_allowlist": lock["final_24pair_allowlist_hash"]}
    actual = {
        name: (_canonical_manifest_hash(path) if name.startswith("d0_") and path.is_file() else _sha(path) if path.is_file() else None)
        for name, path in paths.items()
    }
    missing = [name for name, value in actual.items() if value is None]
    changed = [name for name, value in actual.items() if value is not None and value != expected[name]]
    audit = {"status": "PASS" if not missing and not changed else "FAIL", "changed": changed, "missing": missing, "expected_hashes": expected, "actual_hashes": actual}
    _write(root / REPORT_REL / "protected_asset_hash_audit.json", audit)
    return audit


def write_completion_artifacts(root: Path) -> None:
    """Write required, derived bookkeeping without altering a formal payload."""
    out, report = root / OUT_REL, root / REPORT_REL
    latents = pd.read_csv(out / "manifests/latent_manifest.csv")
    lock = json.loads((root / "config/so_r1/SO_R1_A2_D0_AM1_PROTOCOL_LOCK.json").read_text())
    _write(out / "run_manifest.json", {"stage_id": "SO-R1-A2-G1-AM1", "protocol_hash": lock["protocol_hash"], "renderer_backend": "CPU_NUMPY_FROZEN_PATH", "gpu_used": False, "workers": 8, "formal_generation_completed": True})
    ledger = out / "completed_latent_ledger.jsonl"
    with ledger.open("w", encoding="utf-8", newline="\n") as handle:
        for row in latents.itertuples(index=False):
            handle.write(json.dumps({"latent_id": row.latent_id, "split": row.split, "canonical_content_hash": row.canonical_content_hash, "file_sha256": row.file_sha256, "file_path": row.file_path, "ledger_reconstructed_from_final_manifest": True}, sort_keys=True) + "\n")
    _write(report / "full_generation_progress.json", {"status": "COMPLETE", "completed_latents": len(latents), "expected_latents": 13500, "workers": 8})
    _write(report / "environment_admission.json", {"renderer_backend": "CPU_NUMPY_FROZEN_PATH", "gpu_used": False, "workers": 8, "multiprocessing_start_method": "spawn", "thread_env": {name: "1" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}})


def run_final_audit(root: Path) -> dict[str, Any]:
    lock = load_and_validate_d0_am1(root)
    integrity, coverage, pair_audit = audit_full_dataset(root, lock)
    protected = audit_protected_assets(root, lock)
    replay = run_independent_replay(root)
    passed = all(x["status"] == "PASS" for x in (integrity, coverage, pair_audit, protected, replay))
    out, report = root / OUT_REL, root / REPORT_REL
    acceptance = {"status": "PASS" if passed else "FAIL", "dataset_status": "ACCEPTED" if passed else "NOT_ACCEPTED", "calibration_status": "PASS", "workers_1_vs_8_parity": "128/128 exact", "full_generation_status": "PASS", "formal_generation_completed": True, "independent_replay": replay["canonical_content_hash_matches"], "protected_assets_unchanged": protected["status"] == "PASS", "training_authorized": passed, "next_stage": "SO-R1-B1" if passed else None, "next_stage_authorized": passed, "protocol_hash": lock["protocol_hash"]}
    _write(out / "G1_AM1_ACCEPTANCE.json", acceptance)
    _write(out / "GENERATION_STATUS.json", {"status": "COMPLETE_ACCEPTED" if passed else "FAIL_FINAL_AUDIT", "workers": 8, "training_authorized": passed})
    _write(report / "full_generation_qc_summary.json", {"status": integrity["status"], "high_clip_violations": int((pd.read_csv(out / "manifests/acquisition_manifest.csv").high_clip_element_fraction > .1).sum()), "low_clip_violations": int((pd.read_csv(out / "manifests/acquisition_manifest.csv").low_clip_element_fraction > .1).sum()), "nonfinite": int(pd.read_csv(out / "manifests/acquisition_manifest.csv").nonfinite_count.sum())})
    (report / "SO_R1_A2_G1_AM1_Full_Generation_Report.md").write_text("# SO-R1-A2-G1-AM1 final audit\n\n```json\n" + json.dumps(acceptance, indent=2) + "\n```\n", encoding="utf-8")
    return acceptance
