"""Finalize an S1-2 target-domain development gate without overstating validation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def finalize_target_domain_profile(
    decision_path: str | Path,
    *,
    reviewer: str,
    notes: str,
) -> dict[str, Any]:
    """Approve a post-Validation target-domain profile for S1-3 development only."""

    if not reviewer.strip() or not notes.strip():
        raise ValueError("Reviewer and notes are required")
    decision_file = Path(decision_path).resolve()
    decision = _read_json(decision_file)
    if decision.get("stage") != "S1-2" or decision.get("split") != "valid":
        raise ValueError("Expected an S1-2 Validation target-profile decision")
    if decision.get("automatic_decision") != "PASS":
        raise ValueError("Cannot approve a target profile that failed automatic QC")
    if decision.get("manual_review", {}).get("status") != "pending":
        raise ValueError("Expected a pending target-profile manual review")
    evidence_policy = decision.get("evidence_policy", {})
    if evidence_policy.get("independent_validation_claim_allowed") is not False:
        raise ValueError("Target profile must explicitly prohibit an independent-validation claim")
    if evidence_policy.get("permitted_use") != "development_gate_for_s1_3":
        raise ValueError("Target profile is not authorized as an S1-3 development gate")

    source_valid_decision_file = Path(decision["source_valid_decision_path"]).resolve()
    if sha256_file(source_valid_decision_file) != decision["source_valid_decision_sha256"]:
        raise ValueError("Source Validation decision hash has changed")
    source_valid_decision = _read_json(source_valid_decision_file)
    valid_manifest_file = Path(decision["manifest_path"]).resolve()
    if sha256_file(valid_manifest_file) != decision["manifest_sha256"]:
        raise ValueError("Validation manifest hash has changed")
    if source_valid_decision.get("manifest_sha256") != decision["manifest_sha256"]:
        raise ValueError("Target profile and source Validation decision disagree")

    frozen_file = Path(decision["frozen_train_protocol_provenance"]).resolve()
    frozen = _read_json(frozen_file)
    if frozen.get("status") != "FROZEN" or frozen.get("test_access_count") != 0:
        raise ValueError("Invalid frozen Train provenance")
    train_decision_file = Path(frozen["source_train_decision"]).resolve()
    train_manifest_file = Path(frozen["source_train_manifest"]).resolve()
    if sha256_file(train_decision_file) != frozen["source_train_decision_sha256"]:
        raise ValueError("Frozen Train decision hash has changed")
    if sha256_file(train_manifest_file) != frozen["source_train_manifest_sha256"]:
        raise ValueError("Frozen Train manifest hash has changed")

    train_frame = pd.read_parquet(train_manifest_file)
    valid_frame = pd.read_parquet(valid_manifest_file)
    if set(train_frame["split"].unique()) != {"train"} or set(valid_frame["split"].unique()) != {"valid"}:
        raise ValueError("Train/Validation split labels are invalid")
    if set(train_frame["sample_id"]) & set(valid_frame["sample_id"]):
        raise ValueError("Train/Validation sample overlap detected")
    if set(train_frame["subject_id"]) & set(valid_frame["subject_id"]):
        raise ValueError("Train/Validation subject leakage detected")
    if list(train_frame.columns) != list(valid_frame.columns):
        raise ValueError("Train/Validation mask manifest schemas differ")

    combined = pd.concat([train_frame, valid_frame], ignore_index=True)
    target = decision["target_domain"]
    target_flag = combined["expression"].eq(target["expression"]) & combined["direction"].eq(target["direction"])
    combined["s1_2_target_domain_flag"] = target_flag
    combined["s1_2_usable_flag"] = ~combined["status"].eq("FAIL")
    combined["s1_2_analysis_role"] = np.select(
        [
            combined["split"].eq("train") & target_flag,
            combined["split"].eq("train") & ~target_flag,
            combined["split"].eq("valid") & target_flag,
        ],
        ["primary_development", "stress_development", "primary_validation"],
        default="stress_validation",
    )
    combined["s1_2_gate_profile"] = decision["profile_id"]

    output_root = decision_file.parents[2]
    final_manifest = output_root / "manifests" / "mask_manifest.parquet"
    final_manifest_csv = final_manifest.with_suffix(".csv")
    final_decision_file = output_root / "freeze" / "s1_2_target_domain_final_decision.json"
    for output in (final_manifest, final_manifest_csv, final_decision_file):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite final S1-2 artifact: {output}")

    reviewed_utc = datetime.now(timezone.utc).isoformat()
    review_path = Path(decision["manual_review"]["path"]).resolve()
    review = pd.read_csv(review_path, keep_default_na=False)
    review["reviewer_decision"] = "PASS_FOR_DEVELOPMENT"
    review["notes"] = notes
    review["reviewer"] = reviewer
    review["reviewed_utc"] = reviewed_utc
    review.to_csv(review_path, index=False, encoding="utf-8-sig")

    final_manifest.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(final_manifest, index=False)
    combined.to_csv(final_manifest_csv, index=False, encoding="utf-8-sig")
    decision["manual_review"] = {
        **decision["manual_review"],
        "status": "PASS_FOR_DEVELOPMENT",
        "reviewer": reviewer,
        "notes": notes,
        "reviewed_utc": reviewed_utc,
    }
    decision["status"] = "PASS_FOR_DEVELOPMENT"
    decision["next_stage_allowed"] = True
    decision["authorized_next_stage"] = "S1-3"
    decision["independent_validation_claim_allowed"] = False
    decision["final_manifest_path"] = str(final_manifest)
    decision["final_manifest_sha256"] = sha256_file(final_manifest)
    decision["final_decision_path"] = str(final_decision_file)
    _write_json(decision_file, decision)

    role_counts = combined["s1_2_analysis_role"].value_counts().to_dict()
    stress_failures = combined.loc[
        combined["s1_2_analysis_role"].isin(["stress_development", "stress_validation"])
        & combined["status"].eq("FAIL"),
        "sample_id",
    ].tolist()
    final_decision = {
        "schema_version": 1,
        "stage": "S1-2",
        "status": "PASS_FOR_DEVELOPMENT",
        "created_utc": reviewed_utc,
        "next_stage_allowed": True,
        "authorized_next_stage": "S1-3",
        "profile_id": decision["profile_id"],
        "target_domain": target,
        "evidence_policy": evidence_policy,
        "independent_validation_claim_allowed": False,
        "reviewer": reviewer,
        "notes": notes,
        "target_profile_decision_path": str(decision_file),
        "target_profile_decision_sha256": sha256_file(decision_file),
        "frozen_train_protocol_provenance_path": str(frozen_file),
        "frozen_train_protocol_provenance_sha256": sha256_file(frozen_file),
        "manifest_path": str(final_manifest),
        "manifest_sha256": sha256_file(final_manifest),
        "manifest_csv_path": str(final_manifest_csv),
        "manifest_csv_sha256": sha256_file(final_manifest_csv),
        "sample_counts": {
            "train": int(len(train_frame)),
            "valid": int(len(valid_frame)),
            "total": int(len(combined)),
        },
        "analysis_role_counts": {key: int(value) for key, value in role_counts.items()},
        "stress_failure_sample_ids": stress_failures,
        "test_access_count": 0,
    }
    final_decision_file.parent.mkdir(parents=True, exist_ok=True)
    _write_json(final_decision_file, final_decision)
    return final_decision
