from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from skin_optics_hsi.s1_target_profile import finalize_target_domain_profile, sha256_file


def test_finalize_target_profile_marks_roles_and_preserves_claim_boundary(tmp_path: Path) -> None:
    root = tmp_path / "stage1"
    masks = root / "masks" / "r5"
    manifests = root / "manifests"
    freeze = root / "freeze"
    masks.mkdir(parents=True)
    manifests.mkdir(parents=True)
    freeze.mkdir(parents=True)
    columns = ["sample_id", "subject_id", "split", "expression", "direction", "status"]
    train = pd.DataFrame(
        [
            ["p001_neutral_front", "p001", "train", "neutral", "front", "PASS"],
            ["p001_smile_left", "p001", "train", "smile", "left", "PASS"],
        ],
        columns=columns,
    )
    valid = pd.DataFrame(
        [
            ["p101_neutral_front", "p101", "valid", "neutral", "front", "PASS"],
            ["p101_smile_right", "p101", "valid", "smile", "right", "FAIL"],
        ],
        columns=columns,
    )
    train_manifest = manifests / "mask_manifest_train_r2.parquet"
    valid_manifest = manifests / "mask_manifest_valid_r2.parquet"
    train.to_parquet(train_manifest, index=False)
    valid.to_parquet(valid_manifest, index=False)
    train_decision_file = root / "masks" / "r2" / "s1_2_train_decision.json"
    train_decision_file.parent.mkdir(parents=True)
    train_decision_file.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    frozen_file = freeze / "s1_2_mask_protocol_provenance.json"
    frozen_file.write_text(
        json.dumps(
            {
                "status": "FROZEN",
                "test_access_count": 0,
                "source_train_decision": str(train_decision_file),
                "source_train_decision_sha256": sha256_file(train_decision_file),
                "source_train_manifest": str(train_manifest),
                "source_train_manifest_sha256": sha256_file(train_manifest),
            }
        ),
        encoding="utf-8",
    )
    source_valid_decision_file = root / "masks" / "r2" / "s1_2_valid_decision.json"
    source_valid_decision_file.write_text(
        json.dumps({"manifest_sha256": sha256_file(valid_manifest)}), encoding="utf-8"
    )
    review_file = masks / "manual_review_valid_target_front_neutral.csv"
    pd.DataFrame([{"sample_id": "p101_neutral_front"}]).to_csv(review_file, index=False)
    decision_file = masks / "s1_2_valid_target_front_neutral_decision.json"
    decision_file.write_text(
        json.dumps(
            {
                "stage": "S1-2",
                "split": "valid",
                "profile_id": "target_front_neutral_v3",
                "automatic_decision": "PASS",
                "manual_review": {"status": "pending", "path": str(review_file)},
                "target_domain": {"expression": "neutral", "direction": "front"},
                "evidence_policy": {
                    "independent_validation_claim_allowed": False,
                    "permitted_use": "development_gate_for_s1_3",
                },
                "source_valid_decision_path": str(source_valid_decision_file),
                "source_valid_decision_sha256": sha256_file(source_valid_decision_file),
                "manifest_path": str(valid_manifest),
                "manifest_sha256": sha256_file(valid_manifest),
                "frozen_train_protocol_provenance": str(frozen_file),
            }
        ),
        encoding="utf-8",
    )
    result = finalize_target_domain_profile(
        decision_file,
        reviewer="QinghaoLi",
        notes="Approve for S1-3 development; not independent validation.",
    )
    combined = pd.read_parquet(manifests / "mask_manifest.parquet")
    assert result["status"] == "PASS_FOR_DEVELOPMENT"
    assert result["independent_validation_claim_allowed"] is False
    assert result["stress_failure_sample_ids"] == ["p101_smile_right"]
    assert set(combined["s1_2_analysis_role"]) == {
        "primary_development",
        "stress_development",
        "primary_validation",
        "stress_validation",
    }
    assert result["test_access_count"] == 0
