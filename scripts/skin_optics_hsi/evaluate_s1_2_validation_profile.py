"""Re-score an existing frozen-protocol Validation manifest for a target domain."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from skin_optics_hsi.s1_masks import _save_contact_sheet  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--valid-decision", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-revision", default="r3")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_decision_path = args.valid_decision.resolve()
    profile_path = args.profile.resolve()
    source_decision = json.loads(source_decision_path.read_text(encoding="utf-8"))
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    if source_decision.get("split") != "valid":
        raise ValueError("The source decision must be a Validation decision")
    if source_decision.get("automatic_decision") not in {"PASS", "REVISE"}:
        raise ValueError("Unexpected source Validation decision")
    manifest_path = Path(source_decision["manifest_path"]).resolve()
    if sha256_file(manifest_path) != source_decision["manifest_sha256"]:
        raise ValueError("Source Validation manifest hash has changed")
    if profile.get("stage") != "S1-2" or not str(profile.get("profile_id", "")).startswith("target_front_neutral_"):
        raise ValueError("Unexpected target-domain profile")

    frame = pd.read_parquet(manifest_path)
    target = profile["target_domain"]
    primary = frame.loc[
        frame["expression"].eq(target["expression"]) & frame["direction"].eq(target["direction"])
    ].copy()
    if len(primary) < int(profile["primary_gate"]["minimum_sample_count"]):
        raise ValueError("Target primary domain has too few Validation samples")
    primary["usable_flag"] = ~primary["status"].eq("FAIL")
    primary_counts = primary["status"].value_counts().to_dict()
    primary_failure_fraction = float((primary["status"] == "FAIL").mean())
    primary_review_fraction = float((primary["status"] == "REVIEW").mean())
    primary_usable_fraction = float(primary["usable_flag"].mean())
    gate = profile["primary_gate"]
    allowed_review_codes = set(gate.get("allowed_review_codes", []))
    review_code_sets = primary["review_codes"].fillna("").map(
        lambda value: {code for code in str(value).split("|") if code}
    )
    review_codes_allowed = bool(
        all(
            status != "REVIEW" or codes.issubset(allowed_review_codes)
            for status, codes in zip(primary["status"], review_code_sets)
        )
    )
    minimum_cheek_pixels = int(gate.get("minimum_each_cheek_pixels", 0))
    cheeks_usable = bool(
        (
            primary["left_cheek_visible"].eq(True)
            & primary["right_cheek_visible"].eq(True)
            & primary["left_cheek_pixels"].fillna(0).ge(minimum_cheek_pixels)
            & primary["right_cheek_pixels"].fillna(0).ge(minimum_cheek_pixels)
        ).all()
    )
    automatic_checks = {
        "primary_scope_nonempty": len(primary) >= int(gate["minimum_sample_count"]),
        "primary_failure_fraction": primary_failure_fraction <= float(gate["maximum_failure_fraction"]),
        "primary_review_fraction": primary_review_fraction <= float(gate["maximum_review_fraction"]),
        "primary_usable_fraction": primary_usable_fraction >= float(gate["minimum_usable_fraction"]),
        "primary_review_codes_allowed": review_codes_allowed,
        "primary_cheeks_usable": cheeks_usable if gate.get("require_both_cheeks_visible", False) else True,
    }
    stress = frame.loc[~frame.index.isin(primary.index)].copy()
    stress_counts = stress["status"].value_counts().to_dict()
    stress_strata = (
        stress.assign(usable_flag=~stress["status"].eq("FAIL"))
        .groupby(["expression", "direction"], as_index=False)
        .agg(
            samples=("sample_id", "size"),
            pass_count=("status", lambda values: int((values == "PASS").sum())),
            review_count=("status", lambda values: int((values == "REVIEW").sum())),
            fail_count=("status", lambda values: int((values == "FAIL").sum())),
            usable_fraction=("usable_flag", "mean"),
        )
    )
    automatic_pass = all(automatic_checks.values())
    output_root = args.output_root.resolve()
    revision_root = output_root / "masks" / args.output_revision
    if revision_root.exists():
        raise FileExistsError(f"Refusing to overwrite target-domain review revision: {revision_root}")
    revision_root.mkdir(parents=True)
    decision_path = revision_root / "s1_2_valid_target_front_neutral_decision.json"
    review_path = revision_root / "manual_review_valid_target_front_neutral.csv"
    contact_sheet_path = revision_root / "valid_target_front_neutral_contact_sheet.png"
    primary["reviewer_decision"] = ""
    primary["notes"] = ""
    primary.to_csv(review_path, index=False, encoding="utf-8-sig")
    panel_rows = primary.loc[primary["qc_panel_path"].notna()]
    if not panel_rows.empty:
        _save_contact_sheet(contact_sheet_path, panel_rows)
    stress_strata.to_csv(revision_root / "valid_stress_stratum_qc.csv", index=False, encoding="utf-8-sig")
    decision = {
        "schema_version": 1,
        "stage": "S1-2",
        "split": "valid",
        "output_revision": args.output_revision,
        "profile_id": profile["profile_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "AUTOMATIC_PASS_PRIMARY_DOMAIN_STRESS_NONBLOCKING_REVIEW" if automatic_pass else "REVISE",
        "automatic_decision": "PASS" if automatic_pass else "REVISE",
        "automatic_checks": automatic_checks,
        "target_domain": target,
        "primary_scope": {
            "expression": target["expression"],
            "direction": target["direction"],
            "sample_count": int(len(primary)),
            "sample_counts": {key: int(value) for key, value in primary_counts.items()},
            "failure_fraction": primary_failure_fraction,
            "review_fraction": primary_review_fraction,
            "usable_fraction": primary_usable_fraction,
            "allowed_review_codes": sorted(allowed_review_codes),
            "all_review_codes_allowed": review_codes_allowed,
            "both_cheeks_usable": cheeks_usable,
        },
        "stress_scope": {
            "sample_count": int(len(stress)),
            "sample_counts": {key: int(value) for key, value in stress_counts.items()},
            "failure_count": int((stress["status"] == "FAIL").sum()),
            "stratum_qc_path": str(revision_root / "valid_stress_stratum_qc.csv"),
            "role": profile["stress_domain"]["role"],
        },
        "source_valid_decision_path": str(source_decision_path),
        "source_valid_decision_sha256": sha256_file(source_decision_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": source_decision["manifest_sha256"],
        "profile_path": str(profile_path),
        "profile_sha256": sha256_file(profile_path),
        "frozen_train_protocol_provenance": source_decision["frozen_train_protocol_provenance"],
        "manual_review": {
            "status": "pending",
            "path": str(review_path),
            "contact_sheet_path": str(contact_sheet_path),
        },
        "protocol_frozen": True,
        "next_stage_allowed": False,
        "test_access_count": 0,
        "evidence_policy": profile.get("evidence_policy", {}),
    }
    decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0 if automatic_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
