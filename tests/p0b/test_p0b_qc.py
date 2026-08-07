from pathlib import Path

import pandas as pd

from p0b_deca.qc import create_blind_review


def test_blind_review_contains_only_anonymous_audit_ids(tmp_path: Path):
    path = tmp_path / "blind_review.csv"
    create_blind_review(path, ["P0B-001", "P0B-002"])
    table = pd.read_csv(path)
    assert table.audit_id.tolist() == ["P0B-001", "P0B-002"]
    forbidden = {"sample_id", "patient_group_id", "camera_make", "binary_name", "NYHA", "sex"}
    assert not forbidden.intersection(table.columns)
