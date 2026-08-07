from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from p2_counterfactual.path_utils import normalize_path_for_comparison, resolve_project_path


def test_resolve_project_path_supports_wsl_windows_and_relative_formats(tmp_path: Path) -> None:
    project_root = tmp_path / "face2"
    target = project_root / "data" / "a.npz"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"npz-placeholder")
    known_roots = ["/mnt/e/projects/face2", "E:/projects/face2", r"E:\projects\face2"]
    stored_paths = [
        "/mnt/e/projects/face2/data/a.npz",
        "E:/projects/face2/data/a.npz",
        r"E:\projects\face2\data\a.npz",
        "data/a.npz",
    ]
    for stored in stored_paths:
        assert resolve_project_path(stored, project_root, known_project_roots=known_roots, require_exists=True) == target.resolve()


def test_resolve_project_path_error_contains_case_column_stored_and_resolved(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as exc:
        resolve_project_path(
            "/mnt/e/projects/face2/data/missing.npz",
            tmp_path / "face2",
            require_exists=True,
            case_id="case_x",
            column_name="maps_npz_path",
        )
    message = str(exc.value)
    assert "case_x" in message
    assert "maps_npz_path" in message
    assert "/mnt/e/projects/face2/data/missing.npz" in message
    assert "resolved_path" in message


def test_normalize_path_for_comparison_ignores_known_project_root_prefixes(tmp_path: Path) -> None:
    known_roots = ["/mnt/e/projects/face2", "E:/projects/face2", r"E:\projects\face2"]
    tails = {
        normalize_path_for_comparison(path, project_root=tmp_path / "face2", known_project_roots=known_roots)
        for path in [
            "/mnt/e/projects/face2/data/a.npz",
            "E:/projects/face2/data/a.npz",
            r"E:\projects\face2\data\a.npz",
        ]
    }
    assert tails == {"data/a.npz"}
