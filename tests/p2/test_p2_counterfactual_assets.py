from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p2_counterfactual.assets import (
    P2_ROOT,
    ROOT,
    P2CounterfactualDataset,
    deca_add_shlight_numpy,
    derive_relighted_shading,
    get_preset_index,
)
from p2_counterfactual.path_utils import resolve_project_path


def test_preset_lookup_uses_names() -> None:
    names = ["neutral_front", "left", "right", "top", "dim_front", "bright_front"]
    shuffled = ["right", "neutral_front", "bright_front", "left", "top", "dim_front"]
    assert get_preset_index(shuffled, "left") == 3
    assert get_preset_index(names, "bright_front") == 5
    with pytest.raises(KeyError):
        get_preset_index(names, "missing")
    with pytest.raises(ValueError):
        get_preset_index(["left", "left"], "left")


def test_deca_add_shlight_shape_dtype_finite_and_deterministic() -> None:
    rng = np.random.default_rng(2026)
    normal = rng.normal(size=(224, 224, 3)).astype(np.float32)
    coeff = rng.normal(size=(9, 3)).astype(np.float32)
    a = deca_add_shlight_numpy(normal, coeff)
    b = deca_add_shlight_numpy(normal, coeff)
    assert a.shape == (224, 224, 3)
    assert a.dtype == np.float32
    assert np.isfinite(a).all()
    assert np.array_equal(a, b)


def test_derive_relighted_shading_contract() -> None:
    normal = np.zeros((224, 224, 3), dtype=np.float32)
    normal[..., 2] = 1.0
    coeff = np.ones((6, 9, 3), dtype=np.float32)
    out = derive_relighted_shading(normal, coeff)
    assert out.shape == (6, 224, 224, 3)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_dataset_smoke_from_minimal_npz(tmp_path: Path) -> None:
    maps_path = tmp_path / "maps.npz"
    relight_path = tmp_path / "relighting.npz"
    shading_path = tmp_path / "relit_shading.npz"
    names = np.asarray(["neutral_front", "left", "right", "top", "dim_front", "bright_front"])
    np.savez(
        maps_path,
        input_aligned_rgb=np.ones((224, 224, 3), np.float32),
        shading_like=np.ones((224, 224, 3), np.float32) * 0.5,
        signed_residual=np.ones((224, 224, 3), np.float32) * -0.1,
    )
    np.savez(relight_path, preset_names=names, relighted_images=np.ones((6, 224, 224, 3), np.float32))
    np.savez(shading_path, preset_names=names, relighted_shading=np.ones((6, 224, 224, 3), np.float32) * 0.2)
    frame = pd.DataFrame(
        [
            {
                "case_id": "case_a",
                "maps_npz_path": str(maps_path),
                "relighting_npz_path": str(relight_path),
                "relighted_shading_path": str(shading_path),
                "binary_label": 1,
                "fold": 2,
            }
        ]
    )
    original = P2CounterfactualDataset(frame, mode="original")[0]
    counter = P2CounterfactualDataset(frame, mode="counterfactual", preset_name="right")[0]
    assert original["original_rgb"].shape == (3, 224, 224)
    assert counter["relighted_rgb_k"].shape == (3, 224, 224)
    assert counter["relighted_shading_k"].shape == (3, 224, 224)
    assert counter["original_residual"].shape == (3, 224, 224)
    assert counter["preset_name"] == "right"


def test_protocol_json_when_generated() -> None:
    path = P2_ROOT / "metadata/P2_Relighting_Protocol_v1.json"
    if not path.is_file():
        pytest.skip("P2 protocol has not been generated")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["protocol_version"] == "v1"
    assert data["normal_source"] == "normal_coarse"
    assert data["source_renderer_function"] != "unknown"
    assert data["preset_names"] == ["neutral_front", "left", "right", "top", "dim_front", "bright_front"]
    json.dumps(data)


def test_generated_manifest_when_present() -> None:
    path = P2_ROOT / "manifests/p2_training_manifest.csv"
    if not path.is_file():
        pytest.skip("P2 manifest has not been generated")
    frame = pd.read_csv(path, dtype={"case_id": str})
    assert len(frame) == 500
    assert frame["case_id"].nunique() == 500
    assert int(frame["p2_training_ready"].sum()) == 500
    assert int(frame["p1_qc_flag"].sum()) == 16
    assert int(frame["boundary_uncertain"].sum()) == 2
    for column in ("maps_npz_path", "relighting_npz_path", "relighted_shading_path"):
        assert frame[column].map(lambda value: resolve_project_path(value, ROOT, require_exists=True).is_file()).all()


def test_generated_rgb_and_shading_preset_pairing_when_present() -> None:
    path = P2_ROOT / "manifests/p2_training_manifest.csv"
    if not path.is_file():
        pytest.skip("P2 manifest has not been generated")
    frame = pd.read_csv(path, dtype={"case_id": str}).head(12)
    for row in frame.itertuples():
        relighting_path = resolve_project_path(row.relighting_npz_path, ROOT, require_exists=True)
        relighted_shading_path = resolve_project_path(row.relighted_shading_path, ROOT, require_exists=True)
        with np.load(relighting_path, allow_pickle=False) as rel, np.load(relighted_shading_path, allow_pickle=False) as shade:
            rel_names = [str(x) for x in rel["preset_names"].tolist()]
            shade_names = [str(x) for x in shade["preset_names"].tolist()]
            assert rel_names == shade_names
            assert get_preset_index(rel_names, "bright_front") == shade_names.index("bright_front")
            assert shade["relighted_shading"].shape == (6, 224, 224, 3)
            assert shade["relighted_shading"].dtype == np.float32
            assert np.isfinite(shade["relighted_shading"]).all()
