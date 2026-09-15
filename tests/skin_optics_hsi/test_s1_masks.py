from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from skin_optics_hsi.s1_masks import (
    DEFAULT_MASK_CONFIG,
    build_illumination_layer,
    build_radiometric_layer,
    build_roi_geometry,
    build_semantic_layers,
    evaluate_sample_qc,
    finalize_s1_2_train_review,
    finalize_s1_2_validation_review,
    sha256_file,
    _select_manual_review_rows,
    _validate_upstream,
)
from skin_optics_real_preprocess import facemesh_regions


def _config() -> dict:
    return yaml.safe_load(Path(DEFAULT_MASK_CONFIG).read_text(encoding="utf-8"))


def _synthetic_landmarks(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    points = np.full((478, 2), [width / 2, height / 2], dtype=np.float32)
    angles = np.linspace(0, 2 * np.pi, len(facemesh_regions.FACE_OVAL_INDICES), endpoint=False)
    oval = np.column_stack([width / 2 + width * 0.32 * np.cos(angles), height / 2 + height * 0.42 * np.sin(angles)])
    points[np.asarray(facemesh_regions.FACE_OVAL_INDICES)] = oval
    points[np.asarray(facemesh_regions.IMAGE_LEFT_EYE_INDICES)] = [width * 0.38, height * 0.40]
    points[np.asarray(facemesh_regions.IMAGE_RIGHT_EYE_INDICES)] = [width * 0.62, height * 0.40]
    points[np.asarray(facemesh_regions.LEFT_BROW_POLYGON)] = [width * 0.38, height * 0.34]
    points[np.asarray(facemesh_regions.RIGHT_BROW_POLYGON)] = [width * 0.62, height * 0.34]
    points[facemesh_regions.IMAGE_LEFT_MOUTH_INDEX] = [width * 0.43, height * 0.67]
    points[facemesh_regions.IMAGE_RIGHT_MOUTH_INDEX] = [width * 0.57, height * 0.67]
    return points


def test_layered_masks_are_intersections_and_qc_passes() -> None:
    shape = (256, 256)
    config = _config()
    config["roi"]["minimum_whole_skin_pixels"] = 100
    config["roi"]["minimum_cheek_pixels"] = 20
    config["roi"]["minimum_forehead_pixels"] = 20
    landmarks = _synthetic_landmarks(shape)
    labels = np.zeros(shape, dtype=np.uint8)
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    face = ((xx - 128) / 80) ** 2 + ((yy - 128) / 105) ** 2 <= 1
    labels[face] = 1
    anatomical, semantic_valid, _ = build_semantic_layers(labels, landmarks, config)
    geometry = build_roi_geometry(shape, landmarks, config)
    cube = np.full((*shape, 31), 0.35, dtype=np.float32)
    radiometric_valid, parts = build_radiometric_layer(cube, config)
    rgb = np.full((*shape, 3), [150, 105, 85], dtype=np.uint8)
    illumination_valid, _ = build_illumination_layer(rgb, parts["broadband"], semantic_valid, config)
    final = anatomical & semantic_valid & radiometric_valid & illumination_valid
    regions = {
        "whole_skin": final,
        "left_cheek": final & geometry["left_cheek_geometry"],
        "right_cheek": final & geometry["right_cheek_geometry"],
        "forehead": final & geometry["forehead_geometry"],
    }
    qc = evaluate_sample_qc("front", anatomical, semantic_valid, radiometric_valid, illumination_valid, regions, config)
    assert qc["status"] == "PASS"
    assert np.all(regions["left_cheek"] <= regions["whole_skin"])
    assert np.all(regions["right_cheek"] <= regions["whole_skin"])
    assert qc["left_cheek_visible"] and qc["right_cheek_visible"]


def test_radiometric_and_illumination_reject_known_artifacts() -> None:
    config = _config()
    cube = np.full((32, 32, 31), 0.4, dtype=np.float32)
    cube[0, 0, :] = np.nan
    cube[1, 1, :] = 1.0
    cube[2, 2, :] = 0.0
    valid, parts = build_radiometric_layer(cube, config)
    assert not valid[0, 0]
    assert not valid[1, 1]
    assert not valid[2, 2]
    rgb = np.full((32, 32, 3), 120, dtype=np.uint8)
    rgb[3, 3, :] = 255
    broadband = np.nan_to_num(parts["broadband"], nan=0.0)
    broadband[4, 4] = 0.0
    illumination, diagnostics = build_illumination_layer(rgb, broadband, np.ones((32, 32), dtype=bool), config)
    assert not illumination[3, 3]
    assert not illumination[4, 4]
    assert diagnostics["rgb_specular"][3, 3]
    assert diagnostics["shadow"][4, 4]


def test_manual_review_selection_covers_each_pass_stratum() -> None:
    rows = []
    for expression in ("neutral", "smile"):
        for direction in ("front", "left", "right"):
            for index in range(4):
                rows.append(
                    {
                        "sample_id": f"{expression}_{direction}_{index}",
                        "expression": expression,
                        "direction": direction,
                        "status": "PASS",
                        "qc_panel_path": "panel.png",
                        "final_to_anatomical_fraction": 0.7 + index * 0.01,
                        "largest_component_fraction": 0.9,
                    }
                )
    rows.append(
        {
            "sample_id": "review_case",
            "expression": "neutral",
            "direction": "front",
            "status": "REVIEW",
            "qc_panel_path": "panel.png",
            "final_to_anatomical_fraction": 0.2,
            "largest_component_fraction": 0.4,
        }
    )
    selected = _select_manual_review_rows(pd.DataFrame(rows), maximum=12)
    assert "review_case" in set(selected["sample_id"])
    pass_strata = set(zip(selected.loc[selected["status"] == "PASS", "expression"], selected.loc[selected["status"] == "PASS", "direction"]))
    assert len(pass_strata) == 6


def test_upstream_requires_explicit_s1_1_authorization(tmp_path: Path) -> None:
    manifest = tmp_path / "split_manifest.csv"
    manifest.write_text("sample_id,split\np001,train\n", encoding="utf-8")
    contract = {"status": "PASS", "manifest": {"path": str(manifest), "sha256": sha256_file(manifest)}}
    registration = {
        "status": "PASS",
        "coordinate_mapping_frozen": True,
        "frozen_transform": "transpose",
        "input_manifest_sha256": contract["manifest"]["sha256"],
        "next_stage_allowed": False,
    }
    try:
        _validate_upstream(contract, registration)
    except ValueError as error:
        assert "authorize" in str(error)
    else:
        raise AssertionError("S1-2 accepted an unauthorized S1-1 decision")


def test_validation_finalize_builds_traceable_combined_manifest(tmp_path: Path) -> None:
    root = tmp_path / "stage1"
    masks = root / "masks" / "r2"
    manifests = root / "manifests"
    masks.mkdir(parents=True)
    manifests.mkdir(parents=True)
    train_manifest = manifests / "mask_manifest_train_r2.parquet"
    valid_manifest = manifests / "mask_manifest_valid_r2.parquet"
    columns = ["sample_id", "subject_id", "split", "status"]
    pd.DataFrame([["p001_neutral_front", "p001", "train", "PASS"]], columns=columns).to_parquet(
        train_manifest, index=False
    )
    pd.DataFrame([["p101_neutral_front", "p101", "valid", "PASS"]], columns=columns).to_parquet(
        valid_manifest, index=False
    )
    checkpoint = tmp_path / "parser.pth"
    checkpoint.write_bytes(b"checkpoint")
    config = tmp_path / "mask.yaml"
    config.write_text(yaml.safe_dump({"runtime": {"parser_checkpoint": str(checkpoint)}}), encoding="utf-8")
    contract = tmp_path / "contract.json"
    registration = tmp_path / "registration.json"
    contract.write_text("{}", encoding="utf-8")
    registration.write_text("{}", encoding="utf-8")
    train_review = masks / "manual_review_train.csv"
    pd.DataFrame([{"sample_id": "p001_neutral_front"}]).to_csv(train_review, index=False)
    train_decision_file = masks / "s1_2_train_decision.json"
    train_decision = {
        "split": "train",
        "automatic_decision": "PASS",
        "manual_review": {"status": "pending", "path": str(train_review)},
        "config_path": str(config),
        "config_sha256": sha256_file(config),
        "manifest_path": str(train_manifest),
        "manifest_sha256": sha256_file(train_manifest),
        "contract_path": str(contract),
        "contract_sha256": sha256_file(contract),
        "registration_path": str(registration),
        "registration_sha256": sha256_file(registration),
        "parser_checkpoint_sha256": sha256_file(checkpoint),
    }
    train_decision_file.write_text(json.dumps(train_decision), encoding="utf-8")
    finalized_train = finalize_s1_2_train_review(
        train_decision_file, reviewer="reviewer", approve=True, notes="Train masks reviewed"
    )
    assert finalized_train["validation_allowed"] is True
    frozen_provenance = root / "freeze" / "s1_2_mask_protocol_provenance.json"
    valid_review = masks / "manual_review_valid.csv"
    pd.DataFrame([{"sample_id": "p101_neutral_front"}]).to_csv(valid_review, index=False)
    valid_decision_file = masks / "s1_2_valid_decision.json"
    valid_decision = {
        "split": "valid",
        "automatic_decision": "PASS",
        "manual_review": {"status": "pending", "path": str(valid_review)},
        "manifest_path": str(valid_manifest),
        "manifest_sha256": sha256_file(valid_manifest),
        "frozen_train_protocol_provenance": str(frozen_provenance),
    }
    valid_decision_file.write_text(json.dumps(valid_decision), encoding="utf-8")
    finalized_valid = finalize_s1_2_validation_review(
        valid_decision_file, reviewer="reviewer", approve=True, notes="Validation masks reviewed"
    )
    combined = pd.read_parquet(root / "manifests" / "mask_manifest.parquet")
    assert finalized_valid["next_stage_allowed"] is True
    assert combined["split"].tolist() == ["train", "valid"]
    final_decision = json.loads((root / "freeze" / "s1_2_final_decision.json").read_text(encoding="utf-8"))
    assert final_decision["status"] == "PASS"
    assert final_decision["sample_counts"] == {"train": 1, "valid": 1, "total": 2}
