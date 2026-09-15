from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from skin_optics_so_r1.camera_light_selection import (  # noqa: E402
    _join_audit,
    apply_candidate_scope,
    apply_quality_gate,
    audit_lights,
    audit_selection_sensitivity,
    build_camera_light_allowlist,
    compute_sam_distance_matrix,
    compute_tv_distance_matrix,
    load_config,
    normalize_channel_response,
    read_inputs,
    select_seen_cameras,
    select_unseen_cameras,
    validate_inputs,
)


def _state():
    config = load_config(ROOT / "config/so_r1/camera_light_selection_v1.yaml")
    inputs = read_inputs(ROOT, config)
    inventory = _join_audit(inputs)
    inventory, pairs = apply_quality_gate(inputs, inventory)
    inventory, candidates = apply_candidate_scope(inventory, config["scope_exclusions"])
    normalized = normalize_channel_response(inputs.responses)
    sam = compute_sam_distance_matrix(inputs.cameras, normalized)
    tv = compute_tv_distance_matrix(inputs.cameras, normalized)
    return config, inputs, inventory, pairs, candidates, sam, tv


def test_native_input_identity_and_light_validation():
    result = validate_inputs(ROOT, load_config(ROOT / "config/so_r1/camera_light_selection_v1.yaml"))
    assert result["status"] == "PASS"
    assert result["camera_identities"] == 28
    assert result["quality_qualified_cameras"] == 27
    assert result["consumer_mobile_candidates"] == 24
    assert result["lights"] == ["D65", "A", "FL2", "FL11"]


def test_distance_matrices_are_named_symmetric_and_order_invariant():
    _, inputs, _, _, _, sam, tv = _state()
    for matrix in (sam, tv):
        assert matrix.index.tolist() == matrix.columns.tolist()
        assert np.isfinite(matrix.to_numpy()).all()
        assert (matrix.to_numpy() >= 0).all()
        assert np.allclose(matrix.to_numpy(), matrix.to_numpy().T)
        assert np.allclose(np.diag(matrix), 0)
    reverse = list(reversed(range(len(inputs.cameras))))
    reversed_sam = compute_sam_distance_matrix(
        [inputs.cameras[i] for i in reverse], normalize_channel_response(inputs.responses[reverse])
    )
    assert np.allclose(sam.sort_index().sort_index(axis=1), reversed_sam.sort_index().sort_index(axis=1))


def test_scope_and_deterministic_sam_selection():
    config, _, inventory, _, candidates, sam, tv = _state()
    assert len(candidates) == 24
    assert "Point Grey Grasshopper 50S5C" not in candidates
    assert not set(config["scope_exclusions"]) & set(candidates)
    assert inventory.loc[inventory.camera_name == "Point Grey Grasshopper 50S5C", "quality_gate_pass"].item() is False
    seen, trace = select_seen_cameras(candidates, sam, config["forced_seen_anchors"], config["tie_tolerance"])
    unseen, unseen_trace = select_unseen_cameras(candidates, sam, seen, config["tie_tolerance"])
    assert seen[:2] == config["forced_seen_anchors"]
    assert len(seen) == 6 and len(unseen) == 2 and not set(seen) & set(unseen)
    assert len(trace) == 4 and len(unseen_trace) == 2
    seen_again, _ = select_seen_cameras(list(reversed(candidates)), sam, config["forced_seen_anchors"], config["tie_tolerance"])
    unseen_again, _ = select_unseen_cameras(list(reversed(candidates)), sam, seen_again, config["tie_tolerance"])
    assert seen == seen_again and unseen == unseen_again
    sensitivity, _, _ = audit_selection_sensitivity(candidates, sam, tv, config["forced_seen_anchors"], config["tie_tolerance"])
    assert sensitivity["anchors_preserved"]


def test_32_pair_allowlist_has_required_ood_partitions():
    config, _, _, pairs, candidates, sam, _ = _state()
    seen, _ = select_seen_cameras(candidates, sam, config["forced_seen_anchors"], config["tie_tolerance"])
    unseen, _ = select_unseen_cameras(candidates, sam, seen, config["tie_tolerance"])
    allowlist = build_camera_light_allowlist(seen, unseen, pairs, config["seen_lights"], config["unseen_lights"])
    assert len(allowlist) == 32
    assert allowlist.allowed.all()
    assert allowlist.split_role.value_counts().to_dict() == {"ID": 18, "CAMERA_OOD": 6, "LIGHT_OOD": 6, "JOINT_OOD": 2}
    assert not allowlist.duplicated(["camera_name", "light_name"]).any()
