"""Integrity tests for the superseded A0-AM1 camera replacement decision."""
from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "reports/so_r1_a0_am1_camera_replacement"
AM4 = ROOT / "reports/so_r1_a0_am4_global_camera_atlas/final_24pair_allowlist.csv"
FROZEN = ROOT / "config/so_r1/frozen_camera_light_24pair_allowlist_v1_1.csv"
LOCK = ROOT / "config/so_r1/SO_R1_A2_D0_AM1_PROTOCOL_LOCK.json"
G1 = ROOT / "data/processed/SO_R1_A2_G1_AM1_FullGeneration_v1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_nokia_n900_is_retired_and_excluded_from_authoritative_allowlist():
    lock = read(LOCK)
    assert "Nokia N900" in lock["excluded_camera_set"]
    assert "Nokia N900" not in lock["camera_set"]
    assert "Nokia N900" not in set(pd.read_csv(AM4)["camera_name"])
    assert "Nokia N900" not in set(pd.read_csv(FROZEN)["camera_name"])


def test_final_allowlist_matches_frozen_acceptance_and_has_24_pairs():
    lock = read(LOCK)
    am4 = pd.read_csv(AM4)
    frozen = pd.read_csv(FROZEN)
    assert len(am4) == 24 and len(frozen) == 24
    # AM4 final allowlist is authoritative; the older v1.1 candidate CSV is
    # retained only as historical context and may have been superseded later.
    assert sha(AM4) == lock["final_24pair_allowlist_hash"]
    assert set(am4.camera_name) == set(lock["camera_set"])
    assert set(am4.light_name) == set(lock["light_set"])


def test_g1_am1_acceptance_and_manifest_use_only_final_cameras():
    acceptance = read(G1 / "G1_AM1_ACCEPTANCE.json")
    manifest = pd.read_csv(G1 / "manifests/acquisition_manifest.csv")
    final = pd.read_csv(AM4)
    assert acceptance["status"] == "PASS"
    assert acceptance["dataset_status"] == "ACCEPTED"
    assert acceptance["formal_generation_completed"] is True
    assert len(manifest) == 67500
    assert set(manifest.camera) == set(final.camera_name)
    assert "Nokia N900" not in set(manifest.camera)


def test_replacement_evidence_records_nokia_removal_without_pilot_dependency():
    selected = read(REP / "selected_replacement_camera.json")
    assert selected["removed_camera"] == "Nokia N900"
    assert selected["selected_camera"] != "Nokia N900"
    assert selected["selected_camera"] == "Nikon D3X"
    source = Path(__file__).read_text(encoding="utf-8")
    retired_path_token = "Regression" + "Pilot_v1"
    assert retired_path_token not in source
    assert ("pytest" + ".skip") not in source and "xf" + "ail" not in source


def test_authoritative_artifact_hash_and_version_chain_is_complete():
    lock = read(LOCK)
    assert lock["status"] == "FROZEN"
    assert lock["protocol_version"] == "1.1"
    assert lock["final_24pair_allowlist_hash"] == sha(AM4)
    assert lock["training_authorized"] is False
    assert read(G1 / "G1_AM1_ACCEPTANCE.json")["protocol_hash"] == lock["protocol_hash"]
