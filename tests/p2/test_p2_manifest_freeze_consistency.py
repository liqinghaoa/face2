from __future__ import annotations

import hashlib
import json
from pathlib import Path

from torch.utils.data import DataLoader

from p2_counterfactual.dataset import P2ASingleRGBDataset, p2_a_collate
from p2_counterfactual.path_utils import normalize_path_for_comparison, resolve_project_path

MANIFEST = Path("data/processed/P2_Counterfactual_Relighting500_v1/manifests/p2_training_manifest.csv")
FREEZE = Path("data/processed/P2_Counterfactual_Relighting500_v1/metadata/P2_COUNTERFACTUAL_TRAINING_ASSETS_FROZEN.json")
AUDIT = Path("metadata/p2_manifest_path_migration_audit.json")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_restored_manifest_hash_matches_frozen_record() -> None:
    recorded = json.loads(FREEZE.read_text(encoding="utf-8"))["manifest_sha256"]
    current = sha256_file(MANIFEST)
    assert current == recorded
    first_asset_path = MANIFEST.read_text(encoding="utf-8").splitlines()[1].split(",")[4]
    assert resolve_project_path(first_asset_path, Path.cwd(), require_exists=True).is_file()


def test_path_migration_audit_confirms_semantic_invariance() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert audit["row_count_before"] == audit["row_count_after"] == 500
    assert audit["unique_case_count_before"] == audit["unique_case_count_after"] == 500
    assert audit["case_ids_identical"] is True
    assert audit["patient_groups_identical"] is True
    assert audit["folds_identical"] is True
    assert audit["labels_identical"] is True
    assert audit["qc_flags_identical"] is True
    assert audit["boundary_flags_identical"] is True
    assert audit["asset_relative_tail_paths_identical"] is True
    assert audit["only_path_prefix_changed"] is True
    assert audit["strict_inverse_recovered_manifest_sha256"] == audit["recorded_manifest_sha256"]


def test_dataset_and_dataloader_do_not_rewrite_manifest_for_all_modes() -> None:
    before = sha256_file(MANIFEST)
    modes = [
        {"input_mode": "original", "color_jitter_enabled": False},
        {"input_mode": "original", "color_jitter_enabled": True},
        {"input_mode": "relight_mix", "color_jitter_enabled": False},
        {"input_mode": "paired", "color_jitter_enabled": False},
    ]
    for mode in modes:
        ds = P2ASingleRGBDataset(
            MANIFEST,
            fold=0,
            split="train",
            input_mode=mode["input_mode"],
            training=True,
            color_jitter_enabled=mode["color_jitter_enabled"],
            max_cases=8,
            seed=2026,
        )
        _ = ds[0]
        loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=p2_a_collate)
        for batch_index, _batch in enumerate(loader):
            if batch_index >= 1:
                break
    after = sha256_file(MANIFEST)
    assert before == after


def test_manifest_asset_tail_paths_remain_comparable_after_prefix_recovery() -> None:
    text = MANIFEST.read_text(encoding="utf-8").splitlines()[1]
    assert normalize_path_for_comparison(text.split(",")[4]).endswith("data/processed/p0_physics_audit_v1/images/aligned_scene_224/100037382.png")
