from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from skin_optics_so1.decomposition.contract import build_dataset_contract
from skin_optics_so1.decomposition.synthetic_dataset import SO1DecompositionDataset


def _write_split(root: Path, split: str, rows: int = 2, size: int = 8) -> None:
    split_dir = root / split
    split_dir.mkdir(parents=True)
    linear = np.zeros((rows, 3, size, size), dtype=np.float16)
    target = np.zeros((rows, 4, size, size), dtype=np.float16)
    mask = np.ones((rows, 1, size, size), dtype=np.uint8)
    written = np.ones((rows,), dtype=np.uint8)
    for index in range(rows):
        linear[index, :, :, :] = np.float16(0.1 * (index + 1))
        target[index, :, :, :] = np.float16(0.2 * (index + 1))
    np.save(split_dir / "linear_rgb.f16.npy", linear)
    np.save(split_dir / "target_mhsp.f16.npy", target)
    np.save(split_dir / "valid_mask.u8.npy", mask)
    np.save(split_dir / "written.u8.npy", written)
    pd.DataFrame(
        {
            "sample_id": [f"{split}_{index:06d}" for index in range(rows)],
            "split": [split] * rows,
            "split_index": list(range(rows)),
            "base_latent_id": list(range(rows)),
            "acquisition_variant_id": [0] * rows,
        }
    ).to_csv(split_dir / "metadata.csv", index=False)


def test_so1_decomposition_dataset_reads_by_split_index(tmp_path: Path) -> None:
    _write_split(tmp_path, "train", rows=2, size=8)
    metadata_path = tmp_path / "train" / "metadata.csv"
    pd.read_csv(metadata_path).iloc[[1, 0]].to_csv(metadata_path, index=False)
    dataset = SO1DecompositionDataset(tmp_path, "train", require_full_count=False)
    assert len(dataset) == 2
    sample = dataset[0]
    assert tuple(sample["input"].shape) == (3, 8, 8)
    assert tuple(sample["target"].shape) == (4, 8, 8)
    assert tuple(sample["mask"].shape) == (1, 8, 8)
    assert sample["sample_id"] == "train_000000"
    assert sample["split_index"] == 0
    assert float(sample["input"].mean()) < 0.11


def test_build_dataset_contract_records_expected_semantics(tmp_path: Path) -> None:
    for split in ("train", "validation", "id_test", "camera_ood", "light_ood", "joint_ood"):
        (tmp_path / split).mkdir(parents=True)
    (tmp_path / "COMPLETED.json").write_text(
        '{"status":"COMPLETED","rows":27000,"full_generation_started":true,'
        '"smoke":false,"train_final_pair_collision_count":0}',
        encoding="utf-8",
    )
    (tmp_path / "run_hashes.json").write_text(
        '{"generator_config_hash":"78256a4f18072257d407c46f311c5903a66e0242ee945ff62e1d1a18afcc5cdf"}',
        encoding="utf-8",
    )
    (tmp_path / "camera_light_split.json").write_text(
        '{"seen_cameras":["cam_a"],"unseen_cameras":["cam_b"],'
        '"seen_lights":["D65"],"unseen_lights":["FL11"],'
        '"qualified_pairs":["cam_a / D65","cam_a / FL11","cam_b / D65","cam_b / FL11"],'
        '"excluded_pairs":[]}',
        encoding="utf-8",
    )
    contract = build_dataset_contract(tmp_path, patch_size=8, integrity_records={})
    assert contract["schema_version"] == "SO1_Decomposition_DatasetContract_v1"
    assert contract["target"]["channel_order"] == ["M", "H", "S_norm", "P_norm"]
    assert contract["input"]["color_space"] == "linear RGB"
