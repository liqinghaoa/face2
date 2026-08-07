"""Frozen data contract for SO-1 synthetic decomposition."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


SPLITS = ("train", "validation", "id_test", "camera_ood", "light_ood", "joint_ood")
EXPECTED_SPLIT_COUNTS = {
    "train": 20_000,
    "validation": 2_000,
    "id_test": 2_000,
    "camera_ood": 1_000,
    "light_ood": 1_000,
    "joint_ood": 1_000,
}
EXPECTED_TOTAL_ROWS = 27_000
EXPECTED_GENERATOR_CONFIG_HASH = (
    "78256a4f18072257d407c46f311c5903a66e0242ee945ff62e1d1a18afcc5cdf"
)


@dataclass(frozen=True)
class ArraySpec:
    filename: str
    key: str
    dtype: str
    channels: int | None
    semantic: str

    def expected_shape(self, rows: int, patch_size: int) -> tuple[int, ...]:
        if self.channels is None:
            return (rows,)
        return (rows, self.channels, patch_size, patch_size)


ARRAY_SPECS = {
    "linear_rgb": ArraySpec(
        filename="linear_rgb.f16.npy",
        key="input",
        dtype="float16",
        channels=3,
        semantic="linear RGB, CHW, channel order [R, G, B], range [0, 1]",
    ),
    "target_mhsp": ArraySpec(
        filename="target_mhsp.f16.npy",
        key="target",
        dtype="float16",
        channels=4,
        semantic=(
            "masked SO-0 target, CHW, channel order "
            "[M, H, (S - 0.25) / 1.75, P / 0.10], nominal range [0, 1]"
        ),
    ),
    "valid_mask": ArraySpec(
        filename="valid_mask.u8.npy",
        key="mask",
        dtype="uint8",
        channels=1,
        semantic="valid skin mask, CHW, binary values {0, 1}",
    ),
    "written": ArraySpec(
        filename="written.u8.npy",
        key="written",
        dtype="uint8",
        channels=None,
        semantic="sample write sentinel, shape [N], all values must be 1",
    ),
}


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json_hash(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(text)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _integrity_key(data_root: Path, path_value: str) -> str:
    path_value = str(path_value).replace("\\", "/")
    root_name = data_root.name
    marker = f"/{root_name}/"
    if marker in path_value:
        return path_value.split(marker, 1)[1]
    if path_value.startswith(f"{root_name}/"):
        return path_value[len(root_name) + 1 :]
    return path_value


def load_integrity_records(data_root: str | Path) -> dict[str, dict[str, Any]]:
    root = Path(data_root)
    path = root / "file_integrity_sha256.csv"
    if not path.is_file():
        return {}
    frame = pd.read_csv(path)
    records: dict[str, dict[str, Any]] = {}
    for row in frame.to_dict("records"):
        key = _integrity_key(root, str(row["path"]))
        records[key] = {
            "path": str(row["path"]),
            "sha256": str(row["sha256"]),
            "bytes": int(row["bytes"]),
        }
    return records


def _file_record(
    data_root: Path,
    rel_path: str,
    integrity: dict[str, dict[str, Any]],
    *,
    hash_small_file: bool,
) -> dict[str, Any]:
    path = data_root / rel_path
    record: dict[str, Any] = {
        "relative_path": rel_path,
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
    }
    if rel_path in integrity:
        record["sha256"] = integrity[rel_path]["sha256"]
        record["sha256_source"] = "file_integrity_sha256.csv"
        record["integrity_bytes"] = integrity[rel_path]["bytes"]
    elif hash_small_file and path.is_file():
        record["sha256"] = sha256_file(path)
        record["sha256_source"] = "computed"
    else:
        record["sha256"] = None
        record["sha256_source"] = None
    return record


def build_dataset_contract(
    data_root: str | Path,
    *,
    patch_size: int = 256,
    integrity_records: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the immutable contract expected by SO-1 decomposition code."""

    root = Path(data_root)
    integrity = integrity_records if integrity_records is not None else load_integrity_records(root)
    completed = load_json(root / "COMPLETED.json")
    run_hashes = load_json(root / "run_hashes.json") if (root / "run_hashes.json").is_file() else {}
    camera_light_split = load_json(root / "camera_light_split.json")

    split_files: dict[str, Any] = {}
    for split, rows in EXPECTED_SPLIT_COUNTS.items():
        split_files[split] = {
            name: {
                **_file_record(root, f"{split}/{spec.filename}", integrity, hash_small_file=False),
                "expected_shape": list(spec.expected_shape(rows, patch_size)),
                "dtype": spec.dtype,
                "key": spec.key,
                "semantic": spec.semantic,
            }
            for name, spec in ARRAY_SPECS.items()
        }
        split_files[split]["metadata"] = _file_record(
            root, f"{split}/metadata.csv", integrity, hash_small_file=True
        )

    contract = {
        "schema_version": "SO1_Decomposition_DatasetContract_v1",
        "dataset_version": "SO1_Synthetic_Generator_v1_localfull",
        "data_root": str(root),
        "split_names": list(SPLITS),
        "split_counts": dict(EXPECTED_SPLIT_COUNTS),
        "total_rows": EXPECTED_TOTAL_ROWS,
        "patch_size": patch_size,
        "input": {
            "key": "linear_rgb",
            "shape": ["N", 3, patch_size, patch_size],
            "dtype": "float16_on_disk_float32_tensor",
            "channel_order": ["R", "G", "B"],
            "color_space": "linear RGB",
            "range": [0.0, 1.0],
        },
        "target": {
            "key": "target_mhsp",
            "shape": ["N", 4, patch_size, patch_size],
            "dtype": "float16_on_disk_float32_tensor",
            "channel_order": ["M", "H", "S_norm", "P_norm"],
            "normalization": {
                "M": "identity",
                "H": "identity",
                "S_norm": "(S - 0.25) / 1.75",
                "P_norm": "P / 0.10",
            },
            "range": [0.0, 1.0],
        },
        "mask": {
            "key": "valid_mask",
            "shape": ["N", 1, patch_size, patch_size],
            "dtype": "uint8_on_disk_float32_tensor",
            "values": [0, 1],
        },
        "metadata_rules": {
            "split_index": "0-based dense index into memmap arrays within each split",
            "sample_id": "unique globally and stable; expected prefix is '<split>_'",
            "train_pairing": (
                "train rows are paired by base_latent_id with acquisition_variant_id 0/1 "
                "and distinct final_camera_light_pair"
            ),
            "cross_split_leakage": "base_latent_id must not appear in more than one split",
        },
        "camera_light_rules": {
            "train_validation_id_test": "seen cameras and seen lights only",
            "camera_ood": "unseen cameras and seen lights only",
            "light_ood": "seen cameras and unseen FL11 only",
            "joint_ood": "unseen cameras and unseen FL11 only",
            "seen_cameras": camera_light_split["seen_cameras"],
            "unseen_cameras": camera_light_split["unseen_cameras"],
            "seen_lights": camera_light_split["seen_lights"],
            "unseen_lights": camera_light_split["unseen_lights"],
            "excluded_pairs": camera_light_split["excluded_pairs"],
        },
        "provenance": {
            "completed_status": completed.get("status"),
            "completed_rows": completed.get("rows"),
            "completed_full_generation_started": completed.get("full_generation_started"),
            "completed_smoke": completed.get("smoke"),
            "train_final_pair_collision_count": completed.get("train_final_pair_collision_count"),
            "previous_completion_invalidated": completed.get("previous_completion_invalidated"),
            "repair_reason": completed.get("repair_reason"),
            "repair_base_latent_ids": completed.get("repair_base_latent_ids"),
            "repair_sample_ids": completed.get("repair_sample_ids"),
            "so0_version": completed.get("so0_version"),
            "so0_config_hash": completed.get("so0_config_hash"),
            "generator_config_hash": run_hashes.get(
                "generator_config_hash", completed.get("generator_config_hash")
            ),
            "expected_generator_config_hash": EXPECTED_GENERATOR_CONFIG_HASH,
            "camera_light_split_hash": run_hashes.get("camera_light_split_hash"),
            "source_generation_platform": run_hashes.get("environment", {}).get("platform"),
            "source_generation_executable": run_hashes.get("environment", {}).get("executable"),
            "synthetic_data_symlink_targets": run_hashes.get("synthetic_data_symlink_targets"),
        },
        "file_hashes": {
            "COMPLETED.json": _file_record(root, "COMPLETED.json", integrity, hash_small_file=True),
            "run_hashes.json": _file_record(root, "run_hashes.json", integrity, hash_small_file=True),
            "camera_light_split.json": _file_record(
                root, "camera_light_split.json", integrity, hash_small_file=True
            ),
            "synthetic_manifest.csv": _file_record(
                root, "synthetic_manifest.csv", integrity, hash_small_file=True
            ),
            "synthetic_manifest.jsonl": _file_record(
                root, "synthetic_manifest.jsonl", integrity, hash_small_file=True
            ),
            "split_files": split_files,
            "integrity_table": _file_record(
                root, "file_integrity_sha256.csv", integrity, hash_small_file=True
            ),
        },
        "contract_notes": [
            "The synthetic_data directory is not part of the training contract.",
            "Large array sha256 values are taken from file_integrity_sha256.csv and size-checked.",
            "SO-0 and SO-1 synthetic generator outputs are upstream-frozen and must not be rewritten.",
        ],
    }
    contract["contract_sha256"] = canonical_json_hash(contract)
    return contract
