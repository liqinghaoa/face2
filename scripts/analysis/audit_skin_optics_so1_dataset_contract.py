"""Audit the frozen SO-1 synthetic decomposition data contract."""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from skin_optics_so1.decomposition.contract import (  # noqa: E402
    ARRAY_SPECS,
    EXPECTED_GENERATOR_CONFIG_HASH,
    EXPECTED_SPLIT_COUNTS,
    EXPECTED_TOTAL_ROWS,
    SPLITS,
    build_dataset_contract,
    load_integrity_records,
    load_json,
    sha256_file,
)


DEFAULT_DATA_ROOT = (
    PROJECT_ROOT / "data/processed/SO1_Synthetic_Generator_v1_localfull"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "experiments/skin_optics_so1/SO1_Decomposition_UNet_v1/preflight"
)
REQUIRED_COMPLETED_FIELDS = {
    "status": "COMPLETED",
    "rows": EXPECTED_TOTAL_ROWS,
    "full_generation_started": True,
    "smoke": False,
    "train_final_pair_collision_count": 0,
}
REQUIRED_REPAIR_FIELDS = (
    "previous_completion_invalidated",
    "repair_reason",
    "repair_base_latent_ids",
    "repair_sample_ids",
)
METADATA_COLUMNS = {
    "row_index",
    "sample_id",
    "split",
    "split_index",
    "base_latent_id",
    "acquisition_variant_id",
    "camera_name",
    "light_name",
    "camera_light_pair",
    "expected_valid_fraction",
    "so0_version",
    "so0_config_hash",
    "generator_config_hash",
    "actual_valid_fraction",
    "generation_status",
}
TRAIN_FINAL_METADATA_COLUMNS = {
    "initial_camera_light_pair",
    "final_camera_light_pair",
    "final_camera_name",
    "final_light_name",
    "retry_count",
}


class AuditFailure(RuntimeError):
    pass


def _failures_to_status(failures: list[str]) -> str:
    return "PASS" if not failures else "FAIL"


def _record_check(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any = None) -> None:
    checks.append({"name": name, "status": "PASS" if passed else "FAIL", "detail": detail})


def _read_metadata(split_dir: Path, split: str) -> pd.DataFrame:
    path = split_dir / "metadata.csv"
    frame = pd.read_csv(path)
    missing = METADATA_COLUMNS.difference(frame.columns)
    if missing:
        raise AuditFailure(f"{split}/metadata.csv missing columns: {sorted(missing)}")
    if split == "train":
        missing_train = TRAIN_FINAL_METADATA_COLUMNS.difference(frame.columns)
        if missing_train:
            raise AuditFailure(
                f"{split}/metadata.csv missing train repair columns: {sorted(missing_train)}"
            )
    return frame


def _allowed_pair_sets(camera_light_split: dict[str, Any]) -> dict[str, set[str]]:
    seen_cameras = set(camera_light_split["seen_cameras"])
    unseen_cameras = set(camera_light_split["unseen_cameras"])
    seen_lights = set(camera_light_split["seen_lights"])
    unseen_lights = set(camera_light_split["unseen_lights"])
    qualified = set(camera_light_split["qualified_pairs"])

    def combine(cameras: set[str], lights: set[str]) -> set[str]:
        return {pair for pair in qualified if pair.split(" / ", 1)[0] in cameras and pair.rsplit(" / ", 1)[1] in lights}

    return {
        "train": combine(seen_cameras, seen_lights),
        "validation": combine(seen_cameras, seen_lights),
        "id_test": combine(seen_cameras, seen_lights),
        "camera_ood": combine(unseen_cameras, seen_lights),
        "light_ood": combine(seen_cameras, unseen_lights),
        "joint_ood": combine(unseen_cameras, unseen_lights),
    }


def _audit_completed(data_root: Path, checks: list[dict[str, Any]], failures: list[str]) -> dict[str, Any]:
    path = data_root / "COMPLETED.json"
    completed = load_json(path)
    for key, expected in REQUIRED_COMPLETED_FIELDS.items():
        passed = completed.get(key) == expected
        _record_check(checks, f"completed.{key}", passed, {"actual": completed.get(key), "expected": expected})
        if not passed:
            failures.append(f"COMPLETED.json {key}={completed.get(key)!r}, expected {expected!r}")
    for key in REQUIRED_REPAIR_FIELDS:
        passed = key in completed and completed.get(key) not in (None, [], "")
        _record_check(checks, f"completed.{key}", passed, completed.get(key))
        if not passed:
            failures.append(f"COMPLETED.json missing required repair field {key}")
    return completed


def _audit_files_and_metadata(
    data_root: Path,
    checks: list[dict[str, Any]],
    failures: list[str],
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    integrity = load_integrity_records(data_root)
    metadata_by_split: dict[str, pd.DataFrame] = {}
    file_rows: list[dict[str, Any]] = []
    for split, expected_count in EXPECTED_SPLIT_COUNTS.items():
        split_dir = data_root / split
        if not split_dir.is_dir():
            failures.append(f"Missing split directory: {split_dir}")
            _record_check(checks, f"{split}.directory_exists", False, str(split_dir))
            continue
        _record_check(checks, f"{split}.directory_exists", True, str(split_dir))
        try:
            frame = _read_metadata(split_dir, split)
        except Exception as exc:
            failures.append(str(exc))
            _record_check(checks, f"{split}.metadata_columns", False, str(exc))
            continue
        metadata_by_split[split] = frame
        _record_check(checks, f"{split}.metadata_columns", True, sorted(frame.columns))
        row_count_ok = len(frame) == expected_count
        _record_check(checks, f"{split}.metadata_count", row_count_ok, {"actual": len(frame), "expected": expected_count})
        if not row_count_ok:
            failures.append(f"{split} metadata rows {len(frame)} != {expected_count}")
        split_ok = set(frame["split"].astype(str)) == {split}
        _record_check(checks, f"{split}.metadata_split_values", split_ok, sorted(set(frame["split"].astype(str))))
        if not split_ok:
            failures.append(f"{split} metadata split column contains other values")
        dense_ok = sorted(frame["split_index"].astype(int).tolist()) == list(range(len(frame)))
        unique_index_ok = frame["split_index"].is_unique
        _record_check(checks, f"{split}.split_index_dense_unique", dense_ok and unique_index_ok, None)
        if not (dense_ok and unique_index_ok):
            failures.append(f"{split} split_index is not dense and unique")
        sample_ok = frame["sample_id"].astype(str).is_unique and frame["sample_id"].astype(str).str.startswith(f"{split}_").all()
        _record_check(checks, f"{split}.sample_id_unique_prefix", bool(sample_ok), None)
        if not sample_ok:
            failures.append(f"{split} sample_id must be unique and start with {split}_")
        status_ok = set(frame["generation_status"].astype(str)) == {"SUCCESS"}
        _record_check(checks, f"{split}.generation_status_success", status_ok, sorted(set(frame["generation_status"].astype(str))))
        if not status_ok:
            failures.append(f"{split} contains non-SUCCESS generation_status")
        generator_ok = set(frame["generator_config_hash"].astype(str)) == {EXPECTED_GENERATOR_CONFIG_HASH}
        _record_check(checks, f"{split}.generator_config_hash", generator_ok, sorted(set(frame["generator_config_hash"].astype(str))))
        if not generator_ok:
            failures.append(f"{split} generator_config_hash mismatch")

        for name, spec in ARRAY_SPECS.items():
            path = split_dir / spec.filename
            exists = path.is_file()
            _record_check(checks, f"{split}.{spec.filename}.exists", exists, str(path))
            if not exists:
                failures.append(f"Missing array file: {path}")
                continue
            try:
                array = np.load(path, mmap_mode="r")
            except Exception as exc:
                failures.append(f"Cannot open {path}: {exc}")
                _record_check(checks, f"{split}.{spec.filename}.open", False, str(exc))
                continue
            expected_shape = spec.expected_shape(expected_count, 256)
            shape_ok = tuple(array.shape) == expected_shape
            dtype_ok = str(array.dtype) == spec.dtype
            _record_check(
                checks,
                f"{split}.{spec.filename}.shape_dtype",
                shape_ok and dtype_ok,
                {"shape": list(array.shape), "expected_shape": list(expected_shape), "dtype": str(array.dtype), "expected_dtype": spec.dtype},
            )
            if not shape_ok or not dtype_ok:
                failures.append(f"{split}/{spec.filename} shape or dtype mismatch")
            rel = f"{split}/{spec.filename}"
            integrity_record = integrity.get(rel)
            size_ok = integrity_record is None or int(integrity_record["bytes"]) == path.stat().st_size
            _record_check(
                checks,
                f"{split}.{spec.filename}.integrity_size",
                size_ok,
                {"actual_bytes": path.stat().st_size, "record": integrity_record},
            )
            if not size_ok:
                failures.append(f"{split}/{spec.filename} size does not match file_integrity_sha256.csv")
            file_rows.append(
                {
                    "split": split,
                    "file": spec.filename,
                    "bytes": path.stat().st_size,
                    "dtype": str(array.dtype),
                    "shape": "x".join(map(str, array.shape)),
                    "sha256": integrity_record.get("sha256") if integrity_record else None,
                    "sha256_source": "file_integrity_sha256.csv" if integrity_record else None,
                }
            )
    return metadata_by_split, file_rows


def _audit_metadata_relations(
    metadata_by_split: dict[str, pd.DataFrame],
    camera_light_split: dict[str, Any],
    checks: list[dict[str, Any]],
    failures: list[str],
) -> None:
    all_rows = []
    for split, frame in metadata_by_split.items():
        subset = frame.copy()
        subset["_source_split"] = split
        all_rows.append(subset)
    if not all_rows:
        failures.append("No metadata loaded")
        return
    combined = pd.concat(all_rows, ignore_index=True)
    sample_unique = combined["sample_id"].astype(str).is_unique
    _record_check(checks, "global.sample_id_unique", bool(sample_unique), int(combined["sample_id"].nunique()))
    if not sample_unique:
        failures.append("sample_id is not globally unique")

    split_counts = combined.groupby("base_latent_id")["split"].nunique()
    leak_count = int((split_counts > 1).sum())
    _record_check(checks, "global.no_cross_split_base_latent_leakage", leak_count == 0, {"leaking_base_latent_ids": leak_count})
    if leak_count:
        failures.append(f"{leak_count} base_latent_id values occur in multiple splits")

    allowed = _allowed_pair_sets(camera_light_split)
    excluded = set(camera_light_split["excluded_pairs"])
    for split, frame in metadata_by_split.items():
        pairs = set(frame["camera_light_pair"].astype(str))
        illegal = sorted(pairs - allowed[split])
        excluded_present = sorted(pairs & excluded)
        _record_check(checks, f"{split}.camera_light_allowed", not illegal, illegal[:10])
        _record_check(checks, f"{split}.excluded_pairs_absent", not excluded_present, excluded_present[:10])
        if illegal:
            failures.append(f"{split} contains disallowed camera/light pairs: {illegal[:5]}")
        if excluded_present:
            failures.append(f"{split} contains excluded failed pairs: {excluded_present[:5]}")
        if split == "train":
            final_pairs = set(frame["final_camera_light_pair"].astype(str))
            final_illegal = sorted(final_pairs - allowed[split])
            final_excluded = sorted(final_pairs & excluded)
            changed_count = int(
                (
                    frame["camera_light_pair"].astype(str)
                    != frame["final_camera_light_pair"].astype(str)
                ).sum()
            )
            _record_check(
                checks,
                "train.final_camera_light_allowed",
                not final_illegal,
                final_illegal[:10],
            )
            _record_check(
                checks,
                "train.final_excluded_pairs_absent",
                not final_excluded,
                final_excluded[:10],
            )
            _record_check(
                checks,
                "train.initial_final_pair_difference_recorded",
                True,
                {"changed_rows": changed_count},
            )
            if final_illegal:
                failures.append(f"train contains disallowed final camera/light pairs: {final_illegal[:5]}")
            if final_excluded:
                failures.append(f"train contains excluded final failed pairs: {final_excluded[:5]}")

    train = metadata_by_split.get("train")
    if train is not None:
        grouped = train.groupby("base_latent_id")
        group_size_ok = grouped.size().eq(2).all()
        variant_ok = grouped["acquisition_variant_id"].apply(lambda s: sorted(map(int, s.tolist())) == [0, 1]).all()
        distinct_pair_ok = grouped["final_camera_light_pair"].nunique().eq(2).all()
        _record_check(checks, "train.base_latent_pairs_size_2", bool(group_size_ok), int(grouped.ngroups))
        _record_check(checks, "train.acquisition_variants_0_1", bool(variant_ok), None)
        _record_check(checks, "train.final_camera_light_pair_distinct_within_pair", bool(distinct_pair_ok), None)
        if not group_size_ok:
            failures.append("Train base_latent_id groups are not all size 2")
        if not variant_ok:
            failures.append("Train acquisition_variant_id groups are not all [0, 1]")
        if not distinct_pair_ok:
            failures.append("Train final camera/light pair collision remains after repair")


def _audit_arrays(
    data_root: Path,
    metadata_by_split: dict[str, pd.DataFrame],
    checks: list[dict[str, Any]],
    failures: list[str],
    *,
    chunk_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, frame in metadata_by_split.items():
        frame = frame.sort_values("split_index", kind="mergesort").reset_index(drop=True)
        split_dir = data_root / split
        linear = np.load(split_dir / "linear_rgb.f16.npy", mmap_mode="r")
        target = np.load(split_dir / "target_mhsp.f16.npy", mmap_mode="r")
        mask = np.load(split_dir / "valid_mask.u8.npy", mmap_mode="r")
        written = np.load(split_dir / "written.u8.npy", mmap_mode="r")
        written_ok = bool(np.all(written[:] == 1))
        _record_check(checks, f"{split}.written_all_one", written_ok, int(written.sum()))
        if not written_ok:
            failures.append(f"{split} has unwritten samples")

        stats = {
            "linear_min": math.inf,
            "linear_max": -math.inf,
            "target_min": math.inf,
            "target_max": -math.inf,
            "mask_sum_min": math.inf,
            "mask_sum_max": -math.inf,
            "linear_outside_mask_max_abs": 0.0,
            "target_outside_mask_max_abs": 0.0,
            "valid_fraction_max_abs_delta": 0.0,
            "linear_finite": True,
            "target_finite": True,
            "mask_binary": True,
            "nonempty_mask": True,
        }
        linear_channel_sum = np.zeros(3, dtype=np.float64)
        linear_channel_sumsq = np.zeros(3, dtype=np.float64)
        target_channel_sum = np.zeros(4, dtype=np.float64)
        target_channel_sumsq = np.zeros(4, dtype=np.float64)
        pixel_count = 0

        for start in range(0, len(frame), chunk_size):
            stop = min(start + chunk_size, len(frame))
            rgb = linear[start:stop].astype(np.float32, copy=False)
            y = target[start:stop].astype(np.float32, copy=False)
            m = mask[start:stop].astype(np.uint8, copy=False)
            m_float = m.astype(np.float32, copy=False)
            outside = 1.0 - m_float

            stats["linear_finite"] = stats["linear_finite"] and bool(np.isfinite(rgb).all())
            stats["target_finite"] = stats["target_finite"] and bool(np.isfinite(y).all())
            stats["mask_binary"] = stats["mask_binary"] and bool(np.isin(m, [0, 1]).all())
            mask_sums = m.reshape(m.shape[0], -1).sum(axis=1)
            stats["nonempty_mask"] = stats["nonempty_mask"] and bool(np.all(mask_sums > 0))
            stats["mask_sum_min"] = min(stats["mask_sum_min"], float(mask_sums.min()))
            stats["mask_sum_max"] = max(stats["mask_sum_max"], float(mask_sums.max()))
            stats["linear_min"] = min(stats["linear_min"], float(np.min(rgb)))
            stats["linear_max"] = max(stats["linear_max"], float(np.max(rgb)))
            stats["target_min"] = min(stats["target_min"], float(np.min(y)))
            stats["target_max"] = max(stats["target_max"], float(np.max(y)))
            stats["linear_outside_mask_max_abs"] = max(
                stats["linear_outside_mask_max_abs"], float(np.max(np.abs(rgb * outside)))
            )
            stats["target_outside_mask_max_abs"] = max(
                stats["target_outside_mask_max_abs"], float(np.max(np.abs(y * outside)))
            )
            valid_fraction = mask_sums.astype(np.float64) / float(m.shape[-1] * m.shape[-2])
            expected_fraction = frame.iloc[start:stop]["actual_valid_fraction"].astype(float).to_numpy()
            stats["valid_fraction_max_abs_delta"] = max(
                stats["valid_fraction_max_abs_delta"],
                float(np.max(np.abs(valid_fraction - expected_fraction))),
            )
            linear_channel_sum += rgb.sum(axis=(0, 2, 3), dtype=np.float64)
            linear_channel_sumsq += np.square(rgb, dtype=np.float32).sum(axis=(0, 2, 3), dtype=np.float64)
            target_channel_sum += y.sum(axis=(0, 2, 3), dtype=np.float64)
            target_channel_sumsq += np.square(y, dtype=np.float32).sum(axis=(0, 2, 3), dtype=np.float64)
            pixel_count += rgb.shape[0] * rgb.shape[-1] * rgb.shape[-2]

        range_ok = (
            stats["linear_min"] >= -1.0e-4
            and stats["linear_max"] <= 1.0 + 1.0e-4
            and stats["target_min"] >= -1.0e-4
            and stats["target_max"] <= 1.0 + 1.0e-4
        )
        outside_ok = (
            stats["linear_outside_mask_max_abs"] <= 1.0e-4
            and stats["target_outside_mask_max_abs"] <= 1.0e-4
        )
        valid_fraction_ok = stats["valid_fraction_max_abs_delta"] <= 1.0e-6
        array_ok = (
            stats["linear_finite"]
            and stats["target_finite"]
            and stats["mask_binary"]
            and stats["nonempty_mask"]
            and range_ok
            and outside_ok
            and valid_fraction_ok
        )
        _record_check(checks, f"{split}.array_values", array_ok, stats)
        if not array_ok:
            failures.append(f"{split} array value audit failed")

        linear_mean = linear_channel_sum / max(pixel_count, 1)
        target_mean = target_channel_sum / max(pixel_count, 1)
        linear_std = np.sqrt(np.maximum(linear_channel_sumsq / max(pixel_count, 1) - linear_mean**2, 0.0))
        target_std = np.sqrt(np.maximum(target_channel_sumsq / max(pixel_count, 1) - target_mean**2, 0.0))
        row = {
            "split": split,
            **stats,
            "linear_channel_mean": json.dumps(linear_mean.tolist()),
            "linear_channel_std": json.dumps(linear_std.tolist()),
            "target_channel_mean": json.dumps(target_mean.tolist()),
            "target_channel_std": json.dumps(target_std.tolist()),
        }
        rows.append(row)
    return rows


def write_report(output_dir: Path, audit: dict[str, Any]) -> None:
    lines = [
        "# SO-1A Dataset Contract Audit",
        "",
        f"Status: {audit['status']}",
        f"Data root: `{audit['data_root']}`",
        f"Rows: {audit['total_rows']}",
        f"Generator config hash: `{audit['generator_config_hash']}`",
        f"Contract sha256: `{audit['contract_sha256']}`",
        "",
        "## Split Counts",
        "",
        "| split | rows |",
        "| --- | ---: |",
    ]
    for split, count in audit["split_counts"].items():
        lines.append(f"| {split} | {count} |")
    lines.extend(["", "## Failures", ""])
    if audit["failures"]:
        lines.extend(f"- {failure}" for failure in audit["failures"])
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- SO-1 synthetic data and SO-0 upstream assets were audited read-only.",
            "- Large array hashes were read from file_integrity_sha256.csv and file sizes were checked.",
            "- The synthetic_data symlink compatibility directory is not used by the decomposition dataset.",
            "",
        ]
    )
    (output_dir / "SO1A_dataset_contract_audit_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def audit_dataset_contract(
    data_root: str | Path,
    output_dir: str | Path,
    *,
    chunk_size: int = 64,
) -> dict[str, Any]:
    root = Path(data_root).resolve()
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    checks: list[dict[str, Any]] = []
    failures: list[str] = []
    completed = _audit_completed(root, checks, failures)
    run_hashes = load_json(root / "run_hashes.json")
    generator_hash = run_hashes.get("generator_config_hash")
    generator_ok = generator_hash == EXPECTED_GENERATOR_CONFIG_HASH
    _record_check(
        checks,
        "run_hashes.generator_config_hash",
        generator_ok,
        {"actual": generator_hash, "expected": EXPECTED_GENERATOR_CONFIG_HASH},
    )
    if not generator_ok:
        failures.append("run_hashes.json generator_config_hash mismatch")

    config_path = root / "config" / "so1_synthetic_generation_v1.yaml"
    if config_path.is_file():
        _record_check(checks, "frozen_generator_config.present", True, str(config_path))
    else:
        _record_check(checks, "frozen_generator_config.present", False, str(config_path))
        failures.append("Frozen config/so1_synthetic_generation_v1.yaml is missing")

    camera_light_split = load_json(root / "camera_light_split.json")
    metadata_by_split, file_rows = _audit_files_and_metadata(root, checks, failures)
    _audit_metadata_relations(metadata_by_split, camera_light_split, checks, failures)
    array_rows = _audit_arrays(root, metadata_by_split, checks, failures, chunk_size=chunk_size)

    split_counts = {split: int(len(frame)) for split, frame in metadata_by_split.items()}
    contract = build_dataset_contract(root)
    contract_path = out / "SO1_Decomposition_DatasetContract_v1.json"
    contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(checks).to_csv(out / "SO1A_contract_checks.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(file_rows).to_csv(out / "SO1A_file_manifest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(array_rows).to_csv(out / "SO1A_array_statistics.csv", index=False, encoding="utf-8-sig")

    audit = {
        "schema_version": "SO1A_DatasetContractAudit_v1",
        "status": _failures_to_status(failures),
        "data_root": str(root),
        "output_dir": str(out),
        "total_rows": int(sum(split_counts.values())),
        "split_counts": split_counts,
        "generator_config_hash": generator_hash,
        "contract_path": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "completed_json": completed,
        "python": sys.version,
        "platform": platform.platform(),
        "failures": failures,
        "checks": checks,
    }
    audit_path = out / "SO1A_dataset_contract_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(out, audit)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunk-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = audit_dataset_contract(args.data_root, args.output_dir, chunk_size=args.chunk_size)
    print(json.dumps({"status": audit["status"], "output_dir": audit["output_dir"]}, ensure_ascii=False))
    if audit["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
