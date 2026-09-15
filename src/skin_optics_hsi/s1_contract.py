"""S1-0: build an evidence-labelled, reproducible Hyper-Skin data contract."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
from PIL import Image

from .data_contracts import OFFICIAL_SPLITS, HyperSkinSample, audit_data_contract, discover_hyperskin_vis


DEFAULT_EXPECTED_COUNTS = {"train": 264, "valid": 18, "test": 24}
VIS_WAVELENGTH_CENTERS_NM = tuple(float(value) for value in range(400, 701, 10))
VIS_WAVELENGTH_CODE_EVIDENCE = {
    "repository": "https://github.com/hyperspectral-skin/Hyper-Skin-2023",
    "commit": "380ff1f97a81aefc074e8ddd8e44f3c7e104afe9",
    "path": "evaluations/vis_evaluation_mstpp_retrained.ipynb",
    "git_blob": "13b0a938e279739c4073ea14fff0bfcc42fd4cd2",
    "expression": "band_31 = np.arange(400, 710, 10)",
}
MANIFEST_FIELDS = (
    "sample_id",
    "subject_id",
    "expression",
    "direction",
    "split",
    "rgb_path",
    "hsi_path",
    "rgb_relative_path",
    "hsi_relative_path",
    "rgb_bytes",
    "hsi_bytes",
    "rgb_sha256",
    "hsi_sha256",
    "rgb_width",
    "rgb_height",
    "rgb_mode",
    "hsi_dataset_key",
    "hsi_stored_shape",
    "hsi_loader_shape_before_s1_1",
    "hsi_dtype",
    "hsi_chunks",
    "hsi_compression",
    "hsi_min",
    "hsi_max",
    "hsi_nan_count",
    "hsi_inf_count",
    "hsi_below_zero_count",
    "hsi_above_one_count",
    "value_scan",
)


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _scan_values(dataset: h5py.Dataset, mode: str) -> dict[str, Any]:
    if mode == "none":
        return {
            "hsi_min": "",
            "hsi_max": "",
            "hsi_nan_count": "",
            "hsi_inf_count": "",
            "hsi_below_zero_count": "",
            "hsi_above_one_count": "",
        }
    if mode == "sampled":
        slices = tuple(slice(None, None, max(1, int(size) // 128)) for size in dataset.shape)
        values = np.asarray(dataset[slices])
    elif mode == "full":
        values = np.asarray(dataset)
    else:
        raise ValueError("value_scan must be one of: none, sampled, full")
    finite = np.isfinite(values)
    finite_values = values[finite]
    return {
        "hsi_min": float(finite_values.min()) if finite_values.size else "",
        "hsi_max": float(finite_values.max()) if finite_values.size else "",
        "hsi_nan_count": int(np.isnan(values).sum()),
        "hsi_inf_count": int(np.isinf(values).sum()),
        "hsi_below_zero_count": int((finite_values < 0).sum()),
        "hsi_above_one_count": int((finite_values > 1).sum()),
    }


def inspect_sample(sample: HyperSkinSample, data_root: Path, dataset_key: str, value_scan: str) -> dict[str, Any]:
    if sample.rgb_path is None:
        raise FileNotFoundError(f"Missing paired RGB for {sample.sample_id}")
    with Image.open(sample.rgb_path) as image:
        rgb_width, rgb_height = image.size
        rgb_mode = image.mode
        image.verify()
    with h5py.File(sample.hsi_path, "r") as archive:
        if dataset_key not in archive:
            raise ValueError(f"Missing HDF5 dataset key '{dataset_key}': {sample.hsi_path}")
        dataset = archive[dataset_key]
        stored_shape = tuple(int(value) for value in dataset.shape)
        band_axes = [index for index, size in enumerate(stored_shape) if size == 31]
        if len(band_axes) != 1:
            loader_shape: tuple[int, ...] = ()
        else:
            loader_shape = tuple(size for index, size in enumerate(stored_shape) if index != band_axes[0]) + (31,)
        value_stats = _scan_values(dataset, value_scan)
        metadata = {
            "hsi_dataset_key": dataset_key,
            "hsi_stored_shape": json.dumps(stored_shape),
            "hsi_loader_shape_before_s1_1": json.dumps(loader_shape),
            "hsi_dtype": str(dataset.dtype),
            "hsi_chunks": json.dumps(dataset.chunks) if dataset.chunks else "",
            "hsi_compression": str(dataset.compression or "none"),
        }
    return {
        "sample_id": sample.sample_id,
        "subject_id": sample.subject_id,
        "expression": sample.expression,
        "direction": sample.direction,
        "split": sample.split,
        "rgb_path": str(sample.rgb_path),
        "hsi_path": str(sample.hsi_path),
        "rgb_relative_path": _relative(sample.rgb_path, data_root),
        "hsi_relative_path": _relative(sample.hsi_path, data_root),
        "rgb_bytes": sample.rgb_path.stat().st_size,
        "hsi_bytes": sample.hsi_path.stat().st_size,
        "rgb_sha256": sha256_file(sample.rgb_path),
        "hsi_sha256": sha256_file(sample.hsi_path),
        "rgb_width": rgb_width,
        "rgb_height": rgb_height,
        "rgb_mode": rgb_mode,
        **metadata,
        **value_stats,
        "value_scan": value_scan,
    }


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _evidence_markdown(contract: dict[str, Any]) -> str:
    evidence = contract["evidence"]
    return "\n".join(
        [
            "# Hyper-Skin 阶段一数据合同证据",
            "",
            f"生成时间：`{contract['created_utc']}`",
            "",
            "## 可以确认的事实",
            "",
            "- 下载内容是 Hyper-Skin RGB/VIS 发布集；HSI 文件是 MATLAB v7.3/HDF5，数据键为 `cube`。",
            "- 论文与采集文档将发布光谱描述为暗场校正、白参考归一化后的光谱反射率。这里的 `confirmed` 表示文档证据充分，不表示当前文件携带独立校准元数据。",
            "- 发布 VIS 范围为 400–700 nm，共 31 波段；原始 FX10 数据为 400–1000 nm、448 波段，发布数据经过插值。",
            "- 作者官方评估代码在固定提交中明确使用 `band_31 = np.arange(400, 710, 10)`，因此发布 VIS 的标称中心波长及升序通道顺序确认为 `400, 410, ..., 700 nm`。",
            "",
            "## 仍缺失或仅部分确认",
            "",
            "- 下载目录未提供 31 波段 SRF/有效带宽、原始 wavelength vector、精确插值函数与端点策略。原始系统约 5.5 nm FWHM 不能直接当作发布 31 波段的有效带宽。",
            "- HDF5 读出后的两个空间轴与 RGB 的唯一映射留给 S1-1 冻结；S1-0 不提前把转置规则写成事实。",
            "- 原始数据只读；极小的 `[0,1]` 越界只在派生数据中按已记录规则裁剪，绝不改写原文件。",
            "",
            "## 证据来源",
            "",
            *[f"- **{item['status']}** — {item['claim']}  来源：`{item['source']}`（{item['location']}）" for item in evidence],
            "",
            "## 使用限制",
            "",
            "数据提供方邮件说明：除 p012、p019、p027 外，不将其他受试者图像用于出版图。所有样本仍可在许可范围内用于内部分析；任何公开材料必须再次核对原许可文本。",
            "",
        ]
    )


def build_s1_0_contract(
    data_root: str | Path,
    output_root: str | Path,
    *,
    expected_counts: dict[str, int] | None = None,
    expected_spatial_shape: tuple[int, int] = (1024, 1024),
    dataset_key: str = "cube",
    value_scan: str = "full",
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Create S1-0 artifacts. Existing artifacts are never overwritten."""

    root = Path(data_root).resolve()
    output = Path(output_root).resolve()
    contracts_dir = output / "contracts"
    manifests_dir = output / "manifests"
    contract_path = contracts_dir / "data_contract.json"
    evidence_path = contracts_dir / "contract_evidence.md"
    manifest_path = manifests_dir / "split_manifest.csv"
    existing = [path for path in (contract_path, evidence_path, manifest_path) if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing S1-0 artifacts: {existing}")
    contracts_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    expected = DEFAULT_EXPECTED_COUNTS if expected_counts is None else expected_counts
    samples = discover_hyperskin_vis(root)
    base_audit = audit_data_contract(samples, expected, require_masks=False, require_rgb=True)
    rows: list[dict[str, Any]] = []
    inspection_failures: list[dict[str, str]] = []
    for index, sample in enumerate(samples, start=1):
        try:
            rows.append(inspect_sample(sample, root, dataset_key, value_scan))
        except Exception as error:
            inspection_failures.append({"sample_id": sample.sample_id, "error": repr(error)})
        if progress and (index == 1 or index % 25 == 0 or index == len(samples)):
            progress(f"S1-0 inspected {index}/{len(samples)} samples")

    _write_manifest(manifest_path, rows)
    manifest_sha256 = sha256_file(manifest_path)
    issues = list(base_audit["issues"])
    if inspection_failures:
        issues.append(
            {
                "severity": "critical",
                "code": "FILE_INSPECTION_FAILURE",
                "message": f"{len(inspection_failures)} paired samples failed file inspection",
            }
        )
    stored_shapes = sorted({row["hsi_stored_shape"] for row in rows})
    loader_shapes = sorted({row["hsi_loader_shape_before_s1_1"] for row in rows})
    hsi_dtypes = sorted({row["hsi_dtype"] for row in rows})
    rgb_shapes = sorted({f"{row['rgb_height']}x{row['rgb_width']}" for row in rows})
    expected_stored_shape = json.dumps((31, *expected_spatial_shape))
    expected_loader_shape = json.dumps((*expected_spatial_shape, 31))
    if stored_shapes != [expected_stored_shape] or loader_shapes != [expected_loader_shape]:
        issues.append({"severity": "critical", "code": "HSI_SHAPE_MISMATCH", "message": str(stored_shapes)})
    expected_rgb_shape = f"{expected_spatial_shape[0]}x{expected_spatial_shape[1]}"
    if rgb_shapes != [expected_rgb_shape]:
        issues.append({"severity": "critical", "code": "RGB_SHAPE_MISMATCH", "message": str(rgb_shapes)})
    nan_total = sum(int(row["hsi_nan_count"] or 0) for row in rows)
    inf_total = sum(int(row["hsi_inf_count"] or 0) for row in rows)
    if value_scan != "none" and (nan_total or inf_total):
        issues.append({"severity": "critical", "code": "NONFINITE_HSI", "message": f"NaN={nan_total}, Inf={inf_total}"})
    below_zero = sum(int(row["hsi_below_zero_count"] or 0) for row in rows)
    above_one = sum(int(row["hsi_above_one_count"] or 0) for row in rows)
    if below_zero or above_one:
        issues.append(
            {
                "severity": "warning",
                "code": "FLOAT_BOUNDARY_DEVIATION",
                "message": f"Values below 0: {below_zero}; values above 1: {above_one}. Clip derived arrays only.",
            }
        )
    issues.extend(
        [
            {"severity": "warning", "code": "EFFECTIVE_SRF_MISSING", "message": "31-band effective SRF/bandwidth is not available"},
            {"severity": "pending", "code": "SPATIAL_MAPPING_PENDING_S1_1", "message": "RGB-HSI transform must be frozen by S1-1"},
        ]
    )
    status = "FAIL" if any(issue["severity"] == "critical" for issue in issues) else "PASS"
    paper_root = Path("face2_research/01_Background_and_Literature/Key_Papers/11_Ng_2023_Hyper_Skin")
    contract: dict[str, Any] = {
        "schema_version": 1,
        "stage": "S1-0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "readiness": {
            "s1_1_allowed": status == "PASS",
            "formal_test_allowed": False,
            "formal_test_blockers": ["effective_srf_missing", "spatial_mapping_pending_s1_1"],
        },
        "dataset": {
            "name": "Hyper-Skin RGB_VIS release",
            "version": "downloaded_release_version_not_labelled",
            "raw_root": str(root),
            "raw_data_policy": "read_only",
            "license_publication_subject_allowlist": ["p012", "p019", "p027"],
        },
        "counts": {
            "samples": len(samples),
            "expected_by_split": expected,
            "actual_by_split": base_audit["split_counts"],
            "subjects_by_split": base_audit["subject_counts"],
            "subject_ids_by_split": base_audit["subjects_by_split"],
        },
        "pairing": {"rgb_vis_one_to_one": base_audit["missing_rgb_count"] == 0, "missing_rgb_count": base_audit["missing_rgb_count"]},
        "hsi_storage": {
            "container": "MATLAB v7.3 / HDF5",
            "dataset_key": dataset_key,
            "stored_shapes": stored_shapes,
            "loader_shapes_before_s1_1": loader_shapes,
            "dtypes": hsi_dtypes,
            "axis_contract": {
                "stored_band_axis": 0,
                "loader_operation": "move unique 31-length axis to last",
                "rgb_spatial_mapping": {"status": "pending", "to_be_frozen_by": "S1-1"},
            },
        },
        "physical_quantity": {
            "name": "normalized spectral reflectance",
            "status": "confirmed_from_dataset_documentation",
            "file_embedded_calibration_metadata": "missing",
            "raw_expected_range": [0.0, 1.0],
            "derived_clip_range": [0.0, 1.0],
            "clip_policy": "derived_arrays_only_never_raw",
        },
        "wavelength": {
            "coverage_nm": {"start": 400.0, "stop": 700.0, "status": "confirmed"},
            "band_count": {"value": 31, "status": "confirmed_from_files_and_documentation"},
            "centers_nm": {
                "value": list(VIS_WAVELENGTH_CENTERS_NM),
                "status": "confirmed_from_official_code",
                "evidence": VIS_WAVELENGTH_CODE_EVIDENCE,
            },
            "ordering": "ascending",
            "ordering_status": "confirmed_from_official_code",
            "raw_system_fwhm_nm": {"value": 5.5, "status": "confirmed_for_raw_FX10_not_release_bands"},
            "release_effective_bandwidth_or_srf": {"value": None, "status": "missing"},
        },
        "value_audit": {
            "mode": value_scan,
            "global_min": min((float(row["hsi_min"]) for row in rows if row["hsi_min"] != ""), default=None),
            "global_max": max((float(row["hsi_max"]) for row in rows if row["hsi_max"] != ""), default=None),
            "nan_count": nan_total if value_scan != "none" else None,
            "inf_count": inf_total if value_scan != "none" else None,
            "below_zero_count": below_zero if value_scan != "none" else None,
            "above_one_count": above_one if value_scan != "none" else None,
        },
        "manifest": {"path": str(manifest_path), "sha256": manifest_sha256, "row_count": len(rows), "hash_algorithm": "SHA-256"},
        "evidence": [
            {"field": "physical_quantity", "status": "confirmed", "claim": "暗场校正和白参考归一化后的光谱反射率", "source": str(paper_root / "Hyper-Skin_A_Hyperspectral_Dataset_for_Reconstructing_Facial_Skin-Spectra_from_RGB_Images.pdf"), "location": "Methods / hyperspectral acquisition and preprocessing"},
            {"field": "raw_sensor", "status": "confirmed", "claim": "Specim FX10；400–1000 nm；448 波段；约 1.34 nm sampling；约 5.5 nm raw FWHM", "source": str(paper_root / "SupportingInformation/Appendix A1 - Hyperspectral_Data_Acquisition_Documentation.pdf"), "location": "camera and acquisition specification"},
            {"field": "release_vis", "status": "confirmed", "claim": "发布 VIS 为 400–700 nm、31 波段；由原始波段插值得到", "source": str(paper_root / "SupportingInformation/Appendix A1 - Hyper_Skin_Datasheets.pdf"), "location": "VIS dataset description"},
            {
                "field": "wavelength_centers_and_ordering",
                "status": "confirmed_from_official_code",
                "claim": "发布 VIS 的 31 个标称中心为 400:10:700 nm，按升序映射到通道 0:31",
                "source": VIS_WAVELENGTH_CODE_EVIDENCE["repository"],
                "location": (
                    f"commit {VIS_WAVELENGTH_CODE_EVIDENCE['commit']}; "
                    f"{VIS_WAVELENGTH_CODE_EVIDENCE['path']}; "
                    f"{VIS_WAVELENGTH_CODE_EVIDENCE['expression']}"
                ),
            },
            {"field": "release_srf", "status": "missing", "claim": "31 波段有效 SRF/带宽未提供", "source": "downloaded release and supplied documents", "location": "not found"},
        ],
        "inspection_failures": inspection_failures,
        "issues": issues,
    }
    contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    evidence_path.write_text(_evidence_markdown(contract), encoding="utf-8")
    return contract
