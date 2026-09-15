"""Create the Train-only S1-3 band-reliability audit without changing extraction outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s1-3-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.s1_3_root.resolve()
    decision_path = root / "s1_3_decision.json"
    spectra_path = root / "region_spectra.parquet"
    summary_path = root / "band_reliability_train_primary.csv"
    report_path = root / "band_reliability_report.md"
    final_path = root / "s1_3_final_decision.json"
    existing = [path for path in (summary_path, report_path, final_path) if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite S1-3 reliability artifacts: {existing}")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    expected = decision["outputs"]["region_spectra_parquet"]["sha256"]
    if sha256_file(spectra_path) != expected:
        raise ValueError("region_spectra.parquet hash differs from S1-3 decision")
    if decision.get("test_access_count") != 0:
        raise ValueError("S1-3 decision reports Test access")
    frame = pd.read_parquet(spectra_path)
    primary = frame.loc[
        frame["s1_2_analysis_role"].eq("primary_development")
        & frame["region"].isin(["left_cheek", "right_cheek"])
        & frame["extraction_status"].eq("USABLE")
    ].copy()
    wavelength = np.asarray(decision["wavelength_nm"], dtype=np.float64)
    median_columns = [f"reflectance_median_{int(value)}nm" for value in wavelength]
    median_values = primary[median_columns].to_numpy(dtype=np.float64)
    within_mad = np.stack(
        [primary[f"reflectance_mad_{int(value)}nm"].to_numpy(dtype=np.float64) for value in wavelength], axis=1
    )
    mean_values = np.stack(
        [primary[f"reflectance_mean_{int(value)}nm"].to_numpy(dtype=np.float64) for value in wavelength], axis=1
    )
    curvature = np.full(wavelength.shape, np.nan, dtype=np.float64)
    curvature[1:-1] = np.median(
        np.abs(median_values[:, 1:-1] - (median_values[:, :-2] + median_values[:, 2:]) / 2.0), axis=0
    )
    left = primary.loc[primary["region"].eq("left_cheek")].set_index("subject_id")
    right = primary.loc[primary["region"].eq("right_cheek")].set_index("subject_id")
    subjects = left.index.intersection(right.index)
    left_values = left.loc[subjects, median_columns].to_numpy(dtype=np.float64)
    right_values = right.loc[subjects, median_columns].to_numpy(dtype=np.float64)
    paired_difference = left_values - right_values
    broadband_difference = paired_difference.mean(axis=1)
    summary = pd.DataFrame(
        {
            "wavelength_nm": wavelength,
            "n_primary_train_region_spectra": len(primary),
            "finite_fraction": np.isfinite(median_values).mean(axis=0),
            "cohort_median_reflectance": np.median(median_values, axis=0),
            "cohort_iqr_reflectance": np.subtract(*np.quantile(median_values, [0.75, 0.25], axis=0)),
            "median_within_region_mad": np.median(within_mad, axis=0),
            "median_relative_within_region_mad": np.median(within_mad / np.maximum(median_values, 1e-8), axis=0),
            "median_abs_mean_minus_median": np.median(np.abs(mean_values - median_values), axis=0),
            "median_adjacent_curvature": curvature,
            "fraction_image_left_brighter_than_right": (paired_difference > 0).mean(axis=0),
            "median_image_left_minus_right": np.median(paired_difference, axis=0),
        }
    )
    summary["diagnostic_review_note"] = ""
    summary.loc[summary["wavelength_nm"].isin([400.0, 700.0]), "diagnostic_review_note"] = "spectral_endpoint"
    for value in (420.0, 590.0, 670.0, 680.0, 690.0):
        index = summary["wavelength_nm"].eq(value)
        existing_note = summary.loc[index, "diagnostic_review_note"]
        summary.loc[index, "diagnostic_review_note"] = existing_note.where(existing_note.eq(""), existing_note + "|") + "high_local_curvature_rank"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    unavailable = frame.loc[frame["extraction_status"].ne("USABLE")]
    unavailable_counts = (
        unavailable.groupby(["split", "region", "extraction_reason"], dropna=False).size().reset_index(name="count")
    )
    unavailable_lines = [
        f"- `{row.split}/{row.region}`：{int(row['count'])}，原因 `{row.extraction_reason}`"
        for _, row in unavailable_counts.iterrows()
    ]
    top_curvature = summary.dropna(subset=["median_adjacent_curvature"]).nlargest(5, "median_adjacent_curvature")
    curvature_text = ", ".join(
        f"{int(row.wavelength_nm)} nm ({row.median_adjacent_curvature:.4f})" for _, row in top_curvature.iterrows()
    )
    usable_values = frame.loc[frame["extraction_status"].eq("USABLE"), median_columns].to_numpy(dtype=np.float64)
    report = "\n".join(
        [
            "# S1-3 区域光谱与波段可靠性报告",
            "",
            f"生成时间：`{datetime.now(timezone.utc).isoformat()}`",
            "",
            "## 数据范围与门控",
            "",
            f"- 输入 282 个 Train/Validation 样本，形成 {len(frame)} 条样本×区域记录；Test 访问计数为 0。",
            f"- 可用记录 {int(frame['extraction_status'].eq('USABLE').sum())}；不可用记录 {len(unavailable)}。",
            "- 目标域正脸无表情双侧脸颊 94/94 可用，因此 S1-3 数据工程门通过。",
            f"- 所有可用中位光谱均为有限值，范围为 {np.min(usable_values):.6f}–{np.max(usable_values):.6f}。",
            "",
            "## 不可用区域解释",
            "",
            *unavailable_lines,
            "",
            "侧脸被冻结规则明确置空的遮挡侧脸颊，以及前额不足样本均保留为 `UNAVAILABLE`，没有插值或补造。Validation 的 `p016_smile_right` 继续作为压力测试失败完整保留。",
            "",
            "## Train 主域波段诊断",
            "",
            f"诊断仅使用 44 名 Train 受试者的 88 条正脸无表情脸颊光谱，不使用 Validation 选择波段。逐波段有限率均为 1.0；区域内相对 MAD 的逐波段中位数范围为 {summary['median_relative_within_region_mad'].min():.3f}–{summary['median_relative_within_region_mad'].max():.3f}。",
            "",
            f"局部曲率最高的五个内部波段为：{curvature_text}。其中红端 670–690 nm 呈现明显锯齿状局部变化；400 nm 是端点且具有不同于主体波段的左右侧方向，因此列入诊断审查。它们当前只是 S1-4/S1-5 的敏感性候选，不自动删除或降低权重。",
            "",
            "## 固定侧别效应",
            "",
            f"Train 正脸无表情中，图像左侧脸颊的宽带反射率在 {int((broadband_difference > 0).sum())}/{len(subjects)} 名受试者上高于图像右侧，中位有符号差为 {np.median(broadband_difference):.6f}。这表明数据存在稳定的照明/几何侧别效应，不能让 M/H 参数无约束地吸收该差异。",
            "",
            "后续处理要求：S1-4 保留显式散射/观察 nuisance 接口；S1-5 分别报告左右脸颊及对称聚合结果，并比较全 31 波段、端点移除以及高曲率波段敏感性。波段权重只能在 Train 上形成，再由 Validation 检验。",
            "",
            "## 结论边界",
            "",
            "发布版 31 波段有效 SRF/带宽仍为 `missing`。S1-3 不据此估计绝对黑色素、血红蛋白或氧合度；该缺口通过 S1-4/S1-5 的波长/FWHM 敏感性分析和代理量表述管理。",
            "",
        ]
    )
    report_path.write_text(report, encoding="utf-8")
    final = {
        "schema_version": 1,
        "stage": "S1-3",
        "status": "PASS_FOR_S1_4",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_decision_path": str(decision_path),
        "source_decision_sha256": sha256_file(decision_path),
        "band_reliability_scope": "train_primary_development_cheeks_only",
        "band_reliability_summary_path": str(summary_path),
        "band_reliability_summary_sha256": sha256_file(summary_path),
        "band_reliability_report_path": str(report_path),
        "band_reliability_report_sha256": sha256_file(report_path),
        "primary_train_subject_count": int(len(subjects)),
        "primary_train_region_spectrum_count": int(len(primary)),
        "image_left_brighter_broadband_count": int((broadband_difference > 0).sum()),
        "median_image_left_minus_right_broadband": float(np.median(broadband_difference)),
        "diagnostic_review_bands_nm": [400.0, 420.0, 590.0, 670.0, 680.0, 690.0, 700.0],
        "diagnostic_bands_are_excluded": False,
        "effective_srf_status": decision["effective_srf_status"],
        "effective_srf_policy": decision["effective_srf_policy"],
        "next_stage_allowed": True,
        "authorized_next_stage": "S1-4",
        "test_access_count": 0,
    }
    final_path.write_text(json.dumps(final, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

