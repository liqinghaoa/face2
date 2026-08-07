from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import json_safe


def manual_review_template(pilot: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "sample_id",
        "core_skin_e0_ok",
        "core_skin_e2_ok",
        "core_skin_e4_ok",
        "forehead_roi_ok",
        "canvas_left_cheek_roi_ok",
        "canvas_right_cheek_roi_ok",
        "nose_roi_ok",
        "raw_mapping_ok",
        "bright_candidate_visual_match",
        "dark_candidate_visual_match",
        "shadow_candidate_visual_match",
        "specular_candidate_visual_match",
        "preferred_skin_erosion",
        "preferred_bright_threshold",
        "preferred_dark_threshold",
        "preferred_shadow_ratio",
        "preferred_specular_y_threshold",
        "preferred_specular_chroma_threshold",
        "reviewer_notes",
        "final_review_status",
    ]
    out = pd.DataFrame({"sample_id": pilot["sample_id"]})
    for col in cols:
        if col not in out:
            out[col] = ""
    return out[cols]


def write_report(output_dir: Path, summary: dict[str, Any]) -> None:
    reports = output_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "stage2_pilot32_machine_summary.json").write_text(json.dumps(json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Lighting QC Stage2 Pilot32 Report",
        "",
        "## 1. Pilot目的",
        "开发图像级光照质量控制指标和候选阈值分布；本轮只运行 Pilot32。",
        "",
        "## 2. 与方案B的关系",
        "直接读取冻结 RealFace 256x320 方案B产物，不重新检测、不重新仿射、不修改方案B图片。",
        "",
        "## 3. 输入资产和冻结约束",
        f"- scheme_b_root: `{summary['inputs']['scheme_b_root']}`",
        f"- raw_scene_dir: `{summary['inputs']['raw_scene_dir']}`",
        f"- split_csv: `{summary['inputs']['split_csv']}`",
        f"- exif_xlsx: `{summary['inputs']['exif_xlsx']}`",
        "",
        "## 4. Pilot32选择方法",
        "按 fold×label 覆盖、EXIF极端、方案B warning/几何边缘和固定随机补足确定性选择；未使用OOF预测挑选样本。",
        "",
        "## 5. parsing类别映射",
        "`preflight/detected_parsing_class_map.json` 记录从 legacy parser 源码解析得到的类别映射。",
        "",
        "## 6-8. core skin、ROI和颜色方法",
        "core skin = parsing skin 类 AND source_valid AND face_valid；鼻部单独作为 ROI，不进入 core skin 主指标。sRGB 按标准 inverse transfer 转线性 RGB，并用相对亮度 Y 计算主指标。",
        "",
        "## 9-11. 输入空间、原图空间和反映射",
        "连续指标见 `metrics/`；canvas mask 通过方案B affine 矩阵反映射到原图，原图指标仅在反映射 mask 内计算。",
        "",
        "## 12. ROI面积与有效性分布",
        f"- invalid_roi_rows: {summary['roi']['invalid_roi_rows']}",
        f"- min_core_skin_e2_pixels: {summary['roi']['min_core_skin_e2_pixels']}",
        "",
        "## 13. 腐蚀尺度比较",
        "erosion 0/2/4 均已输出，不冻结最终腐蚀尺度。",
        "",
        "## 14-17. 候选阈值分布",
        "过亮、欠曝、阴影ratio、镜面高光候选组合的分布见 `reports/threshold_candidate_summary.csv`。",
        "",
        "## 18. 原图和256x320一致性",
        f"- min_roundtrip_iou: {summary['roundtrip']['min_iou']}",
        f"- max_centroid_distance: {summary['roundtrip']['max_centroid_distance']}",
        "一致性表见 `metrics/pilot32_raw_input_consistency.csv` 和 `metrics/pilot32_consistency_summary.csv`。",
        "",
        "## 19. 极端样本清单",
        "`reports/pilot32_extreme_cases.csv`。",
        "",
        "## 20. 需要人工确认的阈值",
        "需要人工审查 skin erosion、bright/dark threshold、shadow ratio、specular Y/chroma 组合；本轮没有冻结阈值。",
        "",
        "## 21. 已知限制",
        "无FaceMesh关键点重算，ROI为固定对齐空间候选区域；镜面高光只是候选，不等同真实物理镜面分量。",
        "",
        "## 22. Full-500前门控清单",
        "人工审查QC图、确认ROI方向和阈值候选、修订配置后，才能进入下一轮阈值冻结；本轮停止于Pilot32。",
    ]
    (reports / "stage2_pilot32_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
