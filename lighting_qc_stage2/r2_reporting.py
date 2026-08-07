from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def write_r2_reports(r2_root: Path, summary: dict[str, Any]) -> None:
    reports = r2_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    repair = summary["repair"]
    exif = summary["exif"]
    lines = [
        "# Stage2 R2 Statistical Repair and EXIF Decomposition",
        "",
        "## 1. R2 Purpose",
        "R2 repairs Stage2-Full500 statistical reporting and adds fixed-fivefold cross-fitted EXIF decomposition. Traditional EXIF covariate adjustment was intentionally not performed in R2. The analysis focused on cross-fitted decomposition into EXIF-predicted and EXIF-residual components.",
        "",
        "## 2-3. Frozen Inputs and Constraints",
        "The frozen Stage2-R1 image metrics were not modified. R2 did not re-extract image features, did not train or modify the RGB classifier, did not delete samples, and did not enter Stage3.",
        "",
        "## 4-8. Erosion Stratification Repair",
        "Erosion 0/2/4 sensitivity stratification was regenerated with fold-specific train-only q33/q67 cutpoints. Stable groups require n_total>=30, n_control>=20, n_patient>=20, and n_patient_groups>=25; severely unstable groups have either class below 10 or a missing class.",
        f"- all-group max DeltaAUC: {repair['all_group_max_delta_auc']}",
        f"- stable-only max DeltaAUC: {repair['stable_only_max_delta_auc']}",
        f"- stable-only max DeltaAUC CI: {repair['stable_only_max_delta_auc_ci']}",
        "",
        "## 9. FDR Hit Deduplication",
        f"- RGB logit single-model significant unique metrics: {repair['rgb_logit_single_sig']}",
        f"- RGB error single-model significant unique metrics: {repair['rgb_error_single_sig']}",
        f"- RGB error multivariable independent significant terms: {repair['rgb_error_multi_sig']}",
        "",
        "## 10. Revised Dependence Terminology",
        f"- dependence term: {repair['dependence_term']}",
        f"- dependence level: {repair['dependence_level']}",
        "The correct wording is high image luminance-associated dependence risk, not proven environmental-lighting dependence.",
        "",
        "## 11-14. EXIF Fields and Cross-fitting",
        "The EXIF Ridge model used only log2_exposure_time, log2_iso, brightness_value, and missing indicators. It did not use label, RGB prediction, camera model, device ID, acquisition date, or batch. Ridge alpha was fixed at 1.0.",
        exif["performance_markdown"],
        "",
        "## 15-19. EXIF Components",
        "EXIF-predicted and EXIF-residual components were analyzed separately for label association, RGB logit, RGB error risk, Brier contribution, and attenuation. Residuals denote variation unexplained by the recorded EXIF exposure fields.",
        "",
        "## 20-23. Interpretation and Stage3",
        f"- revised Stage3 recommendation: {summary['stage3_recommendation']}",
        "Stage3 should prioritize controlled exposure/gamma perturbations where EXIF-predicted components explain variance, and local/appearance perturbations where residual component risks remain.",
        "",
        "## 24-26. Limitations and Non-actions",
        "Image luminance metrics mix illumination, exposure, camera ISP, skin tone/appearance, true phenotype, and unrecorded acquisition factors. Observational associations are not causal. No traditional EXIF covariate adjustment, no EXIF+RGB classifier, no RGB retraining, and no Stage3 execution were performed.",
    ]
    (reports / "stage2_r2_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (reports / "stage2_r2_machine_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (reports / "stage2_r2_revised_stage3_decision.md").write_text(
        "# Stage2 R2 Revised Stage3 Decision\n\n"
        f"- dependence term: {repair['dependence_term']}\n"
        f"- dependence level: {repair['dependence_level']}\n"
        f"- recommendation: {summary['stage3_recommendation']}\n"
        "- Traditional EXIF covariate adjustment was not performed.\n"
        "- Stage3 was not executed.\n",
        encoding="utf-8",
    )
    inventory = {str(p.relative_to(r2_root)): p.stat().st_size for p in r2_root.rglob("*") if p.is_file()}
    (reports / "stage2_r2_output_inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")


def markdown_table(frame: pd.DataFrame, cols: list[str]) -> str:
    return frame[cols].to_markdown(index=False)
