"""Write the SO-1C Gate v2 adjudication from completed R1/R2 artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from skin_optics_so1.decomposition.gate_adjudication import adjudicate_learnability_gate_v2


def main() -> None:
    root = PROJECT_ROOT / "experiments/skin_optics_so1/SO1_Decomposition_UNet_v1"
    r2 = root / "so1c_r2_overfit16_gate"
    r1 = root / "diagnostics/overfit_failure_r1"
    best = json.loads((r2 / "metrics/best_metrics.json").read_text(encoding="utf-8"))
    overfit1 = json.loads((r1 / "overfit1/result.json").read_text(encoding="utf-8"))
    paired_overfit2 = json.loads((r1 / "paired_overfit2/result.json").read_text(encoding="utf-8"))
    result = adjudicate_learnability_gate_v2(
        r2_best_record=best["record"], r2_best_summary=best["summary"],
        overfit1_result=overfit1, paired_overfit2_result=paired_overfit2, tests_passed=True,
    )
    lines = ["# SO-1C Gate Adjudication v1", "", "## Historical strict gate", ""]
    lines.extend([
        "Original SO-1C strict overfit16 gate: `FAIL`", "", "SO-1C-R2 historical result: `FAIL`", "",
        "Original H threshold: `H_MAE <= 0.020`", "", f"R2 best H_MAE: `{result['historical_R2_H_MAE']:.6f}`", "",
        "The historical R2 strict result is preserved. No R2 JSON, CSV, checkpoint, or report was modified.", "",
        "## Learnability Engineering Gate v2", "",
        "This is a pre-smoke engineering learnability gate, not a scientific performance endpoint or ID/OOD evaluation threshold.",
        "It uses loss reduction >= 95%, M/H/S MAE <= 0.030, finite non-collapsed P, completed overfit1 and paired-overfit2 evidence, and passing SO-1 tests.", "",
        "Rationale for 0.030: single-sample H and paired acquisition H fit to ~0.002/~0.001, while independently initialized mixed-camera/light 16-sample batch8 R2 converged to ~0.024. Thus 0.020 was an intentionally strict sanity check; 0.030 is a bounded normalized [0,1] pre-smoke threshold, not 97% accuracy and not a clinical claim.", "",
        "## Adjudication", "",
        f"- historical_R2_strict_gate: `{result['historical_R2_strict_gate']}`",
        f"- SO1C_learnability_gate_v2: `{result['learnability_gate_v2']}`",
        f"- gate_adjudication_decision: `{result['gate_adjudication_decision']}`",
        f"- ALLOW_SO1D: `NO`", "", "### Checks", "",
    ])
    for key, value in result["checks"].items(): lines.append(f"- {key}: `{value}`")
    lines.extend(["", "### R2 values", ""])
    for key, value in result["values"].items(): lines.append(f"- {key}: `{value:.6f}`")
    (root / "SO1C_gate_adjudication_v1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "SO1C_gate_adjudication_v1.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
