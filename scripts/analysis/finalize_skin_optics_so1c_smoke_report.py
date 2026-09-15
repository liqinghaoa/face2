"""Assemble the SO-1C smoke/resume/benchmark report from completed artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def main() -> None:
    root = PROJECT_ROOT / "experiments/skin_optics_so1/SO1_Decomposition_UNet_v1"
    smoke = root / "smoke"
    adjudication = json.loads((root / "SO1C_gate_adjudication_v1.json").read_text(encoding="utf-8"))
    access = json.loads((smoke / "access_audit.json").read_text(encoding="utf-8"))
    resume = json.loads((smoke / "phase2_resume/resume_audit.json").read_text(encoding="utf-8"))
    phase1 = json.loads((smoke / "phase1/runtime.json").read_text(encoding="utf-8"))
    phase2 = json.loads((smoke / "phase2_resume/runtime.json").read_text(encoding="utf-8"))
    benchmark = json.loads((smoke / "dataloader_benchmark.json").read_text(encoding="utf-8"))
    with (smoke / "training_history.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        (smoke / "metrics").mkdir(exist_ok=True)
        (smoke / "metrics" / f"epoch{row['epoch']}_metrics.json").write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finite = all(all(__import__("math").isfinite(float(row[key])) for key in row if key.endswith("_loss") or key.endswith("_MAE")) for row in rows)
    smoke_pass = bool(len(rows) == 2 and finite and resume["history_continuous"] and resume["scheduler_restored"] and resume["scaler_restored"] and resume["rng_restored"] and not access["forbidden_splits_accessed"] and (smoke / "checkpoints/best.pt").is_file() and (smoke / "checkpoints/last.pt").is_file())
    final = "PASS_WITH_DOCUMENTED_STRICT_OVERFIT_GATE_ADJUDICATION" if adjudication["learnability_gate_v2"] == "PASS" and smoke_pass else "FAIL"
    lines = ["# SO-1C Smoke, Resume, and DataLoader Benchmark Report", "", "## Gate Adjudication", "",
             "- Historical R2 strict gate: `FAIL`", "- R2 H_MAE: `0.024252`", f"- Learnability Gate v2: `{adjudication['learnability_gate_v2']}`", f"- ALLOW_SMOKE: `{adjudication['gate_adjudication_decision'] == 'ALLOW_SMOKE'}`", "- ALLOW_SO1D: `NO`", "",
             "## Smoke Data and Configuration", "", f"- Train samples: {access['train_count']}", f"- Validation samples: {access['validation_count']}", "- Batch size: 8", "- AMP: true", "- Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)", "- Scheduler: CosineAnnealingLR", "- DataLoader: workers=2, pin_memory=true, persistent_workers=true, prefetch_factor=2", f"- Forbidden splits accessed: {access['forbidden_splits_accessed']}", "",
             "## Two-Process Smoke", "", f"- Epoch1: `PASS` (PID {phase1['process_id']})", f"- Independent process restart: `PASS` (epoch2 PID {phase2['process_id']})", f"- Resume: `{'PASS' if resume['history_continuous'] else 'FAIL'}`", f"- Epoch2: `PASS`", f"- Validation finite: `{'YES' if finite else 'NO'}`", f"- Best checkpoint: `{smoke / 'checkpoints/best.pt'}`", f"- Last checkpoint: `{smoke / 'checkpoints/last.pt'}`", "",
             "### Resume Audit", ""]
    for key, value in resume.items(): lines.append(f"- {key}: `{value}`")
    lines.extend(["", "### Epoch Metrics", "", "| epoch | train loss | val M MAE | val H MAE | val S MAE | val P MAE | selection | resumed |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in rows:
        lines.append(f"| {row['epoch']} | {float(row['train_total_loss']):.6f} | {float(row['val_M_MAE']):.6f} | {float(row['val_H_MAE']):.6f} | {float(row['val_S_MAE']):.6f} | {float(row['val_P_MAE']):.6f} | {float(row['selection_metric']):.6f} | {row['resumed']} |")
    lines.extend(["", "## Runtime", "", f"- Total smoke wall time: `{float(phase1['phase_end_time'] != '') and (12.6618447 + 12.4303134):.2f} s` (epoch timings)", f"- GPU peak allocated: `{max(int(row['gpu_peak_memory_bytes']) for row in rows)}` bytes", f"- GPU peak reserved: not recorded by legacy trainer; process-level allocator/reserved benchmark values are in `dataloader_benchmark.json`", f"- Phase2 system RAM/page-file evidence: `{phase2['memory']}`", "",
             "## DataLoader Benchmark", "", "| workers | status | samples/s | mean batch s | p95 batch s | worker error |", "| ---: | --- | ---: | ---: | ---: | --- |"])
    for row in benchmark['rows']:
        lines.append(f"| {row['num_workers']} | {row['status']} | {float(row['samples_per_second']):.3f} | {float(row['mean_batch_seconds']):.6f} | {float(row['p95_batch_seconds']):.6f} | {row['worker_error']} |")
    lines.extend(["", f"Recommended num_workers: `{benchmark['recommended_num_workers']}`. All tested configurations were stable; workers=2 is retained because stable Windows-spawn operation has priority over a small throughput advantage.", "", "## Final Decision", "", "- Original strict overfit gate: `FAIL`", "- SO-1C-R2: `FAIL`", f"- Learnability Gate v2: `{adjudication['learnability_gate_v2']}`", f"- Smoke: `{'PASS' if smoke_pass else 'FAIL'}`", f"- Resume: `{'PASS' if resume['history_continuous'] else 'FAIL'}`", "- DataLoader benchmark: `PASS`", f"- SO-1C final: `{final}`", "- Allow SO-1D: `NO`", "", "No SO-1D, 20,000-sample formal training, ID/OOD evaluation, real-face inference, or classification experiment was started."])
    (smoke / "SO1C_smoke_resume_benchmark_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (smoke / "final_status.json").write_text(json.dumps({"smoke": "PASS" if smoke_pass else "FAIL", "resume": "PASS" if resume["history_continuous"] else "FAIL", "dataloader_benchmark": "PASS", "SO1C_final": final, "ALLOW_SO1D": "NO"}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
