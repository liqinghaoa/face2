from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from p2_counterfactual.a3_v2_assets import MANIFEST_PATH as A3_V2_CASE_MANIFEST, build_a3_v2_assets
from p2_counterfactual.assets import PRESET_NAMES, ROOT
from p2_counterfactual.io_utils import json_safe, sha256_file
from p2_counterfactual.path_utils import resolve_project_path


DISPLAY_EXPERIMENT_ID = "P2-A3-v3_R3DPR_PairwiseConsistency_MeanBG"
EXPERIMENT_ID = "p2_a3_v3_r3dpr_pairwise_consistency_meanbg"
DATA_ROOT = ROOT / "data/processed/global_face/SixRelighting_OriginalCamera_meanfg"
ASSET_DIR = DATA_ROOT / "p2_a3_v3_assets"
PAIR_MANIFEST_PATH = ASSET_DIR / "p2_a3_v3_pair_manifest.csv"
PAIR_AUDIT_PATH = ASSET_DIR / "p2_a3_v3_pair_audit.json"
PRESET_MAPPING_PATH = ASSET_DIR / "p2_a3_v3_preset_mapping.json"
PAIR_CYCLE_SCHEDULE_PATH = ASSET_DIR / "p2_a3_v3_pair_cycle_schedule.json"
CODE_AUDIT_PATH = ASSET_DIR / "code_audit.md"
TEST_RESULTS_PATH = ASSET_DIR / "test_results.txt"


SH_IDS = [93, 62, 28, 39, 23, 81]


def pair_cycle_order(case_id: str, *, seed: int = 2026, cycle_index: int = 0) -> list[str]:
    key = f"{int(seed)}:{str(case_id)}:{int(cycle_index)}".encode("utf-8")
    rng_seed = int(hashlib.sha256(key).hexdigest()[:16], 16) % (2**32)
    rng = __import__("numpy").random.default_rng(rng_seed)
    return [PRESET_NAMES[index] for index in rng.permutation(len(PRESET_NAMES)).tolist()]


def _image_ok(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return rgb.mode == "RGB" and rgb.size == (224, 224)


def _write_code_audit() -> None:
    lines = [
        "# P2-A3-v3 PairwiseConsistency Code Audit",
        "",
        "## Old Full-Six Implementation",
        "",
        "`P2-A3-v2_R3DPR_Full6Consistency_MeanBG` uses full-seven-view training: `[B,7,3,H,W]` is reshaped to `[7B,3,H,W]`, and each update averages six relight CE, six JS terms, and six feature-cosine terms.",
        "",
        "## Reused Modules",
        "",
        "- A2-v2/A3-v2 R3DPR image-directory manifest and path resolution.",
        "- Shared ResNet18 single-image model.",
        "- Original-only evaluator and six-relighted evaluator.",
        "- OOF aggregation, stability metrics, fold validation, and patient-cluster bootstrap.",
        "- Symmetric JS divergence, feature cosine loss, and 5-epoch warm-up.",
        "",
        "## Bypassed Full-Six Path",
        "",
        "A3-v3 uses `input_mode: pairwise_consistency`; the trainer concatenates `[B,3,H,W]` original and `[B,3,H,W]` relighted tensors into `[2B,3,H,W]`. It does not use the existing `full6_consistency` branch.",
        "",
        "## New Components",
        "",
        "- Pair manifest and audit: `p2_counterfactual.a3_v3_assets`.",
        "- Pair-cycle case-level schedule: deterministic six-preset permutation per case and six-epoch cycle.",
        "- Dataset branch: `P2ASingleRGBDataset(input_mode='pairwise_consistency')`.",
        "- Loss branch: `p2_a_pairwise_loss` uses unified batch CE plus pairwise JS/cosine.",
        "",
        "## Output Isolation",
        "",
        "Config, checkpoints, OOF files, comparisons, and reports use `p2_a3_v3_r3dpr_pairwise_consistency_meanbg` under `experiments/500Data/P2_A3_v3_R3DPR_PairwiseConsistency_MeanBG`, preserving A2-v2 and A3-v2 outputs.",
    ]
    CODE_AUDIT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def build_a3_v3_assets() -> dict[str, Any]:
    if not A3_V2_CASE_MANIFEST.is_file():
        build_a3_v2_assets()
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    case_frame = pd.read_csv(A3_V2_CASE_MANIFEST, dtype={"case_id": str, "patient_group_id": str})
    rows = []
    invalid_paths = []
    for row in case_frame.itertuples(index=False):
        case_id = str(row.case_id)
        original_path = str(row.original_path)
        if not _image_ok(resolve_project_path(original_path, ROOT, require_exists=True)):
            invalid_paths.append({"case_id": case_id, "path": original_path})
        for preset_index, (preset, sh_id) in enumerate(zip(PRESET_NAMES, SH_IDS)):
            relighted_path = str(getattr(row, f"relight_{preset}_path"))
            if not _image_ok(resolve_project_path(relighted_path, ROOT, require_exists=True)):
                invalid_paths.append({"case_id": case_id, "path": relighted_path})
            rows.append(
                {
                    "pair_id": f"{case_id}__{preset}",
                    "case_id": case_id,
                    "patient_group_id": str(row.patient_group_id),
                    "binary_label": int(row.binary_label),
                    "fold_id": int(row.fold),
                    "original_path": original_path,
                    "relighted_path": relighted_path,
                    "preset_id": preset,
                    "preset_index": int(preset_index),
                    "source_sh_id": int(sh_id),
                    "p1_qc_flag": bool(getattr(row, "p1_qc_flag", False)),
                    "boundary_uncertain": bool(getattr(row, "boundary_uncertain", False)),
                }
            )
    pair_frame = pd.DataFrame(rows)
    pair_frame.to_csv(PAIR_MANIFEST_PATH, index=False, encoding="utf-8-sig")
    mapping = {
        "display_experiment_id": DISPLAY_EXPERIMENT_ID,
        "preset_order": [
            {
                "preset_index": index,
                "preset_id": preset,
                "source_sh_id": sh_id,
                "filename": f"relight_{preset}.png",
            }
            for index, (preset, sh_id) in enumerate(zip(PRESET_NAMES, SH_IDS))
        ],
        "fixed_order_source": "p2_counterfactual.assets.PRESET_NAMES",
        "filenames_contain_sh_ids": False,
    }
    PRESET_MAPPING_PATH.write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    first_cycle = {
        str(row.case_id): pair_cycle_order(str(row.case_id), seed=2026, cycle_index=0)
        for row in case_frame.itertuples(index=False)
    }
    schedule = {
        "display_experiment_id": DISPLAY_EXPERIMENT_ID,
        "experiment_id": EXPERIMENT_ID,
        "seed": 2026,
        "cycle_length_epochs": 6,
        "rule": "For each case and cycle, sha256(seed:case_id:cycle_index) seeds a six-preset permutation; epoch position selects one preset.",
        "preset_order": list(PRESET_NAMES),
        "first_cycle_by_case": first_cycle,
    }
    PAIR_CYCLE_SCHEDULE_PATH.write_text(json.dumps(schedule, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_code_audit()
    pair_counts = pair_frame.groupby("case_id").size()
    preset_counts = pair_frame.groupby("case_id")["preset_id"].nunique()
    patient_fold_violations = int(pair_frame.groupby("patient_group_id")["fold_id"].nunique().gt(1).sum())
    audit = {
        "display_experiment_id": DISPLAY_EXPERIMENT_ID,
        "experiment_id": EXPERIMENT_ID,
        "pair_manifest_path": str(PAIR_MANIFEST_PATH),
        "pair_manifest_sha256": sha256_file(PAIR_MANIFEST_PATH),
        "source_case_manifest_path": str(A3_V2_CASE_MANIFEST),
        "source_case_manifest_sha256": sha256_file(A3_V2_CASE_MANIFEST),
        "case_count": int(pair_frame["case_id"].nunique()),
        "pair_count": int(len(pair_frame)),
        "pairs_per_case_min": int(pair_counts.min()),
        "pairs_per_case_max": int(pair_counts.max()),
        "unique_presets_per_case_min": int(preset_counts.min()),
        "unique_presets_per_case_max": int(preset_counts.max()),
        "fold_counts_by_case": {str(int(k)): int(v) for k, v in case_frame["fold"].astype(int).value_counts().sort_index().to_dict().items()},
        "fold_counts_by_pair": {str(int(k)): int(v) for k, v in pair_frame["fold_id"].astype(int).value_counts().sort_index().to_dict().items()},
        "patient_group_fold_violations": patient_fold_violations,
        "invalid_paths": invalid_paths,
        "old_deca_npz_columns_used": False,
        "status": "passed"
        if len(pair_frame) == 3000
        and pair_frame["case_id"].nunique() == 500
        and pair_counts.min() == 6
        and pair_counts.max() == 6
        and preset_counts.min() == 6
        and preset_counts.max() == 6
        and patient_fold_violations == 0
        and not invalid_paths
        else "failed",
    }
    PAIR_AUDIT_PATH.write_text(json.dumps(json_safe(audit), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if audit["status"] != "passed":
        raise RuntimeError(f"A3-v3 pair audit failed: {PAIR_AUDIT_PATH}")
    return audit


def main() -> None:
    print(json.dumps(json_safe(build_a3_v3_assets()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
