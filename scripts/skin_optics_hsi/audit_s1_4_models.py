"""Build the immutable S1-4 candidate-model contract and unit audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from skin_optics_hsi.model_registry import DEFAULT_MODEL_REGISTRY_PATH  # noqa: E402
from skin_optics_hsi.s1_model_audit import build_s1_4_model_audit  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s1-3-decision", required=True)
    parser.add_argument("--data-contract", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--registry", default=str(DEFAULT_MODEL_REGISTRY_PATH))
    parser.add_argument("--supersedes-decision")
    args = parser.parse_args()
    decision = build_s1_4_model_audit(
        args.s1_3_decision,
        args.data_contract,
        args.output_dir,
        args.registry,
        args.supersedes_decision,
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0 if decision["status"] == "PASS_FOR_S1_5" else 2


if __name__ == "__main__":
    raise SystemExit(main())
