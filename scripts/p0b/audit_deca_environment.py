"""Write the B0 environment evidence and return 2 when the hard gate is blocked."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from p0b_deca.config import load_config
from p0b_deca.environment_audit import audit_environment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config/p0b/p0b_deca_environment_v1.yaml")
    args = parser.parse_args()
    config = load_config(args.config, ROOT)
    output = config.output_root / "environment"
    output.mkdir(parents=True, exist_ok=True)
    result = audit_environment(config)

    canonical = output / "p0b_deca_environment_audit.json"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if canonical.exists():
        archive = output / f"p0b_deca_environment_audit.pre_{stamp}.json"
        counter = 1
        while archive.exists():
            archive = output / f"p0b_deca_environment_audit.pre_{stamp}_{counter}.json"
            counter += 1
        shutil.copy2(canonical, archive)
    timestamped = output / f"p0b_deca_environment_audit.{stamp}.json"
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    timestamped.write_text(serialized, encoding="utf-8")
    canonical.write_text(
        serialized, encoding="utf-8"
    )
    with (output / "p0b_deca_asset_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "path", "exists", "size_bytes", "sha256", "load_status", "required"])
        writer.writeheader()
        writer.writerows(result["assets"])
    lines = [
        "# P0-B0 environment audit",
        "",
        f"Passed: `{result['passed']}`",
        "",
        f"Blocked: `{result['blocked']}`",
        "",
        f"Block reason: `{result['block_reason']}`",
        "",
        "## Check results",
        "",
        *[f"- `{name}`: `{value}`" for name, value in result["checks"].items()],
    ]
    (output / "p0b_deca_environment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    metadata = config.output_root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    with (metadata / "environment_status.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audit_stage", "passed", "blocked", "block_reason", "license_status"])
        writer.writeheader()
        writer.writerow({"audit_stage": "P0-B0", "passed": result["passed"], "blocked": result["blocked"], "block_reason": result["block_reason"], "license_status": result["license_status"]})
    print(result)
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
