from __future__ import annotations

import subprocess
import sys


def test_validate_only_smoke() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "preprocessing/build_skin_optics_realface_highres_1024x1280_v1.py",
            "--config",
            "config/preprocess/skin_optics_realface_highres_1024x1280_v1.yaml",
            "--validate-only",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[validate] OK" in result.stdout
