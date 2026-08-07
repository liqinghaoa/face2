from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from skin_optics_so1.synthetic_storage import prepare_split_storage, write_sample


def test_memmap_storage_write_sample(tmp_path: Path) -> None:
    stores = prepare_split_storage(tmp_path, {"train": 2}, 16, overwrite=True)
    linear = np.ones((3, 16, 16), dtype=np.float32) * 0.5
    target = np.ones((4, 16, 16), dtype=np.float32) * 0.25
    mask = np.ones((16, 16), dtype=np.uint8)
    write_sample(stores, "train", 1, linear, target, mask)
    assert int(stores["train"]["written"][1]) == 1
    assert stores["train"]["linear_rgb"].dtype == np.float16
    assert stores["train"]["target_mhsp"].shape == (2, 4, 16, 16)


def test_cli_validate_only_smoke_config(tmp_path: Path) -> None:
    env = os.environ.copy()
    root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root / "src"), str(root / "src/skin_optics_so0/src")]
    )
    out = tmp_path / "validate"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "skin_optics_so1.build_synthetic_dataset",
            "--config",
            str(root / "configs/so1_synthetic_smoke_v1.yaml"),
            "--output-root",
            str(out),
            "--validate-only",
            "--overwrite-confirmed",
        ],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert (out / "synthetic_manifest.csv").exists()
    assert (out / "camera_light_split.json").exists()
    assert (out / "dataset_statistics.json").exists()
