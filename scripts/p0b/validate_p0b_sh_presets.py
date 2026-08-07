"""Validate frozen directional SH presets on a synthetic normal sphere."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from p0b_deca.sh_lighting import DirectionalToSHProjector, render_sh


def _display(shading: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Produce a display-only clipped grayscale PNG; raw shading is retained in metrics."""
    value = np.zeros(shading.shape, dtype=np.uint8)
    value[valid] = np.rint(np.clip(shading[valid], 0.0, 1.0) * 255).astype(np.uint8)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config/p0b/p0b_relighting_presets_v1.yaml")
    args = parser.parse_args()
    raw_bytes = args.config.read_bytes()
    raw = yaml.safe_load(raw_bytes)
    config_hash = hashlib.sha256(raw_bytes).hexdigest()
    output = ROOT / "data/processed/P0B_DECA_Pilot12_v1/environment/sh_validation"
    output.mkdir(parents=True, exist_ok=True)

    size = 256
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    z = np.sqrt(np.clip(1 - x * x - y * y, 0, None))
    normals = np.stack((x, y, z), axis=-1)
    valid = (x * x + y * y) <= 1
    sphere = np.zeros((size, size, 3), dtype=np.uint8)
    sphere[valid] = np.rint((normals[valid] + 1.0) * 127.5).astype(np.uint8)
    cv2.imwrite(str(output / "synthetic_normal_sphere.png"), cv2.cvtColor(sphere, cv2.COLOR_RGB2BGR))

    projector = DirectionalToSHProjector(int(raw["sample_count"]))
    rows = []
    for name, value in raw["presets"].items():
        coefficients, rmse = projector.fit(np.asarray(value["direction"]), float(value["ambient"]), float(value["diffuse"]))
        shading = render_sh(normals, coefficients).mean(axis=-1)
        maximum = np.unravel_index(np.argmax(np.where(valid, shading, -np.inf)), shading.shape)
        rows.append({
            "preset_name": name, "fit_rmse": rmse, "mean_shading": float(shading[valid].mean()),
            "max_x": float(x[maximum]), "max_y": float(y[maximum]),
            "negative_fraction": float((shading[valid] < 0).mean()), "above_one_fraction": float((shading[valid] > 1).mean()),
            "config_sha256": config_hash,
        })
        np.save(output / f"{name}_coefficients.npy", coefficients)
        cv2.imwrite(str(output / f"{name}_shading.png"), _display(shading, valid))

    table = pd.DataFrame(rows)
    table.to_csv(output / "sh_preset_validation.csv", index=False)
    lookup = table.set_index("preset_name")
    passed = (
        lookup.loc["dim_front", "mean_shading"] < lookup.loc["neutral_front", "mean_shading"] < lookup.loc["bright_front", "mean_shading"]
        and lookup.loc["left", "max_x"] < 0 and lookup.loc["right", "max_x"] > 0 and lookup.loc["top", "max_y"] < 0
    )
    (output / "sh_coordinate_convention.json").write_text(json.dumps({
        "passed": bool(passed), "basis": "canonical_real_second_order_sh", "renderer_basis_verified": False, "image_x": "positive right", "image_y": "positive down",
        "left_preset_peak_x": float(lookup.loc["left", "max_x"]), "right_preset_peak_x": float(lookup.loc["right", "max_x"]),
        "top_preset_peak_y": float(lookup.loc["top", "max_y"]), "config_sha256": config_hash,
    }, indent=2), encoding="utf-8")
    print({"passed": bool(passed), "rows": len(rows)})
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
