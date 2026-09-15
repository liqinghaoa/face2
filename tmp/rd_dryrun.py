"""Cheap dry run of the R-D post-freeze code path.

The 44-subject joint freeze costs ~25 minutes and is deterministic.  This script
replaces only that estimator with a stub returning the same structure, then
executes the whole validation/direction/emission path against a scratch output
directory.  It is a development aid and is never part of the frozen artifact set.
"""

from pathlib import Path
import shutil
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.skin_optics_hsi import km_bio_v2r1_stage_rd as rd


def main() -> None:
    config = yaml.safe_load((ROOT / "configs/skin_optics_hsi/km_bio_v2r1_stage_rd.yaml").read_text(encoding="utf-8"))
    scratch = ROOT / "tmp/rd_dryrun_output"
    if scratch.exists():
        shutil.rmtree(scratch)
    config["output_directory"] = "tmp/rd_dryrun_output"
    config["solver"]["individual"]["sobol_starts"] = 2
    temp_config = ROOT / "tmp/rd_dryrun_config.yaml"
    temp_config.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")

    real = rd._shape_joint_fit

    def stub(observed, optical, wavelength, initial_theta, config):
        count = observed.shape[0]
        a_profile = [
            {"A_s": float(value), "centered_logrmse": 0.06 + abs(value - 1.483456), "solver_success": True, "nfev": 0, "grid_kind": "dryrun"}
            for value in np.linspace(0.6, 1.6, 11)
        ]
        delta_profile = [
            {"delta_bs": float(value), "centered_logrmse": 0.06 + abs(value - 0.5), "solver_success": True, "nfev": 0, "grid_kind": "dryrun"}
            for value in np.linspace(-0.5, 0.5, 11)
        ]
        return {
            "A_s": 1.483456, "delta_bs": 0.5, "theta": np.tile(np.array([0.09, 0.09]), (count, 1)),
            "centered_logrmse": 0.06, "start_index": 0, "optimizer_success": True,
            "A_s_profile": a_profile, "delta_bs_profile": delta_profile,
        }

    rd._shape_joint_fit = stub
    try:
        decision = rd.run_stage_rd(temp_config, ROOT)
        print("DRYRUN_STATUS:", decision["status"])
        print("DRYRUN_STATE:", decision["decision_state"])
        print("DIRECTION:", decision["direction_consistency"]["direction_consistent"])
        print("SPECTRAL_CHECKS:", decision["direction_consistency"]["spectral_checks"])
        print("PARAMETER_CHECKS:", decision["direction_consistency"]["parameter_checks"])
        written = sorted(path.name for path in scratch.iterdir())
        print("ARTIFACTS:", len(written))
        for name in written:
            print("  ", name)
        missing = [name for name in config["required_outputs"] if not (scratch / name).is_file()]
        print("MISSING_REQUIRED:", missing)
    finally:
        rd._shape_joint_fit = real


if __name__ == "__main__":
    main()
