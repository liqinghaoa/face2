from __future__ import annotations

from pathlib import Path

P2_A_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config" / "p2" / "p2_a"

P2_A_EXPERIMENTS = {
    "p2_a1_colorjitter": P2_A_CONFIG_DIR / "p2_a1_colorjitter.yaml",
    "p2_a2_relighting": P2_A_CONFIG_DIR / "p2_a2_relighting.yaml",
    "p2_a3_full_consistency": P2_A_CONFIG_DIR / "p2_a3_full_consistency.yaml",
}


def get_p2_a_config_path(experiment_id: str) -> Path:
    try:
        return P2_A_EXPERIMENTS[str(experiment_id)]
    except KeyError as exc:
        raise KeyError(f"Unknown P2-A experiment_id: {experiment_id}") from exc
