from __future__ import annotations

import pandas as pd

from skin_optics_so1.decomposition.diagnostic_selection import select_overfit1, select_paired_overfit2


def _metadata() -> pd.DataFrame:
    rows = []
    for index in range(4):
        rows.append({
            "sample_id": f"train_{index:06d}", "split_index": index,
            "base_latent_id": index // 2, "acquisition_variant_id": index % 2,
            "generation_status": "SUCCESS", "retry_count": 0, "total_clip_fraction": 0.0,
            "m_std": 0.1, "h_std": 0.1, "m_mean": 0.5, "h_mean": 0.5,
            "s_mean": 0.5, "s_std": 0.1, "p_mean": 0.01, "p_std": 0.01,
            "p_nonzero_fraction": 0.1, "valid_mask_type": "full",
            "m_seed": 1 if index < 2 else 2, "h_seed": 3 if index < 2 else 4,
            "mask_seed": 5 if index < 2 else 6, "s_seed": 10 + index,
            "p_seed": 20 + index, "final_camera_light_pair": f"pair{index}",
        })
    return pd.DataFrame(rows)


def test_overfit1_selection_reuses_existing_id_file(tmp_path) -> None:
    path = tmp_path / "overfit1_id.txt"
    first = select_overfit1(_metadata(), path)
    second = select_overfit1(_metadata().iloc[0:0], path)
    assert first == second
    assert (tmp_path / "overfit1_selection.json").is_file()


def test_paired_selection_is_deterministic_and_persists_requested_name(tmp_path) -> None:
    path = tmp_path / "paired_overfit2_ids.txt"
    selected = select_paired_overfit2(_metadata(), path)
    assert selected == ["train_000000", "train_000001"]
    assert (tmp_path / "paired_overfit2_selection.json").is_file()
