from src.skin_optics_so_r1.full_generation_g1_am1 import _replay_selection


def test_replay_selection_is_fixed_and_split_balanced():
    quotas = {"Train": 12, "Validation": 4, "ID Test": 4, "Camera-OOD": 4, "Light-OOD": 4, "Joint-OOD": 4}
    rows = [
        {"latent_id": f"{split}-{index}", "split": split, "split_index": index}
        for split, count in quotas.items() for index in range(count + 4)
    ]
    selected = _replay_selection(rows)
    assert len(selected) == 32
    assert _replay_selection(rows) == selected
    assert {split: sum(row["split"] == split for row in selected) for split in quotas} == quotas
