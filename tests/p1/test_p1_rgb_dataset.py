from pathlib import Path

from datasets.nyha_3class_face_dataset import build_transforms
from datasets.p1_rgb_binary_dataset import P1RGBBinaryDataset
from utils.p1_rgb_audit import EXPECTED_SPLIT_SHA, load_p1_frame, sha

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data/processed/P1_Component_Audit_v1/manifests/p1_master_manifest.csv"
SPLIT = ROOT / "data/processed/P0_Physics_Audit_v1/splits_500/nyha_3class_sex_stratified_group_5fold.csv"


def test_p1_master_and_fixed_split_are_the_only_500_case_source():
    frame = load_p1_frame(MANIFEST, SPLIT)
    assert sha(SPLIT) == EXPECTED_SPLIT_SHA
    assert len(frame) == frame.case_id.nunique() == 500
    assert frame.patient_group_id.nunique() == 483
    assert (frame.group_id.astype(str) == frame.patient_group_id.astype(str)).all()
    assert frame.groupby("patient_group_id").fold.nunique().max() == 1
    assert frame.groupby("patient_group_id").size().sum() == 500
    for fold in range(5):
        val = frame[frame.fold == fold]
        train = frame[frame.fold != fold]
        assert (len(train), len(val)) == (400, 100)
        assert ((train.label_binary == 0).sum(), (train.label_binary == 1).sum()) == (92, 308)
        assert ((val.label_binary == 0).sum(), (val.label_binary == 1).sum()) == (23, 77)


def test_rgb_dataset_item_shape_and_fields_are_stable():
    frame = load_p1_frame(MANIFEST, SPLIT)
    ds = P1RGBBinaryDataset(frame.head(1), build_transforms("val"))
    item = ds[0]
    assert set(item) == {
        "image",
        "label_binary",
        "case_id",
        "patient_group_id",
        "fold",
        "label_original",
        "label_3class",
        "image_path",
    }
    assert tuple(item["image"].shape) == (3, 224, 224)
    assert item["case_id"] == frame.case_id.iloc[0]
