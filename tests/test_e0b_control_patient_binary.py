import numpy as np
import pytest
from PIL import Image

from datasets.nyha_3class_face_dataset import build_transforms
from datasets.control_patient_binary_dataset import map_three_class_to_binary
from losses.classification_losses import compute_class_weights
from metrics.binary_classification_metrics import compute_binary_metrics


def test_binary_label_mapping_and_invalid_label():
    assert [map_three_class_to_binary(x) for x in (0, 1, 2)] == [0, 1, 1]
    with pytest.raises(ValueError): map_three_class_to_binary(3)


def test_binary_metrics_use_probability_and_argmax():
    metrics = compute_binary_metrics([0, 1, 0, 1], [[.9, .1], [.2, .8], [.7, .3], [.1, .9]])
    assert metrics["macro_auc"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["confusion_matrix"].tolist() == [[2, 0], [0, 2]]


def test_probability_sum_is_checked_and_weights_are_two_class_train_only():
    with pytest.raises(ValueError): compute_binary_metrics([0, 1], [[.8, .8], [.1, .9]])
    assert np.allclose(compute_class_weights([0, 0, 1, 1, 1], 2).numpy(), [1.25, 5 / 6])


def test_build_transforms_supports_rectangular_e0b_input():
    image = Image.new("RGB", (256, 320))
    square = build_transforms("val", 224)(image)
    rectangular = build_transforms("val", [320, 256])(image)
    assert tuple(square.shape) == (3, 224, 224)
    assert tuple(rectangular.shape) == (3, 320, 256)
