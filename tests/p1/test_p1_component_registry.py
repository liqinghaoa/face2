from utils.p1_component_registry import (
    EXPERIMENT_ORDER,
    get_component_spec,
    runnable_component_keys,
    validate_component_registry,
)


def test_registry_contains_all_requested_components():
    result = validate_component_registry()
    assert result["experiment_order"] == list(EXPERIMENT_ORDER)
    assert result["specular_status"] == "SKIPPED_UNAVAILABLE_BY_CURRENT_FRONTEND"
    assert runnable_component_keys() == ("p1_a", "p1_n", "p1_l", "p1_s", "p1_r", "p1_rgb_a")


def test_registry_specular_is_explicitly_unavailable():
    spec = get_component_spec("P1-Spec")
    assert spec.enabled is False
    assert spec.availability == "unavailable_by_current_frontend"
    assert spec.stage_two_status == "SKIPPED_UNAVAILABLE_BY_CURRENT_FRONTEND"

