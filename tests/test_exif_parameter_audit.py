"""Small rule tests for the EXIF parameter audit."""

from scripts.analysis.audit_exif_parameter_completeness import (
    parse_number,
    validate_parameter,
)


def test_numeric_and_fraction_parsing() -> None:
    assert parse_number("1/50") == 0.02
    assert parse_number("' -25") == -25.0
    assert parse_number("0/0") is None


def test_explicit_invalid_and_warning_rules() -> None:
    assert validate_parameter("Orientation", "0")[0] == "invalid"
    assert validate_parameter("ExposureBiasValue", "-25")[0] == "invalid"
    assert validate_parameter("DigitalZoomRatio", "100")[0] == "warning"
    assert validate_parameter("LightSource", "255")[0] == "warning"


def test_valid_core_values() -> None:
    assert validate_parameter("ExposureTime", "0.02")[0] == "valid"
    assert validate_parameter("FNumber", "1.9")[0] == "valid"
    assert validate_parameter("ISOSpeedRatings", "320")[0] == "valid"
    assert validate_parameter("DateTimeOriginal", "2025:01:02 03:04:05")[0] == "valid"
