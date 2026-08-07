from dataclasses import replace
from pathlib import Path

import pytest

from p0b_deca.config import load_config
from p0b_deca.deca_frontend import DecaFrontend
from p0b_deca.environment_audit import audit_environment


def test_missing_deca_source_is_explicitly_blocked(tmp_path: Path):
    config = replace(load_config(Path("config/p0b/p0b_deca_environment_v1.yaml"), Path.cwd()), deca_root=tmp_path / "DECA")
    result = audit_environment(config)
    assert result["passed"] is False and result["blocked"] is True
    assert result["checks"]["deca_root_exists"] is False
    assert len(result["assets"]) >= 1 and all(not asset["exists"] for asset in result["assets"])
    with pytest.raises(RuntimeError, match="failed_environment"):
        DecaFrontend(config.deca_root, "cuda")
