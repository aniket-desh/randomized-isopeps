import tomllib
from pathlib import Path


def test_quimb_extras_require_device_safe_autoray():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    extras = project["project"]["optional-dependencies"]

    assert "autoray>=0.9.0" in extras["quimb"]
    assert "autoray>=0.9.0" in extras["all"]
