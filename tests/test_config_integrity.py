from pathlib import Path


def test_project_version_is_current():
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "4.7.0"' in text
