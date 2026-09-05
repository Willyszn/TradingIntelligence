from pathlib import Path

from trading_intelligence.data.paths import data_root, market_root, reports_root


def test_persistent_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("TI_DATA_ROOT", str(tmp_path / "TI"))
    assert data_root() == (tmp_path / "TI").resolve()
    assert market_root() == (tmp_path / "TI" / "market").resolve()
    assert reports_root() == (tmp_path / "TI" / "reports").resolve()
