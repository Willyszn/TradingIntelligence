import pandas as pd

from trading_intelligence.data.audit import audit_market_csv


def test_audit_passes_clean_csv(tmp_path):
    path = tmp_path / "clean.csv"
    pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True),
        "symbol": ["AAA", "AAA"],
        "open": [100, 101],
        "high": [102, 103],
        "low": [99, 100],
        "close": [101, 102],
        "volume": [1000, 1100],
    }).to_csv(path, index=False)
    audit = audit_market_csv(path)
    assert audit.status == "PASS_WITH_WARNINGS"
    assert audit.duplicate_keys == 0
    assert audit.invalid_ohlc_rows == 0
    assert audit.has_adjusted_close is False


def test_audit_flags_duplicates_and_bad_ohlc(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-01", "2026-01-01"], utc=True),
        "symbol": ["AAA", "AAA"],
        "open": [100, 100],
        "high": [99, 99],
        "low": [98, 98],
        "close": [100, 100],
        "volume": [1000, 1000],
    }).to_csv(path, index=False)
    audit = audit_market_csv(path)
    assert audit.status == "FAIL"
    assert audit.duplicate_keys == 1
    assert audit.invalid_ohlc_rows == 2


def test_crypto_audit_does_not_apply_equity_adjusted_close_warning(tmp_path):
    path = tmp_path / "binance_btcusdt_daily.csv"
    pd.DataFrame({
        "timestamp": pd.to_datetime(["2020-03-12", "2020-03-13"], utc=True),
        "symbol": ["BTCUSDT", "BTCUSDT"],
        "open": [7934.58, 4800],
        "high": [7966.17, 5600],
        "low": [4410, 4300],
        "close": [4800, 5600],
        "volume": [1000, 1200],
    }).to_csv(path, index=False)
    audit = audit_market_csv(path)
    assert audit.asset_class == "crypto"
    assert audit.has_adjusted_close is False
    assert not any("adjusted-close" in w for w in audit.warnings)
    assert audit.extreme_return_rows == 0
