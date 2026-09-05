import pandas as pd

from trading_intelligence.research.universe import build_liquid_candidate_universe


def _write(path, symbol, rows=800, price=100.0, volume=500_000):
    idx = pd.date_range("2020-01-01", periods=rows, freq="D", tz="UTC")
    df = pd.DataFrame({
        "timestamp": idx,
        "open": price,
        "high": price + 1,
        "low": price - 1,
        "close": price,
        "volume": volume,
        "symbol": symbol,
    })
    df.to_csv(path, index=False)


def test_liquidity_screen_selects_strong_candidate(tmp_path):
    _write(tmp_path / "aaa.csv", "AAA", price=100, volume=500_000)  # $50m median dollar volume
    _write(tmp_path / "bbb.csv", "BBB", price=2, volume=500_000)
    result = build_liquid_candidate_universe(tmp_path, max_symbols=10)
    row = result[result.symbol == "AAA"].iloc[0]
    assert bool(row.eligible)
    assert bool(row.selected)


def test_liquidity_screen_rejects_short_history_and_warrant_like_symbols(tmp_path):
    _write(tmp_path / "short.csv", "SHORT", rows=100)
    _write(tmp_path / "warrant.csv", "ABC-WS", rows=800, price=100, volume=500_000)
    result = build_liquid_candidate_universe(tmp_path, max_symbols=10)
    short = result[result.symbol == "SHORT"].iloc[0]
    warrant = result[result.symbol == "ABC-WS"].iloc[0]
    assert short.exclusion_reason == "insufficient_history"
    assert warrant.exclusion_reason == "obvious_non_common_symbol_pattern"


def test_core_profile_excludes_known_leveraged_inverse_symbols(tmp_path):
    _write(tmp_path / "tqqq.csv", "TQQQ", rows=800, price=50, volume=500_000)
    _write(tmp_path / "aaa.csv", "AAA", rows=800, price=100, volume=500_000)
    result = build_liquid_candidate_universe(tmp_path, max_symbols=10, profile="core_equity_etf")
    tqqq = result[result.symbol == "TQQQ"].iloc[0]
    aaa = result[result.symbol == "AAA"].iloc[0]
    assert not bool(tqqq.eligible)
    assert tqqq.exclusion_reason == "profile_excluded_leveraged_inverse"
    assert bool(aaa.eligible)
