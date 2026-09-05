import pandas as pd

from trading_intelligence.research.assembly import assemble_research_dataset, discover_market_files


def _write_market(path, symbol, start="2020-01-01", n=30):
    ts = pd.date_range(start, periods=n, freq="D", tz="UTC")
    df = pd.DataFrame({
        "timestamp": ts,
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 1000,
        "symbol": symbol,
    })
    df.to_csv(path, index=False)


def test_assembly_discovers_multiple_files(tmp_path):
    d = tmp_path / "market"
    d.mkdir()
    _write_market(d / "a.csv", "AAA")
    _write_market(d / "b.csv", "BBB")
    assert len(discover_market_files(d)) == 2
    dataset, result = assemble_research_dataset(d, horizon=3)
    assert result.symbols == 2
    assert result.market_files == 2
    assert len(dataset) == 60
    assert "forward_return" in dataset.columns


def test_assembly_can_join_point_in_time_news(tmp_path):
    d = tmp_path / "market"
    d.mkdir()
    _write_market(d / "a.csv", "AAA", n=30)
    news = pd.DataFrame({
        "timestamp": ["2020-01-10T12:00:00Z", "2020-01-20T12:00:00Z"],
        "asset": ["AAA", "AAA"],
        "sentiment": [1.0, -0.5],
        "sentiment_change": [0.4, -0.2],
        "surprise": [0.2, -0.1],
    })
    news_path = tmp_path / "news.csv"
    news.to_csv(news_path, index=False)
    dataset, result = assemble_research_dataset(d, news_csv=news_path, horizon=3)
    assert result.news_rows == 2
    assert "news_count_24h" in dataset.columns
    # News from the future of the first few rows must not be visible.
    assert dataset.loc[0, "news_count_24h"] == 0
