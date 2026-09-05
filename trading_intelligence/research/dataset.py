from __future__ import annotations

from pathlib import Path
import pandas as pd

REQUIRED_MARKET = ["timestamp", "open", "high", "low", "close", "volume", "symbol"]


def load_market_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_MARKET if c not in df.columns]
    if missing:
        raise ValueError(f"Missing market columns: {missing}")
    df = df[REQUIRED_MARKET].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "symbol"])
    return (
        df.sort_values(["symbol", "timestamp"])
        .drop_duplicates(["symbol", "timestamp"], keep="last")
        .reset_index(drop=True)
    )


def build_supervised_market_dataset(market: pd.DataFrame, horizon: int = 3) -> pd.DataFrame:
    """Build causal features and a canonical next-bar-entry forward-return target."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    df = market.copy().sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    g = df.groupby("symbol", group_keys=False)

    df["ret_1"] = g["close"].pct_change(1)
    df["ret_3"] = g["close"].pct_change(3)
    df["ret_5"] = g["close"].pct_change(5)
    df["vol_10"] = g["ret_1"].transform(lambda s: s.rolling(10, min_periods=5).std())
    ma_10 = g["close"].transform(lambda s: s.rolling(10, min_periods=5).mean())
    df["close_to_ma10"] = df["close"] / ma_10 - 1.0
    df["range_pct"] = (df["high"] - df["low"]) / df["close"].replace(0, pd.NA)
    vol_ma = g["volume"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    df["volume_ratio_20"] = df["volume"] / vol_ma.replace(0, pd.NA)

    df["entry_timestamp"] = g["timestamp"].shift(-1)
    df["entry_price"] = g["open"].shift(-1)
    df["target_timestamp"] = g["timestamp"].shift(-horizon)
    exit_close = g["close"].shift(-horizon)
    df["target_return"] = exit_close / df["entry_price"] - 1.0
    df["forward_return"] = df["target_return"]
    df["forward_positive"] = (df["target_return"] > 0).astype("Int64")
    return df


def join_point_in_time_news(
    market: pd.DataFrame,
    news: pd.DataFrame,
    lookback_hours: int = 24,
) -> pd.DataFrame:
    """Attach aggregated news signals using only news seen on/before market time."""
    out = market.copy().sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    defaults = {"news_count_24h": 0.0, "sentiment_mean_24h": 0.0, "sentiment_shock_24h": 0.0}
    if news is None or news.empty:
        for k, v in defaults.items():
            out[k] = v
        return out

    n = news.copy()
    if "asset" not in n.columns or "timestamp" not in n.columns:
        for k, v in defaults.items():
            out[k] = v
        return out
    n["timestamp"] = pd.to_datetime(n["timestamp"], utc=True, errors="coerce")
    n = n.dropna(subset=["timestamp", "asset"]).copy()
    n["asset"] = n["asset"].astype(str)
    for c in ["sentiment", "sentiment_change", "surprise"]:
        if c not in n.columns:
            n[c] = 0.0
        n[c] = pd.to_numeric(n[c], errors="coerce").fillna(0.0)

    rows: list[dict] = []
    window = pd.Timedelta(hours=lookback_hours)
    for r in out.itertuples(index=False):
        asset_news = n[
            (n["asset"] == str(r.symbol))
            & (n["timestamp"] <= r.timestamp)
            & (n["timestamp"] > r.timestamp - window)
        ]
        if asset_news.empty:
            rows.append(defaults.copy())
            continue
        rows.append(
            {
                "news_count_24h": float(len(asset_news)),
                "sentiment_mean_24h": float(asset_news["sentiment"].mean()),
                "sentiment_shock_24h": float(
                    asset_news["sentiment_change"].mean() + asset_news["surprise"].mean()
                ),
            }
        )
    return pd.concat([out, pd.DataFrame(rows, index=out.index)], axis=1)
