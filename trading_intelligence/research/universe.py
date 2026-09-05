from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re

import pandas as pd

from trading_intelligence.research.universe_profiles import classify_profile


@dataclass(frozen=True)
class UniverseCandidate:
    symbol: str
    path: str
    rows: int
    first_timestamp: str
    last_timestamp: str
    last_close: float
    median_close_lookback: float
    median_dollar_volume_lookback: float
    avg_dollar_volume_lookback: float
    eligible: bool
    exclusion_reason: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_OBVIOUS_NON_COMMON = re.compile(r"(?:-WS|-WT|-WW|-WSA|-WTA|-UN|-U$|-R$|-P$|_)", re.IGNORECASE)


def _read_market_file(path: Path, lookback: int) -> tuple[UniverseCandidate | None, str | None]:
    try:
        df = pd.read_csv(path, usecols=["timestamp", "open", "high", "low", "close", "volume", "symbol"])
    except Exception as exc:
        return None, f"unreadable: {exc}"
    if df.empty:
        return None, "empty"
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    numeric = df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
    work = pd.DataFrame({
        "timestamp": ts,
        "open": numeric["open"],
        "high": numeric["high"],
        "low": numeric["low"],
        "close": numeric["close"],
        "volume": numeric["volume"],
        "symbol": df["symbol"].astype(str),
    }).dropna(subset=["timestamp", "close", "volume"])
    if work.empty:
        return None, "no_valid_rows"
    work = work.sort_values("timestamp")
    symbol = str(work["symbol"].iloc[-1]).upper()
    tail = work.tail(max(1, lookback)).copy()
    dollar_volume = tail["close"].abs() * tail["volume"].abs()
    candidate = UniverseCandidate(
        symbol=symbol,
        path=str(path),
        rows=int(len(work)),
        first_timestamp=work["timestamp"].min().isoformat(),
        last_timestamp=work["timestamp"].max().isoformat(),
        last_close=float(work["close"].iloc[-1]),
        median_close_lookback=float(tail["close"].median()),
        median_dollar_volume_lookback=float(dollar_volume.median()),
        avg_dollar_volume_lookback=float(dollar_volume.mean()),
        eligible=False,
        exclusion_reason=None,
    )
    return candidate, None


def build_liquid_candidate_universe(
    root: str | Path,
    min_history_rows: int = 750,
    min_median_close: float = 5.0,
    min_median_dollar_volume: float = 20_000_000.0,
    lookback: int = 90,
    exclude_obvious_non_common: bool = True,
    max_symbols: int | None = 500,
    profile: str = "broad",
) -> pd.DataFrame:
    root = Path(root)
    records: list[dict[str, object]] = []
    for path in sorted(root.glob("*.csv")):
        cand, err = _read_market_file(path, lookback)
        if cand is None:
            continue
        reason: str | None = None
        if cand.rows < min_history_rows:
            reason = "insufficient_history"
        elif cand.last_close < min_median_close or cand.median_close_lookback < min_median_close:
            reason = "low_price"
        elif cand.median_dollar_volume_lookback < min_median_dollar_volume:
            reason = "low_liquidity"
        elif exclude_obvious_non_common and _OBVIOUS_NON_COMMON.search(cand.symbol):
            reason = "obvious_non_common_symbol_pattern"
        else:
            profile_ok, profile_reason = classify_profile(cand.symbol, profile=profile)
            if not profile_ok:
                reason = profile_reason
        eligible = reason is None
        row = cand.to_dict()
        row["eligible"] = eligible
        row["exclusion_reason"] = reason
        records.append(row)

    df = pd.DataFrame(records)
    if df.empty:
        return df
    eligible = df[df["eligible"]].sort_values(
        ["median_dollar_volume_lookback", "rows"], ascending=[False, False]
    )
    if max_symbols is not None:
        selected_symbols = set(eligible.head(max_symbols)["symbol"])
        df["selected"] = df["symbol"].isin(selected_symbols)
    else:
        df["selected"] = df["eligible"]
    return df.sort_values(["selected", "eligible", "median_dollar_volume_lookback"], ascending=[False, False, False])
