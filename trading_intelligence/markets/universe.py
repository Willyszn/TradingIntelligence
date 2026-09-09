from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

MarketKind = Literal["equity", "fx", "crypto", "future", "etf", "option", "commodity", "index"]


@dataclass(frozen=True)
class MarketUniverse:
    name: str
    asset_class: MarketKind
    symbols: tuple[str, ...]
    primary_currency: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


DEFAULT_UNIVERSES = {
    "us_equities": MarketUniverse(
        "us_equities", "equity", tuple(), metadata={"scope": "S&P 500 + Nasdaq 100; constituent lists supplied by data provider"}
    ),
    "fx_major": MarketUniverse(
        "fx_major", "fx", ("EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD", "EURGBP", "EURJPY", "GBPJPY")
    ),
    "crypto_liquid": MarketUniverse(
        "crypto_liquid", "crypto", ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT")
    ),
    "major_futures": MarketUniverse(
        "major_futures", "future", ("ES", "NQ", "RTY", "CL", "GC", "ZN", "6E"), metadata={"note": "continuous-contract methodology must be defined by data provider"}
    ),
    "major_etfs": MarketUniverse(
        "major_etfs", "etf", ("SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "USO", "XLE", "XLK", "XLF")
    ),
}

