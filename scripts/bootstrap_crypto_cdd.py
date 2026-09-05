from __future__ import annotations

import warnings

warnings.warn(
    "CryptoDataDownload bootstrap is deprecated for this project; use scripts\bootstrap_public_data.py for Binance public data.",
    DeprecationWarning,
    stacklevel=2,
)

from trading_intelligence.data.acquisition.crypto_data_download import download_daily_binance

__all__ = ["download_daily_binance"]
