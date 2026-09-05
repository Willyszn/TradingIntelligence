from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
from pathlib import Path
import os

from trading_intelligence.data.acquisition.binance_public import download_daily_klines
from trading_intelligence.data.acquisition.gdelt_events import download_event_days, gdelt_archive_manifest

DEFAULT_CRYPTO = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT"]


def main() -> int:
    p = argparse.ArgumentParser(description="Bootstrap official/public Binance daily crypto and GDELT event data.")
    p.add_argument("--crypto", nargs="+", default=DEFAULT_CRYPTO)
    p.add_argument("--crypto-start", default="2017-01-01")
    p.add_argument("--crypto-end", default=None)
    p.add_argument("--gdelt-start", default=None, help="Inclusive YYYY-MM-DD; omit to skip GDELT downloads")
    p.add_argument("--gdelt-end", default=None)
    default_out = Path(os.environ.get("TI_DATA_ROOT", str(Path.home() / "TradingIntelligenceData")))
    p.add_argument("--out", type=Path, default=default_out, help="Persistent data root. Can also be set with TI_DATA_ROOT.")
    args = p.parse_args()
    args.out = args.out.expanduser().resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Persistent data root: {args.out}")

    crypto_out = args.out / "market"
    for symbol in args.crypto:
        destination = crypto_out / f"binance_{symbol.lower()}_daily.csv"
        print(f"Binance {symbol} -> {destination}")
        download_daily_klines(
            symbol,
            destination,
            start=datetime.fromisoformat(args.crypto_start).replace(tzinfo=timezone.utc),
            end=(datetime.fromisoformat(args.crypto_end).replace(tzinfo=timezone.utc) if args.crypto_end else None),
        )

    if args.gdelt_start:
        if not args.gdelt_end:
            raise SystemExit("--gdelt-end is required when --gdelt-start is supplied")
        start = date.fromisoformat(args.gdelt_start)
        end = date.fromisoformat(args.gdelt_end)
        gdelt_dir = args.out / "news" / "gdelt" / "events"
        paths = download_event_days(start, end, gdelt_dir)
        manifest = gdelt_archive_manifest(paths)
        manifest_path = gdelt_dir / "manifest.csv"
        manifest.to_csv(manifest_path, index=False)
        print(f"GDELT archives: {len(paths)}; manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
