from __future__ import annotations

import argparse
from pathlib import Path

from trading_intelligence.data.acquisition.stooq import download_daily

# Conservative liquid starter universe. This is deliberately a bootstrap list,
# not an assertion that these are the only securities worth researching.
DEFAULT_SYMBOLS = [
    "aapl.us", "msft.us", "nvda.us", "amzn.us", "meta.us", "googl.us", "tsla.us", "avgo.us",
    "brk-b.us", "jpm.us", "xom.us", "unh.us", "cost.us", "wmt.us", "jnj.us", "lly.us",
    "pg.us", "ko.us", "pep.us", "hd.us", "crm.us", "orcl.us", "adbe.us", "nflx.us", "amd.us",
    "intc.us", "qcom.us", "csco.us", "txn.us", "cat.us", "ba.us", "ge.us", "lin.us", "nee.us",
    "spy.us", "qqq.us", "iwm.us", "dia.us", "xlf.us", "xlk.us", "xle.us", "gld.us", "slv.us",
]


def main() -> int:
    p = argparse.ArgumentParser(description="Bootstrap a multi-asset US equity/ETF daily dataset from Stooq.")
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--out", type=Path, default=Path("data/raw/market"))
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    failures: list[tuple[str, str]] = []
    for symbol in args.symbols:
        destination = args.out / f"stooq_{symbol.replace('.', '_')}_daily.csv"
        print(f"Downloading {symbol} -> {destination}")
        try:
            download_daily(symbol, destination)
        except Exception as exc:  # noqa: BLE001 - report and continue across a large universe
            failures.append((symbol, str(exc)))
            print(f"FAILED {symbol}: {exc}")

    if failures:
        print("\nFailures:")
        for symbol, reason in failures:
            print(f"- {symbol}: {reason}")
    print(f"\nCompleted: {len(args.symbols) - len(failures)}/{len(args.symbols)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
