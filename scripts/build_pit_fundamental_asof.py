from __future__ import annotations

import argparse
import json

from trading_intelligence.research.pit_fundamental_asof import build_from_csv


def main() -> None:
    p = argparse.ArgumentParser(description="Align canonical PIT SEC observations to explicit market decision timestamps.")
    p.add_argument("observations_csv")
    p.add_argument("decisions_csv", help="CSV containing at least cik and decision_time in UTC.")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    print(json.dumps(build_from_csv(args.observations_csv, args.decisions_csv, args.output), indent=2, default=str))


if __name__ == "__main__":
    main()
