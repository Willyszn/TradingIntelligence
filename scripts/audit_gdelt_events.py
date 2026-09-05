from __future__ import annotations

import argparse
import json
import pandas as pd

from trading_intelligence.data.acquisition.gdelt_events import audit_gdelt_event_table


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    args = p.parse_args()
    df = pd.read_csv(args.csv, nrows=None)
    print(json.dumps(audit_gdelt_event_table(df), indent=2))


if __name__ == "__main__":
    main()
