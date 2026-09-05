from __future__ import annotations

import argparse
import pandas as pd

from trading_intelligence.research.fred_macro import audit_macro_vintages


def main() -> None:
    p = argparse.ArgumentParser(description="Audit a downloaded FRED vintage dataset.")
    p.add_argument("csv")
    args = p.parse_args()
    df = pd.read_csv(args.csv)
    print(audit_macro_vintages(df))


if __name__ == "__main__":
    main()
