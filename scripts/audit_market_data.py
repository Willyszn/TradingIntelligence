from __future__ import annotations

import argparse
import json

from trading_intelligence.data.audit import audit_market_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a market CSV before research use.")
    parser.add_argument("path")
    args = parser.parse_args()
    audit = audit_market_csv(args.path)
    print(json.dumps(audit.to_dict(), indent=2))
    if audit.warnings:
        print("\nWarnings:")
        for warning in audit.warnings:
            print(f"- {warning}")
    raise SystemExit(0 if audit.status != "FAIL" else 2)


if __name__ == "__main__":
    main()
