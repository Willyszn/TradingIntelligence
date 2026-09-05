from __future__ import annotations

import argparse
import json
from pathlib import Path

from trading_intelligence.data.audit import audit_market_csv
from trading_intelligence.data.paths import market_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit all normalized market CSV files in a directory.")
    parser.add_argument("path", nargs="?", type=Path, default=market_root())
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON report path.")
    args = parser.parse_args()

    directory = args.path.expanduser().resolve()
    files = sorted(directory.glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files found under {directory}")

    audits = [audit_market_csv(f).to_dict() for f in files]
    payload = json.dumps(audits, indent=2)
    print(payload)
    fails = sum(a["status"] == "FAIL" for a in audits)
    warnings = sum(a["status"] == "PASS_WITH_WARNINGS" for a in audits)
    print(f"\nAudited {len(audits)} files; failures: {fails}; warnings: {warnings}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
        print(f"Saved audit -> {args.out}")
    raise SystemExit(2 if fails else 0)


if __name__ == "__main__":
    main()
