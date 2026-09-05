from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from trading_intelligence.data.acquisition.gdelt_events import download_event_days, gdelt_archive_manifest
from trading_intelligence.data.paths import data_root


def main() -> None:
    p = argparse.ArgumentParser(description="Download GDELT 2.0 daily event archives.")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    p.add_argument("--output", help="Override raw output directory")
    args = p.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    root = data_root()
    out_dir = Path(args.output) if args.output else root / "news" / "gdelt_events" / "raw"
    paths = download_event_days(start, end, out_dir)
    manifest = gdelt_archive_manifest(paths)
    manifest_path = out_dir.parent / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    print(json.dumps({"downloaded": len(paths), "start": args.start, "end": args.end, "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
