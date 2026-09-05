from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

from trading_intelligence.env import load_project_env
from trading_intelligence.data.paths import data_root


def run_module(module: str, args: list[str]) -> None:
    cmd = [sys.executable, "-m", module, *args]
    result = subprocess.run(cmd, check=False)
    if result.returncode:
        raise SystemExit(result.returncode)


def main() -> None:
    p = argparse.ArgumentParser(description="One-command bootstrap for Trading Intelligence research data.")
    p.add_argument("--skip-fred", action="store_true")
    p.add_argument("--gdelt-start", help="Optional GDELT event start date YYYY-MM-DD")
    p.add_argument("--gdelt-end", help="Optional GDELT event end date YYYY-MM-DD")
    p.add_argument("--skip-gdelt", action="store_true")
    args = p.parse_args()

    env_path = load_project_env()
    root = data_root()
    root.mkdir(parents=True, exist_ok=True)

    print(json.dumps({"project_env": str(env_path) if env_path else None, "data_root": str(root)}, indent=2))

    if not args.skip_fred:
        from trading_intelligence.research.fred_macro import download_macro_vintages
        from trading_intelligence.research.fred_macro import DEFAULT_SERIES
        import os
        if not os.environ.get("FRED_API_KEY"):
            raise RuntimeError("FRED_API_KEY is missing. Copy .env.example to .env and set it locally.")
        output = root / "macro" / "fred_macro_vintages.csv"
        print("\n=== FRED / ALFRED ===")
        print(download_macro_vintages(output, series=DEFAULT_SERIES))

    if not args.skip_gdelt and args.gdelt_start and args.gdelt_end:
        from trading_intelligence.data.acquisition.gdelt_events import download_event_days, gdelt_archive_manifest
        start = date.fromisoformat(args.gdelt_start)
        end = date.fromisoformat(args.gdelt_end)
        if end < start:
            raise ValueError("--gdelt-end must be on or after --gdelt-start")
        out_dir = root / "news" / "gdelt_events" / "raw"
        print("\n=== GDELT EVENTS ===")
        paths = download_event_days(start, end, out_dir)
        manifest = gdelt_archive_manifest(paths)
        manifest_path = root / "news" / "gdelt_events" / "manifest.csv"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(manifest_path, index=False)
        print(json.dumps({"downloaded": len(paths), "manifest": str(manifest_path)}, indent=2))
    elif not args.skip_gdelt:
        print("\n=== GDELT EVENTS ===")
        print("Skipped download: provide --gdelt-start and --gdelt-end to avoid accidentally downloading a massive history.")

    print("\nBootstrap complete.")


if __name__ == "__main__":
    main()
