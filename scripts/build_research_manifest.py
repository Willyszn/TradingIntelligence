from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="data/raw/market")
    ap.add_argument("--output", default="data/manifests/market_manifest.json")
    args = ap.parse_args()
    root = Path(args.input_dir)
    rows = []
    for path in sorted(root.glob("*.csv")):
        try:
            df = pd.read_csv(path, usecols=["timestamp", "symbol"])
            rows.append({
                "file": str(path), "sha256": sha256(path), "rows": int(len(df)),
                "symbols": sorted(df["symbol"].dropna().astype(str).unique().tolist()),
                "min_timestamp": str(df["timestamp"].min()), "max_timestamp": str(df["timestamp"].max()),
            })
        except Exception as exc:
            rows.append({"file": str(path), "error": str(exc), "sha256": sha256(path)})
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"files": rows}, indent=2), encoding="utf-8")
    print(f"Wrote {out} for {len(rows)} files")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
