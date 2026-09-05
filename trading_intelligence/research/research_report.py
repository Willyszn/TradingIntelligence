from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import pandas as pd

@dataclass(frozen=True)
class ResearchRecord:
    name: str
    status: str
    metrics: dict
    warnings: list[str]

def write_records(records: list[ResearchRecord], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([asdict(r) for r in records], indent=2, default=str), encoding="utf-8")

def write_table(records: list[ResearchRecord], path: str | Path) -> None:
    rows = []
    for r in records:
        row = {"name": r.name, "status": r.status, **r.metrics, "warnings": "; ".join(r.warnings)}
        rows.append(row)
    df = pd.DataFrame(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
