from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class ModelCard:
    name: str
    version: str
    market: str
    horizon_bars: int
    features: list[str]
    training_start: str
    training_end: str
    validation_method: str
    notes: str = ""

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
