from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


SECURITY_TYPES = {
    "common_equity",
    "etf",
    "leveraged_etf",
    "inverse_etf",
    "adr",
    "preferred",
    "fund",
    "warrant",
    "other",
}


@dataclass(frozen=True)
class SecurityRecord:
    symbol: str
    cik: str | None = None
    security_type: str = "other"
    exchange: str | None = None
    currency: str | None = None
    active_from: pd.Timestamp | None = None
    active_to: pd.Timestamp | None = None
    is_common_equity: bool = False
    is_etf: bool = False
    is_leveraged: bool = False
    is_inverse: bool = False
    is_adr: bool = False

    def __post_init__(self) -> None:
        if self.security_type not in SECURITY_TYPES:
            raise ValueError(f"Unsupported security_type: {self.security_type}")


def build_security_master(
    records: list[SecurityRecord] | tuple[SecurityRecord, ...],
) -> pd.DataFrame:
    """Build a deterministic canonical security-master table."""
    rows = []
    for record in records:
        rows.append(
            {
                "symbol": record.symbol.upper().strip(),
                "cik": record.cik,
                "security_type": record.security_type,
                "exchange": record.exchange,
                "currency": record.currency,
                "active_from": record.active_from,
                "active_to": record.active_to,
                "is_common_equity": bool(record.is_common_equity),
                "is_etf": bool(record.is_etf),
                "is_leveraged": bool(record.is_leveraged),
                "is_inverse": bool(record.is_inverse),
                "is_adr": bool(record.is_adr),
            }
        )

    columns = [
        "symbol",
        "cik",
        "security_type",
        "exchange",
        "currency",
        "active_from",
        "active_to",
        "is_common_equity",
        "is_etf",
        "is_leveraged",
        "is_inverse",
        "is_adr",
    ]

    out = pd.DataFrame(rows, columns=columns)
    if out.empty:
        return out

    out["symbol"] = out["symbol"].astype(str)
    out["cik"] = out["cik"].astype("string")

    for column in ("active_from", "active_to"):
        out[column] = pd.to_datetime(out[column], utc=True, errors="coerce")

    out = (
        out.sort_values(
            ["symbol", "cik", "security_type"],
            kind="mergesort",
            na_position="last",
        )
        .drop_duplicates(["symbol"], keep="first")
        .reset_index(drop=True)
    )

    return out


def validate_security_master(frame: pd.DataFrame) -> dict[str, object]:
    required = {
        "symbol",
        "cik",
        "security_type",
        "exchange",
        "currency",
        "active_from",
        "active_to",
        "is_common_equity",
        "is_etf",
        "is_leveraged",
        "is_inverse",
        "is_adr",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Security master missing columns: {missing}")

    df = frame.copy()
    violations: list[str] = []

    if df["symbol"].duplicated().any():
        violations.append("duplicate symbols")

    if (~df["security_type"].isin(SECURITY_TYPES)).any():
        violations.append("unsupported security types")

    # A security cannot be both common equity and an ETF, and leveraged/inverse
    # instruments must never be admitted as plain common equity.
    bad_common = df["is_common_equity"] & (
        df["is_etf"] | df["is_leveraged"] | df["is_inverse"]
    )
    if bad_common.any():
        violations.append("invalid common-equity classification")

    bad_leveraged_flags = (
        df["is_leveraged"]
        & ~df["security_type"].isin({"leveraged_etf", "etf"})
    )
    if bad_leveraged_flags.any():
        violations.append("leveraged flag conflicts with security type")

    bad_inverse_flags = (
        df["is_inverse"]
        & ~df["security_type"].isin({"inverse_etf", "etf"})
    )
    if bad_inverse_flags.any():
        violations.append("inverse flag conflicts with security type")

    return {
        "rows": int(len(df)),
        "symbols": int(df["symbol"].nunique()),
        "violations": tuple(violations),
        "status": "PASS" if not violations else "FAIL",
    }


def load_security_master(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.copy()

    for column in ("active_from", "active_to"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(
                frame[column], utc=True, errors="coerce"
            )

    boolean_columns = [
        "is_common_equity",
        "is_etf",
        "is_leveraged",
        "is_inverse",
        "is_adr",
    ]
    for column in boolean_columns:
        if column in frame.columns:
            frame[column] = frame[column].astype(bool)

    result = validate_security_master(frame)
    if result["status"] != "PASS":
        raise ValueError(f"Invalid security master: {result['violations']}")

    return frame.sort_values(["symbol"], kind="mergesort").reset_index(drop=True)
