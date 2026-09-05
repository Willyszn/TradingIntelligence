from __future__ import annotations

from pathlib import Path
import pandas as pd

OBS_REQUIRED = ["cik", "metric", "unit", "start", "end", "information_time", "val_num"]
DECISION_REQUIRED = ["cik", "decision_time"]


def _prepare_observations(observations: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in OBS_REQUIRED if c not in observations.columns]
    if missing:
        raise ValueError(f"Missing observation columns: {missing}")
    out = observations.copy()
    out["cik"] = pd.to_numeric(out["cik"], errors="coerce").astype("Int64")
    out["information_time"] = pd.to_datetime(out["information_time"], utc=True, errors="coerce")
    out["start"] = pd.to_datetime(out["start"], utc=True, errors="coerce")
    out["end"] = pd.to_datetime(out["end"], utc=True, errors="coerce")
    out["val_num"] = pd.to_numeric(out["val_num"], errors="coerce")
    out = out.dropna(subset=["cik", "metric", "unit", "information_time", "end", "val_num"]).copy()
    # Canonical observations may contain multiple economic periods. Keep each period,
    # because point-in-time selection must not silently substitute a later period.
    return out


def _prepare_decisions(decisions: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in DECISION_REQUIRED if c not in decisions.columns]
    if missing:
        raise ValueError(f"Missing decision columns: {missing}")
    out = decisions.copy()
    out["cik"] = pd.to_numeric(out["cik"], errors="coerce").astype("Int64")
    out["decision_time"] = pd.to_datetime(out["decision_time"], utc=True, errors="coerce")
    out = out.dropna(subset=["cik", "decision_time"]).copy()
    return out


def asof_latest_observations(observations: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    """Return the latest disclosed observation available by each decision time.

    Causal rule: information_time <= decision_time. Revisions remain usable because the
    lookup selects the latest disclosed version known at that exact decision timestamp.
    Economic periods are kept separate, so this layer does not yet impose TTM/fiscal-year
    feature semantics.
    """
    obs = _prepare_observations(observations)
    dec = _prepare_decisions(decisions)
    if obs.empty or dec.empty:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    group_cols = ["cik", "metric", "unit", "period_type"] if "period_type" in obs.columns else ["cik", "metric", "unit"]
    for key, ogrp in obs.groupby(group_cols, dropna=False, sort=False):
        d_cik = key[0] if isinstance(key, tuple) else key
        dgrp = dec[dec["cik"] == d_cik].copy()
        if dgrp.empty:
            continue
        # merge_asof handles each metric/period group independently and enforces the
        # causal cutoff. A secondary deterministic sort on accession is applied after merge.
        ogrp = ogrp.sort_values("information_time", kind="mergesort")
        dgrp = dgrp.sort_values("decision_time", kind="mergesort")
        # Avoid duplicate join-key columns by removing `cik` from the observation side.
        ojoin = ogrp.drop(columns=["cik"], errors="ignore")
        out = pd.merge_asof(
            dgrp,
            ojoin,
            left_on="decision_time",
            right_on="information_time",
            direction="backward",
            allow_exact_matches=True,
        )
        out["available"] = out["information_time"].notna() & (out["information_time"] <= out["decision_time"])
        for c in ["metric", "unit", "period_type"]:
            if c in ogrp.columns and c not in out.columns:
                out[c] = ogrp[c].iloc[0]
        frames.append(out)

    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True, sort=False)
    preferred = [
        "cik", "decision_time", "metric", "unit", "period_type", "start", "end",
        "val_num", "information_time", "filed", "form", "accn", "tag", "entity_name", "available"
    ]
    cols = [c for c in preferred if c in result.columns] + [c for c in result.columns if c not in preferred]
    return result[cols].sort_values(["cik", "decision_time", "metric"], kind="mergesort").reset_index(drop=True)


def build_from_csv(observations_csv: str | Path, decisions_csv: str | Path, output_csv: str | Path) -> dict:
    obs = pd.read_csv(observations_csv, low_memory=False)
    decisions = pd.read_csv(decisions_csv, low_memory=False)
    out = asof_latest_observations(obs, decisions)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return {
        "observation_rows": int(len(obs)),
        "decision_rows": int(len(decisions)),
        "output_rows": int(len(out)),
        "available_rows": int(out["available"].sum()) if not out.empty else 0,
        "output": str(output_csv),
    }
