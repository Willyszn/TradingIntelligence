from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


@dataclass(frozen=True)
class UniverseProfile:
    name: str
    description: str
    exclude_patterns: tuple[str, ...] = ()
    exclude_symbols: frozenset[str] = frozenset()


# Conservative: this list is intentionally limited to instruments whose primary
# purpose is leveraged/inverse exposure. It is not used to claim a complete ETF
# taxonomy; the SEC/company metadata layer will provide the authoritative type.
LEVERAGED_INVERSE_SYMBOLS = frozenset({
    "TQQQ", "SQQQ", "SOXL", "SOXS", "UPRO", "SPXU", "SSO", "SDS",
    "QLD", "QID", "FAS", "FAZ", "LABU", "LABD", "TECL", "TECS",
    "TMF", "TMV", "UVXY", "SVXY", "BITX", "BITU", "ETHU", "ETHD",
    "NUGT", "DUST", "JNUG", "JDST", "BOIL", "KOLD", "UCO", "SCO",
    "ERX", "ERY", "DRIP", "GUSH", "YINN", "YANG", "FNGU", "FNGD",
})

PROFILES = {
    "broad": UniverseProfile(
        "broad",
        "Liquidity-screened research candidates; instrument type is not inferred beyond symbol heuristics.",
    ),
    "core_equity_etf": UniverseProfile(
        "core_equity_etf",
        "Core research pool excluding obvious non-common patterns and a conservative leveraged/inverse list.",
        exclude_symbols=LEVERAGED_INVERSE_SYMBOLS,
        exclude_patterns=(r"(?:-WS|-WT|-WW|-UN|-U$|-R$|-P$|_)",),
    ),
}


def classify_profile(symbol: str, profile: str = "broad") -> tuple[bool, str | None]:
    key = symbol.upper().split(".", 1)[0]
    cfg = PROFILES.get(profile)
    if cfg is None:
        raise ValueError(f"Unknown universe profile: {profile}")
    if key in cfg.exclude_symbols:
        return False, "profile_excluded_leveraged_inverse"
    for pattern in cfg.exclude_patterns:
        if re.search(pattern, key, flags=re.IGNORECASE):
            return False, "profile_excluded_non_common_pattern"
    return True, None


def known_leveraged_inverse_symbols() -> set[str]:
    return set(LEVERAGED_INVERSE_SYMBOLS)
