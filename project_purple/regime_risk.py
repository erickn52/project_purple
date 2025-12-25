# project_purple/regime_risk.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


# Canonical regime labels (single source of truth)
REGIME_BULL = "BULL"
REGIME_CHOPPY = "CHOPPY"
REGIME_BEAR = "BEAR"
REGIME_UNKNOWN = "UNKNOWN"

ALL_REGIMES = (REGIME_BULL, REGIME_CHOPPY, REGIME_BEAR, REGIME_UNKNOWN)


# Canonical risk multipliers (single source of truth)
RISK_MULTIPLIER_BY_REGIME: Dict[str, float] = {
    REGIME_BULL: 1.0,
    REGIME_CHOPPY: 0.5,
    REGIME_BEAR: 0.0,
    REGIME_UNKNOWN: 1.0,
}


# Canonical trade permission (single source of truth)
TRADE_LONG_ALLOWED_BY_REGIME: Dict[str, bool] = {
    REGIME_BULL: True,
    REGIME_CHOPPY: True,
    REGIME_BEAR: False,
    REGIME_UNKNOWN: True,
}


def normalize_regime(regime: str) -> str:
    r = str(regime).strip().upper()
    if r in RISK_MULTIPLIER_BY_REGIME:
        return r
    return REGIME_UNKNOWN


@dataclass(frozen=True)
class RegimePolicy:
    regime: str
    trade_long: bool
    risk_multiplier: float


def get_regime_policy(regime: str) -> RegimePolicy:
    r = normalize_regime(regime)
    return RegimePolicy(
        regime=r,
        trade_long=bool(TRADE_LONG_ALLOWED_BY_REGIME.get(r, True)),
        risk_multiplier=float(RISK_MULTIPLIER_BY_REGIME.get(r, 1.0)),
    )


def get_risk_multiplier(regime: str) -> float:
    return get_regime_policy(regime).risk_multiplier


def is_trade_long_allowed(regime: str) -> bool:
    return get_regime_policy(regime).trade_long
