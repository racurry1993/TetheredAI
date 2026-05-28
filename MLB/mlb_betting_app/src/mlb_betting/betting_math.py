from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np


def american_to_decimal(price: float) -> float:
    price = float(price)
    if price > 0:
        return 1.0 + price / 100.0
    if price < 0:
        return 1.0 + 100.0 / abs(price)
    raise ValueError("American odds price cannot be zero")


def american_to_implied_prob(price: float) -> float:
    price = float(price)
    if price > 0:
        return 100.0 / (price + 100.0)
    if price < 0:
        return abs(price) / (abs(price) + 100.0)
    raise ValueError("American odds price cannot be zero")


def implied_prob_to_american(prob: float) -> float:
    prob = float(prob)
    if not 0 < prob < 1:
        raise ValueError("Probability must be between 0 and 1")
    if prob >= 0.5:
        return -100.0 * prob / (1.0 - prob)
    return 100.0 * (1.0 - prob) / prob


def no_vig_two_way(prob_a: float, prob_b: float) -> tuple[float, float]:
    total = float(prob_a) + float(prob_b)
    if total <= 0:
        return math.nan, math.nan
    return float(prob_a) / total, float(prob_b) / total


def profit_if_win(price: float, stake: float = 1.0) -> float:
    decimal = american_to_decimal(price)
    return stake * (decimal - 1.0)


def expected_value_per_unit(model_prob: float, american_price: float) -> float:
    """Expected net profit per 1 unit staked."""
    p = float(model_prob)
    win_profit = profit_if_win(american_price, stake=1.0)
    return p * win_profit - (1.0 - p) * 1.0


def kelly_fraction(model_prob: float, american_price: float, fraction: float = 0.25) -> float:
    """Fractional Kelly stake. Returns 0 for negative edge."""
    p = float(model_prob)
    b = american_to_decimal(american_price) - 1.0
    q = 1.0 - p
    full = (b * p - q) / b
    return max(0.0, full * fraction)


def safe_mean(values: Iterable[float]) -> Optional[float]:
    arr = np.array(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return None
    return float(np.mean(arr))
