"""Power, minimum detectable effect (MDE) and duration for the proposed holdout tests (notebook 06).

Everything uses the normal approximation to a two-sided test at level alpha. A share `holdout` of the
randomized units is held out (not treated); the rest are treated.

Per-unit metrics (a 30-day purchase or return visit, revenue per visitor), n units in total:

    SE     = sd * sqrt(1 / ((1 - holdout) * n) + 1 / (holdout * n))     SE of treated mean - holdout mean
    power  = Phi(lift * mean / SE - z_a) + Phi(-lift * mean / SE - z_a)  lift relative to the holdout mean
    MDE    = (z_a + z_b) * SE / mean                                     relative lift detected with power 1 - b

with z_a = z(1 - alpha/2), and the holdout's SD used for both arms. A 0/1 metric with rate p has
sd = sqrt(p (1 - p)): the two-proportion z-test.

Counts (Poisson), e.g. purchases per week of a whole randomized audience. Over `periods`, the treated arm
counts X_T ~ Poisson((1 - holdout) r_T periods) and the holdout X_H ~ Poisson(holdout r_H periods), where r_T
and r_H are full-audience rates per period. X_T / (1 - holdout) - X_H / holdout estimates the full-scale
difference (r_T - r_H) periods with variance

    Var    = vif * periods * (r_T / (1 - holdout) + r_H / holdout)

and power follows as above. vif = 1 is pure Poisson; vif > 1 allows for units that repeat (overdispersion).

Durations: power rises with the number of units (or periods), so duration_for_power() solves power = target.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def z_star(alpha: float = 0.05, power: float = 0.80) -> float:
    """z(1 - alpha/2) + z(power): the effect, in standard errors, detected with that power (2.80 by default)."""
    return float(norm.ppf(1 - alpha / 2) + norm.ppf(power))


def _two_sided_power(z_effect, alpha: float):
    z_a = norm.ppf(1 - alpha / 2)
    return norm.cdf(z_effect - z_a) + norm.cdf(-z_effect - z_a)


# ---------------------------------------------------------------- per-unit metrics

def diff_se(sd, n, holdout: float = 0.5):
    """SE of (treated mean - holdout mean) with n units in total and per-unit SD sd."""
    return sd * np.sqrt(1 / ((1 - holdout) * n) + 1 / (holdout * n))


def two_sample_power(mean, sd, lift, n, holdout: float = 0.5, alpha: float = 0.05):
    """Power to detect a relative lift in a per-unit mean (z-test)."""
    return _two_sided_power(lift * mean / diff_se(sd, n, holdout), alpha)


def two_sample_mde(mean, sd, n, holdout: float = 0.5, alpha: float = 0.05, power: float = 0.80):
    """Smallest relative lift in a per-unit mean detected with the given power."""
    return z_star(alpha, power) * diff_se(sd, n, holdout) / mean


def proportion_sd(p):
    return np.sqrt(p * (1 - p))


def two_proportion_power(p, lift, n, holdout: float = 0.5, alpha: float = 0.05):
    """Power of the two-proportion z-test to detect a relative lift in a rate p (the holdout's rate)."""
    return two_sample_power(p, proportion_sd(p), lift, n, holdout, alpha)


def two_proportion_mde(p, n, holdout: float = 0.5, alpha: float = 0.05, power: float = 0.80):
    """Smallest relative lift in a rate p detected with the given power."""
    return two_sample_mde(p, proportion_sd(p), n, holdout, alpha, power)


# ---------------------------------------------------------------- counts

def poisson_diff_se(rate_treated, rate_holdout, periods, holdout: float = 0.5, vif: float = 1.0):
    """SE of the full-scale difference in counts over `periods` (rates are per period, whole audience)."""
    return np.sqrt(vif * periods * (rate_treated / (1 - holdout) + rate_holdout / holdout))


def poisson_power(rate_treated, rate_holdout, periods, holdout: float = 0.5, vif: float = 1.0,
                  alpha: float = 0.05):
    """Power to detect the difference between two Poisson rates (full-audience counts per period)."""
    effect = (rate_treated - rate_holdout) * periods
    return _two_sided_power(effect / poisson_diff_se(rate_treated, rate_holdout, periods, holdout, vif), alpha)


def poisson_mde(rate, periods, holdout: float = 0.5, vif: float = 1.0, alpha: float = 0.05,
                power: float = 0.80):
    """Smallest difference in a full-audience rate per period detected with the given power (small effects)."""
    return z_star(alpha, power) * poisson_diff_se(rate, rate, periods, holdout, vif) / periods


# ---------------------------------------------------------------- duration

def duration_for_power(power_at: Callable[[float], float], target: float = 0.80,
                       max_duration: float = 1e7) -> float:
    """Shortest duration d with power_at(d) = target, for a power that rises with d (inf if beyond max_duration)."""
    if power_at(max_duration) < target:
        return float("inf")
    return float(brentq(lambda d: power_at(d) - target, 1e-9, max_duration, xtol=1e-12, rtol=1e-12))
