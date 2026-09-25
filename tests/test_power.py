"""Tests for src/power.py: power, minimum detectable effect and duration, against hand calculations and statsmodels."""

from __future__ import annotations

import numpy as np
import pytest
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.rates import power_poisson_diff_2indep

from src.power import (diff_se, duration_for_power, poisson_mde, poisson_power, two_proportion_mde,
                       two_proportion_power, two_sample_mde, two_sample_power, z_star)


def test_z_star_is_the_sum_of_the_two_normal_quantiles():
    assert z_star() == pytest.approx(1.959964 + 0.841621, abs=1e-6)          # 5% two-sided, 80% power
    assert z_star(alpha=0.01, power=0.90) == pytest.approx(2.575829 + 1.281552, abs=1e-6)


def test_diff_se_by_hand():
    assert diff_se(1.0, 400) == pytest.approx(np.sqrt(1 / 200 + 1 / 200))    # 200 per arm
    assert diff_se(1.0, 400, holdout=0.1) == pytest.approx(np.sqrt(1 / 360 + 1 / 40))


def test_two_proportion_mde_by_hand():
    # 2% conversion, 20,000 visitors split 50/50: SE = sqrt(0.02 * 0.98 * (1/10,000 + 1/10,000)) = 0.00198
    se = np.sqrt(0.02 * 0.98 * 2 / 10_000)
    assert two_proportion_mde(0.02, 20_000) == pytest.approx(z_star() * se / 0.02)
    assert two_proportion_mde(0.02, 20_000) == pytest.approx(0.2773, abs=1e-4)


def test_a_10_90_split_needs_2_8_times_the_visitors_of_50_50():
    # SE factor sqrt(1/0.9 + 1/0.1) = 10/3 against sqrt(1/0.5 + 1/0.5) = 2
    assert two_proportion_mde(0.02, 50_000, holdout=0.1) / two_proportion_mde(0.02, 50_000) == pytest.approx(10 / 6)
    n_5050 = duration_for_power(lambda n: two_proportion_power(0.02, 0.10, n))
    n_1090 = duration_for_power(lambda n: two_proportion_power(0.02, 0.10, n, holdout=0.1))
    assert n_1090 / n_5050 == pytest.approx((10 / 6) ** 2, rel=1e-4)


def test_mde_shrinks_with_the_square_root_of_the_sample():
    assert two_proportion_mde(0.03, 40_000) == pytest.approx(two_proportion_mde(0.03, 10_000) / 2)


@pytest.mark.parametrize("p, lift, n, holdout", [
    (0.0156, 0.10, 107_000, 0.5),    # about 12 months of the top-20% audience, staff excluded
    (0.0156, 0.10, 107_000, 0.1),
    (0.20, 0.10, 5_000, 0.3),
    (0.05, -0.15, 8_000, 0.5),       # a drop is detected as well as a rise
])
def test_two_proportion_power_matches_statsmodels(p, lift, n, holdout):
    effect = lift * p / np.sqrt(p * (1 - p))                  # standardized by the holdout's SD
    n_treated, n_holdout = (1 - holdout) * n, holdout * n
    expected = NormalIndPower().power(effect, nobs1=n_treated, alpha=0.05, ratio=n_holdout / n_treated)
    assert two_proportion_power(p, lift, n, holdout) == pytest.approx(expected, rel=1e-10)


def test_power_at_the_mde_is_the_target_power():
    for holdout in (0.1, 0.5):
        mde = two_proportion_mde(0.02, 60_000, holdout)
        assert two_proportion_power(0.02, mde, 60_000, holdout) == pytest.approx(0.80, abs=1e-5)   # + far tail
    mde = two_sample_mde(3.0, 45.0, 100_000, power=0.9)
    assert two_sample_power(3.0, 45.0, mde, 100_000) == pytest.approx(0.90, abs=1e-5)


def test_no_effect_has_power_alpha():
    assert two_proportion_power(0.02, 0.0, 10_000) == pytest.approx(0.05)
    assert poisson_power(117.0, 117.0, 12) == pytest.approx(0.05)
    assert poisson_power(117.0, 117.0, 12, alpha=0.10) == pytest.approx(0.10)


@pytest.mark.parametrize("rate_treated, rate_holdout, periods, holdout", [
    (117.0, 114.85, 12, 0.5),        # store-wide purchases per week, all Display purchases incremental
    (117.0, 114.85, 4, 0.1),
    (15_800.0, 15_700.0, 12, 0.5),   # sessions per week
    (50.0, 40.0, 6, 0.2),
])
def test_poisson_power_matches_statsmodels(rate_treated, rate_holdout, periods, holdout):
    # statsmodels compares rates per unit exposure: arm exposures are (1 - holdout) * periods and holdout * periods
    expected = power_poisson_diff_2indep(rate_treated, rate_holdout, nobs1=(1 - holdout) * periods,
                                         nobs_ratio=holdout / (1 - holdout), method_var="alt",
                                         return_results=False)
    assert poisson_power(rate_treated, rate_holdout, periods, holdout) == pytest.approx(expected, rel=1e-10)


def test_poisson_power_by_hand():
    # 50/50 for 12 weeks, S = 117 purchases/week, D = 2.15 of them lost in the holdout
    var = 117.0 * 12 / 0.5 + (117.0 - 2.15) * 12 / 0.5
    z = 2.15 * 12 / np.sqrt(var)
    assert poisson_power(117.0, 117.0 - 2.15, 12) == pytest.approx(
        _phi(z - 1.959964) + _phi(-z - 1.959964), abs=1e-6)


def _phi(x: float) -> float:
    from math import erf, sqrt
    return 0.5 * (1 + erf(x / sqrt(2)))


def test_overdispersion_acts_like_less_data():
    # variance x 4 has the same power as a quarter of the periods
    assert poisson_power(100.0, 97.0, 12, vif=4.0) == pytest.approx(poisson_power(100.0, 97.0, 3))


def test_poisson_mde_is_detected_with_80_percent_power():
    # rates S +/- mde/2 keep the 50/50 variance at the no-effect level the MDE assumes
    s, mde = 15_800.0, poisson_mde(15_800.0, 12)
    assert poisson_power(s + mde / 2, s - mde / 2, 12) == pytest.approx(0.80, abs=1e-5)
    assert poisson_power(s, s - mde, 12) == pytest.approx(0.80, abs=0.005)   # small effect: nearly the same
    assert poisson_mde(s, 12, vif=2.0) == pytest.approx(mde * np.sqrt(2))


def test_duration_for_power_matches_the_closed_form():
    # months = z*^2 p (1 - p) (1/(1 - h) + 1/h) / (lift p)^2 / visitors per month; the solver also counts the
    # negligible far tail, so it lands a hair below the closed form
    p, lift, per_month, h = 0.0156, 0.10, 8_940.0, 0.5
    closed = z_star() ** 2 * p * (1 - p) * (1 / (1 - h) + 1 / h) / (lift * p) ** 2 / per_month
    months = duration_for_power(lambda m: two_proportion_power(p, lift, per_month * m, h))
    assert months == pytest.approx(closed, rel=1e-4)
    assert two_proportion_power(p, lift, per_month * months, h) == pytest.approx(0.80, abs=1e-9)

    weeks = duration_for_power(lambda w: poisson_power(117.0, 114.85, w, vif=1.27))
    assert weeks == pytest.approx(z_star() ** 2 * 1.27 * (117.0 / 0.5 + 114.85 / 0.5) / 2.15 ** 2, rel=1e-4)


def test_duration_matches_the_statsmodels_sample_size():
    p, lift, h = 0.02, 0.10, 0.1
    effect = lift * p / np.sqrt(p * (1 - p))
    n_treated = NormalIndPower().solve_power(effect, nobs1=None, alpha=0.05, power=0.80, ratio=h / (1 - h))
    n_total = duration_for_power(lambda n: two_proportion_power(p, lift, n, h))
    assert n_total == pytest.approx(n_treated / (1 - h), rel=1e-6)


def test_unreachable_power_returns_infinity():
    assert duration_for_power(lambda d: 0.05 + 0.0 * d) == float("inf")
