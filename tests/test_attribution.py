"""Tests for src/attribution.py: the credit rules, the Markov chain, conservation, and the presence check.

The toy journeys used throughout: "A > B" buys ($30), "A" doesn't buy, "B" buys ($90).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.attribution import (HEURISTIC_MODELS, MODELS, Attribution, credit_vs_presence, markov_holdout_loglik,
                             markov_transitions, starter_closer)

# The heuristics that credit only channels in the path (GA's label can name a campaign from outside it)
PATH_MODELS = [m for m in HEURISTIC_MODELS if m != "ga_last_click"]


def journeys_frame(rows: list[tuple[str, bool, float, str]]) -> pd.DataFrame:
    """Journeys table from (arrival_path, converted, capped revenue, ga_last_channel) rows."""
    return pd.DataFrame(rows, columns=["arrival_path", "converted", "revenue_capped_usd", "ga_last_channel"])


def random_journeys(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Paths of 1-6 touches over five channels; about a third convert. GA's label is sometimes off-path."""
    rng = np.random.default_rng(seed)
    channels = ["Direct", "Organic Search", "Paid Search", "Referral", "Social"]
    paths = [list(rng.choice(channels, rng.integers(1, 7))) for _ in range(n)]
    converted = rng.random(n) < 0.35
    return pd.DataFrame({
        "arrival_path": [" > ".join(p) for p in paths],
        "converted": converted,
        "revenue_capped_usd": np.where(converted, rng.gamma(2.0, 40.0, n).round(2), 0.0),
        "ga_last_channel": [p[-1] if rng.random() < 0.7 else str(rng.choice(channels)) for p in paths],
    })


def credit(att: Attribution, model: str, measure: str = "conversions") -> dict[str, float]:
    return att.credit_table()[(model, measure)].to_dict()


@pytest.fixture
def toy() -> Attribution:
    """First-order chain on the toy journeys. GA labels the "A > B" purchase session with A (a relabelled return)."""
    return Attribution(journeys_frame([("A > B", True, 30.0, "A"), ("A", False, 0.0, "A"), ("B", True, 90.0, "B")]),
                       order=1)


# ---------------------------------------------------------------- Markov chain

def test_markov_removal_effects_match_hand_calculation(toy):
    # Chain: start -> A 2/3, start -> B 1/3; A -> B 1/2, A -> null 1/2; B -> purchase 1.
    # P(purchase) = 2/3 * 1/2 + 1/3 = 2/3.
    # Without A only start -> B converts (1/3): A's removal effect is 1 - (1/3) / (2/3) = 0.5.
    # Without B nothing converts: B's removal effect is 1.
    conv_effect, value_effect = toy.markov_removal_effects()
    assert conv_effect == pytest.approx([0.5, 1.0])
    assert value_effect == pytest.approx([0.5, 1.0])   # every purchase is made from B, so value is lost alike


def test_markov_credit_is_proportional_to_removal_effect(toy):
    # 2 purchases and $120 shared 0.5 : 1.0
    assert credit(toy, "markov") == pytest.approx({"A": 2 / 3, "B": 4 / 3})
    assert credit(toy, "markov", "revenue") == pytest.approx({"A": 40.0, "B": 80.0})
    table = toy.credit_table()
    assert table[("markov", "removal_effect")].tolist() == pytest.approx([0.5, 1.0])


def test_markov_revenue_credit_keeps_the_order_values_that_follow_each_channel():
    # "A" buys $100; "B" buys $10, buys $30, and once doesn't buy. From B, 2/3 of visits buy, averaging $20.
    # P(purchase) = 1/4 + 3/4 * 2/3 = 3/4; expected value = 1/4 * 100 + 3/4 * 2/3 * 20 = 35.
    # Without A: 1/2 and $10, so removal effects 1/3 and 5/7. Without B: 1/4 and $25, so 2/3 and 2/7.
    att = Attribution(journeys_frame([("A", True, 100.0, "A"), ("B", True, 10.0, "B"), ("B", True, 30.0, "B"),
                                      ("B", False, 0.0, "B")]))
    conv_effect, value_effect = att.markov_removal_effects()
    assert conv_effect == pytest.approx([1 / 3, 2 / 3])
    assert value_effect == pytest.approx([5 / 7, 2 / 7])
    # with single-touch journeys every channel keeps exactly what its own journeys produced
    assert credit(att, "markov") == pytest.approx({"A": 1.0, "B": 2.0})
    assert credit(att, "markov", "revenue") == pytest.approx({"A": 100.0, "B": 40.0})


def test_markov_states_remember_the_last_k_channels():
    # Order 2: a state is the last two channels, padded with -1 at the start of a journey.
    frm, to, journey, last_channel, state_ids = markov_transitions([[0, 1, 1], [1]], np.array([True, False]), order=2)
    assert state_ids == {(-1, -1): 0, (-1, 0): 1, (0, 1): 2, (1, 1): 3, (-1, 1): 4}
    conv, null = 5, 6   # the two absorbing states follow the transient ones
    assert list(zip(frm.tolist(), to.tolist())) == [(0, 1), (1, 2), (2, 3), (3, conv), (0, 4), (4, null)]
    assert journey.tolist() == [0, 0, 0, 0, 1, 1]
    assert last_channel.tolist() == [-1, 0, 1, 1, 1, -2, -3]


def test_higher_order_chain_separates_histories():
    # X never converts but is followed by D, which does. A first-order chain can't tell "D after X" from
    # "D first", so removing X loses half the purchases; a second-order chain remembers X and loses none.
    j = journeys_frame([("X > D", False, 0.0, "D"), ("D", True, 50.0, "D")])
    first, _ = Attribution(j, order=1).markov_removal_effects()
    second, _ = Attribution(j, order=2).markov_removal_effects()
    assert first.tolist() == pytest.approx([1.0, 0.5])    # channels sorted: D, X
    assert second.tolist() == pytest.approx([1.0, 0.0])


def test_holdout_loglik_hand_calculation():
    # Fit on one converting journey [0]: start -> 0 once, 0 -> purchase once. With a 0.1 pseudo-count over
    # n_channels + 2 = 3 targets, each observed step has probability (1 + 0.1) / (1 + 0.3).
    ll = markov_holdout_loglik([[0]], np.array([True]), [[0]], np.array([True]), n_channels=1, order=1)
    assert ll == pytest.approx(2 * np.log(1.1 / 1.3))


def test_holdout_loglik_smooths_unseen_steps():
    # Channel 1 never appears in training: the step into it gets 0.1 / (1 + 4 * 0.1), and its unseen state
    # spreads probability evenly over the 4 targets.
    ll = markov_holdout_loglik([[0]], np.array([True]), [[1]], np.array([False]), n_channels=2, order=1)
    assert ll == pytest.approx(np.log(0.1 / 1.4) + np.log(1 / 4))


# ---------------------------------------------------------------- heuristic models

@pytest.mark.parametrize("model, conversions, revenue", [
    ("first_touch", {"A": 1.0, "B": 1.0}, {"A": 30.0, "B": 90.0}),
    ("last_touch", {"A": 0.0, "B": 2.0}, {"A": 0.0, "B": 120.0}),
    ("linear", {"A": 0.5, "B": 1.5}, {"A": 15.0, "B": 105.0}),
    ("position_based", {"A": 0.5, "B": 1.5}, {"A": 15.0, "B": 105.0}),
    ("ga_last_click", {"A": 1.0, "B": 1.0}, {"A": 30.0, "B": 90.0}),
])
def test_heuristic_credit_on_toy(toy, model, conversions, revenue):
    assert credit(toy, model) == pytest.approx(conversions)
    assert credit(toy, model, "revenue") == pytest.approx(revenue)


@pytest.mark.parametrize("path, expected", [
    ("A", {"A": 1.0}),
    ("A > B", {"A": 0.5, "B": 0.5}),
    ("A > B > C", {"A": 0.4, "B": 0.2, "C": 0.4}),
    ("A > B > C > D", {"A": 0.4, "B": 0.1, "C": 0.1, "D": 0.4}),
    ("A > B > A", {"A": 0.8, "B": 0.2}),   # a repeated channel collects every share it holds
])
def test_position_based_splits_40_20_40(path, expected):
    att = Attribution(journeys_frame([(path, True, 100.0, "A")]))
    got = credit(att, "position_based")
    assert got == pytest.approx({c: expected.get(c, 0.0) for c in got})
    assert credit(att, "position_based", "revenue") == pytest.approx({c: 100 * expected.get(c, 0.0) for c in got})


@pytest.mark.parametrize("path, first, last", [("A", "A", "A"), ("A > B > C", "A", "C"), ("B > A > B", "B", "B")])
def test_first_and_last_touch(path, first, last):
    att = Attribution(journeys_frame([(path, True, 10.0, "A")]))
    assert credit(att, "first_touch") == {c: float(c == first) for c in att.channels}
    assert credit(att, "last_touch") == {c: float(c == last) for c in att.channels}


def test_ga_last_click_credits_ga_label_not_the_path():
    # GA labels a direct return with the earlier campaign, so the purchase session reads "Organic Search";
    # its label can even name a channel that isn't in the arrival path at all.
    att = Attribution(journeys_frame([
        ("Organic Search > Direct", True, 50.0, "Organic Search"),
        ("Direct", True, 20.0, "Referral"),
        ("Social", False, 0.0, "Social"),
    ]))
    assert credit(att, "ga_last_click") == {"Direct": 0.0, "Organic Search": 1.0, "Referral": 1.0, "Social": 0.0}
    assert credit(att, "ga_last_click", "revenue") == {"Direct": 0.0, "Organic Search": 50.0, "Referral": 20.0,
                                                       "Social": 0.0}
    assert credit(att, "last_touch") == {"Direct": 2.0, "Organic Search": 0.0, "Referral": 0.0, "Social": 0.0}


def test_starter_closer_counts_roles_in_multi_touch_purchases():
    j = pd.DataFrame({"arrival_path": ["A > B > C", "A > C", "C > B > B > A", "B", "A > B"],
                      "converted": [True, True, True, True, False]})
    j["n_touches"] = j.arrival_path.str.count(" > ") + 1
    roles = starter_closer(j)   # the single-touch and the non-converting journey are left out
    assert roles.index.tolist() == ["A", "C", "B"]
    assert roles[["starts", "closes", "assists"]].to_dict("index") == {
        "A": {"starts": 2, "closes": 1, "assists": 0},
        "C": {"starts": 1, "closes": 2, "assists": 0},
        "B": {"starts": 0, "closes": 0, "assists": 2},   # once per journey, however often it repeats
    }
    assert roles.start_to_close_ratio[["A", "C"]].tolist() == [2.0, 0.5]
    assert np.isnan(roles.start_to_close_ratio["B"])


# ---------------------------------------------------------------- conservation and bootstrap

@pytest.mark.parametrize("order", [1, 2, 3])
def test_every_model_conserves_conversions_and_revenue(order):
    j = random_journeys()
    table = Attribution(j, order=order).credit_table()
    conv = j[j.converted]
    for model in MODELS:
        assert table[(model, "conversions")].sum() == pytest.approx(len(conv)), model
        assert table[(model, "revenue")].sum() == pytest.approx(conv.revenue_capped_usd.sum()), model


def test_weighted_credit_conserves_weighted_totals():
    j = random_journeys()
    w = np.random.default_rng(1).poisson(1.0, len(j)).astype(float)
    table = Attribution(j, order=2).credit_table(w)
    conv = j.converted.to_numpy()
    for model in MODELS:
        assert table[(model, "conversions")].sum() == pytest.approx(w[conv].sum()), model
        assert table[(model, "revenue")].sum() == pytest.approx((w * j.revenue_capped_usd)[conv].sum()), model


def test_integer_journey_weights_equal_duplicated_journeys():
    j = random_journeys(n=150)
    w = np.random.default_rng(2).integers(1, 4, len(j))
    weighted = Attribution(j, order=2).credit_table(w.astype(float))
    duplicated = Attribution(j.loc[j.index.repeat(w)], order=2).credit_table()
    pd.testing.assert_frame_equal(weighted, duplicated)


def test_shares_divide_each_model_by_its_total(toy):
    assert toy.shares()["markov"].tolist() == pytest.approx([1 / 3, 2 / 3])
    assert toy.shares(measure="revenue")["first_touch"].tolist() == pytest.approx([0.25, 0.75])


def test_bootstrap_shares_reweight_journeys_with_poisson_draws():
    att = Attribution(random_journeys(), order=2)
    boot = att.bootstrap_shares(n_boot=20, seed=7)
    assert boot.shape == (20, len(att.channels), len(MODELS))
    assert boot.sum(axis=1) == pytest.approx(np.ones((20, len(MODELS))))
    first_draw = np.random.default_rng(7).poisson(1.0, att.n_journeys).astype(float)
    assert boot[0] == pytest.approx(att.shares(first_draw)[MODELS].to_numpy())
    assert np.array_equal(boot, att.bootstrap_shares(n_boot=20, seed=7))


# ---------------------------------------------------------------- credit vs presence

def test_presence_counts_the_converting_journeys_that_contain_each_channel(toy):
    check = credit_vs_presence(toy)
    assert check.presence.to_dict() == {"A": 1.0, "B": 2.0}   # the lone "A" journey didn't buy
    assert check.credited.to_dict() == pytest.approx({"A": 2 / 3, "B": 4 / 3})
    assert not check.exceeds_presence.any()
    assert credit_vs_presence(toy, measure="revenue").presence.to_dict() == {"A": 30.0, "B": 120.0}


def test_presence_flag_fires_when_markov_credits_a_channel_beyond_its_journeys():
    # X appears in no purchasing journey, yet the first-order chain gives it removal effect 0.5 against D's 1.0,
    # so X is credited with 0.5 / 1.5 of the one purchase and of its $60.
    j = journeys_frame([("X > D", False, 0.0, "D"), ("D", True, 60.0, "D")])
    att = Attribution(j, order=1)
    check = credit_vs_presence(att)
    assert check.loc["X", "credited"] == pytest.approx(1 / 3)
    assert check.loc["X", "presence"] == 0
    assert check.exceeds_presence.to_dict() == {"D": False, "X": True}
    revenue = credit_vs_presence(att, measure="revenue")
    assert revenue.loc["X", "credited"] == pytest.approx(20.0)
    assert revenue.exceeds_presence.to_dict() == {"D": False, "X": True}
    # A second-order chain remembers that D-after-X never converts, and the flag goes quiet.
    assert not credit_vs_presence(Attribution(j, order=2)).exceeds_presence.any()


def test_presence_flag_fires_for_a_ga_label_outside_the_path():
    att = Attribution(journeys_frame([("Direct", True, 20.0, "Referral")]))
    assert credit_vs_presence(att, "ga_last_click").exceeds_presence.to_dict() == {"Direct": False, "Referral": True}


@pytest.mark.parametrize("measure", ["conversions", "revenue"])
def test_path_based_credit_never_exceeds_presence(measure):
    j = random_journeys()
    att = Attribution(j, order=2)
    conv = j[j.converted]
    weight = conv.revenue_capped_usd if measure == "revenue" else pd.Series(1.0, index=conv.index)
    paths = conv.arrival_path.str.split(" > ")
    expected = {c: weight[paths.map(lambda p: c in p)].sum() for c in att.channels}
    for model in PATH_MODELS:
        check = credit_vs_presence(att, model, measure)
        assert check.presence.to_dict() == pytest.approx(expected), model
        assert not check.exceeds_presence.any(), model


def test_presence_check_rejects_other_measures(toy):
    with pytest.raises(ValueError, match="measure"):
        credit_vs_presence(toy, measure="removal_effect")
