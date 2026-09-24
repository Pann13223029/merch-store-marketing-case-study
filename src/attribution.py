"""Multi-touch attribution for D1: which channels deserve credit for purchases?

Input is data/processed/journeys.parquet (src/journeys.py): one row per journey of external
visitors, with the channel path three ways (arrival / conservative / GA labels), the outcome, and
revenue capped per order at the 99th percentile (D-P2).

Models (credit per purchase sums to 1):
  ga_last_click   GA's own report: the channel GA assigned to the purchase session
                  (last non-direct click, because GA relabels direct returns with the previous campaign)
  last_touch      the arrival channel of the purchase session
  first_touch     the arrival channel that started the journey
  linear          equal credit to every touch
  position_based  40% first, 40% last, 20% shared by the middle (50/50 for two touches)
  markov          data-driven: order-k Markov chain over all journeys (converting or not), with k
                  chosen by held-out likelihood. A channel's conversion credit is proportional to its
                  removal effect: the share of conversions lost if it were removed from every path.
                  Revenue credit uses the same chain with each step into a purchase carrying the
                  average order value of purchases made from that state, so channels keep the order
                  values that actually follow them (value removal effect).

Uncertainty: a Poisson bootstrap reweights journeys (weight ~ Poisson(1)) and recomputes every model
on the same weights, so differences between models are paired.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HEURISTIC_MODELS = ["ga_last_click", "last_touch", "first_touch", "linear", "position_based"]
MODELS = HEURISTIC_MODELS + ["markov"]


def split_paths(journeys: pd.DataFrame, path_col: str) -> pd.Series:
    return journeys[path_col].str.split(" > ")


class Attribution:
    """Precomputes per-journey credit and Markov transitions so bootstraps are fast."""

    def __init__(self, journeys: pd.DataFrame, path_col: str = "arrival_path",
                 revenue_col: str = "revenue_capped_usd", order: int = 1):
        self.path_col = path_col
        self.order = order
        paths = split_paths(journeys, path_col)
        ga_last = journeys["ga_last_channel"]
        self.channels = sorted(set(ch for p in paths for ch in p) | set(ga_last))
        self.idx = {c: i for i, c in enumerate(self.channels)}
        n_ch = len(self.channels)
        self.converted = journeys["converted"].to_numpy(dtype=bool)
        self.revenue = journeys[revenue_col].to_numpy(dtype=float)
        self.revenue_all = np.where(self.converted, self.revenue, 0.0)
        self.n_journeys = len(journeys)

        # Heuristic credit matrices over converting journeys: {model: (n_converting, n_channels)}
        conv_rows = np.flatnonzero(self.converted)
        self.conv_rows = conv_rows
        credit = {m: np.zeros((len(conv_rows), n_ch)) for m in HEURISTIC_MODELS}
        for r, j in enumerate(conv_rows):
            p = [self.idx[c] for c in paths.iat[j]]
            k = len(p)
            credit["ga_last_click"][r, self.idx[ga_last.iat[j]]] = 1.0
            credit["last_touch"][r, p[-1]] += 1.0
            credit["first_touch"][r, p[0]] += 1.0
            for c in p:
                credit["linear"][r, c] += 1.0 / k
            if k == 1:
                credit["position_based"][r, p[0]] += 1.0
            elif k == 2:
                credit["position_based"][r, p[0]] += 0.5
                credit["position_based"][r, p[1]] += 0.5
            else:
                credit["position_based"][r, p[0]] += 0.4
                credit["position_based"][r, p[-1]] += 0.4
                for c in p[1:-1]:
                    credit["position_based"][r, c] += 0.2 / (k - 2)
        self.credit = credit

        # Markov transitions over all journeys. A state is the tuple of the last `order` channels
        # (padded with -1 at the start of a journey); two absorbing states: conversion and null.
        self.t_from, self.t_to, self.t_journey, self.state_last_channel, _ = markov_transitions(
            [[self.idx[c] for c in p] for p in paths], self.converted, order)
        self.n_states = len(self.state_last_channel)
        self.CONV, self.NULL = self.n_states - 2, self.n_states - 1

    # ------------------------------------------------------------------ Markov
    def _absorption(self, P: np.ndarray, value: np.ndarray) -> tuple[float, float]:
        """(conversion probability, expected purchase value) starting from the journey start."""
        transient = np.arange(0, self.CONV)   # state 0 is the journey start
        Q = P[np.ix_(transient, transient)]
        start = np.zeros(len(transient)); start[0] = 1.0
        visits = np.linalg.solve((np.eye(len(transient)) - Q).T, start)   # expected visits per state
        to_conv = P[transient, self.CONV]
        return float(visits @ to_conv), float(visits @ (to_conv * value[transient]))

    def markov_removal_effects(self, journey_weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Share of conversions, and of expected purchase value, lost when each channel is removed."""
        w = np.ones(self.n_journeys) if journey_weights is None else journey_weights
        tw = w[self.t_journey]
        counts = np.bincount(self.t_from * self.n_states + self.t_to, weights=tw,
                             minlength=self.n_states ** 2).reshape(self.n_states, self.n_states)
        out = counts.sum(axis=1, keepdims=True)
        P = np.divide(counts, out, out=np.zeros_like(counts), where=out > 0)
        # average order value of the purchases made from each state
        is_conv = self.t_to == self.CONV
        conv_n = np.bincount(self.t_from[is_conv], weights=tw[is_conv], minlength=self.n_states)
        conv_v = np.bincount(self.t_from[is_conv], weights=(tw * self.revenue_all[self.t_journey])[is_conv],
                             minlength=self.n_states)
        value = np.divide(conv_v, conv_n, out=np.zeros_like(conv_v), where=conv_n > 0)
        base_p, base_v = self._absorption(P, value)
        conv_effect, value_effect = np.zeros(len(self.channels)), np.zeros(len(self.channels))
        for c in range(len(self.channels)):
            Pr = P.copy()
            into_c = np.flatnonzero(self.state_last_channel == c)
            Pr[:, self.NULL] += Pr[:, into_c].sum(axis=1)   # every visit to the removed channel is lost
            Pr[:, into_c] = 0.0
            p_c, v_c = self._absorption(Pr, value)
            conv_effect[c] = 1.0 - p_c / base_p if base_p > 0 else 0.0
            value_effect[c] = 1.0 - v_c / base_v if base_v > 0 else 0.0
        return conv_effect, value_effect

    # ------------------------------------------------------------------ credit
    def credit_table(self, journey_weights: np.ndarray | None = None) -> pd.DataFrame:
        """Conversions and capped revenue credited to each channel under every model."""
        w = np.ones(self.n_journeys) if journey_weights is None else journey_weights
        wc = w[self.conv_rows]
        total_conv = wc.sum()
        total_rev = (wc * self.revenue[self.conv_rows]).sum()
        rows = {}
        for m in HEURISTIC_MODELS:
            conv = (self.credit[m] * wc[:, None]).sum(axis=0)
            rev = (self.credit[m] * (wc * self.revenue[self.conv_rows])[:, None]).sum(axis=0)
            rows[(m, "conversions")] = conv
            rows[(m, "revenue")] = rev
        conv_effect, value_effect = self.markov_removal_effects(w)
        rows[("markov", "conversions")] = conv_effect / conv_effect.sum() * total_conv
        rows[("markov", "revenue")] = value_effect / value_effect.sum() * total_rev
        rows[("markov", "removal_effect")] = conv_effect
        rows[("markov", "value_removal_effect")] = value_effect
        table = pd.DataFrame(rows, index=self.channels)
        table.columns = pd.MultiIndex.from_tuples(table.columns, names=["model", "measure"])
        return table

    def shares(self, journey_weights: np.ndarray | None = None, measure: str = "conversions") -> pd.DataFrame:
        t = self.credit_table(journey_weights).xs(measure, axis=1, level="measure")
        return t / t.sum()

    def bootstrap_shares(self, n_boot: int = 1000, seed: int = 42, measure: str = "conversions") -> np.ndarray:
        """Array (n_boot, n_channels, n_models) of credit shares under Poisson journey weights."""
        rng = np.random.default_rng(seed)
        out = np.empty((n_boot, len(self.channels), len(MODELS)))
        for b in range(n_boot):
            s = self.shares(rng.poisson(1.0, self.n_journeys).astype(float), measure)
            out[b] = s[MODELS].to_numpy()
        return out


def markov_transitions(paths: list[list[int]], converted: np.ndarray, order: int):
    """Transitions between order-k states for every journey.

    Returns (from_state, to_state, journey_index, last_channel_of_each_state, state_ids). State 0 is
    the start; the last two states are conversion and null (last channel -2 and -3); state_ids maps
    each transient state tuple to its index.
    """
    state_id = {(-1,) * order: 0}
    frm, to, jid = [], [], []
    for j, p in enumerate(paths):
        prev = (-1,) * order
        for c in p:
            nxt = prev[1:] + (c,)
            frm.append(state_id.setdefault(prev, len(state_id)))
            to.append(state_id.setdefault(nxt, len(state_id)))
            jid.append(j)
            prev = nxt
        frm.append(state_id.setdefault(prev, len(state_id)))
        to.append(-2 if converted[j] else -1)      # placeholders for the absorbing states
        jid.append(j)
    n_transient = len(state_id)
    to = np.asarray(to)
    to = np.where(to == -2, n_transient, np.where(to == -1, n_transient + 1, to))
    last = np.full(n_transient + 2, -3)
    for st, i in state_id.items():
        last[i] = st[-1]
    last[n_transient] = -2
    return np.asarray(frm), to, np.asarray(jid), last, state_id


def markov_holdout_loglik(train_paths, train_conv, test_paths, test_conv, n_channels: int, order: int,
                          alpha: float = 0.1) -> float:
    """Mean log-likelihood per test journey under an order-k chain fit on training journeys.

    Transition counts are smoothed with a small pseudo-count so unseen transitions keep probability.
    """
    f, t, _, _, state_key = markov_transitions(train_paths, train_conv, order)
    from collections import Counter
    counts = Counter(zip(f.tolist(), t.tolist()))
    outflow = Counter(f.tolist())
    n_targets = n_channels + 2   # next channel, conversion, or null
    n_transient = len(state_key)
    total = 0.0
    for p, conv in zip(test_paths, test_conv):
        prev = (-1,) * order
        ll = 0.0
        for step in list(p) + ["END"]:
            s_from = state_key.get(prev)
            if step == "END":
                target = n_transient if conv else n_transient + 1
            else:
                nxt = prev[1:] + (step,)
                target = state_key.get(nxt, -99)
            c = counts.get((s_from, target), 0) if s_from is not None and target != -99 else 0
            o = outflow.get(s_from, 0) if s_from is not None else 0
            ll += np.log((c + alpha) / (o + alpha * n_targets))
            if step != "END":
                prev = prev[1:] + (step,)
        total += ll
    return total / len(test_paths)


def starter_closer(journeys: pd.DataFrame, path_col: str = "arrival_path") -> pd.DataFrame:
    """For multi-touch converting journeys: how often each channel starts, assists, or closes."""
    conv = journeys[journeys.converted & (journeys.n_touches > 1)]
    paths = split_paths(conv, path_col)
    rows = []
    for p in paths:
        rows.append(("starts", p[0]))
        rows.append(("closes", p[-1]))
        for c in set(p[1:-1]):
            rows.append(("assists", c))
    t = pd.DataFrame(rows, columns=["role", "channel"]).value_counts().unstack("role").fillna(0).astype(int)
    t["start_to_close_ratio"] = t["starts"] / t["closes"].where(t["closes"] > 0)
    return t.sort_values("starts", ascending=False)
