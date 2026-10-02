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
on the same weights, so differences between models are paired. Passing the visitor as the cluster gives
all of a visitor's journeys one shared weight, so repeat buyers are resampled as a unit.

Paid Search keywords (search_keyword_class) sort each ad click's keyword into brand or store-name
searches, other readable keywords, targeting placeholders, obfuscated IDs, and missing values.
"""

from __future__ import annotations

import re

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
                             minlength=self.n_states ** 2).astype(float).reshape(self.n_states, self.n_states)
        out = counts.sum(axis=1, keepdims=True)
        P = np.divide(counts, out, out=np.zeros_like(counts), where=out > 0)
        # average order value of the purchases made from each state
        is_conv = self.t_to == self.CONV
        conv_n = np.bincount(self.t_from[is_conv], weights=tw[is_conv], minlength=self.n_states).astype(float)
        conv_v = np.bincount(self.t_from[is_conv], weights=(tw * self.revenue_all[self.t_journey])[is_conv],
                             minlength=self.n_states).astype(float)
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
        if total_conv == 0:
            raise ValueError("no converting journeys (or all have weight 0); Markov credit is undefined")
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

    def bootstrap_shares(self, n_boot: int = 1000, seed: int = 42, measure: str = "conversions",
                         clusters: np.ndarray | pd.Series | None = None) -> np.ndarray:
        """Array (n_boot, n_channels, n_models) of credit shares under Poisson journey weights.

        With `clusters` (one label per journey, such as the visitor), each cluster draws one Poisson(1)
        weight that all its journeys share, so a visitor's journeys are resampled together.
        """
        if clusters is None:
            codes, n_draws = np.arange(self.n_journeys), self.n_journeys
        else:
            if len(clusters) != self.n_journeys:
                raise ValueError(f"clusters needs one label per journey ({self.n_journeys}), got {len(clusters)}")
            codes, labels = pd.factorize(np.asarray(clusters))
            n_draws = len(labels)
        rng = np.random.default_rng(seed)
        out = np.empty((n_boot, len(self.channels), len(MODELS)))
        for b in range(n_boot):
            s = self.shares(rng.poisson(1.0, n_draws)[codes].astype(float), measure)
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


def credit_vs_presence(att: Attribution, model: str = "markov", measure: str = "conversions") -> pd.DataFrame:
    """Each channel's credit against the converting journeys it actually appears in.

    A channel can't earn more purchases than the purchasing journeys that contain it, nor more revenue
    than those journeys brought in. The path-based models (first and last touch, linear, position-based)
    stay within that bound by construction. GA's last click can exceed it, because GA may label the
    purchase session with a campaign from outside the journey, and so can the Markov chain, which
    recombines observed steps into paths nobody took (the third-order chain credits Affiliates with
    16.1 purchases from 4 purchasing journeys).

    Returns one row per channel: credited (credit under `model`), presence (converting journeys that
    contain the channel, or their capped revenue when measure="revenue"), and exceeds_presence.
    """
    if measure not in ("conversions", "revenue"):
        raise ValueError(f"measure must be 'conversions' or 'revenue', not {measure!r}")
    # A channel appears in a journey exactly when linear attribution gives it credit there.
    present = att.credit["linear"] > 0
    weight = att.revenue[att.conv_rows] if measure == "revenue" else np.ones(len(att.conv_rows))
    presence = (present * weight[:, None]).sum(axis=0)
    credited = att.credit_table()[(model, measure)].to_numpy()
    return pd.DataFrame({
        "credited": credited,
        "presence": presence,
        "exceeds_presence": (credited > presence) & ~np.isclose(credited, presence),   # beyond float rounding
    }, index=att.channels)


# ---------------------------------------------------------------------- Paid Search keywords
KEYWORD_CLASSES = ["Brand or store name", "Readable, no brand", "Targeting or automatic", "Obfuscated ID", "Missing"]
BRAND_TERMS = re.compile(r"\b(?:google|goggle|youtube|android|chrome|nest)\b")   # the store's brands, and a typo
AD_CATEGORIES = {"arts & entertainment"}   # Google Ads audience categories that appear as keywords


def search_keyword_class(keyword: object) -> str:
    """Sort a Paid Search click's keyword (trafficSource.keyword) into one of KEYWORD_CLASSES.

      Missing                 no keyword, "(not provided)" or "(not set)"
      Targeting or automatic  a placeholder in parentheses such as "(automatic matching)" or
                              "(Remarketing/Content targeting)", a product-feed rule ("category_l1==166",
                              "brand==nest"), or an audience category ("Arts & Entertainment")
      Obfuscated ID           a 16-character token such as "6qEhsCssdK0z36ri", which hides the query
      Brand or store name     names one of the store's brands (Google, YouTube, Android, Chrome, Nest), for
                              example "google merchandise store", "+youtube +merch", "google stickers"
      Readable, no brand      any other readable keyword, for example "+mens +sunglasses"
    """
    if keyword is None or pd.isna(keyword) or not str(keyword).strip():
        return "Missing"
    text = str(keyword).strip()
    lower = text.lower()
    if lower in ("(not provided)", "(not set)"):
        return "Missing"
    if (text.startswith("(") and text.endswith(")")) or "==" in text or lower in AD_CATEGORIES:
        return "Targeting or automatic"
    if re.fullmatch(r"[A-Za-z0-9_-]{16}", text) and re.search(r"\d", text):
        return "Obfuscated ID"
    if BRAND_TERMS.search(lower.replace("+", " ")):
        return "Brand or store name"
    return "Readable, no brand"
