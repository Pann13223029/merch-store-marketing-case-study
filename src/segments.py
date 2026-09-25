"""Outside visitors kept out of the channel analysis and reported as their own segment, like employees.

A key account is an outside buyer large enough that it alone decides a channel's numbers. Found by the
visitor-concentration check (src/validate.py:visitor_concentration) in notebook 05:

  * 1957458976293878100: one US desktop visiting on weekday office hours, 278 sessions and 16
    purchase sessions worth $128,413 (15% of outside revenue in the attribution period), up to $47,082
    each. It was already buying before its only Display click (2017-03-10); GA then labelled its next
    15 purchases Display, which is 89% of Display's GA-credited revenue. Its purchases reflect an
    existing corporate relationship, not what Display or any other channel caused.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KEY_ACCOUNTS: frozenset[str] = frozenset({"1957458976293878100"})

EXTERNAL = "External"
EMPLOYEES = "Internal (Google employees)"
KEY_ACCOUNT = "Key account"


def is_key_account(full_visitor_id: pd.Series) -> pd.Series:
    """True for rows that belong to a key account."""
    return full_visitor_id.astype(str).isin(KEY_ACCOUNTS)


def segment(sessions: pd.DataFrame) -> pd.Series:
    """Each session's segment: External, Internal (Google employees), or Key account.

    Marketing figures use External sessions only; the other two are reported on their own. Employees take
    precedence, so a key account can only hold outside sessions.
    """
    internal = sessions.is_internal.astype(bool).to_numpy()
    key = is_key_account(sessions.full_visitor_id).to_numpy()
    return pd.Series(np.select([internal, key], [EMPLOYEES, KEY_ACCOUNT], EXTERNAL), index=sessions.index,
                     name="segment")
