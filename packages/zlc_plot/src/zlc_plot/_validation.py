"""Validators for backend-neutral plotting records.

What a real number or an integer IS belongs to zlc_data, which this package
already depends on, and is not restated here.  It was: two copies of the same
two rules, in packages one of which imports the other, drifting quietly -- one
grew a minimum, the other grew an optional, and neither knew.

What is here is what plotting genuinely owns: text that may be padded because a
human typed it, and array storage that must not be written through.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np

from zlc_data import finite_real as _finite_real
from zlc_data import integer as _integer


def finite_real(value: object, field: str) -> float:
    """One finite real, by zlc_data's rule.

    Zero is left exactly as given: a plot limit of -0.0 came from data, and
    canonicalising it is a storage concern rather than a drawing one.
    """

    return _finite_real(value, field, normalize_zero=False)


def integer(
    value: object,
    field: str,
    *,
    minimum: int | None = None,
    optional: bool = False,
) -> int | None:
    """One integer, by zlc_data's rule, optionally absent.

    Absence is the plotting-side addition: a panel with no pinned row count is
    a real state, whereas stored data either has the number or is malformed.
    """

    if value is None and optional:
        return None
    if optional and (isinstance(value, bool) or not isinstance(value, Integral)):
        raise TypeError(f"{field} must be an integer or None")
    return _integer(value, field, minimum=minimum)


def text(value: object, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be text")
    result = value.strip()
    if not result and not allow_empty:
        raise ValueError(f"{field} must be non-empty")
    return result


def optional_nonempty_text(value: object, field: str) -> str | None:
    return None if value is None else text(value, field)


def readonly_copy(values: Any, *, dtype: Any | None = None) -> np.ndarray:
    """Return independent array storage under a read-only view contract."""

    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result
