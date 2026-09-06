"""The phase solver's prepared-input cache: identity stands in for an input
only while nothing can write that input."""

from __future__ import annotations

import numpy as np
import pytest

from zlc_atom.devices.slm import solver
from zlc_atom.devices.slm.solver import solve_phase, validate_target


def _two_spots() -> np.ndarray:
    target = np.zeros((32, 32), dtype=np.float32)
    target[8, 8] = target[24, 24] = 1.0
    return target


def test_a_cleared_writeable_flag_is_not_immutability_for_the_prepared_cache() -> None:
    """An owned array made read-only can be made writable again and changed;
    the cache used to keep answering from its old contents.

    Solved once as read-only, then made writable and zeroed, the SAME
    object was solved "again" from the cached two-spot target -- the
    input's own validation ("positive intensity") never ran.  Only memory
    that has no writable form is remembered by identity.
    """

    target = _two_spots()
    target.setflags(write=False)
    solve_phase(target, objective_kind="spots", iterations=1, seed=0)
    assert id(target) not in solver._PREPARED_TARGETS

    target.setflags(write=True)
    target[:] = 0.0
    with pytest.raises(ValueError, match="positive intensity"):
        solve_phase(target, objective_kind="spots", iterations=1, seed=0)


def test_a_frozen_input_is_remembered_by_identity() -> None:
    """What the cache is for: a feedback run hands one frozen target to every
    candidate solve, and a bytes-backed array cannot be written by anyone."""

    frozen = validate_target(_two_spots())
    with pytest.raises(ValueError):
        frozen.setflags(write=True)
    first, _metadata = solve_phase(frozen, objective_kind="spots", iterations=1, seed=0)
    assert id(frozen) in solver._PREPARED_TARGETS
    second, _metadata = solve_phase(frozen, objective_kind="spots", iterations=1, seed=0)
    np.testing.assert_array_equal(first, second)
