"""The six scalar/mapping validators owned by zlc_data."""

from __future__ import annotations

import numpy as np
import pytest

from zlc_data.validation import (
    DIGEST_HEX,
    canonical_text,
    exact_mapping,
    finite_real,
    integer,
    nonnegative_integer,
    positive_integer,
    digest_text,
)


def test_validation_accepts_canonical_values_and_normalizes_numeric_scalars():
    digest = "ab" * (DIGEST_HEX // 2)
    assert canonical_text("owner-id", "owner") == "owner-id"
    assert canonical_text("", "optional", empty=True) == ""
    assert finite_real(np.float64(-0.0), "value") == 0.0
    assert nonnegative_integer(np.int64(3), "count") == 3
    assert positive_integer(2, "count") == 2
    assert digest_text(digest, "digest") == digest
    assert digest_text(None, "digest", optional=True) is None


def test_canonical_text_rejects_padding_and_wrong_types():
    with pytest.raises(ValueError):
        canonical_text(" padded ", "owner")
    with pytest.raises(ValueError):
        canonical_text("", "owner")
    with pytest.raises(ValueError):
        canonical_text(3, "owner")


def test_exact_mapping_owns_shape_and_discriminator_admission():
    value = {"schema": "owner-format", "payload": 3}
    assert exact_mapping(value, {"schema", "payload"}, "owner-format") is value
    with pytest.raises(ValueError, match="exactly"):
        exact_mapping({**value, "extra": 4}, {"schema", "payload"}, "owner-format")
    with pytest.raises(ValueError, match="expected schema"):
        exact_mapping(value, {"schema", "payload"}, "other-format")

    nested = {"name": "center", "fixed": None}
    assert exact_mapping(
        nested,
        {"name", "fixed"},
        "nested constraint",
        discriminator=None,
    ) is nested
    with pytest.raises(ValueError, match="exactly"):
        exact_mapping(
            {**nested, "extra": 1},
            {"name", "fixed"},
            "nested constraint",
            discriminator=None,
        )


def test_finite_real_enforces_finiteness_and_bounds():
    assert finite_real(2.5, "value", minimum=2.0) == 2.5
    assert finite_real(2.5, "value", positive=True) == 2.5
    with pytest.raises(TypeError):
        finite_real(True, "value")
    with pytest.raises(ValueError, match="finite"):
        finite_real(float("nan"), "value")
    with pytest.raises(ValueError, match="positive"):
        finite_real(0.0, "value", positive=True)
    with pytest.raises(ValueError, match="at least"):
        finite_real(1.0, "value", minimum=2.0)


def test_integer_validators_reject_booleans_and_wrong_bounds():
    assert integer(np.int64(3), minimum=0) == 3
    with pytest.raises(ValueError):
        integer(0, minimum=1)
    with pytest.raises(TypeError):
        nonnegative_integer(True, "count")
    with pytest.raises(ValueError):
        nonnegative_integer(-1, "count")
    with pytest.raises(ValueError):
        positive_integer(0, "count")
    with pytest.raises(TypeError):
        positive_integer(1.5, "count")


def test_digest_text_requires_lowercase_digest_shape():
    digest = "ab" * (DIGEST_HEX // 2)
    with pytest.raises(ValueError):
        digest_text(digest.upper(), "digest")
    with pytest.raises(ValueError):
        digest_text(digest[:-1], "digest")
    with pytest.raises(ValueError):
        digest_text(None, "digest")
