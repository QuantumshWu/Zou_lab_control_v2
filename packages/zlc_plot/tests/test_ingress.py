from __future__ import annotations

import numpy as np
import pytest

from data_factory import Axis, DatasetSchema, PointTable, snapshot_validity, snapshot_values
from zlc_plot._live_channel import LatestIngress
from zlc_plot.errors import RevisionError


@pytest.fixture
def schema() -> DatasetSchema:
    return DatasetSchema.create(
        Axis.create("repeat", size=2),
        PointTable.from_columns({"x": [0.0, 1.0, 2.0]}),
        dtype=np.float32,
        generation="ingress-test",
    )


def test_patch_updates_values_and_validity_without_mutating_previous(schema: DatasetSchema) -> None:
    ingress = LatestIngress(schema)
    initial = ingress.replace(
        np.zeros(schema.shape, dtype=np.float32),
        revision=0,
        validity=np.zeros(schema.shape, dtype=np.bool_),
    )
    next_snapshot = ingress.patch(
        (slice(None), 1, 0),
        np.array([5.0, 6.0], dtype=np.float32),
        revision=1,
        valid=np.array([True, False], dtype=np.bool_),
    )
    assert np.array_equal(snapshot_values(initial), np.zeros(schema.shape, dtype=np.float32))
    assert np.array_equal(snapshot_values(next_snapshot)[:, 1], [[5.0], [6.0]])
    assert np.array_equal(snapshot_validity(next_snapshot)[:, 1], [[True], [False]])
    assert np.all(snapshot_validity(next_snapshot)[:, 0] == False)
    assert snapshot_values(next_snapshot).flags.writeable is False


def test_rolling_replaces_tail_and_propagates_validity(schema: DatasetSchema) -> None:
    ingress = LatestIngress(schema)
    ingress.replace(
        np.array([[[0], [1], [2]], [[10], [11], [12]]], dtype=np.float32),
        revision=0,
        validity=np.array([[[True], [True], [False]], [[True], [False], [True]]], dtype=np.bool_),
    )
    snapshot = ingress.rolling(
        np.array([[[20]], [[30]]], dtype=np.float32),
        revision=1,
        axis=1,
        validity=np.array([[[False]], [[True]]], dtype=np.bool_),
    )
    assert np.array_equal(snapshot_values(snapshot), [[[1], [2], [20]], [[11], [12], [30]]])
    assert np.array_equal(snapshot_validity(snapshot), [[[True], [False], [False]], [[False], [True], [True]]])


def test_ingress_rejects_stale_revision_and_invalid_rolling_shape(schema: DatasetSchema) -> None:
    ingress = LatestIngress(schema)
    ingress.replace(np.zeros(schema.shape, dtype=np.float32), revision=3)
    with pytest.raises(RevisionError):
        ingress.replace(np.zeros(schema.shape, dtype=np.float32), revision=3)
    with pytest.raises(RevisionError):
        ingress.rolling(np.zeros((2, 1, 1), dtype=np.float32), revision=2)
    with pytest.raises(ValueError):
        ingress.rolling(np.zeros((2, 4, 1), dtype=np.float32), revision=4)
