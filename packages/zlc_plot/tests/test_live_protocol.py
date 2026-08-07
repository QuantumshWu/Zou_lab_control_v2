from __future__ import annotations

import numpy as np
import pytest

from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable
from zlc_plot import AxisRef, CurvePlot, PlotSession


def _snap(revision: int) -> DatasetSnapshot:
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"x": [0.0, 1.0, 2.0]}),
        dtype=np.float64,
        generation="live-protocol",
    )
    return DatasetSnapshot(
        schema,
        np.array([[1.0, 2.0, 3.0]], dtype=np.float64) + revision,
        revision=revision,
    )


def _session() -> PlotSession:
    return PlotSession(_snap(0), CurvePlot(AxisRef.point("x")))


def test_prepare_commit_abort_is_atomic_and_abort_restores_front() -> None:
    session = _session()
    try:
        prepared = session.prepare_live_frame(_snap(1)).result(timeout=10)
        finalization = session.commit_live_frame(prepared)
        assert finalization is not None
        assert session.data_revision == 1
        session.abort_live_frame(finalization)
        assert session.data_revision == 0

        prepared = session.prepare_live_frame(_snap(1)).result(timeout=10)
        finalization = session.commit_live_frame(prepared)
        assert finalization is not None
        session.finalize_live_frame(finalization)
        assert session.data_revision == 1
    finally:
        session.close()


def test_stale_revision_is_rejected_before_prepare_and_commit() -> None:
    session = _session()
    try:
        with pytest.raises(ValueError, match="must increase"):
            session.prepare_live_frame(_snap(0)).result(timeout=10)
        prepared = session.prepare_live_frame(_snap(1)).result(timeout=10)
        session.update_data(_snap(2))
        assert session.commit_live_frame(prepared) is None
        assert session.data_revision == 2
    finally:
        session.close()
