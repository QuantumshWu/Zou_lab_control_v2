"""A frame prepared under one spec may not be committed under another.

A live frame is prepared off the owner thread and committed later.  The
commit already refuses a frame whose display parameters, data revision or
image overlay have moved underneath it -- but not one whose SPEC has.  A
semantic edit (an axis fate) replaces the spec through ``replace_spec``,
which no parameter-schema check can see, so an in-flight frame landed its
old-spec payload beside the new spec.  The session then held a spec and a
payload that were never one accepted view, and the first thing to ask
them a question -- a selector, wanting its selection subject -- said so:
"selection subject payload differs from FacetGrid spec".
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
import pytest

from data_factory import (
    Axis,
    DatasetSchema,
    DatasetSnapshot,
    PointTable,
    PointTopology,
)
from zlc_data import AxisId
from zlc_plot import AxisRef, CurvePlot, FacetGridPlot, PlotSession


def _snapshot(revision: int):
    rows = [(i % 4, i // 4) for i in range(12)]
    schema = DatasetSchema.create(
        Axis.create("repeat", size=2),
        PointTable.from_columns({
            "ax": np.asarray([float(r[0]) for r in rows]),
            "ay": np.asarray([float(r[1]) for r in rows]),
        }),
        data_axes=(Axis.create("site", values=[0.0, 1.0, 2.0]),),
        dtype=np.float64,
        generation="spec-currency",
        point_topology=PointTopology(
            (AxisId("ax"), AxisId("ay")),
            ((0.0, 1.0, 2.0, 3.0), (0.0, 1.0, 2.0)),
            tuple(rows),
        ),
    )
    rng = np.random.default_rng(revision)
    return DatasetSnapshot(
        schema, rng.normal(size=(2, len(rows), 3)), revision=revision
    )


def test_a_frame_prepared_under_the_old_spec_is_refused() -> None:
    session = PlotSession(
        _snapshot(1),
        FacetGridPlot(AxisRef.data("site"), CurvePlot(AxisRef.point("ax"))),
    )
    try:
        # A frame is prepared while the operator is still looking at the
        # grid they authored.
        prepared = session.prepare_live_frame(_snapshot(2)).result(timeout=20)

        # Then they hand x to another axis: a NEW spec, and with it a new
        # projection and payload.
        session.apply_semantic("fate:point_dimension:ay", "x")
        current_spec = session.spec

        # The in-flight frame must not land its old-spec payload here.
        assert session.commit_live_frame(prepared) is None
        assert session.spec == current_spec

        # And the session is still one accepted view: the thing that used to
        # break -- asking for the selection subject -- works.
        session._selection_subject()
    finally:
        session.close()


def test_a_frame_prepared_under_the_current_spec_still_commits() -> None:
    """The refusal is about the SPEC, not about live frames."""

    session = PlotSession(
        _snapshot(1),
        FacetGridPlot(AxisRef.data("site"), CurvePlot(AxisRef.point("ax"))),
    )
    try:
        prepared = session.prepare_live_frame(_snapshot(2)).result(timeout=20)
        finalization = session.commit_live_frame(prepared)
        assert finalization is not None
        session.publish_live_frame(finalization)
        assert session.data_revision == 2
        session._selection_subject()
    finally:
        session.close()
