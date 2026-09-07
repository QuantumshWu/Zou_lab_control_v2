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
    axis,
    make_dataset_schema,
    make_snapshot,
    mapped_domain_from_columns,
    repeat_domain,
)
from zlc_data import REPEAT, SITE
from zlc_plot import AxisRef, CurvePlot, FacetGridPlot, HistogramPlot, PlotSession

def _snapshot(revision: int):
    rows = [(i % 4, i // 4) for i in range(12)]
    schema = make_dataset_schema(
        repeat_domain(size=2),
        mapped_domain_from_columns({
            "ax": np.asarray([float(r[0]) for r in rows]),
            "ay": np.asarray([float(r[1]) for r in rows]),
        }),
        cell_axes=(axis("site", values=[0.0, 1.0, 2.0], role=SITE),),
        dtype=np.float64,
    )
    rng = np.random.default_rng(revision)
    return make_snapshot(
        schema, rng.normal(size=(2, len(rows), 3)), revision=revision
    )

def test_a_frame_prepared_under_the_old_spec_is_refused() -> None:
    session = PlotSession(
        _snapshot(1),
        FacetGridPlot(AxisRef.cell_data("site"), CurvePlot(AxisRef.point("ax"))),
    )
    try:
        # A frame is prepared while the operator is still looking at the
        # grid they authored.
        prepared = session.prepare_live_frame(_snapshot(2)).result(timeout=20)

        # Then they hand x to another axis: a NEW spec, and with it a new
        # projection and payload.
        session.apply_semantic("fate:point:ay", "x")
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
        FacetGridPlot(AxisRef.cell_data("site"), CurvePlot(AxisRef.point("ax"))),
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

def test_a_frame_prepared_under_another_kind_is_refused_not_a_key_error() -> None:
    """The refusal comes before the old frame's display fields are read.

    A frame prepared as a Curve carries no ``bin_count``; asked about it
    before the spec check, the commit raised KeyError where the contract
    says "return None".
    """

    session = PlotSession(_snapshot(1), CurvePlot(AxisRef.point("ax")))
    try:
        prepared = session.prepare_live_frame(_snapshot(2)).result(timeout=20)
        session.replace_spec(HistogramPlot())
        assert session.commit_live_frame(prepared) is None
        assert session.spec == HistogramPlot()
        assert session.data_revision == 1
    finally:
        session.close()

def _run(generation: str, revision: int):
    rows = [(i % 4, i // 4) for i in range(12)]
    schema = make_dataset_schema(
        repeat_domain(size=2),
        mapped_domain_from_columns({
            "ax": np.asarray([float(r[0]) for r in rows]),
            "ay": np.asarray([float(r[1]) for r in rows]),
        }),
        dtype=np.float64,
    )
    return make_snapshot(
        schema,
        np.full((2, len(rows)), float(revision)),
        revision=revision,
        generation=generation,
    )

def test_a_frame_prepared_over_the_previous_run_does_not_land_on_the_next() -> None:
    """The base a frame stands on is a generation AND a revision.

    A new run restarts revisions, so run A's frame prepared over A@1 read
    B@1 as the same base and, being of another run, skipped the monotonic
    guard too: the panel went back to run A.  A prepared frame names the
    run it was prepared over, and a frame over another run is stale.
    """

    session = PlotSession(_run("run-a", 1), CurvePlot(AxisRef.point("ax")))
    try:
        prepared = session.prepare_live_frame(_run("run-a", 2)).result(timeout=20)
        session.update_data(_run("run-b", 1))
        assert session.commit_live_frame(prepared) is None
        assert session.data_generation == "run-b"
        assert session.data_revision == 1

        # The same frame over its OWN base still commits.
        prepared = session.prepare_live_frame(_run("run-b", 2)).result(timeout=20)
        finalization = session.commit_live_frame(prepared)
        assert finalization is not None
        session.publish_live_frame(finalization)
        assert session.data_revision == 2
    finally:
        session.close()
