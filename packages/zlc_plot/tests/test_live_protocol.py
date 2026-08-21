from __future__ import annotations

from threading import Event, Thread

import numpy as np
import pytest

from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable
from zlc_plot import AxisRef, CurvePlot, ImagePlot, PlotSession
from zlc_plot.fit import FitCancelled


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


def _fit_snapshot(model_id: str, revision: int) -> tuple[DatasetSnapshot, object]:
    generation = f"live-fit-order-{model_id}"
    if model_id == "radial_gaussian_center":
        x = np.linspace(-2.0, 2.0, 25)
        y = np.linspace(-2.0, 2.0, 27)
        schema = DatasetSchema.create(
            Axis.create("repeat", size=1),
            PointTable.from_columns({"sample": [0.0]}),
            data_axes=(
                Axis.create("x", values=x, canonical_unit="m"),
                Axis.create("y", values=y, canonical_unit="m"),
            ),
            dtype=np.float64,
            generation=generation,
        )
        xx, yy = np.meshgrid(x, y)
        values = 0.2 + (2.0 + 0.01 * revision) * np.exp(
            -2.0 * ((xx - 0.25) ** 2 + (yy + 0.35) ** 2) / 0.9**2
        )
        return (
            DatasetSnapshot(schema, values.T[None, None], revision=revision),
            ImagePlot(AxisRef.data("x"), AxisRef.data("y")),
        )

    x = np.linspace(0.0, 8.0, 101)
    if model_id == "lorentzian":
        x = x - 4.0
        values = 0.2 + 2.0 / (1.0 + (2.0 * (x - 0.3) / 1.1) ** 2)
    else:
        values = 0.1 + 2.0 * np.exp(-x / 2.2)
    values = values + 0.001 * revision
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"x": x}),
        dtype=np.float64,
        generation=generation,
    )
    return DatasetSnapshot(schema, values[None], revision=revision), CurvePlot(
        AxisRef.point("x")
    )


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
        assert session.data_revision == 1
    finally:
        session.close()


@pytest.mark.parametrize(
    "model_id",
    ("radial_gaussian_center", "lorentzian", "exponential_decay"),
)
def test_live_fit_event_precedes_blocked_main_render_and_pair_stays_atomic(
    monkeypatch, model_id: str
) -> None:
    initial, spec = _fit_snapshot(model_id, 0)
    updated, _same_spec = _fit_snapshot(model_id, 1)
    session = PlotSession(initial, spec)
    render_entered = Event()
    release_render = Event()
    outcome: list[object] = []
    events: list[object | None] = []
    release_subscription = None
    thread = None
    try:
        model = next(item for item in session.fit_models if item.model_id == model_id)
        assert session.fit(model, live=True).success
        release_subscription = session.subscribe_fit(events.append)
        resolved_fit = session._resolved_accepted_fit
        resolved_revisions: list[int] = []

        def count_resolved(*args, **kwargs):
            resolved_revisions.append(int(args[0].source_revision))
            return resolved_fit(*args, **kwargs)

        monkeypatch.setattr(session, "_resolved_accepted_fit", count_resolved)

        prepared = session.prepare_live_frame(updated).result(timeout=10)
        solve = session.solve_live_frame(prepared)
        assert solve is not None
        solved = solve.result(timeout=30)
        old_fit = session.last_fit
        assert old_fit is not None and old_fit.source_revision == 0

        present = session._present_projection_transaction

        def blocked_present(*args, **kwargs):
            render_entered.set()
            assert release_render.wait(10)
            return present(*args, **kwargs)

        monkeypatch.setattr(session, "_present_projection_transaction", blocked_present)

        def commit() -> None:
            try:
                outcome.append(session.commit_live_frame(prepared, solved))
            except BaseException as error:
                outcome.append(error)

        thread = Thread(target=commit)
        thread.start()
        assert render_entered.wait(10)
        assert len(events) == 1
        assert events[0].result.source_revision == 1
        assert session.data_revision == 0
        assert session.last_fit is old_fit

        release_render.set()
        thread.join(10)
        assert not thread.is_alive()
        assert len(outcome) == 1 and not isinstance(outcome[0], BaseException)
        finalization = outcome[0]
        assert finalization is not None
        assert session.data_revision == 1
        assert session.last_fit is not None
        assert session.last_fit.source_revision == session.data_revision
        session.publish_live_frame(finalization)
        assert len(events) == 1, "front promotion notified the solved fit twice"
        assert resolved_revisions == [1]
        third, _third_spec = _fit_snapshot(model_id, 2)
        session.update_data(third)
        assert [event.result.source_revision for event in events] == [1, 2]
        assert resolved_revisions == [1, 2]
        fourth, _fourth_spec = _fit_snapshot(model_id, 3)
        prepared = session.prepare_live_frame(fourth).result(timeout=10)
        solve = session.solve_live_frame(prepared)
        assert solve is not None
        solved = solve.result(timeout=30)
        session.clear_fit()
        with pytest.raises(FitCancelled, match="request changed"):
            session.commit_live_frame(prepared, solved)
        assert events[-1] is None
    finally:
        release_render.set()
        if thread is not None:
            thread.join(10)
        if release_subscription is not None:
            release_subscription()
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
