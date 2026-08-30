from __future__ import annotations

import base64
from threading import Event, Thread
import zlib

import numpy as np
import pytest

from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable
from zlc_plot import AxisRef, CurvePlot, ImagePlot, PlotSession
from zlc_plot.fit import FitCancelled, FitEngine, RegularImageFitInput
from zlc_plot.selectors import NumericRange, SelectorKind


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


def test_unsuccessful_exact_retry_does_not_invalidate_public_fit_event() -> None:
    """A failed no-improvement retry cannot erase an exact solve's success."""

    # Frozen as bytes so this neutral Plot test does not import Atom simulation.
    encoded = (
        "c-k$LyKWOv5Qb;%TqLAWAW=a<0VO;D4RmxAJOd5l1?Z@G2%aR6#Llkw>Rs<T>$@GoMT&?71vGSg^V_n#d(NDhfBy"
        "Mr&K8ziXcOC5WEVEI#xlFKA<uJ*d6(D+_Qg{6sa5P2_6^UNRb~mcoLz#gVLbvV=HJ*EQA?0Rtf%&qlgtWEj)H?=2"
        "ySXM*k@Sg_8s2~?PK04F`3P<#jJa_ws(6^yGt+~!_RW^CFD=6#mDolStUeVaX#ezFSvj7a({_uX=9iVt<Sr_w_;b"
        "bi$L=_TX2`;0VlY7__(ve-czPyn6)6sQesYYqr4+fJx9X(#@+`D&|I66lNhWq_7O^E_(xh-FmxSS;zG1+@EPpXjZ"
        "wbBy0Fje3Q#wyEW_d%a#j_LE9u=Mwo-h8t&v|r_FPKT_;{5m=ozaK8ERon#c!_N05&aq@35=j7_RG;_)OHt6cz){"
        "MzHeUPVE}pKDJ!ixOV4s&pYqxj6v#YjxP8W($oJV)(cKk+Yjyp1^CX}xwbq0E08AYS}YYaWsT7B(q^64ow<07HJ2"
        "&xnkt-2jt-7rW9{#{JhfZFhr?Ihp{69V6Jl3VrPUOd;_CAr$riFs?Md))?^BTIJT>Ro65TR^B{>Rpb;i@#dMBpTe"
        "Pa)UzeEhNZt(N@y4H-$6+2K5JVU*!w-EA79nP65XW&y*u-;af;TiKR!JNX>Z-h@tPkr%zhKg~Na92l+y4z|qSglo"
        "6#e~J&xs~QPBrYcBgk6dEnv65W_&hGi(C98z(#WfBr0&LkVM|s2Z!sKl?z3A;+X)q~iBB~#jk<o^ebe2jBaJ2_rU"
        "SKoRks^iBchhnR$%w3c;2~p``N)CDVvF{Go5IPLn>-C^_lLRe;;yUA9P+EzU`cV>l5iG+Z*7wd_=|{R5}&Db3X+m"
        "X0{&%mF|ANQ*xx<H|lXj*XEqLo+Y|f;y>oQl&jY5H>rkOUvqFy2A^@iJw5}j(|=W_5h("
    )
    image = np.frombuffer(
        zlib.decompress(base64.b85decode(encoded)), dtype="<u2"
    ).reshape(26, 26)
    x = np.arange(51.0, 77.0)
    y = np.arange(35.0, 61.0)
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"sample": [0.0]}),
        data_axes=(Axis.create("x", values=x), Axis.create("y", values=y)),
        dtype=image.dtype,
        generation="full-resolution-success-retry",
    )
    snapshot = DatasetSnapshot(schema, image.T[None, None], revision=175)
    spec = ImagePlot(AxisRef.data("x"), AxisRef.data("y"))
    fit_input = RegularImageFitInput(x, y, image)
    expected = np.asarray(
        (2704.0, 226.6475384850741, 0.9993684055038514, 55.15224030304722,
         57.128628322987566)
    )
    reference = FitEngine().fit(
        "radial_gaussian_center", fit_input, initial=expected, data_revision=175
    )
    session = PlotSession(snapshot, spec)
    events = []
    release = session.subscribe_fit(events.append)
    try:
        session.set_area_selector(
            NumericRange(51.0, 76.0), NumericRange(35.0, 60.0), display=False
        )
        result = session.fit("radial_gaussian_center", live=False)
        assert reference.success and result.success
        assert len(events) == 1
        event = events[0]
        assert event.result.success and event.result.source_revision == 175
        assert event.selection.sample_count == 26 * 26
        assert event.selection.selector_kind is SelectorKind.AREA
        np.testing.assert_allclose(
            result.parameter_values,
            reference.parameter_values,
            rtol=5e-11,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            result.covariance,
            reference.covariance,
            rtol=2e-9,
            atol=1e-9,
        )
        assert result.reduced_chi_square == pytest.approx(
            reference.reduced_chi_square, rel=0.0, abs=1e-9
        )
    finally:
        release()
        session.close()


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
