"""A drag on a LIVE panel keeps drawing while the data keeps arriving.

A rolling trace's shot axis grows with every shot and a live curve's
limits follow its data, so on those panels the axes move under a gesture
constantly -- through no act of the operator's.  The widget used to read
any axis move as "the surface changed under this gesture" and drop the
front, so from the press until the release the picture froze: the box the
operator was dragging stopped following the pointer, and what they saw
was a selection that would not open.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable
from zlc_plot import AxisRef, CurvePlot, Qt5PlotWidget, RollingPlot
from zlc_plot.raster import RasterPlotHost


def _schema(sites: int) -> DatasetSchema:
    return DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"site": np.arange(sites, dtype=np.int64)}),
        data_axes=(),
        dtype=np.float64,
        generation="live-drag",
    )


@pytest.mark.gui
@pytest.mark.parametrize(
    "spec_name", ("rolling", "curve"), ids=("rolling", "dynamic-curve")
)
def test_a_live_panel_keeps_presenting_while_a_selector_is_dragged(
    spec_name,
) -> None:
    from zlc_plot import ensure_qt5_application

    try:
        app = ensure_qt5_application([])
        from PyQt5 import QtCore, QtGui
    except Exception as error:  # pragma: no cover - environment dependent
        pytest.skip(f"Qt5 offscreen unavailable: {error}")

    sites = 6
    schema = _schema(sites)
    rng = np.random.default_rng(4)
    scale = [1.0]

    def shot(revision: int) -> DatasetSnapshot:
        # Growing values, so an autoscaled curve's limits MOVE with the data
        # exactly as a rolling trace's shot axis does.
        scale[0] *= 1.6
        return DatasetSnapshot(
            schema, rng.random((1, sites)) * scale[0], revision=revision
        )

    spec = RollingPlot() if spec_name == "rolling" else CurvePlot(
        AxisRef.point("site")
    )
    host = RasterPlotHost.from_plot(shot(1), spec)
    widget = Qt5PlotWidget(host)
    widget.show()

    def pump(seconds: float) -> None:
        deadline = time.perf_counter() + seconds
        while time.perf_counter() < deadline:
            app.processEvents()
            time.sleep(0.002)

    try:
        for revision in range(2, 8):
            host.update_data(shot(revision)).result(timeout=20)
        pump(0.5)
        front = widget.presented_front
        assert front is not None
        axis = front.interaction.axes[0]
        left, top, right, bottom = axis.bounds

        def at(fx: float, fy: float):
            return QtCore.QPointF(
                (left + fx * (right - left)) * widget.width(),
                (top + fy * (bottom - top)) * widget.height(),
            )

        def send(kind, point, button, buttons) -> None:
            app.sendEvent(
                widget,
                QtGui.QMouseEvent(
                    kind, point, button, buttons, QtCore.Qt.NoModifier
                ),
            )

        send(
            QtGui.QMouseEvent.MouseButtonPress,
            at(0.25, 0.25),
            QtCore.Qt.LeftButton,
            QtCore.Qt.LeftButton,
        )
        pump(0.3)
        pressed_revision = widget.presented_front.identity.data_revision

        # The bench keeps running while the button is down.
        for revision in range(8, 12):
            host.update_data(shot(revision)).result(timeout=20)
            send(
                QtGui.QMouseEvent.MouseMove,
                at(0.25 + 0.1 * (revision - 7), 0.25 + 0.1 * (revision - 7)),
                QtCore.Qt.NoButton,
                QtCore.Qt.LeftButton,
            )
            pump(0.25)

        shown = widget.presented_front.identity.data_revision
        assert shown > pressed_revision, (
            "the panel froze for the whole drag: every live front was dropped "
            f"(still showing revision {shown})"
        )
    finally:
        widget.close_adapter()
        host.close()


def _sliding_history(first_shot: int, rows: int = 5):
    """One bounded shot history: absolute shot numbers that SLIDE."""

    from zlc_data import (
        AxisId,
        AxisSpec,
        PointColumn,
        PRIMARY_INDEX,
        REPEAT,
        ValueSchema,
    )
    from zlc_data import DatasetSchema as RoleDatasetSchema
    from zlc_data import PointTable as RolePointTable
    from zlc_data.value import owned_snapshot_from_arrays

    column = PointColumn(
        AxisId("zlc_data.primary-index"), "source index", PRIMARY_INDEX,
        PointColumn.NUMERIC,
        tuple(float(first_shot + index) for index in range(rows)),
    )
    schema = RoleDatasetSchema(
        AxisSpec(AxisId("repeat"), "repeat", REPEAT, 1, (0,)),
        RolePointTable(rows, (column,)),
        None,
        ValueSchema.scalar(np.dtype("float64"), "1"),
    )
    return schema, column


@pytest.mark.gui
def test_a_sliding_shot_history_is_not_a_new_geometry_under_a_drag() -> None:
    """A rolling panel past its window renames its own coordinates.

    Its x IS the absolute shot number, so once the window is full every
    shot slides the whole axis forward -- and the schema's full fingerprint
    with it, because that name includes every coordinate value.  A source
    that also re-generations each publication therefore declared a NEW
    GEOMETRY on every shot, and a new geometry cancels the gesture: the
    area the operator was dragging open vanished the moment a shot landed.
    Sliding coordinates are not a new world; the axes, their roles and
    units and the shape are all exactly what they were.
    """

    from zlc_data.value import owned_snapshot_from_arrays
    from zlc_plot import RollingPlot, ensure_qt5_application

    try:
        app = ensure_qt5_application([])
        from PyQt5 import QtCore, QtGui
    except Exception as error:  # pragma: no cover - environment dependent
        pytest.skip(f"Qt5 offscreen unavailable: {error}")

    rng = np.random.default_rng(3)
    rows = 5

    def shot(first: int, revision: int):
        schema, _column = _sliding_history(first, rows)
        return owned_snapshot_from_arrays(
            schema=schema,
            values=rng.random((1, rows, 1)),
            revision=revision,
            # A new generation per publication, as a re-cut bounded history
            # hands one over.
            stream_generation=f"slide-{first}",
        )

    host = RasterPlotHost.from_plot(shot(0, 1), RollingPlot())
    host.set_parameter("window", rows).result(timeout=20)
    widget = Qt5PlotWidget(host)
    widget.resize(520, 380)
    widget.show()

    def pump(seconds: float) -> None:
        deadline = time.perf_counter() + seconds
        while time.perf_counter() < deadline:
            app.processEvents()
            time.sleep(0.002)

    try:
        pump(0.4)
        front = widget.presented_front
        assert front is not None
        axis = front.interaction.axes[0]
        opened = axis.x_limits
        left, top, right, bottom = axis.bounds

        def at(fx: float, fy: float):
            return QtCore.QPointF(
                (left + fx * (right - left)) * widget.width(),
                (top + fy * (bottom - top)) * widget.height(),
            )

        def send(kind, point, button, buttons) -> None:
            app.sendEvent(
                widget,
                QtGui.QMouseEvent(
                    kind, point, button, buttons, QtCore.Qt.NoModifier
                ),
            )

        send(
            QtGui.QMouseEvent.MouseButtonPress, at(0.2, 0.2),
            QtCore.Qt.LeftButton, QtCore.Qt.LeftButton,
        )
        pump(0.3)
        assert widget._candidate is not None, "the press opened no selector"

        for step in range(4):
            host.update_data(shot(step + 1, step + 2)).result(timeout=20)
            send(
                QtGui.QMouseEvent.MouseMove,
                at(0.2 + 0.12 * (step + 1), 0.2 + 0.12 * (step + 1)),
                QtCore.Qt.NoButton, QtCore.Qt.LeftButton,
            )
            pump(0.25)
            assert widget._candidate is not None, (
                f"the drag was cancelled by shot {step + 1}: a sliding shot "
                "history was read as a new geometry"
            )

        # the window really did slide while the drag was held
        assert widget.presented_front.interaction.axes[0].x_limits != opened
        send(
            QtGui.QMouseEvent.MouseButtonRelease, at(0.85, 0.85),
            QtCore.Qt.LeftButton, QtCore.Qt.NoButton,
        )
        pump(0.4)
        committed = host.selectors().result(timeout=20).value
        assert [item.kind.value for item in committed] == ["area"], committed
    finally:
        widget.close_adapter()
        host.close()
