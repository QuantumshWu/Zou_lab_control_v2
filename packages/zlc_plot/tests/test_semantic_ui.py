from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from zlc_plot import (
    AxisRef,
    CurvePlot,
    FacetGridPlot,
    ImagePlot,
    PlotSession,
    Qt5ParameterPanel,
    ensure_qt5_application,
)
from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable


def _session() -> PlotSession:
    schema = DatasetSchema.create(
        Axis.create("repeat", size=2, label="Repeat"),
        PointTable.from_columns(
            {"scan": np.linspace(0.0, 1.0, 4)},
            units={"scan": "V"},
            display_units={"scan": "mV"},
            labels={"scan": "Scan"},
        ),
        dtype=np.float64,
        generation="semantic-ui-tests",
    )
    snapshot = DatasetSnapshot(schema, np.arange(8.0).reshape(2, 4), revision=0)
    return PlotSession(snapshot, CurvePlot(AxisRef.point("scan")))


@pytest.mark.gui
def test_offscreen_semantic_combos_use_unique_labeled_choices() -> None:
    try:
        ensure_qt5_application([])
    except Exception as error:  # pragma: no cover - environment-dependent
        pytest.skip(f"Qt5 offscreen unavailable: {error}")
    session = _session()
    panel = None
    try:
        panel = Qt5ParameterPanel(session.describe_display())
        for name in panel.semantic_names:
            editor = panel.semantic_editor(name)
            if not hasattr(editor, "count"):
                continue
            labels = tuple(editor.itemText(index) for index in range(editor.count()))
            assert len(labels) == len(set(labels))
            assert all("AxisRef(" not in label for label in labels)
        # One row per axis: the row is NAMED after the axis and its options
        # are what can become of it, the first being what it already is when
        # nobody gave it a role.
        scan = panel.semantic_editor("fate:scan")
        assert scan.itemText(0) == "(reduced)"
        assert "X axis" in tuple(scan.itemText(index) for index in range(scan.count()))
    finally:
        if panel is not None:
            panel.deleteLater()
        session.close()


@pytest.mark.gui
def test_offscreen_semantic_kind_combo_offers_only_usable_kinds() -> None:
    """The kind combo lists exactly the kinds that can be switched to.

    An editor never shows an option that cannot be used.  A scalar series has
    no non-repeat facet axis left, so FacetGrid is not offered merely because
    repeats exist.  An image likewise cannot be drawn from a scalar point
    table with no second dimension.
    """

    try:
        ensure_qt5_application([])
    except Exception as error:  # pragma: no cover - environment-dependent
        pytest.skip(f"Qt5 offscreen unavailable: {error}")

    session = _session()
    panel = None
    try:
        panel = Qt5ParameterPanel(session.describe_display())
        editor = panel.semantic_editor("kind")
        assert editor.findData(FacetGridPlot.kind) < 0
    finally:
        if panel is not None:
            panel.deleteLater()
        session.close()

    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"scan": np.linspace(0.0, 1.0, 4)}),
        dtype=np.float64,
        generation="semantic-ui-no-facet-default",
    )
    snapshot = DatasetSnapshot(schema, np.arange(4.0).reshape(1, 4), revision=0)
    session = PlotSession(snapshot, CurvePlot(AxisRef.point("scan")))
    panel = None
    try:
        panel = Qt5ParameterPanel(session.describe_display())
        editor = panel.semantic_editor("kind")
        assert editor.findData(FacetGridPlot.kind) < 0
        assert editor.findData(ImagePlot.kind) < 0
    finally:
        if panel is not None:
            panel.deleteLater()
        session.close()
