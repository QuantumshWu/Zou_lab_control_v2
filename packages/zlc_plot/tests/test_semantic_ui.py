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
from data_factory import (
    make_dataset_schema,
    make_snapshot,
    mapped_domain_from_columns,
    repeat_domain,
)
from zlc_plot.semantics import fate_field_name

def _session() -> PlotSession:
    schema = make_dataset_schema(
        repeat_domain(size=2),
        mapped_domain_from_columns(
            {"scan": np.linspace(0.0, 1.0, 4)},
            units={"scan": "V"},
        ),
        dtype=np.float64,
    )
    snapshot = make_snapshot(schema, np.arange(8.0).reshape(2, 4), revision=0)
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
        scan = panel.semantic_editor(
            fate_field_name(AxisRef.point("scan"))
        )
        assert scan.itemText(0) == "(reduced)"
        assert "X axis" in tuple(scan.itemText(index) for index in range(scan.count()))
    finally:
        if panel is not None:
            panel.deleteLater()
        session.close()

@pytest.mark.gui
def test_offscreen_semantic_kind_combo_lists_every_kind_the_data_admits() -> None:
    """The kind combo lists every kind this dataset can host.

    Whether a kind has an OBVIOUS default is a different question from
    whether the operator may choose it: a grid the data does not
    obviously want is still one they may build, and the fate table is
    where they say what it faces.  What stays out is a kind the data
    cannot host at all -- an image still needs a second dimension.
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
        assert editor.findData(FacetGridPlot.kind) >= 0
    finally:
        if panel is not None:
            panel.deleteLater()
        session.close()

    schema = make_dataset_schema(
        repeat_domain(size=1),
        mapped_domain_from_columns({"scan": np.linspace(0.0, 1.0, 4)}),
        dtype=np.float64,
    )
    snapshot = make_snapshot(schema, np.arange(4.0).reshape(1, 4), revision=0)
    session = PlotSession(snapshot, CurvePlot(AxisRef.point("scan")))
    panel = None
    try:
        panel = Qt5ParameterPanel(session.describe_display())
        editor = panel.semantic_editor("kind")
        # No automatic grid exists for a scalar sweep with nothing to face
        # across, and the operator may still ask for one.
        assert editor.findData(FacetGridPlot.kind) >= 0
        # An image needs two dimensions; this dataset has one.
        assert editor.findData(ImagePlot.kind) < 0
    finally:
        if panel is not None:
            panel.deleteLater()
        session.close()
