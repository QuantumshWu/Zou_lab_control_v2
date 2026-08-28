"""A histogram is the distribution of every acquired value."""

from __future__ import annotations

import numpy as np

from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable
from zlc_plot import HistogramPlot, PlotSession
from zlc_plot.data_view import DataView


def _snapshot(revision: int = 0) -> DatasetSnapshot:
    schema = DatasetSchema.create(
        Axis.create("repeat", size=2),
        PointTable.from_columns({"point": [0.0, 1.0]}),
        data_axes=(Axis.create("scan", values=[0.0, 1.0]),),
        dtype=np.float64,
        generation="histogram-pool",
    )
    values = np.arange(8, dtype=np.float64).reshape(schema.shape)
    return DatasetSnapshot(schema, values, revision=revision)


def test_histogram_pools_the_whole_box() -> None:
    """Repeat x points x data axes all land in the pool: 8 values, 8 counts."""

    histogram = DataView(_snapshot()).histogram(bins=4)
    assert int(np.asarray(histogram.counts).sum()) == 8


def test_histogram_spec_needs_no_axis_declaration() -> None:
    """No axis takes a ROLE here -- and every axis still has a fate.

    A histogram pools whatever box it is given, so every axis row offers
    pooling and a coordinate to pin, and none of them offers a role.
    """

    spec = HistogramPlot()
    session = PlotSession(_snapshot(), spec)
    try:
        description = session.describe_semantics()
        names = tuple(field.name for field in description.fields)
        # The kind row, then one row per axis, then the reduction the
        # collapsed axes are read under.  What a histogram declares no
        # axis for is a ROLE: there is no x, y, group or facet to assign.
        assert names[0] == "kind" and names[-1] == "reduction"
        assert tuple(name for _axis, name in description.fate_rows) == names[1:-1]
        assert all(
            description.field(name).value == "pool"
            for _axis, name in description.fate_rows
        )
        assert description.axes_offering("x") == ()
    finally:
        session.close()


def test_a_nonindexed_window_never_invents_cross_publication_history() -> None:
    """Only a Runtime primary-index Dataset carries earlier publications."""

    session = PlotSession(_snapshot(), HistogramPlot())
    try:
        assert int(np.asarray(session._payload.counts).sum()) == 8

        session.set_parameter("window", 3)
        for revision in range(1, 8):
            session.update_data(_snapshot(revision=revision))
        payload = session._payload
        assert int(np.asarray(payload.counts).sum()) == 8

        session.set_parameter("window", 1)
        assert int(np.asarray(session._payload.counts).sum()) == 8
        session.set_parameter("window", 3)
        assert int(np.asarray(session._payload.counts).sum()) == 8
    finally:
        session.close()


def test_representation_toggles_refit_the_count_axis() -> None:
    """Density/cumulative/bin edits re-fit the axis; they are not jitter.

    The expand/shrink hysteresis exists for shot-to-shot noise on live
    data.  A representation change alters what one count MEANS, so the
    axis snaps to the new scale — and a density peak far below one count
    must fill the axis instead of being pinned under a counts floor.
    """

    rng = np.random.default_rng(3)
    schema = DatasetSchema.create(
        Axis.create("repeat", size=300),
        PointTable.from_columns({"x": np.arange(1.0)}),
        dtype=np.float64,
        generation="histogram-refit",
    )
    snapshot = DatasetSnapshot(schema, rng.normal(50, 8, (300, 1)), revision=0)
    session = PlotSession(snapshot, HistogramPlot())
    try:
        axes = session._renderer.primary_axes

        def ceiling() -> float:
            return float(axes.get_ylim()[1])

        counts_ceiling = ceiling()
        assert counts_ceiling > 10.0

        session.set_parameter("density", True)
        density_ceiling = ceiling()
        assert density_ceiling < 1.0  # fills the axis, no counts floor
        assert density_ceiling > 0.0

        session.set_parameter("density", False)
        assert ceiling() == counts_ceiling

        session.set_parameter("cumulative", True)
        assert abs(ceiling() - 300 * 1.08) < 1e-6

        session.set_parameter("cumulative", False)
        assert ceiling() == counts_ceiling

        session.set_parameter("bin_count", 15)
        assert ceiling() > counts_ceiling  # fewer bins, taller peaks, refit
    finally:
        session.close()


def test_only_bin_edits_reproject_histogram_samples(monkeypatch) -> None:
    """Density/cumulative transform bins; only a bin edit reprojects samples."""

    schema = DatasetSchema.create(
        Axis.create("repeat", size=512),
        PointTable.from_columns({"point": [0.0]}),
        dtype=np.float64,
        generation="histogram-parameter-projection",
    )
    samples = np.linspace(-5.0, 7.0, 512, dtype=np.float64)[:, np.newaxis]
    session = PlotSession(
        DatasetSnapshot(schema, samples, revision=0),
        HistogramPlot(),
    )
    try:
        native_rebuild = session._rebuild_projection
        rebuilds = 0

        def counted_rebuild(*, payload_only: bool = False) -> None:
            nonlocal rebuilds
            rebuilds += 1
            native_rebuild(payload_only=payload_only)

        monkeypatch.setattr(session, "_rebuild_projection", counted_rebuild)
        projected = session._payload
        counts = np.asarray(projected.counts).copy()
        edges = np.asarray(projected.edges.display).copy()

        session.set_parameter("density", True)
        assert session._payload is projected
        session.set_parameter("cumulative", True)
        assert session._payload is projected
        assert rebuilds == 0
        np.testing.assert_array_equal(session._payload.counts, counts)
        np.testing.assert_array_equal(session._payload.edges.display, edges)

        for expected_rebuilds, bin_count in enumerate((32, 64, 128), start=1):
            previous = session._payload
            session.set_parameter("bin_count", bin_count)
            assert rebuilds == expected_rebuilds
            assert session._payload is not previous
            assert len(session._payload.counts) == bin_count
            assert len(session._payload.edges.display) == bin_count + 1
    finally:
        session.close()


def test_an_axis_may_be_collapsed_before_the_values_are_binned() -> None:
    """Pooling is the default fate of a histogram axis, not the only one.

    "The distribution of every shot" and "the distribution of each site's
    mean over shots" are two different measurements of the same data, and
    the fate table is where the operator says which one they are looking
    at: an axis pools, or it collapses under the reduction first.
    """

    import numpy as np

    from zlc_plot import AxisRef
    from zlc_plot.semantics import describe_semantics, fate_field_name, updated_spec

    rng = np.random.default_rng(3)
    per_site = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    values = per_site[None, None, :] + rng.normal(scale=0.02, size=(4, 1, 5))
    schema = DatasetSchema.create(
        Axis.create("repeat", size=4),
        PointTable.from_columns({"shot": [0.0]}),
        data_axes=(Axis.create("site", values=[0.0, 1.0, 2.0, 3.0, 4.0]),),
        dtype=np.float64,
    )
    session = PlotSession(
        DatasetSnapshot(schema, values, revision=1),
        HistogramPlot(),
        parameters={"bin_count": 12},
    )
    try:
        pooled = session._projection._payload
        assert int(np.sum(pooled.counts)) == 20  # four shots of five sites

        session.replace_spec(
            updated_spec(
                schema,
                session.spec,
                fate_field_name(AxisRef.repeat()),
                "reduce",
            )
        )
        collapsed = session._projection._payload
        assert int(np.sum(collapsed.counts)) == 5  # one mean per site

        description = describe_semantics(schema, session.spec)
        repeat_row = description.field(fate_field_name(AxisRef.repeat()))
        assert repeat_row.value == "reduce"
        assert "pool" in [value for value, _label in repeat_row.choices]
        assert session.rgba() is not None
    finally:
        session.close()


def _noisy_camera(revision: int, seed: int) -> DatasetSnapshot:
    """Integer counts with a jittering maximum -- a camera frame's shape."""

    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"shot": [0.0]}),
        data_axes=(
            Axis.create("y", values=[float(i) for i in range(40)]),
            Axis.create("x", values=[float(i) for i in range(50)]),
        ),
        dtype=np.uint16,
        generation="histogram-domain",
    )
    # Poisson counts and nothing else: the maximum of two thousand samples
    # wanders by a count or two between revisions, which is the jitter a
    # camera actually shows and the jitter the domain has to absorb.
    rng = np.random.default_rng(seed)
    values = rng.poisson(6.0, size=(1, 1, 40, 50)).astype(np.uint16)
    return DatasetSnapshot(schema, values, revision=revision)


def test_a_steady_value_axis_holds_its_bins_across_revisions() -> None:
    """The bins an operator is reading must not be re-cut every shot.

    A camera's maximum is the largest of a million noisy samples and moves
    by a count or two every revision.  Binning between the raw minimum and
    maximum therefore re-cut every bar under the operator, and moved the x
    limits with them -- and a moving limit marks the axes chrome dirty,
    which took the panel out of the composed-background path and into a
    full figure redraw: measured at 14.63 ms of wall and 11.27 of CPU per
    frame, against 3.57 and 1.40 once the domain holds.

    Two things had to be true for the retention that was already written to
    ever run.  It compared the bin count it PRODUCED against the count that
    was REQUESTED, and integer-aligned bins produce fewer than requested, so
    on a camera the two never matched.  And it held the produced edges as
    the next revision's domain, though integer bins round a domain up to a
    whole number of them -- so each revision widened the span it was handed,
    and a value axis grew from 30 counts to 1200 in ninety frames.
    """

    session = PlotSession(_noisy_camera(1, 1), HistogramPlot())
    try:
        session.set_size("2x2")
        session.set_parameters({"bin_count": 60, "x_relim_mode": "normal"})
        session.rgba()

        # Let the domain absorb the first few maxima, then it must stand.
        for revision, seed in enumerate((2, 3, 4), start=2):
            session.update_data(_noisy_camera(revision, seed))
            session.rgba()
        held = session._renderer._artists["histogram:projection"][0]

        seen = []
        for revision, seed in enumerate((5, 6, 7, 8, 9, 10, 11, 12), start=5):
            session.update_data(_noisy_camera(revision, seed))
            session.rgba()
            seen.append(session._renderer._artists["histogram:projection"][0])
        assert all(np.array_equal(edges, held) for edges in seen), (
            "the bins were re-cut on %d of %d settled revisions"
            % (sum(1 for e in seen if not np.array_equal(e, held)), len(seen))
        )
        # And no ratchet: a domain that is handed back its own widened edges
        # grows every revision even when nothing breaches it.
        spans = [float(edges[-1] - edges[0]) for edges in seen]
        assert max(spans) == min(spans) == float(held[-1] - held[0])

        # Tight means tight here too: the operator who asks for the data's
        # own range every revision still gets it.
        session.set_parameters({"x_relim_mode": "tight"})
        session.rgba()
        moved = []
        for revision, seed in enumerate((13, 14, 15, 16), start=13):
            session.update_data(_noisy_camera(revision, seed))
            session.rgba()
            moved.append(
                tuple(session._renderer._artists["histogram:projection"][0][[0, -1]])
            )
        assert len(set(moved)) > 1, (
            "tight stopped following the data: %s" % (moved,)
        )
    finally:
        session.close()


def test_the_value_axis_mode_is_the_operator_s_own_control() -> None:
    """It reaches the editor, and it governs only its own limits.

    The value domain used to be retained (or not) by the COUNT axis's mode,
    so an operator asking for a steady count scale silently also asked for a
    steady value domain and could not ask for either alone.
    """

    from zlc_plot.specs import parameter_schema_for_kind
    from zlc_plot.style import build_plot_style
    from zlc_plot.ui import parameter_controls

    schema = parameter_schema_for_kind("histogram", style=build_plot_style())
    assert "x_relim_mode" in schema
    assert "x_min" in schema and "x_max" in schema

    values = {name: spec.default for name, spec in schema.items()}
    values["relim_mode"] = "fixed"
    values["y_min"], values["y_max"] = 0.0, 10.0
    controls = {control.name: control for control in
                parameter_controls(schema, values)}

    assert controls["x_relim_mode"].label == "Value limits"
    assert controls["x_relim_mode"].choices == controls["relim_mode"].choices
    # The count axis is fixed, so ITS limits are editable and the value
    # axis's are not.  One mode gating both was the editor's own copy of
    # the pairing; the vocabulary owns it now.
    assert controls["y_min"].unavailable_reason == ""
    assert controls["x_min"].unavailable_reason != ""

    values["x_relim_mode"] = "fixed"
    values["x_min"], values["x_max"] = 0.0, 5.0
    controls = {control.name: control for control in
                parameter_controls(schema, values)}
    assert controls["x_min"].unavailable_reason == ""
