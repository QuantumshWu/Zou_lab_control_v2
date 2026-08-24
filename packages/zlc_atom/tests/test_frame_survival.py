"""Frame survival: every forward pair, denominator in the validity.

The processor consumes judged occupancy (cycles x frames x sites, bool)
and publishes one dataset whose LABELLED pair cell axis carries the
forward frame pairs ("0-1", "0-2", "1-2"), one identity per pair.  The
pinned identity is the one the design stands on: a MEAN over the
published validity equals the pooled survival fraction computed from the
raw pool -- averaging the dataset IS pooling the data.
"""

from __future__ import annotations

import numpy as np
import pytest
from zlc_data import (
    AxisId,
    AxisSpec,
    DatasetSchema,
    PointColumn,
    PointTable,
    READOUT_EVENT,
    REPEAT,
    SITE,
    ValidityContract,
    ValueSchema,
    owned_snapshot_from_arrays,
)
from zlc_runtime import DatasetCoverage, SignalValue

from zlc_atom.nodes.frame_survival import FrameSurvivalProcessor
from zlc_atom.nodes.frame_survival.processor import _forward_pairs


def _occupied_snapshot(
    occupied: np.ndarray,
    valid: np.ndarray | None = None,
    *,
    revision: int = 0,
):
    cycles, frames, sites = occupied.shape
    site_axis = AxisSpec(AxisId("occupancy.site"), "site", SITE, sites)
    schema = DatasetSchema(
        AxisSpec(AxisId("camera.cycle"), "cycle", REPEAT, cycles),
        PointTable(
            frames,
            (
                PointColumn(
                    AxisId("camera.frames.frame"),
                    "frame",
                    READOUT_EVENT,
                    PointColumn.NUMERIC,
                    tuple(range(frames)),
                ),
            ),
        ),
        None,
        ValueSchema(
            (site_axis,),
            ValidityContract.components(site_axis.axis_id),
            np.dtype("?"),
            "1",
        ),
    )
    return owned_snapshot_from_arrays(
        schema,
        occupied,
        revision,
        validity=np.ones_like(occupied, dtype=bool) if valid is None else valid,
    )


def test_forward_pairs_enumerate_every_combination() -> None:
    assert _forward_pairs(2) == ((0, 1),)
    assert _forward_pairs(3) == ((0, 1), (0, 2), (1, 2))
    assert _forward_pairs(4) == (
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
    )


def test_pairing_identity_per_entry() -> None:
    rng = np.random.default_rng(3)
    occupied = rng.random((40, 3, 6)) < 0.5
    processor = FrameSurvivalProcessor()
    survival = processor._pair(_occupied_snapshot(occupied))
    values = np.asarray(survival.block.values)
    validity = np.asarray(survival.expanded_validity())
    assert values.shape == (40, 1, 3, 6)  # (cycles, 1 point, pairs, sites)
    for entry, (condition, value) in enumerate(_forward_pairs(3)):
        eligible = occupied[:, condition, :]
        np.testing.assert_array_equal(validity[:, 0, entry, :], eligible)
        np.testing.assert_array_equal(
            values[:, 0, entry, :][eligible], occupied[:, value, :][eligible]
        )
        assert np.all(np.isnan(values[:, 0, entry, :][~eligible]))


def test_unjudgeable_frames_leave_the_denominator() -> None:
    occupied = np.ones((4, 2, 3), dtype=bool)
    valid = np.ones_like(occupied)
    valid[1, 0, :] = False  # condition frame unjudgeable in cycle 1
    valid[2, 1, 0] = False  # value frame unjudgeable at one site of cycle 2
    survival = FrameSurvivalProcessor()._pair(
        _occupied_snapshot(occupied, valid)
    )
    validity = np.asarray(survival.expanded_validity())[:, 0, 0, :]
    assert not validity[1].any()
    assert not validity[2, 0]
    assert validity[0].all() and validity[3].all()


def test_mean_over_validity_is_the_pooled_survival() -> None:
    """The design's central identity: averaging the dataset IS pooling."""

    rng = np.random.default_rng(11)
    occupied = rng.random((60, 3, 5)) < 0.6
    survival = FrameSurvivalProcessor()._pair(_occupied_snapshot(occupied))
    values = np.asarray(survival.block.values)
    validity = np.asarray(survival.expanded_validity())
    for entry, (condition, value) in enumerate(_forward_pairs(3)):
        loaded = occupied[:, condition, :]
        pooled = (occupied[:, value, :] & loaded).sum() / loaded.sum()
        projected = np.nanmean(
            values[:, 0, entry, :][validity[:, 0, entry, :]]
        )
        np.testing.assert_allclose(projected, pooled, rtol=1e-12)


def test_pair_axis_carries_one_label_per_pair() -> None:
    occupied = np.zeros((2, 3, 2), dtype=bool)
    survival = FrameSurvivalProcessor(producer="fs")._pair(
        _occupied_snapshot(occupied)
    )
    schema = survival.block.schema
    assert schema.point_table.row_count == 1
    assert schema.point_table.columns == ()
    pair_axis, site_axis = schema.cell_schema.data_axes
    assert pair_axis.axis_id == AxisId("fs.pair")
    assert pair_axis.size == 3
    assert pair_axis.coordinate_labels == ("0-1", "0-2", "1-2")
    assert site_axis.axis_id == AxisId("occupancy.site")


def test_single_frame_and_wrong_shapes_are_refused() -> None:
    processor = FrameSurvivalProcessor()
    with pytest.raises(ValueError, match="at least two frames"):
        processor._pair(_occupied_snapshot(np.zeros((3, 1, 2), dtype=bool)))
    non_boolean = _occupied_snapshot(np.zeros((3, 2, 2), dtype=bool))
    float_schema = non_boolean.block.schema
    float_snapshot = owned_snapshot_from_arrays(
        DatasetSchema(
            float_schema.repeat_axis,
            float_schema.point_table,
            None,
            ValueSchema(
                float_schema.cell_schema.data_axes,
                float_schema.cell_schema.validity_contract,
                np.dtype("<f8"),
                "1",
            ),
        ),
        np.zeros((3, 2, 2)),
        0,
    )
    with pytest.raises(ValueError, match="occupied"):
        processor._pair(float_snapshot)


def test_connecting_judged_frames_names_the_right_signal() -> None:
    """The natural wrong pick -- frame_judged -- must say what to select."""

    from zlc_data import SPATIAL_X, SPATIAL_Y

    frames = _occupied_snapshot(np.zeros((2, 2, 3), dtype=bool))
    schema = frames.block.schema
    pixel_schema = DatasetSchema(
        schema.repeat_axis,
        schema.point_table,
        None,
        ValueSchema(
            (
                AxisSpec(AxisId("cam.y"), "y", SPATIAL_Y, 4),
                AxisSpec(AxisId("cam.x"), "x", SPATIAL_X, 5),
            ),
            ValidityContract.components(AxisId("cam.y")),
            np.dtype("<u2"),
            "1",
        ),
    )
    pixels = owned_snapshot_from_arrays(
        pixel_schema, np.zeros((2, 2, 4, 5), dtype=np.uint16), 0
    )
    with pytest.raises(ValueError, match="occupied.*frame_judged|frame_judged"):
        FrameSurvivalProcessor()._pair(pixels)


def test_evaluate_translates_exact_coverage_by_whole_cycles() -> None:
    # One fired cycle arrives as the event; the canonical run holds five.
    event = _occupied_snapshot(np.zeros((1, 3, 4), dtype=bool))
    canonical_source = _occupied_snapshot(np.zeros((5, 3, 4), dtype=bool))
    signal = SignalValue(
        "@logic/occupancy/occupied",
        event,
        DatasetCoverage(2 * 3, 5 * 3),  # 2 of 5 cycles written, 3 frames each
        canonical_schema=canonical_source.block.schema,
        cell_origin=(1, 0),
    )
    outputs = FrameSurvivalProcessor().evaluate(signal)
    survival = outputs["survival"]
    assert survival.coverage == DatasetCoverage(2, 5)  # one cell per cycle
    assert survival.cell_origin == (1, 0)
    assert survival.canonical_schema.point_table.row_count == 1
    assert survival.canonical_schema.repeat_axis.size == 5


def test_evaluate_refuses_partial_cycle_coverage() -> None:
    occupied = np.zeros((5, 3, 4), dtype=bool)
    snapshot = _occupied_snapshot(occupied)
    signal = SignalValue(
        "@logic/occupancy/occupied",
        _occupied_snapshot(np.zeros((1, 3, 4), dtype=bool)),
        DatasetCoverage(4, 15),  # not a whole number of cycles
        canonical_schema=snapshot.block.schema,
        cell_origin=(0, 0),
    )
    with pytest.raises(ValueError, match="whole cycles"):
        FrameSurvivalProcessor().evaluate(signal)


def test_terminal_dataset_evaluates_frozen() -> None:
    rng = np.random.default_rng(5)
    occupied = rng.random((8, 2, 3)) < 0.5
    snapshot = _occupied_snapshot(occupied)
    outputs = FrameSurvivalProcessor().evaluate(
        SignalValue("@logic/occupancy/occupied", snapshot, None)
    )
    survival = outputs["survival"]
    assert survival.coverage == DatasetCoverage(8, 8)  # one cell per cycle
    values = np.asarray(survival.snapshot.block.values)
    assert values.shape == (8, 1, 1, 3)  # (cycles, 1 point, 1 pair, sites)


def test_discovered_as_a_logic_node() -> None:
    from zlc_atom.nodes import discover_logic_nodes

    names = {descriptor.api_name for descriptor in discover_logic_nodes()}
    assert "frame_survival" in names


def test_plot_mean_projection_gives_pooled_rate_and_binomial_band() -> None:
    """End to end: the published dataset needs NOTHING downstream -- the
    plot's MEAN is the pooled rate and the uncertainty band is binomial."""

    from zlc_plot.data_view import DataView
    from zlc_plot.kinds import AxisRef

    rng = np.random.default_rng(7)
    occupied = rng.random((80, 3, 6)) < 0.55
    survival = FrameSurvivalProcessor()._pair(_occupied_snapshot(occupied))
    view = DataView(survival)
    series = view.curve(
        AxisRef.data("frame_survival.pair"), uncertainty=True
    ).series[0]
    assert len(series.y.canonical) == 3  # one plotted point per pair
    for entry, (condition, value) in enumerate(_forward_pairs(3)):
        loaded = occupied[:, condition, :]
        pooled = (occupied[:, value, :] & loaded).sum() / loaded.sum()
        np.testing.assert_allclose(
            float(series.y.canonical[entry]), pooled, rtol=1e-12
        )
        count = int(loaded.sum())
        binomial = np.sqrt(pooled * (1.0 - pooled) / (count - 1))
        np.testing.assert_allclose(
            float(series.sem[entry]), binomial, rtol=1e-12
        )
        assert int(series.counts[entry]) == count


def test_monitor_source_translates_coverage_to_own_geometry() -> None:
    """The real-bench failure: a camera-monitor chain hands MonitorCoverage
    counted in (cycles x frames); the published ledger must count THIS
    output's geometry (one row per cycle) or the runtime refuses it."""

    from zlc_runtime import MonitorCoverage

    occupied = np.zeros((4, 3, 2), dtype=bool)
    snapshot = _occupied_snapshot(occupied)
    outputs = FrameSurvivalProcessor().evaluate(
        SignalValue(
            "@logic/occupancy/occupied",
            snapshot,
            MonitorCoverage(4 * 3, 4 * 3),
        )
    )
    survival = outputs["survival"]
    assert isinstance(survival.coverage, MonitorCoverage)
    assert survival.coverage.total_cells == 4  # cycles x one point row
    assert survival.coverage.written_cells == 4
    # The constructor itself validates ledger-vs-geometry, so constructing
    # the LiveDatasetOutput above IS the regression proof.
