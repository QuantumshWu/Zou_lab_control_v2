"""Frame survival: every forward pair, denominator in the validity.

The processor consumes judged occupancy (cycles x frames x sites, bool)
and publishes one dataset whose LABELLED pair cell axis carries the
forward frame pairs ("0-1", "0-2", "1-2"), one identity per pair.  The
pinned identity is the one the design stands on: a MEAN over the
published validity equals the pooled survival fraction computed from the
raw pool -- averaging the dataset IS pooling the data.
"""

from __future__ import annotations

from dataclasses import replace

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
from zlc_atom.nodes.occupancy_agreement import OccupancyAgreementProcessor


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


def test_occupancy_agreement_filters_sampled_counts_and_allows_one_frame_noop() -> None:
    occupied_values = np.asarray(
        [[[False, True, False, True, True],
          [True, False, True, False, True],
          [False, True, True, False, True]]],
        dtype=bool,
    )
    occupied_valid = np.ones_like(occupied_values)
    occupied_valid[:, :, 4] = False
    occupied = _occupied_snapshot(occupied_values, occupied_valid)
    occupied_schema = occupied.block.schema
    counts_schema = replace(
        occupied_schema,
        cell_schema=ValueSchema(
            occupied_schema.cell_schema.data_axes,
            occupied_schema.cell_schema.validity_contract,
            np.dtype("<f8"),
            "count",
        ),
    )
    count_values = np.asarray(
        [[[1, 2, 3, 4, 5], [11, 22, 33, 44, 55], [6, 7, 8, 9, 10]]],
        dtype="<f8",
    )
    counts = owned_snapshot_from_arrays(
        counts_schema,
        count_values,
        occupied.block.revision,
        validity=np.ones_like(occupied_values),
        stream_generation=occupied.ref.stream_generation,
    )
    outputs = OccupancyAgreementProcessor().evaluate_inputs(
        {
            "counts": SignalValue("@logic/occupancy/counts", counts, None),
            "occupied": SignalValue("@logic/occupancy/occupied", occupied, None),
        }
    )
    consistent_occupied = outputs["consistent_occupied"].snapshot
    consistent_counts = outputs["consistent_counts"].snapshot
    np.testing.assert_array_equal(
        consistent_occupied.block.values[:, 0, :2], [[False, True]]
    )
    np.testing.assert_array_equal(
        consistent_occupied.expanded_validity(),
        [[[True, True, False, False, False]]],
    )
    np.testing.assert_array_equal(
        consistent_counts.expanded_validity(),
        [[[True, True, False, False, False]]],
    )
    np.testing.assert_array_equal(
        consistent_counts.block.values[:, 0, :2], [[11.0, 22.0]]
    )
    assert np.isnan(consistent_counts.block.values[:, 0, 2:]).all()
    assert consistent_counts.block.schema.point_table.columns[0].values == (1,)

    one_frame_occupied = _occupied_snapshot(occupied_values[:, :1, :], occupied_valid[:, :1, :])
    one_frame_counts = owned_snapshot_from_arrays(
        replace(
            one_frame_occupied.block.schema,
            cell_schema=counts_schema.cell_schema,
        ),
        count_values[:, :1, :],
        one_frame_occupied.block.revision,
        validity=np.ones_like(occupied_values[:, :1, :]),
        stream_generation=one_frame_occupied.ref.stream_generation,
    )
    no_op = OccupancyAgreementProcessor(
        first_occupancy_frame=0,
        counts_frame=0,
        second_occupancy_frame=0,
    ).evaluate_inputs(
        {
            "counts": SignalValue("@logic/occupancy/counts", one_frame_counts, None),
            "occupied": SignalValue("@logic/occupancy/occupied", one_frame_occupied, None),
        }
    )
    np.testing.assert_array_equal(
        no_op["consistent_counts"].snapshot.block.values[:, 0, :4],
        count_values[:, 0, :4],
    )
    np.testing.assert_array_equal(
        no_op["consistent_occupied"].snapshot.block.values[:, 0, :4],
        occupied_values[:, 0, :4],
    )


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


def test_live_monitor_chain_camera_occupancy_survival() -> None:
    """The bench chain that failed on first Start, end to end on product
    hosts: a live (repeat-zero) camera measurement feeds a hosted occupancy
    processor feeds a hosted frame-survival processor, all over the real
    signal plane.  This is the third delivery path -- monitor -- exercised
    for real instead of by hand-built SignalValues."""

    import time
    from threading import Event

    from zlc_runtime import NodeHost, SignalDataPlane

    from zlc_atom.install import create_installation
    from zlc_atom.nodes.calibration import (
        FrameContract,
        ReadoutModel,
        ReadoutModelKind,
        SiteMap,
        TrapCalibration,
    )
    from zlc_atom.nodes.camera_measurement import (
        CameraMeasurementNode,
        CameraMeasurementRequest,
    )
    from zlc_atom.nodes.camera_measurement.measurement import (
        CAMERA_FRAMES_OUTPUT,
    )
    from zlc_atom.nodes.occupancy import OccupancyProcessor
    from zlc_atom.nodes.occupancy.processor import OCCUPANCY_OUTPUTS
    from zlc_atom.nodes.frame_survival import (
        SURVIVAL_OUTPUTS,
        FrameSurvivalProcessor,
    )
    from tests.pulse_fixture import (
        CALIBRATION_FRAMES_PER_CYCLE,
        build_calibration_pulse,
    )

    plane = SignalDataPlane()
    installation = create_installation("virtual")
    hosts = []

    def _await(predicate, message, seconds=10.0):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            # freeze() drains the latest-processor lane -- the pump the
            # workbench runs; a test drives it explicitly.
            plane.freeze()
            for running in hosts:
                running.poll()
            found = predicate()
            if found is not None:
                return found
            time.sleep(0.005)
        raise AssertionError(message)

    try:
        camera = installation.capability("camera.adapter")
        sequencer = installation.device("sequencer")
        program = build_calibration_pulse(sequencer.describe())
        sequencer.load(program)

        node = CameraMeasurementNode(
            camera=camera,
            request=CameraMeasurementRequest(
                camera_key="camera",
                exposure_seconds=0.02,
                roi_xywh=None,
                repeat=0,
                frames_per_cycle=CALIBRATION_FRAMES_PER_CYCLE,
            ),
            signal_plane=plane,
            producer="cm-chain",
        )
        camera_host = NodeHost(
            node,
            plane,
            Event().set,
            instance_id=node.instance_id,
            kind="measurement",
            dataset_output_declarations=(CAMERA_FRAMES_OUTPUT,),
        )
        hosts.append(camera_host)
        camera_host.start()
        _await(
            lambda: True if camera.capture_state() else None,
            "camera never armed",
        )
        sequencer.fire()
        sequencer.wait_done(1.0)
        frames_key = camera_host.signal_key("frames")
        frames_value = _await(
            lambda: plane.freeze().value(frames_key),
            "camera published no live frames",
        )

        working = node.actual_working_point
        assert working is not None
        height, width = np.asarray(
            frames_value.snapshot.block.values
        ).shape[-2:]
        roi_y, roi_x = working.roi_origin_yx
        roi_height, roi_width = working.roi_shape_yx
        site_ids = ("site_0000",)
        calibration = TrapCalibration(
            SiteMap(
                site_ids,
                np.asarray([[width // 2, height // 2]], dtype=float),
                [True],
                [1.0],
            ),
            (ReadoutModel(site_ids, [0.0], [-1.0], [1.0], [True], [1.0]),),
            ReadoutModelKind.BOX,
            FrameContract(
                (height, width),
                sensor_shape=working.sensor_shape_yx,
                roi_xywh=(roi_x, roi_y, roi_width, roi_height),
                binning_yx=working.binning_yx,
                exposure_seconds=working.exposure_seconds,
                camera_id=node.camera_key,
                readout_mode=working.readout_mode,
            ),
        )

        occupancy = OccupancyProcessor(
            calibration, producer="occ-chain", source_signal=frames_key
        )
        occupancy_host = NodeHost(
            occupancy,
            plane,
            Event().set,
            instance_id=occupancy.instance_id,
            kind="processor",
            dataset_output_declarations=OCCUPANCY_OUTPUTS,
            input_signal=frames_key,
            input_delivery="latest",
        )
        hosts.append(occupancy_host)
        occupancy_host.start()
        occupied_key = occupancy_host.signal_key("occupied")
        _await(
            lambda: plane.freeze().value(occupied_key),
            "occupancy published no verdicts",
        )

        survival_node = FrameSurvivalProcessor(
            producer="fs-chain", source_signal=occupied_key
        )
        survival_host = NodeHost(
            survival_node,
            plane,
            Event().set,
            instance_id=survival_node.instance_id,
            kind="processor",
            dataset_output_declarations=SURVIVAL_OUTPUTS,
            input_signal=occupied_key,
            input_delivery="latest",
        )
        hosts.append(survival_host)
        survival_host.start()
        survival_value = _await(
            lambda: plane.freeze().value(survival_host.signal_key("survival")),
            "frame survival published nothing on the live chain",
        )

        values = np.asarray(survival_value.snapshot.block.values)
        cycles = values.shape[0]
        pairs = len(_forward_pairs(CALIBRATION_FRAMES_PER_CYCLE))
        assert values.shape == (cycles, 1, pairs, 1)
        schema = survival_value.snapshot.block.schema
        pair_axis = schema.cell_schema.data_axes[0]
        assert pair_axis.coordinate_labels == ("0-1", "0-2", "1-2")
        coverage = survival_value.coverage
        assert coverage.total_cells == cycles

        # The declared source-index history is what lets a rolling panel
        # replay every retained shot when its projection changes: lease it,
        # fire another cycle, and the survival dataset must carry one
        # primary-indexed row per shot.
        from zlc_data.snapshot_projection import PRIMARY_INDEX_AXIS_ID

        survival_key = survival_host.signal_key("survival")
        assert plane.supports_indexed_history(survival_key)
        lease = plane.acquire_indexed_history(survival_key, 16)
        try:
            sequencer.fire()
            sequencer.wait_done(1.0)

            def _indexed():
                publication = plane.latest_publication(survival_key)
                if publication is None:
                    return None
                snapshot = plane.current_dataset(survival_key, publication)
                columns = snapshot.block.schema.point_table.columns
                has_index = any(
                    column.coordinate_id == PRIMARY_INDEX_AXIS_ID
                    for column in columns
                )
                rows = snapshot.block.schema.point_table.row_count
                return snapshot if has_index and rows >= 2 else None

            def _diagnose():
                publication = plane.latest_publication(survival_key)
                if publication is None:
                    return "no publication"
                snapshot = plane.current_dataset(survival_key, publication)
                columns = tuple(
                    str(column.coordinate_id)
                    for column in snapshot.block.schema.point_table.columns
                )
                def _seq(name):
                    pub = plane.latest_publication(name)
                    return None if pub is None else pub.event_ref.sequence

                return (
                    f"seq={publication.event_ref.sequence} "
                    f"rows={snapshot.block.schema.point_table.row_count} "
                    f"columns={columns} "
                    f"chain: frames={_seq(frames_key)} "
                    f"occupied={_seq(occupied_key)} "
                    f"survival={_seq(survival_key)} "
                    f"camera_armed={camera.capture_state()}"
                )

            try:
                indexed = _await(
                    _indexed,
                    "leased survival never grew a primary-indexed history",
                )
            except AssertionError as error:
                raise AssertionError(f"{error}; state: {_diagnose()}") from None
            assert indexed.block.schema.point_table.row_count >= 2
        finally:
            lease.close()
    finally:
        for running in hosts:
            try:
                running.cancel("chain test done")
            except Exception:
                pass
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not all(
            running.observation.terminal for running in hosts
        ):
            for running in hosts:
                running.poll()
            time.sleep(0.005)
        for running in hosts:
            running.shutdown()
        installation.close()
        plane.close()
