"""Frame survival: every forward pair, denominator in the validity.

The processor consumes judged occupancy (cycles x frames x sites, bool)
and publishes one dataset whose LABELLED pair point axis carries the
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
    DomainSpec,
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
    repeat_domain = AxisSpec(AxisId("camera.cycle"), "cycle", REPEAT, cycles)
    frame_axis = AxisSpec(
        AxisId("camera.frames.frame"),
        "frame",
        READOUT_EVENT,
        frames,
        tuple(range(frames)),
    )
    schema = DatasetSchema(
        DomainSpec((cycles,), (repeat_domain,), (tuple(range(cycles)),)),
        DomainSpec((frames,), (frame_axis,), (tuple(range(frames)),)),
        DomainSpec((site_axis.size,), (site_axis,)),
        ValueSchema(
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
    assert values.shape == (40, 3, 6)  # (cycles, pairs, sites)
    for entry, (condition, value) in enumerate(_forward_pairs(3)):
        eligible = occupied[:, condition, :]
        np.testing.assert_array_equal(validity[:, entry, :], eligible)
        np.testing.assert_array_equal(
            values[:, entry, :][eligible], occupied[:, value, :][eligible]
        )
        # Outside the denominator there was no trial, and the validity is
        # what says so.  The value carries the occupancy verdict's own
        # dtype, so "no trial" cannot be spelled in it a second time.
        assert values.dtype == np.dtype("?")
        assert not values[:, entry, :][~eligible].any()


def test_unjudgeable_frames_leave_the_denominator() -> None:
    occupied = np.ones((4, 2, 3), dtype=bool)
    valid = np.ones_like(occupied)
    valid[1, 0, :] = False  # condition frame unjudgeable in cycle 1
    valid[2, 1, 0] = False  # value frame unjudgeable at one site of cycle 2
    survival = FrameSurvivalProcessor()._pair(
        _occupied_snapshot(occupied, valid)
    )
    validity = np.asarray(survival.expanded_validity())[:, 0, :]
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
            values[:, entry, :][validity[:, entry, :]]
        )
        np.testing.assert_allclose(projected, pooled, rtol=1e-12)


def test_pair_axis_carries_one_label_per_pair() -> None:
    occupied = np.zeros((2, 3, 2), dtype=bool)
    survival = FrameSurvivalProcessor(producer="fs")._pair(
        _occupied_snapshot(occupied)
    )
    schema = survival.block.schema
    assert schema.point_domain.size == 3
    (pair_axis,) = schema.point_domain.axes
    assert pair_axis.axis_id == AxisId("fs.pair")
    assert pair_axis.role == READOUT_EVENT
    assert pair_axis.coordinates == (0, 1, 2)
    assert pair_axis.coordinate_labels == ("0-1", "0-2", "1-2")
    (site_axis,) = schema.cell_domain.axes
    assert site_axis.axis_id == AxisId("occupancy.site")

    # A frame axis may carry names as its typed coordinates; the labels then
    # carry those names rather than a number format the name cannot take.
    source = _occupied_snapshot(np.zeros((1, 2, 2), dtype=bool)).block.schema
    named = replace(
        source,
        point_domain=DomainSpec(
            (2,),
            (replace(source.point_domain.axes[0], coordinates=("before", "after")),),
            ((0, 1),),
        ),
    )
    labelled = FrameSurvivalProcessor(producer="fs")._pair(
        owned_snapshot_from_arrays(named, np.zeros((1, 2, 2), dtype=bool), 0)
    )
    assert labelled.block.schema.point_domain.axes[0].coordinate_labels == (
        "before-after",
    )
    from zlc_atom.nodes.scan.dataset import scan_dataset_schema

    value_schema = labelled.block.schema.value_schema
    assert value_schema.name == "survival"
    scanned = scan_dataset_schema(
        labelled.block.schema, ((1.0,), (2.0,)), (("power", "mW"),),
        run_repeats=3,
    )
    assert scanned.value_schema is value_schema
    assert scanned.physical_shape == (3, 2, 2)


def test_single_frame_and_wrong_shapes_are_refused() -> None:
    processor = FrameSurvivalProcessor()
    with pytest.raises(ValueError, match="at least two frames"):
        processor._pair(_occupied_snapshot(np.zeros((3, 1, 2), dtype=bool)))
    non_boolean = _occupied_snapshot(np.zeros((3, 2, 2), dtype=bool))
    float_schema = non_boolean.block.schema
    float_snapshot = owned_snapshot_from_arrays(
        DatasetSchema(
            float_schema.repeat_domain,
            float_schema.point_domain,
            float_schema.cell_domain,
            ValueSchema(
                float_schema.value_schema.validity_contract,
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
    y_axis = AxisSpec(AxisId("cam.y"), "y", SPATIAL_Y, 4)
    x_axis = AxisSpec(AxisId("cam.x"), "x", SPATIAL_X, 5)
    pixel_schema = DatasetSchema(
        schema.repeat_domain,
        schema.point_domain,
        DomainSpec((y_axis.size, x_axis.size), (y_axis, x_axis)),
        ValueSchema(
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
    assert survival.coverage == DatasetCoverage(2 * 3, 5 * 3)  # cycles x pairs
    assert survival.cell_origin == (1, 0)
    assert survival.canonical_schema.point_domain.size == 3
    assert survival.canonical_schema.repeat_domain.size == 5


@pytest.mark.parametrize("frames", (2, 3, 4))
def test_scan_pairing_preserves_coordinates_and_live_terminal_placement(frames, monkeypatch) -> None:
    import zlc_atom.nodes.frame_survival.processor as processor_module
    from types import SimpleNamespace
    from zlc_atom.nodes.scan import SCAN_OUTPUT, ScanDatasetWriter
    from zlc_atom.nodes.frame_survival import SURVIVAL_OUTPUTS
    from zlc_runtime import SignalDataPlane

    rng = np.random.default_rng(17)
    occupied = rng.random((2, 4, frames, 3)) < 0.65
    valid = rng.random(occupied.shape) < 0.85
    base = _occupied_snapshot(occupied[0, 0][None]).block.schema
    frame_axis = replace(base.point_domain.axes[0], name="probe window")
    # Physical rows need not be in frame-coordinate order.
    frame_codes = tuple(reversed(range(frames)))
    base = replace(base, point_domain=DomainSpec((frames,), (frame_axis,), (frame_codes,)))
    writer = ScanDatasetWriter(
        ((8, 3), (8, 7), (2, 3), (2, 7)),
        (("frame", "V"), ("other", "Hz")), run_repeats=2,
    )
    processor = FrameSurvivalProcessor(producer="survival")
    layouts = []
    original_rows = processor_module._frame_rows
    def tracked_rows(schema, axis):
        layouts.append(schema)
        return original_rows(schema, axis)
    monkeypatch.setattr(processor_module, '_frame_rows', tracked_rows)
    scan = SimpleNamespace(instance_id="scan", dataset_output_declarations=(SCAN_OUTPUT,),
                           signal_key=lambda name: f"@logic/scan/{name}")
    result = SimpleNamespace(instance_id="survival", dataset_output_declarations=SURVIVAL_OUTPUTS,
                             signal_key=lambda name: f"@logic/survival/{name}")
    plane = SignalDataPlane()
    pair_count = frames * (frames - 1) // 2
    expected = np.zeros((2, 4, pair_count, 3), dtype=bool)
    eligible = np.zeros_like(expected)
    try:
        plane.begin_generation(scan)
        plane.begin_generation(result)
        for repeat in range(2):
            for point in range(4):
                snapshot = owned_snapshot_from_arrays(
                    base, occupied[repeat, point][None], repeat * 4 + point,
                    validity=valid[repeat, point][None],
                )
                event = writer.write(SignalValue("occupied", snapshot, None),
                                     row=point, scan_repeat=0, run_repeat=repeat)
                plane.commit_live(scan, {"scan": event})
                source_publication = plane.latest_publication("@logic/scan/scan")
                output = processor.evaluate(source_publication.value("@logic/scan/scan"))["survival"]
                assert output.cell_origin == (repeat, point * pair_count)
                assert output.coverage == DatasetCoverage(
                    (repeat * 4 + point + 1) * pair_count, 8 * pair_count,
                )
                plane.commit_live(result, {"survival": output})
                entry = 0
                for earlier in range(frames):
                    for later in range(earlier + 1, frames):
                        before, after = frame_codes.index(earlier), frame_codes.index(later)
                        trial = (occupied[repeat, point, before]
                                 & valid[repeat, point, before] & valid[repeat, point, after])
                        eligible[repeat, point, entry] = trial
                        expected[repeat, point, entry] = trial & occupied[repeat, point, after]
                        entry += 1
                live = plane.current_dataset("@logic/survival/survival")
                np.testing.assert_array_equal(live.block.values, expected.reshape(2, -1, 3))
                np.testing.assert_array_equal(live.expanded_validity(), eligible.reshape(2, -1, 3))
        assert len(layouts) == 2, 'unchanged event/canonical geometry was rebuilt per shot'
        plane.seal_committed(scan)
        source = plane.current_dataset("@logic/scan/scan")
        terminal = processor.evaluate(SignalValue("scan", source, None))["survival"]
        assert len(layouts) == 3, 'terminal input must plan its actual complete geometry'
        assert terminal.snapshot.block.schema == live.block.schema
        assert terminal.coverage == DatasetCoverage(8 * pair_count, 8 * pair_count)
        np.testing.assert_array_equal(terminal.snapshot.block.values, live.block.values)
        np.testing.assert_array_equal(terminal.snapshot.expanded_validity(), live.expanded_validity())
        output_domain = live.block.schema.point_domain
        assert output_domain.axes[1:] == source.block.schema.point_domain.axes[1:]
        assert output_domain.axis_codes[1:] == tuple(
            tuple(code for code in codes[::frames] for _ in range(pair_count))
            for codes in source.block.schema.point_domain.axis_codes[1:]
        )
    finally:
        plane.close()


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
    assert survival.coverage == DatasetCoverage(8, 8)  # cycles x one pair
    values = np.asarray(survival.snapshot.block.values)
    assert values.shape == (8, 1, 3)  # (cycles, 1 pair, sites)


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
        AxisRef.point("frame_survival.pair"), uncertainty=True
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
    output's geometry (one pair row per cycle) or the runtime refuses it."""

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
    assert survival.coverage.total_cells == 4 * 3  # cycles x pair rows
    assert survival.coverage.written_cells == 4 * 3
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
        program = build_calibration_pulse(sequencer)
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
        sequencer.fire(run_repeats=1, scan_repeats=1)
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
        assert values.shape == (cycles, pairs, 1)
        schema = survival_value.snapshot.block.schema
        pair_axis = next(
            axis
            for axis in schema.point_domain.axes
            if axis.role == READOUT_EVENT
        )
        assert pair_axis.coordinate_labels == ("0-1", "0-2", "1-2")
        coverage = survival_value.coverage
        assert coverage.total_cells == cycles * pairs

        # The declared source-index history is what lets a rolling panel
        # replay every retained shot when its projection changes: lease it,
        # fire another cycle, and the survival dataset must carry one
        # primary-indexed row per shot.
        from zlc_data.snapshot_projection import PRIMARY_INDEX_AXIS_ID

        survival_key = survival_host.signal_key("survival")
        assert plane.supports_indexed_history(survival_key)
        lease = plane.acquire_indexed_history(survival_key, 16)
        try:
            sequencer.fire(run_repeats=1, scan_repeats=1)
            sequencer.wait_done(1.0)

            def _indexed():
                publication = plane.latest_publication(survival_key)
                if publication is None:
                    return None
                snapshot = plane.current_dataset(survival_key, publication)
                axes = snapshot.block.schema.point_domain.axes
                has_index = any(
                    axis.axis_id == PRIMARY_INDEX_AXIS_ID
                    for axis in axes
                )
                rows = snapshot.block.schema.point_domain.size
                return snapshot if has_index and rows >= 2 else None

            def _diagnose():
                publication = plane.latest_publication(survival_key)
                if publication is None:
                    return "no publication"
                snapshot = plane.current_dataset(survival_key, publication)
                axes = tuple(
                    str(axis.axis_id)
                    for axis in snapshot.block.schema.point_domain.axes
                )
                def _seq(name):
                    pub = plane.latest_publication(name)
                    return None if pub is None else pub.event_ref.sequence

                return (
                    f"seq={publication.event_ref.sequence} "
                    f"rows={snapshot.block.schema.point_domain.size} "
                    f"axes={axes} "
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
            assert indexed.block.schema.point_domain.size >= 2
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
