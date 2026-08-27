from __future__ import annotations

from dataclasses import dataclass, replace
from threading import Event, Thread
import time

import numpy as np
import pytest

from zlc_data import (
    COMPONENT,
    READOUT_EVENT,
    SITE,
    REPEAT,
    SCAN_POINT,
    SPATIAL_X,
    SPATIAL_Y,
    AxisId,
    AxisSpec,
    BlockId,
    CellValidity,
    DataBlock,
    DatasetRevision,
    DatasetSchema,
    GridTopology,
    OwnedSnapshot,
    PointColumn,
    PointTable,
    StreamGenerationId,
    ValidityContract,
    ValueSchema,
)

from zlc_runtime.dataset import DatasetCoverage, MonitorCoverage
from zlc_runtime.dataset_output import (
    DatasetOutputDeclaration,
    LiveDatasetOutput,
)
from zlc_runtime.plane import SignalDataPlane
from zlc_runtime.selection_bridge import (
    FacetCondition,
    FitEventValue,
    SelectionBridge,
    SelectionChange,
    SelectionRange,
    SelectionState,
    selection_output_catalog,
)
from zlc_runtime.selection_bridge import _StaleFit


@dataclass
class _Source:
    declaration: DatasetOutputDeclaration

    instance_id = "camera"

    @property
    def dataset_output_declarations(self):
        return (self.declaration,)

    def signal_key(self, name: str) -> str:
        return f"camera/{name}"


class _Events:
    def __init__(self) -> None:
        self._fit_callbacks = []
        self._bridge: SelectionBridge | None = None

    def subscribe_fit(self, callback):
        self._fit_callbacks.append(callback)
        owner = getattr(callback, "__self__", None)
        if isinstance(owner, SelectionBridge):
            self._bridge = owner

        def unsubscribe() -> None:
            if callback in self._fit_callbacks:
                self._fit_callbacks.remove(callback)

        return unsubscribe

    def emit_selection(self, change: SelectionChange, state: SelectionState) -> None:
        bridge = self._bridge
        if bridge is None:
            raise RuntimeError("selection test bridge has not started")
        if change in {SelectionChange.ADDED, SelectionChange.UPDATED}:
            return
        if change is SelectionChange.REMOVED:
            bridge.clear_selection()
            return
        publication = bridge._plane.latest_publication(bridge._source_signal)
        if publication is None:
            raise RuntimeError("selection test source has no publication")
        bridge.commit_selection(state, source_publication=publication)

    def emit_fit(self, event: FitEventValue | None) -> None:
        for callback in tuple(self._fit_callbacks):
            callback(event)


def _snapshot(
    name: str,
    revision: int,
    schema: DatasetSchema,
    values: np.ndarray,
) -> OwnedSnapshot:
    block = DataBlock(
        BlockId(f"{name}-{revision}"),
        DatasetRevision(revision),
        np.asarray(values, dtype=np.float64),
        CellValidity(
            np.ones(
                (schema.repeat_axis.size, schema.point_table.row_count),
                dtype=np.bool_,
            )
        ),
        schema,
    )
    return OwnedSnapshot(
        block.ref(StreamGenerationId(f"{name}-generation")),
        block,
    )


def _source_setup(
    schema: DatasetSchema,
    values: np.ndarray,
):
    declaration = DatasetOutputDeclaration("frame", "test.camera.frame")
    state = {
        "frame": LiveDatasetOutput(
            declaration,
            _snapshot("frame", 1, schema, values),
            MonitorCoverage(schema.repeat_axis.size * schema.point_table.row_count,
                            schema.repeat_axis.size * schema.point_table.row_count),
        )
    }
    source = _Source(declaration)
    plane = SignalDataPlane()
    plane.reserve(source)
    plane.commit_live(source, state)
    initial = plane.freeze()
    return plane, source, None, state, initial


def _source_generation(plane: SignalDataPlane) -> str:
    publication = plane.latest_publication("camera/frame")
    assert publication is not None
    value = publication.value("camera/frame")
    assert value is not None
    return str(value.snapshot.ref.stream_generation.value)


def _scalar_fit_event(
    plane: SignalDataPlane,
    value: float,
    error: float,
    batch_revision: int,
) -> FitEventValue:
    return FitEventValue(
        parameter_names=("x0",),
        parameter_units={"x0": "pixel"},
        parameter_values={"x0": np.asarray([value])},
        parameter_errors={"x0": np.asarray([error])},
        success=np.asarray([True]),
        sample_axis_domain="",
        sample_axis_id="",
        sample_axis_name="",
        sample_coordinates=np.asarray([0.0]),
        sample_unit="",
        sample_labels=None,
        source_generation=_source_generation(plane),
        source_revision=1,
        batch_revision=batch_revision,
    )


def _finite_source_setup(
    canonical_schema: DatasetSchema,
    event_schema: DatasetSchema,
    values: np.ndarray,
    *,
    origin: tuple[int, int],
):
    declaration = DatasetOutputDeclaration("frame", "test.camera.frame")
    total = (
        canonical_schema.repeat_axis.size
        * canonical_schema.point_table.row_count
    )
    event_cells = (
        event_schema.repeat_axis.size
        * event_schema.point_table.row_count
    )
    state = {
        "frame": LiveDatasetOutput(
            declaration,
            _snapshot("frame", 1, event_schema, values),
            DatasetCoverage(event_cells, total),
            canonical_schema=canonical_schema,
            cell_origin=origin,
        )
    }
    source = _Source(declaration)
    plane = SignalDataPlane()
    plane.reserve(source)
    plane.commit_live(source, state)
    return plane, source, state, plane.freeze()


def _commit_source(
    plane: SignalDataPlane,
    source: _Source,
    state: dict[str, LiveDatasetOutput],
) -> None:
    plane.commit_live(source, state)


def _seal_source(
    plane: SignalDataPlane,
    source: _Source,
    output: LiveDatasetOutput,
) -> None:
    plane.retire(source)
    plane.begin_generation(source)
    schema = output.snapshot.block.schema
    cells = schema.repeat_axis.size * schema.point_table.row_count
    plane.commit_live(
        source,
        {
            "frame": LiveDatasetOutput(
                output.declaration,
                output.snapshot,
                DatasetCoverage(cells, cells),
                output.run_record,
                schema,
                (0, 0),
            )
        },
    )
    plane.seal_committed(source)


def _image_schema() -> DatasetSchema:
    repeat = AxisSpec(AxisId("repeat"), "repeat", REPEAT, 1, (0,))
    x = AxisSpec(AxisId("x"), "x", SPATIAL_X, 4, (-1.0, 0.0, 1.0, 2.0))
    y = AxisSpec(AxisId("y"), "y", SPATIAL_Y, 3, (10.0, 20.0, 30.0))
    return DatasetSchema(
        repeat,
        PointTable(1),
        None,
        ValueSchema((x, y), ValidityContract.value(), np.dtype("float64"), "counts"),
    )


def _curve_schema(with_facet: bool = False) -> DatasetSchema:
    repeat = AxisSpec(AxisId("repeat"), "repeat", REPEAT, 1, (0,))
    columns = [
        PointColumn(
            AxisId("x"),
            "x",
            SCAN_POINT,
            PointColumn.NUMERIC,
            (0.0, 1.0, 2.0, 3.0, 4.0),
        )
    ]
    if with_facet:
        columns.append(
            PointColumn(
                AxisId("facet"),
                "facet",
                SCAN_POINT,
                PointColumn.NUMERIC,
                (0.0, 0.0, 1.0, 1.0, 1.0),
            )
        )
    return DatasetSchema(
        repeat,
        PointTable(5, tuple(columns)),
        None,
        ValueSchema.scalar(np.dtype("float64"), "counts"),
    )


def _wait_for_signal(plane, signal_name: str, revision: int):
    deadline = time.monotonic() + 2.0
    while True:
        front = plane.freeze()
        value = front.value(signal_name)
        if value is not None and value.snapshot.ref.revision.value == revision:
            return front
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{signal_name} did not reach revision {revision}")
        time.sleep(0.001)


def _close(bridge: SelectionBridge, plane: SignalDataPlane, source: _Source) -> None:
    bridge.close()
    plane.retire(source)
    plane.close()


def test_image_area_materializes_closed_roi_and_mean_with_lineage() -> None:
    schema = _image_schema()
    values = np.arange(12, dtype=np.float64).reshape(1, 1, 4, 3)
    plane, source, _slot, _state, initial = _source_setup(schema, values)
    events = _Events()
    plane.set_front_signals(
        {
            "camera/frame",
            "@logic/image/roi_frame",
            "@logic/image/roi_mean",
        }
    )
    bridge = SelectionBridge(
        plane,
        "camera/frame",
        events,
        bridge_id="image",
    )
    bridge.start()
    state = SelectionState(
        "image",
        "area",
        (
            SelectionRange("x", 0.0, 1.0, domain="data"),
            SelectionRange("y", 20.0, 30.0, domain="data"),
        ),
        revision=1,
    )
    try:
        events.emit_selection(SelectionChange.COMMITTED, state)
        front = plane.freeze()
        roi_frame = front.value("@logic/image/roi_frame")
        roi_mean = front.value("@logic/image/roi_mean")
        assert roi_frame is not None and roi_mean is not None
        np.testing.assert_array_equal(
            roi_frame.snapshot.block.values,
            values[:, :, 1:3, 1:3],
        )
        assert float(roi_mean.snapshot.block.values.reshape(-1)[0]) == 6.0
        roi_publication = front.publication("@logic/image/roi_frame")
        assert roi_publication is not None
        assert roi_publication.direct_parent_refs == (initial.publication("camera/frame").event_ref,)
    finally:
        _close(bridge, plane, source)


def test_image_area_catalog_statistics_and_publication_choice_share_one_owner() -> None:
    schema = _image_schema()
    values = np.arange(12, dtype=np.float64).reshape(1, 1, 4, 3)
    plane, source, _slot, _state, _initial = _source_setup(schema, values)
    events = _Events()
    derived = {name for name, _label in selection_output_catalog("area")}
    assert derived == {
        "roi_frame",
        "roi_mean",
        "roi_min",
        "roi_max",
        "roi_min_10_mean",
        "roi_max_10_mean",
    }
    plane.set_front_signals(
        {"camera/frame", *(f"@logic/image/{name}" for name in derived)}
    )
    bridge = SelectionBridge(plane, "camera/frame", events, bridge_id="image")
    bridge.start()
    selection = SelectionState(
        "image",
        "area",
        (
            SelectionRange("x", -1.0, 2.0, domain="data"),
            SelectionRange("y", 10.0, 30.0, domain="data"),
        ),
        revision=1,
    )
    try:
        events.emit_selection(SelectionChange.COMMITTED, selection)
        front = plane.freeze()
        expected = {
            "roi_mean": 5.5,
            "roi_min": 0.0,
            "roi_max": 11.0,
            "roi_min_10_mean": 4.5,
            "roi_max_10_mean": 6.5,
        }
        for name, scalar in expected.items():
            value = front.value(f"@logic/image/{name}")
            assert value is not None
            assert float(value.snapshot.block.values.reshape(-1)[0]) == scalar
            assert value.snapshot.block.schema.cell_schema.value_unit == "counts"

        bridge.configure_outputs({name: name == "roi_max" for name in derived})
        front = plane.freeze()
        assert front.value("@logic/image/roi_max") is not None
        assert all(
            front.value(f"@logic/image/{name}") is None
            for name in derived - {"roi_max"}
        )

        bridge.configure_outputs({name: True for name in derived})
        front = plane.freeze()
        assert all(front.value(f"@logic/image/{name}") is not None for name in derived)
    finally:
        _close(bridge, plane, source)


def test_selection_commit_republishes_same_source_and_source_revision_follows() -> None:
    schema = _image_schema()
    values = np.arange(12, dtype=np.float64).reshape(1, 1, 4, 3)
    plane, source, slot, state, _initial = _source_setup(schema, values)
    events = _Events()
    plane.set_front_signals({"camera/frame", "@logic/image/roi_mean"})
    bridge = SelectionBridge(plane, "camera/frame", events, bridge_id="image")
    bridge.start()
    first_state = SelectionState(
        "image",
        "area",
        (
            SelectionRange("x", -1.0, 0.0, domain="data"),
            SelectionRange("y", 10.0, 20.0, domain="data"),
        ),
        revision=1,
    )
    second_state = SelectionState(
        "image",
        "area",
        (
            SelectionRange("x", 1.0, 2.0, domain="data"),
            SelectionRange("y", 20.0, 30.0, domain="data"),
        ),
        revision=2,
    )
    try:
        events.emit_selection(SelectionChange.COMMITTED, first_state)
        first = plane.freeze().publication("@logic/image/roi_mean")
        assert first is not None
        events.emit_selection(SelectionChange.COMMITTED, second_state)
        second_front = plane.freeze()
        second = second_front.publication("@logic/image/roi_mean")
        assert second is not None
        assert second.event_ref.generation != first.event_ref.generation
        assert second.event_ref.sequence == 1
        assert second.direct_parent_refs == first.direct_parent_refs
        assert float(second_front.value("@logic/image/roi_mean").snapshot.block.values.reshape(-1)[0]) == 9.0

        state["frame"] = LiveDatasetOutput(
            state["frame"].declaration,
            _snapshot("frame", 2, schema, values + 100.0),
            MonitorCoverage(1, 1),
        )
        _commit_source(plane, source, state)
        front = _wait_for_signal(plane, "@logic/image/roi_mean", 2)
        publication = front.publication("@logic/image/roi_mean")
        assert publication is not None
        assert publication.direct_parent_refs[0].sequence == 2
        assert float(front.value("@logic/image/roi_mean").snapshot.block.values.reshape(-1)[0]) == 109.0
    finally:
        _close(bridge, plane, source)


@pytest.mark.parametrize("phase", ("commit", "processor"))
def test_close_does_not_wait_for_selection_materialization_or_publish_stale(
    monkeypatch, phase: str,
) -> None:
    schema = _image_schema()
    values = np.arange(12, dtype=np.float64).reshape(1, 1, 4, 3)
    plane, source, _slot, source_state, _initial = _source_setup(schema, values)
    events = _Events()
    bridge = SelectionBridge(
        plane,
        "camera/frame",
        events,
        bridge_id="close-race",
    )
    bridge.start()
    entered = Event()
    release = Event()
    close_done = Event()
    callback_errors: list[BaseException] = []
    close_errors: list[BaseException] = []
    selection = SelectionState(
        "image",
        "area",
        (
            SelectionRange("x", 0.0, 1.0, domain="data"),
            SelectionRange("y", 20.0, 30.0, domain="data"),
        ),
        revision=1,
    )
    if phase == "processor":
        events.emit_selection(SelectionChange.COMMITTED, selection)
    real_materialize = bridge._materialize_selection_outputs

    def gated_materialize(snapshot, state):
        entered.set()
        if not release.wait(2.0):
            raise TimeoutError("selection materialization gate did not open")
        return real_materialize(snapshot, state)

    monkeypatch.setattr(
        bridge,
        "_materialize_selection_outputs",
        gated_materialize,
    )

    def commit_selection() -> None:
        try:
            if phase == "commit":
                events.emit_selection(SelectionChange.COMMITTED, selection)
            else:
                source_state["frame"] = LiveDatasetOutput(
                    source_state["frame"].declaration,
                    _snapshot("frame", 2, schema, values + 1.0),
                    MonitorCoverage(1, 1),
                )
                _commit_source(plane, source, source_state)
                deadline = time.monotonic() + 1.0
                while not entered.is_set() and time.monotonic() < deadline:
                    plane.freeze()
                    time.sleep(0.001)
        except BaseException as error:
            callback_errors.append(error)

    worker = Thread(target=commit_selection, daemon=True)

    def close_bridge() -> None:
        try:
            bridge.close()
        except BaseException as error:
            close_errors.append(error)
        finally:
            close_done.set()

    closer = Thread(target=close_bridge, daemon=True)
    worker.start()
    assert entered.wait(1.0), callback_errors
    closer.start()
    closed_without_waiting = close_done.wait(0.5)
    release.set()
    worker.join(2.0)
    closer.join(2.0)
    try:
        assert closed_without_waiting, "SelectionBridge.close waited on numeric work"
        assert not callback_errors
        assert not close_errors
        assert (
            plane.latest_publication("@logic/close-race/roi_frame") is None
        ), "materialization completed after close and published a stale result"
    finally:
        bridge.close()
        plane.retire(source)
        plane.close()


def test_selection_derives_from_the_canonical_repeat_prefix_not_the_event_chunk() -> None:
    event_schema = _image_schema()
    canonical_schema = replace(
        event_schema,
        repeat_axis=replace(
            event_schema.repeat_axis,
            size=3,
            coordinates=(0, 1, 2),
        ),
    )
    first = np.arange(12, dtype=np.float64).reshape(1, 1, 4, 3)
    plane, source, state, _initial = _finite_source_setup(
        canonical_schema,
        event_schema,
        first,
        origin=(0, 0),
    )
    events = _Events()
    signal = "@logic/repeats/roi_mean"
    plane.set_front_signals({"camera/frame", signal})
    bridge = SelectionBridge(
        plane,
        "camera/frame",
        events,
        bridge_id="repeats",
    )
    bridge.start()
    selection = SelectionState(
        "image",
        "area",
        (
            SelectionRange("x", -1.0, 2.0, domain="data"),
            SelectionRange("y", 10.0, 30.0, domain="data"),
        ),
        revision=1,
    )
    try:
        events.emit_selection(SelectionChange.COMMITTED, selection)
        first_derived = plane.freeze().value(signal)
        assert first_derived is not None, bridge.last_error
        assert first_derived.snapshot.block.values.shape == (3, 1, 1)
        np.testing.assert_array_equal(
            first_derived.snapshot.expanded_validity()[..., 0],
            np.asarray([[True], [False], [False]]),
        )

        second = first + 100.0
        state["frame"] = LiveDatasetOutput(
            state["frame"].declaration,
            _snapshot("frame", 2, event_schema, second),
            DatasetCoverage(2, 3),
            canonical_schema=canonical_schema,
            cell_origin=(1, 0),
        )
        _commit_source(plane, source, state)
        front = _wait_for_signal(plane, signal, 2)
        derived = front.value(signal)
        assert derived is not None, bridge.last_error
        np.testing.assert_allclose(
            derived.snapshot.block.values[:2, 0, 0],
            np.asarray([first.mean(), second.mean()]),
        )
        np.testing.assert_array_equal(
            derived.snapshot.expanded_validity()[..., 0],
            np.asarray([[True], [True], [False]]),
        )
    finally:
        _close(bridge, plane, source)


def test_restored_selection_starts_on_displayed_prefix_then_catches_live_latest() -> None:
    """A restored panel answers for its screen before following newer data."""

    event_schema = _image_schema()
    canonical_schema = replace(
        event_schema,
        repeat_axis=replace(
            event_schema.repeat_axis,
            size=3,
            coordinates=(0, 1, 2),
        ),
    )
    first = np.full(event_schema.physical_shape, 1.0)
    plane, source, state, initial = _finite_source_setup(
        canonical_schema,
        event_schema,
        first,
        origin=(0, 0),
    )
    displayed = initial.publication("camera/frame")
    assert displayed is not None

    second = np.full(event_schema.physical_shape, 2.0)
    state["frame"] = LiveDatasetOutput(
        state["frame"].declaration,
        _snapshot("frame", 2, event_schema, second),
        DatasetCoverage(2, 3),
        canonical_schema=canonical_schema,
        cell_origin=(1, 0),
    )
    _commit_source(plane, source, state)
    latest = plane.latest_publication("camera/frame")
    assert latest is not None and latest is not displayed

    events = _Events()
    signal = "@logic/restored-prefix/roi_mean"
    plane.set_front_signals({"camera/frame", signal})
    bridge = SelectionBridge(
        plane,
        "camera/frame",
        events,
        bridge_id="restored-prefix",
    )
    selection = SelectionState(
        "image",
        "area",
        (
            SelectionRange("x", -1.0, 2.0, domain="data"),
            SelectionRange("y", 10.0, 30.0, domain="data"),
        ),
        revision=1,
    )
    try:
        bridge.start(
            initial_selection=selection,
            initial_publication=displayed,
        )
        initial_derived = plane.latest_publication(signal)
        assert initial_derived is not None
        assert plane.direct_parent_publications(initial_derived) == (displayed,)
        generation = initial_derived.event_ref.generation
        initial_value = initial_derived.value(signal)
        assert initial_value is not None
        np.testing.assert_allclose(
            initial_value.snapshot.block.values[:, 0, 0],
            np.asarray([1.0, 0.0, 0.0]),
        )
        np.testing.assert_array_equal(
            initial_value.snapshot.expanded_validity()[:, 0, 0],
            np.asarray([True, False, False]),
        )

        caught_up = _wait_for_signal(plane, signal, 2)
        caught_up_publication = caught_up.publication(signal)
        assert caught_up_publication is not None
        assert caught_up_publication.event_ref.generation == generation
        assert plane.direct_parent_publications(caught_up_publication) == (latest,)
        caught_up_value = caught_up.value(signal)
        assert caught_up_value is not None
        np.testing.assert_allclose(
            caught_up_value.snapshot.block.values[:, 0, 0],
            np.asarray([1.0, 2.0, 0.0]),
        )

        third = np.full(event_schema.physical_shape, 3.0)
        state["frame"] = LiveDatasetOutput(
            state["frame"].declaration,
            _snapshot("frame", 3, event_schema, third),
            DatasetCoverage(3, 3),
            canonical_schema=canonical_schema,
            cell_origin=(2, 0),
        )
        _commit_source(plane, source, state)
        final = _wait_for_signal(plane, signal, 3)
        final_publication = final.publication(signal)
        assert final_publication is not None
        assert final_publication.event_ref.generation == generation
        assert final_publication.direct_parent_refs[0].sequence == 3
        final_value = final.value(signal)
        assert final_value is not None
        np.testing.assert_allclose(
            final_value.snapshot.block.values[:, 0, 0],
            np.asarray([1.0, 2.0, 3.0]),
        )
    finally:
        _close(bridge, plane, source)


def test_delayed_selection_of_publication_n_never_reads_publication_n_plus_one(
    monkeypatch,
) -> None:
    event_schema = _image_schema()
    canonical_schema = replace(
        event_schema,
        repeat_axis=replace(
            event_schema.repeat_axis,
            size=3,
            coordinates=(0, 1, 2),
        ),
    )
    first = np.full(event_schema.physical_shape, 1.0)
    plane, source, state, _initial = _finite_source_setup(
        canonical_schema,
        event_schema,
        first,
        origin=(0, 0),
    )
    events = _Events()
    signal = "@logic/exact-prefix/roi_mean"
    plane.set_front_signals({"camera/frame", signal})
    bridge = SelectionBridge(
        plane,
        "camera/frame",
        events,
        bridge_id="exact-prefix",
    )
    bridge.start()
    selection = SelectionState(
        "image",
        "area",
        (
            SelectionRange("x", -1.0, 2.0, domain="data"),
            SelectionRange("y", 10.0, 30.0, domain="data"),
        ),
        revision=1,
    )
    events.emit_selection(SelectionChange.COMMITTED, selection)
    plane.freeze()

    entered = Event()
    release = Event()
    observed: list[OwnedSnapshot] = []
    original_current_dataset = plane.current_dataset

    def delayed_current_dataset(name, publication=None):
        if publication is not None and publication.event_ref.sequence == 2:
            entered.set()
            assert release.wait(2.0)
        snapshot = original_current_dataset(name, publication)
        if publication is not None and publication.event_ref.sequence == 2:
            observed.append(snapshot)
        return snapshot

    monkeypatch.setattr(plane, "current_dataset", delayed_current_dataset)
    try:
        state["frame"] = LiveDatasetOutput(
            state["frame"].declaration,
            _snapshot("frame", 2, event_schema, np.full(event_schema.physical_shape, 2.0)),
            DatasetCoverage(2, 3),
            canonical_schema=canonical_schema,
            cell_origin=(1, 0),
        )
        _commit_source(plane, source, state)
        plane.freeze()
        assert entered.wait(2.0)

        state["frame"] = LiveDatasetOutput(
            state["frame"].declaration,
            _snapshot("frame", 3, event_schema, np.full(event_schema.physical_shape, 3.0)),
            DatasetCoverage(3, 3),
            canonical_schema=canonical_schema,
            cell_origin=(2, 0),
        )
        _commit_source(plane, source, state)
        release.set()
        deadline = time.monotonic() + 2.0
        while not observed and time.monotonic() < deadline:
            plane.freeze()
            time.sleep(0.001)
        assert observed
        snapshot = observed[0]
        np.testing.assert_array_equal(
            snapshot.block.values[:, 0, 0, 0],
            np.asarray([1.0, 2.0, 0.0]),
        )
        np.testing.assert_array_equal(
            snapshot.expanded_validity()[:, 0, 0, 0],
            np.asarray([True, True, False]),
        )
    finally:
        release.set()
        _close(bridge, plane, source)


def test_curve_range_and_facet_condition_select_point_rows_inclusive() -> None:
    schema = _curve_schema(with_facet=True)
    values = np.asarray([[[1.0], [2.0], [10.0], [20.0], [30.0]]])
    plane, source, _slot, _state, _initial = _source_setup(schema, values)
    events = _Events()
    plane.set_front_signals({"camera/frame", "@logic/curve/roi_mean"})
    bridge = SelectionBridge(plane, "camera/frame", events, bridge_id="curve")
    bridge.start()
    selection = SelectionState(
        "curve",
        "x_range",
        (SelectionRange("x", 0.0, 3.0, domain="point_coordinate"),),
        (FacetCondition("facet", 1.0, "point_coordinate"),),
        revision=1,
    )
    try:
        events.emit_selection(SelectionChange.COMMITTED, selection)
        front = plane.freeze()
        value = front.value("@logic/curve/roi_mean")
        assert value is not None
        # The range and the facet cut POINT ROWS, and a cut of the point axis
        # is not a pooling of it: rows 2 and 3 survive as two points.
        np.testing.assert_array_equal(
            value.snapshot.block.values,
            np.asarray([[[10.0], [20.0]]]),
        )
        derived = value.snapshot.block.schema
        assert derived.point_table.row_count == 2
        assert {column.name: column.values for column in derived.point_table.columns} == {
            "x": (2.0, 3.0),
            "facet": (1.0, 1.0),
        }
    finally:
        _close(bridge, plane, source)


def test_fit_event_publishes_parameter_and_error_scalars() -> None:
    schema = _curve_schema()
    values = np.asarray([[[0.0], [0.0], [0.0], [0.0], [0.0]]])
    plane, source, _slot, _state, _initial = _source_setup(schema, values)
    events = _Events()
    plane.set_front_signals(
        {"camera/frame", "@logic/fit/x0", "@logic/fit/x0_err"}
    )
    bridge = SelectionBridge(plane, "camera/frame", events, bridge_id="fit")
    bridge.start()
    try:
        events.emit_fit(
            _scalar_fit_event(plane, 2.5, 0.1, 1)
        )
        front = plane.freeze()
        parameter = front.value("@logic/fit/x0")
        error = front.value("@logic/fit/x0_err")
        assert parameter is not None and error is not None
        assert float(parameter.snapshot.block.values.reshape(-1)[0]) == 2.5
        assert float(error.snapshot.block.values.reshape(-1)[0]) == 0.1
        publication = front.publication("@logic/fit/x0")
        assert publication is not None
        assert publication.direct_parent_refs[0].sequence == 1
        bridge.configure_outputs({"x0": False})
        hidden = plane.freeze()
        assert hidden.value("@logic/fit/x0") is None
        assert hidden.value("@logic/fit/x0_err") is not None
        bridge.configure_outputs({"x0": True})
        replayed = plane.freeze().value("@logic/fit/x0")
        assert replayed is not None
        assert float(replayed.snapshot.block.values.reshape(-1)[0]) == 2.5
        publication = plane.freeze().publication("@logic/fit/x0")
        assert publication is not None
        events.emit_selection(
            SelectionChange.REMOVED,
            SelectionState(
                "curve",
                "x_range",
                (SelectionRange("x", 0.0, 1.0, domain="point_coordinate"),),
                revision=1,
            ),
        )
        assert plane.freeze().value("@logic/fit/x0") is not None
        events.emit_fit(
            _scalar_fit_event(plane, 3.5, 0.2, 2)
        )
        second_front = plane.freeze()
        second_parameter = second_front.value("@logic/fit/x0")
        second_publication = second_front.publication("@logic/fit/x0")
        assert second_parameter is not None and second_publication is not None
        assert second_publication.event_ref.generation == publication.event_ref.generation
        assert second_publication.event_ref.sequence > publication.event_ref.sequence
        assert float(second_parameter.snapshot.block.values.reshape(-1)[0]) == 3.5
        assert second_parameter.snapshot.ref.revision.value == 2
    finally:
        _close(bridge, plane, source)


def _batch_fit_event(
    plane: SignalDataPlane,
    *,
    source_revision: int,
    batch_revision: int = 1,
    text_samples: bool = False,
) -> FitEventValue:
    coordinates = np.asarray([0.0, 1.0, 2.0])
    labels = None
    unit = "V"
    if text_samples:
        labels = ("red", "green", "blue")
        unit = ""
    else:
        coordinates = np.asarray([10.0, 20.0, 35.0])
    return FitEventValue(
        parameter_names=("center", "width"),
        parameter_units={"center": "pixel", "width": ""},
        parameter_values={
            "center": np.asarray([1.0, np.nan, 3.0]),
            "width": np.asarray([0.5, np.nan, 0.7]),
        },
        parameter_errors={
            "center": np.asarray([0.1, np.nan, np.nan]),
            "width": np.asarray([0.05, np.nan, 0.07]),
        },
        success=np.asarray([True, False, True]),
        sample_axis_domain="point_row",
        sample_axis_id="",
        sample_axis_name="facet",
        sample_coordinates=coordinates,
        sample_unit=unit,
        sample_labels=labels,
        source_generation=_source_generation(plane),
        source_revision=source_revision,
        batch_revision=batch_revision,
    )


def test_a_withdrawn_fit_takes_its_outputs_with_it() -> None:
    """The fit is no longer on screen, so it is no longer published.

    A removed box retires its processor through SelectionChange.REMOVED; an
    un-armed fit says the same thing by delivering ``None``.  Without it the
    parameters stayed on the plane, frozen at an answer nobody is making any
    more, and their names stayed owned so re-arming could not reclaim them.
    """

    schema = _curve_schema()
    values = np.asarray([[[0.0], [0.0], [0.0], [0.0], [0.0]]])
    plane, _source, _slot, _state, _initial = _source_setup(schema, values)
    events = _Events()
    plane.set_front_signals({"camera/frame", "@logic/fit/x0", "@logic/fit/x0_err"})
    bridge = SelectionBridge(plane, "camera/frame", events, bridge_id="fit")
    bridge.start()
    try:
        def emit(x0: float, batch_revision: int) -> None:
            events.emit_fit(_scalar_fit_event(plane, x0, 0.1, batch_revision))

        emit(2.5, 1)
        assert plane.freeze().value("@logic/fit/x0") is not None

        events.emit_fit(None)
        withdrawn = plane.freeze()
        assert withdrawn.value("@logic/fit/x0") is None
        assert withdrawn.value("@logic/fit/x0_err") is None

        emit(3.5, 2)
        replayed = plane.freeze().value("@logic/fit/x0")
        assert replayed is not None
        assert float(replayed.snapshot.block.values.reshape(-1)[0]) == 3.5
    finally:
        bridge.close()


def test_fit_event_batch_publishes_vectors_with_units_validity_and_lineage() -> None:
    schema = _curve_schema()
    values = np.zeros((1, 5, 1), dtype=np.float64)
    plane, source, _slot, _state, initial = _source_setup(schema, values)
    events = _Events()
    plane.set_front_signals(
        {
            "camera/frame",
            "@logic/batch/center",
            "@logic/batch/center_err",
            "@logic/batch/width",
            "@logic/batch/width_err",
        }
    )
    bridge = SelectionBridge(plane, "camera/frame", events, bridge_id="batch")
    bridge.start()
    try:
        events.emit_fit(
            _batch_fit_event(plane, source_revision=1)
        )
        front = plane.freeze()
        for name in ("center", "center_err", "width", "width_err"):
            signal = front.value(f"@logic/batch/{name}")
            publication = front.publication(f"@logic/batch/{name}")
            assert signal is not None and publication is not None
            assert signal.snapshot.block.values.shape == (1, 3, 1)
            assert signal.snapshot.ref.revision.value == 1
            assert publication.direct_parent_refs == (
                initial.publication("camera/frame").event_ref,
            )

        center = front.value("@logic/batch/center")
        center_error = front.value("@logic/batch/center_err")
        width = front.value("@logic/batch/width")
        assert center is not None and center_error is not None and width is not None
        np.testing.assert_allclose(
            center.snapshot.block.values.reshape(-1),
            np.asarray([1.0, np.nan, 3.0]),
            equal_nan=True,
        )
        np.testing.assert_array_equal(
            center.snapshot.block.validity.mask,
            np.asarray([[True, False, True]]),
        )
        np.testing.assert_allclose(
            center_error.snapshot.block.values.reshape(-1),
            np.asarray([0.1, np.nan, np.nan]),
            equal_nan=True,
        )
        np.testing.assert_array_equal(
            center_error.snapshot.block.validity.mask,
            np.asarray([[True, False, False]]),
        )
        assert center.snapshot.block.schema.cell_schema.value_unit == "pixel"
        assert width.snapshot.block.schema.cell_schema.value_unit is None
        column = center.snapshot.block.schema.point_table.columns[0]
        assert column.name == "facet"
        assert column.values == (10.0, 20.0, 35.0)
        assert column.unit == "V"

        for name in ("center", "center_err", "width", "width_err"):
            signal = front.value(f"@logic/batch/{name}")
            assert signal is not None
            values = signal.snapshot.block.values.reshape(1, -1)
            valid = signal.snapshot.block.validity.mask
            assert np.all(np.isfinite(values[valid]))
    finally:
        _close(bridge, plane, source)


def test_fit_event_batch_text_samples_use_numeric_indices_and_preserve_labels() -> None:
    schema = _curve_schema()
    values = np.zeros((1, 5, 1), dtype=np.float64)
    plane, source, _slot, _state, _initial = _source_setup(schema, values)
    events = _Events()
    plane.set_front_signals(
        {"camera/frame", "@logic/text/center", "@logic/text/center_err"}
    )
    bridge = SelectionBridge(plane, "camera/frame", events, bridge_id="text")
    bridge.start()
    try:
        event = _batch_fit_event(
            plane,
            source_revision=1,
            text_samples=True,
        )
        assert event.sample_labels == ("red", "green", "blue")
        events.emit_fit(event)
        front = plane.freeze()
        value = front.value("@logic/text/center")
        assert value is not None
        column = value.snapshot.block.schema.point_table.columns[0]
        assert column.values == (0.0, 1.0, 2.0)
        assert column.unit is None
        label_column = value.snapshot.block.schema.point_table.columns[1]
        assert label_column.name == "facet_label"
        assert label_column.value_kind == PointColumn.TEXT
        assert label_column.values == ("red", "green", "blue")
    finally:
        _close(bridge, plane, source)


def _single_cell_facet_event(
    plane: SignalDataPlane,
    *,
    source_revision: int,
    batch_revision: int,
) -> FitEventValue:
    return FitEventValue(
        parameter_names=("center",),
        parameter_units={"center": "pixel"},
        parameter_values={"center": np.asarray([4.0])},
        parameter_errors={"center": np.asarray([0.25])},
        success=np.asarray([True]),
        sample_axis_domain="point_row",
        sample_axis_id="",
        sample_axis_name="facet",
        sample_coordinates=np.asarray([42.0]),
        sample_unit="V",
        sample_labels=None,
        source_generation=_source_generation(plane),
        source_revision=source_revision,
        batch_revision=batch_revision,
    )


def test_single_cell_facet_is_a_valid_vector_fit() -> None:
    schema = _curve_schema()
    values = np.zeros((1, 5, 1), dtype=np.float64)
    plane, source, _slot, _state, _initial = _source_setup(schema, values)
    events = _Events()
    plane.set_front_signals(
        {"camera/frame", "@logic/one/center", "@logic/one/center_err"}
    )
    bridge = SelectionBridge(plane, "camera/frame", events, bridge_id="one")
    bridge.start()
    try:
        events.emit_fit(
            _single_cell_facet_event(
                plane,
                source_revision=1,
                batch_revision=1,
            )
        )
        front = plane.freeze()
        value = front.value("@logic/one/center")
        error = front.value("@logic/one/center_err")
        assert value is not None and error is not None
        assert value.snapshot.block.schema.point_table.columns[0].name == "facet"
        assert value.snapshot.block.schema.point_table.columns[0].values == (42.0,)
        assert float(value.snapshot.block.values.reshape(-1)[0]) == 4.0
        assert float(error.snapshot.block.values.reshape(-1)[0]) == 0.25
    finally:
        _close(bridge, plane, source)


def test_late_stale_fit_failure_cannot_withdraw_newer_batch(monkeypatch) -> None:
    schema = _curve_schema()
    values = np.zeros((1, 5, 1), dtype=np.float64)
    plane, source, slot, state, _initial = _source_setup(schema, values)
    events = _Events()
    plane.set_front_signals({"camera/frame", "@logic/race/center"})
    bridge = SelectionBridge(plane, "camera/frame", events, bridge_id="race")
    bridge.start()
    gate = Event()
    original_evaluate = bridge._evaluate_processor

    def delayed_evaluate(processor, signal, publication):
        if signal.snapshot.ref.revision.value == 2:
            gate.wait(2.0)
        return original_evaluate(processor, signal, publication)

    monkeypatch.setattr(bridge, "_evaluate_processor", delayed_evaluate)
    try:
        events.emit_fit(
            _batch_fit_event(plane, source_revision=1, batch_revision=1)
        )
        processor = bridge._fit_processor
        assert processor is not None

        state["frame"] = LiveDatasetOutput(
            state["frame"].declaration,
            _snapshot("frame", 2, schema, values + 1.0),
            MonitorCoverage(1, 5),
        )
        _commit_source(plane, source, state)
        plane.freeze()

        events.emit_fit(
            _batch_fit_event(plane, source_revision=2, batch_revision=2)
        )
        bridge._accept_processor_failure(processor, _StaleFit(1))
        front = plane.freeze()
        newer = front.value("@logic/race/center")
        assert newer is not None
        np.testing.assert_allclose(
            newer.snapshot.block.values.reshape(-1),
            np.asarray([1.0, np.nan, 3.0]),
            equal_nan=True,
        )
        assert newer.snapshot.ref.revision.value == 2
        assert bridge.last_error is None
    finally:
        gate.set()
        _close(bridge, plane, source)


def test_fit_batch_revision_is_retained_within_source_and_resets_on_restart() -> None:
    schema = _curve_schema()
    values = np.zeros((1, 5, 1), dtype=np.float64)
    plane, source, slot, state, _initial = _source_setup(schema, values)
    events = _Events()
    plane.set_front_signals({"camera/frame", "@logic/revision/center"})
    bridge = SelectionBridge(plane, "camera/frame", events, bridge_id="revision")
    bridge.start()
    try:
        first_event = _batch_fit_event(
            plane,
            source_revision=1,
            batch_revision=1,
        )
        events.emit_fit(first_event)
        first = plane.freeze().value("@logic/revision/center")
        assert first is not None
        plane.freeze()
        first_names = tuple(
            item.name
            for item in plane.describe_signals()
            if item.owner_id.startswith("revision:fit:")
        )

        state["frame"] = LiveDatasetOutput(
            state["frame"].declaration,
            _snapshot("frame", 2, schema, values + 1.0),
            MonitorCoverage(1, 5),
        )
        _commit_source(plane, source, state)
        plane.freeze()
        between = plane.freeze()
        assert between.value("@logic/revision/center") is first
        assert tuple(
            item.name
            for item in plane.describe_signals()
            if item.owner_id.startswith("revision:fit:")
        ) == first_names
        events.emit_fit(
            _batch_fit_event(plane, source_revision=2, batch_revision=2)
        )
        front = _wait_for_signal(plane, "@logic/revision/center", 2)
        second = front.value("@logic/revision/center")
        assert second is not None
        assert second.snapshot.ref.revision.value > first.snapshot.ref.revision.value

        previous_generation = _source_generation(plane)
        plane.retire(source)
        plane.begin_generation(source)
        state["frame"] = LiveDatasetOutput(
            state["frame"].declaration,
            _snapshot("restarted-frame", 1, schema, values + 2.0),
            MonitorCoverage(1, 5),
        )
        _commit_source(plane, source, state)
        restarted_source = plane.latest_publication("camera/frame")
        assert restarted_source is not None
        assert _source_generation(plane) != previous_generation
        events.emit_fit(
            _batch_fit_event(plane, source_revision=1, batch_revision=1)
        )
        restarted_front = plane.freeze()
        restarted = restarted_front.value("@logic/revision/center")
        restarted_publication = restarted_front.publication(
            "@logic/revision/center"
        )
        assert restarted is not None and restarted_publication is not None
        assert restarted.snapshot.ref.revision.value == 1
        assert restarted_publication.direct_parent_refs == (
            restarted_source.event_ref,
        )
    finally:
        _close(bridge, plane, source)


def test_a_trailing_fit_publishes_against_the_exact_shot_it_fitted() -> None:
    """A fit accepted AFTER its source advanced still publishes, honestly.

    The fit trails its source by construction: at 10 Hz a fit for shot N can
    be accepted while the camera already published N+1.  The panel port that
    rendered the fit is the causal holder of its exact parent publication
    (the accept fires inside revision N's own commit), so the bridge asks
    the port's resolver — never a plane-side retention window — and lineage
    stays truthful: fit@N names camera@N.
    """

    schema = _curve_schema()
    values = np.zeros((1, 5, 1), dtype=np.float64)
    plane, source, slot, state, _initial = _source_setup(schema, values)
    events = _Events()
    plane.set_front_signals({"camera/frame", "@logic/trail/center"})

    # The port's role in miniature: it holds every publication it staged or
    # presented, keyed by the data revision the renderer saw.
    held: dict[tuple[str, int], object] = {}

    def remember_current() -> None:
        publication = plane.latest_publication("camera/frame")
        assert publication is not None
        value = publication.value("camera/frame")
        assert value is not None
        held[
            (
                str(value.snapshot.ref.stream_generation.value),
                value.snapshot.ref.revision.value,
            )
        ] = publication

    bridge = SelectionBridge(
        plane,
        "camera/frame",
        events,
        bridge_id="trail",
        source_publication_for=lambda generation, revision: held.get(
            (generation, revision)
        ),
    )
    bridge.start()
    try:
        remember_current()
        # The camera advances to revision 2 BEFORE the fit of revision 1
        # arrives -- the live-cadence ordering.
        state["frame"] = LiveDatasetOutput(
            state["frame"].declaration,
            _snapshot("frame", 2, schema, values + 1.0),
            MonitorCoverage(1, 5),
        )
        _commit_source(plane, source, state)
        plane.freeze()
        remember_current()

        events.emit_fit(
            _batch_fit_event(plane, source_revision=1, batch_revision=1)
        )
        assert bridge.last_error is None
        front = plane.freeze()
        published = front.value("@logic/trail/center")
        assert published is not None, "the trailing fit was dropped"
        publication = front.publication("@logic/trail/center")
        parent_ref = publication.direct_parent_refs[0]
        assert parent_ref.sequence == 1  # camera@1: the shot it was fitted on

        # The next shot's fit follows normally against camera@2.
        events.emit_fit(
            _batch_fit_event(plane, source_revision=2, batch_revision=2)
        )
        assert bridge.last_error is None
        second = plane.freeze().publication("@logic/trail/center")
        assert second.direct_parent_refs[0].sequence == 2

        # A revision no panel ever held reports, not silently vanishes.
        events.emit_fit(
            _batch_fit_event(plane, source_revision=77, batch_revision=3)
        )
        assert bridge.last_error is not None
    finally:
        _close(bridge, plane, source)


def test_added_and_updated_selection_events_do_not_publish() -> None:
    schema = _curve_schema()
    values = np.asarray([[[1.0], [2.0], [3.0], [4.0], [5.0]]])
    plane, source, _slot, _state, _initial = _source_setup(schema, values)
    events = _Events()
    plane.set_front_signals({"camera/frame", "@logic/curve/roi_mean"})
    bridge = SelectionBridge(plane, "camera/frame", events, bridge_id="curve")
    bridge.start()
    state = SelectionState(
        "curve",
        "x_range",
        (SelectionRange("x", 1.0, 3.0, domain="point_coordinate"),),
        revision=1,
    )
    try:
        events.emit_selection(SelectionChange.ADDED, state)
        events.emit_selection(
            SelectionChange.UPDATED,
            SelectionState(
                "curve",
                "x_range",
                (SelectionRange("x", 0.0, 4.0, domain="point_coordinate"),),
                revision=2,
            ),
        )
        assert plane.latest_publication("@logic/curve/roi_mean") is None
    finally:
        _close(bridge, plane, source)


def test_removed_selection_retires_derived_signals_and_unknown_axis_is_loud() -> None:
    schema = _curve_schema()
    values = np.asarray([[[1.0], [2.0], [3.0], [4.0], [5.0]]])
    plane, source, _slot, _state, _initial = _source_setup(schema, values)
    events = _Events()
    plane.set_front_signals({"camera/frame", "@logic/curve/roi_mean"})
    bridge = SelectionBridge(plane, "camera/frame", events, bridge_id="curve")
    bridge.start()
    selection = SelectionState(
        "curve",
        "x_range",
        (SelectionRange("x", 1.0, 3.0, domain="point_coordinate"),),
        revision=1,
    )
    try:
        events.emit_selection(SelectionChange.COMMITTED, selection)
        assert plane.latest_publication("@logic/curve/roi_mean") is not None
        events.emit_selection(SelectionChange.REMOVED, selection)
        plane.freeze()
        assert plane.latest_publication("@logic/curve/roi_mean") is None
        with pytest.raises(ValueError, match="not present"):
            events.emit_selection(
                SelectionChange.COMMITTED,
                SelectionState(
                    "curve",
                    "x_range",
                    (
                        SelectionRange(
                            "missing",
                            0.0,
                            1.0,
                            domain="point_coordinate",
                        ),
                    ),
                    revision=2,
                ),
            )
    finally:
        _close(bridge, plane, source)


def test_a_box_on_a_finished_run_is_answered_once() -> None:
    """The common case that used to be refused instead of answered.

    A finite measurement publishes exactly once and is terminal from the
    moment it exists, so a region boxed on it can never be a live derivation:
    no further parent publication will arrive to re-cut it.  It is still a
    real question, and it gets its answer as a terminal generation of its own
    rather than an exception about machinery the operator never asked for.
    """

    schema = _image_schema()
    values = np.arange(12, dtype=np.float64).reshape(1, 1, 4, 3)
    plane, source, slot, state_map, _initial = _source_setup(schema, values)
    plane.set_front_signals(
        {"camera/frame", "@logic/frozen/roi_frame", "@logic/frozen/roi_mean"}
    )
    _seal_source(plane, source, state_map["frame"])
    assert not plane.is_generation_live("camera/frame")

    events = _Events()
    bridge = SelectionBridge(plane, "camera/frame", events, bridge_id="frozen")
    bridge.start()
    try:
        events.emit_selection(
            SelectionChange.COMMITTED,
            SelectionState(
                "image",
                "area",
                (
                    SelectionRange("x", 0.0, 1.0, domain="data"),
                    SelectionRange("y", 20.0, 30.0, domain="data"),
                ),
                revision=1,
            ),
        )
        front = plane.freeze()
        roi_frame = front.value("@logic/frozen/roi_frame")
        roi_mean = front.value("@logic/frozen/roi_mean")
        assert roi_frame is not None, bridge.last_error
        np.testing.assert_array_equal(
            roi_frame.snapshot.block.values, values[:, :, 1:3, 1:3]
        )
        assert float(roi_mean.snapshot.block.values.reshape(-1)[0]) == 6.0
        # The answer is the same cut the live path makes; only its lifetime
        # differs, and claiming to be live would be a lie about a run that is
        # over.
        assert not plane.is_generation_live("@logic/frozen/roi_frame")
    finally:
        bridge.close()
        plane.retire(source)
        plane.close()


def test_a_second_box_on_a_finished_run_replaces_the_first() -> None:
    """Re-drawing must supersede, not collide with, the previous answer."""

    schema = _image_schema()
    values = np.arange(12, dtype=np.float64).reshape(1, 1, 4, 3)
    plane, source, _slot, state_map, _initial = _source_setup(schema, values)
    plane.set_front_signals(
        {"camera/frame", "@logic/frozen/roi_frame", "@logic/frozen/roi_mean"}
    )
    _seal_source(plane, source, state_map["frame"])
    events = _Events()
    bridge = SelectionBridge(plane, "camera/frame", events, bridge_id="frozen")
    bridge.start()
    try:
        for revision, upper in ((1, 1.0), (2, 2.0)):
            events.emit_selection(
                SelectionChange.COMMITTED,
                SelectionState(
                    "image",
                    "area",
                    (
                        SelectionRange("x", 0.0, upper, domain="data"),
                        SelectionRange("y", 20.0, 30.0, domain="data"),
                    ),
                    revision=revision,
                ),
            )
        roi_frame = plane.freeze().value("@logic/frozen/roi_frame")
        assert roi_frame is not None, bridge.last_error
        np.testing.assert_array_equal(
            roi_frame.snapshot.block.values, values[:, :, 1:4, 1:3]
        )
    finally:
        bridge.close()
        plane.retire(source)
        plane.close()


def test_the_plane_can_say_who_is_producing_what() -> None:
    """The list nothing could ask for before.

    Every chooser, legend and status line needs "what exists" before it can
    need "what is in this one signal", and the plane holds the only complete
    answer.  It answers in copies: a view handed the live state would read it
    at whatever moment it painted and show a mixture of two instants.
    """

    schema = _image_schema()
    values = np.arange(12, dtype=np.float64).reshape(1, 1, 4, 3)
    plane, source, _slot, _state, _initial = _source_setup(schema, values)
    plane.set_front_signals({"camera/frame", "@logic/topology/roi_frame"})
    events = _Events()
    bridge = SelectionBridge(plane, "camera/frame", events, bridge_id="topology")
    bridge.start()
    try:
        described = {item.name: item for item in plane.describe_signals()}
        assert "camera/frame" in described
        frame = described["camera/frame"]
        assert frame.owner_id == "camera"
        assert frame.kind == "producer"
        assert frame.live and not frame.derived
        assert frame.shape == (1, 1, 4, 3)

        events.emit_selection(
            SelectionChange.COMMITTED,
            SelectionState(
                "image",
                "area",
                (
                    SelectionRange("x", 0.0, 1.0, domain="data"),
                    SelectionRange("y", 20.0, 30.0, domain="data"),
                ),
                revision=1,
            ),
        )
        derived = {
            item.name: item
            for item in plane.describe_signals()
            if item.name.startswith("@logic/topology/")
        }
        assert derived, "a derived signal is invisible to anything offering a choice"
        for item in derived.values():
            assert item.derived and item.source_name == "camera/frame"

        # Copies, not windows.  A description already handed out must keep
        # describing the moment it was taken, and must not be writable by the
        # view holding it.
        before = {item.name: item for item in plane.describe_signals()}
        with pytest.raises((AttributeError, TypeError)):
            before["camera/frame"].live = False
        events.emit_selection(
            SelectionChange.COMMITTED,
            SelectionState(
                "image",
                "area",
                (
                    SelectionRange("x", 0.0, 2.0, domain="data"),
                    SelectionRange("y", 20.0, 30.0, domain="data"),
                ),
                revision=2,
            ),
        )
        assert before["camera/frame"].live is True
    finally:
        bridge.close()
        plane.retire(source)
        plane.close()


def test_a_contiguous_roi_is_a_view_not_four_gathers() -> None:
    """resolve_selection_indices returns a range so a contiguous run need not
    be expanded.  Flattening it to a tuple made every axis a gather -- a full
    copy, per axis, including the axes nobody selected -- so a 512x512 box on a
    1200x1920 frame cost milliseconds of pure copying per published frame."""

    import numpy as np

    from zlc_data.snapshot_projection import restricted_values

    values = np.zeros((1, 1, 120, 160), dtype=np.uint16)

    class _Schema:
        class repeat_axis:
            axis_id = "r"

        class cell_schema:
            data_axes = ()

    taken = restricted_values(values, _Schema, range(0, 1), range(0, 1), {})

    assert taken.base is not None, "a contiguous selection copied the frame"
    assert taken.shape == (1, 1, 120, 160)


def _heatmap_schema(repeats: int = 2, *, with_columns: bool = True) -> DatasetSchema:
    """A 3x3 scan grid: 9 point rows over dimensions (bias_x, grad)."""

    repeat = AxisSpec(
        AxisId("repeat"), "repeat", REPEAT, repeats, tuple(range(repeats))
    )
    columns: tuple[PointColumn, ...] = ()
    if with_columns:
        columns = (
            PointColumn(
                AxisId("bias_x"),
                "bias_x",
                SCAN_POINT,
                PointColumn.NUMERIC,
                (-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
            ),
            PointColumn(
                AxisId("grad"),
                "grad",
                SCAN_POINT,
                PointColumn.NUMERIC,
                (10.0, 20.0, 30.0, 10.0, 20.0, 30.0, 10.0, 20.0, 30.0),
            ),
        )
    topology = GridTopology(
        (AxisId("bias_x"), AxisId("grad")),
        ((-1.0, 0.0, 1.0), (10.0, 20.0, 30.0)),
        tuple((i, j) for i in range(3) for j in range(3)),
    )
    return DatasetSchema(
        repeat,
        PointTable(9, columns),
        topology,
        ValueSchema.scalar(np.dtype("float64"), None),
    )


def test_selection_derives_from_incremental_scan_points_in_the_canonical_grid() -> None:
    canonical_schema = _heatmap_schema(repeats=1)
    event_schema = DatasetSchema(
        canonical_schema.repeat_axis,
        PointTable(1),
        None,
        canonical_schema.cell_schema,
    )
    plane, source, state, _initial = _finite_source_setup(
        canonical_schema,
        event_schema,
        np.asarray([[[10.0]]]),
        origin=(0, 0),
    )
    events = _Events()
    signal = "@logic/point-prefix/roi_frame"
    plane.set_front_signals({"camera/frame", signal})
    bridge = SelectionBridge(
        plane,
        "camera/frame",
        events,
        bridge_id="point-prefix",
    )
    bridge.start()
    selection = SelectionState(
        "image",
        "area",
        (
            SelectionRange("grad", 10.0, 30.0, domain="point_dimension"),
            SelectionRange("bias_x", -1.0, 1.0, domain="point_dimension"),
        ),
        revision=1,
    )
    try:
        events.emit_selection(SelectionChange.COMMITTED, selection)
        first = plane.freeze().value(signal)
        assert first is not None, bridge.last_error
        assert first.snapshot.block.values.shape == (1, 9, 1)
        assert first.snapshot.block.schema.grid_topology == canonical_schema.grid_topology
        np.testing.assert_array_equal(
            first.snapshot.expanded_validity()[..., 0],
            np.asarray([[True, False, False, False, False, False, False, False, False]]),
        )

        state["frame"] = LiveDatasetOutput(
            state["frame"].declaration,
            _snapshot(
                "frame",
                2,
                event_schema,
                np.asarray([[[44.0]]]),
            ),
            DatasetCoverage(2, 9),
            canonical_schema=canonical_schema,
            cell_origin=(0, 4),
        )
        _commit_source(plane, source, state)
        front = _wait_for_signal(plane, signal, 2)
        derived = front.value(signal)
        assert derived is not None, bridge.last_error
        assert float(derived.snapshot.block.values[0, 0, 0]) == 10.0
        assert float(derived.snapshot.block.values[0, 4, 0]) == 44.0
        np.testing.assert_array_equal(
            derived.snapshot.expanded_validity()[..., 0],
            np.asarray([[True, False, False, False, True, False, False, False, False]]),
        )
    finally:
        _close(bridge, plane, source)


def test_image_area_over_grid_dimensions_cuts_the_point_rows() -> None:
    """A box on a scan heatmap selects the sub-grid, not an exception.

    The heatmap cell's axes are grid-topology DIMENSIONS -- point-row
    quantities -- and the bridge used to refuse them with 'image area axes
    must be source data axes', so every committed box on a scan heatmap
    silently derived nothing.
    """

    schema = _heatmap_schema()
    values = np.arange(18, dtype=np.float64).reshape(2, 9, 1)
    plane, source, _slot, _state, _initial = _source_setup(schema, values)
    events = _Events()
    plane.set_front_signals(
        {"camera/frame", "@logic/grid/roi_frame", "@logic/grid/roi_mean"}
    )
    bridge = SelectionBridge(plane, "camera/frame", events, bridge_id="grid")
    bridge.start()
    try:
        events.emit_selection(
            SelectionChange.COMMITTED,
            SelectionState(
                "image",
                "area",
                (
                    SelectionRange(
                        "grad", 15.0, 25.0, domain="point_dimension"
                    ),
                    SelectionRange(
                        "bias_x", -0.5, 1.5, domain="point_dimension"
                    ),
                ),
                revision=1,
            ),
        )
        front = plane.freeze()
        roi_frame = front.value("@logic/grid/roi_frame")
        roi_mean = front.value("@logic/grid/roi_mean")
        assert roi_frame is not None, bridge.last_error
        assert roi_mean is not None
        # Rows inside BOTH ranges: grad == 20 and bias_x in {0, 1} -> rows 4, 7.
        np.testing.assert_array_equal(
            roi_frame.snapshot.block.values, values[:, (4, 7)]
        )
        derived_schema = roi_frame.snapshot.block.schema
        assert derived_schema.point_table.row_count == 2
        assert derived_schema.grid_topology is not None
        assert len(derived_schema.grid_topology.row_to_cell) == 2
        # The sub-grid keeps both surviving scan points and both repeats; a
        # cell here is already scalar, so the mean consumes nothing.
        np.testing.assert_array_equal(
            roi_mean.snapshot.block.values, values[:, (4, 7)]
        )
        assert roi_mean.snapshot.block.schema.point_table == derived_schema.point_table
    finally:
        _close(bridge, plane, source)


def test_repeat_index_narrows_a_grid_cut_to_the_focused_repeat() -> None:
    """The repeat restriction is structural: row k of dimension 0, no name.

    The repeat axis is never name-addressed anywhere -- it is identified by
    its role and position -- so a focused repeat facet crosses the bridge as
    ``repeat_index`` and the derived data is exactly the k-th repeat slice.
    """

    schema = _heatmap_schema()
    values = np.arange(18, dtype=np.float64).reshape(2, 9, 1)
    plane, source, _slot, _state, _initial = _source_setup(schema, values)
    events = _Events()
    plane.set_front_signals(
        {"camera/frame", "@logic/focus/roi_frame", "@logic/focus/roi_mean"}
    )
    bridge = SelectionBridge(plane, "camera/frame", events, bridge_id="focus")
    bridge.start()
    try:
        events.emit_selection(
            SelectionChange.COMMITTED,
            SelectionState(
                "image",
                "area",
                (
                    SelectionRange(
                        "grad", 15.0, 25.0, domain="point_dimension"
                    ),
                    SelectionRange(
                        "bias_x", -0.5, 0.5, domain="point_dimension"
                    ),
                ),
                repeat_index=1,
                revision=1,
            ),
        )
        front = plane.freeze()
        roi_frame = front.value("@logic/focus/roi_frame")
        roi_mean = front.value("@logic/focus/roi_mean")
        assert roi_frame is not None, bridge.last_error
        # The focused slice only: repeat 1, row 4 (grad 20, bias_x 0).
        np.testing.assert_array_equal(
            roi_frame.snapshot.block.values, values[1:2, 4:5]
        )
        assert roi_frame.snapshot.block.schema.repeat_axis.size == 1
        assert float(roi_mean.snapshot.block.values.reshape(-1)[0]) == 13.0

        # A row the source does not have fails loudly, not as an empty cut.
        with pytest.raises(IndexError):
            events.emit_selection(
                SelectionChange.COMMITTED,
                SelectionState(
                    "image",
                    "area",
                    (
                    SelectionRange(
                        "grad", 15.0, 25.0, domain="point_dimension"
                    ),
                    SelectionRange(
                        "bias_x", -0.5, 0.5, domain="point_dimension"
                    ),
                    ),
                    repeat_index=7,
                    revision=2,
                ),
            )
    finally:
        _close(bridge, plane, source)


def test_grid_dimensions_resolve_without_matching_point_columns() -> None:
    """The topology itself declares the dimensions; columns are optional."""

    schema = _heatmap_schema(with_columns=False)
    values = np.arange(18, dtype=np.float64).reshape(2, 9, 1)
    plane, source, _slot, _state, _initial = _source_setup(schema, values)
    events = _Events()
    plane.set_front_signals({"camera/frame", "@logic/bare/roi_mean"})
    bridge = SelectionBridge(plane, "camera/frame", events, bridge_id="bare")
    bridge.start()
    try:
        events.emit_selection(
            SelectionChange.COMMITTED,
            SelectionState(
                "image",
                "area",
                (
                    SelectionRange(
                        "grad", 15.0, 25.0, domain="point_dimension"
                    ),
                    SelectionRange(
                        "bias_x", -0.5, 1.5, domain="point_dimension"
                    ),
                ),
                revision=1,
            ),
        )
        roi_mean = plane.freeze().value("@logic/bare/roi_mean")
        assert roi_mean is not None, bridge.last_error
        np.testing.assert_array_equal(
            roi_mean.snapshot.block.values, values[:, (4, 7)]
        )
    finally:
        _close(bridge, plane, source)


def _frames_on_point_axis_schema(cycles: int = 2, frames: int = 3) -> DatasetSchema:
    """Camera frames use (cycles, frame POINTS, y, x)."""

    repeat = AxisSpec(
        AxisId("cycle"), "cycle", REPEAT, cycles, tuple(range(cycles))
    )
    frame = PointColumn(
        AxisId("frame"),
        "frame",
        READOUT_EVENT,
        PointColumn.NUMERIC,
        tuple(float(index) for index in range(frames)),
    )
    y = AxisSpec(AxisId("y"), "y", SPATIAL_Y, 4, (0.0, 1.0, 2.0, 3.0))
    x = AxisSpec(AxisId("x"), "x", SPATIAL_X, 5, (0.0, 1.0, 2.0, 3.0, 4.0))
    return DatasetSchema(
        repeat,
        PointTable(frames, (frame,)),
        None,
        ValueSchema((y, x), ValidityContract.value(), np.dtype("float64"), "counts"),
    )


def test_frames_on_point_axis_keep_deriving_and_facet_by_frame() -> None:
    """Point rows represent frames.

    An image cell over the DATA spatial axes must still cut spatially, keep
    every frame row, and a facet condition on the 'frame' point column must
    select exactly one frame's rows.
    """

    schema = _frames_on_point_axis_schema()
    values = np.arange(120, dtype=np.float64).reshape(2, 3, 4, 5)
    plane, source, _slot, _state, _initial = _source_setup(schema, values)
    events = _Events()
    plane.set_front_signals(
        {"camera/frame", "@logic/frames/roi_frame", "@logic/frames/roi_mean"}
    )
    bridge = SelectionBridge(plane, "camera/frame", events, bridge_id="frames")
    bridge.start()
    try:
        events.emit_selection(
            SelectionChange.COMMITTED,
            SelectionState(
                "image",
                "area",
                (
                    SelectionRange("x", 1.0, 3.0, domain="data"),
                    SelectionRange("y", 1.0, 2.0, domain="data"),
                ),
                revision=1,
            ),
        )
        roi_frame = plane.freeze().value("@logic/frames/roi_frame")
        assert roi_frame is not None, bridge.last_error
        np.testing.assert_array_equal(
            roi_frame.snapshot.block.values, values[:, :, 1:3, 1:4]
        )
        assert roi_frame.snapshot.block.schema.point_table.row_count == 3

        events.emit_selection(
            SelectionChange.COMMITTED,
            SelectionState(
                "image",
                "area",
                (
                    SelectionRange("x", 1.0, 3.0, domain="data"),
                    SelectionRange("y", 1.0, 2.0, domain="data"),
                ),
                (FacetCondition("frame", 1.0, "point_coordinate"),),
                revision=2,
            ),
        )
        focused = plane.freeze().value("@logic/frames/roi_frame")
        assert focused is not None, bridge.last_error
        np.testing.assert_array_equal(
            focused.snapshot.block.values, values[:, 1:2, 1:3, 1:4]
        )
        column = focused.snapshot.block.schema.point_table.columns[0]
        assert column.name == "frame"
        assert column.values == (1.0,)
    finally:
        _close(bridge, plane, source)


def test_roi_mean_keeps_one_value_per_frame_point() -> None:
    """A box consumes the IMAGE axes; it does not pool a cycle's frames.

    The frames of one acquisition cycle are POINTS -- they fire at different
    moments of the pulse, so their physics differs.  ``roi_mean`` used to
    average the whole point axis into a single scalar, so a scan watching it
    silently mixed physically distinct frames into one number.
    """

    schema = _frames_on_point_axis_schema()
    values = np.arange(120, dtype=np.float64).reshape(2, 3, 4, 5)
    plane, source, _slot, _state, _initial = _source_setup(schema, values)
    events = _Events()
    plane.set_front_signals(
        {"camera/frame", "@logic/mean/roi_frame", "@logic/mean/roi_mean"}
    )
    bridge = SelectionBridge(plane, "camera/frame", events, bridge_id="mean")
    bridge.start()
    try:
        events.emit_selection(
            SelectionChange.COMMITTED,
            SelectionState(
                "image",
                "area",
                (
                    SelectionRange("x", 1.0, 3.0, domain="data"),
                    SelectionRange("y", 1.0, 2.0, domain="data"),
                ),
                revision=1,
            ),
        )
        front = plane.freeze()
        roi_frame = front.value("@logic/mean/roi_frame")
        roi_mean = front.value("@logic/mean/roi_mean")
        assert roi_frame is not None, bridge.last_error
        assert roi_mean is not None

        # One value per (cycle, frame): the two repeats and the three frame
        # points survive; only the two image data axes are consumed.
        assert roi_mean.snapshot.block.values.shape == (2, 3, 1)
        np.testing.assert_allclose(
            roi_mean.snapshot.block.values.reshape(2, 3),
            values[:, :, 1:3, 1:4].mean(axis=(2, 3)),
        )

        mean_schema = roi_mean.snapshot.block.schema
        frame_schema = roi_frame.snapshot.block.schema
        # The point column is the PARENT's frame column, not a fresh axis.
        assert mean_schema.point_table == frame_schema.point_table
        (column,) = mean_schema.point_table.columns
        assert column.name == "frame"
        assert column.role == READOUT_EVENT
        assert column.values == (0.0, 1.0, 2.0)
        assert mean_schema.repeat_axis == frame_schema.repeat_axis
        assert mean_schema.cell_schema.is_scalar
        assert mean_schema.cell_schema.value_unit == "counts"
    finally:
        _close(bridge, plane, source)


def test_roi_mean_invalidity_is_per_point_not_pooled() -> None:
    """One unusable frame invalidates its own point, not the whole answer."""

    schema = _frames_on_point_axis_schema(cycles=1, frames=2)
    values = np.arange(40, dtype=np.float64).reshape(1, 2, 4, 5)
    values[0, 1] = np.nan
    plane, source, _slot, _state, _initial = _source_setup(schema, values)
    events = _Events()
    plane.set_front_signals({"camera/frame", "@logic/partial/roi_mean"})
    bridge = SelectionBridge(
        plane, "camera/frame", events, bridge_id="partial"
    )
    bridge.start()
    try:
        events.emit_selection(
            SelectionChange.COMMITTED,
            SelectionState(
                "image",
                "area",
                (
                    SelectionRange("x", 1.0, 3.0, domain="data"),
                    SelectionRange("y", 1.0, 2.0, domain="data"),
                ),
                revision=1,
            ),
        )
        roi_mean = plane.freeze().value("@logic/partial/roi_mean")
        assert roi_mean is not None, bridge.last_error
        block = roi_mean.snapshot.block
        assert block.values.shape == (1, 2, 1)
        assert float(block.values[0, 0, 0]) == float(values[0, 0, 1:3, 1:4].mean())
        np.testing.assert_array_equal(
            block.validity.mask, np.asarray([[True, False]])
        )
    finally:
        _close(bridge, plane, source)


def _frame_faceted_fit(
    plane: SignalDataPlane,
    sample_axis_name: str,
    sample_coordinates: np.ndarray,
    *,
    source_revision: int = 1,
    batch_revision: int = 1,
) -> FitEventValue:
    count = int(sample_coordinates.size)
    sample_domain = "repeat" if sample_axis_name == "cycle" else "point_coordinate"
    sample_axis_id = "" if sample_domain == "repeat" else sample_axis_name
    return FitEventValue(
        parameter_names=("center",),
        parameter_units={"center": "pixel"},
        parameter_values={"center": np.arange(count, dtype=np.float64)},
        parameter_errors={"center": np.full(count, 0.5)},
        success=np.ones(count, dtype=np.bool_),
        sample_axis_domain=sample_domain,
        sample_axis_id=sample_axis_id,
        sample_axis_name=sample_axis_name,
        sample_coordinates=sample_coordinates,
        sample_unit="",
        sample_labels=None,
        source_generation=_source_generation(plane),
        source_revision=source_revision,
        batch_revision=batch_revision,
    )


def test_a_faceted_fit_takes_its_sample_role_from_the_axis_it_was_cut_along() -> None:
    """A fit's samples mean whatever the axis they were faceted over means.

    Every fit used to publish SCAN_POINT, so a fit over a camera cycle's
    frames claimed the experiment had scanned over them.
    """

    schema = _frames_on_point_axis_schema()
    values = np.arange(120, dtype=np.float64).reshape(2, 3, 4, 5)
    plane, source, _slot, _state, _initial = _source_setup(schema, values)
    events = _Events()
    plane.set_front_signals(
        {"camera/frame", "@logic/perframe/center", "@logic/perframe/center_err"}
    )
    bridge = SelectionBridge(
        plane, "camera/frame", events, bridge_id="perframe"
    )
    bridge.start()
    try:
        events.emit_fit(
            _frame_faceted_fit(
                plane,
                "frame",
                np.asarray([0.0, 1.0, 2.0]),
            )
        )
        value = plane.freeze().value("@logic/perframe/center")
        assert value is not None, bridge.last_error
        (column,) = value.snapshot.block.schema.point_table.columns
        assert column.name == "frame"
        assert column.role == READOUT_EVENT
        assert value.snapshot.block.values.shape == (1, 3, 1)
    finally:
        _close(bridge, plane, source)


def test_a_scan_faceted_fit_still_publishes_a_scan_point_column() -> None:
    """The scan case is inherited from the parent column, not hardcoded."""

    schema = _curve_schema()
    values = np.zeros((1, 5, 1), dtype=np.float64)
    plane, source, _slot, _state, _initial = _source_setup(schema, values)
    events = _Events()
    plane.set_front_signals(
        {"camera/frame", "@logic/perscan/center", "@logic/perscan/center_err"}
    )
    bridge = SelectionBridge(
        plane, "camera/frame", events, bridge_id="perscan"
    )
    bridge.start()
    try:
        events.emit_fit(
            _frame_faceted_fit(
                plane,
                "x",
                np.asarray([0.0, 1.0, 2.0]),
            )
        )
        value = plane.freeze().value("@logic/perscan/center")
        assert value is not None, bridge.last_error
        (column,) = value.snapshot.block.schema.point_table.columns
        assert column.name == "x"
        assert column.role == SCAN_POINT
    finally:
        _close(bridge, plane, source)


def test_a_repeat_faceted_fit_keeps_the_repeat_identity() -> None:
    """Samples that ARE repeats stay on the repeat axis.

    A repeat is the same conditions measured again, so restating it as a
    point row would claim the experiment varied it.
    """

    schema = _frames_on_point_axis_schema()
    values = np.arange(120, dtype=np.float64).reshape(2, 3, 4, 5)
    plane, source, _slot, _state, _initial = _source_setup(schema, values)
    events = _Events()
    plane.set_front_signals(
        {"camera/frame", "@logic/percycle/center", "@logic/percycle/center_err"}
    )
    bridge = SelectionBridge(
        plane, "camera/frame", events, bridge_id="percycle"
    )
    bridge.start()
    try:
        events.emit_fit(
            _frame_faceted_fit(
                plane,
                "cycle",
                np.asarray([0.0, 1.0]),
            )
        )
        value = plane.freeze().value("@logic/percycle/center")
        assert value is not None, bridge.last_error
        fit_schema = value.snapshot.block.schema
        assert fit_schema.repeat_axis.axis_id == AxisId("cycle")
        assert fit_schema.repeat_axis.size == 2
        assert fit_schema.repeat_axis.coordinates == (0, 1)
        assert fit_schema.point_table.row_count == 1
        assert fit_schema.point_table.columns == ()
        assert value.snapshot.block.values.shape == (2, 1, 1)
        np.testing.assert_array_equal(
            value.snapshot.block.values.reshape(-1), np.asarray([0.0, 1.0])
        )
    finally:
        _close(bridge, plane, source)


def test_a_mixed_kind_image_area_is_refused_loudly() -> None:
    """One data axis plus one point axis is no image surface anywhere."""

    schema = _frames_on_point_axis_schema()
    values = np.arange(120, dtype=np.float64).reshape(2, 3, 4, 5)
    plane, source, _slot, _state, _initial = _source_setup(schema, values)
    events = _Events()
    plane.set_front_signals({"camera/frame", "@logic/mixed/roi_mean"})
    bridge = SelectionBridge(plane, "camera/frame", events, bridge_id="mixed")
    bridge.start()
    try:
        with pytest.raises(ValueError, match="both be"):
            events.emit_selection(
                SelectionChange.COMMITTED,
                SelectionState(
                    "image",
                    "area",
                    (
                        SelectionRange("x", 1.0, 3.0, domain="data"),
                        SelectionRange(
                            "frame", 0.0, 1.0, domain="point_coordinate"
                        ),
                    ),
                    revision=1,
                ),
            )
    finally:
        _close(bridge, plane, source)


def test_an_implicit_axis_cropped_to_a_run_stays_implicit() -> None:
    """Writing the coordinates out here undid, one layer down, exactly what the
    producer stopped doing: build a tuple, validate every element, digest all of
    it -- per frame, for an axis that is still just indexed 0..n-1 from later."""

    from zlc_data import SPATIAL_X
    from zlc_data.snapshot_projection import restricted_schema
    from zlc_data.axis import AxisId, AxisSpec
    from zlc_data.schema import DatasetSchema, PointTable, ValueSchema
    from zlc_data.validity import ValidityContract

    axis = AxisSpec(AxisId("cam.x"), "x", SPATIAL_X, 1920)
    schema = DatasetSchema(
        AxisSpec(AxisId("r"), "repeat", REPEAT, 1),
        PointTable(1),
        None,
        ValueSchema((axis,), ValidityContract.value(), np.dtype("uint16")),
    )

    cropped = restricted_schema(
        schema, range(1), range(1), {axis.axis_id: range(400, 912)}
    ).cell_schema.data_axes[0]

    assert cropped.coordinates is None
    assert cropped.size == 512
    assert cropped.index_origin == 400
    assert cropped.coordinate_at(0) == 400


def test_two_fit_events_in_flight_never_own_one_name_twice() -> None:
    """A fit publication CLAIMS its names before it knows it is still wanted.

    Attaching (or reserving) registers the output names in the plane, and
    only afterwards does the bridge decide whether that claim survives.
    Two events in flight -- an operator arming a fit while the shot the
    plot was still fitting lands -- therefore both held the same names for
    a moment, and the plane refused the second one:

        signal '@logic/fit/x0' is already owned by 'fit:fit:1'

    Recorded per event and re-reported every beat, that is a panel whose
    fit is broken and stays broken.  One route per role means publishing
    one is not a concurrent operation.
    """

    schema = _curve_schema()
    values = np.asarray([[[0.0], [0.0], [0.0], [0.0], [0.0]]])
    plane, source, _slot, _state, _initial = _source_setup(schema, values)
    plane.set_front_signals(
        {"camera/frame", "@logic/fit/x0", "@logic/fit/x0_err"}
    )
    events = _Events()
    bridge = SelectionBridge(plane, "camera/frame", events, bridge_id="fit")
    bridge.start()
    claimed, release = Event(), Event()
    original = plane.attach_latest_only_processor

    def holding_attach(processor, **kwargs):
        result = original(processor, **kwargs)
        claimed.set()
        release.wait(20.0)
        return result

    plane.attach_latest_only_processor = holding_attach
    failures: list[BaseException] = []

    def emit(value: float, error: float, revision: int) -> None:
        try:
            events.emit_fit(_scalar_fit_event(plane, value, error, revision))
        except BaseException as caught:  # pragma: no cover - the defect
            failures.append(caught)

    first = Thread(target=emit, args=(2.5, 0.1, 1))
    first.start()
    try:
        assert claimed.wait(20.0), "the first fit never claimed its names"
        second = Thread(target=emit, args=(3.5, 0.2, 2))
        second.start()
        # The second must be WAITING for the first to finish, not racing
        # it: without that it claims '@logic/fit/x0' while the first still
        # holds it, and the plane refuses the fit outright.
        second.join(0.5)
        assert second.is_alive()
    finally:
        release.set()
        first.join(20.0)
    second.join(20.0)
    plane.attach_latest_only_processor = original
    try:
        assert not failures, failures
        assert bridge.last_error is None, bridge.last_error
        parameter = plane.freeze().value("@logic/fit/x0")
        assert parameter is not None
        assert float(parameter.snapshot.block.values.reshape(-1)[0]) == 3.5
    finally:
        _close(bridge, plane, source)


def test_retiring_a_fit_route_frees_its_names_before_the_slot_reads_empty() -> None:
    """The RELEASE half of one claim, under the claim's own lock.

    A fit whose sample coordinates moved -- an ordinary shot -- replaces
    its route: the bridge empties the slot and withdraws the processor
    holding the names.  Done above the publish lock, those two steps are
    not one operation, and a second event arriving between them reads an
    empty slot while the plane still holds the names:

        signal '@logic/fit/x0' is already owned by 'fit:fit:1'

    The claim side was serialized and the ``_release_route`` helper was
    taught the same lock; this release, written inline in the publish
    path, was not.  It is the one an operator meets, because it fires
    whenever the fitted samples change.
    """

    schema = _curve_schema()
    values = np.asarray([[[0.0], [0.0], [0.0], [0.0], [0.0]]])
    plane, source, _slot, _state, _initial = _source_setup(schema, values)
    plane.set_front_signals(
        {"camera/frame", "@logic/fit/x0", "@logic/fit/x0_err"}
    )
    events = _Events()
    bridge = SelectionBridge(plane, "camera/frame", events, bridge_id="fit")
    bridge.start()

    def moved(value: float, revision: int, coordinate: float) -> FitEventValue:
        """The same fit, one shot later: same names, moved samples."""

        event = _scalar_fit_event(plane, value, 0.1, revision)
        return replace(event, sample_coordinates=np.asarray([coordinate]))

    events.emit_fit(_scalar_fit_event(plane, 2.5, 0.1, 1))
    assert bridge.last_error is None, bridge.last_error
    assert plane.freeze().value("@logic/fit/x0") is not None

    withdrawing, proceed = Event(), Event()
    original = bridge._withdraw_processor

    def holding_withdraw(processor):
        withdrawing.set()
        proceed.wait(20.0)
        return original(processor)

    bridge._withdraw_processor = holding_withdraw
    failures: list[BaseException] = []

    def emit(event: FitEventValue) -> None:
        try:
            events.emit_fit(event)
        except BaseException as caught:  # pragma: no cover - the defect
            failures.append(caught)

    first = Thread(target=emit, args=(moved(3.5, 2, 1.0),))
    first.start()
    try:
        assert withdrawing.wait(20.0), "the replaced route never withdrew"
        second = Thread(target=emit, args=(moved(4.5, 3, 1.0),))
        second.start()
        # The second must WAIT for the names to be free, not attach over
        # them: the slot being empty is not the same fact as the plane
        # having let them go.
        second.join(0.5)
        assert second.is_alive(), (
            "a second fit claimed names the retiring route still held"
        )
    finally:
        proceed.set()
        first.join(20.0)
    second.join(20.0)
    bridge._withdraw_processor = original
    try:
        assert not failures, failures
        assert bridge.last_error is None, bridge.last_error
        parameter = plane.freeze().value("@logic/fit/x0")
        assert parameter is not None
        assert float(parameter.snapshot.block.values.reshape(-1)[0]) == 4.5
    finally:
        _close(bridge, plane, source)


def test_a_box_drawn_on_a_retired_run_answers_instead_of_raising() -> None:
    """A panel outlives its run, and its selectors go with it.

    The picture stays on the card after the camera measurement ends and
    its data is retired; the box the operator then drags asks the plane
    for a dataset it no longer holds.  Asked as an exception, the answer
    arrived as ``LookupError`` -- a class the console's interaction drain
    did not name -- and left a Qt slot, which is where PyQt ends the
    process.  It is not a defect at all: it is what "this run is gone"
    looks like from inside a derivation, and it is reported.
    """

    schema = _image_schema()
    values = np.arange(12, dtype=float).reshape(1, 1, 4, 3)
    plane, source, _slot, _state, initial = _source_setup(schema, values)
    plane.set_front_signals({"camera/frame", "@logic/box/roi_frame"})
    events = _Events()
    bridge = SelectionBridge(plane, "camera/frame", events, bridge_id="box")
    bridge.start()
    try:
        assert plane.retains("camera/frame")
        assert plane.is_generation_live("camera/frame")

        publication = initial.publication("camera/frame")
        assert publication is not None
        plane.retire(source)
        # The two questions a derivation has to tell apart: a run that
        # ENDED still holds its data; a run that was retired does not.
        assert not plane.is_generation_live("camera/frame")
        assert not plane.retains("camera/frame")

        bridge.commit_selection(
            SelectionState(
                "image",
                "area",
                (
                    SelectionRange("x", 0.0, 1.0, domain="data"),
                    SelectionRange("y", 20.0, 30.0, domain="data"),
                ),
                revision=1,
            ),
            source_publication=publication,
        )
        assert bridge.last_error is not None
        assert "no longer held" in str(bridge.last_error)
        assert plane.freeze().value("@logic/box/roi_frame") is None
    finally:
        bridge.close()
        plane.close()


def _component_schema() -> DatasetSchema:
    """A survival-shaped parent: its data axes carry COMPONENT and SITE."""

    repeat = AxisSpec(AxisId("repeat"), "repeat", REPEAT, 1, (0,))
    pair = AxisSpec(
        AxisId("pair"), "pair", COMPONENT, 3,
        coordinate_labels=("0-1", "0-2", "1-2"),
    )
    site = AxisSpec(AxisId("site"), "site", SITE, 2, (0, 1))
    cell = ValueSchema(
        (pair, site),
        ValidityContract.components(pair.axis_id, site.axis_id),
        np.dtype("float64"),
        "1",
    )
    return DatasetSchema(repeat, PointTable(4), None, cell)


def test_a_fit_faceted_over_a_component_axis_publishes() -> None:
    """A grid may be faceted over ANY axis; the fit must survive that.

    The sample column inherits the faceted axis's ROLE, but the point-row
    domain admits a narrower vocabulary than the axis vocabulary as a
    whole -- a component, like a repeat or the implicit scalar, is not an
    independent variable.  Inheriting one raised "point column role is
    outside the point-domain role set" from inside the fit, where an
    operator reads it as the fit being broken.
    """

    schema = _component_schema()
    values = np.zeros((1, 4, 3, 2), dtype=np.float64)
    plane, source, _slot, _state, _initial = _source_setup(schema, values)
    plane.set_front_signals(
        {"camera/frame", "@logic/fit/center", "@logic/fit/center_err",
         "@logic/fit/width", "@logic/fit/width_err"}
    )
    events = _Events()
    bridge = SelectionBridge(plane, "camera/frame", events, bridge_id="fit")
    bridge.start()
    try:
        event = replace(
            _batch_fit_event(plane, source_revision=1),
            sample_axis_domain="data",
            sample_axis_id="pair",
            sample_axis_name="pair",
            sample_labels=("0-1", "0-2", "1-2"),
            sample_unit="",
            sample_coordinates=np.asarray([0.0, 1.0, 2.0]),
        )
        events.emit_fit(event)
        assert bridge.last_error is None, bridge.last_error
        published = plane.freeze().value("@logic/fit/center")
        assert published is not None
        column = published.snapshot.block.schema.point_table.columns[0]
        assert column.name == "pair"
        # No role to inherit: the point ordinal's own role stands in.
        assert column.role == SCAN_POINT
    finally:
        _close(bridge, plane, source)


def test_the_histogram_summary_gives_the_array_path_s_numbers_exactly() -> None:
    """Same region, same five scalars, whichever way they are counted.

    Two of the catalogue's scalars -- the mean of the ten smallest samples
    and of the ten largest -- were each answered by partitioning the whole
    region, so ten numbers cost two full reorderings of it.  On a 346x345
    camera window that was 95 per cent of the reduction, and on a
    whole-frame selection twenty of its twenty-two milliseconds.

    A camera frame is small unsigned integers, so one ``bincount`` pass
    answers all five.  The array path stays for every other dtype and is
    the specification here: the two must agree EXACTLY, ties, degenerate
    sizes and single-valued regions included.
    """

    from zlc_runtime.selection_bridge import _Sample

    rng = np.random.default_rng(11)
    questions = ("mean", "minimum", "maximum", "bottom_mean", "top_mean")
    regions = [
        np.array([7], dtype=np.uint8),
        np.full(5, 3, dtype=np.uint8),
        np.full(2048, 200, dtype=np.uint8),
        np.arange(10, dtype=np.uint8),
        np.array([0, 255] * 7, dtype=np.uint8),
        np.zeros(1000, dtype=np.uint8),
        np.full(30, 65535, dtype=np.uint16),
        rng.integers(0, 4096, size=5000, dtype=np.uint16),
    ]
    regions += [
        rng.integers(0, int(rng.integers(2, 256)),
                     size=int(rng.integers(1, 3000)), dtype=np.uint8)
        for _ in range(60)
    ]
    for region in regions:
        counted = _Sample(region)
        assert counted._counts is not None, (
            "the histogram path must engage for %s" % region.dtype
        )
        # The same values as floats take the array path by construction.
        measured = _Sample(region.astype(np.float64))
        assert measured._counts is None
        for question in questions:
            assert getattr(counted, question)() == getattr(measured, question)(), (
                "%s disagrees on a %d-sample %s region"
                % (question, region.size, region.dtype)
            )
