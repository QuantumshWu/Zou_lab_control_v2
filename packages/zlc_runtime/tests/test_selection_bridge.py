from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Event
import time

import numpy as np
import pytest
import zlc_runtime.selection_bridge as selection_bridge_module

from zlc_data import (
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
    OwnedSnapshot,
    PointColumn,
    PointTable,
    StreamGenerationId,
    ValidityContract,
    ValueSchema,
)

from zlc_runtime.dataset import MonitorCoverage
from zlc_runtime.dataset_output import (
    DatasetOutputDeclaration,
    FinalDatasetOutput,
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


class _Slot:
    notification_failure = None

    def __init__(self, state: dict[str, LiveDatasetOutput]) -> None:
        self.state = state

    def freeze_live_outputs(self):
        return dict(self.state)

    def close(self) -> None:
        return None


class _Events:
    def __init__(self) -> None:
        self.current: SelectionState | None = None
        self._selection_callbacks = []
        self._fit_callbacks = []

    def subscribe_selection(self, callback):
        self._selection_callbacks.append(callback)

        def unsubscribe() -> None:
            if callback in self._selection_callbacks:
                self._selection_callbacks.remove(callback)

        return unsubscribe

    def subscribe_fit(self, callback):
        self._fit_callbacks.append(callback)

        def unsubscribe() -> None:
            if callback in self._fit_callbacks:
                self._fit_callbacks.remove(callback)

        return unsubscribe

    def selector_data(self, _kind: str) -> SelectionState:
        if self.current is None:
            raise RuntimeError("no current selection")
        return self.current

    def emit_selection(self, change: SelectionChange, state: SelectionState) -> None:
        self.current = state
        for callback in tuple(self._selection_callbacks):
            callback(change, state)

    def emit_fit(self, event: FitEventValue) -> None:
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
    slot = _Slot(state)
    plane = SignalDataPlane()
    plane.reserve(source)
    plane.attach(source, slot)
    plane.mark_changed(source, slot)
    initial = plane.freeze()
    return plane, source, slot, state, initial


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


def _wait_for_signal(plane, bridge, signal_name: str, revision: int):
    deadline = time.monotonic() + 2.0
    while True:
        front = plane.freeze()
        value = front.value(signal_name)
        if value is not None and value.snapshot.ref.revision.value == revision:
            return front
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise TimeoutError(f"{signal_name} did not reach revision {revision}")
        bridge.wait_for_wake(remaining)


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
            "@logic/image/roi_value",
        }
    )
    bridge = SelectionBridge(
        plane,
        "camera/frame",
        events,
        events,
        bridge_id="image",
    )
    bridge.start()
    state = SelectionState(
        "image",
        "area",
        (
            SelectionRange("x", 0.0, 1.0),
            SelectionRange("y", 20.0, 30.0),
        ),
        revision=1,
    )
    try:
        events.emit_selection(SelectionChange.COMMITTED, state)
        front = plane.freeze()
        roi_frame = front.value("@logic/image/roi_frame")
        roi_value = front.value("@logic/image/roi_value")
        assert roi_frame is not None and roi_value is not None
        np.testing.assert_array_equal(
            roi_frame.snapshot.block.values,
            values[:, :, 1:3, 1:3],
        )
        assert float(roi_value.snapshot.block.values.reshape(-1)[0]) == 6.0
        roi_publication = front.publication("@logic/image/roi_frame")
        assert roi_publication is not None
        assert roi_publication.direct_parent_refs == (initial.publication("camera/frame").event_ref,)
    finally:
        _close(bridge, plane, source)


def test_selection_commit_republishes_same_source_and_source_revision_follows() -> None:
    schema = _image_schema()
    values = np.arange(12, dtype=np.float64).reshape(1, 1, 4, 3)
    plane, source, slot, state, _initial = _source_setup(schema, values)
    events = _Events()
    plane.set_front_signals({"camera/frame", "@logic/image/roi_value"})
    bridge = SelectionBridge(plane, "camera/frame", events, events, bridge_id="image")
    bridge.start()
    first_state = SelectionState(
        "image",
        "area",
        (SelectionRange("x", -1.0, 0.0), SelectionRange("y", 10.0, 20.0)),
        revision=1,
    )
    second_state = SelectionState(
        "image",
        "area",
        (SelectionRange("x", 1.0, 2.0), SelectionRange("y", 20.0, 30.0)),
        revision=2,
    )
    try:
        events.emit_selection(SelectionChange.COMMITTED, first_state)
        first = plane.freeze().publication("@logic/image/roi_value")
        assert first is not None
        events.emit_selection(SelectionChange.COMMITTED, second_state)
        second_front = plane.freeze()
        second = second_front.publication("@logic/image/roi_value")
        assert second is not None
        assert second.event_ref.generation != first.event_ref.generation
        assert second.event_ref.sequence == 1
        assert second.direct_parent_refs == first.direct_parent_refs
        assert float(second_front.value("@logic/image/roi_value").snapshot.block.values.reshape(-1)[0]) == 9.0

        state["frame"] = LiveDatasetOutput(
            state["frame"].declaration,
            _snapshot("frame", 2, schema, values + 100.0),
            MonitorCoverage(1, 1),
        )
        plane.mark_changed(source, slot)
        front = _wait_for_signal(plane, bridge, "@logic/image/roi_value", 2)
        publication = front.publication("@logic/image/roi_value")
        assert publication is not None
        assert publication.direct_parent_refs[0].sequence == 2
        assert float(front.value("@logic/image/roi_value").snapshot.block.values.reshape(-1)[0]) == 109.0
    finally:
        _close(bridge, plane, source)


def test_curve_range_and_facet_condition_select_point_rows_inclusive() -> None:
    schema = _curve_schema(with_facet=True)
    values = np.asarray([[[1.0], [2.0], [10.0], [20.0], [30.0]]])
    plane, source, _slot, _state, _initial = _source_setup(schema, values)
    events = _Events()
    plane.set_front_signals({"camera/frame", "@logic/curve/roi_value"})
    bridge = SelectionBridge(plane, "camera/frame", events, events, bridge_id="curve")
    bridge.start()
    selection = SelectionState(
        "curve",
        "x_range",
        (SelectionRange("x", 0.0, 3.0),),
        (FacetCondition("facet", 1.0),),
        revision=1,
    )
    try:
        events.emit_selection(SelectionChange.COMMITTED, selection)
        front = plane.freeze()
        value = front.value("@logic/curve/roi_value")
        assert value is not None
        assert float(value.snapshot.block.values.reshape(-1)[0]) == 15.0
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
    bridge = SelectionBridge(plane, "camera/frame", events, events, bridge_id="fit")
    bridge.start()
    try:
        events.emit_fit(
            FitEventValue(
                parameter_names=("x0",),
                parameter_units={"x0": "pixel"},
                parameter_values={"x0": np.asarray([2.5])},
                parameter_errors={"x0": np.asarray([0.1])},
                success=np.asarray([True]),
                sample_axis_name="",
                sample_coordinates=np.asarray([0.0]),
                sample_unit="",
                sample_labels=None,
                source_revision=1,
                batch_revision=1,
            )
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
        events.emit_selection(
            SelectionChange.REMOVED,
            SelectionState(
                "curve",
                "x_range",
                (SelectionRange("x", 0.0, 1.0),),
                revision=1,
            ),
        )
        assert plane.freeze().value("@logic/fit/x0") is not None
        events.emit_fit(
            FitEventValue(
                parameter_names=("x0",),
                parameter_units={"x0": "pixel"},
                parameter_values={"x0": np.asarray([3.5])},
                parameter_errors={"x0": np.asarray([0.2])},
                success=np.asarray([True]),
                sample_axis_name="",
                sample_coordinates=np.asarray([0.0]),
                sample_unit="",
                sample_labels=None,
                source_revision=1,
                batch_revision=2,
            )
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
        sample_axis_name="facet",
        sample_coordinates=coordinates,
        sample_unit=unit,
        sample_labels=labels,
        source_revision=source_revision,
        batch_revision=batch_revision,
    )


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
    bridge = SelectionBridge(plane, "camera/frame", events, events, bridge_id="batch")
    bridge.start()
    try:
        events.emit_fit(_batch_fit_event(source_revision=1))
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
    bridge = SelectionBridge(plane, "camera/frame", events, events, bridge_id="text")
    bridge.start()
    try:
        event = _batch_fit_event(
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


def _single_cell_facet_event(*, source_revision: int, batch_revision: int) -> FitEventValue:
    return FitEventValue(
        parameter_names=("center",),
        parameter_units={"center": "pixel"},
        parameter_values={"center": np.asarray([4.0])},
        parameter_errors={"center": np.asarray([0.25])},
        success=np.asarray([True]),
        sample_axis_name="facet",
        sample_coordinates=np.asarray([42.0]),
        sample_unit="V",
        sample_labels=None,
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
    bridge = SelectionBridge(plane, "camera/frame", events, events, bridge_id="one")
    bridge.start()
    try:
        events.emit_fit(_single_cell_facet_event(source_revision=1, batch_revision=1))
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


def test_scalar_and_single_cell_facet_use_the_same_vector_materializer(monkeypatch) -> None:
    original = SelectionBridge._materialize_fit_vector
    calls: dict[str, list[str]] = {"sample": [], "facet": []}

    def spy(self, source, fit_source_ref, name, values, schema, validity, contract_id, coverage):
        axis_name = schema.point_table.columns[0].name
        calls[axis_name].append(name)
        return original(
            self,
            source,
            fit_source_ref,
            name,
            values,
            schema,
            validity,
            contract_id,
            coverage,
        )

    monkeypatch.setattr(SelectionBridge, "_materialize_fit_vector", spy)

    scalar_plane, scalar_source, _slot, _state, scalar_initial = _source_setup(
        _curve_schema(), np.zeros((1, 5, 1), dtype=np.float64)
    )
    scalar_events = _Events()
    scalar_bridge = SelectionBridge(
        scalar_plane,
        "camera/frame",
        scalar_events,
        scalar_events,
        bridge_id="scalar-path",
    )
    facet_plane, facet_source, _slot, _state, facet_initial = _source_setup(
        _curve_schema(), np.zeros((1, 5, 1), dtype=np.float64)
    )
    facet_events = _Events()
    facet_bridge = SelectionBridge(
        facet_plane,
        "camera/frame",
        facet_events,
        facet_events,
        bridge_id="facet-path",
    )
    try:
        scalar_bridge._materialize_fit_outputs(
            scalar_initial.value("camera/frame").snapshot,
            FitEventValue(
                parameter_names=("center",),
                parameter_units={"center": "pixel"},
                parameter_values={"center": np.asarray([1.0])},
                parameter_errors={"center": np.asarray([0.1])},
                success=np.asarray([True]),
                sample_axis_name="",
                sample_coordinates=np.asarray([0.0]),
                sample_unit="",
                sample_labels=None,
                source_revision=1,
                batch_revision=1,
            ),
        )
        facet_bridge._materialize_fit_outputs(
            facet_initial.value("camera/frame").snapshot,
            _single_cell_facet_event(source_revision=1, batch_revision=1),
        )
        assert calls["sample"] == ["center", "center_err"]
        assert calls["sample"] == calls["facet"]
    finally:
        _close(scalar_bridge, scalar_plane, scalar_source)
        _close(facet_bridge, facet_plane, facet_source)


def test_late_stale_fit_failure_cannot_withdraw_newer_batch(monkeypatch) -> None:
    schema = _curve_schema()
    values = np.zeros((1, 5, 1), dtype=np.float64)
    plane, source, slot, state, _initial = _source_setup(schema, values)
    events = _Events()
    plane.set_front_signals({"camera/frame", "@logic/race/center"})
    bridge = SelectionBridge(plane, "camera/frame", events, events, bridge_id="race")
    bridge.start()
    gate = Event()
    original_evaluate = bridge._evaluate_processor

    def delayed_evaluate(processor, signal):
        if signal.snapshot.ref.revision.value == 2:
            gate.wait(2.0)
        return original_evaluate(processor, signal)

    monkeypatch.setattr(bridge, "_evaluate_processor", delayed_evaluate)
    try:
        events.emit_fit(_batch_fit_event(source_revision=1, batch_revision=1))
        processor = bridge._fit_processor
        assert processor is not None

        state["frame"] = LiveDatasetOutput(
            state["frame"].declaration,
            _snapshot("frame", 2, schema, values + 1.0),
            MonitorCoverage(1, 5),
        )
        plane.mark_changed(source, slot)
        plane.freeze()

        events.emit_fit(_batch_fit_event(source_revision=2, batch_revision=2))
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


def test_fit_contract_fields_and_exports_are_exact() -> None:
    assert tuple(FitEventValue.__dataclass_fields__) == (
        "parameter_names",
        "parameter_units",
        "parameter_values",
        "parameter_errors",
        "success",
        "sample_axis_name",
        "sample_coordinates",
        "sample_unit",
        "sample_labels",
        "source_revision",
        "batch_revision",
    )
    assert tuple(selection_bridge_module.__all__) == (
        "FacetCondition",
        "FitEventValue",
        "SelectionBridge",
        "SelectionChange",
        "SelectionDataReader",
        "SelectionEventSource",
        "SelectionRange",
        "SelectionState",
    )
    assert not hasattr(selection_bridge_module, "FitParameter")
    contract = (Path(__file__).parents[1] / "docs" / "contract.md").read_text(
        encoding="utf-8"
    )
    assert "source_revision: int" in contract
    assert "batch_revision: int" in contract
    assert "success AND isfinite(error)" in contract
    assert "{sample_axis_name}_label` TEXT point column" in contract


def test_fit_event_batch_revision_is_retained_across_source_updates() -> None:
    schema = _curve_schema()
    values = np.zeros((1, 5, 1), dtype=np.float64)
    plane, source, slot, state, _initial = _source_setup(schema, values)
    events = _Events()
    plane.set_front_signals({"camera/frame", "@logic/revision/center"})
    bridge = SelectionBridge(plane, "camera/frame", events, events, bridge_id="revision")
    bridge.start()
    try:
        first_event = _batch_fit_event(source_revision=1, batch_revision=1)
        events.emit_fit(first_event)
        first = plane.freeze().value("@logic/revision/center")
        assert first is not None
        assert bridge.wait_for_wake(2.0)
        plane.freeze()
        bridge.wait_for_wake(0.0)
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
        plane.mark_changed(source, slot)
        plane.freeze()
        assert bridge.wait_for_wake(2.0)
        between = plane.freeze()
        assert between.value("@logic/revision/center") is first
        assert tuple(
            item.name
            for item in plane.describe_signals()
            if item.owner_id.startswith("revision:fit:")
        ) == first_names
        events.emit_fit(_batch_fit_event(source_revision=2, batch_revision=2))
        front = _wait_for_signal(plane, bridge, "@logic/revision/center", 2)
        second = front.value("@logic/revision/center")
        assert second is not None
        assert second.snapshot.ref.revision.value > first.snapshot.ref.revision.value
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
    held: dict[int, object] = {}

    def remember_current() -> None:
        publication = plane.latest_publication("camera/frame")
        assert publication is not None
        value = publication.value("camera/frame")
        assert value is not None
        held[value.snapshot.ref.revision.value] = publication

    bridge = SelectionBridge(
        plane,
        "camera/frame",
        events,
        events,
        bridge_id="trail",
        source_publication_for=lambda revision: held.get(revision),
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
        plane.mark_changed(source, slot)
        plane.freeze()
        remember_current()

        events.emit_fit(_batch_fit_event(source_revision=1, batch_revision=1))
        assert bridge.last_error is None
        front = plane.freeze()
        published = front.value("@logic/trail/center")
        assert published is not None, "the trailing fit was dropped"
        publication = front.publication("@logic/trail/center")
        parent_ref = publication.direct_parent_refs[0]
        assert parent_ref.sequence == 1  # camera@1: the shot it was fitted on

        # The next shot's fit follows normally against camera@2.
        events.emit_fit(_batch_fit_event(source_revision=2, batch_revision=2))
        assert bridge.last_error is None
        second = plane.freeze().publication("@logic/trail/center")
        assert second.direct_parent_refs[0].sequence == 2

        # A revision no panel ever held reports, not silently vanishes.
        events.emit_fit(_batch_fit_event(source_revision=77, batch_revision=3))
        assert bridge.last_error is not None
    finally:
        _close(bridge, plane, source)


def test_fit_event_value_freezes_batch_inputs_and_rejects_invalid_tables() -> None:
    values = np.asarray([1.0, np.nan])
    errors = np.asarray([0.1, np.nan])
    success = np.asarray([True, False])
    event = FitEventValue(
        parameter_names=("center",),
        parameter_units={"center": "pixel"},
        parameter_values={"center": values},
        parameter_errors={"center": errors},
        success=success,
        sample_axis_name="facet",
        sample_coordinates=np.asarray([0.0, 1.0]),
        sample_unit="",
        sample_labels=("a", "b"),
        source_revision=1,
        batch_revision=1,
    )
    values[0] = 99.0
    errors[0] = 99.0
    success[0] = False
    assert event.parameter_values["center"][0] == 1.0
    assert event.parameter_errors["center"][0] == 0.1
    assert bool(event.success[0]) is True
    with pytest.raises(ValueError):
        event.parameter_values["center"][0] = 4.0
    with pytest.raises(ValueError):
        event.success[0] = False
    with pytest.raises(TypeError):
        event.parameter_values["other"] = np.asarray([1.0, 2.0])

    with pytest.raises(ValueError, match="match value length"):
        FitEventValue(
            parameter_names=("center",),
            parameter_units={"center": "pixel"},
            parameter_values={"center": np.asarray([1.0, 2.0])},
            parameter_errors={"center": np.asarray([0.1])},
            success=np.asarray([True, False]),
            sample_axis_name="facet",
            sample_coordinates=np.asarray([0.0, 1.0]),
            sample_unit="",
            sample_labels=None,
            source_revision=1,
            batch_revision=1,
        )


def test_added_and_updated_selection_events_do_not_publish() -> None:
    schema = _curve_schema()
    values = np.asarray([[[1.0], [2.0], [3.0], [4.0], [5.0]]])
    plane, source, _slot, _state, _initial = _source_setup(schema, values)
    events = _Events()
    plane.set_front_signals({"camera/frame", "@logic/curve/roi_value"})
    bridge = SelectionBridge(plane, "camera/frame", events, events, bridge_id="curve")
    bridge.start()
    state = SelectionState(
        "curve",
        "x_range",
        (SelectionRange("x", 1.0, 3.0),),
        revision=1,
    )
    try:
        events.emit_selection(SelectionChange.ADDED, state)
        events.emit_selection(
            SelectionChange.UPDATED,
            SelectionState(
                "curve",
                "x_range",
                (SelectionRange("x", 0.0, 4.0),),
                revision=2,
            ),
        )
        assert bridge.selection is None
        assert plane.latest_publication("@logic/curve/roi_value") is None
    finally:
        _close(bridge, plane, source)


def test_removed_selection_retires_derived_signals_and_unknown_axis_is_loud() -> None:
    schema = _curve_schema()
    values = np.asarray([[[1.0], [2.0], [3.0], [4.0], [5.0]]])
    plane, source, _slot, _state, _initial = _source_setup(schema, values)
    events = _Events()
    plane.set_front_signals({"camera/frame", "@logic/curve/roi_value"})
    bridge = SelectionBridge(plane, "camera/frame", events, events, bridge_id="curve")
    bridge.start()
    selection = SelectionState(
        "curve",
        "x_range",
        (SelectionRange("x", 1.0, 3.0),),
        revision=1,
    )
    try:
        events.emit_selection(SelectionChange.COMMITTED, selection)
        assert plane.latest_publication("@logic/curve/roi_value") is not None
        events.emit_selection(SelectionChange.REMOVED, selection)
        plane.freeze()
        assert plane.latest_publication("@logic/curve/roi_value") is None
        with pytest.raises(ValueError, match="not uniquely present"):
            events.emit_selection(
                SelectionChange.COMMITTED,
                SelectionState(
                    "curve",
                    "x_range",
                    (SelectionRange("missing", 0.0, 1.0),),
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
        {"camera/frame", "@logic/frozen/roi_frame", "@logic/frozen/roi_value"}
    )
    plane.publish_final(
        source,
        {
            "frame": FinalDatasetOutput(
                state_map["frame"].declaration,
                state_map["frame"].snapshot,
            )
        },
    )
    assert not plane.is_generation_live("camera/frame")

    events = _Events()
    bridge = SelectionBridge(plane, "camera/frame", events, events, bridge_id="frozen")
    bridge.start()
    try:
        events.emit_selection(
            SelectionChange.COMMITTED,
            SelectionState(
                "image",
                "area",
                (SelectionRange("x", 0.0, 1.0), SelectionRange("y", 20.0, 30.0)),
                revision=1,
            ),
        )
        front = plane.freeze()
        roi_frame = front.value("@logic/frozen/roi_frame")
        roi_value = front.value("@logic/frozen/roi_value")
        assert roi_frame is not None, bridge.last_error
        np.testing.assert_array_equal(
            roi_frame.snapshot.block.values, values[:, :, 1:3, 1:3]
        )
        assert float(roi_value.snapshot.block.values.reshape(-1)[0]) == 6.0
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
        {"camera/frame", "@logic/frozen/roi_frame", "@logic/frozen/roi_value"}
    )
    plane.publish_final(
        source,
        {
            "frame": FinalDatasetOutput(
                state_map["frame"].declaration,
                state_map["frame"].snapshot,
            )
        },
    )
    events = _Events()
    bridge = SelectionBridge(plane, "camera/frame", events, events, bridge_id="frozen")
    bridge.start()
    try:
        for revision, upper in ((1, 1.0), (2, 2.0)):
            events.emit_selection(
                SelectionChange.COMMITTED,
                SelectionState(
                    "image",
                    "area",
                    (
                        SelectionRange("x", 0.0, upper),
                        SelectionRange("y", 20.0, 30.0),
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
    bridge = SelectionBridge(plane, "camera/frame", events, events, bridge_id="topology")
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
                (SelectionRange("x", 0.0, 1.0), SelectionRange("y", 20.0, 30.0)),
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
                (SelectionRange("x", 0.0, 2.0), SelectionRange("y", 20.0, 30.0)),
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

    from zlc_runtime.selection_bridge import SelectionBridge

    values = np.zeros((1, 1, 120, 160), dtype=np.uint16)

    class _Schema:
        class repeat_axis:
            axis_id = "r"

        class cell_schema:
            data_axes = ()

    taken = SelectionBridge._take(values, range(0, 1), range(0, 1), {}, _Schema)

    assert taken.base is not None, "a contiguous selection copied the frame"
    assert taken.shape == (1, 1, 120, 160)


def test_an_implicit_axis_cropped_to_a_run_stays_implicit() -> None:
    """Writing the coordinates out here undid, one layer down, exactly what the
    producer stopped doing: build a tuple, validate every element, digest all of
    it -- per frame, for an axis that is still just indexed 0..n-1 from later."""

    from zlc_data import SPATIAL_X
    from zlc_data.axis import AxisId, AxisSpec
    from zlc_runtime.selection_bridge import SelectionBridge

    axis = AxisSpec(AxisId("cam.x"), "x", SPATIAL_X, 1920)

    cropped = SelectionBridge._subset_axis(axis, range(400, 912))

    assert cropped.coordinates is None
    assert cropped.size == 512
    assert cropped.index_origin == 400
    assert cropped.coordinate_at(0) == 400
