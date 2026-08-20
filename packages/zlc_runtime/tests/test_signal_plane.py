"""Canonical commit/current/seal contracts for the signal plane."""

from __future__ import annotations

import gc
import threading
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from zlc_data import (
    REPEAT,
    SCAN_POINT,
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
    ValueSchema,
)
from zlc_runtime.dataset import DatasetCoverage, MonitorCoverage
from zlc_runtime.dataset_output import DatasetOutputDeclaration, LiveDatasetOutput
from zlc_runtime.plane import SignalDataPlane


def _event(name: str, value: float) -> OwnedSnapshot:
    repeat = AxisSpec(AxisId(f"{name}.repeat"), "repeat", REPEAT, 1, (0,))
    point = PointColumn(
        AxisId(f"{name}.point"),
        "point",
        SCAN_POINT,
        PointColumn.NUMERIC,
        (0,),
    )
    schema = DatasetSchema(
        repeat,
        PointTable(1, (point,)),
        None,
        ValueSchema.scalar(np.dtype("float64"), "count"),
    )
    block = DataBlock(
        BlockId(f"{name}.plugin"),
        DatasetRevision(77),
        np.asarray([[[value]]], dtype=np.float64),
        CellValidity(np.ones((1, 1), dtype=np.bool_)),
        schema,
    )
    return OwnedSnapshot(block.ref(StreamGenerationId("plugin-generation")), block)


def _finite(
    declaration: DatasetOutputDeclaration,
    *,
    value: float,
    total: int,
    origin: int,
    written: int,
    run_record: dict[str, object] | None = None,
) -> LiveDatasetOutput:
    event = _event(declaration.name, value)
    schema = event.block.schema
    canonical = DatasetSchema(
        AxisSpec(
            schema.repeat_axis.axis_id,
            schema.repeat_axis.name,
            schema.repeat_axis.role,
            total,
            tuple(range(total)),
        ),
        schema.point_table,
        schema.grid_topology,
        schema.cell_schema,
    )
    return LiveDatasetOutput(
        declaration,
        event,
        DatasetCoverage(written, total),
        run_record,
        canonical,
        (origin, 0),
    )


def _latest(
    declaration: DatasetOutputDeclaration,
    value: float,
    run_record: dict[str, object] | None = None,
) -> LiveDatasetOutput:
    return LiveDatasetOutput(
        declaration,
        _event(declaration.name, value),
        MonitorCoverage(1, 1),
        run_record,
    )


def _finite_grid_point(
    declaration: DatasetOutputDeclaration,
    *,
    value: float,
    point_origin: int,
    written: int,
) -> LiveDatasetOutput:
    event = _event(declaration.name, value)
    schema = event.block.schema
    x_id = AxisId(f"{declaration.name}.grid-x")
    y_id = AxisId(f"{declaration.name}.grid-y")
    canonical = DatasetSchema(
        schema.repeat_axis,
        PointTable(
            4,
            (
                PointColumn(
                    x_id,
                    "x",
                    SCAN_POINT,
                    PointColumn.NUMERIC,
                    (0.0, 1.0, 0.0, 1.0),
                ),
                PointColumn(
                    y_id,
                    "y",
                    SCAN_POINT,
                    PointColumn.NUMERIC,
                    (0.0, 0.0, 1.0, 1.0),
                ),
            ),
        ),
        GridTopology(
            (x_id, y_id),
            ((0.0, 1.0), (0.0, 1.0)),
            ((0, 0), (1, 0), (0, 1), (1, 1)),
        ),
        schema.cell_schema,
    )
    return LiveDatasetOutput(
        declaration,
        event,
        DatasetCoverage(written, 4),
        canonical_schema=canonical,
        cell_origin=(0, point_origin),
    )


def _node(instance_id: str, *declarations: DatasetOutputDeclaration):
    return SimpleNamespace(
        instance_id=instance_id,
        dataset_output_declarations=declarations,
        signal_key=lambda name: f"{instance_id}/{name}",
    )


def test_commit_mints_runtime_identity_and_freezes_run_record() -> None:
    declaration = DatasetOutputDeclaration("frame", "test.frame")
    node = _node("camera", declaration)
    record = {"camera": {"gain": 1}, "shape": [1, 1]}
    plane = SignalDataPlane()
    try:
        generation = plane.reserve(node)
        value = plane.commit_live(
            node,
            {
                "frame": _finite(
                    declaration,
                    value=1.0,
                    total=1,
                    origin=0,
                    written=1,
                    run_record=record,
                )
            },
        )["camera/frame"]
        record["camera"]["gain"] = 9
        assert value.run_record["camera"]["gain"] == 1
        assert value.snapshot.ref.stream_generation == generation
        assert value.snapshot.ref.revision.value == 1
        assert value.snapshot.ref.block_id == BlockId("camera/frame.event")
        assert plane.seal_committed(node)
    finally:
        plane.close()


def test_partial_current_has_invalid_future_and_overlap_is_rejected() -> None:
    declaration = DatasetOutputDeclaration("frame", "test.frame")
    node = _node("partial", declaration)
    plane = SignalDataPlane()
    try:
        plane.reserve(node)
        plane.commit_live(
            node,
            {
                "frame": _finite(
                    declaration,
                    value=10.0,
                    total=4,
                    origin=0,
                    written=1,
                )
            },
        )
        current = plane.current_dataset("partial/frame")
        assert current.block.values[:, 0, 0].tolist() == [10.0, 0.0, 0.0, 0.0]
        assert current.expanded_validity()[:, 0, 0].tolist() == [
            True,
            False,
            False,
            False,
        ]
        with pytest.raises(ValueError, match="overlaps"):
            plane.commit_live(
                node,
                {
                    "frame": _finite(
                        declaration,
                        value=99.0,
                        total=4,
                        origin=0,
                        written=2,
                    )
                },
            )
        assert plane.seal_committed(node, cut_short=True)
    finally:
        plane.close()


def test_finite_signal_reports_full_repeat_geometry_from_first_event_through_stop() -> None:
    declaration = DatasetOutputDeclaration("frame", "test.frame")
    node = _node("repeat-display", declaration)
    plane = SignalDataPlane()
    try:
        plane.reserve(node)
        event_value = plane.commit_live(
            node,
            {
                "frame": _finite(
                    declaration,
                    value=10.0,
                    total=30,
                    origin=0,
                    written=1,
                )
            },
        )["repeat-display/frame"]

        # Exact Processors still receive the one-event chunk.
        assert event_value.shape == (1, 1, 1)
        description = plane.describe_signals()[0]
        assert description.shape == (30, 1, 1)
        current = plane.current_dataset(description.name)
        assert current.block.values.shape == (30, 1, 1)
        assert current.expanded_validity()[:, 0, 0].tolist() == [
            True,
            *([False] * 29),
        ]

        assert plane.seal_committed(node, cut_short=True)
        stopped = plane.describe_signals()[0]
        assert not stopped.live
        assert stopped.shape == (30, 1, 1)
    finally:
        plane.close()


def test_finite_signal_reports_full_point_grid_geometry_while_cells_arrive() -> None:
    declaration = DatasetOutputDeclaration("scan", "test.scan")
    node = _node("grid-display", declaration)
    plane = SignalDataPlane()
    try:
        plane.reserve(node)
        plane.commit_live(
            node,
            {
                "scan": _finite_grid_point(
                    declaration,
                    value=10.0,
                    point_origin=0,
                    written=1,
                )
            },
        )
        description = plane.describe_signals()[0]
        assert description.shape == (1, 4, 1)
        first = plane.current_dataset(description.name)
        assert first.block.schema.grid_topology is not None
        assert first.block.schema.grid_topology.logical_shape == (2, 2)
        assert first.expanded_validity()[0, :, 0].tolist() == [
            True,
            False,
            False,
            False,
        ]

        plane.commit_live(
            node,
            {
                "scan": _finite_grid_point(
                    declaration,
                    value=40.0,
                    point_origin=3,
                    written=2,
                )
            },
        )
        second = plane.current_dataset(description.name)
        assert second.block.values[0, :, 0].tolist() == [10.0, 0.0, 0.0, 40.0]
        assert second.expanded_validity()[0, :, 0].tolist() == [
            True,
            False,
            False,
            True,
        ]
        assert plane.seal_committed(node, cut_short=True)
        assert plane.describe_signals()[0].shape == (1, 4, 1)
    finally:
        plane.close()


def test_monitor_to_finite_generation_changes_from_event_to_authored_shape() -> None:
    declaration = DatasetOutputDeclaration("frame", "test.frame")
    node = _node("restart-shape", declaration)
    plane = SignalDataPlane()
    try:
        plane.begin_generation(node)
        plane.commit_live(node, {"frame": _latest(declaration, 1.0)})
        old_publication = plane.latest_publication("restart-shape/frame")
        assert old_publication is not None
        assert plane.describe_signals()[0].shape == (1, 1, 1)

        plane.retire(node)
        plane.begin_generation(node)
        plane.commit_live(
            node,
            {
                "frame": _finite(
                    declaration,
                    value=2.0,
                    total=30,
                    origin=0,
                    written=1,
                )
            },
        )
        assert plane.describe_signals()[0].shape == (30, 1, 1)
        with pytest.raises(ValueError, match="another signal generation"):
            plane.current_dataset("restart-shape/frame", old_publication)
    finally:
        plane.close()


def test_one_canonical_prefix_is_reused_across_later_event_commits(monkeypatch) -> None:
    import zlc_runtime.plane as plane_module

    declaration = DatasetOutputDeclaration("frame", "test.frame")
    node = _node("presentation-cache", declaration)
    calls = 0
    real = plane_module.owned_snapshot_from_arrays

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(plane_module, "owned_snapshot_from_arrays", counted)
    plane = SignalDataPlane()
    try:
        plane.reserve(node)
        plane.commit_live(
            node,
            {
                "frame": _finite(
                    declaration,
                    value=1.0,
                    total=3,
                    origin=0,
                    written=1,
                )
            },
        )
        first_publication = plane.latest_publication("presentation-cache/frame")
        assert first_publication is not None
        first = plane.current_dataset(
            "presentation-cache/frame",
            first_publication,
        )
        assert calls == 1

        plane.commit_live(
            node,
            {
                "frame": _finite(
                    declaration,
                    value=2.0,
                    total=3,
                    origin=1,
                    written=2,
                )
            },
        )
        # Committing an event neither materializes a prefix nor discards the
        # accepted display prefix.  Semantic/Edit/Save reads of that same
        # publication therefore reuse one immutable snapshot.
        assert calls == 1
        assert (
            plane.current_dataset(
                "presentation-cache/frame",
                first_publication,
            )
            is first
        )
        assert calls == 1

        second = plane.current_dataset("presentation-cache/frame")
        assert calls == 2
        assert second.block.values[:, 0, 0].tolist() == [1.0, 2.0, 0.0]
        assert plane.seal_committed(node, cut_short=True)
        assert calls == 2
    finally:
        plane.close()


def test_canonical_prefix_is_bound_to_its_publication_when_next_event_wins_race() -> None:
    declaration = DatasetOutputDeclaration("frame", "test.frame")
    node = _node("publication-prefix", declaration)
    plane = SignalDataPlane()
    try:
        plane.reserve(node)
        plane.commit_live(
            node,
            {
                "frame": _finite(
                    declaration,
                    value=1.0,
                    total=3,
                    origin=0,
                    written=1,
                )
            },
        )
        first_publication = plane.latest_publication("publication-prefix/frame")
        assert first_publication is not None
        plane.commit_live(
            node,
            {
                "frame": _finite(
                    declaration,
                    value=2.0,
                    total=3,
                    origin=1,
                    written=2,
                )
            },
        )

        first = plane.current_dataset(
            "publication-prefix/frame",
            first_publication,
        )
        latest = plane.current_dataset("publication-prefix/frame")
        assert first.block.values[:, 0, 0].tolist() == [1.0, 0.0, 0.0]
        assert first.expanded_validity()[:, 0, 0].tolist() == [
            True,
            False,
            False,
        ]
        assert latest.block.values[:, 0, 0].tolist() == [1.0, 2.0, 0.0]
    finally:
        plane.close()


def test_repeat_100_publication_cost_and_retained_arrays_stay_linear(monkeypatch) -> None:
    import zlc_runtime.plane as plane_module

    declaration = DatasetOutputDeclaration("frame", "test.frame")
    node = _node("linear", declaration)
    calls = 0
    real = plane_module.owned_snapshot_from_arrays

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(plane_module, "owned_snapshot_from_arrays", counted)
    plane = SignalDataPlane()
    try:
        plane.reserve(node)
        published_value_bytes = 0
        for index in range(100):
            value = plane.commit_live(
                node,
                {
                    "frame": _finite(
                        declaration,
                        value=float(index),
                        total=100,
                        origin=index,
                        written=index + 1,
                    )
                },
            )["linear/frame"]
            published_value_bytes += value.snapshot.block.values.nbytes
            plane.freeze()
        assert calls == 0
        assert published_value_bytes == 100 * np.dtype(np.float64).itemsize

        # Runtime retains exactly 100 one-cell event arrays plus one 100-cell
        # placement mask.  It has not built (or retained) 1+2+...+100 prefixes.
        state = plane._states["linear"]
        retained = [
            value
            for _sequence, value in state.commit_values["linear/frame"]
        ]
        assert len(retained) == 100
        assert state.materialized == {}
        assert state.publication is not None
        assert state.publication.value("linear/frame") is retained[-1]
        retained_array_bytes = sum(
            value.snapshot.block.values.nbytes
            + value.snapshot.block.validity.mask.nbytes
            for value in retained
        )
        placement_bytes = state.occupied_cells["linear/frame"].nbytes
        assert retained_array_bytes + placement_bytes == 100 * (8 + 1) + 100

        current = plane.current_dataset("linear/frame")
        assert calls == 1
        assert current.expanded_validity().all()
        assert plane.seal_committed(node)
        assert calls == 1
    finally:
        plane.close()


@pytest.mark.parametrize("operation", ("current", "seal"))
def test_full_materialization_does_not_hold_plane_lock(monkeypatch, operation: str) -> None:
    import zlc_runtime.plane as plane_module

    declaration = DatasetOutputDeclaration("frame", "test.frame")
    node = _node(f"nonblocking-{operation}", declaration)
    entered = threading.Event()
    release = threading.Event()
    reader_done = threading.Event()
    errors: list[BaseException] = []
    real = plane_module.owned_snapshot_from_arrays

    def gated(*args, **kwargs):
        entered.set()
        if not release.wait(2.0):
            raise TimeoutError("materialization gate did not open")
        return real(*args, **kwargs)

    plane = SignalDataPlane()
    try:
        plane.reserve(node)
        plane.commit_live(
            node,
            {
                "frame": _finite(
                    declaration,
                    value=1.0,
                    total=1,
                    origin=0,
                    written=1,
                )
            },
        )
        monkeypatch.setattr(plane_module, "owned_snapshot_from_arrays", gated)

        def materialize() -> None:
            try:
                if operation == "seal":
                    plane.seal_committed(node)
                else:
                    plane.current_dataset(node.signal_key("frame"))
            except BaseException as error:
                errors.append(error)

        worker = threading.Thread(target=materialize)
        worker.start()
        assert entered.wait(1.0)

        def read_plane() -> None:
            plane.freeze()
            plane.describe_signals()
            reader_done.set()

        reader = threading.Thread(target=read_plane)
        reader.start()
        assert reader_done.wait(1.0), "materialization held the Plane lock"
        release.set()
        worker.join(2.0)
        reader.join(2.0)
        assert not errors
    finally:
        release.set()
        plane.close()


def test_mixed_exact_and_latest_siblings_share_one_event_without_retention() -> None:
    history_declaration = DatasetOutputDeclaration("history", "test.history")
    phase_declaration = DatasetOutputDeclaration("phase", "test.phase")
    node = _node("mixed", history_declaration, phase_declaration)
    plane = SignalDataPlane()
    first_ref = None
    first = None
    try:
        plane.reserve(node)
        for index in range(2):
            values = plane.commit_live(
                node,
                {
                    "history": _finite(
                        history_declaration,
                        value=float(index + 1),
                        total=2,
                        origin=index,
                        written=index + 1,
                    ),
                    "phase": _latest(phase_declaration, float(index + 1)),
                },
            )
            assert values["mixed/history"].canonical_schema is not None
            assert values["mixed/phase"].canonical_schema is None
            publication = plane.latest_publication("mixed/history")
            assert publication is not None
            if index == 0:
                first = publication
                first_ref = weakref.ref(publication)
        del first
        gc.collect()
        assert first_ref is not None and first_ref() is None
        assert plane.current_dataset("mixed/history").block.values[:, 0, 0].tolist() == [
            1.0,
            2.0,
        ]
        assert plane.seal_committed(node)
    finally:
        plane.close()


def test_latest_processors_run_parallel_per_node_serial_and_coalesce() -> None:
    source_declaration = DatasetOutputDeclaration("frame", "test.frame")
    source = _node("latest-source", source_declaration)
    release_initial = threading.Event()

    class Processor:
        def __init__(self, instance_id: str) -> None:
            self.instance_id = instance_id
            self.declaration = DatasetOutputDeclaration("derived", "test.derived")
            self.entered_initial = threading.Event()
            self.entered_latest = threading.Event()
            self.wake = threading.Event()
            self.lock = threading.Lock()
            self.calls: list[int] = []
            self.accepted: list[int] = []
            self.failures: list[Exception] = []
            self.active = 0
            self.max_active = 0

        @property
        def dataset_output_declarations(self):
            return (self.declaration,)

        def signal_key(self, name: str) -> str:
            return f"{self.instance_id}/{name}"

        def validate_processor_source(self, _source) -> None:
            return None

        def evaluate_processor(self, selected, _publication):
            sequence = selected.snapshot.ref.revision.value
            with self.lock:
                self.calls.append(sequence)
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                first = len(self.calls) == 1
            try:
                if first:
                    self.entered_initial.set()
                    assert release_initial.wait(2.0)
                else:
                    self.entered_latest.set()
                return {"derived": _latest(self.declaration, float(sequence))}
            finally:
                with self.lock:
                    self.active -= 1

        def accept_processor_result(
            self,
            _source,
            publication,
            _result,
        ) -> None:
            self.accepted.append(publication.event_ref.sequence)

        def accept_processor_failure(self, error: Exception) -> None:
            self.failures.append(error)

        def accept_processor_cancelled(self) -> None:
            return None

        def request_processor_owner_wake(self) -> None:
            self.wake.set()

    first = Processor("latest-first")
    second = Processor("latest-second")
    plane = SignalDataPlane()
    try:
        plane.reserve(source)
        plane.commit_live(
            source,
            {"frame": _latest(source_declaration, 1.0)},
        )
        initial = plane.latest_publication("latest-source/frame")
        assert initial is not None
        plane.attach_latest_only_processor(
            first,
            source_name="latest-source/frame",
            initial_publication=initial,
        )
        plane.attach_latest_only_processor(
            second,
            source_name="latest-source/frame",
            initial_publication=initial,
        )

        # Both blocked evaluations have entered before either is released:
        # separate latest Processors therefore do not share one serial worker.
        assert first.entered_initial.wait(2.0)
        assert second.entered_initial.wait(2.0)

        plane.commit_live(
            source,
            {"frame": _latest(source_declaration, 2.0)},
        )
        plane.freeze()
        plane.commit_live(
            source,
            {"frame": _latest(source_declaration, 3.0)},
        )
        plane.freeze()

        release_initial.set()
        assert first.wake.wait(2.0)
        assert second.wake.wait(2.0)
        first.wake.clear()
        second.wake.clear()
        plane.freeze()

        assert first.entered_latest.wait(2.0)
        assert second.entered_latest.wait(2.0)
        assert first.wake.wait(2.0)
        assert second.wake.wait(2.0)
        plane.freeze()

        for processor in (first, second):
            assert processor.calls == [1, 3]
            assert processor.accepted == [1, 3]
            assert processor.max_active == 1
            assert not processor.failures
    finally:
        release_initial.set()
        plane.close()


def test_direct_latest_commit_retires_without_cleanup_callbacks() -> None:
    declaration = DatasetOutputDeclaration("preview", "test.preview")
    node = _node("preview", declaration)
    plane = SignalDataPlane()
    try:
        plane.reserve(node)
        plane.commit_live(node, {"preview": _latest(declaration, 1.0)})
        assert plane.retire(node) == frozenset({"preview/preview"})
        assert plane.latest_publication("preview/preview") is None
    finally:
        plane.close()
