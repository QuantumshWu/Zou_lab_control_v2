"""Canonical commit/current/seal contracts for the signal plane."""

from __future__ import annotations

import gc
import threading
from types import MappingProxyType
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from zlc_data import (
    PRIMARY_INDEX,
    REPEAT,
    SCAN_POINT,
    SPATIAL_X,
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
    event_record: dict[str, object] | None = None,
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
        event_record,
    )


def _latest(
    declaration: DatasetOutputDeclaration,
    value: float,
    run_record: dict[str, object] | None = None,
    *,
    event_record: dict[str, object] | None = None,
) -> LiveDatasetOutput:
    return LiveDatasetOutput(
        declaration,
        _event(declaration.name, value),
        MonitorCoverage(1, 1),
        run_record,
        event_record=event_record,
    )


def _large_latest(
    declaration: DatasetOutputDeclaration,
    value: float,
) -> LiveDatasetOutput:
    scalar = _event(declaration.name, value)
    data_axis = AxisSpec(
        AxisId(f"{declaration.name}.sample"),
        "sample",
        SPATIAL_X,
        200_000,
    )
    schema = DatasetSchema(
        scalar.block.schema.repeat_axis,
        scalar.block.schema.point_table,
        None,
        ValueSchema(
            (data_axis,),
            ValidityContract.value(),
            np.dtype("float64"),
            "count",
        ),
    )
    block = DataBlock(
        BlockId(f"{declaration.name}.large"),
        DatasetRevision(77),
        np.full((1, 1, 200_000), value, dtype=np.float64),
        CellValidity(np.ones((1, 1), dtype=np.bool_)),
        schema,
    )
    snapshot = OwnedSnapshot(
        block.ref(StreamGenerationId("plugin-generation")),
        block,
    )
    return LiveDatasetOutput(
        declaration,
        snapshot,
        MonitorCoverage(1, 1),
    )


def test_derived_monitor_materializes_every_source_primary_index() -> None:
    source_declaration = DatasetOutputDeclaration("frame", "test.frame")
    derived_declaration = DatasetOutputDeclaration(
        "value",
        "test.value",
        index_by_source=True,
    )
    latest_declaration = DatasetOutputDeclaration(
        "latest",
        "test.latest",
        index_by_source=True,
    )

    class Source:
        instance_id = "indexed-source"
        dataset_output_declarations = (source_declaration,)

        @staticmethod
        def signal_key(name: str) -> str:
            return f"indexed-source/{name}"

    class Derived:
        instance_id = "indexed-derived"
        dataset_output_declarations = (derived_declaration, latest_declaration)

        @staticmethod
        def signal_key(name: str) -> str:
            return f"indexed-derived/{name}"

        @staticmethod
        def validate_processor_source(_source) -> None:
            return None

        @staticmethod
        def evaluate_processor(_source, _publication):
            raise AssertionError("paused processor must not evaluate")

        @staticmethod
        def accept_processor_result(_source, _publication, _result) -> None:
            return None

        @staticmethod
        def accept_processor_failure(error: Exception) -> None:
            raise error

        @staticmethod
        def accept_processor_cancelled() -> None:
            return None

        @staticmethod
        def request_processor_owner_wake() -> None:
            return None

    source = Source()
    derived = Derived()
    plane = SignalDataPlane()
    wakes: list[int] = []
    unsubscribe = plane.subscribe_publications(lambda: wakes.append(1))
    history = small_history = None

    def settings(epoch: int) -> dict[str, object]:
        return {
            "device_settings": {
                "camera": {
                    "device_session_id": "camera-session",
                    "epoch_ranges": ((epoch, epoch),),
                }
            }
        }
    try:
        plane.reserve(source)
        plane.commit_live(source, {"frame": _latest(source_declaration, 1.0)})
        first = plane.latest_publication("indexed-source/frame")
        assert first is not None
        plane.attach_latest_only_processor(
            derived,
            source_name="indexed-source/frame",
            initial_publication=first,
            paused=True,
        )
        plane.commit_processor(
            derived,
            {
                "value": _latest(derived_declaration, 11.0),
                "latest": _latest(latest_declaration, 111.0),
            },
            source_publication=first,
        )
        plane.commit_live(
            source,
            {"frame": _latest(source_declaration, 2.0)},
        )
        second = plane.latest_publication("indexed-source/frame")
        assert second is not None
        plane.commit_processor(
            derived,
            {
                "value": _latest(
                    derived_declaration, 22.0, event_record=settings(2)
                ),
                "latest": _latest(
                    latest_declaration, 222.0, event_record=settings(2)
                ),
            },
            source_publication=second,
        )
        second_derived = plane.latest_publication("indexed-derived/value")
        assert second_derived is not None
        before_demand = plane.current_dataset("indexed-derived/value")
        assert before_demand.block.values.item() == 22.0
        assert all(
            str(column.coordinate_id) != "zlc_data.primary-index"
            for column in before_demand.block.schema.point_table.columns
        )
        assert plane.supports_indexed_history("indexed-derived/value")
        history = plane.acquire_indexed_history("indexed-derived/value", 4)

        for revision in (3, 4):
            plane.commit_live(
                source,
                {"frame": _latest(source_declaration, float(revision))},
            )
        fourth = plane.latest_publication("indexed-source/frame")
        assert fourth is not None
        plane.commit_processor(
            derived,
            {
                "value": _latest(
                    derived_declaration, 44.0, event_record=settings(4)
                ),
                "latest": _latest(
                    latest_declaration, 444.0, event_record=settings(4)
                ),
            },
            source_publication=fourth,
        )

        publication = plane.latest_publication("indexed-derived/value")
        assert publication is not None
        snapshot, snapshot_record = plane.current_dataset_view(
            "indexed-derived/value",
            publication,
        )
        source_index = snapshot.block.schema.point_table.column(
            AxisId("zlc_data.primary-index")
        )
        assert source_index.role == PRIMARY_INDEX
        assert source_index.values == (2, 3, 4)
        np.testing.assert_allclose(
            snapshot.block.values.reshape(-1),
            (22.0, 0.0, 44.0),
        )
        np.testing.assert_array_equal(
            snapshot.expanded_validity().reshape(-1),
            (True, False, True),
        )
        assert snapshot_record["device_settings"]["camera"]["epoch_ranges"] == (
            (2, 2),
            (4, 4),
        )
        old_publication = publication
        plane.commit_processor(
            derived,
            {
                "value": _latest(
                    derived_declaration, 55.0, event_record=settings(5)
                ),
                "latest": _latest(
                    latest_declaration, 555.0, event_record=settings(5)
                ),
            },
            source_publication=fourth,
            trigger=("refit", 1),
        )
        new_publication = plane.latest_publication("indexed-derived/value")
        assert new_publication is not None
        old_view, old_record = plane.current_dataset_view(
            "indexed-derived/value", old_publication
        )
        new_view, new_record = plane.current_dataset_view(
            "indexed-derived/value", new_publication
        )
        assert old_view.block.schema == new_view.block.schema
        np.testing.assert_allclose(old_view.block.values.reshape(-1)[-1], 44.0)
        np.testing.assert_allclose(new_view.block.values.reshape(-1)[-1], 55.0)
        assert bool(old_view.expanded_validity().reshape(-1)[-1])
        assert bool(new_view.expanded_validity().reshape(-1)[-1])
        assert old_record["device_settings"]["camera"]["epoch_ranges"] == (
            (2, 2),
            (4, 4),
        )
        assert new_record["device_settings"]["camera"]["epoch_ranges"] == (
            (2, 2),
            (5, 5),
        )
        latest_value = new_publication.value("indexed-derived/latest")
        assert latest_value is not None and latest_value.primary_index == 4
        latest = plane.current_dataset("indexed-derived/latest", new_publication)
        assert latest is latest_value.snapshot
        assert latest.block.values.shape == (1, 1, 1)
        assert latest.block.values.item() == 555.0
        assert all(
            str(column.coordinate_id) != "zlc_data.primary-index"
            for column in latest.block.schema.point_table.columns
        )
        small_history = plane.acquire_indexed_history("indexed-derived/value", 2)
        cached_view = plane.current_dataset("indexed-derived/value", new_publication)
        small_history.resize(3)
        assert (
            plane.current_dataset("indexed-derived/value", new_publication)
            is cached_view
        )
        small_history.resize(2)
        history.close()
        trimmed, trimmed_record = plane.current_dataset_view(
            "indexed-derived/value"
        )
        assert trimmed.block.schema.point_table.column(
            AxisId("zlc_data.primary-index")
        ).values == (3, 4)
        assert trimmed_record["device_settings"]["camera"]["epoch_ranges"] == (
            (5, 5),
        )
        with pytest.raises(ValueError, match="precedes retained"):
            plane.current_dataset("indexed-derived/value", second_derived)
        small_history.close()
        released = plane.current_dataset("indexed-derived/value")
        assert released.block.values.item() == 55.0
        assert all(
            str(column.coordinate_id) != "zlc_data.primary-index"
            for column in released.block.schema.point_table.columns
        )
        assert len(wakes) == 8  # four source and four atomic derived publications
    finally:
        if history is not None:
            history.close()
        if small_history is not None:
            small_history.close()
        unsubscribe()
        plane.close()


def test_indexed_history_retains_only_the_requested_window() -> None:
    source_declaration = DatasetOutputDeclaration("frame", "test.frame")
    derived_declaration = DatasetOutputDeclaration(
        "value",
        "test.value",
        index_by_source=True,
    )
    source = _node("bounded-source", source_declaration)

    class Derived:
        instance_id = "bounded-derived"
        dataset_output_declarations = (derived_declaration,)

        @staticmethod
        def signal_key(name: str) -> str:
            return f"bounded-derived/{name}"

        validate_processor_source = staticmethod(lambda _source: None)
        evaluate_processor = staticmethod(
            lambda _source, _publication: (_ for _ in ()).throw(AssertionError())
        )
        accept_processor_result = staticmethod(lambda *_args: None)
        accept_processor_failure = staticmethod(lambda error: (_ for _ in ()).throw(error))
        accept_processor_cancelled = staticmethod(lambda: None)
        request_processor_owner_wake = staticmethod(lambda: None)

    derived = Derived()
    plane = SignalDataPlane()
    history = None
    try:
        plane.reserve(source)
        plane.commit_live(source, {"frame": _latest(source_declaration, 1.0)})
        publication = plane.latest_publication("bounded-source/frame")
        assert publication is not None
        plane.attach_latest_only_processor(
            derived,
            source_name="bounded-source/frame",
            initial_publication=publication,
            paused=True,
        )
        for revision in range(1, 151):
            if revision > 1:
                plane.commit_live(
                    source,
                    {"frame": _latest(source_declaration, float(revision))},
                )
                publication = plane.latest_publication("bounded-source/frame")
                assert publication is not None
            plane.commit_processor(
                derived,
                {"value": _latest(derived_declaration, float(revision))},
                source_publication=publication,
            )
            if revision == 1:
                history = plane.acquire_indexed_history(
                    "bounded-derived/value",
                    100,
                )
        snapshot = plane.current_dataset("bounded-derived/value")
        primary = snapshot.block.schema.point_table.column(
            AxisId("zlc_data.primary-index")
        )
        assert primary.values == tuple(range(51, 151))
    finally:
        if history is not None:
            history.close()
        plane.close()


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
    with pytest.raises(ValueError, match="requires canonical placement"):
        LiveDatasetOutput(
            declaration,
            _event("unplaced", 0.0),
            DatasetCoverage(1, 1),
        )
    node = _node("camera", declaration)
    mutable = {"camera": {"gain": 1}, "shape": [1, 1]}
    record = MappingProxyType(mutable)
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
        mutable["camera"]["gain"] = 9
        mutable["new"] = "late"
        assert value.run_record["camera"]["gain"] == 1
        assert "new" not in value.run_record
        assert isinstance(value.run_record, MappingProxyType)
        assert isinstance(value.run_record["camera"], MappingProxyType)
        assert value.snapshot.ref.stream_generation == generation
        assert value.snapshot.ref.revision.value == 1
        assert value.snapshot.ref.block_id == BlockId("camera/frame.event")
        assert plane.seal_committed(node)
    finally:
        plane.close()


def test_finite_prefix_merges_event_epochs_without_changing_run_identity() -> None:
    declaration = DatasetOutputDeclaration("frame", "test.frame")
    node = _node("epoch-camera", declaration)
    plane = SignalDataPlane()

    def event(epoch: int) -> dict[str, object]:
        return {
            "device_settings": {
                "camera": {
                    "device_session_id": "camera-session",
                    "epoch_ranges": [[epoch, epoch]],
                    "mixed": False,
                }
            }
        }

    try:
        plane.reserve(node)
        first = plane.commit_live(
            node,
            {
                "frame": _finite(
                    declaration,
                    value=1.0,
                    total=2,
                    origin=0,
                    written=1,
                    run_record={"run": "same"},
                    event_record=event(0),
                )
            },
        )["epoch-camera/frame"]
        second = plane.commit_live(
            node,
            {
                "frame": _finite(
                    declaration,
                    value=2.0,
                    total=2,
                    origin=1,
                    written=2,
                    run_record={"run": "same"},
                    event_record=event(2),
                )
            },
        )["epoch-camera/frame"]
        assert first.event_record["device_settings"]["camera"][
            "epoch_ranges"
        ] == ((0, 0),)
        camera = second.event_record["device_settings"]["camera"]
        assert camera["epoch_ranges"] == ((2, 2),)
        assert camera["mixed"] is False
        publication = plane.latest_publication("epoch-camera/frame")
        assert publication is not None
        _snapshot, prefix_record = plane.current_dataset_view(
            "epoch-camera/frame",
            publication,
        )
        prefix_camera = prefix_record["device_settings"]["camera"]
        assert prefix_camera["epoch_ranges"] == ((0, 0), (2, 2))
        assert prefix_camera["mixed"] is True
        assert first.run_record == second.run_record == {"run": "same"}
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
            for _sequence, value, _origin, _parents in state.commit_chunks[
                "linear/frame"
            ]
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


def test_late_exact_replay_keeps_slim_causal_roots_and_drops_monitor_sibling() -> None:
    source_declaration = DatasetOutputDeclaration("frame", "test.frame")
    history_declaration = DatasetOutputDeclaration("history", "test.history")
    phase_declaration = DatasetOutputDeclaration("phase", "test.phase")
    source = _node("causal-source", source_declaration)
    first_processor = _node(
        "causal-first",
        history_declaration,
        phase_declaration,
    )
    downstream = _node(
        "causal-downstream",
        DatasetOutputDeclaration("result", "test.result"),
    )
    plane = SignalDataPlane()
    first_tap = None
    downstream_tap = None
    try:
        plane.reserve(source)
        plane.commit_live(
            source,
            {
                "frame": _finite(
                    source_declaration,
                    value=1.0,
                    total=2,
                    origin=0,
                    written=1,
                )
            },
        )
        first_root = plane.latest_publication("causal-source/frame")
        assert first_root is not None
        first_tap = plane.reserve_follow_processor(
            first_processor,
            source_name="causal-source/frame",
            source_publication=first_root,
        )
        plane.commit_processor(
            first_processor,
            {
                "history": _finite(
                    history_declaration,
                    value=10.0,
                    total=2,
                    origin=0,
                    written=1,
                ),
                "phase": _large_latest(phase_declaration, 100.0),
            },
            source_publication=first_root,
        )
        first_derived = plane.latest_publication("causal-first/history")
        assert first_derived is not None
        phase = first_derived.value("causal-first/phase")
        assert phase is not None

        plane.commit_live(
            source,
            {
                "frame": _finite(
                    source_declaration,
                    value=2.0,
                    total=2,
                    origin=1,
                    written=2,
                )
            },
        )
        second_root = plane.latest_publication("causal-source/frame")
        assert second_root is not None
        plane.commit_processor(
            first_processor,
            {
                "history": _finite(
                    history_declaration,
                    value=20.0,
                    total=2,
                    origin=1,
                    written=2,
                ),
                "phase": _large_latest(phase_declaration, 200.0),
            },
            source_publication=second_root,
        )
        latest_derived = plane.latest_publication("causal-first/history")
        assert latest_derived is not None
        del phase, first_derived

        downstream_tap = plane.reserve_follow_processor(
            downstream,
            source_name="causal-first/history",
            source_publication=latest_derived,
        )
        replayed = downstream_tap.next(timeout=0.0)
        assert replayed.event_ref.sequence == 1
        assert replayed.value("causal-first/phase") is None
        assert replayed.direct_parent_refs == (first_root.event_ref,)
        assert plane.publication_roots(replayed) == frozenset(
            {first_root.event_ref}
        )
    finally:
        if downstream_tap is not None:
            downstream_tap.close()
        if first_tap is not None:
            first_tap.close()
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


def test_freeze_preserves_publication_committed_while_processor_route_runs(
    monkeypatch,
) -> None:
    declaration = DatasetOutputDeclaration("frame", "test.frame")
    source = _node("route-race-source", declaration)
    plane = SignalDataPlane()
    route_entered = threading.Event()
    release_route = threading.Event()
    routed_sequences: list[int] = []
    freeze_errors: list[BaseException] = []
    real_route = plane._lane.route

    def gated_route(publications) -> None:
        publication = publications[source.signal_key("frame")]
        routed_sequences.append(publication.event_ref.sequence)
        if len(routed_sequences) == 1:
            route_entered.set()
            assert release_route.wait(2.0), "processor route gate did not open"
        real_route(publications)

    monkeypatch.setattr(plane._lane, "route", gated_route)
    try:
        plane.reserve(source)
        plane.commit_live(source, {"frame": _latest(declaration, 1.0)})

        def freeze_first_publication() -> None:
            try:
                plane.freeze()
            except BaseException as error:
                freeze_errors.append(error)

        worker = threading.Thread(target=freeze_first_publication)
        worker.start()
        assert route_entered.wait(2.0), "freeze never entered processor routing"

        # This commit lands after freeze cleared the work it captured, but
        # before it publishes the newly built front.  It must leave another
        # routing turn owed instead of being cleared by freeze's second lock.
        plane.commit_live(source, {"frame": _latest(declaration, 2.0)})
        release_route.set()
        worker.join(2.0)
        assert not worker.is_alive(), "freeze did not leave the route gate"
        assert not freeze_errors

        plane.freeze()
        assert routed_sequences == [1, 2]
    finally:
        release_route.set()
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
