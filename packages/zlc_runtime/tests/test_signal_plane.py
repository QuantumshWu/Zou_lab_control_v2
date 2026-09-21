"""Canonical commit/current/seal contracts for the signal plane."""

from __future__ import annotations

import gc
from dataclasses import replace
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
    INVALID,
    VALID,
    AxisId,
    AxisSpec,
    BlockId,
    CellValidity,
    DataBlock,
    DatasetRevision,
    DatasetComponentValidity,
    DatasetSchema,
    IndexedWindow,
    DomainSpec,
    OwnedSnapshot,
    SCALAR_DOMAIN,
    StreamGenerationId,
    ValidityContract,
    ValueSchema,
)
from zlc_data.snapshot_projection import PRIMARY_INDEX_AXIS_ID
from zlc_runtime.dataset import DatasetCoverage, MonitorCoverage
from zlc_runtime.dataset_output import DatasetOutputDeclaration, LiveDatasetOutput
from zlc_runtime.plane import SignalDataPlane


def _event(name: str, value: float) -> OwnedSnapshot:
    repeat = AxisSpec(AxisId(f"{name}.repeat"), "repeat", REPEAT, 1, (0,))
    point = AxisSpec(
        AxisId(f"{name}.point"),
        "point",
        SCAN_POINT,
        1,
        (0,),
    )
    schema = DatasetSchema(
        DomainSpec((1,), (repeat,), ((0,),)),
        DomainSpec((1,), (point,), ((0,),)),
        SCALAR_DOMAIN,
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
    event_record: dict[str, object] | None = None,
) -> LiveDatasetOutput:
    event = _event(declaration.name, value)
    schema = event.block.schema
    (repeat,) = schema.repeat_domain.axes
    canonical = DatasetSchema(
        DomainSpec(
            (total,),
            (
                AxisSpec(
                    repeat.axis_id,
                    repeat.name,
                    repeat.role,
                    total,
                    tuple(range(total)),
                ),
            ),
            (tuple(range(total)),),
        ),
        schema.point_domain,
        schema.cell_domain,
        schema.value_schema,
    )
    return LiveDatasetOutput(
        declaration,
        event,
        DatasetCoverage(written, total),
        canonical,
        (origin, 0),
        event_record,
    )


def _latest(
    declaration: DatasetOutputDeclaration,
    value: float,
    *,
    event_record: dict[str, object] | None = None,
) -> LiveDatasetOutput:
    return LiveDatasetOutput(
        declaration,
        _event(declaration.name, value),
        MonitorCoverage(1, 1),
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
        scalar.block.schema.repeat_domain,
        scalar.block.schema.point_domain,
        DomainSpec((data_axis.size,), (data_axis,)),
        ValueSchema(
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


def _camera_epoch_record(epoch: int) -> dict[str, object]:
    """The event record of one shot taken at one camera settings epoch."""

    return {
        "device_settings": {
            "camera": {
                "device_session_id": "camera-session",
                "epoch_ranges": ((epoch, epoch),),
            }
        }
    }


def _paused_lane(instance_id: str, declaration: DatasetOutputDeclaration):
    """A latest-lane node whose results are committed by the test itself.

    The paused lane is the existing public contract for a display-paced
    derivation that already holds its answers; evaluating is an error.
    """

    def refuse(*_arguments):
        raise AssertionError("paused processor must not evaluate")

    return SimpleNamespace(
        instance_id=instance_id,
        dataset_output_declarations=(declaration,),
        signal_key=lambda name: f"{instance_id}/{name}",
        validate_processor_source=lambda _source: None,
        evaluate_processor=refuse,
        accept_processor_result=lambda *_arguments: None,
        accept_processor_failure=refuse,
        accept_processor_cancelled=lambda: None,
        accept_processor_ended=lambda _error: None,
        request_processor_owner_wake=lambda: None,
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

        accept_processor_ended = staticmethod(lambda _error: None)

        @staticmethod
        def request_processor_owner_wake() -> None:
            return None

    source = Source()
    derived = Derived()
    plane = SignalDataPlane()
    wakes: list[int] = []
    unsubscribe = plane.subscribe_publications(lambda: wakes.append(1))
    history = small_history = None

    try:
        plane.begin_generation(source)
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
                    derived_declaration, 22.0, event_record=_camera_epoch_record(2)
                ),
                "latest": _latest(
                    latest_declaration, 222.0, event_record=_camera_epoch_record(2)
                ),
            },
            source_publication=second,
        )
        second_derived = plane.latest_publication("indexed-derived/value")
        assert second_derived is not None
        before_demand = plane.current_dataset("indexed-derived/value")
        assert before_demand.block.values.item() == 22.0
        assert all(
            str(axis.axis_id) != "zlc_data.primary-index"
            for axis in before_demand.block.schema.point_domain.axes
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
                    derived_declaration, 44.0, event_record=_camera_epoch_record(4)
                ),
                "latest": _latest(
                    latest_declaration, 444.0, event_record=_camera_epoch_record(4)
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
        source_index = snapshot.block.schema.point_domain.axis(
            AxisId("zlc_data.primary-index")
        )
        assert source_index.role == PRIMARY_INDEX
        assert tuple(source_index.coordinate_values()) == (-2, -1, 0)
        np.testing.assert_allclose(
            snapshot.materialize().block.values.reshape(-1),
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
                    derived_declaration, 55.0, event_record=_camera_epoch_record(5)
                ),
                "latest": _latest(
                    latest_declaration, 555.0, event_record=_camera_epoch_record(5)
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
        np.testing.assert_allclose(old_view.materialize().block.values.reshape(-1)[-1], 44.0)
        np.testing.assert_allclose(new_view.materialize().block.values.reshape(-1)[-1], 55.0)
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
            str(axis.axis_id) != "zlc_data.primary-index"
            for axis in latest.block.schema.point_domain.axes
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
        assert tuple(trimmed.block.schema.point_domain.axis(
            AxisId("zlc_data.primary-index")
        ).coordinate_values()) == (-1, 0)
        assert trimmed_record["device_settings"]["camera"]["epoch_ranges"] == (
            (5, 5),
        )
        from zlc_runtime import RetainedPublicationExpired

        assert plane.retains("indexed-derived/value", new_publication)
        assert not plane.retains("indexed-derived/value", second_derived)
        with pytest.raises(RetainedPublicationExpired, match="precedes retained"):
            plane.current_dataset("indexed-derived/value", second_derived)
        small_history.close()
        released = plane.current_dataset("indexed-derived/value")
        assert released.block.values.item() == 55.0
        assert all(
            str(axis.axis_id) != "zlc_data.primary-index"
            for axis in released.block.schema.point_domain.axes
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
        accept_processor_ended = staticmethod(lambda _error: None)
        request_processor_owner_wake = staticmethod(lambda: None)

    derived = Derived()
    plane = SignalDataPlane()
    history = None
    try:
        plane.begin_generation(source)
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
        primary = snapshot.block.schema.point_domain.axis(
            AxisId("zlc_data.primary-index")
        )
        assert tuple(primary.coordinate_values()) == tuple(range(-99, 1))
    finally:
        if history is not None:
            history.close()
        plane.close()


def _indexed_lane(window: int, *, prefix: str):
    """A source, a paused derivation of it holding index 1, and a lease."""

    source_declaration = DatasetOutputDeclaration("frame", "test.frame")
    derived_declaration = DatasetOutputDeclaration(
        "value",
        "test.value",
        index_by_source=True,
    )
    source = _node(f"{prefix}-source", source_declaration)
    derived = _paused_lane(f"{prefix}-derived", derived_declaration)
    plane = SignalDataPlane()
    plane.begin_generation(source)
    plane.commit_live(source, {"frame": _latest(source_declaration, 1.0)})
    parent = plane.latest_publication(f"{prefix}-source/frame")
    assert parent is not None
    plane.attach_latest_only_processor(
        derived,
        source_name=f"{prefix}-source/frame",
        initial_publication=parent,
        paused=True,
    )
    plane.commit_processor(
        derived,
        {
            "value": _latest(
                derived_declaration, 1.0, event_record=_camera_epoch_record(1)
            )
        },
        source_publication=parent,
    )
    history = plane.acquire_indexed_history(f"{prefix}-derived/value", window)
    return plane, source, source_declaration, derived, derived_declaration, history


def test_a_source_that_jumped_past_the_cached_window_leaves_legal_holes() -> None:
    """A cached basis is reused only where the new window overlaps it.

    A display-paced derivation may skip source indices, and its last
    materialization is the basis of the next.  Admitted on sequence and a
    forward start alone, a basis that ended BEFORE the new window began was
    still sliced for its overlap: an empty slice broadcast into two cells
    of the three-wide window, and the skipped indices -- invalid slots by
    contract -- came back as a materialization error instead.
    """

    plane, source, source_declaration, derived, derived_declaration, history = (
        _indexed_lane(3, prefix="jump")
    )
    try:
        cached = plane.current_dataset("jump-derived/value")
        assert cached.materialize().block.values.reshape(-1).tolist() == [1.0]
        for index in range(2, 6):
            plane.commit_live(
                source, {"frame": _latest(source_declaration, float(index))}
            )
        plane.commit_processor(
            derived,
            {"value": _latest(derived_declaration, 5.0)},
            source_publication=plane.latest_publication("jump-source/frame"),
        )
        snapshot = plane.current_dataset("jump-derived/value")
        assert tuple(snapshot.block.schema.point_domain.axis(
            AxisId("zlc_data.primary-index")
        ).coordinate_values()) == (-2, -1, 0)
        np.testing.assert_allclose(
            snapshot.materialize().block.values.reshape(-1), (0.0, 0.0, 5.0)
        )
        np.testing.assert_array_equal(
            snapshot.expanded_validity().reshape(-1), (False, False, True)
        )
    finally:
        history.close()
        plane.close()


def test_a_rolled_window_s_record_names_only_the_rows_it_kept() -> None:
    """Provenance is a fact of the window, not of how often it was read.

    Reading after every shot makes each materialization the basis of the
    next.  The VALUES of a rolled window copy only the rows that stayed;
    the record was merged from the basis's whole record plus the new
    events -- a union of epoch ranges, which cannot subtract the rows that
    left -- so a two-shot window read every shot claimed epochs 1 to 4
    while the same window read once claimed 3 to 4.
    """

    records = {}
    for read_every_shot in (True, False):
        plane, source, source_declaration, derived, derived_declaration, history = (
            _indexed_lane(2, prefix="roll")
        )
        try:
            if read_every_shot:
                plane.current_dataset_view("roll-derived/value")
            for index in (2, 3, 4):
                plane.commit_live(
                    source, {"frame": _latest(source_declaration, float(index))}
                )
                plane.commit_processor(
                    derived,
                    {
                        "value": _latest(
                            derived_declaration,
                            float(index),
                            event_record=_camera_epoch_record(index),
                        )
                    },
                    source_publication=plane.latest_publication(
                        "roll-source/frame"
                    ),
                )
                if read_every_shot:
                    plane.current_dataset_view("roll-derived/value")
            snapshot, deferred = plane.current_dataset_view("roll-derived/value", defer_record=True)
            assert snapshot.block.values is None
        finally:
            history.close()
            plane.close()
        # The captured record belongs to this exact window, even after its
        # source and lease are gone; it never re-queries a later publication.
        assert snapshot.materialize().block.values.reshape(-1).tolist() == [3.0, 4.0]
        record = deferred() if callable(deferred) else deferred
        records[read_every_shot] = record
        assert (deferred() if callable(deferred) else deferred) == record
    for record in records.values():
        assert record["device_settings"]["camera"]["epoch_ranges"] == ((3, 4),)
    assert records[True] == records[False]


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
        schema.repeat_domain,
        DomainSpec(
            (4,),
            (
                AxisSpec(x_id, "x", SCAN_POINT, 2, (0.0, 1.0)),
                AxisSpec(y_id, "y", SCAN_POINT, 2, (0.0, 1.0)),
            ),
            ((0, 1, 0, 1), (0, 0, 1, 1)),
        ),
        schema.cell_domain,
        schema.value_schema,
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
        generation = plane.begin_generation(node)
        plane.set_run_record(node, record)
        with pytest.raises(RuntimeError, match="already been declared"):
            plane.set_run_record(node, record)
        value = plane.commit_live(
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
        )["camera/frame"]
        mutable["camera"]["gain"] = 9
        mutable["new"] = "late"
        assert value.run_record["camera"]["gain"] == 1
        assert "new" not in value.run_record
        assert isinstance(value.run_record, MappingProxyType)
        assert isinstance(value.run_record["camera"], MappingProxyType)
        publication = plane.latest_publication("camera/frame")
        assert publication.run_record is value.run_record
        assert publication.event_record is value.event_record
        assert value.snapshot.ref.stream_generation == generation
        assert value.snapshot.ref.revision.value == 1
        assert value.snapshot.ref.block_id == BlockId("camera/frame.event")
        assert plane.seal_committed(node)
    finally:
        plane.close()


def test_finite_prefix_merges_event_epochs_without_changing_run_identity(monkeypatch) -> None:
    import zlc_runtime.plane as plane_module

    declaration = DatasetOutputDeclaration("frame", "test.frame")
    sibling = DatasetOutputDeclaration("counts", "test.counts")
    node = _node("epoch-camera", declaration, sibling)
    plane = SignalDataPlane()
    merge_calls = []
    real_merge = plane_module._merge_event_records

    def counted(records):
        records = tuple(records)
        merge_calls.append(len(records))
        return real_merge(records)

    monkeypatch.setattr(plane_module, "_merge_event_records", counted)

    def event(epoch: int) -> dict[str, object]:
        return {
            "record_timing": {"camera": {str(epoch): {"record_time_seconds": float(epoch)}}},
            "device_settings": {
                "camera": {
                    "device_session_id": "camera-session",
                    "epoch_ranges": [[epoch, epoch]],
                    "mixed": False,
                }
            }
        }

    try:
        plane.begin_generation(node)
        plane.set_run_record(node, {"run": "same"})
        first = plane.commit_live(
            node,
            {
                "frame": _finite(
                    declaration,
                    value=1.0,
                    total=2,
                    origin=0,
                    written=1,
                    event_record=event(0),
                ),
                "counts": _finite(sibling, value=3.0, total=2, origin=0, written=1, event_record=event(0)),
            },
        )["epoch-camera/frame"]
        first_publication = plane.latest_publication("epoch-camera/frame")
        _, first_record = plane.current_dataset_view("epoch-camera/frame")
        assert plane.current_dataset_view("epoch-camera/counts")[1] is first_record
        assert merge_calls == [1]
        second = plane.commit_live(
            node,
            {
                "frame": _finite(
                    declaration,
                    value=2.0,
                    total=2,
                    origin=1,
                    written=2,
                    event_record=event(2),
                ),
                "counts": _finite(sibling, value=4.0, total=2, origin=1, written=2, event_record=event(2)),
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
        prepared = plane.current_dataset("epoch-camera/frame")
        assert merge_calls == [1]
        _snapshot, prefix_record = plane.current_dataset_view(
            "epoch-camera/frame",
            publication,
        )
        prefix_camera = prefix_record["device_settings"]["camera"]
        assert _snapshot is prepared
        assert plane.current_dataset_view("epoch-camera/counts")[1] is prefix_record
        assert merge_calls == [1, 2]
        assert prefix_camera["epoch_ranges"] == ((0, 0), (2, 2))
        assert prefix_camera["mixed"] is True
        assert first.run_record is second.run_record
        for value, key in ((first, "0"), (second, "2")):
            assert prefix_record["record_timing"]["camera"][key] == value.event_record["record_timing"]["camera"][key]
        with pytest.raises(TypeError):
            prefix_record["record_timing"]["camera"]["0"]["record_time_seconds"] = 99
        assert first.canonical_schema is second.canonical_schema
        assert first.run_record == {"run": "same"}
        assert tuple(first_record["record_timing"]["camera"]) == ("0",)
        old_snapshot, old_record = plane.current_dataset_view("epoch-camera/frame", first_publication)
        assert tuple(old_record["record_timing"]["camera"]) == ("0",)
        assert old_snapshot.expanded_validity()[:, 0, 0].tolist() == [True, False]
        assert plane.current_dataset_view("epoch-camera/frame")[1] is prefix_record
    finally:
        plane.close()


def test_partial_current_has_invalid_future_and_overlap_is_rejected() -> None:
    declaration = DatasetOutputDeclaration("frame", "test.frame")
    node = _node("partial", declaration)
    plane = SignalDataPlane()
    try:
        plane.begin_generation(node)
        output = _finite(declaration, value=10.0, total=4, origin=0, written=1)
        snapshot = output.snapshot
        output = replace(output, snapshot=OwnedSnapshot(
            snapshot.ref, snapshot.block.replacing(validity=VALID)))
        plane.commit_live(
            node,
            {"frame": output},
        )
        current = plane.current_dataset("partial/frame")
        assert current.materialize().block.values[:, 0, 0].tolist() == [10.0, 0.0, 0.0, 0.0]
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
        plane.begin_generation(node)
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
        assert current.block.schema.physical_shape == (30, 1, 1)
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
        plane.begin_generation(node)
        waiting_directory = plane.describe_signals()
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
        directory = plane.describe_signals()
        assert directory is not waiting_directory
        assert plane.describe_signals() is directory
        description = directory[0]
        assert description.shape == (1, 4, 1)
        first = plane.current_dataset(description.name)
        assert first.block.schema.point_domain.logical_shape == (2, 2)
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
        assert plane.describe_signals() is directory, "new values do not rebuild the directory"
        assert second.materialize().block.values[0, :, 0].tolist() == [10.0, 0.0, 0.0, 40.0]
        assert second.expanded_validity()[0, :, 0].tolist() == [
            True,
            False,
            False,
            True,
        ]
        assert plane.seal_committed(node, cut_short=True)
        assert plane.describe_signals() is not directory
        assert plane.describe_signals()[0].shape == (1, 4, 1)
    finally:
        plane.close()


@pytest.mark.parametrize("point_codes", ((0, 1), (0, 1, 0)))
@pytest.mark.parametrize("mask_kind", ("components", "subset", "cell", "global"))
def test_repeat_counts_follow_written_cells_not_survival_eligibility(point_codes, mask_kind) -> None:
    declaration = DatasetOutputDeclaration("survival", "test.survival")
    node = _node("repeat-title", declaration)
    plane = SignalDataPlane()
    repeat = AxisSpec(AxisId("scan"), "repeat", REPEAT, 3, (0, 1, 2))
    run = AxisSpec(AxisId("run"), "repeat", REPEAT, 2, (0, 1))
    point = AxisSpec(AxisId("power"), "power", SCAN_POINT, 2, (135, 247))
    site = AxisSpec(AxisId("site"), "site", SPATIAL_X, 2, (0, 1))
    channel = AxisSpec(AxisId("channel"), "channel", SPATIAL_X, 3, (0, 1, 2))
    canonical = DatasetSchema(
        DomainSpec((6,), (repeat, run), ((0, 0, 1, 1, 2, 2), (0, 1, 0, 1, 0, 1))),
        DomainSpec((len(point_codes),), (point,), (point_codes,)), DomainSpec((2, 3), (site, channel)),
        ValueSchema(ValidityContract.components(site.axis_id, channel.axis_id), np.dtype(bool)),
    )
    scalar_event = _event("survival", 0.0).block.schema
    event_schema = replace(canonical, repeat_domain=scalar_event.repeat_domain,
                           point_domain=scalar_event.point_domain)
    # Missing scan cells stay absent; false/invalid sites in a committed cell
    # do not erase the acquired trial. The two Repeat axes share a name.
    steps = ((0, 0, (1, 1)), (0, 1, (1, 1)), (1, 0, (1, 2)),
             (2, 0, (2, 1)), (3, 1, (1, 1)), (4, 0, (3, 1)), (5, 0, (2, 2)))
    if len(point_codes) == 3:
        steps += ((5, 2, (2, 2)),)
    saved = []
    try:
        plane.begin_generation(node)
        for written, (row, column, expected) in enumerate(steps, 1):
            eligible = np.array([[[written % 2 == 0, False]]], dtype=bool)
            validity = (INVALID if mask_kind == "global" else
                        CellValidity(np.zeros((1, 1), dtype=bool)) if mask_kind == "cell" else
                        DatasetComponentValidity((site.axis_id,), eligible) if mask_kind == "subset" else
                        DatasetComponentValidity((site.axis_id, channel.axis_id), np.broadcast_to(eligible[..., None], (1, 1, 2, 3))))
            block = DataBlock(BlockId("survival"), DatasetRevision(0),
                              np.zeros((1, 1, 2, 3), dtype=bool), validity, event_schema,
                              window=IndexedWindow(written - 1, written, written - 1))
            snapshot = OwnedSnapshot(block.ref(StreamGenerationId("source")), block)
            value = plane.commit_live(node, {"survival": LiveDatasetOutput(
                declaration, snapshot, DatasetCoverage(written, 6 * len(point_codes)),
                canonical_schema=canonical, cell_origin=(row, column),
                shot_time_seconds=written / 10,
            )})[node.signal_key("survival")]
            assert value.repeat_counts == expected
            saved.append((value, expected))
        _, tap = plane.follow_publications(node.signal_key("survival"))
        assert plane.seal_committed(node, cut_short=True)
        snapshot = plane.current_dataset(node.signal_key("survival"))
        assert not snapshot.expanded_validity()[..., -1, :].any()
        assert all(value.repeat_counts == expected for value, expected in saved)
        try:
            for original, expected in saved:
                replayed = tap.next(0).value(node.signal_key("survival"))
                assert replayed.repeat_counts == expected
                assert replayed.coverage == original.coverage
                assert replayed.primary_index == original.primary_index
                assert replayed.shot_time == original.shot_time
                assert replayed.snapshot.block.window == original.snapshot.block.window
                np.testing.assert_array_equal(replayed.snapshot.expanded_validity(), original.snapshot.expanded_validity())
        finally:
            tap.close()
    finally:
        plane.close()


def test_watching_a_run_grow_costs_the_shot_not_the_run() -> None:
    """Assembling the view again places what ARRIVED, not everything.

    A canonical Dataset never rewrites a cell it already holds, so the
    previous assembly answers for every cell in it.  Re-placing all of
    them anyway made one redraw cost the whole run, which on a long scan
    is seconds a shot -- the panel falls behind the producer and never
    catches up, and the cost grows for as long as the run does.
    """

    from zlc_runtime import plane as plane_module

    declaration = DatasetOutputDeclaration("scan", "test.scan")
    node = _node("grid-cost", declaration)
    plane = SignalDataPlane()
    # Counted on EVERY assembler the plane owns, named or not, so the
    # measurement is of the work done and not of which function did it.
    placed: list[int] = []
    merged: list[int] = []
    merge_records = plane_module._merge_event_records

    def merge_new_records(records):
        records = tuple(records)
        merged.append(len(records))
        return merge_records(records)

    plane_module._merge_event_records = merge_new_records
    assemblers = {
        name: getattr(plane_module, name)
        for name in ("_assembled_planes", "_extended_planes")
        if hasattr(plane_module, name)
    }

    def counting(original):
        def assemble(*arguments):
            head, tail = arguments[:-1], tuple(arguments[-1])
            placed.append(len(tail))
            return original(*head, tail)

        return assemble

    for name, original in assemblers.items():
        setattr(plane_module, name, counting(original))
    try:
        plane.begin_generation(node)
        for point in range(4):
            plane.commit_live(
                node,
                {
                    "scan": _finite_grid_point(
                        declaration,
                        value=float(point),
                        point_origin=point,
                        written=point + 1,
                    )
                },
            )
            # A values-only consumer asks for the run without provenance.
            view = plane.current_dataset("grid-cost/scan")
            assert view.materialize().block.values[0, point, 0] == float(point)
        assert placed == [1, 1, 1, 1], placed
        assert merged == [], "snapshot-only reads must not build discarded event records"
        snapshot, record = plane.current_dataset_view("grid-cost/scan")
        assert snapshot is view, "requesting provenance must not rebuild prepared pixels"
        assert merged == [4]
        assert record == {}
        assert plane.current_dataset_view("grid-cost/scan")[1] is record
        assert merged == [4], "the same exact record is prepared once"
        np.testing.assert_allclose(
            plane.current_dataset("grid-cost/scan").materialize().block.values[0, :, 0],
            (0.0, 1.0, 2.0, 3.0),
        )
    finally:
        plane_module._merge_event_records = merge_records
        for name, original in assemblers.items():
            setattr(plane_module, name, original)
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
    real = plane_module.SignalDataPlane._materialize_dataset

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(plane_module.SignalDataPlane, "_materialize_dataset", staticmethod(counted))
    plane = SignalDataPlane()
    try:
        plane.begin_generation(node)
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
        assert second.materialize().block.values[:, 0, 0].tolist() == [1.0, 2.0, 0.0]
        assert plane.seal_committed(node, cut_short=True)
        assert calls == 2
    finally:
        plane.close()


def test_canonical_prefix_is_bound_to_its_publication_when_next_event_wins_race() -> None:
    declaration = DatasetOutputDeclaration("frame", "test.frame")
    node = _node("publication-prefix", declaration)
    plane = SignalDataPlane()
    try:
        plane.begin_generation(node)
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
        assert first.materialize().block.values[:, 0, 0].tolist() == [1.0, 0.0, 0.0]
        assert first.expanded_validity()[:, 0, 0].tolist() == [
            True,
            False,
            False,
        ]
        assert latest.materialize().block.values[:, 0, 0].tolist() == [1.0, 2.0, 0.0]
    finally:
        plane.close()


def test_repeat_100_publication_cost_and_retained_arrays_stay_linear(monkeypatch) -> None:
    import zlc_runtime.plane as plane_module

    declaration = DatasetOutputDeclaration("frame", "test.frame")
    node = _node("linear", declaration)
    calls = 0
    real = plane_module.SignalDataPlane._materialize_dataset

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(plane_module.SignalDataPlane, "_materialize_dataset", staticmethod(counted))
    plane = SignalDataPlane()
    try:
        plane.begin_generation(node)
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

        # Runtime retains exactly 100 one-cell byte planes plus one 100-cell
        # placement mask.  It has not built (or retained) 1+2+...+100 prefixes.
        state = plane._states["linear"]
        retained = state.commit_chunks["linear/frame"][0]
        assert len(retained) == 2 * 100
        assert state.materialized == {}
        assert state.publication is not None
        assert state.publication.value("linear/frame").snapshot.block.values.tobytes() == retained[-2]
        parts = iter(retained)
        retained_array_bytes = sum(
            len(values)
            for values, _sigma in zip(parts, parts)
        )
        placement_bytes = state.occupied_cells["linear/frame"].nbytes
        assert retained_array_bytes + placement_bytes == 100 * 8 + 100

        assert plane.seal_committed(node)
        assert calls == 0, "ending production must not consume an unread Dataset"
        assert state.materialized == {}
        current = plane.current_dataset("linear/frame")
        assert calls == 1
        assert current.expanded_validity().all()
    finally:
        plane.close()


@pytest.mark.parametrize("operation", ["_retained_snapshot", "_merge_event_records"])
def test_full_materialization_does_not_hold_plane_lock(monkeypatch, operation) -> None:
    import zlc_runtime.plane as plane_module

    declaration = DatasetOutputDeclaration("frame", "test.frame")
    node = _node("nonblocking-current", declaration)
    entered = threading.Event()
    release = threading.Event()
    reader_done = threading.Event()
    errors: list[BaseException] = []
    real = getattr(plane_module, operation)

    def gated(*args, **kwargs):
        entered.set()
        if not release.wait(2.0):
            raise TimeoutError("materialization gate did not open")
        return real(*args, **kwargs)

    plane = SignalDataPlane()
    try:
        plane.begin_generation(node)
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
        monkeypatch.setattr(plane_module, operation, gated)

        def materialize() -> None:
            try:
                plane.current_dataset_view(node.signal_key("frame"))
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
        plane.begin_generation(node)
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
        assert plane.current_dataset("mixed/history").materialize().block.values[:, 0, 0].tolist() == [
            1.0,
            2.0,
        ]
        assert plane.seal_committed(node)
    finally:
        plane.close()


def test_late_exact_replay_keeps_slim_causal_roots_and_drops_monitor_sibling() -> None:
    from zlc_runtime.streams import SourceFailed
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
    live_tap = overflow_tap = None
    try:
        plane.begin_generation(source)
        plane.commit_live(
            source,
            {
                "frame": _large_latest(source_declaration, 1.0)
            },
        )
        first_root = plane.latest_publication("causal-source/frame")
        assert first_root is not None
        first_pixels = weakref.ref(first_root.value("causal-source/frame").snapshot.block.values)
        first_event = first_root.event_ref
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
        _, live_tap = plane.follow_publications(
            "causal-first/history", replay=False, max_bytes=2 * 1024 * 1024,
        )
        _, overflow_tap = plane.follow_publications(
            "causal-first/history", replay=False, max_bytes=1024,
        )

        plane.commit_live(
            source,
            {
                "frame": _large_latest(source_declaration, 2.0)
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
        needed = live_tap.next(0.0)
        assert tuple(needed.signals) == ("causal-first/history",)
        # The large unused sibling is not queued; the needed scalar's actual
        # strong camera parent still counts towards its bounded input budget.
        with pytest.raises(SourceFailed, match="payload bytes"):
            overflow_tap.next(0.0)
        assert not overflow_tap._queue and overflow_tap._queued_bytes == 0
        del phase, first_derived, _
        first_tap.close()
        first_tap = None

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
        # An actually held Frozen/accepted parent still resolves its pixels.
        assert plane.direct_parent_publications(replayed)[0] is first_root
        del first_root
        gc.collect()
        assert first_pixels() is None, "finite scalar replay must not retain the old camera array"
        parent = plane.direct_parent_publications(replayed)[0]
        assert parent.event_ref == first_event
        assert parent.signal_names == ("causal-source/frame",)
        assert not parent.signals
        assert plane.publication_roots(replayed) == frozenset({first_event})
        visible = _node("visible", history_declaration, phase_declaration)
        plane.begin_generation(visible)
        plane.set_front_signals({"visible/history", "visible/phase"})
        _baseline, visible_tap = plane.follow_publications("visible/history", replay=False)
        try:
            plane.commit_live(visible, {
                "history": _latest(history_declaration, 3.0),
                "phase": _large_latest(phase_declaration, 300.0),
            })
            pending = visible_tap.next(0.0)
            assert set(pending.signals) == {"visible/history", "visible/phase"}
        finally:
            visible_tap.close()
    finally:
        if live_tap is not None:
            live_tap.close()
        if overflow_tap is not None:
            overflow_tap.close()
        if downstream_tap is not None:
            downstream_tap.close()
        if first_tap is not None:
            first_tap.close()
        plane.close()


@pytest.mark.parametrize("source_error", (None, RuntimeError("source failed after committed data")))
def test_latest_processors_run_parallel_per_node_serial_and_coalesce(source_error) -> None:
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
            self.ended: list[Exception | None] = []
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

        def accept_processor_ended(self, error: Exception | None) -> None:
            self.ended.append(error)

        def request_processor_owner_wake(self) -> None:
            self.wake.set()

    first = Processor("latest-first")
    second = Processor("latest-second")
    plane = SignalDataPlane()
    try:
        plane.begin_generation(source)
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

        plane.seal_committed(source, cut_short=True, error=source_error)
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
            assert len(processor.ended) == 1
            assert (processor.ended[0] is None) == (source_error is None)
            if source_error is not None:
                assert str(source_error) in str(processor.ended[0])
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
        plane.begin_generation(source)
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
        plane.begin_generation(node)
        plane.commit_live(node, {"preview": _latest(declaration, 1.0)})
        assert plane.retire(node) == frozenset({"preview/preview"})
        assert plane.latest_publication("preview/preview") is None
    finally:
        plane.close()


def test_slimming_reads_the_commit_s_recorded_selection_not_the_live_state() -> None:
    """The bench-killing crash: "derived publication has no selected source signal".

    A measurement producer commits with ``worker_source``: its publication
    is DERIVED (it consumed one signal of the worker's input publication),
    but that selection lived only inside the one ``commit_live`` call --
    the producer state's ``source_name`` is None.  When a downstream
    processor (a panel ROI) then committed, slimming the causal chain
    asked the LIVE state table for the selection, got None, and raised out
    of the commit.  The selection is a fact of the commit, recorded on the
    publication itself, so the walk now survives a live worker-fed
    producer and an ancestor stream that re-armed since -- the retained
    lineage still reaches the true root.
    """

    wire_declaration = DatasetOutputDeclaration("frame", "test.frame")
    measurement_declaration = DatasetOutputDeclaration("frame", "test.frame")
    roi_declaration = DatasetOutputDeclaration("value", "test.value")
    fit_declaration = DatasetOutputDeclaration("fit", "test.fit")
    wire = _node("wire", wire_declaration)
    measurement = _node("measurement", measurement_declaration)
    roi = _node("roi", roi_declaration)
    fit = _node("fit", fit_declaration)
    plane = SignalDataPlane()
    roi_tap = None
    fit_tap = None
    try:
        plane.begin_generation(wire)
        plane.commit_live(
            wire,
            {"frame": _finite(wire_declaration, value=1.0, total=1, origin=0, written=1)},
        )
        wire_publication = plane.latest_publication("wire/frame")
        assert wire_publication is not None

        plane.begin_generation(measurement)
        plane.commit_live(
            measurement,
            {
                "frame": _finite(
                    measurement_declaration, value=2.0, total=1, origin=0, written=1
                )
            },
            worker_source=("wire/frame", wire_publication),
        )
        measured = plane.latest_publication("measurement/frame")
        assert measured is not None
        assert measured.direct_parent_refs == (wire_publication.event_ref,)

        # The commit that crashed the bench: a processor over the live
        # worker-fed producer.
        roi_tap = plane.reserve_follow_processor(
            roi, source_name="measurement/frame", source_publication=measured
        )
        plane.commit_processor(
            roi,
            {"value": _finite(roi_declaration, value=3.0, total=1, origin=0, written=1)},
            source_publication=measured,
        )
        derived = plane.latest_publication("roi/value")
        assert derived is not None
        assert plane.publication_roots(derived) == frozenset(
            {wire_publication.event_ref}
        )

        # An ancestor stream that MOVED ON is the same fact from the other
        # side: the wire ends and re-arms, then a deeper commit walks the
        # retained chain -- whose selections were recorded, not re-derived.
        plane.seal_committed(wire)
        plane.begin_generation(wire)
        fit_tap = plane.reserve_follow_processor(
            fit, source_name="roi/value", source_publication=derived
        )
        plane.commit_processor(
            fit,
            {"fit": _finite(fit_declaration, value=4.0, total=1, origin=0, written=1)},
            source_publication=derived,
        )
        answered = plane.latest_publication("fit/fit")
        assert answered is not None
        assert plane.publication_roots(answered) == frozenset(
            {wire_publication.event_ref}
        )
    finally:
        if fit_tap is not None:
            fit_tap.close()
        if roi_tap is not None:
            roi_tap.close()
        plane.close()


def test_indexed_history_stamps_its_window_and_the_last_replacement() -> None:
    """The stamp a consumer keeps work across revisions by.

    ``start``..``latest`` name the retained shots; ``stable_since`` is -1
    until a retained index is overwritten, and then the sequence that
    overwrote it -- the fence past which no carried work is valid.
    """

    source_declaration = DatasetOutputDeclaration("frame", "test.frame")
    derived_declaration = DatasetOutputDeclaration(
        "value",
        "test.value",
        index_by_source=True,
    )
    source = _node("stamped-source", source_declaration)

    class Derived:
        instance_id = "stamped-derived"
        dataset_output_declarations = (derived_declaration,)

        @staticmethod
        def signal_key(name: str) -> str:
            return f"stamped-derived/{name}"

        validate_processor_source = staticmethod(lambda _source: None)
        evaluate_processor = staticmethod(
            lambda _source, _publication: (_ for _ in ()).throw(AssertionError())
        )
        accept_processor_result = staticmethod(lambda *_args: None)
        accept_processor_failure = staticmethod(lambda error: (_ for _ in ()).throw(error))
        accept_processor_cancelled = staticmethod(lambda: None)
        accept_processor_ended = staticmethod(lambda _error: None)
        request_processor_owner_wake = staticmethod(lambda: None)

    derived = Derived()
    plane = SignalDataPlane()
    history = None
    try:
        plane.begin_generation(source)
        plane.commit_live(source, {"frame": _latest(source_declaration, 1.0)})
        publication = plane.latest_publication("stamped-source/frame")
        plane.attach_latest_only_processor(
            derived,
            source_name="stamped-source/frame",
            initial_publication=publication,
            paused=True,
        )
        for revision in range(1, 7):
            if revision > 1:
                plane.commit_live(
                    source, {"frame": _latest(source_declaration, float(revision))}
                )
                publication = plane.latest_publication("stamped-source/frame")
            plane.commit_processor(
                derived,
                {"value": _latest(derived_declaration, float(revision))},
                source_publication=publication,
            )
            if revision == 1:
                history = plane.acquire_indexed_history("stamped-derived/value", 4)
        before = plane.current_dataset("stamped-derived/value")
        window = before.block.window
        assert window is not None
        assert window.latest - window.start == 3
        assert window.stable_since == -1

        # The latest index published again with another value -- a re-run
        # for the same parent -- is a REPLACEMENT of a retained shot, which
        # every table carried across revisions must notice.
        plane.commit_processor(
            derived,
            {"value": _latest(derived_declaration, 60.0)},
            source_publication=publication,
            trigger=("rerun", 1),
        )
        after = plane.current_dataset("stamped-derived/value")
        assert after.block.window.start == window.start
        assert after.block.window.latest == window.latest
        assert DatasetRevision(after.block.window.stable_since) == after.block.revision
        assert float(after.materialize().block.values.reshape(-1)[-1]) == 60.0
    finally:
        if history is not None:
            history.close()
        plane.close()


def test_a_stamped_history_window_carries_when_each_shot_was_taken() -> None:
    """A monitor that stamps its shots gets a shot-time axis beside the index.

    One coordinate per shot on the primary index's rows, in seconds from
    the run's first shot, so a panel can place the shots along the time
    they happened instead of counting them; a history that stamps none
    carries no such axis, and one cannot start stamping halfway.
    """

    from zlc_data import SHOT_TIME
    from zlc_data.snapshot_projection import SHOT_TIME_AXIS_ID, indexed_history_layout

    declaration = DatasetOutputDeclaration("field", "test.field", index_by_source=True)
    source = _node("stamped-source", declaration)
    plane = SignalDataPlane()
    lease = None
    try:
        plane.begin_generation(source)
        lease = plane.acquire_indexed_history("stamped-source/field", 3)
        for value, seconds in ((1.0, 0.0), (2.0, 0.1), (3.0, 0.25), (4.0, 0.4)):
            plane.commit_live(
                source,
                {
                    "field": LiveDatasetOutput(
                        declaration,
                        _event("field", value),
                        MonitorCoverage(1, 1),
                        shot_time_seconds=seconds,
                    )
                },
            )
        publication = plane.latest_publication("stamped-source/field")
        assert publication is not None
        snapshot, _record = plane.current_dataset_view("stamped-source/field", publication)
        schema = snapshot.block.schema
        times = schema.point_domain.axis(SHOT_TIME_AXIS_ID)
        assert times.role == SHOT_TIME and times.unit == "s"
        assert tuple(times.coordinate_values()) == (0.1, 0.25, 0.4)
        assert np.array_equal(
            schema.point_domain.codes(SHOT_TIME_AXIS_ID),
            schema.point_domain.codes(PRIMARY_INDEX_AXIS_ID),
        )
        layout = indexed_history_layout(schema)
        assert layout is not None and layout.times is not None
        assert layout.times.tolist() == [0.1, 0.25, 0.4]
        assert np.asarray(snapshot.materialize().block.values).reshape(-1).tolist() == [2.0, 3.0, 4.0]
        with pytest.raises(ValueError, match="every shot"):
            plane.commit_live(
                source,
                {"field": LiveDatasetOutput(declaration, _event("field", 5.0), MonitorCoverage(1, 1))},
            )
        with pytest.raises(ValueError, match="advance in time"):
            plane.commit_live(
                source,
                {
                    "field": LiveDatasetOutput(
                        declaration, _event("field", 5.0), MonitorCoverage(1, 1),
                        shot_time_seconds=0.4,
                    )
                },
            )
    finally:
        if lease is not None:
            lease.close()
        plane.close()


def test_a_stamped_window_gives_every_row_a_distinct_ordered_time() -> None:
    """Rows a history does not hold still get a coordinate on the time axis.

    A derivation that evaluated only the latest shot, or a source that
    skipped, leaves holes; their rows are invalid and never drawn, but an
    axis is a coordinate per row, unique and in order.  A hole between two
    held shots lies on the line between them; one before the first held
    shot lies a nanosecond earlier.
    """

    from zlc_runtime.plane import _row_times

    assert _row_times([0.1, 0.25, 0.4]) == (0.1, 0.25, 0.4)
    assert _row_times([0.1, None, None, 0.4]) == pytest.approx((0.1, 0.2, 0.3, 0.4))
    front = _row_times([None, None, 0.5, None, 0.9])
    assert front[2:] == (0.5, 0.7, 0.9)
    assert front[0] < front[1] < front[2]
    assert len(set(front)) == len(front)
    # Distinct at any magnitude: a run half a year in still tells its rows apart.
    late = _row_times([None, None, 16777217.0])
    assert late[0] < late[1] < late[2] and len(set(late)) == 3
