"""Current signal-plane contracts: exact publications and causal fronts."""

from __future__ import annotations

import ast
import os
import pathlib
import threading
import time
from types import SimpleNamespace

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
    OwnedSnapshot,
    PointColumn,
    PointTable,
    StreamGenerationId,
    ValueSchema,
)
from zlc_runtime.dataset_output import (
    DatasetOutputDeclaration,
    LiveDatasetOutput,
)
from zlc_runtime.plane import (
    SignalDataPlane,
    SignalFront,
    SignalPublication,
)
from zlc_runtime.dataset import MonitorCoverage


REPO = pathlib.Path(__file__).resolve().parents[1]


def _live_output(
    name: str,
    revision: int,
    *,
    generation: str | None = None,
) -> LiveDatasetOutput:
    repeat = AxisSpec(AxisId(f"{name}.repeat"), "repeat", REPEAT, 1, (0,))
    point = AxisSpec(AxisId(f"{name}.point"), "point", SCAN_POINT, 1, (0,))
    schema = DatasetSchema(
        repeat,
        PointTable(
            1,
            (
                PointColumn(
                    point.axis_id,
                    point.name,
                    point.role,
                    PointColumn.NUMERIC,
                    point.coordinates,
                ),
            ),
        ),
        None,
        ValueSchema.scalar(np.dtype("float64"), "count"),
    )
    values = np.asarray([[[float(revision)]]], dtype=np.float64)
    block = DataBlock(
        BlockId(f"{name}-block"),
        DatasetRevision(revision),
        values,
        CellValidity(np.ones((1, 1), dtype=np.bool_)),
        schema,
    )
    snapshot = OwnedSnapshot(
        block.ref(
            StreamGenerationId(
                f"{name}-generation" if generation is None else generation
            )
        ),
        block,
    )
    return LiveDatasetOutput(
        DatasetOutputDeclaration(name, f"test.{name}"),
        snapshot,
        MonitorCoverage(1, 1),
    )


def _node(instance_id: str, outputs: tuple[LiveDatasetOutput, ...]):
    return SimpleNamespace(
        instance_id=instance_id,
        dataset_output_declarations=tuple(output.declaration for output in outputs),
        signal_key=lambda name: f"{instance_id}/{name}",
    )


def _attach_live(
    plane: SignalDataPlane,
    node,
    state: dict[str, LiveDatasetOutput],
    *,
    calls: list[object] | None = None,
):
    def freeze_live_outputs():
        if calls is not None:
            calls.append(object())
        return dict(state)

    slot = SimpleNamespace(
        freeze_live_outputs=freeze_live_outputs,
        close=lambda: None,
        notification_failure=None,
    )
    plane.reserve(node)
    plane.attach(node, slot)
    plane.mark_changed(node, slot)
    return slot


def _live_source_plane():
    plane = SignalDataPlane()
    state = {
        "frame": _live_output("frame", 1),
        "frame_aux": _live_output("frame_aux", 1),
    }
    node = _node("camera", tuple(state.values()))
    slot = _attach_live(
        plane,
        node,
        state,
    )
    first = plane.freeze()
    return plane, state, node, slot, first


def test_a_followed_slot_publishes_every_update_between_two_freezes() -> None:
    """A follower asked for every shot; a display asks for the newest one.

    Both used to be served by freeze alone, so a burst -- a board playing a
    whole scan table from one fire -- collapsed into ONE publication and the
    pixels of the earlier cycles were overwritten in the slot before anyone
    could see them.  A scan that counts one publication per played shot then
    waits forever for shots that no longer exist.
    """

    plane, state, node, slot, _first = _live_source_plane()
    try:
        _baseline, tap = plane.follow_publications("camera/frame")
        for revision in range(2, 5):
            state["frame"] = _live_output("frame", revision)
            plane.mark_changed(node, slot)

        received = []
        while True:
            try:
                received.append(tap.next(0.0).payload)
            except TimeoutError:
                break
        assert len(received) == 3, (
            "every update between two freezes must reach the follower, "
            f"got {len(received)}"
        )

        # And the display still sees the newest one, exactly once more.
        front = plane.freeze()
        value = front.value("camera/frame")
        assert value is not None
        try:
            extra = tap.next(0.0).payload
        except TimeoutError:
            extra = None
        assert extra is None, "freeze must not republish what was already sent"
    finally:
        plane.close()


def test_detach_with_slow_slot_cannot_withdraw_or_wake_replacement() -> None:
    """Retirement is generation-linearized before a live slot joins."""

    plane = SignalDataPlane()
    old_output = _live_output("frame", 1)
    new_output = _live_output("frame", 2)
    node = _node("camera", (old_output,))
    close_entered = threading.Event()
    allow_close = threading.Event()
    errors = []
    old_slot = SimpleNamespace(
        freeze_live_outputs=lambda: {"frame": old_output},
        close=lambda: (close_entered.set(), allow_close.wait(2.0)),
        notification_failure=None,
    )
    new_slot = SimpleNamespace(
        freeze_live_outputs=lambda: {"frame": new_output},
        close=lambda: None,
        notification_failure=None,
    )
    try:
        first_generation = plane.reserve(node)
        plane.attach(node, old_slot)

        def detach() -> None:
            try:
                plane.detach_live(node)
            except BaseException as error:
                errors.append(error)

        worker = threading.Thread(target=detach)
        worker.start()
        assert close_entered.wait(1.0)
        assert plane.reserve(node) != first_generation
        plane.attach(node, new_slot)

        plane.mark_changed(node, old_slot)
        assert plane.freeze().value("camera/frame") is None
        plane.mark_changed(node, new_slot)
        assert plane.freeze().value("camera/frame").snapshot.ref.revision.value == 2

        allow_close.set()
        worker.join(2.0)
        assert not worker.is_alive()
        assert not errors
        assert plane.latest_publication("camera/frame") is not None
    finally:
        allow_close.set()
        plane.close()


class _GatedProcessorNode:
    def __init__(
        self,
        plane: SignalDataPlane,
        *,
        instance_id: str,
        source_name: str,
        declared_outputs: tuple[str, ...],
        gates: dict[int, threading.Event],
        published_outputs: tuple[str, ...] | None = None,
    ) -> None:
        self.instance_id = instance_id
        self._plane = plane
        self._source_name = source_name
        self._gates = gates
        self._published_outputs = (
            declared_outputs if published_outputs is None else published_outputs
        )
        self.ready_events = {
            revision: threading.Event()
            for revision in gates
        }
        self.dataset_output_declarations = tuple(
            _live_output(name, 1).declaration
            for name in declared_outputs
        )
        self.failure: Exception | None = None
        self.cancelled = False
        self.ready = False
        self.wake = threading.Event()
        self.ready_event = threading.Event()
        self.cancelled_event = threading.Event()
        self.failure_event = threading.Event()

    def signal_key(self, name: str) -> str:
        return f"{self.instance_id}/{name}"

    def validate_processor_source(self, source) -> None:
        if source.name != self._source_name:
            raise ValueError("wrong test source")

    def evaluate_processor(self, source):
        self.validate_processor_source(source)
        self.ready = True
        self.ready_event.set()
        revision = source.snapshot.ref.revision.value
        self.ready_events.setdefault(revision, threading.Event()).set()
        gate = self._gates.setdefault(revision, threading.Event())
        if not gate.wait(2.0):
            raise TimeoutError("test Processor gate did not open")
        return {
            name: _live_output(
                name,
                revision,
                generation=f"{name}-generation-{revision}",
            )
            for name in self._published_outputs
        }

    def accept_processor_result(
        self,
        _source,
        source_publication: SignalPublication,
        outputs,
    ) -> None:
        self._plane.publish_processor(
            self,
            outputs,
            source_publication=source_publication,
        )

    def accept_processor_failure(self, error) -> None:
        self.failure = error
        self.failure_event.set()

    def accept_processor_cancelled(self) -> None:
        self.cancelled = True
        self.cancelled_event.set()

    def request_processor_owner_wake(self) -> None:
        self.wake.set()


def _wait_for_signal_revision(
    plane: SignalDataPlane,
    name: str,
    revision: int,
    wake: threading.Event,
) -> SignalFront:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        front = plane.freeze()
        value = front.value(name)
        if value is not None and value.snapshot.ref.revision.value == revision:
            return front
        if not wake.wait(max(0.0, deadline - time.monotonic())):
            break
        wake.clear()
    raise AssertionError(f"{name} revision {revision} did not reach the data front")


def _wait_for_processor_ready(
    plane: SignalDataPlane,
    processor: _GatedProcessorNode,
) -> None:
    assert processor.ready_event.wait(2.0)
    assert processor.ready


def _wait_for_processor_cancelled(
    plane: SignalDataPlane,
    processor: _GatedProcessorNode,
) -> None:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        plane.freeze()
        if processor.cancelled_event.is_set():
            break
        if not processor.wake.wait(max(0.0, deadline - time.monotonic())):
            break
        processor.wake.clear()
    assert processor.cancelled


def test_unchanged_sources_reuse_their_exact_publication_and_front() -> None:
    plane = SignalDataPlane()
    output = _live_output("frame", 1)
    state = {"frame": output}
    node = _node("camera", (output,))
    calls: list[object] = []
    slot = _attach_live(
        plane,
        node,
        state,
        calls=calls,
    )
    try:
        first = plane.freeze()
        publication = first.publication("camera/frame")
        assert isinstance(publication, SignalPublication)
        assert publication.value("camera/frame") is first.value("camera/frame")
        assert plane.freeze() is first
        assert len(calls) == 1

        state["frame"] = _live_output("frame", 2)
        plane.mark_changed(node, slot)
        second = plane.freeze()
        assert second is not first
        assert second.publication("camera/frame") is not publication
        assert len(calls) == 2
    finally:
        plane.close()


def test_the_data_plane_holds_no_toolkit_or_domain_authority() -> None:
    tree = ast.parse(
        (REPO / "src" / "zlc_runtime" / "plane.py").read_text(
            encoding="utf-8"
        )
    )
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    assert not roots & {"PyQt5", "matplotlib", "zlc_storage", "zlc_neutral_atom"}, roots


def test_preemption_and_device_association_protocols_are_absent() -> None:
    removed = (
        "bind_generation_source",
        "release_generation_source",
        "withdraw_dependency_closure",
        "finish_dependency_retirement",
        "require_active_generation",
        "bind_processor_event_source",
        "signal_event_binding",
        "has_event_association",
        "bind_event_derived",
        "publish_event_derived",
    )
    assert all(not hasattr(SignalDataPlane, name) for name in removed)


def test_live_slot_cannot_publish_an_undeclared_output_contract() -> None:
    declared = _live_output("frame", 1)
    undeclared = _live_output("roi", 1)
    node = _node("camera", (declared,))
    plane = SignalDataPlane()
    _attach_live(
        plane,
        node,
        {"roi": undeclared},
    )
    try:
        front = plane.freeze()
        assert front.names() == ()
        assert "undeclared output" in front.failures["camera"]
    finally:
        plane.close()


def test_independent_producers_keep_independent_publications() -> None:
    plane = SignalDataPlane()
    slow = _live_output("frame", 3)
    fast = _live_output("frame", 91, generation="fast-generation")
    slow_node = _node("slow", (slow,))
    fast_node = _node("fast", (fast,))
    _attach_live(
        plane,
        slow_node,
        {"frame": slow},
    )
    _attach_live(
        plane,
        fast_node,
        {"frame": fast},
    )
    try:
        front = plane.freeze()
        slow_publication = front.publication("slow/frame")
        fast_publication = front.publication("fast/frame")
        assert slow_publication is not None and fast_publication is not None
        assert slow_publication is not fast_publication
        assert slow_publication.direct_parent_refs == ()
        assert fast_publication.direct_parent_refs == ()
        assert slow_publication.event_ref.stream_id.value == "slow"
        assert fast_publication.event_ref.stream_id.value == "fast"
        assert not hasattr(front.value("slow/frame"), "run_id")
        assert not hasattr(front, "shot")
    finally:
        plane.close()


def test_processor_advances_with_its_exact_source_publication() -> None:
    plane, state, source_node, source_slot, first = _live_source_plane()
    gates = {1: threading.Event(), 2: threading.Event()}
    gates[1].set()
    processor = _GatedProcessorNode(
        plane,
        instance_id="occupancy",
        source_name="camera/frame",
        declared_outputs=("occupied",),
        gates=gates,
    )
    plane.set_front_signals(
        {"camera/frame", "camera/frame_aux", "occupancy/occupied"}
    )
    first_publication = first.publication("camera/frame")
    assert first_publication is not None
    plane.attach_latest_only_processor(
        processor,
        source_name="camera/frame",
        initial_publication=first_publication,
    )
    try:
        admitted = _wait_for_signal_revision(plane, "occupancy/occupied", 1, processor.wake)
        derived_publication = admitted.publication("occupancy/occupied")
        assert derived_publication is not None
        assert plane.direct_parent_publications(derived_publication) == (
            first_publication,
        )

        state["frame"] = _live_output("frame", 2)
        state["frame_aux"] = _live_output("frame_aux", 2)
        plane.mark_changed(source_node, source_slot)
        staged = plane.freeze()
        assert staged.value("camera/frame").snapshot.ref.revision.value == 1
        assert staged.value("camera/frame_aux").snapshot.ref.revision.value == 1
        assert staged.value("occupancy/occupied").snapshot.ref.revision.value == 1

        gates[2].set()
        advanced = _wait_for_signal_revision(plane, "occupancy/occupied", 2, processor.wake)
        source_publication = advanced.publication("camera/frame")
        result_publication = advanced.publication("occupancy/occupied")
        assert source_publication is not None and result_publication is not None
        assert plane.direct_parent_publications(result_publication) == (
            source_publication,
        )
        assert advanced.value("camera/frame_aux").snapshot.ref.revision.value == 2
    finally:
        for gate in gates.values():
            gate.set()
        plane.close()


def test_fanout_leaf_disagreement_rolls_back_the_complete_family() -> None:
    """One fast and one slow sibling cannot tear a requested family apart."""

    plane, state, source_node, source_slot, first = _live_source_plane()
    fast_gates = {1: threading.Event(), 2: threading.Event()}
    slow_gates = {1: threading.Event(), 2: threading.Event()}
    fast_gates[1].set()
    fast_gates[2].set()
    slow_gates[1].set()
    fast = _GatedProcessorNode(
        plane,
        instance_id="roi",
        source_name="camera/frame",
        declared_outputs=("value",),
        gates=fast_gates,
    )
    slow = _GatedProcessorNode(
        plane,
        instance_id="fit",
        source_name="camera/frame",
        declared_outputs=("value",),
        gates=slow_gates,
    )
    requested = {
        "camera/frame",
        "camera/frame_aux",
        "roi/value",
        "fit/value",
    }
    plane.set_front_signals(requested)
    first_publication = first.publication("camera/frame")
    assert first_publication is not None
    plane.attach_latest_only_processor(
        fast,
        source_name="camera/frame",
        initial_publication=first_publication,
    )
    plane.attach_latest_only_processor(
        slow,
        source_name="camera/frame",
        initial_publication=first_publication,
    )
    try:
        initial = _wait_for_signal_revision(plane, "fit/value", 1, slow.wake)
        assert initial.value("roi/value") is not None
        fast.wake.clear()
        slow.wake.clear()

        state["frame"] = _live_output("frame", 2)
        state["frame_aux"] = _live_output("frame_aux", 2)
        plane.mark_changed(source_node, source_slot)
        plane.freeze()
        assert slow.ready_events[2].wait(2.0)
        assert fast.wake.wait(2.0)
        fast.wake.clear()
        staged = plane.freeze()

        fast_publication = plane.latest_publication("roi/value")
        slow_publication = plane.latest_publication("fit/value")
        source_publication = plane.latest_publication("camera/frame")
        assert fast_publication is not None
        assert slow_publication is not None
        assert source_publication is not None
        assert fast_publication.value("roi/value").snapshot.ref.revision.value == 2
        assert slow_publication.value("fit/value").snapshot.ref.revision.value == 1
        assert source_publication.value("camera/frame").snapshot.ref.revision.value == 2
        assert plane.direct_parent_publications(fast_publication) == (
            source_publication,
        )

        assert {
            name: staged.value(name).snapshot.ref.revision.value
            for name in requested
        } == {name: 1 for name in requested}

        slow_gates[2].set()
        recovered = _wait_for_signal_revision(plane, "fit/value", 2, slow.wake)
        assert {
            name: recovered.value(name).snapshot.ref.revision.value
            for name in requested
        } == {name: 2 for name in requested}
    finally:
        for gate in (*fast_gates.values(), *slow_gates.values()):
            gate.set()
        plane.close()


def test_source_retirement_removes_the_complete_publication_closure() -> None:
    plane, _state, source_node, _source_slot, first = _live_source_plane()
    gate = threading.Event()
    processor = _GatedProcessorNode(
        plane,
        instance_id="occupancy",
        source_name="camera/frame",
        declared_outputs=("occupied",),
        gates={1: gate},
    )
    publication = first.publication("camera/frame")
    assert publication is not None
    plane.attach_latest_only_processor(
        processor,
        source_name="camera/frame",
        initial_publication=publication,
    )
    try:
        _wait_for_processor_ready(plane, processor)
        retired = plane.retire(source_node)
        assert retired == frozenset(
            {"camera/frame", "camera/frame_aux", "occupancy/occupied"}
        )
        assert plane.freeze().names() == ()
        gate.set()
        _wait_for_processor_cancelled(plane, processor)
        assert plane.freeze().names() == ()
    finally:
        gate.set()
        plane.close()


def test_explicit_processor_cancel_waits_for_inflight_work_acknowledgement() -> None:
    plane, _state, _source_node, _source_slot, first = _live_source_plane()
    gate = threading.Event()
    processor = _GatedProcessorNode(
        plane,
        instance_id="occupancy",
        source_name="camera/frame",
        declared_outputs=("occupied",),
        gates={1: gate},
    )
    publication = first.publication("camera/frame")
    assert publication is not None
    plane.attach_latest_only_processor(
        processor,
        source_name="camera/frame",
        initial_publication=publication,
    )
    try:
        _wait_for_processor_ready(plane, processor)
        assert not plane.cancel_latest_only_processor(processor)
        assert plane.freeze().value("occupancy/occupied") is None
        assert not processor.cancelled
        gate.set()
        _wait_for_processor_cancelled(plane, processor)
        assert plane.freeze().value("occupancy/occupied") is None
    finally:
        gate.set()
        plane.close()


def test_fan_out_binds_the_same_exact_visible_publication() -> None:
    plane, _state, _source_node, _source_slot, first = _live_source_plane()
    source_publication = first.publication("camera/frame")
    assert source_publication is not None
    processors = []
    try:
        for name in ("left", "right"):
            gate = threading.Event()
            gate.set()
            processor = _GatedProcessorNode(
                plane,
                instance_id=name,
                source_name="camera/frame",
                declared_outputs=("value",),
                gates={1: gate},
            )
            processors.append(processor)
            plane.attach_latest_only_processor(
                processor,
                source_name="camera/frame",
                initial_publication=source_publication,
            )
        for name in ("left", "right"):
            front = _wait_for_signal_revision(plane, f"{name}/value", 1, processor.wake)
            publication = front.publication(f"{name}/value")
            assert publication is not None
            assert plane.direct_parent_publications(publication) == (
                source_publication,
            )
    finally:
        plane.close()


def test_processor_publication_requires_every_declared_sibling() -> None:
    plane, _state, _source_node, _source_slot, first = _live_source_plane()
    gate = threading.Event()
    gate.set()
    processor = _GatedProcessorNode(
        plane,
        instance_id="partial",
        source_name="camera/frame",
        declared_outputs=("left", "right"),
        published_outputs=("left",),
        gates={1: gate},
    )
    publication = first.publication("camera/frame")
    assert publication is not None
    plane.attach_latest_only_processor(
        processor,
        source_name="camera/frame",
        initial_publication=publication,
    )
    try:
        deadline = time.monotonic() + 2.0
        while processor.failure is None and time.monotonic() < deadline:
            plane.freeze()
            if processor.failure is not None:
                break
            if not processor.wake.wait(max(0.0, deadline - time.monotonic())):
                break
            processor.wake.clear()
        assert isinstance(processor.failure, ValueError)
        assert "complete frozen output vocabulary" in str(processor.failure)
        front = plane.freeze()
        assert front.value("partial/left") is None
        assert front.value("partial/right") is None
    finally:
        plane.close()
