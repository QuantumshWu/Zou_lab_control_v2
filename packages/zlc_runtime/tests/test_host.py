"""Worker and source-bound processor NodeHost lifecycle contracts."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
from threading import Event, get_ident
from types import SimpleNamespace
import time

import numpy as np
import pytest

from zlc_data import (
    BlockId,
)


from zlc_runtime.dataset import DatasetCoverage, MonitorCoverage
from zlc_runtime.dataset_output import (
    DatasetOutputDeclaration,
    FinalDatasetOutput,
    LiveDatasetOutput,
)
from zlc_runtime.host import LogicNodeObservation, NodeHost
from zlc_runtime.owner_mailbox import OwnerCompletion
from zlc_runtime.plane import SignalDataPlane

from _snapshots import snapshot as _snapshot


def _live_output(
    name: str,
    revision: int,
    declaration: DatasetOutputDeclaration | None = None,
    run_record: dict[str, object] | None = None,
) -> LiveDatasetOutput:
    declaration = declaration or DatasetOutputDeclaration(name, f"test.{name}")
    return LiveDatasetOutput(
        declaration,
        _snapshot(name, revision),
        MonitorCoverage(1, 1),
        run_record,
    )


def _dataset_output(
    name: str,
    revision: int,
    declaration: DatasetOutputDeclaration | None = None,
    run_record: dict[str, object] | None = None,
) -> LiveDatasetOutput:
    declaration = declaration or DatasetOutputDeclaration(name, f"test.{name}")
    return LiveDatasetOutput(
        declaration,
        _snapshot(name, revision),
        DatasetCoverage(1, 1),
        run_record,
    )


def _final_output(
    name: str,
    revision: int,
    declaration: DatasetOutputDeclaration,
    run_record: dict[str, object] | None = None,
) -> FinalDatasetOutput:
    return FinalDatasetOutput(
        declaration,
        _snapshot(name, revision),
        run_record,
    )


def _wait_worker(host: NodeHost, wake: Event) -> LogicNodeObservation:
    deadline = time.monotonic() + 2.0
    while not host.terminal:
        observation = host.poll()
        if observation.terminal:
            return observation
        remaining = deadline - time.monotonic()
        if remaining <= 0.0 or not wake.wait(remaining):
            break
        wake.clear()
    observation = host.poll()
    assert observation.terminal
    return observation


def _wait_processor(
    plane: SignalDataPlane,
    wake: Event,
    predicate,
):
    deadline = time.monotonic() + 2.0
    while True:
        front = plane.freeze()
        if predicate(front):
            return front
        remaining = deadline - time.monotonic()
        if remaining <= 0.0 or not wake.wait(remaining):
            break
        wake.clear()
    front = plane.freeze()
    assert predicate(front)
    return front


@dataclass
class _ImmediateHandle:
    value: object
    cancelled: bool = False

    def snapshot(self):
        return SimpleNamespace(phase="done")

    def cancel(self, reason: str | None = None) -> None:
        self.cancelled = True

    def result(self):
        return self.value


def test_worker_without_kind_publishes_final_and_records_context_capabilities() -> None:
    declaration = DatasetOutputDeclaration("frame", "test.frame")
    run_record = {
        "authored_parameters": {"exposure_us": 1250.0},
        "named_devices": {"camera": "mot_camera"},
        "actual_device_snapshots": {"camera": {"exposure_us": 1248.0}},
    }
    wake = Event()
    plane = SignalDataPlane()

    class Node:
        instance_id = "camera"
        dataset_output_declarations = (declaration,)
        artifact_output_names = ("report",)

        def execute(self, context):
            assert all(
                callable(getattr(context, name))
                for name in (
                    "cancel_requested",
                    "start_and_wait",
                    "attach_live_outputs",
                    "open_live_dataset",
                    "open_exact_dataset",
                    "publish_final",
                    "warn",
                )
            )
            assert context.cancel_requested() is False
            context.warn("one warning")
            assert context.start_and_wait(
                lambda: _ImmediateHandle("hardware result")
            ) == "hardware result"
            context.publish_final(
                {"frame": _final_output("frame", 7, declaration, run_record)}
            )
            return {"status": "ok"}

    node = Node()
    host = NodeHost(node, plane, wake.set)
    try:
        assert host.node is node
        assert host.source_signal is None
        host.start()
        observation = _wait_worker(host, wake)
        assert observation == host.observation
        assert observation.phase == "done"
        assert observation.warnings == ("one warning",)
        assert host.final_result == {"status": "ok"}
        assert host.signal_key("frame") == "@logic/camera/frame"
        with pytest.raises(KeyError, match="undeclared node output"):
            host.signal_key("report")
        publication = plane.latest_publication("@logic/camera/frame")
        assert publication is not None
        value = publication.value("@logic/camera/frame")
        assert value is not None
        assert publication.run_record == run_record
        assert value.run_record == run_record
        run_record["late_mutation"] = True
        assert "late_mutation" not in publication.run_record
        assert "late_mutation" not in value.run_record
        with pytest.raises(TypeError):
            publication.run_record["forbidden"] = True
        with pytest.raises(TypeError):
            value.run_record["forbidden"] = True
        assert host.worker_idle
    finally:
        host.shutdown()
        plane.close()


@pytest.mark.parametrize(
    "execute, expected",
    [
        (lambda _context: (_ for _ in ()).throw(ValueError("bad node")), "failed"),
        (lambda _context: "missing final", "failed"),
    ],
)
def test_worker_failure_and_declared_output_guard(execute, expected: str) -> None:
    declaration = DatasetOutputDeclaration("frame", "test.frame")
    wake = Event()
    plane = SignalDataPlane()

    class Node:
        instance_id = "camera"
        dataset_output_declarations = (declaration,)

        def execute(self, context):
            return execute(context)

    host = NodeHost(Node(), plane, wake.set)
    try:
        host.start()
        observation = _wait_worker(host, wake)
        assert observation.phase == expected
        assert observation.error is not None
        if "bad node" in observation.error:
            assert "ValueError" in observation.error
        else:
            assert "did not publish final outputs" in observation.error
        assert plane.latest_publication("@logic/camera/frame") is None
    finally:
        host.shutdown()
        plane.close()


def test_worker_cancel_and_shutdown_refuses_pending_worker() -> None:
    wake = Event()
    started = Event()
    release = Event()
    plane = SignalDataPlane()

    class Node:
        instance_id = "slow"

        def execute(self, context):
            started.set()
            release.wait(2.0)
            if context.cancel_requested():
                raise RuntimeError("cancelled by host")
            return "unexpected"

    host = NodeHost(Node(), plane, wake.set)
    try:
        host.start()
        assert started.wait(2.0)
        with pytest.raises(RuntimeError, match="before terminal"):
            host.shutdown()
        assert host.phase == "stopping"
        release.set()
        observation = _wait_worker(host, wake)
        assert observation.phase == "cancelled"
        assert not host.running
        host.shutdown()
    finally:
        release.set()
        if not host.terminal:
            _wait_worker(host, wake)
        if not getattr(host, "_closed", False):
            host.shutdown()
        plane.close()


def test_stale_mailbox_completion_cannot_replace_new_generation() -> None:
    wake = Event()
    second_ready = Event()
    second_release = Event()
    calls = 0
    plane = SignalDataPlane()

    class Node:
        instance_id = "restartable"

        def execute(self, _context):
            nonlocal calls
            calls += 1
            if calls == 1:
                return "first"
            second_ready.set()
            second_release.wait(2.0)
            return "second"

    host = NodeHost(Node(), plane, wake.set)
    try:
        host.start()
        _wait_worker(host, wake)
        first_generation = host._owner.generation
        host.start()
        assert second_ready.wait(2.0)
        stale = Future()
        stale.set_result("stale")
        mailbox = host._owner
        with mailbox._lock:
            mailbox._completions.append(
                OwnerCompletion("execute", first_generation, stale)
            )
        wake.clear()
        host.poll()
        assert host.running
        assert not host.final_result_resolved
        second_release.set()
        observation = _wait_worker(host, wake)
        assert observation.phase == "done"
        assert host.final_result == "second"
    finally:
        second_release.set()
        if not host.terminal:
            _wait_worker(host, wake)
        host.shutdown()
        plane.close()


class _SourceSlot:
    notification_failure = None

    def __init__(self, state: dict[str, LiveDatasetOutput]) -> None:
        self._state = state
        self.closed = False

    def freeze_live_outputs(self):
        return dict(self._state)

    def set_change_listener(self, listener) -> None:
        self._listener = listener

    def close(self) -> None:
        self.closed = True


def _source_plane(*, run_record: dict[str, object] | None = None):
    declaration = DatasetOutputDeclaration("frame", "test.frame")
    state = {"frame": _live_output("frame", 1, declaration, run_record)}
    source = SimpleNamespace(
        instance_id="camera",
        dataset_output_declarations=(declaration,),
        signal_key=lambda name: f"camera/{name}",
    )
    plane = SignalDataPlane()
    slot = _SourceSlot(state)
    plane.reserve(source)
    plane.attach(source, slot)
    plane.mark_changed(source, slot)
    first = plane.freeze()
    return plane, source, slot, state, first


def _final_source_plane():
    declaration = DatasetOutputDeclaration("frame", "test.final.frame")
    source = SimpleNamespace(
        instance_id="final-camera",
        dataset_output_declarations=(declaration,),
        signal_key=lambda name: f"final-camera/{name}",
    )
    plane = SignalDataPlane()
    plane.reserve(source)
    plane.publish_final(
        source,
        {"frame": _final_output("frame", 1, declaration)},
    )
    publication = plane.latest_publication("final-camera/frame")
    assert publication is not None
    return plane, publication


def test_source_bound_processor_without_kind_follows_latest_publication() -> None:
    source_record = {"named_devices": {"camera": "camera"}}
    processor_record = {"authored_parameters": {"roi": [2, 3, 8, 9]}}
    plane, source, source_slot, state, first = _source_plane(
        run_record=source_record,
    )
    wake = Event()
    declaration = DatasetOutputDeclaration("roi", "test.roi")
    revisions: list[int] = []

    class Node:
        instance_id = "roi"
        input_signal = "camera/frame"
        dataset_output_declarations = (declaration,)

        def evaluate(self, value):
            assert value.run_record == source_record
            revision = value.snapshot.ref.revision.value
            revisions.append(revision)
            return {
                "roi": _live_output(
                    "roi",
                    revision,
                    declaration,
                    processor_record,
                )
            }

    host = NodeHost(Node(), plane, wake.set)
    try:
        assert host.source_signal == "camera/frame"
        host.start()
        first_front = _wait_processor(
            plane,
            wake,
            lambda front: front.value("@logic/roi/roi") is not None,
        )
        assert first_front.value("@logic/roi/roi").snapshot.ref.revision.value == 1

        wake.clear()
        state["frame"] = _live_output(
            "frame",
            2,
            source.dataset_output_declarations[0],
            source_record,
        )
        plane.mark_changed(source, source_slot)
        plane.freeze()
        second_front = _wait_processor(
            plane,
            wake,
            lambda front: (
                front.value("@logic/roi/roi") is not None
                and front.value("@logic/roi/roi").snapshot.ref.revision.value == 2
            ),
        )
        assert second_front.value("@logic/roi/roi").snapshot.ref.revision.value == 2
        assert revisions == [1, 2]
        child_publication = second_front.publication("@logic/roi/roi")
        assert child_publication is not None
        assert child_publication.run_record == processor_record
        assert second_front.value("@logic/roi/roi").run_record == processor_record
        (parent_publication,) = plane.direct_parent_publications(child_publication)
        assert parent_publication.run_record == source_record
        assert parent_publication.value("camera/frame").run_record == source_record
        assert host.running
        host.cancel("test cancellation")
        assert host.terminal
        assert host.phase == "cancelled"
    finally:
        if not host.terminal:
            host.cancel("cleanup")
            plane.freeze()
        host.shutdown()
        plane.close()


def test_processor_latest_only_skips_busy_intermediate_publications() -> None:
    plane, source, source_slot, state, _first = _source_plane()
    wake = Event()
    first_started = Event()
    third_started = Event()
    release_first = Event()
    declaration = DatasetOutputDeclaration("roi", "test.roi.jump")
    revisions: list[int] = []

    class Node:
        instance_id = "roi-jump"
        input_signal = "camera/frame"
        dataset_output_declarations = (declaration,)

        def evaluate(self, value):
            revision = value.snapshot.ref.revision.value
            revisions.append(revision)
            if revision == 1:
                first_started.set()
                assert release_first.wait(2.0)
            elif revision == 3:
                third_started.set()
            return {"roi": _live_output("roi", revision, declaration)}

    host = NodeHost(Node(), plane, wake.set)
    try:
        host.start()
        assert first_started.wait(2.0)
        wake.clear()

        state["frame"] = _live_output("frame", 2, source.dataset_output_declarations[0])
        plane.mark_changed(source, source_slot)
        plane.freeze()
        state["frame"] = _live_output("frame", 3, source.dataset_output_declarations[0])
        plane.mark_changed(source, source_slot)
        plane.freeze()

        release_first.set()
        _wait_processor(plane, wake, lambda _front: third_started.is_set())
        front = _wait_processor(
            plane,
            wake,
            lambda current: (
                current.value("@logic/roi-jump/roi") is not None
                and current.value("@logic/roi-jump/roi").snapshot.ref.revision.value == 3
            ),
        )
        assert revisions == [1, 3]
        result_publication = front.publication("@logic/roi-jump/roi")
        source_publication = front.publication("camera/frame")
        assert result_publication is not None and source_publication is not None
        assert plane.direct_parent_publications(result_publication) == (
            source_publication,
        )
    finally:
        release_first.set()
        if not host.terminal:
            host.cancel("cleanup")
            plane.freeze()
        host.shutdown()
        plane.close()


def test_worker_cancel_then_restart_resets_the_generation_stop_event() -> None:
    wake = Event()
    first_started = Event()
    release_first = Event()
    calls = 0
    plane = SignalDataPlane()

    class Node:
        instance_id = "restart-stop"

        def execute(self, context):
            nonlocal calls
            calls += 1
            if calls == 1:
                first_started.set()
                assert release_first.wait(2.0)
                if context.cancel_requested():
                    raise RuntimeError("first generation cancelled")
            return calls

    host = NodeHost(Node(), plane, wake.set)
    try:
        host.start()
        assert first_started.wait(2.0)
        host.cancel("restart test")
        release_first.set()
        cancelled = _wait_worker(host, wake)
        assert cancelled.phase == "cancelled"

        host.start()
        completed = _wait_worker(host, wake)
        assert completed.phase == "done"
        assert host.final_result == 2
    finally:
        release_first.set()
        if not host.terminal:
            _wait_worker(host, wake)
        host.shutdown()
        plane.close()


class _LiveSpec:
    block_id = BlockId("host-live")
    dataset_edge = object()


class _LiveOwner:
    def live_dataset_outputs(self, _frozen):
        return {}


def test_worker_live_attachment_is_singleton_per_generation() -> None:
    declaration = DatasetOutputDeclaration("frame", "test.live.frame")
    wake = Event()
    plane = SignalDataPlane()

    class Node:
        instance_id = "live-once"
        dataset_output_declarations = (declaration,)

        def execute(self, context):
            context.open_live_dataset(
                _LiveSpec(),
                output_owner=_LiveOwner(),
            )
            with pytest.raises(RuntimeError, match="one Node generation"):
                context.open_live_dataset(
                    _LiveSpec(),
                    output_owner=_LiveOwner(),
                )
            return "live opened once"

    host = NodeHost(Node(), plane, wake.set)
    try:
        host.start()
        observation = _wait_worker(host, wake)
        assert observation.phase == "done"
        assert host.final_result == "live opened once"
    finally:
        host.shutdown()
        plane.close()


def test_cancelled_worker_detaches_app_owned_live_outputs_before_terminal() -> None:
    declaration = DatasetOutputDeclaration("frame", "test.live.final")
    wake = Event()
    started = Event()
    release = Event()
    plane = SignalDataPlane()
    slot = _SourceSlot({"frame": _live_output("frame", 1, declaration)})

    class Node:
        instance_id = "live-terminal"
        dataset_output_declarations = (declaration,)

        def execute(self, context):
            context.attach_live_outputs(slot)
            started.set()
            assert release.wait(2.0)
            return "capture loop stopped"

    host = NodeHost(Node(), plane, wake.set)
    try:
        host.start()
        assert started.wait(2.0)
        host.cancel("stop infinite capture")
        release.set()
        observation = _wait_worker(host, wake)
        assert observation.phase == "cancelled"
        assert observation.error is None
        assert not host.final_result_resolved
        assert slot.closed
        assert plane.latest_publication("@logic/live-terminal/frame") is None
    finally:
        release.set()
        host.shutdown()
        plane.close()


def test_dataset_coverage_processor_follows_every_publication_then_finishes() -> None:
    source_declaration = DatasetOutputDeclaration("frame", "test.exact.frame")
    source_record = {
        "authored_parameters": {"repeat": 3},
        "actual_device_snapshots": {"camera": {"binning": [1, 1]}},
    }
    processor_record = {
        "authored_parameters": {"calibration_path": "calibration.json"}
    }
    source_state = {
        "frame": _dataset_output(
            "frame",
            1,
            source_declaration,
            source_record,
        )
    }
    source = SimpleNamespace(
        instance_id="exact-camera",
        dataset_output_declarations=(source_declaration,),
        signal_key=lambda name: f"exact-camera/{name}",
    )
    source_slot = _SourceSlot(source_state)
    plane = SignalDataPlane()
    plane.reserve(source)
    plane.attach(source, source_slot)
    plane.mark_changed(source, source_slot)
    plane.freeze()

    wake = Event()
    first_started = Event()
    release_first = Event()
    declaration = DatasetOutputDeclaration("derived", "test.exact.derived")
    revisions: list[int] = []

    class Node:
        instance_id = "exact-derived"
        input_signal = "exact-camera/frame"
        dataset_output_declarations = (declaration,)

        def evaluate(self, value):
            assert value.run_record == source_record
            revision = value.snapshot.ref.revision.value
            revisions.append(revision)
            if revision == 1:
                first_started.set()
                assert release_first.wait(2.0)
            return {
                "derived": _dataset_output(
                    "derived",
                    revision,
                    declaration,
                    processor_record,
                )
            }

    host = NodeHost(Node(), plane, wake.set)
    try:
        host.start()
        assert first_started.wait(2.0)
        for revision in (2, 3):
            source_state["frame"] = _dataset_output(
                "frame",
                revision,
                source_declaration,
                source_record,
            )
            plane.mark_changed(source, source_slot)
            plane.freeze()
        release_first.set()
        plane.finish_live(source)
        completed = _wait_worker(host, wake)
        assert completed.phase == "done"
        assert revisions == [1, 2, 3]

        front = plane.freeze()
        publication = front.publication("@logic/exact-derived/derived")
        value = front.value("@logic/exact-derived/derived")
        source_publication = plane.latest_publication("exact-camera/frame")
        assert publication is not None and value is not None
        assert source_publication is not None
        assert value.coverage is None and value.transient is False
        assert publication.run_record == processor_record
        assert value.run_record == processor_record
        assert source_publication.run_record == source_record
        assert source_publication.value("exact-camera/frame").run_record == source_record
        assert not plane.is_generation_live("@logic/exact-derived/derived")
        assert plane.direct_parent_publications(publication) == (source_publication,)
    finally:
        release_first.set()
        if host.running:
            host.cancel("cleanup")
            _wait_worker(host, wake)
        host.shutdown()
        plane.close()


def test_frozen_final_processor_runs_once_on_worker_and_cleans_up() -> None:
    plane, source_publication = _final_source_plane()
    wake = Event()
    started = Event()
    release = Event()
    declaration = DatasetOutputDeclaration("derived", "test.frozen.derived")
    caller_thread = get_ident()
    behavior = "success"
    processor_record = {"authored_parameters": {"mode": "frozen"}}
    calls: list[tuple[str, int]] = []

    class Node:
        instance_id = "frozen"
        input_signal = "final-camera/frame"
        dataset_output_declarations = (declaration,)

        def evaluate(self, value):
            calls.append((behavior, get_ident()))
            started.set()
            if behavior == "failure":
                raise ValueError("frozen processor failed")
            assert release.wait(2.0)
            return {
                "derived": _live_output(
                    "derived",
                    1,
                    declaration,
                    processor_record,
                )
            }

    host = NodeHost(Node(), plane, wake.set)
    try:
        host.start()
        assert started.wait(2.0)
        assert len(calls) == 1 and calls[0][0] == "success"
        assert calls[0][1] != caller_thread
        assert host.running
        release.set()
        completed = _wait_worker(host, wake)
        assert completed.phase == "done"

        front = plane.freeze()
        value = front.value("@logic/frozen/derived")
        publication = front.publication("@logic/frozen/derived")
        assert value is not None and publication is not None
        assert value.coverage is None
        assert value.transient is False
        assert value.run_record == processor_record
        assert publication.run_record == processor_record
        assert not plane.is_generation_live("@logic/frozen/derived")
        assert plane.direct_parent_publications(publication) == (source_publication,)

        behavior = "cancel"
        started.clear()
        release.clear()
        host.start()
        assert started.wait(2.0)
        host.cancel("cancel frozen processor")
        release.set()
        cancelled = _wait_worker(host, wake)
        assert cancelled.phase == "cancelled"
        assert plane.latest_publication("@logic/frozen/derived") is None

        behavior = "failure"
        started.clear()
        release.clear()
        host.start()
        assert started.wait(2.0)
        failed = _wait_worker(host, wake)
        assert failed.phase == "failed"
        assert failed.error is not None and "frozen processor failed" in failed.error
        assert plane.latest_publication("@logic/frozen/derived") is None
    finally:
        release.set()
        if host.running:
            host.cancel("cleanup")
            _wait_worker(host, wake)
        host.shutdown()
        plane.close()


def test_processor_failure_and_inflight_cancel_are_terminal_callbacks() -> None:
    for mode in ("failure", "cancel"):
        plane, _source, _source_slot, _state, _first = _source_plane()
        wake = Event()
        evaluate_started = Event()
        release = Event()
        declaration = DatasetOutputDeclaration("roi", f"test.roi.{mode}")

        class Node:
            instance_id = f"roi-{mode}"
            input_signal = "camera/frame"
            dataset_output_declarations = (declaration,)

            def evaluate(self, _value):
                evaluate_started.set()
                if mode == "failure":
                    raise ValueError("processor failed")
                release.wait(2.0)
                return {"roi": _live_output("roi", 1, declaration)}

        host = NodeHost(Node(), plane, wake.set)
        try:
            host.start()
            if mode == "failure":
                _wait_processor(
                    plane,
                    wake,
                    lambda _front: host.terminal,
                )
                assert host.phase == "failed"
                assert host.last_error is not None
                assert "processor failed" in host.last_error
            else:
                assert evaluate_started.wait(2.0)
                host.cancel("stop processor")
                release.set()
                _wait_processor(
                    plane,
                    wake,
                    lambda _front: host.terminal,
                )
                assert host.phase == "cancelled"
        finally:
            release.set()
            if not host.terminal:
                host.cancel("cleanup")
                plane.freeze()
            host.shutdown()
            plane.close()
