"""Finite and reactive NodeHost lifecycle contracts."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
from threading import Event
from types import SimpleNamespace
import time

import numpy as np
import pytest

from zlc_data import (
    BlockId,
)


from zlc_runtime.dataset import MonitorCoverage
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
) -> LiveDatasetOutput:
    declaration = declaration or DatasetOutputDeclaration(name, f"test.{name}")
    return LiveDatasetOutput(
        declaration,
        _snapshot(name, revision),
        MonitorCoverage(1, 1, 0, False),
    )


def _final_output(
    name: str,
    revision: int,
    declaration: DatasetOutputDeclaration,
) -> FinalDatasetOutput:
    return FinalDatasetOutput(declaration, _snapshot(name, revision))


def _wait_finite(host: NodeHost, wake: Event) -> LogicNodeObservation:
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


def _wait_reactive(
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


def test_finite_success_publishes_final_and_records_context_capabilities() -> None:
    declaration = DatasetOutputDeclaration("frame", "test.frame")
    wake = Event()
    plane = SignalDataPlane()

    class Node:
        kind = "finite"
        instance_id = "camera"
        dataset_output_declarations = (declaration,)

        def execute(self, context):
            assert all(
                callable(getattr(context, name))
                for name in (
                    "cancel_requested",
                    "start_and_wait",
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
                {"frame": _final_output("frame", 7, declaration)}
            )
            return {"status": "ok"}

    host = NodeHost(Node(), plane, wake.set)
    try:
        host.start()
        observation = _wait_finite(host, wake)
        assert observation == host.observation
        assert observation.phase == "done"
        assert observation.warnings == ("one warning",)
        assert host.final_result == {"status": "ok"}
        assert host.signal_key("frame") == "@logic/camera/frame"
        publication = plane.latest_publication("@logic/camera/frame")
        assert publication is not None
        assert publication.value("@logic/camera/frame") is not None
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
def test_finite_failure_and_declared_output_guard(execute, expected: str) -> None:
    declaration = DatasetOutputDeclaration("frame", "test.frame")
    wake = Event()
    plane = SignalDataPlane()

    class Node:
        kind = "finite"
        instance_id = "camera"
        dataset_output_declarations = (declaration,)

        def execute(self, context):
            return execute(context)

    host = NodeHost(Node(), plane, wake.set)
    try:
        host.start()
        observation = _wait_finite(host, wake)
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


def test_finite_cancel_and_shutdown_refuses_pending_worker() -> None:
    wake = Event()
    started = Event()
    release = Event()
    plane = SignalDataPlane()

    class Node:
        kind = "finite"
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
        observation = _wait_finite(host, wake)
        assert observation.phase == "cancelled"
        assert not host.running
        host.shutdown()
    finally:
        release.set()
        if not host.terminal:
            _wait_finite(host, wake)
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
        kind = "finite"
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
        _wait_finite(host, wake)
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
        observation = _wait_finite(host, wake)
        assert observation.phase == "done"
        assert host.final_result == "second"
    finally:
        second_release.set()
        if not host.terminal:
            _wait_finite(host, wake)
        host.shutdown()
        plane.close()


class _SourceSlot:
    notification_failure = None

    def __init__(self, state: dict[str, LiveDatasetOutput]) -> None:
        self._state = state
        self.closed = False

    def freeze_live_outputs(self):
        return dict(self._state)

    def close(self) -> None:
        self.closed = True


def _source_plane():
    declaration = DatasetOutputDeclaration("frame", "test.frame")
    state = {"frame": _live_output("frame", 1, declaration)}
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


def test_reactive_node_follows_latest_publication_and_can_jump() -> None:
    plane, source, source_slot, state, first = _source_plane()
    wake = Event()
    declaration = DatasetOutputDeclaration("roi", "test.roi")
    revisions: list[int] = []

    class Node:
        kind = "reactive"
        instance_id = "roi"
        input_signal = "camera/frame"
        dataset_output_declarations = (declaration,)

        def evaluate(self, value):
            revision = value.snapshot.ref.revision.value
            revisions.append(revision)
            return {"roi": _live_output("roi", revision, declaration)}

    host = NodeHost(Node(), plane, wake.set)
    try:
        host.start()
        first_front = _wait_reactive(
            plane,
            wake,
            lambda front: front.value("@logic/roi/roi") is not None,
        )
        assert first_front.value("@logic/roi/roi").snapshot.ref.revision.value == 1

        wake.clear()
        state["frame"] = _live_output("frame", 2, source.dataset_output_declarations[0])
        plane.mark_changed(source, source_slot)
        plane.freeze()
        second_front = _wait_reactive(
            plane,
            wake,
            lambda front: (
                front.value("@logic/roi/roi") is not None
                and front.value("@logic/roi/roi").snapshot.ref.revision.value == 2
            ),
        )
        assert second_front.value("@logic/roi/roi").snapshot.ref.revision.value == 2
        assert revisions == [1, 2]
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


def test_reactive_latest_only_skips_busy_intermediate_publications() -> None:
    plane, source, source_slot, state, _first = _source_plane()
    wake = Event()
    first_started = Event()
    third_started = Event()
    release_first = Event()
    declaration = DatasetOutputDeclaration("roi", "test.roi.jump")
    revisions: list[int] = []

    class Node:
        kind = "reactive"
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
        _wait_reactive(plane, wake, lambda _front: third_started.is_set())
        front = _wait_reactive(
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


def test_finite_cancel_then_restart_resets_the_generation_stop_event() -> None:
    wake = Event()
    first_started = Event()
    release_first = Event()
    calls = 0
    plane = SignalDataPlane()

    class Node:
        kind = "finite"
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
        cancelled = _wait_finite(host, wake)
        assert cancelled.phase == "cancelled"

        host.start()
        completed = _wait_finite(host, wake)
        assert completed.phase == "done"
        assert host.final_result == 2
    finally:
        release_first.set()
        if not host.terminal:
            _wait_finite(host, wake)
        host.shutdown()
        plane.close()


class _LiveSpec:
    block_id = BlockId("host-live")
    dataset_edge = object()


class _LiveOwner:
    def live_dataset_outputs(self, _frozen):
        return {}


def test_finite_live_attachment_is_singleton_per_generation() -> None:
    declaration = DatasetOutputDeclaration("frame", "test.live.frame")
    wake = Event()
    plane = SignalDataPlane()

    class Node:
        kind = "finite"
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
        observation = _wait_finite(host, wake)
        assert observation.phase == "done"
        assert host.final_result == "live opened once"
    finally:
        host.shutdown()
        plane.close()


def test_finite_live_open_without_final_uses_tree_terminal_semantics() -> None:
    declaration = DatasetOutputDeclaration("frame", "test.live.final")
    wake = Event()
    plane = SignalDataPlane()

    class Node:
        kind = "finite"
        instance_id = "live-terminal"
        dataset_output_declarations = (declaration,)

        def execute(self, context):
            context.open_live_dataset(
                _LiveSpec(),
                output_owner=_LiveOwner(),
            )
            return "live terminal"

    host = NodeHost(Node(), plane, wake.set)
    try:
        host.start()
        observation = _wait_finite(host, wake)
        assert observation.phase == "done"
        assert observation.error is None
        assert host.final_result == "live terminal"
        assert plane.latest_publication("@logic/live-terminal/frame") is None
    finally:
        host.shutdown()
        plane.close()


def test_reactive_failure_and_inflight_cancel_are_terminal_callbacks() -> None:
    for mode in ("failure", "cancel"):
        plane, _source, _source_slot, _state, _first = _source_plane()
        wake = Event()
        evaluate_started = Event()
        release = Event()
        declaration = DatasetOutputDeclaration("roi", f"test.roi.{mode}")

        class Node:
            kind = "reactive"
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
                _wait_reactive(
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
                _wait_reactive(
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
