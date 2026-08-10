from __future__ import annotations

from zlc_data.validation import DIGEST_HEX

from dataclasses import dataclass
from threading import Event

import pytest

from zlc_data import BlockId

from zlc_runtime._failure import DetachedFailure, detach_failure
from zlc_runtime.cleanup import CleanupReport, run_cleanup_steps
from zlc_runtime.live_dataset import LiveDatasetPort
from zlc_runtime.owner_mailbox import RunOwnerMailbox
from zlc_runtime.output_name import bare_output_name
from zlc_runtime.preview import (
    ExactDatasetPreviewSpec,
    notify_preview_failure,
)


def test_detached_failure_drops_traceback_references() -> None:
    original = None
    try:
        raise ValueError("bad input")
    except ValueError as error:
        original = error
        detached = detach_failure(error, note_prefix="worker")

    assert isinstance(detached, DetachedFailure)
    assert detached.summary == "ValueError: bad input"
    assert detached.locations
    assert original is not None
    assert original.__traceback__ is None


def test_cleanup_runs_all_steps_and_aggregates_failures() -> None:
    called: list[str] = []

    def first() -> CleanupReport:
        called.append("first")
        return CleanupReport.complete()

    def second() -> CleanupReport:
        called.append("second")
        raise RuntimeError("second failed")

    def third() -> CleanupReport:
        called.append("third")
        return CleanupReport.complete(errors=(ValueError("third failed"),))

    report = run_cleanup_steps(first, second, third)
    assert called == ["first", "second", "third"]
    assert [str(error) for error in report.errors] == ["second failed", "third failed"]


def test_names_and_preview_specs_enforce_boundary_contracts() -> None:
    assert bare_output_name("roi") == "roi"
    with pytest.raises(ValueError):
        bare_output_name("@logic/roi")
    with pytest.raises(ValueError):
        ExactDatasetPreviewSpec("not-a-digest")
    ExactDatasetPreviewSpec("a" * DIGEST_HEX)


def test_preview_failure_notification_is_best_effort() -> None:
    class Preview:
        terminal = False

        def __init__(self) -> None:
            self.message = None

        def fail(self, message: str) -> None:
            self.message = message

    preview = Preview()
    notify_preview_failure(preview, RuntimeError("broken"))
    assert preview.message == "RuntimeError: broken"


def test_live_dataset_port_coalesces_updates_and_closes_source() -> None:
    class Spec:
        block_id = BlockId("camera")
        dataset_edge = object()

    class Source:
        def __init__(self) -> None:
            self.closed = False

        def freeze_current(self):
            return "snapshot"

        def close(self) -> None:
            self.closed = True

    source = Source()
    port = LiveDatasetPort(Spec())
    notifications: list[str] = []
    port.bind(source)
    port.updated()
    port.set_change_listener(lambda: notifications.append("changed"))
    assert notifications == ["changed"]
    assert port.freeze_current() == "snapshot"
    port.close()
    assert source.closed


def test_live_dataset_port_fail_detaches_source_and_notifies_listener() -> None:
    class Spec:
        block_id = BlockId("camera")
        dataset_edge = object()

    class Source:
        def __init__(self) -> None:
            self.closed = False

        def freeze_current(self):
            return "snapshot"

        def close(self) -> None:
            self.closed = True

    source = Source()
    notifications: list[str] = []
    port = LiveDatasetPort(Spec())
    port.bind(source)
    port.set_change_listener(lambda: notifications.append("changed"))

    port.fail("broken")

    assert port.failure == "broken"
    assert port.terminal
    assert not port.dataset_bound
    assert source.closed
    assert notifications == ["changed"]


def test_live_dataset_port_source_terminal_retains_or_withdraws_by_policy() -> None:
    class Spec:
        block_id = BlockId("camera")
        dataset_edge = object()

    class Source:
        def __init__(self) -> None:
            self.closed = False

        def freeze_current(self):
            return "snapshot"

        def close(self) -> None:
            self.closed = True

    retained_source = Source()
    retained = LiveDatasetPort(Spec(), retain_on_terminal=True)
    retained.bind(retained_source)
    retained.source_terminal()
    assert retained.terminal
    assert retained.dataset_bound
    assert not retained.withdrawn
    assert not retained_source.closed
    retained.close()
    assert retained_source.closed

    withdrawn_source = Source()
    withdrawn = LiveDatasetPort(Spec(), retain_on_terminal=False)
    withdrawn.bind(withdrawn_source)
    withdrawn.source_terminal()
    assert withdrawn.terminal
    assert withdrawn.withdrawn
    assert not withdrawn.dataset_bound
    assert withdrawn_source.closed


def test_owner_mailbox_reaps_completion_with_event_wake() -> None:
    wake = Event()
    mailbox = RunOwnerMailbox(wake.set, thread_name_prefix="zlc-test-mailbox")
    try:
        generation = mailbox.begin_generation()
        future = mailbox.submit("work", lambda: 7, generation=generation)
        assert wake.wait(2.0)
        assert future.result() == 7
        completions = mailbox.drain_completions()
        assert len(completions) == 1
        assert completions[0].kind == "work"
        assert completions[0].generation == generation
        mailbox.mark_owner_reaped()
    finally:
        if mailbox.has_pending_owner_work:
            for completion in mailbox.drain_completions():
                completion.future.result()
        mailbox.shutdown()
