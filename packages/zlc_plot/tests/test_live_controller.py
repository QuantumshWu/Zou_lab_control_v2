from __future__ import annotations

from concurrent.futures import Future
from threading import Event

import numpy as np

from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable
from zlc_plot import DEFAULTS
from zlc_plot.live import LivePlotController


def _snapshot(revision: int) -> DatasetSnapshot:
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"x": [0.0, 1.0]}),
        dtype=np.float64,
        generation="controller-test",
    )
    return DatasetSnapshot(schema, np.array([[revision, revision + 1.0]]), revision)


class FakeSession:
    defaults = DEFAULTS

    def __init__(
        self,
        *,
        block_prepare: bool = False,
        armed: bool = False,
    ) -> None:
        self.data_revision = 0
        self.prepared: list[int] = []
        self.presented: list[int] = []
        self.solved: list[int] = []
        self.committed_pairs: list[tuple[int, object | None]] = []
        self.prepare_started = Event()
        self.prepare_future: Future[object] | None = None
        self.block_prepare = block_prepare
        self.armed = armed

    def prepare_live_frame(self, data, *, revision=None, cancelled=None):
        self.prepare_started.set()
        future: Future[object] = Future()
        self.prepare_future = future
        if self.block_prepare:
            return future
        future.set_result((int(revision), data))
        return future

    def solve_live_frame(self, prepared):
        if not self.armed:
            return None
        revision, _data = prepared
        self.solved.append(int(revision))
        future: Future[object] = Future()
        future.set_result(("fit", int(revision)))
        return future

    def commit_live_frame(self, prepared, solved=None):
        revision, _data = prepared
        self.committed_pairs.append((int(revision), solved))
        if revision <= self.data_revision:
            return None
        self.data_revision = revision
        self.prepared.append(revision)
        return revision

    def publish_live_frame(self, finalization):
        self.presented.append(int(finalization))

    def abort_live_frame(self, finalization):
        self.data_revision -= 1


def test_controller_pump_is_latest_only_and_records_metrics() -> None:
    session = FakeSession()
    controller = LivePlotController(session, _snapshot(0), refresh_interval_ms=100)
    try:
        controller.publish(_snapshot(1))
        controller.publish(_snapshot(2))
        assert controller.pump_once() is True
        assert session.presented == [2]
        metrics = controller.metrics()
        assert metrics.coalesced_updates == 1
        assert metrics.successful_updates == 1
        assert metrics.last_presented_revision == 2
    finally:
        controller.close()


def test_controller_commits_the_solved_pair_for_an_armed_session() -> None:
    """The commit receives the solved fit half beside its prepared data."""

    session = FakeSession(armed=True)
    controller = LivePlotController(session, _snapshot(0), refresh_interval_ms=100)
    try:
        controller.publish(_snapshot(1))
        assert controller.pump_once() is True
        assert session.solved == [1]
        assert session.committed_pairs == [(1, ("fit", 1))]
    finally:
        controller.close()


def test_controller_commits_data_only_when_no_fit_is_armed() -> None:
    session = FakeSession(armed=False)
    controller = LivePlotController(session, _snapshot(0), refresh_interval_ms=100)
    try:
        controller.publish(_snapshot(1))
        assert controller.pump_once() is True
        assert session.solved == []
        assert session.committed_pairs == [(1, None)]
    finally:
        controller.close()


def test_controller_stop_during_prepare_cancels_active_work() -> None:
    session = FakeSession(block_prepare=True)
    controller = LivePlotController(session, _snapshot(0), refresh_interval_ms=100)
    controller.start()
    controller.publish(_snapshot(1))
    assert session.prepare_started.wait(2.0)
    assert controller.stop(timeout=2.0)
    metrics = controller.metrics()
    assert metrics.running is False
    assert metrics.cancelled_updates >= 1
    controller.close()
