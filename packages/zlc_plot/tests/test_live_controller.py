from __future__ import annotations

from concurrent.futures import Future
from threading import Event
import time

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

    def __init__(self, *, block_prepare: bool = False) -> None:
        self.data_revision = 0
        self.prepared: list[int] = []
        self.presented: list[int] = []
        self.prepare_started = Event()
        self.prepare_future: Future[object] | None = None
        self.block_prepare = block_prepare

    def prepare_live_frame(self, data, *, revision=None, cancelled=None):
        self.prepare_started.set()
        future: Future[object] = Future()
        self.prepare_future = future
        if self.block_prepare:
            return future
        future.set_result((int(revision), data))
        return future

    def commit_live_frame(self, prepared):
        revision, _data = prepared
        if revision <= self.data_revision:
            return None
        self.data_revision = revision
        self.prepared.append(revision)
        return revision

    def finalize_live_frame(self, finalization):
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
