from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import pytest

from zlc_plot._live_channel import LatestRevisionChannel
from zlc_plot.errors import RevisionError


def test_latest_revision_channel_is_strict_and_latest_only() -> None:
    channel: LatestRevisionChannel[str] = LatestRevisionChannel(initial_revision=2)
    channel.publish(3, "three")
    channel.publish(4, "four")
    metrics = channel.metrics()
    assert metrics.published == 2
    assert metrics.coalesced_updates == 1
    assert metrics.pending is True
    assert channel.take_latest().payload == "four"
    assert channel.take_latest() is None
    assert channel.metrics().consumed == 1

    with pytest.raises(RevisionError):
        channel.publish(4, "again")
    with pytest.raises(RevisionError):
        channel.publish(1, "old")


def test_close_drops_pending_and_wakes_waiters() -> None:
    channel: LatestRevisionChannel[int] = LatestRevisionChannel()
    channel.publish(0, 1)
    channel.close()
    assert channel.take_latest() is None
    assert channel.wait_latest(timeout=0.0) is None
    with pytest.raises(RuntimeError):
        channel.publish(1, 2)
    channel.close()


def test_concurrent_publish_pressure_preserves_monotonic_acceptance() -> None:
    channel: LatestRevisionChannel[int] = LatestRevisionChannel()
    counter = 0
    counter_lock = Lock()

    def publish_batch() -> None:
        nonlocal counter
        for _ in range(250):
            with counter_lock:
                revision = counter
                counter += 1
                # Acceptance of the revision and the monotonic counter are
                # one critical section.  Otherwise another producer can
                # publish a newer revision between counter allocation and
                # channel acceptance on a free-threaded interpreter.
                channel.publish(revision, revision)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: publish_batch(), range(4)))

    metrics = channel.metrics()
    assert metrics.published == 1000
    assert metrics.last_published_revision == 999
    latest = channel.take_latest()
    assert latest is not None
    assert latest.revision == 999
    assert latest.payload == 999
    assert channel.metrics().coalesced_updates == 999
