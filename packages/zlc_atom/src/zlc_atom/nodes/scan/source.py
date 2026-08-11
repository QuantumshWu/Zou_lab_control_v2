"""Following the live signal a scan reads at every point.

Both engines watch ONE live signal and take its publications in order; only
what they do between publications differs.  Three facts live here.

A publication materialises when the plane FREEZES.  Waiting for the display
to freeze would couple acquisition to whether a panel happens to be open --
a scan on a panel-less bench would simply never advance -- so the scan
freezes for itself.  Freezing with nothing staged is cheap by the plane's
own design; unchanged slots reuse their immutable fronts.

Everything published before a point is applied belongs to the old world, so
the backlog is dropped on purpose before the fire that applies the point.

A source that restarts mid-scan is not a source with a gap; it is a
different generation, and the scan says so instead of stitching.
"""

from __future__ import annotations

from zlc_runtime import SignalValue
from zlc_runtime.streams import StreamEndedEarly


def follow_source(signal_plane, signal_name: str, generation: object):
    """Subscribe to the source's future publications, or refuse the restart."""

    baseline, tap = signal_plane.follow_publications(signal_name)
    if baseline.event_ref.generation != generation:
        raise RuntimeError("the source signal restarted before the scan began")
    return tap


def next_source_value(
    signal_plane,
    tap,
    signal_name: str,
    context,
) -> SignalValue:
    """The next publication's value of the watched signal, waiting for it."""

    while True:
        if context.cancel_requested():
            raise RuntimeError("the scan was cancelled")
        signal_plane.freeze()
        try:
            publication = tap.next(0.1).payload
        except TimeoutError:
            continue
        except StreamEndedEarly:
            raise RuntimeError(
                "the source signal restarted during the scan"
            ) from None
        value = publication.value(signal_name)
        if not isinstance(value, SignalValue):
            raise RuntimeError("the source publication lost the selected signal")
        return value


def drain_backlog(signal_plane, tap) -> None:
    """Discard everything already published; the caller knows it is stale."""

    while True:
        signal_plane.freeze()
        try:
            tap.next(0.0)
        except TimeoutError:
            return
        except StreamEndedEarly:
            raise RuntimeError(
                "the source signal restarted during the scan"
            ) from None


__all__ = ["drain_backlog", "follow_source", "next_source_value"]
