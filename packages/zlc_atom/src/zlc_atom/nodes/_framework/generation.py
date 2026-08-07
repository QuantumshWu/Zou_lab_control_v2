"""Starting a run, and stamping what it produces.

A producer generation belongs to ONE run while the node performing it is a
reusable object, so deciding when the previous generation ends is not the node's
business.  Three nodes each reserved for themselves and each could therefore
fire exactly once per process: the second attempt raised "producer generation is
already active", because ``publish_final`` deliberately leaves the finished
generation in place so its result stays readable.

Two facts belong to the run rather than to the array it produces:

``generation``
    which run this data came from.  ``zlc_runtime`` assigns it.

``revision``
    how far this signal has advanced.  It must increase strictly and must NOT
    restart with each run: a live plot rejects a revision that is not newer than
    the one it already holds, and a run that restarted the count would freeze
    every panel downstream of it.

Both used to be optional arguments defaulted to a constant that no caller ever
overrode, so every publication in the system carried generation ``"0"`` and
revision ``0`` forever.
"""

from __future__ import annotations

__all__ = ["ProducerRuns", "begin_run"]


def begin_run(signal_plane: object, node: object) -> object:
    """Begin a producer run for ``node``, superseding its finished predecessor.

    Raises if the plane does not implement the runtime contract rather than
    degrading to a bare ``reserve``, which would silently make the node
    single-use again.
    """

    begin = getattr(signal_plane, "begin_generation", None)
    if not callable(begin):
        raise TypeError(
            "signal_plane must implement the zlc_runtime plane contract "
            "(begin_generation); a plane offering only reserve() cannot run a "
            "node more than once"
        )
    return begin(node)


class ProducerRuns:
    """The generation and revision stamps a producer puts on its output.

    One per producing node.  ``begin`` starts a run; ``next_revision`` stamps
    each publication within it, continuing to count across runs.
    """

    __slots__ = ("_generation", "_revision")

    def __init__(self) -> None:
        self._generation: object | None = None
        self._revision = 0

    def begin(self, signal_plane: object, node: object) -> object:
        """Start a run we own."""

        self._generation = begin_run(signal_plane, node)
        return self._generation

    def adopt(self, generation: object) -> object:
        """Take up a run someone else began.

        A hosted node must not reserve: its NodeHost already did, in the host's
        own name, so a second reservation collides with it.  The host hands the
        generation down instead, because a node that stamps its output has to
        know which run it is in.
        """

        if generation is None:
            raise RuntimeError("the host supplied no generation for this run")
        self._generation = generation
        return self._generation

    @property
    def generation(self) -> str:
        if self._generation is None:
            raise RuntimeError("no run has begun; call begin() before publishing")
        return str(getattr(self._generation, "value", self._generation))

    def next_revision(self) -> int:
        """The next revision for this producer, never repeating or going back."""

        self._revision += 1
        return self._revision
