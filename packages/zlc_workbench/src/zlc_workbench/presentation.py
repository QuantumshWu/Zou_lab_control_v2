"""One panel, connecting the runtime's scheduler to a plotting host.

The two sides were designed to meet and had never met: zlc_runtime schedules
board-coherent updates and knows nothing about drawing; zlc_plot draws and knows
nothing about scheduling.  This is the whole of what sits between them, and it
must stay this small -- the moment it decides WHAT to draw or WHEN a signal is
ready, that decision has been taken from the package that owns it.

Board coherence is the property worth understanding.  A tick freezes ONE signal
front and offers it to every panel; each panel prepares its update off the GUI
thread, and the batch goes up together or not at all.  Panels showing the same
shot therefore always show the same shot -- never half a board from one moment
beside half from the next, which is what makes a multi-panel view trustworthy
while an experiment is running.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from zlc_runtime import SurfaceUpdate


__all__ = ["PlotPanelPort"]


@dataclass(frozen=True)
class _Prepared:
    """What one panel handed to the batch, and what it was drawn from."""

    publication: object
    plot_input: object


def _revision_of(snapshot: object) -> object | None:
    """The data revision a snapshot carries, or None when it carries none."""

    snapshot = getattr(snapshot, "snapshot", snapshot)
    block = getattr(snapshot, "block", None)
    return getattr(block, "revision", None) if block is not None else None


def _publication_generation(publication: object) -> object | None:
    event_ref = getattr(publication, "event_ref", None)
    return getattr(event_ref, "generation", None)


class PlotPanelPort:
    """The scheduler's view of one plotting panel."""

    def __init__(
        self,
        panel_id: str,
        signal_name: str,
        host: Any,
        *,
        display_interval_ms: int,
        shown: object | None = None,
        project_input: Callable[[object, object], object] | None = None,
        replace_host: Callable[[object, object, object], Any] | None = None,
        on_presented: Callable[[object, object], None] | None = None,
    ) -> None:
        self._panel_id = str(panel_id)
        self._signal_name = str(signal_name)
        self._host = host
        self._interval_ms = int(display_interval_ms)
        self._presented: object | None = None
        self._presented_input: object | None = shown
        self._project_input = project_input
        self._replace_host = replace_host
        self._on_presented = on_presented
        #: What the host was BUILT from.  A host constructed from a snapshot is
        #: already holding that revision, and handing it the same one back is
        #: refused -- correctly, but the refusal then arrived as an error on
        #: the card once per beat, forever, because the delivery was never
        #: accepted and so was retried on every tick.
        self._shown_revision = _revision_of(shown)
        self._shown_generation: object | None = None
        self._serial = 0
        self._pending: dict[int, _Prepared] = {}
        self.missing: list[str] = []
        #: The last render this panel could not show.  Read on the beat and put
        #: on the card, because a panel that quietly stopped drawing looks
        #: exactly like a panel whose data stopped arriving.
        self.last_error: BaseException | None = None

    # ------------------------------------------------------------- identity

    @property
    def panel_id(self) -> str:
        return self._panel_id

    @property
    def signal_name(self) -> str:
        return self._signal_name

    @property
    def display_interval_ms(self) -> int:
        return self._interval_ms

    def set_display_interval(self, milliseconds: int) -> None:
        """How often this panel redraws.

        Per panel, not per board: a camera worth watching at 10 Hz sits beside
        a fit result that changes once a run, and one interval for both either
        wastes the machine or hides the camera.
        """

        interval = int(milliseconds)
        if interval <= 0:
            raise ValueError("a display interval must be positive")
        self._interval_ms = interval

    def presented_publication(self) -> object | None:
        """What is on screen now, so the scheduler can skip an unchanged shot."""

        return self._presented

    def presented_input(self) -> object | None:
        """The exact plot input accepted beside ``presented_publication``."""

        return self._presented_input

    @property
    def host(self) -> Any:
        return self._host

    # -------------------------------------------------------------- the tick

    def prepare(self, value: object, publication: object) -> SurfaceUpdate | None:
        """Hand one frozen value to the host and return the batch's handle.

        The host owns its own drawing worker, so the work happens off the GUI
        thread by construction; doing it here would put rendering on the thread
        that has to stay responsive.

        A publication this panel has already shown is not handed over again.
        The host refuses a revision it is already holding -- rightly, since a
        redraw of unchanged data is work with no result -- and that refusal
        arrived as an error on the card once per beat while a producer was
        stopped.  Nothing was wrong: the panel was simply asked the same
        question over and over.
        """

        publication_generation = _publication_generation(publication)
        plot_input = (
            value.snapshot
            if self._project_input is None
            else self._project_input(value, publication)
        )
        revision = _revision_of(plot_input)
        if (
            self._shown_generation is None
            and revision is not None
            and revision == self._shown_revision
        ):
            # The host was constructed from this exact snapshot before the
            # board first saw its publication.  Anchor the generation now so a
            # later Restart with revision=1 cannot be mistaken for this run.
            self._shown_generation = publication_generation
            self._presented = publication
            self._presented_input = plot_input
            if self._on_presented is not None:
                self._on_presented(publication, plot_input)
            return None
        if (
            self._shown_generation is not None
            and publication_generation != self._shown_generation
        ):
            if self._replace_host is None:
                raise RuntimeError(
                    "signal generation changed; the panel host must be replaced"
                )
            replacement = self._replace_host(plot_input, value, publication)
            if replacement is None or not hasattr(replacement, "host_id"):
                raise TypeError("panel host replacement must return a plotting host")
            self._host = replacement
            self._shown_generation = publication_generation
            self._shown_revision = revision
            self._presented = publication
            self._presented_input = plot_input
            self.last_error = None
            if self._on_presented is not None:
                self._on_presented(publication, plot_input)
            return None
        if (
            publication_generation == self._shown_generation
            and revision is not None
            and revision == self._shown_revision
        ):
            return None
        if any(
            prepared.publication is publication
            for prepared in self._pending.values()
        ):
            return None
        self._serial += 1
        serial = self._serial
        self._pending[serial] = _Prepared(publication, plot_input)
        try:
            future = self._host.update_data(plot_input)
        except BaseException:
            self._pending.pop(serial, None)
            raise
        return SurfaceUpdate(
            panel_id=self._panel_id,
            serial=serial,
            host_token=self._host.host_id,
            publication=publication,
            value=value,
            future=future,
            replacement=False,
        )

    def observe(self, update: SurfaceUpdate, operation: object) -> None:
        """Note that a prepared surface completed, drawn or not."""

    def can_accept(self, update: SurfaceUpdate, operation: object) -> bool:
        """Whether this panel will still show that update.

        A panel reconfigured since the batch was prepared says no, and the whole
        batch is abandoned rather than showing one stale panel beside fresh ones.
        """

        return update.host_token == self._host.host_id

    def accept(self, update: SurfaceUpdate, operation: object) -> bool:
        if not self.can_accept(update, operation):
            return False
        self._presented = update.publication
        prepared = self._pending.pop(update.serial, None)
        self._presented_input = (
            update.value.snapshot if prepared is None else prepared.plot_input
        )
        self._shown_generation = _publication_generation(update.publication)
        self._shown_revision = _revision_of(self._presented_input)
        self.last_error = None
        if self._on_presented is not None:
            self._on_presented(update.publication, self._presented_input)
        return True

    def reject(self, update: SurfaceUpdate, error: BaseException | None) -> None:
        """Abandon a prepared update, and remember why.

        It used to be a silent no-op, so a panel whose render failed simply
        stopped changing -- indistinguishable from a signal that had stopped
        arriving, which is the other thing a still panel means.
        """

        serial = getattr(update, "serial", None)
        if serial is not None:
            self._pending.pop(serial, None)
        if error is not None:
            self.last_error = error

    def finish_unpresented(self, update: SurfaceUpdate) -> None:
        """Release a surface the batch decided not to show."""

        serial = getattr(update, "serial", None)
        if serial is not None:
            self._pending.pop(serial, None)

    def report_waiting(self, missing_signal: str) -> None:
        """The signal this panel wants has not arrived on this tick."""

        self.missing.append(str(missing_signal))
