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

from concurrent.futures import CancelledError, Future, InvalidStateError
from dataclasses import dataclass, replace
from threading import Lock
from typing import Any, Callable

from zlc_runtime import SurfaceUpdate


__all__ = ["PlotPanelPort"]


@dataclass(frozen=True)
class _Prepared:
    """What one panel handed to the batch, and what it was drawn from."""

    publication: object
    plot_input: object
    front_refs: tuple[object, ...]
    replacement_host: object | None = None
    operation: object | None = None
    completion: Future | None = None


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
        *,
        display_interval_ms: int,
        companion_signals: Callable[[], tuple[str, ...]] | None = None,
        project_input: Callable[[object, object, object], object] | None = None,
        submit_projection: Callable[[Callable[[], object]], Future],
        current_dataset: Callable[[str, object], object] | None = None,
        replace_host: (
            Callable[[object, object, object], tuple[Any, object]] | None
        ) = None,
        accept_host: (
            Callable[[object, object, object, object], None] | None
        ) = None,
        retire_host: Callable[[object], None] | None = None,
        on_presented: Callable[[object, object], None] | None = None,
        present: Callable[[object], None] | None = None,
    ) -> None:
        self._panel_id = str(panel_id)
        self._signal_name = str(signal_name)
        self._host = None
        self._host_token = object()
        self._interval_ms = int(display_interval_ms)
        self._presented: object | None = None
        self._presented_front_refs: tuple[object, ...] = ()
        self._presented_input: object | None = None
        self._project_input = project_input
        #: Signals this panel READS besides the one it shows -- an image's
        #: annotation.  Asked per tick because the operator can change it;
        #: the scheduler unions them into the plane's coherent front set, so
        #: an annotation is frozen at the shot its picture came from.
        self._companion_signals = companion_signals
        if not callable(submit_projection):
            raise TypeError("plot panel port requires a projection submitter")
        self._current_dataset = current_dataset
        self._submit_projection = submit_projection
        self._replace_host = replace_host
        self._accept_host = accept_host
        self._retire_host = retire_host
        self._on_presented = on_presented
        #: Puts the accepted render's pixels on screen.  The batch is the
        #: same-shot unit: every member of one causal group is accepted in one
        #: owner-thread pass, so presenting from ``accept`` is what makes the
        #: group land as one shot.  ``None`` keeps the port presentation-free
        #: for hosts whose widgets present their own fronts.
        self._present = present
        self._shown_revision = None
        self._shown_generation: object | None = None
        #: The newest revision ever handed to the host, per generation.  The
        #: port is the host's sole feeder and the host's revisions are
        #: strictly monotonic, so a revision at or below this mark can never
        #: be staged again — whether its render presented, was skipped with
        #: its abandoned cohort, or was rejected.  Re-offering it produced a
        #: "data revision must increase" refusal on the card once per beat
        #: until the next shot arrived.
        self._staged_generation: object | None = None
        self._staged_revision: int | None = None
        self._staged_front_refs: tuple[object, ...] = ()
        self._serial = 0
        self._pending: dict[int, _Prepared] = {}
        # Projection completion runs on the board worker while accept, close,
        # and the next tick run on the owner.  This lock protects only the
        # port's small identity/pending record; Runtime materialization and
        # plot rendering happen outside it.
        self._state_lock = Lock()
        self._closed = False
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

        with self._state_lock:
            return self._presented

    def presented_front_refs(self) -> tuple[object, ...]:
        """EventRefs of the primary and companions currently on screen."""

        with self._state_lock:
            return self._presented_front_refs

    def presented_input(self) -> object | None:
        """The exact plot input accepted beside ``presented_publication``."""

        with self._state_lock:
            return self._presented_input

    @staticmethod
    def _revision_value(revision: object) -> int | None:
        """Canonical integer revision for cross-boundary identity.

        The port's snapshots carry ``DatasetRevision`` objects while a fit
        event names its source by the bare integer value; comparing the raw
        objects silently never matches, which starved every fit publication
        whose panel had already staged a newer frame.
        """

        value = getattr(revision, "value", revision)
        return value if isinstance(value, int) else None

    def publication_for_revision(self, revision: object) -> object | None:
        """The publication whose snapshot carries ``revision``, if held here.

        A fit accepted by this panel's renderer fires inside revision N's own
        commit, so N is either still travelling toward the batch (pending) or
        already on screen (presented).  The port is therefore the
        deterministic causal holder of a fit publication's exact parent — no
        plane-side retention window exists or is needed.
        """

        with self._state_lock:
            wanted = self._revision_value(revision)
            if wanted is None:
                return None
            for prepared in self._pending.values():
                if self._revision_value(_revision_of(prepared.plot_input)) == wanted:
                    return prepared.publication
            if (
                self._presented is not None
                and self._revision_value(self._shown_revision) == wanted
            ):
                return self._presented
            return None

    @property
    def host(self) -> Any:
        with self._state_lock:
            return self._host

    # -------------------------------------------------------------- the tick

    @property
    def front_signals(self) -> tuple[str, ...]:
        """Every signal this panel reads from one front, shown one first."""

        companions = () if self._companion_signals is None else self._companion_signals()
        return tuple(
            dict.fromkeys(
                (self._signal_name, *(str(name) for name in companions if name))
            )
        )

    def _front_refs(
        self,
        front: object,
        publication: object,
    ) -> tuple[object, ...]:
        refs = [publication.event_ref]
        companions = self.front_signals[1:]
        if not companions:
            return tuple(refs)
        publication_for = getattr(front, "publication", None)
        if not callable(publication_for):
            raise TypeError("companion presentation requires a SignalFront")
        for name in companions:
            companion = publication_for(name)
            if companion is None:
                raise LookupError(f"front lacks panel signal {name!r}")
            refs.append(companion.event_ref)
        return tuple(refs)

    def _plot_value(self, value: object, publication: object) -> object:
        if getattr(value, "canonical_schema", None) is None:
            return value
        if self._current_dataset is None:
            raise RuntimeError(
                "finite exact presentation requires current_dataset"
            )
        snapshot = self._current_dataset(self._signal_name, publication)
        # This is a presentation-local value.  Clearing placement metadata
        # says its snapshot is already the canonical run view, so a Console
        # projection callback does not ask Runtime to materialize it twice.
        # The SurfaceUpdate still carries the original event value; exact
        # Processor delivery therefore remains chunk based.
        return replace(
            value,
            snapshot=snapshot,
            canonical_schema=None,
            cell_origin=None,
        )

    def prepare(
        self,
        value: object,
        publication: object,
        front: object,
    ) -> SurfaceUpdate | None:
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

        # Resolve companion membership before entering the state critical
        # section.  The callback belongs to Console and may itself inspect the
        # panel; holding the lock here made that ordinary re-entry deadlock.
        with self._state_lock:
            if self._closed:
                raise RuntimeError("plot panel port is closed")
        front_refs = self._front_refs(front, publication)
        publication_generation = _publication_generation(publication)
        revision = _revision_of(value)
        completion: Future = Future()
        notify_presented: tuple[object, object] | None = None
        serial: int | None = None
        host_token: object | None = None

        # Reserve one serial using only in-memory state.  Projection, host work,
        # Future settlement, and Console callbacks all happen after release.
        with self._state_lock:
            if self._closed:
                raise RuntimeError("plot panel port is closed")
            for pending_serial, prepared in tuple(self._pending.items()):
                same_render = prepared.front_refs == front_refs or (
                    _publication_generation(prepared.publication)
                    == publication_generation
                    and _revision_of(prepared.plot_input) == revision
                    and prepared.front_refs[1:] == front_refs[1:]
                )
                if same_render:
                    if prepared.publication is not publication:
                        self._pending[pending_serial] = replace(
                            prepared,
                            publication=publication,
                            front_refs=front_refs,
                        )
                    return None

            replacing_generation = (
                self._shown_generation is not None
                and publication_generation != self._shown_generation
            )
            if replacing_generation:
                if self._replace_host is None:
                    raise RuntimeError(
                        "signal generation changed; the panel host must be replaced"
                    )
                if any(
                    _publication_generation(prepared.publication)
                    != self._shown_generation
                    for prepared in self._pending.values()
                ):
                    return None

            if (
                publication_generation == self._shown_generation
                and revision is not None
                and revision == self._shown_revision
            ):
                if (
                    not self._pending
                    and self._presented is not publication
                    and front_refs[1:] == self._presented_front_refs[1:]
                ):
                    self._presented = publication
                    self._presented_front_refs = front_refs
                    notify_presented = (publication, self._presented_input)
                elif front_refs == self._presented_front_refs:
                    return None

            if notify_presented is None:
                if any(
                    prepared.front_refs == front_refs
                    for prepared in self._pending.values()
                ):
                    return None
                if front_refs == self._staged_front_refs:
                    return None
                staged_revision = self._revision_value(revision)
                if (
                    staged_revision is not None
                    and self._staged_revision is not None
                    and publication_generation == self._staged_generation
                    and staged_revision <= self._staged_revision
                    and publication is not self._presented
                ):
                    return None

                self._serial += 1
                serial = self._serial
                host_token = self._host_token
                self._pending[serial] = _Prepared(
                    publication,
                    value.snapshot,
                    front_refs,
                    completion=completion,
                )

        if notify_presented is not None:
            if self._on_presented is not None:
                self._on_presented(*notify_presented)
            return None
        assert serial is not None and host_token is not None

        def project_and_stage() -> object:
            plot_value = self._plot_value(value, publication)
            plot_input = (
                plot_value.snapshot
                if self._project_input is None
                else self._project_input(plot_value, publication, front)
            )
            with self._state_lock:
                prepared = self._pending.get(serial)
                if (
                    self._closed
                    or completion.cancelled()
                    or prepared is None
                    or prepared.completion is not completion
                ):
                    raise CancelledError()
                host = self._host
                replace_current_host = host is None or (
                    self._shown_generation is not None
                    and publication_generation != self._shown_generation
                )
                shown_revision = self._shown_revision
                is_presented = publication is self._presented

            # Host construction/configuration can be substantial.  Keeping it
            # on this same projection job preserves ordering without holding
            # the identity lock that close and owner acceptance need.
            replacement = None
            if replace_current_host:
                if self._replace_host is None:
                    raise RuntimeError(
                        "signal generation changed; the panel host must be replaced"
                    )
                replacement, rendered = self._replace_host(
                    plot_input,
                    plot_value,
                    publication,
                )
                if replacement is None or not hasattr(replacement, "host_id"):
                    raise TypeError(
                        "panel host replacement must return a plotting host"
                    )
            elif (
                is_presented
                and _revision_of(plot_input) == shown_revision
                and hasattr(plot_input, "overlay")
                and callable(getattr(host, "update_image_overlay", None))
            ):
                rendered = host.update_image_overlay(plot_input.overlay)
            else:
                rendered = host.update_data(plot_input)

            with self._state_lock:
                current = self._pending.get(serial)
                stale = (
                    self._closed
                    or completion.cancelled()
                    or current is None
                    or current.completion is not completion
                    or self._host is not host
                )
                if not stale:
                    self._pending[serial] = replace(
                        current,
                        plot_input=plot_input,
                        replacement_host=replacement,
                        operation=rendered,
                    )
            if stale:
                self._cancel_operation(rendered)
                if replacement is not None:
                    self._close_staged_host(replacement)
                raise CancelledError()
            return rendered

        def render_finished(rendered: Future) -> None:
            if completion.done():
                return
            try:
                operation = rendered.result()
            except CancelledError:
                completion.cancel()
            except BaseException as error:
                try:
                    completion.set_exception(error)
                except InvalidStateError:
                    pass
            else:
                try:
                    completion.set_result(operation)
                except InvalidStateError:
                    pass

        def projection_finished(resolved: Future) -> None:
            if completion.done():
                return
            try:
                rendered = resolved.result()
            except CancelledError:
                completion.cancel()
                return
            except BaseException as error:
                try:
                    completion.set_exception(error)
                except InvalidStateError:
                    pass
                return
            rendered.add_done_callback(render_finished)

        def cancel_inflight(done: Future) -> None:
            if not done.cancelled():
                return
            with self._state_lock:
                prepared = self._pending.get(serial)
                operation = (
                    None
                    if prepared is None or prepared.completion is not completion
                    else prepared.operation
                )
            self._cancel_operation(operation)

        # Register cancellation before submission.  Future invokes callbacks
        # synchronously when already complete, so every registration and every
        # settlement must remain outside the state lock.
        completion.add_done_callback(cancel_inflight)
        try:
            projected = self._submit_projection(project_and_stage)
        except BaseException:
            with self._state_lock:
                current = self._pending.get(serial)
                if current is not None and current.completion is completion:
                    self._pending.pop(serial, None)
            completion.cancel()
            raise

        with self._state_lock:
            current = self._pending.get(serial)
            stale = (
                self._closed
                or completion.cancelled()
                or current is None
                or current.completion is not completion
            )
            if not stale and current.operation is None:
                self._pending[serial] = replace(current, operation=projected)
        if stale:
            self._cancel_operation(projected)
            completion.cancel()
        else:
            projected.add_done_callback(projection_finished)

        return SurfaceUpdate(
            panel_id=self._panel_id,
            serial=serial,
            host_token=host_token,
            publication=publication,
            value=value,
            front_refs=front_refs,
            future=completion,
        )

    def can_accept(self, update: SurfaceUpdate, operation: object) -> bool:
        """Whether this panel will still show that update.

        A panel reconfigured since the batch was prepared says no, and the whole
        batch is abandoned rather than showing one stale panel beside fresh ones.
        """

        with self._state_lock:
            return (
                not self._closed
                and update.host_token == self._host_token
                and update.serial in self._pending
            )

    def _advance_staged(
        self,
        generation: object,
        revision: object,
        front_refs: tuple[object, ...],
    ) -> None:
        """Record that the host now holds ``revision`` of ``generation``."""

        value = self._revision_value(revision)
        self._staged_front_refs = tuple(front_refs)
        if value is None:
            return
        if self._staged_generation != generation:
            self._staged_generation = generation
            self._staged_revision = value
        elif self._staged_revision is None or value > self._staged_revision:
            self._staged_revision = value

    @staticmethod
    def _render_completed(update: SurfaceUpdate) -> bool:
        """Whether the plotting host successfully committed this surface."""

        future = update.future
        if not future.done() or future.cancelled():
            return False
        try:
            if future.exception() is not None:
                return False
        except CancelledError:
            return False
        return True

    def accept(self, update: SurfaceUpdate, operation: object) -> bool:
        """Accept bookkeeping atomically, then run fallible app callbacks unlocked."""

        with self._state_lock:
            if self._closed or update.host_token != self._host_token:
                return False
            prepared = self._pending.pop(update.serial, None)
            if prepared is None:
                return False
            publication = prepared.publication
            replacement = prepared.replacement_host
            old_host = self._host
            if replacement is not None:
                self._host = replacement
                self._host_token = replacement.host_id
            self._presented = publication
            self._presented_input = prepared.plot_input
            accepted_refs = prepared.front_refs
            self._shown_generation = _publication_generation(publication)
            self._shown_revision = _revision_of(self._presented_input)
            self._presented_front_refs = accepted_refs
            self._advance_staged(
                self._shown_generation,
                self._shown_revision,
                accepted_refs,
            )
            self.last_error = None
            presented_input = self._presented_input

        callback_error: BaseException | None = None
        if replacement is not None:
            try:
                if self._accept_host is not None:
                    self._accept_host(
                        old_host,
                        replacement,
                        publication,
                        prepared.plot_input,
                    )
            except BaseException as error:
                # Staging already proved the new host is usable.  Application
                # bookkeeping failure is reported on this panel but must not
                # tear an already-accepted cohort in half.
                callback_error = error
        if self._present is not None:
            # The pixels, on the owner thread, inside the batch's one accept
            # pass -- this is the group's atomic present.  A refused present
            # (the surface changed between staging and now) is remembered as
            # the panel's error, never allowed to unwind the batch bookkeeping
            # mid-pass: the next revision renders against the new surface and
            # heals the pixels.
            try:
                self._present(operation)
            except BaseException as error:
                callback_error = error
        if self._on_presented is not None:
            try:
                self._on_presented(publication, presented_input)
            except BaseException as error:
                callback_error = error
        if callback_error is not None:
            with self._state_lock:
                self.last_error = callback_error
        return True

    def reject(self, update: SurfaceUpdate, error: BaseException | None) -> None:
        """Abandon a prepared update, and remember why.

        It used to be a silent no-op, so a panel whose render failed simply
        stopped changing -- indistinguishable from a signal that had stopped
        arriving, which is the other thing a still panel means.

        A ``CancelledError`` is not remembered.  A cancelled render was
        coalesced away because a newer frame is already queued behind it (or
        the host is shutting down) -- flow control, not failure -- and shown
        red on the card it read as the camera failing once per skipped frame.
        """

        completed = self._render_completed(update)
        prepared = None
        with self._state_lock:
            serial = getattr(update, "serial", None)
            if serial is not None:
                prepared = self._pending.pop(serial, None)
                if (
                    completed
                    and prepared is not None
                    and prepared.replacement_host is None
                ):
                    self._advance_staged(
                        _publication_generation(prepared.publication),
                        _revision_of(prepared.plot_input),
                        prepared.front_refs,
                    )
            if error is not None and not isinstance(error, CancelledError):
                self.last_error = error
        if prepared is not None and prepared.replacement_host is not None:
            self._cancel_operation(prepared.operation)
            self._close_staged_host(prepared.replacement_host)

    def finish_unpresented(self, update: SurfaceUpdate) -> None:
        """Release a surface the batch decided not to show.

        A completed render leaving unpresented (its cohort was abandoned)
        still advanced the host's revision: the staged mark records that so
        the scheduler's re-offer of the same publication is declined instead
        of bouncing off the host as a "revision must increase" refusal once
        per beat until the next shot.
        """

        completed = self._render_completed(update)
        prepared = None
        with self._state_lock:
            serial = getattr(update, "serial", None)
            if serial is not None:
                prepared = self._pending.pop(serial, None)
                if (
                    completed
                    and prepared is not None
                    and prepared.replacement_host is None
                ):
                    self._advance_staged(
                        _publication_generation(prepared.publication),
                        _revision_of(prepared.plot_input),
                        prepared.front_refs,
                    )
        if prepared is not None:
            self._cancel_operation(prepared.operation)
            if prepared.replacement_host is not None:
                self._close_staged_host(prepared.replacement_host)

    @staticmethod
    def _cancel_operation(operation: object | None) -> None:
        cancel = getattr(operation, "cancel", None)
        if callable(cancel):
            try:
                cancel()
            except BaseException:
                pass

    def _close_staged_host(self, host: object) -> None:
        if self._retire_host is not None:
            self._retire_host(host)
            return
        close = getattr(host, "close", None)
        if not callable(close):
            return
        try:
            close(timeout=0.0)
        except TypeError:
            close()

    def close(self) -> None:
        """Cancel unpresented work; the mounted host remains its caller's."""

        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            pending = tuple(self._pending.values())
            self._pending.clear()
        for prepared in pending:
            self._cancel_operation(prepared.operation)
            self._cancel_operation(prepared.completion)
            if prepared.replacement_host is not None:
                self._close_staged_host(prepared.replacement_host)

    def report_waiting(self, missing_signal: str) -> None:
        """The signal this panel wants has not arrived on this tick."""

        self.missing.append(str(missing_signal))
