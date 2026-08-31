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


_UNCHANGED_TARGET = object()


@dataclass(frozen=True)
class _Prepared:
    """One staged or accepted panel transaction record."""

    publication: object
    plot_input: object
    event_records: tuple[tuple[object, object], ...]
    target: object
    front_refs: tuple[object, ...]
    presentation_epoch: int
    host: object | None = None
    description: object | None = None
    replacement_host: object | None = None
    operation: object | None = None
    completion: Future | None = None


def _revision_of(snapshot: object) -> object | None:
    """The data revision a snapshot carries, or None when it carries none."""

    snapshot = getattr(snapshot, "snapshot", snapshot)
    return getattr(getattr(snapshot, "ref", None), "revision", None)


def _schema_fingerprint_of(carrier: object) -> str | None:
    """The dataset GEOMETRY a publication or plot input describes."""

    snapshot = getattr(carrier, "snapshot", carrier)
    block = getattr(snapshot, "block", None)
    schema = getattr(block, "schema", None)
    if schema is None:
        return None
    fingerprint = getattr(schema, "fingerprint", None)
    if callable(fingerprint):
        fingerprint = fingerprint()
    return None if fingerprint is None else str(fingerprint)


def _same_geometry(left: str | None, right: str | None) -> bool:
    """UNKNOWN geometry is not the same geometry: only two provable
    fingerprints suppress a generation replacement."""

    return left is not None and left == right


def _generation_of(snapshot: object) -> object | None:
    snapshot = getattr(snapshot, "snapshot", snapshot)
    ref = getattr(snapshot, "ref", None)
    return getattr(ref, "stream_generation", None)


def _generation_value(generation: object) -> object:
    return getattr(generation, "value", generation)


class PlotPanelPort:
    """The scheduler's view of one plotting panel."""

    def __init__(
        self,
        panel_id: str,
        signal_name: str,
        *,
        initial_target: object = None,
        display_interval_ms: int,
        companion_signals: Callable[[object], tuple[str, ...]] | None = None,
        project_input: (
            Callable[
                [object, object, object, object],
                tuple[object, tuple[tuple[object, object], ...]],
            ]
            | None
        ) = None,
        submit_projection: Callable[[Callable[[], object]], Future],
        replace_host: (
            Callable[[object, object, object, object], tuple[Any, object]] | None
        ) = None,
        accept_host: Callable[[object], None] | None = None,
        retire_host: Callable[[object], None] | None = None,
        on_presented: Callable[[object], None] | None = None,
        present: Callable[[object, object], bool] | None = None,
        invalidate: Callable[[str], None] | None = None,
    ) -> None:
        self._panel_id = str(panel_id)
        self._signal_name = str(signal_name)
        self._projection_target = initial_target
        self._unmounted_token = object()
        self._interval_ms = int(display_interval_ms)
        self._surface: _Prepared | None = None
        #: Signal presentation can change without a new scientific
        #: publication (a Runtime history lease turns event geometry into a
        #: bounded primary-index Dataset).  This local epoch makes that a real
        #: input identity instead of pretending the old host still accepts it.
        self._presentation_epoch = 0
        self._project_input = project_input
        #: Signals this panel READS besides the one it shows -- an image's
        #: annotation.  Asked per tick because the operator can change it;
        #: the scheduler unions them into the plane's coherent front set, so
        #: an annotation is frozen at the shot its picture came from.
        self._companion_signals = companion_signals
        if not callable(submit_projection):
            raise TypeError("plot panel port requires a projection submitter")
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
        self._invalidate = invalidate
        #: The newest revision ever handed to the host, per generation.  The
        #: port is the host's sole feeder and the host's revisions are
        #: strictly monotonic, so a revision at or below this mark can never
        #: be staged again — whether its render presented, was skipped with
        #: its abandoned cohort, or was rejected.  Re-offering it produced a
        #: "data revision must increase" refusal on the card once per beat
        #: until the next shot arrived.
        self._staged_generation: object | None = None
        self._staged_revision: int | None = None
        #: What the HOST has been handed, which is a different fact from
        #: what the screen shows: a render reaches the host on the
        #: projection worker and reaches the screen on the owner thread,
        #: so between those two moments the host holds something the
        #: surface does not yet show.  Only the call that hands data over
        #: can state this, so only it writes here.
        self._held_generation: object | None = None
        self._held_revision: int | None = None
        self._staged_front_refs: tuple[object, ...] = ()
        self._staged_presentation_epoch = -1
        self._serial = 0
        self._pending: dict[int, _Prepared] = {}
        # Projection completion runs on the board worker while accept, close,
        # and the next tick run on the owner.  This lock protects only the
        # port's small identity/pending record; Runtime materialization and
        # plot rendering happen outside it.
        self._state_lock = Lock()
        self._closed = False
        #: The last render this panel could not show.  Read on the beat and put
        #: on the card, because a panel that quietly stopped drawing looks
        #: exactly like a panel whose data stopped arriving.
        self.last_error: BaseException | None = None
        #: The signal this panel is presenting WITHOUT, named -- an overlay
        #: whose producer has not published.  Read on the beat as a standing
        #: card condition, because a frame with no rings and no explanation
        #: is indistinguishable from a defect.
        self.waiting_condition: str = ""

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

    @property
    def presentation_priority(self) -> int:
        """Start a deeper fit surface before same-shot display-only siblings."""

        with self._state_lock:
            fit = getattr(self._projection_target, "fit", None)
        return 1 if fit else 0

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

    def retarget(self, target: object) -> None:
        """The panel is configured differently from now on.

        A projection reads the panel's target when it RUNS, which is a
        moment after it was staged, so a decision the operator has just
        made must reach the port before the next projection -- and the
        work already staged under the decision it replaces has to go.
        That work is not a picture the panel wants: it would draw the
        setting the operator just turned off, over the top of the
        configure that turned it off, and stay there until the producer
        published again.

        Unlike ``invalidate_presentation`` this keeps the host and the
        picture on screen: only the projection target moves.
        """

        pending: tuple[Future, ...]
        with self._state_lock:
            if self._closed or self._projection_target is target:
                return
            self._projection_target = target
            pending = tuple(
                prepared.completion
                for prepared in self._pending.values()
                if prepared.completion is not None
                and prepared.target is not target
            )
        for completion in pending:
            completion.cancel()

    def invalidate_presentation(
        self,
        target: object = _UNCHANGED_TARGET,
    ) -> None:
        """Require the current publication to be projected in a fresh host."""

        pending: tuple[Future, ...]
        with self._state_lock:
            if self._closed:
                return
            if target is not _UNCHANGED_TARGET:
                self._projection_target = target
            self._presentation_epoch += 1
            # The accepted surface remains the exact screen truth.  Board
            # debt, not falsifying its front identity, re-offers the unchanged
            # publication under the new representation epoch.
            self._staged_front_refs = ()
            self._staged_revision = None
            self._staged_generation = None
            self._staged_presentation_epoch = -1
            self._held_generation = None
            self._held_revision = None
            pending = tuple(
                prepared.completion
                for prepared in self._pending.values()
                if prepared.completion is not None
            )
        for completion in pending:
            completion.cancel()

    def presented_front_refs(self) -> tuple[object, ...]:
        """EventRefs of the primary and companions currently on screen."""

        with self._state_lock:
            return () if self._surface is None else self._surface.front_refs

    def accepted_surface(self) -> object | None:
        """The host, pixels and description accepted in one screen commit."""

        with self._state_lock:
            return self._surface

    @property
    def surface_busy(self) -> bool:
        """Whether one heavy surface is already travelling for this panel."""

        with self._state_lock:
            return bool(self._pending)

    @property
    def presentation_current(self) -> bool:
        """Whether the mounted host accepts the current input representation."""

        with self._state_lock:
            return (
                self._surface is not None
                and self._surface.presentation_epoch == self._presentation_epoch
            )

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

    def publication_for_identity(
        self, generation: object, revision: object
    ) -> object | None:
        """The exact generation/revision publication held by this surface.

        Revisions restart at one for every run.  A fit therefore names both
        identities; revision-only lookup could bind a delayed result to a
        different run carrying the same number.
        """

        with self._state_lock:
            wanted = self._revision_value(revision)
            if wanted is None:
                return None
            for prepared in self._pending.values():
                if (
                    _generation_value(
                        _generation_of(prepared.plot_input)
                    )
                    == _generation_value(generation)
                    and self._revision_value(_revision_of(prepared.plot_input))
                    == wanted
                ):
                    return prepared.publication
            if (
                self._surface is not None
                and _generation_value(
                    _generation_of(self._surface.plot_input)
                )
                == _generation_value(generation)
                and self._revision_value(_revision_of(self._surface.plot_input))
                == wanted
            ):
                return self._surface.publication
            return None

    def _host_token_locked(self) -> object:
        surface = self._surface
        return (
            self._unmounted_token
            if surface is None
            else getattr(surface.host, "host_id")
        )

    # -------------------------------------------------------------- the tick

    @property
    def front_signals(self) -> tuple[str, ...]:
        """Every signal this panel reads from one front, shown one first."""

        with self._state_lock:
            target = (
                self._projection_target
                if self._surface is None
                or self._surface.presentation_epoch != self._presentation_epoch
                else self._surface.target
            )
        companions = (
            ()
            if self._companion_signals is None
            else self._companion_signals(target)
        )
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
                # EXCUSED, not a veto -- the same rule the arbiter's
                # _front_refs applies.  Raising here made an overlay that
                # had not yet published in its generation freeze the base
                # panel outright; a base frame shown without its overlay is
                # honest, and the overlay joins atomically through the
                # coherent front with its first commit.
                continue
            refs.append(companion.event_ref)
        return tuple(refs)

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
            presentation_epoch = self._presentation_epoch
        front_refs = self._front_refs(front, publication)
        publication_generation = _generation_of(value)
        revision = _revision_of(value)
        completion: Future = Future()
        notify_presented: _Prepared | None = None
        serial: int | None = None
        host_token: object | None = None

        # Reserve one serial using only in-memory state.  Projection, host work,
        # Future settlement, and Console callbacks all happen after release.
        with self._state_lock:
            if self._closed:
                raise RuntimeError("plot panel port is closed")
            for pending_serial, prepared in tuple(self._pending.items()):
                same_render = prepared.presentation_epoch == presentation_epoch and (
                    prepared.front_refs == front_refs or (
                    _generation_of(prepared.plot_input)
                    == publication_generation
                    and _revision_of(prepared.plot_input) == revision
                    and prepared.front_refs[1:] == front_refs[1:]
                    )
                )
                if same_render:
                    if prepared.publication is not publication:
                        self._pending[pending_serial] = replace(
                            prepared,
                            publication=publication,
                            event_records=(
                                (publication, value.event_record),
                                *prepared.event_records[1:],
                            ),
                            front_refs=front_refs,
                        )
                    return None

            surface = self._surface
            shown_generation = (
                None
                if surface is None
                else _generation_of(surface.plot_input)
            )
            shown_revision = (
                None if surface is None else _revision_of(surface.plot_input)
            )
            replacing_generation = (
                surface is not None
                and publication_generation != shown_generation
                and not _same_geometry(
                    _schema_fingerprint_of(publication),
                    _schema_fingerprint_of(surface.plot_input),
                )
            )
            replacing_presentation = (
                surface is None
                or presentation_epoch != surface.presentation_epoch
            )
            if replacing_generation or replacing_presentation:
                if self._replace_host is None:
                    raise RuntimeError(
                        "signal presentation changed; the panel host must be replaced"
                    )
                if any(
                    prepared.presentation_epoch == presentation_epoch
                    for prepared in self._pending.values()
                ):
                    return None

            if (
                surface is not None
                and publication_generation == shown_generation
                and presentation_epoch == surface.presentation_epoch
                and revision is not None
                and revision == shown_revision
            ):
                if (
                    not self._pending
                    and surface.publication is not publication
                    and front_refs[1:] == surface.front_refs[1:]
                ):
                    notify_presented = replace(
                        surface,
                        publication=publication,
                        event_records=(
                            (publication, value.event_record),
                            *surface.event_records[1:],
                        ),
                        front_refs=front_refs,
                    )
                    self._surface = notify_presented
                elif front_refs == surface.front_refs:
                    return None

            if notify_presented is None:
                if any(
                    prepared.front_refs == front_refs
                    for prepared in self._pending.values()
                ):
                    return None
                if (
                    front_refs == self._staged_front_refs
                    and presentation_epoch == self._staged_presentation_epoch
                ):
                    return None
                staged_revision = self._revision_value(revision)
                if (
                    staged_revision is not None
                    and self._staged_revision is not None
                    and publication_generation == self._staged_generation
                    and presentation_epoch == self._staged_presentation_epoch
                    and staged_revision <= self._staged_revision
                    and (
                        surface is None or publication is not surface.publication
                    )
                ):
                    return None

                self._serial += 1
                serial = self._serial
                host_token = self._host_token_locked()
                # HOW this panel is projected has one owner, and it is
                # not the picture already on screen.  Reading the shown
                # surface's target answered "how was the last frame
                # made", which is the same answer right up to the moment
                # the operator changes something -- and then it is the
                # old answer, so the next frame was drawn to a setting
                # that had just been revoked.
                target = self._projection_target
                self._pending[serial] = _Prepared(
                    publication,
                    value.snapshot,
                    ((publication, value.event_record),),
                    target,
                    front_refs,
                    presentation_epoch,
                    completion=completion,
                )

        if notify_presented is not None:
            if self._on_presented is not None:
                self._on_presented(notify_presented)
            return None
        assert serial is not None and host_token is not None

        #: What this render handed to the host, recorded only if the
        #: host's own operation then completed.
        handed: list[tuple[object, int | None]] = []

        def project_and_stage() -> object:
            if self._project_input is None:
                plot_input = value.snapshot
                event_records = ((publication, value.event_record),)
            else:
                projected = self._project_input(
                    value,
                    publication,
                    front,
                    target,
                )
                if not isinstance(projected, tuple) or len(projected) != 2:
                    raise TypeError(
                        "panel input projection must return (plot_input, event_record)"
                    )
                plot_input, event_records = projected
                if (
                    not isinstance(event_records, tuple)
                    or not event_records
                    or any(
                        not isinstance(item, tuple) or len(item) != 2
                        for item in event_records
                    )
                    or event_records[0][0] is not publication
                ):
                    raise TypeError(
                        "panel event records must start with the primary publication"
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
                surface = self._surface
                host = None if surface is None else surface.host
                shown_generation = (
                    None
                    if surface is None
                    else _generation_of(surface.plot_input)
                )
                # A new RUN with the SAME geometry keeps its host: the
                # session's own projection replacement absorbs the new
                # generation, so the mounted widget, the operator's
                # in-flight gesture and every subscription survive the
                # shot boundary.  Rebuilding the host per generation tore
                # a drag apart whenever a shot landed mid-gesture -- the
                # mouse stayed grabbed by a widget whose host had been
                # retired, and the drag went dead until release.
                #
                # A new presentation EPOCH is the same story told by a
                # different trigger, and it used to replace the host
                # unconditionally.  Turning Selectors on takes a history
                # lease, which moves the signal from windowed to indexed,
                # which invalidates every panel on it -- so the gesture the
                # operator started next went nowhere and the box they had
                # just drawn came back where the remembered one was.  The
                # epoch says RE-PROJECT; the geometry says what came out.
                # A host holds a shape, so the shape decides, whichever
                # trigger asked.
                same_shape = surface is not None and _same_geometry(
                    _schema_fingerprint_of(plot_input),
                    _schema_fingerprint_of(surface.plot_input),
                )
                replace_current_host = surface is None or not same_shape and (
                    publication_generation != shown_generation
                    or presentation_epoch != surface.presentation_epoch
                )
                held_generation = self._held_generation
                held_revision = self._held_revision

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
                    value,
                    publication,
                    target,
                )
                if replacement is None or not hasattr(replacement, "host_id"):
                    raise TypeError(
                        "panel host replacement must return a plotting host"
                    )
                # A fresh host holds exactly what it was built from.
                handed.append(
                    (
                        publication_generation,
                        self._revision_value(_revision_of(plot_input)),
                    )
                )
            else:
                # What the host HOLDS decides what can be done to it, and
                # only the code that hands it data knows that.  Asking
                # instead whether this publication is the one on screen
                # answered a different question: a shot can leave the
                # screen between staging a render and running it, and the
                # answer flipped to "no" for an input whose data the host
                # already had -- which was then pushed as DATA, refused
                # ("data revision must increase") and shown on the card.
                incoming = self._revision_value(_revision_of(plot_input))
                same_stream = (
                    held_revision is not None
                    and incoming is not None
                    and held_generation == publication_generation
                )
                if same_stream and incoming < held_revision:
                    # The picture this describes is gone and is not coming
                    # back.  There is nothing to draw.
                    raise CancelledError()
                if same_stream and incoming == held_revision:
                    if not (
                        hasattr(plot_input, "overlay")
                        and callable(getattr(host, "configure", None))
                    ):
                        raise CancelledError()
                    # Same data, so the only thing that can have changed
                    # is what is drawn OVER it.
                    rendered = host.configure(image_overlay=plot_input.overlay)
                else:
                    rendered = host.update_data(plot_input)
                    handed.append((publication_generation, incoming))

            with self._state_lock:
                current = self._pending.get(serial)
                stale = (
                    self._closed
                    or presentation_epoch != self._presentation_epoch
                    or completion.cancelled()
                    or current is None
                    or current.completion is not completion
                    or (
                        None if self._surface is None else self._surface.host
                    ) is not host
                )
                if not stale:
                    self._pending[serial] = replace(
                        current,
                        plot_input=plot_input,
                        event_records=event_records,
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
                # The host COMMITTED it.  Handing data over is not the
                # same event: a render cancelled while queued never
                # reached the session, and its revision is still free.
                if handed:
                    self._note_held(*handed[-1])
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

        description = getattr(operation, "value", None)
        if not hasattr(description, "spec"):
            raise TypeError(
                "a plotted surface operation must carry DisplayDescription"
            )
        with self._state_lock:
            prepared = self._pending.get(update.serial)
            return (
                not self._closed
                and update.host_token == self._host_token_locked()
                and prepared is not None
                and prepared.presentation_epoch == self._presentation_epoch
            )

    def _note_held(self, generation: object, revision: object) -> None:
        """Record what the plotting host now holds."""

        value = self._revision_value(revision)
        with self._state_lock:
            if self._held_generation != generation:
                self._held_generation = generation
                self._held_revision = value
            elif value is not None and (
                self._held_revision is None or value > self._held_revision
            ):
                self._held_revision = value

    def _advance_staged(
        self,
        generation: object,
        revision: object,
        front_refs: tuple[object, ...],
        presentation_epoch: int,
    ) -> None:
        """Record that the host now holds ``revision`` of ``generation``."""

        value = self._revision_value(revision)
        self._staged_front_refs = tuple(front_refs)
        self._staged_presentation_epoch = int(presentation_epoch)
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
        """Present pixels, then atomically advance the accepted screen truth."""

        with self._state_lock:
            if self._closed or update.host_token != self._host_token_locked():
                return False
            prepared = self._pending.pop(update.serial, None)
            if (
                prepared is None
                or prepared.presentation_epoch != self._presentation_epoch
            ):
                return False
            publication = prepared.publication
            replacement = prepared.replacement_host
            previous = self._surface
            old_host = None if previous is None else previous.host
            target_host = replacement if replacement is not None else old_host

        if target_host is None:
            return False
        # A presented update that carries every declared signal ends the
        # waiting condition; a partial one (base without its overlay) keeps
        # it standing, which is exactly what the card should say.
        if len(update.front_refs) >= len(self.front_signals):
            with self._state_lock:
                self.waiting_condition = ""
        presented, present_error = self._put_on_screen(target_host, operation)
        if not presented:
            if replacement is not None:
                self._close_staged_host(replacement)
            if present_error is not None:
                with self._state_lock:
                    self.last_error = present_error
            # A refusal WITHOUT an exception is the widget's documented
            # answer for a stale race -- the operator zoomed or dragged
            # while this front was in flight, and the host's own newer
            # front paints instead.  Recording it as a panel error turned
            # every continuous gesture into a red-message stream.
            with self._state_lock:
                accepted_surface = self._surface
            if (
                accepted_surface is None
                or accepted_surface.publication is not publication
            ):
                # Only a candidate carrying an UNSEEN publication owes a
                # rebuild.  Re-invalidating for every refused re-describe
                # kept a render-debt churn running underneath the very
                # gesture that made the fronts stale -- the intermittent
                # lag an operator feels as "smooth, stuck, smooth".
                self._request_invalidation()
            return True

        description = operation.value
        accepted = replace(
            prepared,
            host=target_host,
            description=description,
            replacement_host=None,
            operation=None,
            completion=None,
        )
        with self._state_lock:
            self._surface = accepted
            self._projection_target = prepared.target
            self._advance_staged(
                _generation_of(prepared.plot_input),
                _revision_of(prepared.plot_input),
                prepared.front_refs,
                prepared.presentation_epoch,
            )
            self.last_error = None

        callback_error: BaseException | None = None
        if replacement is not None:
            try:
                if self._accept_host is not None:
                    self._accept_host(old_host)
            except BaseException as error:
                # Staging already proved the new host is usable.  Application
                # bookkeeping failure is reported on this panel but must not
                # tear an already-accepted cohort in half.
                callback_error = error
        if self._on_presented is not None:
            try:
                self._on_presented(accepted)
            except BaseException as error:
                callback_error = error
        if callback_error is not None:
            with self._state_lock:
                self.last_error = callback_error
        return True

    def accept_configuration(
        self,
        operation: object,
        target: object,
    ) -> object | None:
        """Commit one live configure result only after its front reaches screen."""

        with self._state_lock:
            surface = self._surface
            if self._closed or surface is None:
                return None
            host = surface.host
        presented, error = self._put_on_screen(host, operation)
        if not presented:
            if error is not None:
                with self._state_lock:
                    self.last_error = error
            # As above: refusal without an exception is stale-race flow
            # control, not a failure to report.
            self._request_invalidation()
            return None
        description = getattr(operation, "value", None)
        if not hasattr(description, "spec"):
            raise TypeError("a live plot configuration must return DisplayDescription")
        with self._state_lock:
            current = self._surface
            if current is None or current.host is not host:
                return None
            accepted = replace(
                current,
                target=target,
                description=description,
            )
            self._surface = accepted
            self._projection_target = target
            self.last_error = None
            return accepted

    def _put_on_screen(
        self, host: object, operation: object
    ) -> tuple[bool, BaseException | None]:
        if self._present is None:
            return True, None
        try:
            return bool(self._present(host, operation)), None
        except BaseException as error:
            return False, error

    def _request_invalidation(self) -> None:
        if self._invalidate is None:
            return
        try:
            self._invalidate(self._panel_id)
        except BaseException as error:
            with self._state_lock:
                if self.last_error is None:
                    self.last_error = error

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
                        _generation_of(prepared.plot_input),
                        _revision_of(prepared.plot_input),
                        prepared.front_refs,
                        prepared.presentation_epoch,
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
                        _generation_of(prepared.plot_input),
                        _revision_of(prepared.plot_input),
                        prepared.front_refs,
                        prepared.presentation_epoch,
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
        """The signal this panel is presenting without, or held back for.

        This used to be ``del missing_signal`` -- a panel frozen behind a
        companion that would never publish showed NOTHING: no error, no
        waiting, an operator staring at a stale frame with no way to learn
        why.  The name is now a standing condition on the card, cleared the
        moment an update carrying every declared signal is presented.
        """

        with self._state_lock:
            self.waiting_condition = (
                f"waiting for {missing_signal!r} to publish"
            )
