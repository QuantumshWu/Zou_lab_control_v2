"""Process-isolated execution of the existing :class:`RasterPlotHost`.

The main process owns Qt, Runtime and the immutable panel record.  This module
keeps the plotting host's public asynchronous surface in that process while
running its unchanged PlotSession, fit and renderer in one dedicated child.
Two instances are used by the application: one for live Monitor surfaces and
one for Edit/export work.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
import ctypes
from dataclasses import dataclass
import multiprocessing
from multiprocessing.connection import Connection
from multiprocessing.shared_memory import SharedMemory
import os
from pathlib import Path
import pickle
from queue import Empty, Queue
from threading import Condition, Event, Lock, RLock, Thread, current_thread
from time import monotonic
import traceback
from types import SimpleNamespace
from uuid import uuid4
import weakref

#: The name every render child is spawned under.  ``_spawn_child`` gives it,
#: and the child reads it back below, before anything numerical has loaded.
_CHILD_NAME_PREFIX = "zlc-render-"

# A RENDER CHILD USES ONE BLAS THREAD.  OpenBLAS commits a scratch buffer
# for every thread it may use the moment its library loads, and numpy and
# scipy each carry a copy of the library: measured, a child on a
# sixteen-core machine committed 1258 MB of address space, about a gigabyte
# of it two thread pools it never touches -- the solvers that multiply
# matrices, the SLM hologram solver and the feedback regressions, live in
# the parent.  With four warm children standing that was four gigabytes of
# commit charge against the page file for nothing.
#
# HERE, because this is the first thing a child imports: its target is in
# this module, so unpickling the Process brings this module in before numpy
# has loaded, and the environment is read when the library loads.  The
# product's own bootstrap never runs in a child -- a package ``__main__`` is
# not re-run by a spawned process -- so the child's environment is the
# parent's plus what this line adds.  The parent bounds its own pool to a
# worker team of four in its bootstrap, and a child inherits that: SET
# here, not defaulted, because an inherited four is the parent's answer
# for the parent, not an operator's word about a child that multiplies
# nothing.
if multiprocessing.current_process().name.startswith(_CHILD_NAME_PREFIX):
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np  # noqa: E402

from .front import (  # noqa: E402
    RasterBuffer,
    RasterFront,
    RasterIdentity,
    RasterInteractionMap,
    RasterOperation,
)


_INPUT_REF = "zlc-render-input"
_IMAGE_FRAME_REF = "zlc-render-image-frame"
_VALUE_TAG = "zlc-render-value"
_MAPPING_OPENED = object()
_MAPPING_RELEASED = object()
_RETIRE_MAPPINGS = object()
_STOP_WRITER = object()
#: How long either end waits for the pipe before looking up from it: the
#: child to notice its writer lost the parent, the parent to ask a silent
#: child whether it is still there.
_POLL_SLICE_SECONDS = 1.0

#: When a child warms its panel's fit: in the first gap this long in the
#: parent's requests after the panel's first front, or at the deadline if
#: the parent never leaves one.  The warm is 170-200 ms of reading kernels
#: off the disk cache on the child's one interpreter, so a frame that
#: overlaps it pays 5-45 ms of contention; a shot is half a second or more
#: apart and the gap after the first frame holds the whole warm.  A
#: producer at 25 Hz never leaves a gap this long, so the deadline starts
#: it anyway -- late enough that the panel's own first frames are clean,
#: soon enough that the operator's first fit finds it done.
_FIT_WARM_QUIET_SECONDS = 0.1
_FIT_WARM_DEADLINE_SECONDS = 1.0


def _encode_message(message: object) -> bytes:
    """One owned pickle, made where the message is made.

    Python 3.13's ``Connection.send`` serialises through
    ``ForkingPickler.dumps``, which returns ``BytesIO.getbuffer()``.  A pipe
    partial-write can keep that exported view alive while the temporary
    BytesIO is finalized, producing an unraisable BufferError.  This protocol
    exchanges only ordinary pickle values after process startup, so an owned
    bytes payload is both sufficient and lifetime-safe.

    Encoded by the SENDER, never by the writer thread: a value that cannot
    cross the pipe fails the call that made it, where that call can answer
    with an error instead.  Encoded by the writer, one such value read as a
    lost peer -- the writer said so once and then drained every later front,
    result and acknowledgement for the rest of the child's life, while the
    child stayed alive and kept rendering into that void.
    """

    return pickle.dumps(message, protocol=5)


def _write_messages(connection: Connection, outbox: Queue, closed: Callable[[], None]) -> None:
    """The ONLY thread that writes to one end of the pipe.

    ``Connection.send_bytes`` blocks until the peer reads, and on Windows it
    waits INFINITE with no timeout to pass.  The pipe buffer is 8192 bytes and
    a legal 64-cell facet grid's front message is 9750, so a write that has to
    wait is ordinary rather than exceptional.  Any thread that ALSO has to
    read -- the child's service loop, the parent's GUI thread -- must therefore
    never perform the write itself: while it waited, it stopped draining, the
    peer's own write filled, and both ends held a send lock forever.  One
    writer per direction, fed by a queue, removes the cycle by construction:
    the readers keep reading no matter how far behind the writes fall, and
    ordering is the queue's.

    It moves bytes its senders already encoded (:func:`_encode_message`), so
    the only failure it can meet is the pipe's own, and that is the peer
    gone.
    """

    while True:
        payload = outbox.get()
        if payload is _STOP_WRITER:
            return
        try:
            connection.send_bytes(payload)
        except BaseException:
            # The peer is gone.  Say so once and drain, so nothing waits on a
            # queue nobody will ever write out; the reader's own EOF is what
            # fails the pending requests.
            closed()
            while True:
                if outbox.get() is _STOP_WRITER:
                    return


def _receive_message(connection: Connection) -> object:
    """Receive from the matching owned-bytes process protocol."""

    return pickle.loads(connection.recv_bytes())


def _plain(value: object) -> object:
    """Copy immutable mapping views into the process wire vocabulary."""

    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_plain(item) for item in value)
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _wire_display_state(state: object) -> tuple[object, ...]:
    return (
        _VALUE_TAG,
        "display-state",
        int(state.revision),
        dict(state.values),
        tuple(state.changed_names),
        int(state.effects),
    )


def _wire_fit_summary(result: object) -> dict[str, object]:
    """The exact parameter table consumers publish, without lazy image planes."""

    outcomes = getattr(result, "results", None)
    if outcomes is None:
        return {
            "kind": "scalar",
            "parameter_names": tuple(result.parameter_names),
            "parameter_units": dict(result.parameter_units),
            "parameter_values": np.asarray(result.parameter_values),
            "standard_errors": np.asarray(result.standard_errors),
            "parameter_error_validity": dict(result.parameter_error_validity),
            "success": bool(result.success),
            "source_revision": int(result.source_revision),
            "batch_revision": int(result.batch_revision),
        }
    return {
        "kind": "batch",
        "parameter_names": tuple(result.parameter_names),
        "parameter_units": dict(result.parameter_units),
        "parameter_values": {
            name: np.asarray(values)
            for name, values in result.parameter_values.items()
        },
        "parameter_errors": {
            name: np.asarray(values)
            for name, values in result.parameter_errors.items()
        },
        "outcomes": tuple(
            None if item is None else bool(item.success) for item in outcomes
        ),
        "facet": result.facet,
        "sample_axis_name": str(result.sample_axis_name),
        "sample_coordinates": (
            None
            if result.sample_coordinates is None
            else np.asarray(result.sample_coordinates)
        ),
        "sample_unit": str(result.sample_unit),
        "sample_labels": result.sample_labels,
        "source_revision": int(result.source_revision),
        "batch_revision": int(result.batch_revision),
    }


def _wire_complete_fit_result(result: object) -> dict[str, object]:
    outcomes = getattr(result, "results", None)
    if outcomes is None:
        return {
            "kind": "scalar",
            "model": result.model,
            "parameter_values": np.asarray(result.parameter_values),
            "standard_errors": np.asarray(result.standard_errors),
            "covariance": np.asarray(result.covariance),
            "fitted_values": np.asarray(result.fitted_values),
            "residuals": np.asarray(result.residuals),
            "selected_indices": np.asarray(result.selected_indices),
            "source_revision": int(result.source_revision),
            "success": bool(result.success),
            "message": str(result.message),
            "reduced_chi_square": float(result.reduced_chi_square),
            "covariance_valid": bool(result.covariance_valid),
            "parameter_units": dict(result.parameter_units),
            "batch_revision": int(result.batch_revision),
            "fixed_parameter_names": tuple(result.fixed_parameter_names),
            "reduced": bool(result.reduced),
            "evidence": float(result.evidence),
        }
    return {
        "kind": "batch",
        "facet": result.facet,
        "facet_values": tuple(result.facet_values),
        "model": result.model,
        "results": tuple(
            None if item is None else _wire_complete_fit_result(item)
            for item in result.results
        ),
        "failure_messages": tuple(result.failure_messages),
        "source_revision": int(result.source_revision),
        "overlays": tuple(result.overlays),
        "parameter_units": dict(result.parameter_units),
        "sample_axis_name": str(result.sample_axis_name),
        "sample_coordinates": result.sample_coordinates,
        "sample_unit": str(result.sample_unit),
        "sample_labels": result.sample_labels,
        "batch_revision": int(result.batch_revision),
    }


def _wire_value(value: object) -> object:
    """Encode only values whose immutable implementation is not pickleable."""

    from .fit import FacetFitBatchResult, FitResult
    from .session import DisplayDescription, SelectionData
    from .state import DisplayState
    from ._session_state import FitEvent

    if isinstance(value, DisplayDescription):
        return (
            _VALUE_TAG,
            "display-description",
            {
                "kind": value.kind,
                "spec": value.spec,
                "size": value.size,
                "size_choices": tuple(value.size_choices),
                "parameter_schema": tuple(
                    {
                        "name": parameter.name,
                        "value_type": parameter.value_type,
                        "effects": int(parameter.effects),
                        "default": parameter.default,
                        "allow_none": parameter.allow_none,
                        "label": parameter.label,
                        "choices": tuple(parameter.choices),
                        "minimum": parameter.minimum,
                        "maximum": parameter.maximum,
                        "step": parameter.step,
                        "portable": parameter.portable,
                    }
                    for parameter in value.parameter_schema.values()
                ),
                "display_state": _wire_display_state(value.display_state),
                "parameter_choices": _plain(value.parameter_choices),
                "automatic_values": _plain(value.automatic_values),
                "limits": value.limits,
                "viewport": value.viewport,
                "semantics": value.semantics,
                "selection_subject": value.selection_subject,
                "selectors": tuple(value.selectors),
                "classifier_thresholds": _plain(value.classifier_thresholds),
                "facet_focus": value.facet_focus,
                "fit": _plain(value.fit),
                "fit_models": tuple(
                    {
                        "model_id": str(model.model_id),
                        "display_name": str(model.display_name),
                        "parameters": tuple(
                            (str(parameter.name), str(parameter.symbol or parameter.name))
                            for parameter in model.parameters
                        ),
                    }
                    for model in value.fit_models
                ),
                "fit_expression": value.fit_expression,
                "fit_expression_error": value.fit_expression_error,
            },
        )
    if isinstance(value, DisplayState):
        return _wire_display_state(value)
    if isinstance(value, FitEvent):
        return (
            _VALUE_TAG,
            "fit-event",
            {
                "result": _wire_fit_summary(value.result),
                "source_generation": value.source_generation,
            },
        )
    if isinstance(value, (FitResult, FacetFitBatchResult)):
        return (_VALUE_TAG, "fit-result", _wire_complete_fit_result(value))
    if isinstance(value, SelectionData):
        return (
            _VALUE_TAG,
            "selection-data",
            {
                "selector": value.selector,
                "mask": value.mask,
                "flat_indices": value.flat_indices,
                "canonical_values": value.canonical_values,
                "display_values": value.display_values,
                "canonical_coordinates": dict(value.canonical_coordinates),
                "display_coordinates": dict(value.display_coordinates),
                "data_revision": value.data_revision,
                "facet_index": value.facet_index,
            },
        )
    # SelectionEvent contains mappingproxy classifier records.
    if type(value).__name__ == "SelectionEvent":
        return (
            _VALUE_TAG,
            "selection-event",
            {
                "change": value.change,
                "selector": value.selector,
                "display_selector": value.display_selector,
                "data_revision": value.data_revision,
                "data_generation": value.data_generation,
                "subject": value.subject,
                "classifier_thresholds": _plain(value.classifier_thresholds),
            },
        )
    return _plain(value)


def _restore_fit_summary(document: Mapping[str, object]) -> object:
    kind = str(document["kind"])
    if kind == "scalar":
        return SimpleNamespace(
            parameter_names=tuple(document["parameter_names"]),
            parameter_units=dict(document["parameter_units"]),
            parameter_values=np.asarray(document["parameter_values"]),
            standard_errors=np.asarray(document["standard_errors"]),
            parameter_error_validity=dict(document["parameter_error_validity"]),
            success=bool(document["success"]),
            source_revision=int(document["source_revision"]),
            batch_revision=int(document["batch_revision"]),
        )
    outcomes = tuple(
        None if item is None else SimpleNamespace(success=bool(item))
        for item in document["outcomes"]
    )
    return SimpleNamespace(
        parameter_names=tuple(document["parameter_names"]),
        parameter_units=dict(document["parameter_units"]),
        parameter_values={
            name: np.asarray(values)
            for name, values in document["parameter_values"].items()
        },
        parameter_errors={
            name: np.asarray(values)
            for name, values in document["parameter_errors"].items()
        },
        results=outcomes,
        facet=document["facet"],
        sample_axis_name=str(document["sample_axis_name"]),
        sample_coordinates=(
            None
            if document["sample_coordinates"] is None
            else np.asarray(document["sample_coordinates"])
        ),
        sample_unit=str(document["sample_unit"]),
        sample_labels=document["sample_labels"],
        source_revision=int(document["source_revision"]),
        batch_revision=int(document["batch_revision"]),
    )


def _restore_complete_fit_result(document: Mapping[str, object]) -> object:
    from .fit import FacetFitBatchResult, FitResult

    if document["kind"] == "scalar":
        return FitResult(
            model=document["model"],
            parameter_values=document["parameter_values"],
            standard_errors=document["standard_errors"],
            covariance=document["covariance"],
            fitted_values=document["fitted_values"],
            residuals=document["residuals"],
            selected_indices=document["selected_indices"],
            source_revision=document["source_revision"],
            success=document["success"],
            message=document["message"],
            reduced_chi_square=document["reduced_chi_square"],
            covariance_valid=document["covariance_valid"],
            parameter_units=document["parameter_units"],
            batch_revision=document["batch_revision"],
            fixed_parameter_names=document["fixed_parameter_names"],
            reduced=document["reduced"],
            evidence=document["evidence"],
        )
    return FacetFitBatchResult(
        facet=document["facet"],
        facet_values=document["facet_values"],
        model=document["model"],
        results=tuple(
            None if item is None else _restore_complete_fit_result(item)
            for item in document["results"]
        ),
        failure_messages=document["failure_messages"],
        source_revision=document["source_revision"],
        overlays=document["overlays"],
        parameter_units=document["parameter_units"],
        sample_axis_name=document["sample_axis_name"],
        sample_coordinates=document["sample_coordinates"],
        sample_unit=document["sample_unit"],
        sample_labels=document["sample_labels"],
        batch_revision=document["batch_revision"],
    )


def _unwire_value(value: object) -> object:
    if not (
        isinstance(value, tuple)
        and len(value) >= 2
        and value[0] == _VALUE_TAG
    ):
        return value
    kind = value[1]
    if kind == "display-state":
        from .parameters import FrozenParameters, RenderEffect
        from .state import DisplayState

        return DisplayState(
            int(value[2]),
            FrozenParameters(value[3]),
            frozenset(value[4]),
            RenderEffect(int(value[5])),
        )
    document = value[2]
    if kind == "display-description":
        from .parameters import ParameterSchema, ParameterSpec, RenderEffect

        state = _unwire_value(document["display_state"])
        spec = document["spec"]
        return SimpleNamespace(
            kind=document["kind"],
            spec=spec,
            size=document["size"],
            size_choices=tuple(document["size_choices"]),
            parameter_schema=ParameterSchema(
                ParameterSpec(
                    name=parameter["name"],
                    value_type=parameter["value_type"],
                    effects=RenderEffect(parameter["effects"]),
                    default=parameter["default"],
                    allow_none=parameter["allow_none"],
                    label=parameter["label"],
                    choices=tuple(parameter["choices"]),
                    minimum=parameter["minimum"],
                    maximum=parameter["maximum"],
                    step=parameter["step"],
                    portable=parameter["portable"],
                )
                for parameter in document["parameter_schema"]
            ),
            display_state=state,
            parameter_choices=dict(document["parameter_choices"]),
            automatic_values=dict(document["automatic_values"]),
            limits=document["limits"],
            viewport=document["viewport"],
            semantics=document["semantics"],
            selection_subject=document["selection_subject"],
            selectors=tuple(document["selectors"]),
            classifier_thresholds=tuple(document["classifier_thresholds"]),
            facet_focus=document["facet_focus"],
            fit=dict(document["fit"]),
            fit_models=tuple(
                SimpleNamespace(
                    model_id=model["model_id"],
                    display_name=model["display_name"],
                    parameters=tuple(
                        SimpleNamespace(name=name, symbol=symbol)
                        for name, symbol in model["parameters"]
                    ),
                    parameter_names=tuple(
                        name for name, _symbol in model["parameters"]
                    ),
                    symbols=tuple(
                        symbol for _name, symbol in model["parameters"]
                    ),
                )
                for model in document["fit_models"]
            ),
            fit_expression=str(document["fit_expression"]),
            fit_expression_error=str(document["fit_expression_error"]),
        )
    if kind == "fit-result":
        return _restore_complete_fit_result(document)
    if kind == "fit-event":
        return SimpleNamespace(
            result=_restore_fit_summary(document["result"]),
            source_generation=document["source_generation"],
        )
    if kind == "selection-data":
        return SimpleNamespace(**document)
    if kind == "selection-event":
        return SimpleNamespace(**document)
    raise RuntimeError(f"unknown render value tag {kind!r}")


def _release_shared_store(
    process: "RenderProcess",
    lease_id: str,
    cache: "_SharedMappingCache",
    name: str,
) -> None:
    process._release_front(str(lease_id))
    cache.release(str(name))


class _SharedMappingCache:
    """One mapping per shared segment, kept while a frontend is reading it.

    A child cycles a bounded set of front segments -- three per Host -- so
    the same names come back frame after frame.  MAPPING one costs nothing,
    about thirty microseconds; TOUCHING it costs the page faults of the
    whole raster, and a fresh mapping faults every page again.  Measured at
    the 2x2 preset, reading a 9.2 MB front through a new mapping is 1.8 to
    2.9 ms of the GUI thread and through a kept one 0.03 ms.  On a
    four-card board at the display rate that was eighty milliseconds a
    second, on the one thread that has to stay answerable.

    A name is never reused for a different segment -- the child names each
    one when it creates it -- so a cached mapping is always the segment its
    name meant.  What a mapping must outlive is the frontend's last view of
    it, which may outlive the RenderProcess itself, so each is counted and
    handed to the retirement lane only once nothing is reading it AND this
    cache has been told to go.  Holding no reference back to the process,
    it is what that process's own finalizer can retire.
    """

    __slots__ = ("_lock", "_entries", "_closing", "_retirements")

    def __init__(self, retirements: Queue) -> None:
        self._lock = Lock()
        #: name -> [mapping, how many stores are reading it]
        self._entries: dict[str, list] = {}
        self._closing = False
        self._retirements = retirements

    def open(self, name: str) -> object:
        """The mapping for one segment.  The reader thread alone calls it."""

        with self._lock:
            entry = self._entries.get(name)
            if entry is not None:
                entry[1] += 1
                return entry[0]
            closing = self._closing
        mapping = _open_shared_mapping(name)
        self._retirements.put(_MAPPING_OPENED)
        if closing:
            # A front that crossed the pipe after this was told to go: read
            # it, and let the mapping follow its store straight out.
            self._retirements.put((_MAPPING_RELEASED, mapping))
            return mapping
        with self._lock:
            self._entries[name] = [mapping, 1]
        return mapping

    def release(self, name: str) -> None:
        """One store finished with a segment; retire the map if it was last."""

        with self._lock:
            entry = self._entries.get(name)
            if entry is None:
                return
            entry[1] -= 1
            if entry[1] > 0 or not self._closing:
                return
            del self._entries[name]
        self._retirements.put((_MAPPING_RELEASED, entry[0]))

    def retire_all(self) -> None:
        """Told to go: release what nothing is reading, and mark the rest."""

        with self._lock:
            self._closing = True
            idle = [
                (name, entry)
                for name, entry in self._entries.items()
                if entry[1] <= 0
            ]
            for name, _entry in idle:
                del self._entries[name]
        for _name, entry in idle:
            self._retirements.put((_MAPPING_RELEASED, entry[0]))
        self._retirements.put(_RETIRE_MAPPINGS)


def _retire_shared_mappings(retirements: Queue) -> None:
    waiting: list[object] = []
    outstanding = 0
    stopping = False
    while True:
        try:
            # Nothing retained means nothing to retry, and the next event is
            # the only thing that can change that: WAITING on it costs one
            # wake instead of fifty a second in the GUI process, whose
            # thread has to stay answerable.
            item = retirements.get(timeout=0.002 if waiting else None)
            if item is _RETIRE_MAPPINGS:
                stopping = True
            elif item is _MAPPING_OPENED:
                outstanding += 1
            else:
                kind, shared = item
                if kind is not _MAPPING_RELEASED or outstanding <= 0:
                    raise RuntimeError("invalid shared mapping retirement event")
                outstanding -= 1
                waiting.append(shared)
        except Empty:
            pass
        if stopping and outstanding == 0 and not waiting:
            return
        if not waiting:
            continue
        retained: list[SharedMemory] = []
        for shared in waiting:
            try:
                shared.close()
            except BufferError:
                retained.append(shared)
            except Exception:
                continue
        waiting = retained


def _open_shared_memory(name: str) -> SharedMemory:
    try:
        return SharedMemory(name=name, create=False, track=False)
    except TypeError:  # Python 3.11/3.12 do not expose ``track``.
        shared = SharedMemory(name=name, create=False)
        try:
            from multiprocessing import resource_tracker

            resource_tracker.unregister(shared._name, "shared_memory")
        except Exception:
            pass
        return shared


def _open_shared_mapping(name: str) -> object:
    """Detach one read mapping from SharedMemory's noisy wrapper lifetime.

    The frontend buffer may deliberately outlive its RenderProcess.  A
    ``SharedMemory`` object prints ``BufferError`` from ``__del__`` if Python
    exits while a QImage/ndarray still exports its mmap.  The mmap itself has
    exactly the lifetime needed here and deallocates quietly; detach it after
    releasing the wrapper's own view, and let the existing retirement lane
    close it once the last frontend owner is gone.
    """

    shared = _open_shared_memory(name)
    view = shared._buf
    mapping = shared._mmap
    if view is None or mapping is None:
        shared.close()
        raise RuntimeError("shared raster mapping is unavailable")
    view.release()
    shared._buf = None
    shared._mmap = None
    file_descriptor = getattr(shared, "_fd", -1)
    if file_descriptor >= 0:
        os.close(file_descriptor)
        shared._fd = -1
    return mapping


@dataclass(slots=True)
class _Pending:
    future: Future
    host_id: str | None = None
    subscription_id: int | None = None
    input_tokens: tuple[int, ...] = ()
    input_transition: str = ""
    raw_result: bool = False


_REMOTE_METHODS = frozenset(
    {
        "update_data", "update_image_overlay", "update_image_frame",
        "set_parameter", "set_parameters", "configure", "describe_display",
        "describe_semantics", "replace_spec", "apply_semantic",
        "resolved_color_limits", "set_labels", "set_relim_mode",
        "set_y_limits", "reset_y_limits", "set_color_limits",
        "reset_color_limits", "set_x_limits", "set_view_limits", "set_size",
        "set_device_pixel_ratio", "set_axis_unit", "set_value_unit",
        "set_time_unit", "save", "clear_fit", "fit_models", "selectors",
        "selector_state", "selector_data", "remove_selector",
        "set_selector_value", "set_area_selector", "set_x_selector",
        "set_threshold_selector", "set_crosshair_selector", "fit",
        "pointer_event", "set_viewport", "focus_facet",
        "show_facet_overview", "reset_viewport",
    }
)


class _RemoteRasterPlotHost:
    """Main-process facade retaining the existing asynchronous Host contract."""

    def __init__(
        self,
        process: "RenderProcess",
        host_id: str,
        defaults: object,
    ) -> None:
        self._process = process
        self._process_pid = process.pid
        self._host_id = host_id
        self._defaults = defaults
        self._front: RasterFront | None = None
        self._front_ready = Event()
        self._front_callbacks: list[Callable[[RasterFront], None]] = []
        self._lock = RLock()
        self._closing = False
        self._closed = False
        #: Whether the child has been ASKED to close this host.  Separate from
        #: ``_closing``, which says the host is on its way out however it got
        #: there: a service failure sets that, and reading it here meant a
        #: failed host never sent its ``close-host``, so the acknowledgement
        #: that is the only thing that sets ``_closed`` could never arrive and
        #: the console waited on that worker for ever.
        self._close_requested = False
        self._startup_error: Exception | None = None
        self._service_failure = False
        self._initial_metadata: tuple[object, object] | None = None
        self._initial_error: BaseException | None = None
        self._interaction_enabled = True
        self._qt_widget = None

    @property
    def host_id(self) -> str:
        return self._host_id

    @property
    def process_pid(self) -> int | None:
        return self._process_pid

    @property
    def process_name(self) -> str:
        return self._process.name

    @property
    def defaults(self) -> object:
        return self._defaults

    @property
    def front(self) -> RasterFront | None:
        with self._lock:
            return self._front

    @property
    def logical_size(self) -> tuple[int, int] | None:
        front = self.front
        return None if front is None else tuple(front.logical_size)

    @property
    def startup_failure(self) -> Exception | None:
        with self._lock:
            return self._startup_error

    @property
    def service_failure(self) -> bool:
        """Whether startup became unusable because its child process died."""

        with self._lock:
            return self._service_failure

    @property
    def initial_state(self) -> tuple[tuple[object, object] | None, BaseException | None]:
        with self._lock:
            return self._initial_metadata, self._initial_error

    @property
    def closing(self) -> bool:
        with self._lock:
            return self._closing or self._closed or not self._process.alive

    @property
    def interaction_enabled(self) -> bool:
        return self._interaction_enabled

    def set_interaction_enabled(self, enabled: bool) -> None:
        self._interaction_enabled = bool(enabled)
        widget = self._qt_widget
        if widget is not None:
            widget.set_interaction_enabled(bool(enabled))

    def qt_widget(self, *, auto_present: bool | None = None):
        widget = self._qt_widget
        if auto_present is not None and not isinstance(auto_present, bool):
            raise TypeError("auto_present must be boolean or None")
        if widget is not None and auto_present is not None and widget._auto_present != auto_present:
            raise ValueError("the host's Qt presentation policy is already fixed")
        if widget is None:
            from .backends import Qt5PlotWidget

            widget = Qt5PlotWidget(self, auto_present=True if auto_present is None else auto_present)
            self._qt_widget = widget
            if not self._interaction_enabled:
                widget.set_interaction_enabled(False)
        return widget

    def wait_for_front(self, timeout: float | None = None) -> RasterFront:
        front = self.front
        if front is not None:
            return front
        if not self._front_ready.wait(timeout):
            raise TimeoutError("remote raster host did not publish its first front")
        front = self.front
        if front is not None:
            return front
        error = self.startup_failure
        if error is not None:
            raise RuntimeError("remote raster host failed to start") from error
        raise RuntimeError("remote raster host closed before its first front")

    def subscribe_front(
        self, callback: Callable[[RasterFront], None]
    ) -> Callable[[], None]:
        if not callable(callback):
            raise TypeError("front callback must be callable")
        with self._lock:
            if self._closing or self._closed:
                raise RuntimeError("remote raster host is closing")
            self._front_callbacks.append(callback)

        def release() -> None:
            with self._lock:
                if callback in self._front_callbacks:
                    self._front_callbacks.remove(callback)

        return release

    def subscribe_display(self, callback: Callable[[object], object]) -> Future:
        return self._process._subscribe(self, "display", callback)

    def subscribe_viewport(self, callback: Callable[[object], object]) -> Future:
        return self._process._subscribe(self, "viewport", callback)

    def subscribe_facet_focus(self, callback: Callable[..., object]) -> Future:
        return self._process._subscribe(self, "facet-focus", callback)

    def subscribe_fit(
        self,
        callback: Callable[[object], object],
        *,
        replay_current: bool = False,
    ) -> Future:
        return self._process._subscribe(
            self, "fit", callback, replay_current=bool(replay_current)
        )

    def subscribe_selection(self, callback: Callable[[object], object]) -> Future:
        return self._process._subscribe(self, "selection", callback)

    def __getattr__(self, name: str) -> object:
        if name not in _REMOTE_METHODS:
            raise AttributeError(name)

        def invoke(*args: object, **kwargs: object) -> Future:
            return self._process._call(self, name, args, kwargs)

        return invoke

    def _accept_front(self, front: RasterFront) -> None:
        with self._lock:
            if self._closed:
                return
            current = self._front
            if current is not None and (
                front.identity.sequence <= current.identity.sequence
            ):
                return
            self._front = front
            callbacks = tuple(self._front_callbacks)
            self._front_ready.set()
        for callback in callbacks:
            try:
                callback(front)
            except Exception:
                continue

    def _created(self, description: object) -> None:
        with self._lock:
            self._initial_metadata = (
                description,
                tuple(getattr(description, "fit_models", ())),
            )

    def _failed(
        self,
        error: BaseException,
        *,
        service_failure: bool = False,
    ) -> None:
        with self._lock:
            if isinstance(error, Exception):
                self._startup_error = error
            self._service_failure = bool(service_failure)
            self._initial_error = error
            self._closing = True
            self._front_ready.set()

    def _mark_closed(self) -> None:
        with self._lock:
            self._closing = True
            self._closed = True
            self._front_ready.set()

    def _finish_local_close(self) -> None:
        widget = self._qt_widget
        if widget is not None:
            try:
                from PyQt5 import QtCore

                if QtCore.QThread.currentThread() == widget.thread():
                    widget.close_adapter()
                    self._qt_widget = None
            except Exception:
                pass
        with self._lock:
            self._front_callbacks.clear()
            self._front = None

    def close(self, *, timeout: float | None = None) -> bool:
        if timeout is not None and timeout < 0.0:
            raise ValueError("timeout must be non-negative or None")
        with self._lock:
            if self._closed:
                self._finish_local_close()
                return True
            first = not self._close_requested
            self._close_requested = True
            self._closing = True
        if first:
            self._process._close_host(self)
        if timeout == 0.0:
            return False
        limit = 30.0 if timeout is None else float(timeout)
        stopped = self._process._wait_host_closed(self._host_id, limit)
        if stopped:
            self._finish_local_close()
        return stopped


#: How many finished input segments one child's transport keeps to fill
#: again rather than destroying.  A producer's shot and its overlay are one
#: or two blocks, and a couple of shots may be in flight, so four covers the
#: rotation; past that a block is given back to the operating system, which
#: is what keeps a session that changed raster size from holding both.
_INPUT_FREE_BLOCKS = 4

#: How many children stand warm and unused before anything is drawing, so
#: a panel never waits for one.  Four, because a board is four cards: the
#: console's own layouts and the saved boards sit at or under four, so a
#: whole board opened at once finds every panel a warm child of its own.
DEFAULT_RENDER_SPARES = 4

#: How many stand warm once a board IS drawing.  A board arrives all at
#: once and then GROWS one panel at a time, so the opening count answers a
#: question nobody asks twice: holding four more idle renderers for the
#: rest of the session is most of a gigabyte against an operator who adds
#: one panel.  Two covers that, and the same number is the mark of a board
#: having arrived -- more than this many drawing is no longer an opening.
DEFAULT_RENDER_SETTLED_SPARES = 2


def _retire_member(member: "RenderProcess") -> None:
    """Tell one child to go, and do not wait for it to finish going.

    NOT a close.  The reclaim can run on a child's own reader thread -- that
    is the thread that reports a Host retired -- and ``close`` waits for that
    very thread to stop, so waiting here is a thirty-second stall ending in a
    terminate.  ``release`` sends the shutdown and returns; the child is
    daemonic and finishes on its own, and the pool's own close is where
    anybody waits.
    """

    try:
        member.release(0.0)
    except BaseException:  # noqa: BLE001 -- a child being let go cannot fail
        pass


class RenderProcessPool:
    """Render children kept warm ahead of the panels that will need them.

    One child held every live panel's renderer, and its workers shared one
    interpreter: the compiled kernels release the GIL, but artist updates,
    chrome drawing and the pickle of each published front do not, so four
    panels' Python ran one at a time however many cores were idle.  Flat out
    that is the whole ceiling -- at the operator's density four children draw
    2.3 to 3.0 times the frames of one (camera 4M 54.5 to 156.6 fps, facet64
    image 35.1 to 104.6, heatmap 44.3 to 102.1, curve 65.8 to 108.2).

    WHY THEY ARE KEPT WARM AHEAD OF TIME.  A child is 2.3 s from spawn to its
    first front.  Started when a panel first needed one, every mount waited
    out its own child's boot: four panels painted 2.4, 4.7, 7.0 and 7.1 s
    after the ask -- a 2.3 s staircase, one child's boot repeated.  Kept warm
    ahead, the same four paint in 0.35 s, the same as one child serving all
    four.  So the pool always holds ``spares`` children that no panel has
    touched, and starts a replacement the moment one is taken.

    WHAT IT COSTS is memory, and only memory: about two hundred megabytes per
    child, whether or not a panel ever lands on it.  The warm count steps
    down to bound that: a board arrives all at once, which is what
    ``spares`` is sized for, and then grows one panel at a time -- so once
    more than ``settled_spares`` children are drawing, the opening count has
    answered its question and ``settled_spares`` stand warm instead.

    ONE PANEL, ONE CHILD, WITHOUT EXCEPTION.  Two panels in one child put
    their Python -- artist updates, chrome, the pickle of every front --
    back on one interpreter, which is the ceiling this pool exists to lift,
    and a fit on one of them then takes frames from the other.  So a panel
    that finds no warm child never joins a busy one: a fresh child is
    started for it and the panel waits.  The wait is a boot, not a warm-up:
    the child answers the create as soon as its imports are in, cutting its
    own warming short.  And it happens only when panels arrive faster than
    replacements come up.

    FOUR RULES hold this together, and each of them is a way it would
    otherwise go wrong:

    * Slow work -- spawning a child, shutting one down -- never happens
      under this pool's lock, so a panel mounting never waits on a child
      being reclaimed.
    * This pool never calls INTO a child while holding its own lock.  A
      child's reader thread reports a retired Host while holding the
      child's, and a pool that asked a child anything under its own lock
      would close that cycle.
    * A child being retired is never handed out again, and a child that
      finishes starting after the pool is closing is retired immediately
      rather than joining it.
    * Every thread this pool starts is joined before ``close`` returns.

    ``build_host``, ``retain``, ``release`` and ``close`` behave exactly as
    one :class:`RenderProcess`'s do, so a caller chooses the counts and
    changes nothing else.
    """

    def __init__(
        self,
        name: str,
        *,
        spares: int = DEFAULT_RENDER_SPARES,
        settled_spares: int | None = None,
    ) -> None:
        selected = str(name).strip()
        if not selected:
            raise ValueError("render pool name must be non-empty")
        wanted = int(spares)
        # A pool that opens with fewer than the standing count has already
        # said what it wants; the default follows it down rather than
        # refusing a perfectly sensible small pool.
        settled = (
            min(DEFAULT_RENDER_SETTLED_SPARES, wanted)
            if settled_spares is None
            else int(settled_spares)
        )
        if wanted < 1 or settled < 1:
            raise ValueError("a pool keeps at least one child warm")
        if settled > wanted:
            raise ValueError("a settled pool cannot keep more warm than an opening one")
        self.name = selected
        self._spares = wanted
        self._settled_spares = settled
        self._lock = RLock()
        self._settled = Condition(self._lock)
        self._members: list[RenderProcess] = []
        #: Handed to a panel whose Host is not registered on them yet: idle
        #: by the child's own count, and not free.
        self._claimed: set[RenderProcess] = set()
        self._retiring: set[RenderProcess] = set()
        self._starting: set[Thread] = set()
        #: Panels waiting for a fresh child, each with a start of its own on
        #: the way.  Those starts are theirs, not the warm count's.
        self._waiting = 0
        self._failures: list[BaseException] = []
        self._serial = 0
        self._owners = 1
        self._closing = False
        self._keep_warm()

    # ------------------------------------------------------------- shaping
    def _idle(self, members: Sequence["RenderProcess"]) -> list["RenderProcess"]:
        """Children no Host is drawing on.  Asked OUTSIDE this pool's lock."""

        return [member for member in members if member.host_count == 0]

    def _warm_count(self, drawing: int) -> int:
        """How many stand warm while ``drawing`` children have a panel on them.

        The opening count answers "a whole board at once"; past that the
        board grows one panel at a time and the answer is the smaller one.
        Asked on every change of shape, so a board whose panels all close
        comes back up to the opening count for the next one.
        """

        return self._spares if drawing <= self._settled_spares else self._settled_spares

    def _keep_warm(self) -> None:
        """Start whatever is missing, retire whatever is spare.

        Called after every change of shape -- a Host built, a Host retired,
        a child started.  It decides under the lock and acts outside it.
        """

        while True:
            with self._lock:
                if self._closing:
                    return
                members = [
                    member for member in self._members
                    if member not in self._retiring
                ]
            unused = self._idle(members)
            with self._lock:
                if self._closing:
                    return
                idle = [member for member in unused if member not in self._claimed]
                pending = len(self._starting)
                warm = self._warm_count(len(members) - len(idle))
                # Every waiting panel holds one of the starts in flight; the
                # rest are the warm count's.
                reserved = min(pending, self._waiting)
                begin = max(0, warm - len(idle) - (pending - reserved))
                # A child is spare only beyond the warm count AND unused, and
                # only ever one at a time: between deciding and acting a panel
                # may have taken it, and the next turn sees that.  Never while
                # a panel is waiting for one: the child it is waiting for
                # arrives idle, and would be the one let go.
                spare = (
                    idle[-1]
                    if len(idle) > warm and not begin and not self._waiting
                    else None
                )
                # The picture was read without the lock, so the child chosen
                # may already have been taken, retired, or replaced.  Only one
                # this pool still holds may be let go.
                surplus = spare if spare in self._members else None
                threads = self._launch(begin)
                if surplus is not None:
                    self._retiring.add(surplus)
                    self._members.remove(surplus)
            for thread in threads:
                thread.start()
            if surplus is None:
                return
            _retire_member(surplus)
            with self._lock:
                self._retiring.discard(surplus)

    def _launch(self, count: int) -> list[Thread]:
        """Threads that each start one child: made under the lock, started outside it."""

        threads = []
        for _ in range(count):
            self._serial += 1
            thread = Thread(
                target=self._start_member,
                args=(self._serial,),
                name=f"zlc-render-{self.name}-start-{self._serial}",
                daemon=True,
            )
            self._starting.add(thread)
            threads.append(thread)
        return threads

    def _start_member(self, serial: int) -> None:
        member: RenderProcess | None = None
        try:
            member = RenderProcess(
                f"{self.name}-{serial}", host_retired=self._host_retired
            )
        except BaseException as error:  # noqa: BLE001 -- raised to the asker
            with self._lock:
                self._failures.append(error)
        stray = None
        with self._lock:
            self._starting.discard(current_thread())
            if member is not None:
                if self._closing:
                    stray = member
                else:
                    self._members.append(member)
            self._settled.notify_all()
        if stray is not None:
            # Closed while this one was starting: it must not outlive the
            # pool merely because it was late.  This IS a close, and it can
            # wait -- it runs on this starting thread, which the pool's own
            # close joins.
            if not stray.release(0.0):
                stray.close(30.0)

    def _host_retired(self) -> None:
        """A child finished with a Host, so the shape may have changed.

        Called from that child's reader thread, which is why this only shapes
        the pool and never waits: the reclaim itself runs here, outside every
        child's lock, and a pool closing takes precedence.
        """

        with self._lock:
            if self._closing:
                return
        self._keep_warm()

    # ------------------------------------------------------------- serving
    def _claim(self) -> "RenderProcess":
        """The child a new Host belongs to: a warm one, else a fresh one.

        Never a busy one.  A panel that finds no idle child starts one and
        waits for it to exist -- a boot, not a warm-up, since the child
        takes the create as soon as its imports are in.  Each waiting panel
        holds one start of its own, so two panels arriving together get two
        children and neither is handed the other's.  A child that fails to
        start fails the panel that was waiting for it.
        """

        failures_seen = len(self._failures)
        waiting = False
        try:
            while True:
                with self._lock:
                    if self._closing:
                        raise RuntimeError("render pool is closing")
                    members = [
                        member for member in self._members
                        if member not in self._retiring
                    ]
                unused = self._idle(members)
                with self._lock:
                    if self._closing:
                        raise RuntimeError("render pool is closing")
                    for member in unused:
                        if (
                            member in self._members
                            and member not in self._retiring
                            and member not in self._claimed
                        ):
                            self._claimed.add(member)
                            return member
                    if len(self._failures) > failures_seen:
                        raise self._failures[-1]
                    if not waiting:
                        self._waiting += 1
                        waiting = True
                    threads = self._launch(
                        max(0, self._waiting - len(self._starting))
                    )
                    seen = len(self._members) + len(self._failures)
                for thread in threads:
                    thread.start()
                with self._lock:
                    self._settled.wait_for(
                        lambda: (
                            self._closing
                            or len(self._members) + len(self._failures) != seen
                        ),
                        30.0,
                    )
        finally:
            if waiting:
                with self._lock:
                    self._waiting -= 1

    def build_host(self, *args: object, **kwargs: object) -> "_RemoteRasterPlotHost":
        member = self._claim()
        try:
            return member.build_host(*args, **kwargs)
        finally:
            with self._lock:
                self._claimed.discard(member)
            # Whether or not that Host was built, the warm count may now be
            # short by one; topping up here is what keeps the next panel from
            # waiting.
            self._keep_warm()

    # ------------------------------------------------------------- closing
    def retain(self) -> None:
        """Add one application-window owner without starting anything."""

        with self._lock:
            if self._closing:
                raise RuntimeError("render pool is closing")
            self._owners += 1

    def _stop(self, timeout: float) -> tuple["RenderProcess", ...]:
        """Close the pool to new work, waiting only if the caller may wait.

        A window's close must return within a Qt turn -- the console asserts
        fifty milliseconds -- and a child takes 2.3 s to start, so a close
        that joined the starting threads unconditionally would hold the GUI
        for seconds whenever the operator shut a console while one was still
        coming up.  With no time to spend, this only shuts the door: a child
        that finishes starting after it sees ``_closing`` retires itself.
        The waiting close, which the app reaches after its Qt turns, is where
        those threads are joined.
        """

        with self._lock:
            self._closing = True
            self._settled.notify_all()
            starting = tuple(self._starting)
        if timeout:
            for thread in starting:
                thread.join(timeout=timeout)
        with self._lock:
            return tuple(self._members)

    def _quiet(self) -> bool:
        """Whether anything is still in flight.

        A child still starting is not a settled pool, and saying it is would
        tell the caller there is nothing left to shut down while a whole
        renderer is on its way up.  It will retire itself when it arrives,
        but the caller has to be told to come back for it.
        """

        with self._lock:
            return not self._starting

    def release(self, timeout: float = 0.0) -> bool:
        """Release one window owner; the last owner shuts every child down."""

        if timeout < 0.0:
            raise ValueError("timeout must be non-negative")
        with self._lock:
            if self._owners > 0:
                self._owners -= 1
            if self._owners > 0:
                return True
        members = self._stop(timeout)
        # Every child is told to go FIRST and only then waited for: told one
        # at a time with the timeout each, a pool would wait one deadline per
        # child for shutdowns that all began at once.
        settled = [member.release(0.0) for member in members]
        if timeout:
            settled = [
                done or member._await_close(timeout)
                for member, done in zip(members, settled, strict=True)
            ]
        return all(settled) and self._quiet()

    def close(self, timeout: float = 0.0) -> bool:
        if timeout < 0.0:
            raise ValueError("timeout must be non-negative")
        members = self._stop(timeout)
        closed = [member.close(0.0) for member in members]
        if timeout:
            closed = [
                done or member._await_close(timeout)
                for member, done in zip(members, closed, strict=True)
            ]
        return all(closed) and self._quiet()


class RenderProcess:
    """One long-lived process containing any number of RasterPlotHosts."""

    #: How long a child may give no sign of life, with requests outstanding,
    #: before it has stopped.  "The service is up" used to have two proxies
    #: and no owner -- the process existing and its pipe not having closed --
    #: and a child that was alive and not answering satisfied both: every
    #: request stayed pending, every panel it served kept its last picture,
    #: and nothing anywhere said so.  The reader owns service failure, so
    #: the reader owns liveness.  The sign of life is a heartbeat from a
    #: thread of the child's that owes nothing else: a dispatch loop busy
    #: for a minute loading one large dataset is alive, and the request
    #: waiting behind it is slow, not dead.  A process that cannot run one
    #: Python thread for ten seconds -- stopped, or wedged -- is not slow.
    SILENCE_DEADLINE_SECONDS = 10.0

    def __init__(
        self,
        name: str,
        *,
        silence_deadline_seconds: float | None = None,
        host_retired: Callable[[], None] | None = None,
    ) -> None:
        selected = str(name).strip()
        if not selected:
            raise ValueError("render process name must be non-empty")
        deadline = float(
            self.SILENCE_DEADLINE_SECONDS
            if silence_deadline_seconds is None
            else silence_deadline_seconds
        )
        if not deadline > 0.0:
            raise ValueError("silence deadline must be positive")
        self.name = selected
        self._silence_deadline = deadline
        #: Told when this child finishes with a Host, so an owner that keeps
        #: several children warm learns that one of them is free again.
        #: Called on the reader thread and OUTSIDE this child's lock -- the
        #: listener is a pool, and a pool asks a child things under its own
        #: lock, so calling it under this one would close that cycle.
        self._host_retired = host_retired
        self._last_alive: float | None = None
        self._lock = RLock()
        self._pending: dict[int, _Pending] = {}
        self._callbacks: dict[int, Callable[..., object]] = {}
        self._subscription_hosts: dict[int, str] = {}
        self._hosts: dict[str, _RemoteRasterPlotHost] = {}
        self._host_closed: dict[str, Event] = {}
        self._request_serial = 0
        self._subscription_serial = 0
        self._input_serial = 0
        self._input_tokens: dict[object, int] = {}
        self._input_keys: dict[int, object] = {}
        # Overlay has no publication ref and is keyed by process-local id;
        # keep its identity owner alive with the token so Python cannot reuse
        # that id for different content before the token is retired.
        self._input_identity_owners: dict[int, object] = {}
        self._input_kinds: dict[int, str] = {}
        self._input_refcounts: dict[int, int] = {}
        self._host_inputs: dict[str, set[int]] = {}
        self._input_uploads: dict[int, tuple[SharedMemory, ...]] = {}
        #: Segments a child has finished reading, kept to be filled again.
        #: A producer publishes the same shape shot after shot, so the block
        #: its bytes are copied into is the same size every time -- and
        #: making one costs the first-touch zero fill of the whole frame:
        #: measured, 4.79 ms to create, fill and destroy eight megabytes
        #: against 0.22 ms to fill one that already exists, and 0.74 against
        #: 0.014 at one megabyte.  Per shot, per child.
        #:
        #: ONE deque searched by size, never a bucket per size: bucketed,
        #: every raster size a session ever used kept a permanent cache of
        #: its own, which is the shape the front pool was already taught not
        #: to have.
        self._input_free: deque[SharedMemory] = deque()
        self._closing = False
        self._owners = 1
        self._close_started: float | None = None
        self._closed = True
        #: The last interaction map each Host was sent, so a front that
        #: repeats one crosses the pipe as None.  Cleared with the child
        #: that filled it: a restart begins the agreement again.
        self._front_interaction: dict[str, RasterInteractionMap] = {}
        self._mapping_retirements: Queue = Queue()
        self._mappings = _SharedMappingCache(self._mapping_retirements)
        self._mapping_retirement_thread = Thread(
            target=_retire_shared_mappings,
            args=(self._mapping_retirements,),
            name=f"zlc-render-{selected}-mapping-retirement",
            daemon=True,
        )
        self._mapping_retirement_thread.start()
        # Bound to the CACHE, which holds no reference back: bound to this
        # process, the finalizer would keep it alive and never run.
        self._mapping_retirement_finalizer = weakref.finalize(
            self,
            self._mappings.retire_all,
        )
        self._spawn_child()

    def _spawn_child(self) -> None:
        """Start one fresh child after the previous reader fully retired."""

        self._front_interaction.clear()
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        stopped = Event()
        process = context.Process(
            target=_render_process_main,
            args=(child, self.name),
            name=f"{_CHILD_NAME_PREFIX}{self.name}",
            # A renderer must never outlive the process it draws for.  Held
            # non-daemonic, multiprocessing's own exit hook JOINS it, so any
            # exit that skips close() -- an exception on the way out, a
            # crash -- hung forever in atexit with the pixels already gone
            # and nothing left to draw.  Daemonic, the same hook terminates
            # it.  An orderly close still shuts its save worker down and
            # waits for it; what this flag decides is only what happens
            # when nobody closed anything.
            daemon=True,
        )
        try:
            process.start()
        except BaseException:
            parent.close()
            child.close()
            raise
        child.close()
        self._connection = parent
        self._process = process
        self._pid = process.pid
        self._reader_stopped = stopped
        self._closed = False
        self._close_started = None
        self._last_alive = None
        self._outbox: Queue = Queue()
        self._writer = Thread(
            target=_write_messages,
            args=(parent, self._outbox, self._mark_write_failed),
            name=f"zlc-render-{self.name}-requests",
            daemon=True,
        )
        self._writer.start()
        self._reader = Thread(
            target=self._read_messages,
            name=f"zlc-render-{self.name}-responses",
            daemon=True,
        )
        self._reader.start()

    def _ensure_running(self) -> None:
        """Restart a crashed service before constructing its replacement Host."""

        with self._lock:
            if self._closing:
                raise RuntimeError("render process is closing")
            if not self._closed and self._process.is_alive():
                return
            if not self._reader_stopped.is_set():
                raise RuntimeError("render process failure is still settling")
            self._spawn_child()

    @property
    def pid(self) -> int | None:
        return self._pid

    @property
    def alive(self) -> bool:
        return bool(self._process.is_alive() and not self._closed)

    @property
    def host_count(self) -> int:
        """How many Hosts this child is currently drawing for."""

        with self._lock:
            return len(self._hosts)

    def retain(self) -> None:
        """Add one application-window owner without creating another process."""

        self._ensure_running()
        with self._lock:
            if self._closing:
                raise RuntimeError("render process is closing")
            self._owners += 1

    def release(self, timeout: float = 0.0) -> bool:
        """Release one window owner; the last owner shuts the process down."""

        if timeout < 0.0:
            raise ValueError("timeout must be non-negative")
        with self._lock:
            if self._owners > 0:
                self._owners -= 1
            remaining = self._owners
            if remaining > 0:
                return True
            if self._closed and not self._process.is_alive():
                return True
            first = not self._closing
            self._closing = True
            if first:
                self._close_started = monotonic()
        if first:
            try:
                self._send(("shutdown",))
            except Exception:
                pass
        return self._await_close(timeout)

    def build_host(
        self,
        plot_input: object,
        spec: object,
        *,
        size: str | None = None,
        parameters: Mapping[str, object] | None = None,
        initial_configuration: Mapping[str, object] | None = None,
        device_pixel_ratio: float = 1.0,
    ) -> _RemoteRasterPlotHost:
        from .config import DEFAULTS  # noqa: PLC0415

        self._ensure_running()
        with self._lock:
            if (
                self._closing
                or self._closed
                or not self._process.is_alive()
            ):
                raise RuntimeError("render process is not running")
            host_id = uuid4().hex
            # The same module constant this process and the child both
            # hold.  It used to ride on EVERY front and be unpacked into two
            # fresh namespaces per front per panel -- a process constant
            # re-sent forty times a second on a four-card board to say what
            # it said when the host was built.
            host = _RemoteRasterPlotHost(self, host_id, DEFAULTS)
            self._hosts[host_id] = host
            self._host_closed[host_id] = Event()
        input_tokens: set[int] = set()
        try:
            input_ref = self._replace_inputs(plot_input, input_tokens)
            self._set_host_inputs(host_id, input_tokens)
            pending = self._request(
                "create",
                host_id,
                input_ref,
                spec,
                size,
                None if parameters is None else dict(parameters),
                float(device_pixel_ratio),
                _plain(initial_configuration),
                host_id=host_id,
                input_tokens=tuple(input_tokens),
                input_transition="create",
            )
        except BaseException:
            with self._lock:
                self._hosts.pop(host_id, None)
                event = self._host_closed.pop(host_id, None)
                current_tokens = tuple(self._host_inputs.pop(host_id, ()))
            self._release_inputs(current_tokens)
            self._release_inputs(tuple(input_tokens))
            if event is not None:
                event.set()
            raise

        def created(done: Future) -> None:
            try:
                operation = done.result()
                host._created(operation.value)
            except BaseException as error:
                host._failed(error)

        pending.add_done_callback(created)
        return host

    def save_figure_artifact(
        self,
        base_path: str | Path,
        *,
        plot_input: object,
        spec: object,
        parameters: Mapping[str, object],
        size: str,
        viewport: object = None,
        classifier_thresholds: object = (),
        facet_focus: int | None = None,
        fit: Mapping[str, object] | None = None,
        lineage: Mapping[str, object] | None = None,
        selectors: object = (),
        source: Mapping[str, object] | None = None,
        host: object | None = None,
    ) -> Future:
        try:
            self._ensure_running()
        except BaseException as error:
            failed = Future()
            failed.set_exception(error)
            return failed
        host_id = None
        if host is not None:
            if not isinstance(host, _RemoteRasterPlotHost) or host._process is not self:
                raise ValueError("save host belongs to another render process")
            host_id = host.host_id
        input_tokens: set[int] = set()
        try:
            input_ref = self._replace_inputs(plot_input, input_tokens)
        except BaseException:
            # `_replace_inputs` may have successfully retained an earlier
            # member of an ImageFrame before a later allocation/pickle fails.
            # No request owns those provisional holds yet.
            self._release_inputs(tuple(input_tokens))
            raise
        return self._request(
            "save",
            str(Path(base_path)),
            input_ref,
            spec,
            dict(parameters),
            str(size),
            viewport,
            _plain(classifier_thresholds),
            facet_focus,
            None if fit is None else dict(fit),
            None if lineage is None else _plain(lineage),
            tuple(selectors),
            None if source is None else _plain(source),
            host_id,
            input_tokens=tuple(input_tokens),
            raw_result=True,
        )

    def save_front(self, path: str | Path, front: RasterFront) -> Future:
        """Encode one already accepted immutable Front in this process.

        FigureViewer's image-only command must preserve the exact pixels the
        operator selected without asking the live Monitor process to render
        again.  The caller captures the Front atomically; this service owns
        the potentially slow image encoding and file write.
        """

        try:
            self._ensure_running()
        except BaseException as error:
            failed = Future()
            failed.set_exception(error)
            return failed
        if not isinstance(front, RasterFront):
            raise TypeError("front must be RasterFront")
        buffer = front.buffer
        return self._request(
            "save-front",
            str(Path(path)),
            int(buffer.width),
            int(buffer.height),
            bytes(buffer.pixels),
            raw_result=True,
        )

    def _call(
        self,
        host: _RemoteRasterPlotHost,
        method: str,
        args: Sequence[object],
        kwargs: Mapping[str, object],
    ) -> Future:
        if host._process is not self:
            raise ValueError("remote host belongs to another render process")
        if host.closing or not self.alive:
            failed = Future()
            failed.set_exception(RuntimeError("remote raster host is not running"))
            return failed
        input_tokens: set[int] = set()
        try:
            encoded_args = self._replace_inputs(tuple(args), input_tokens)
            encoded_kwargs = self._replace_inputs(dict(kwargs), input_tokens)
        except BaseException:
            self._release_inputs(tuple(input_tokens))
            raise
        transition = ""
        if method in {"update_data", "update_image_frame"}:
            transition = "replace"
        elif method == "update_image_overlay":
            transition = "overlay"
        elif method == "configure":
            transition = (
                "replace" if "data" in kwargs else "overlay"
                if "image_overlay" in kwargs else ""
            )
        pending = self._request(
            "call",
            host.host_id,
            str(method),
            encoded_args,
            encoded_kwargs,
            host_id=host.host_id,
            input_tokens=tuple(input_tokens),
            input_transition=transition,
        )
        if method == "configure":
            def remember_configuration(done: Future) -> None:
                try:
                    host._created(done.result().value)
                except BaseException:
                    return

            pending.add_done_callback(remember_configuration)
        return pending

    def _subscribe(
        self,
        host: _RemoteRasterPlotHost,
        channel: str,
        callback: Callable[..., object],
        *,
        replay_current: bool = False,
    ) -> Future:
        if not callable(callback):
            raise TypeError("event callback must be callable")
        with self._lock:
            if self._closing or self._closed or not self._process.is_alive():
                failed = Future()
                failed.set_exception(RuntimeError("render process is not running"))
                return failed
            self._subscription_serial += 1
            subscription_id = self._subscription_serial
            self._callbacks[subscription_id] = callback
            self._subscription_hosts[subscription_id] = host.host_id
        pending = self._request(
            "subscribe",
            host.host_id,
            subscription_id,
            str(channel),
            bool(replay_current),
            host_id=host.host_id,
            subscription_id=subscription_id,
        )

        def discard_rejected(done: Future) -> None:
            try:
                failed = done.cancelled() or done.exception() is not None
            except BaseException:
                failed = True
            if not failed:
                return
            with self._lock:
                if self._subscription_hosts.get(subscription_id) == host.host_id:
                    self._subscription_hosts.pop(subscription_id, None)
                    self._callbacks.pop(subscription_id, None)

        pending.add_done_callback(discard_rejected)
        return pending

    def _unsubscribe(self, subscription_id: int) -> Future:
        with self._lock:
            self._callbacks.pop(int(subscription_id), None)
            host_id = self._subscription_hosts.pop(int(subscription_id), None)
        return self._request(
            "unsubscribe", int(subscription_id), host_id, host_id=host_id
        )

    def _close_host(self, host: _RemoteRasterPlotHost) -> None:
        try:
            self._send(("close-host", host.host_id))
        except Exception as error:
            # The request never went out, so no acknowledgement is owed and
            # none is coming: for this host that IS the end.  Recording only a
            # failure left it un-closable, waiting for an answer nobody was
            # going to send, and the console cannot finish closing until every
            # retired host has answered.
            host._failed(error)
            host._mark_closed()

    def _wait_host_closed(self, host_id: str, timeout: float) -> bool:
        with self._lock:
            event = self._host_closed.get(host_id)
        return True if event is None else event.wait(timeout)

    def _request(
        self,
        action: str,
        *payload: object,
        host_id: str | None = None,
        subscription_id: int | None = None,
        input_tokens: tuple[int, ...] = (),
        input_transition: str = "",
        raw_result: bool = False,
    ) -> Future:
        completion = Future()
        # `_input_reference` already placed one provisional hold on every
        # token in this request, atomically with lookup/registration.  This
        # pending record takes over those holds and releases them on every
        # terminal path; delaying the hold until here lets another Host drop
        # the token from the child in the gap between lookup and submission.
        rejected = False
        with self._lock:
            if self._closing and action not in {"unsubscribe"}:
                rejected = True
                request_id = -1
            else:
                self._request_serial += 1
                request_id = self._request_serial
                self._pending[request_id] = _Pending(
                    completion,
                    host_id=host_id,
                    subscription_id=subscription_id,
                    input_tokens=input_tokens,
                    input_transition=input_transition,
                    raw_result=raw_result,
                )
        if rejected:
            self._release_inputs(input_tokens)
            completion.set_exception(RuntimeError("render process is closing"))
            return completion
        def cancelled(done: Future) -> None:
            if done.cancelled():
                try:
                    self._send(("cancel", request_id))
                except Exception:
                    pass

        completion.add_done_callback(cancelled)
        try:
            self._send(("request", request_id, action, *payload))
        except BaseException as error:
            with self._lock:
                failed = self._pending.pop(request_id, None)
            if failed is not None:
                self._settle_pending_inputs(failed, success=False)
            completion.set_exception(error)
        return completion

    def _send(self, message: object) -> None:
        """Hand one message to the writer.  Never touches the pipe.

        A caller on the Qt owner thread must not wait for the child to read:
        see :func:`_write_messages`.  Ordering is the queue's, so an ``input``
        enqueued before the ``request`` that names its token still arrives
        first.
        """

        if self._closed:
            raise RuntimeError("render process is closed")
        self._outbox.put(_encode_message(message))

    def _mark_write_failed(self) -> None:
        """The writer lost the pipe.  The reader's EOF fails the requests."""

        self._closed = True

    @staticmethod
    def _input_key(value: object) -> object:
        from zlc_data import OwnedSnapshot
        from .primitives import ImagePointOverlay

        if isinstance(value, OwnedSnapshot):
            return "snapshot", value.ref
        if isinstance(value, ImagePointOverlay):
            return "overlay", id(value), value.revision
        raise TypeError(f"unsupported plot input {type(value).__name__}")

    def _reuse_input_token(self, key: object, used: set[int]) -> int | None:
        """The token already published for ``key``, counted once per request."""

        token = self._input_tokens.get(key)
        if token is not None and token not in used:
            self._input_refcounts[token] += 1
            used.add(token)
        return token

    def _input_reference(
        self, value: object, used: set[int]
    ) -> tuple[str, int]:
        """The token a request names this input by, uploading it once.

        A token is visible to other callers only once its ``input`` message
        is in the outbox: publication and enqueueing happen under the one
        lock, as one step.  Published first and uploaded after, a concurrent
        Host's request that saw the token was enqueued ahead of the upload
        it named, and the child refused it as an input released before use.
        The serialization and the shared-memory copy -- the expensive part
        -- run outside the lock, on a value nobody else can see yet; a
        second caller that raced to the same input discards its own copy
        and takes the published token.
        """

        key = self._input_key(value)
        with self._lock:
            token = self._reuse_input_token(key, used)
            if token is not None:
                return _INPUT_REF, token
            if self._closing or self._closed:
                raise RuntimeError("render process is closing")
        buffers: list[pickle.PickleBuffer] = []
        released_buffers = 0
        shared: list[SharedMemory] = []
        descriptors: list[tuple[str, int]] = []

        def discard_blocks() -> None:
            # BACK ON THE FREE LIST, not destroyed.  The commonest way here
            # is two panels sharing one signal: the second finds the token
            # already published and drops the copy it just made -- and that
            # copy is a segment of exactly the size the next shot wants.
            with self._lock:
                closing = self._closing or self._closed
                spare = (
                    tuple(shared)
                    if closing
                    else tuple(
                        block for block in shared
                        if not self._keep_input_block(block)
                    )
                )
            self._discard_input_blocks(spare)

        try:
            payload = pickle.dumps(
                value, protocol=5, buffer_callback=buffers.append
            )
            for item in buffers:
                source = None
                destination = None
                try:
                    source = memoryview(item).cast("B")
                    nbytes = source.nbytes
                    block = self._take_input_block(nbytes)
                    shared.append(block)
                    # A DERIVED view, because releasing a SharedMemory's own
                    # ``buf`` kills it for good -- which did not matter while
                    # every block was destroyed after one use and is exactly
                    # what a block being filled a second time cannot survive.
                    destination = memoryview(block.buf)
                    if nbytes:
                        destination[:nbytes] = source
                    descriptors.append((block.name, nbytes))
                finally:
                    if destination is not None:
                        destination.release()
                    if source is not None:
                        source.release()
                    # PickleBuffer itself owns an export independently of
                    # the derived memoryview; release both as soon as the
                    # shared transport copy is complete.
                    item.release()
                    released_buffers += 1
            buffers.clear()
        except BaseException:
            for item in buffers[released_buffers:]:
                try:
                    item.release()
                except Exception:
                    pass
            buffers.clear()
            discard_blocks()
            raise
        with self._lock:
            token = self._reuse_input_token(key, used)
            if token is not None:
                discard_blocks()
                return _INPUT_REF, token
            if self._closing or self._closed:
                discard_blocks()
                raise RuntimeError("render process is closing")
            self._input_serial += 1
            token = self._input_serial
            try:
                self._send(("input", token, payload, tuple(descriptors)))
            except BaseException:
                discard_blocks()
                raise
            self._input_tokens[key] = token
            self._input_keys[token] = key
            self._input_identity_owners[token] = value
            self._input_kinds[token] = str(key[0])
            self._input_refcounts[token] = 1
            self._input_uploads[token] = tuple(shared)
            used.add(token)
        return _INPUT_REF, token

    def _replace_inputs(self, value: object, used: set[int]) -> object:
        from zlc_data import OwnedSnapshot
        from .primitives import ImageFrame, ImagePointOverlay

        if isinstance(value, ImageFrame):
            return (
                _IMAGE_FRAME_REF,
                self._input_reference(value.snapshot, used),
                self._input_reference(value.overlay, used),
            )
        if isinstance(value, (OwnedSnapshot, ImagePointOverlay)):
            return self._input_reference(value, used)
        if isinstance(value, tuple):
            return tuple(self._replace_inputs(item, used) for item in value)
        if isinstance(value, list):
            return [self._replace_inputs(item, used) for item in value]
        if isinstance(value, Mapping):
            return {
                key: self._replace_inputs(item, used) for key, item in value.items()
            }
        return value

    def _hold_inputs(self, tokens: Sequence[int]) -> None:
        with self._lock:
            for token in tokens:
                self._input_refcounts[token] = self._input_refcounts.get(token, 0) + 1

    def _release_inputs(self, tokens: Sequence[int]) -> None:
        dropped: list[int] = []
        with self._lock:
            for token in tokens:
                if token not in self._input_refcounts:
                    continue
                count = self._input_refcounts[token] - 1
                if count > 0:
                    self._input_refcounts[token] = count
                    continue
                self._input_refcounts.pop(token, None)
                key = self._input_keys.pop(token, None)
                self._input_identity_owners.pop(token, None)
                self._input_kinds.pop(token, None)
                if key is not None and self._input_tokens.get(key) == token:
                    self._input_tokens.pop(key, None)
                dropped.append(token)
        for token in dropped:
            try:
                self._send(("drop-input", token))
            except Exception:
                pass

    def _set_host_inputs(self, host_id: str, tokens: Sequence[int]) -> None:
        selected = set(map(int, tokens))
        with self._lock:
            previous = set(self._host_inputs.get(host_id, ()))
            added = selected - previous
            removed = previous - selected
            self._host_inputs[host_id] = selected
        self._hold_inputs(added)
        self._release_inputs(removed)

    def _settle_pending_inputs(self, pending: _Pending, *, success: bool) -> None:
        tokens = set(pending.input_tokens)
        if success and pending.host_id and pending.input_transition:
            if pending.input_transition == "replace":
                self._set_host_inputs(pending.host_id, tokens)
            elif pending.input_transition == "overlay":
                with self._lock:
                    current = set(self._host_inputs.get(pending.host_id, ()))
                    retained = {
                        token
                        for token in current
                        if self._input_kinds.get(token) != "overlay"
                    }
                self._set_host_inputs(pending.host_id, retained | tokens)
        self._release_inputs(tokens)

    def _require_signs_of_life(self) -> None:
        """Silence with work outstanding is judged against the child's heartbeat.

        Runs on the reader thread whenever the pipe has said nothing for one
        slice.  Every message the child sends is a sign of life, and a
        thread of its own sends one every slice whatever its dispatch loop
        is doing, so a loop busy loading one large dataset is still alive.
        With nothing pending there is nothing to wait for; before the
        child's first word there is nothing to judge, it is still importing.
        A child silent past the deadline with requests outstanding has
        stopped, whatever the process table says, and the reader ends
        exactly as it ends on EOF: every pending request fails, every host
        is marked failed, the child is terminated and the console remounts
        on a fresh one -- said, this time, on the status strip.
        """

        now = monotonic()
        with self._lock:
            if not self._pending or self._last_alive is None:
                return
            silence = now - self._last_alive
            if silence <= self._silence_deadline:
                return
            outstanding = len(self._pending)
        raise TimeoutError(
            f"render process {self.name!r} gave no sign of life for "
            f"{silence:.0f} s with {outstanding} request(s) outstanding"
        )

    def _sign_of_life(self) -> None:
        with self._lock:
            self._last_alive = monotonic()

    def _read_messages(self) -> None:
        failure: BaseException | None = None
        try:
            while True:
                if not self._connection.poll(_POLL_SLICE_SECONDS):
                    self._require_signs_of_life()
                    continue
                message = _receive_message(self._connection)
                self._sign_of_life()
                kind = message[0]
                if kind == "alive":
                    continue
                if kind == "front":
                    self._receive_front(*message[1:])
                elif kind == "result":
                    self._receive_result(*message[1:])
                elif kind == "cancelled":
                    self._receive_cancelled(int(message[1]))
                elif kind == "error":
                    self._receive_error(int(message[1]), message[2])
                elif kind == "event":
                    self._receive_event(int(message[1]), message[2])
                elif kind == "input-ack":
                    self._finish_input_upload(int(message[1]))
                elif kind == "host-closed":
                    self._receive_host_closed(str(message[1]))
                elif kind == "stopped":
                    break
                else:
                    raise RuntimeError(f"unknown render response {kind!r}")
        except (EOFError, OSError) as error:
            failure = error
        except BaseException as error:
            failure = error
        finally:
            self._finish_reader(failure)

    def _receive_front(
        self,
        host_id: str,
        identity: RasterIdentity,
        logical_size: tuple[int, int],
        logical_dpi: float,
        device_pixel_ratio: float,
        interaction: RasterInteractionMap | None,
        lease_id: str,
        shared_name: str,
        nbytes: int,
        width: int,
        height: int,
    ) -> None:
        # None means "the same map you already have": a panel whose limits
        # are not moving repeats it frame after frame, and on a 64-cell grid
        # that is 128 transforms through pickle to say nothing changed.  The
        # child only omits what it has already sent on this connection, and a
        # restarted child sends a full one first.
        #
        # A missing cache therefore means one of two different things, and
        # they must not be treated alike.  For a host this side has already
        # retired it means nothing at all -- the front is dropped below with
        # every other front that crossed the pipe too late.  For a LIVE host
        # it is a protocol violation, and the loud failure is the point: the
        # quiet alternative is a panel keeping its last picture for ever with
        # nothing anywhere saying why.
        with self._lock:
            known = str(host_id) in self._hosts
        if interaction is None:
            interaction = self._front_interaction.get(str(host_id))
            if interaction is None and known:
                raise RuntimeError(
                    "a front repeated an interaction map that was never sent"
                )
        else:
            self._front_interaction[str(host_id)] = interaction
        # Mapped once per SEGMENT, not once per front: see
        # :class:`_SharedMappingCache`.  The retirement lane learns of a new
        # mapping before a Front can escape the reader thread, because a
        # caller may retain only an ndarray view after the RenderProcess is
        # gone, and STOP waits for that last view.
        shared = self._mappings.open(str(shared_name))
        # ctypes exposes the standard buffer protocol on every supported
        # Python (including 3.11), while still giving the finalizer a weak-
        # referenceable owner that all memoryview/NumPy/QImage consumers keep.
        pixels = None
        try:
            store = (ctypes.c_ubyte * int(nbytes)).from_buffer(shared)
            pixels = memoryview(store).cast("B").toreadonly()
            finalizer = weakref.finalize(
                store,
                _release_shared_store,
                self,
                str(lease_id),
                self._mappings,
                str(shared_name),
            )
        except BaseException:
            if pixels is not None:
                pixels.release()
            self._mappings.release(str(shared_name))
            raise
        finalizer.atexit = False
        front = (
            None
            if interaction is None
            else RasterFront(
                identity=identity,
                buffer=RasterBuffer(width, height, pixels),
                logical_size=logical_size,
                logical_dpi=logical_dpi,
                device_pixel_ratio=device_pixel_ratio,
                interaction=interaction,
            )
        )
        with self._lock:
            host = self._hosts.get(str(host_id))
        if host is None or front is None:
            # The host was retired while this front crossed the pipe.  Let the
            # buffer exporter die only after both local views are gone; calling
            # its finalizer here would close SharedMemory under live exports.
            del front
            pixels.release()
            del pixels, store
            return
        host._accept_front(front)

    def _receive_result(
        self, request_id: int, wire_value: object, front_sequence: int | None
    ) -> None:
        with self._lock:
            pending = self._pending.pop(int(request_id), None)
            host = None if pending is None else self._hosts.get(pending.host_id or "")
        if pending is None:
            return
        self._settle_pending_inputs(pending, success=True)
        if pending.future.cancelled():
            return
        try:
            value = _unwire_value(wire_value)
            if pending.raw_result:
                if (
                    isinstance(value, tuple)
                    and len(value) == 3
                    and value[0] == "save-paths"
                ):
                    value = (Path(value[1]), Path(value[2]))
                elif (
                    isinstance(value, tuple)
                    and len(value) == 2
                    and value[0] == "save-path"
                ):
                    value = Path(value[1])
                pending.future.set_result(value)
                return
            if pending.subscription_id is not None:
                subscription_id = pending.subscription_id

                def release() -> Future:
                    return self._unsubscribe(subscription_id)

                value = release
            front = None if host is None else host.front
            if front is None or (
                front_sequence is not None
                and front.identity.sequence != int(front_sequence)
            ):
                raise RuntimeError("render result arrived without its exact front")
            pending.future.set_result(RasterOperation(value, front))
        except BaseException as error:
            pending.future.set_exception(error)

    def _receive_cancelled(self, request_id: int) -> None:
        with self._lock:
            pending = self._pending.pop(request_id, None)
            if pending is not None and pending.subscription_id is not None:
                self._callbacks.pop(pending.subscription_id, None)
                self._subscription_hosts.pop(pending.subscription_id, None)
        if pending is not None:
            self._settle_pending_inputs(pending, success=False)
            pending.future.cancel()

    def _receive_error(self, request_id: int, error: object) -> None:
        with self._lock:
            pending = self._pending.pop(request_id, None)
            if pending is not None and pending.subscription_id is not None:
                self._callbacks.pop(pending.subscription_id, None)
                self._subscription_hosts.pop(pending.subscription_id, None)
        if pending is None:
            return
        self._settle_pending_inputs(pending, success=False)
        if pending.input_transition == "create" and pending.host_id:
            with self._lock:
                tokens = tuple(self._host_inputs.pop(pending.host_id, ()))
            self._release_inputs(tokens)
        if not pending.future.done():
            failure = error if isinstance(error, BaseException) else RuntimeError(str(error))
            pending.future.set_exception(failure)

    def _receive_event(self, subscription_id: int, wire_payload: object) -> None:
        with self._lock:
            callback = self._callbacks.get(subscription_id)
        if callback is None:
            return
        payload = _unwire_value(wire_payload)
        try:
            if isinstance(payload, tuple) and payload[:1] == ("event-args",):
                callback(*payload[1])
            else:
                callback(payload)
        except Exception:
            return

    def _take_input_block(self, nbytes: int) -> SharedMemory:
        """One segment big enough to hold this buffer, reused if one is free."""

        wanted = max(1, int(nbytes))
        with self._lock:
            for index, block in enumerate(self._input_free):
                if block.size >= wanted:
                    del self._input_free[index]
                    return block
        return SharedMemory(create=True, size=wanted)

    def _discard_input_blocks(self, blocks: Sequence[SharedMemory]) -> None:
        for block in blocks:
            try:
                block.close()
                block.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass

    def _finish_input_upload(self, token: int) -> None:
        """The child is done with this input; keep its segments to fill again.

        Only reached once the child has acknowledged the drop, so nothing is
        still reading what goes back on the free list.
        """

        with self._lock:
            blocks = self._input_uploads.pop(token, ())
            if self._closing or self._closed:
                spare: tuple[SharedMemory, ...] = tuple(blocks)
            else:
                spare = tuple(
                    block for block in blocks
                    if not self._keep_input_block(block)
                )
        self._discard_input_blocks(spare)

    def _keep_input_block(self, block: SharedMemory) -> bool:
        """Put one segment back, up to the budget.  Called under the lock."""

        if len(self._input_free) >= _INPUT_FREE_BLOCKS:
            return False
        self._input_free.append(block)
        return True

    def _receive_host_closed(self, host_id: str) -> None:
        with self._lock:
            host = self._hosts.pop(host_id, None)
            self._front_interaction.pop(str(host_id), None)
            event = self._host_closed.pop(host_id, None)
            tokens = tuple(self._host_inputs.pop(host_id, ()))
            subscription_ids = tuple(
                subscription_id
                for subscription_id, owner in self._subscription_hosts.items()
                if owner == host_id
            )
            for subscription_id in subscription_ids:
                self._subscription_hosts.pop(subscription_id, None)
                self._callbacks.pop(subscription_id, None)
        self._release_inputs(tokens)
        if host is not None:
            host._mark_closed()
        if event is not None:
            event.set()
        retired = self._host_retired
        if retired is not None:
            try:
                retired()
            except BaseException:  # noqa: BLE001 -- an owner's bookkeeping
                traceback.print_exc()

    def _release_front(self, lease_id: str) -> None:
        try:
            self._send(("release-front", str(lease_id)))
        except Exception:
            pass

    def _finish_reader(self, failure: BaseException | None) -> None:
        unexpected_stop = failure is not None or not self._closing
        message = RuntimeError(
            f"render process {self.name!r} stopped"
            + ("" if failure is None else f": {failure}")
        )
        with self._lock:
            pending = tuple(self._pending.values())
            self._pending.clear()
            hosts = tuple(self._hosts.values())
            events = tuple(self._host_closed.values())
            host_tokens = tuple(
                token for tokens in self._host_inputs.values() for token in tokens
            )
            self._host_inputs.clear()
            self._hosts.clear()
            self._host_closed.clear()
            self._callbacks.clear()
            self._subscription_hosts.clear()
        for item in pending:
            self._settle_pending_inputs(item, success=False)
            if not item.future.done():
                item.future.set_exception(message)
        self._release_inputs(host_tokens)
        for host in hosts:
            if unexpected_stop:
                host._failed(message, service_failure=True)
            host._mark_closed()
        for event in events:
            event.set()
        for token in tuple(self._input_uploads):
            self._finish_input_upload(token)
        with self._lock:
            free = tuple(self._input_free)
            self._input_free.clear()
        self._discard_input_blocks(free)
        try:
            self._connection.close()
        except Exception:
            pass
        if isinstance(failure, TimeoutError) and self._process.is_alive():
            # A child that did not answer is not going to leave on its own.
            self._process.terminate()
        self._process.join(timeout=5.0)
        if self._process.is_alive() and failure is not None:
            self._process.terminate()
            self._process.join(timeout=5.0)
        with self._lock:
            self._closed = not self._process.is_alive()
        # This generation's writer retires with its pipe, HERE, before a
        # restart replaces the outbox and writer handles: a writer left
        # blocked on the old queue could never be reached again, and it
        # outlived the service it wrote for.  Marking the service closed
        # first (above) is what stops anything enqueueing behind the STOP.
        self._outbox.put(_STOP_WRITER)
        self._writer.join(timeout=5.0)
        self._reader_stopped.set()

    def close(self, timeout: float = 0.0) -> bool:
        if timeout < 0.0:
            raise ValueError("timeout must be non-negative")
        with self._lock:
            if self._closed and not self._process.is_alive():
                return True
            first = not self._closing
            self._closing = True
            if first:
                self._close_started = monotonic()
        if first:
            try:
                self._send(("shutdown",))
            except Exception:
                pass
        return self._await_close(timeout)

    def _await_close(self, timeout: float) -> bool:
        """Finish an already-started close without changing owner state."""

        if timeout:
            self._reader_stopped.wait(float(timeout))
        started = self._close_started
        if (
            self._process.is_alive()
            and started is not None
            and monotonic() - started >= 30.0
        ):
            self._process.terminate()
            self._process.join(timeout=5.0)
        if self._reader_stopped.is_set() and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=max(0.0, float(timeout)))
        if not self._reader_stopped.is_set():
            return False
        # The child is gone; retire its writer so the thread does not outlive
        # the pipe it was created for.  A restart makes a fresh pair.  Mark the
        # service closed FIRST: past this point nothing may enqueue, because
        # nothing will drain, and a message dropped into a dead queue is a
        # request that never fails and never completes.
        self._closed = True
        self._outbox.put(_STOP_WRITER)
        self._process.join(timeout=0.0)
        return not self._process.is_alive()


@dataclass(slots=True)
class _SharedBlock:
    """One shared segment, and the two hands that have to let go of it."""

    memory: SharedMemory
    nbytes: int
    lease_id: str = ""
    store_id: int = 0
    #: The renderer's own read-only view of this block still exists.
    child_holds: bool = False
    #: The frontend was sent this lease and has not released it.
    frontend_holds: bool = False


#: How many blocks one live Host keeps moving: one being written, one on
#: screen, one in flight.  This used to be two pools' business -- the
#: renderer's private buffers were three deep and the shared segments one
#: spare per Host -- and merging them without merging the depths left a
#: single-panel window allocating a fresh segment most frames, which is the
#: six milliseconds of page faults the pooling exists to avoid.
FRONT_DEPTH = 3


class _SharedFrontPool:
    """Blocks that ARE the frontend's memory, freed when both hands let go.

    The renderer used to write a front into private memory and this pool
    copied it into a shared segment: eighteen megabytes written twice, per
    frame, per panel, on the worker that has to keep up with a live camera.
    The two are now one block -- the front is written once, where it will be
    read -- and this pool hands that block out through the same
    ``take(nbytes) -> (writable, published)`` the private pool offers.

    WHO HOLDS A BLOCK is stated here rather than left to the interpreter,
    because two processes hold it and only one of them has an interpreter
    that could say.  A block is free when BOTH have let go:

    * the child lets go when the renderer's read-only view of it dies -- the
      same weakref release the private pool uses, and the reason a retained
      Front cannot have its pixels rewritten underneath it;
    * the frontend lets go when it releases the lease it was sent.

    Neither is trusted to be the last: whichever arrives second returns the
    block.  A block nobody frees is one this pool never sees again -- the
    next ``take`` allocates.  The cost of every mistake here is another
    segment, never somebody else's pixels.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._leased: dict[str, _SharedBlock] = {}
        self._by_store: dict[int, _SharedBlock] = {}
        # ONE list of free blocks across all sizes, searched by size.  A list
        # per size would let every raster size a session ever used keep a
        # permanent spare; a budget is a budget for the pool, not for each
        # shape a panel has passed through.
        self._free: deque[_SharedBlock] = deque()

    # -------------------------------------------------------------- writing
    def take(self, nbytes: int) -> tuple[object, object]:
        """One block to fill: ``(writable, published)``, in shared memory."""

        size = int(nbytes)
        with self._lock:
            block = next(
                (candidate for candidate in self._free if candidate.nbytes == size),
                None,
            )
            if block is not None:
                self._free.remove(block)
        if block is None:
            # Outside the lock: mapping a segment is the expensive half, and
            # four renderer threads queueing behind one allocation is the
            # contention this whole change exists to remove.
            block = _SharedBlock(SharedMemory(create=True, size=size), size)
        # ctypes keeps the owner identity through memoryview/NumPy derivations,
        # so everything made from the published view keeps the block alive and
        # the finalizer fires exactly when the last reader in this process is
        # gone -- never while a retained Front can still be read.
        store = (ctypes.c_ubyte * size).from_buffer(block.memory.buf)
        published = memoryview(store).toreadonly()
        # A Front from a crashed generation may outlive a restarted service.
        # A process-local integer would collide with the new pool and release
        # an unrelated live Front, so every lease is named once, globally.
        block.lease_id = uuid4().hex
        block.store_id = id(store)
        block.child_holds = True
        block.frontend_holds = False
        with self._lock:
            self._leased[block.lease_id] = block
            self._by_store[block.store_id] = block
        # The callback holds the LEASE NAME, never the store: holding the
        # store would keep it alive and the finalizer would never fire.
        returner = weakref.finalize(
            store, self._child_released, block.lease_id, block.store_id
        )
        returner.atexit = False
        return memoryview(block.memory.buf)[:size], published

    def claim(self, pixels: object) -> tuple[str, str, int] | None:
        """This buffer's lease if this pool made it -- and no copy either way.

        A buffer this pool did not make has no lease to give: a Front built
        elsewhere, or one whose session published before the pool existed.
        The caller copies instead, which is what every front used to do.
        """

        owner = getattr(pixels, "obj", None)
        if owner is None:
            return None
        with self._lock:
            block = self._by_store.get(id(owner))
            if block is None or not block.child_holds:
                return None
            block.frontend_holds = True
            return block.lease_id, block.memory.name, block.nbytes

    def publish(self, pixels: object) -> tuple[str, str, int]:
        """Copy a buffer this pool did not make into one it did."""

        source = memoryview(pixels).cast("B")
        writable, published = self.take(source.nbytes)
        writable[:] = source
        del writable
        handoff = self.claim(published)
        assert handoff is not None, "a block this pool just made is its own"
        # Nothing in this process keeps the copy.  Dropping the view here
        # runs the block's own child release, so the frontend's lease is
        # immediately the only hand on it -- no separate path to get wrong.
        del published
        return handoff

    # ------------------------------------------------------------ releasing
    def _child_released(self, lease_id: str, store_id: int) -> None:
        with self._lock:
            block = self._leased.get(str(lease_id))
            # A block already recycled and re-taken carries a NEW store id.
            # Without this the late finalizer of the previous tenant would
            # free a block the renderer is filling right now -- and would
            # unregister an id CPython has since handed to a live store,
            # which costs the next front a copy for no reason.
            if block is None or block.store_id != store_id:
                return
            if self._by_store.get(store_id) is block:
                self._by_store.pop(store_id, None)
            block.child_holds = False
            self._recycle_locked(block)

    def release(self, lease_id: str, free_budget: int) -> None:
        with self._lock:
            block = self._leased.get(str(lease_id))
            if block is None:
                return
            block.frontend_holds = False
            self._recycle_locked(block)
        self.trim_free(free_budget)

    def _recycle_locked(self, block: _SharedBlock) -> None:
        if block.child_holds or block.frontend_holds:
            return
        self._leased.pop(block.lease_id, None)
        # BY IDENTITY.  CPython reuses ctypes addresses aggressively, so a
        # recycled block's stale store id is very likely the id of a store a
        # LIVE block now owns: popped blindly, that block's claim entry goes
        # with it, its next front finds no lease and is copied -- the exact
        # copy this pool exists to remove, disappearing at random intervals
        # with correct pixels and nothing to show for it.
        if self._by_store.get(block.store_id) is block:
            self._by_store.pop(block.store_id, None)
        block.lease_id = ""
        block.store_id = 0
        self._free.append(block)

    def trim_free(self, free_budget: int) -> None:
        retired = []
        with self._lock:
            while len(self._free) > int(free_budget):
                retired.append(self._free.popleft())
        for block in retired:
            _discard_shared(block.memory)

    def close(self) -> None:
        with self._lock:
            blocks = tuple(self._leased.values()) + tuple(self._free)
            self._leased.clear()
            self._by_store.clear()
            self._free.clear()
        for block in blocks:
            _discard_shared(block.memory)


def _discard_shared(memory: SharedMemory) -> None:
    """Give a segment back, whether or not a reader is still holding it.

    ``close`` releases this process's own view and raises BufferError while
    anything still exports it -- and a Front this pool has already dropped
    can still be on its way out.  Retrying that close is what
    ``SharedMemory.__del__`` does, so leaving the wrapper intact turns one
    ordinary race into an unraisable BufferError printed from a destructor.

    Detached instead, the mapping simply outlives its wrapper and
    deallocates when the last export goes, which is the lifetime that was
    wanted all along.  The same detachment the frontend does on the way in.
    """

    try:
        memory.close()
    except BufferError:
        memory._buf = None
        memory._mmap = None
    try:
        memory.unlink()
    except (FileNotFoundError, OSError):
        pass


def _owned_input(value: object, schemas: OrderedDict[str, object]) -> object:
    """Move an IPC-backed PlotInput onto ordinary immutable child storage."""

    from copy import deepcopy
    from zlc_data import (
        CellValidity,
        DataBlock,
        DatasetComponentValidity,
        OwnedSnapshot,
    )
    from zlc_data.codec import dataset_schema_from_tree, dataset_schema_to_tree
    from .primitives import ImageFrame, ImagePointOverlay

    if isinstance(value, OwnedSnapshot):
        # DatasetSchema owns several lazy identity/cache sentinels.  Pickle
        # cannot preserve a module singleton's ``is`` identity, so carrying a
        # warmed schema object across spawn can turn `_NOT_INDEXED` into an
        # arbitrary object that consumers mistake for a real history layout.
        # Rebuild through the data owner's canonical grammar: same scientific
        # schema, fresh process-local caches.
        schema_key = str(value.ref.schema_fingerprint)
        schema = schemas.get(schema_key)
        if schema is None:
            schema = dataset_schema_from_tree(
                dataset_schema_to_tree(value.block.schema)
            )
            schemas[schema_key] = schema
            while len(schemas) > 32:
                schemas.popitem(last=False)
        else:
            schemas.move_to_end(schema_key)
        validity = value.block.validity
        if isinstance(validity, CellValidity):
            validity = CellValidity(np.asarray(validity.mask))
        elif isinstance(validity, DatasetComponentValidity):
            validity = DatasetComponentValidity(
                tuple(validity.axis_ids), np.asarray(validity.mask)
            )
        block = DataBlock(
            value.block.block_id,
            value.block.revision,
            np.asarray(value.block.values),
            validity,
            schema,
            (
                None
                if value.block.sigma is None
                else np.asarray(value.block.sigma)
            ),
            # The producer's own statement of which shots this window
            # holds: immutable metadata, and the admission ticket for the
            # incremental integer-history frequency path.
            value.block.window,
        )
        return OwnedSnapshot(value.ref, block)
    if isinstance(value, ImagePointOverlay):
        return ImagePointOverlay(
            value.revision,
            np.asarray(value.coordinates),
            point_ids=value.point_ids,
            labels=value.labels,
            static_statuses=value.static_statuses,
            status=(
                None
                if value.status is None
                else _owned_input(value.status, schemas)
            ),
        )
    if isinstance(value, ImageFrame):
        return ImageFrame(
            _owned_input(value.snapshot, schemas),
            _owned_input(value.overlay, schemas),
        )
    return deepcopy(value)


def _load_input(
    payload: bytes,
    descriptors: Sequence[tuple[str, int]],
    schemas: OrderedDict[str, object],
) -> object:
    buffers: list[memoryview] = []
    try:
        for name, nbytes in descriptors:
            block = _open_shared_memory(str(name))
            try:
                exported = block.buf[: int(nbytes)]
                try:
                    # Shared memory is only the transport.  The child takes
                    # one bytes-backed immutable copy before constructing the
                    # scientific value, so no OwnedSnapshot can outlive an
                    # input mapping and SharedMemory.close never races an
                    # exported NumPy pointer.
                    owned = bytes(exported)
                finally:
                    exported.release()
            finally:
                block.close()
            buffers.append(memoryview(owned))
        loaded = pickle.loads(payload, buffers=buffers)
        return _owned_input(loaded, schemas)
    finally:
        # DataBlock either retained the immutable bytes backing or copied an
        # incompatible layout.  These temporary view objects own no OS handle.
        buffers.clear()


def _resolve_inputs(value: object, inputs: Mapping[int, object]) -> object:
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and value[0] == _INPUT_REF
    ):
        try:
            return inputs[int(value[1])]
        except KeyError as error:
            raise RuntimeError("render input was released before use") from error
    if (
        isinstance(value, tuple)
        and len(value) == 3
        and value[0] == _IMAGE_FRAME_REF
    ):
        from .primitives import ImageFrame

        return ImageFrame(
            _resolve_inputs(value[1], inputs),
            _resolve_inputs(value[2], inputs),
        )
    if isinstance(value, tuple):
        return tuple(_resolve_inputs(item, inputs) for item in value)
    if isinstance(value, list):
        return [_resolve_inputs(item, inputs) for item in value]
    if isinstance(value, Mapping):
        return {
            key: _resolve_inputs(item, inputs) for key, item in value.items()
        }
    return value


def _send_error(send: Callable[[object], None], request_id: int, error: BaseException) -> None:
    try:
        pickle.dumps(error, protocol=5)
    except Exception:
        error = RuntimeError(f"{type(error).__name__}: {error}")
    send(("error", int(request_id), error))


def _render_process_main(connection: Connection, name: str) -> None:
    """Child entry: multiplex commands over unchanged local RasterPlotHosts."""

    from .config import DEFAULTS
    from .data_contract import snapshot_schema
    from .raster import RasterPlotHost
    from .session import PlotSession
    from . import _raster_kernels as kernels

    state_lock = RLock()
    hosts: dict[str, RasterPlotHost] = {}
    closing_hosts: set[str] = set()
    front_releases: dict[str, Callable[[], None]] = {}
    last_front_sequence: dict[str, int] = {}
    inputs: dict[int, object] = {}
    schemas: OrderedDict[str, object] = OrderedDict()
    pending: dict[int, Future] = {}
    subscriptions: dict[int, tuple[str, Callable[[], object]]] = {}
    fronts = _SharedFrontPool()
    # Imported here, not at module scope: the parent imports this module to
    # talk to the child, and it must not pull Matplotlib in to do it.
    from .rendering import install_publish_pool

    # From now on every renderer built in this process writes its fronts
    # straight into the shared segments, and publishing is a handover.
    install_publish_pool(fronts)
    save_worker = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix=f"zlc-render-{name}-save",
        initializer=kernels.configure_worker_threads,
    )
    closer_threads: set[Thread] = set()

    outbox: Queue = Queue()
    write_failed = Event()
    writer = Thread(
        target=_write_messages,
        args=(connection, outbox, write_failed.set),
        name=f"zlc-render-{name}-out",
        daemon=True,
    )
    writer.start()
    stopping = Event()

    def send(message: object) -> None:
        """Hand one message to the writer, so this loop keeps reading.

        The service loop's own answers -- an input acknowledgement, a refusal
        -- used to wait behind whatever front was mid-write, which stopped it
        draining the parent's requests.  See :func:`_write_messages`.
        Encoded here, on the sender's thread: see :func:`_encode_message`.
        """

        outbox.put(_encode_message(message))

    def reply(request_id: int, message: object) -> None:
        """One request's answer -- or the error that kept it from crossing."""

        try:
            send(message)
        except Exception as error:
            _send_error(send, request_id, error)

    def heartbeat() -> None:
        """A sign of life every slice, from a thread that owes nothing else.

        The parent judges a silent child by this, never by its dispatch
        loop: the loop is legitimately busy for as long as one input takes
        to load or one host takes to build, and a request waiting behind
        that is slow, not dead.
        """

        while not stopping.wait(_POLL_SLICE_SECONDS):
            send(("alive",))

    Thread(target=heartbeat, name=f"zlc-render-{name}-alive", daemon=True).start()

    #: Set by the first create request: from then on the warming below
    #: stops before its next picture, so a panel never queues behind it.
    requested = Event()
    #: The panel this child was given -- its spec and its storage, appended
    #: by the create that builds it -- and the moment its first front went
    #: out.  What the warming goes on to after that is that panel's own fit,
    #: in the first gap of the parent's requests, whose last arrival the
    #: service loop stamps here.
    panel: list[tuple[object, object]] = []
    shown = Event()
    last_request = [monotonic()]

    def warm() -> None:
        """Pay the process's first-render costs now, before a panel asks.

        On a thread of its own, in two levels.  The first is what every
        panel pays, and runs only until a panel asks: a request shares
        this process's one interpreter with the warming, so a picture
        warmed while a panel waits is a picture the panel waited for.  The
        second is that panel's own fit, once its first front is out and
        the parent has gone quiet -- a spare child never reaches it, which
        is what keeps a spare's memory to what every panel needs.  A
        failure in either costs the panel what it would have cost anyway,
        and is written to the child's stderr, not allowed to end the child.

        Masked to the worker team like every other ZLC worker: a panel
        draws on four threads, so what warms for it warms on four, and a
        spare warming beside a live board takes four cores, not sixteen.
        """

        import gc

        from ._kernel_warm import warm_fit, warm_process

        kernels.configure_worker_threads()
        try:
            warm_process(proceed=lambda: not requested.is_set())
        except Exception:  # noqa: BLE001 -- reported, never fatal
            traceback.print_exc()
        # WHAT THE WARMING BUILT IS PERMANENT, so stop rescanning it.  A
        # warmed child holds about two hundred thousand tracked objects --
        # Matplotlib's font and style tables, numba's typing context, the
        # compiled kernels -- none of which will ever become garbage, and
        # the default thresholds walk all of them whenever enough new
        # objects have been made.  Building a sixty-four cell grid makes
        # tens of thousands, so it walked into a full collection partway
        # through: measured, the same panel's first frame took 488 ms at
        # best, 599 in the middle and 767 at worst.  Frozen, and with the
        # collector asked less often now that a sweep is cheap, the same
        # panel is 477 / 484 / 492 -- the tail is not shortened, it is
        # gone.  Cycles made AFTER this are collected as they always were;
        # what is frozen is what was already going to outlive the child.
        gc.collect()
        gc.freeze()
        gc.set_threshold(20000, 50, 50)
        # THE PANEL'S OWN FIT, IN THE FIRST QUIET MOMENT AFTER ITS FIRST
        # FRONT, ON THIS THREAD.  After, because loading a kernel holds
        # numba's compiler lock and a parallel entry's first dispatch
        # saturates the machine: beside the first frame it was paid by the
        # first frame.  On this thread, because the 170-200 ms it takes is
        # numba reading the family's kernels off the disk cache and the
        # kernels are the process's wherever they were read: queued on the
        # panel's worker instead, a frame that landed during it waited the
        # whole 200-250 ms.  In a quiet moment, because here it shares the
        # interpreter with a frame it overlaps -- 5-45 ms on that frame,
        # measured -- and the gap after a panel's first frame is long
        # enough to hold it whole; the deadline is for a producer that
        # never leaves a gap.
        shown.wait()
        started = monotonic()
        while not stopping.is_set():
            quiet = monotonic() - last_request[0]
            if (
                quiet >= _FIT_WARM_QUIET_SECONDS
                or monotonic() - started >= _FIT_WARM_DEADLINE_SECONDS
            ):
                break
            stopping.wait(_FIT_WARM_QUIET_SECONDS - quiet)
        if stopping.is_set():
            return
        spec, storage = panel[0]
        try:
            warm_fit(spec, storage=storage, proceed=lambda: not stopping.is_set())
        except Exception:  # noqa: BLE001 -- reported, never fatal
            traceback.print_exc()
        # As permanent as the pictures' tables: frozen the same way, so the
        # panel's frames never rescan them either.
        gc.collect()
        gc.freeze()

    Thread(target=warm, name=f"zlc-render-{name}-warm", daemon=True).start()

    #: The last interaction map each Host actually sent, so an unchanged one
    #: crosses as ``None``.  Cleared with the Host, because the frontend's
    #: cache is cleared with it too.
    last_interaction: dict[str, object] = {}

    def publish_front(host_id: str, front: RasterFront) -> None:
        sequence = int(front.identity.sequence)
        if sequence <= last_front_sequence.get(host_id, -1):
            return
        last_front_sequence[host_id] = sequence
        handoff = fronts.claim(front.buffer.pixels)
        if handoff is None:
            handoff = fronts.publish(front.buffer.pixels)
        lease_id, shared_name, nbytes = handoff
        # The interaction map is the SAME object frame after frame on a panel
        # whose limits are not moving, and it is the whole non-pixel weight of
        # this message: a 64-cell grid carries 128 transforms through pickle
        # on every frame to say nothing changed.  Send it once and name it.
        interaction = front.interaction
        if interaction == last_interaction.get(host_id):
            interaction = None
        else:
            last_interaction[host_id] = interaction
        send(
            (
                "front",
                host_id,
                front.identity,
                tuple(front.logical_size),
                float(front.logical_dpi),
                float(front.device_pixel_ratio),
                interaction,
                lease_id,
                shared_name,
                nbytes,
                int(front.buffer.width),
                int(front.buffer.height),
            )
        )
        shown.set()

    def complete(request_id: int, completed: Future) -> None:
        pending.pop(request_id, None)
        if completed.cancelled():
            send(("cancelled", request_id))
            return
        try:
            operation = completed.result()
            wire = _wire_value(operation.value)
            sequence = int(operation.front.identity.sequence)
        except BaseException as error:
            _send_error(send, request_id, error)
            return
        reply(request_id, ("result", request_id, wire, sequence))

    def begin(request_id: int, future: Future) -> None:
        pending[request_id] = future
        future.add_done_callback(
            lambda done, request_id=request_id: complete(request_id, done)
        )

    def event(subscription_id: int, *payload: object) -> None:
        if subscription_id not in subscriptions:
            return
        wire = (
            _wire_value(payload[0])
            if len(payload) == 1
            else ("event-args", tuple(_wire_value(item) for item in payload))
        )
        send(("event", subscription_id, wire))

    def start_subscription(
        request_id: int,
        host_id: str,
        subscription_id: int,
        channel: str,
        replay_current: bool,
    ) -> None:
        host = hosts[host_id]
        subscriptions[subscription_id] = (host_id, lambda: None)
        listener = lambda *args: event(subscription_id, *args)
        if channel == "display":
            answer = host.subscribe_display(listener)
        elif channel == "viewport":
            answer = host.subscribe_viewport(listener)
        elif channel == "facet-focus":
            answer = host.subscribe_facet_focus(listener)
        elif channel == "fit":
            answer = host.subscribe_fit(
                listener, replay_current=bool(replay_current)
            )
        elif channel == "selection":
            answer = host.subscribe_selection(listener)
        else:
            raise ValueError(f"unknown render subscription {channel!r}")
        pending[request_id] = answer

        def installed(done: Future) -> None:
            pending.pop(request_id, None)
            if done.cancelled():
                subscriptions.pop(subscription_id, None)
                send(("cancelled", request_id))
                return
            try:
                operation = done.result()
                release = operation.value
                if not callable(release):
                    raise TypeError("render subscription returned no release")
                with state_lock:
                    stale = host_id in closing_hosts or host_id not in hosts
                    if not stale:
                        subscriptions[subscription_id] = (host_id, release)
                if stale:
                    release()
                    raise RuntimeError("subscription host closed before installation")
            except BaseException as error:
                subscriptions.pop(subscription_id, None)
                _send_error(send, request_id, error)
                return
            send(
                (
                    "result",
                    request_id,
                    None,
                    int(operation.front.identity.sequence),
                )
            )

        answer.add_done_callback(installed)

    def unsubscribe(
        request_id: int, subscription_id: int, expected_host_id: str | None
    ) -> None:
        entry = subscriptions.pop(subscription_id, None)
        host_id = expected_host_id if entry is None else entry[0]
        release = None if entry is None else entry[1]
        host = None if host_id is None else hosts.get(str(host_id))
        if release is None:
            front = None if host is None else host.front
            if front is None:
                _send_error(
                    send, request_id, RuntimeError("subscription host is closed")
                )
            else:
                reply(
                    request_id,
                    ("result", request_id, None, int(front.identity.sequence)),
                )
            return
        answer = release()
        if isinstance(answer, Future):
            begin(request_id, answer)
            return
        front = None if host is None else host.front
        if front is None:
            _send_error(send, request_id, RuntimeError("subscription host is closed"))
        else:
            reply(
                request_id,
                ("result", request_id, None, int(front.identity.sequence)),
            )

    def create_host(
        request_id: int,
        host_id: str,
        input_ref: object,
        spec: object,
        size: str | None,
        parameters: Mapping[str, object] | None,
        device_pixel_ratio: float,
        initial_configuration: Mapping[str, object] | None,
    ) -> None:
        requested.set()
        plot_input = _resolve_inputs(input_ref, inputs)

        def factory() -> PlotSession:
            session = PlotSession(
                plot_input,
                spec,
                size=size,
                parameters=parameters,
                defaults=DEFAULTS,
                device_pixel_ratio=device_pixel_ratio,
                initial_configuration=initial_configuration,
            )
            # The one panel this child builds has been built: if it was a
            # grid it took the reserve, and if it was not, nothing ever will.
            from .rendering import CELL_RESERVE  # noqa: PLC0415

            CELL_RESERVE.close()
            return session

        host = RasterPlotHost(factory, host_id=host_id)
        # The warming needs the panel's storage along with its spec: which
        # of a fit's kernels an image takes is decided by its dtype.
        snapshot = getattr(plot_input, "snapshot", plot_input)
        panel.append((spec, snapshot_schema(snapshot).value_schema.dtype))
        with state_lock:
            hosts[host_id] = host
            last_front_sequence[host_id] = -1
            front_releases[host_id] = host.subscribe_front(
                lambda front, host_id=host_id: publish_front(host_id, front)
            )
        current = host.front
        if current is not None:
            publish_front(host_id, current)
        begin(request_id, host.describe_display())

    def close_host(host_id: str) -> None:
        with state_lock:
            host = hosts.get(host_id)
            closing_hosts.add(host_id)
            release = front_releases.pop(host_id, None)
            owned_subscriptions = tuple(
                (subscription_id, subscription_release)
                for subscription_id, (owner, subscription_release)
                in subscriptions.items()
                if owner == host_id
            )
            for subscription_id, _subscription_release in owned_subscriptions:
                subscriptions.pop(subscription_id, None)

        def finish() -> None:
            # Each release owns its own failure, the way the shutdown sweep
            # already does: a raising release must not cost this host the
            # close that is the whole point of the thread.
            stopped = True
            try:
                if release is not None:
                    try:
                        release()
                    except Exception:
                        pass
                for _subscription_id, subscription_release in owned_subscriptions:
                    try:
                        subscription_release()
                    except Exception:
                        pass
                if host is not None:
                    stopped = bool(host.close(timeout=30.0))
            finally:
                try:
                    with state_lock:
                        # A worker that did NOT stop stays in the table: after the
                        # pop this is the only handle on it, and the shutdown
                        # sweep could no longer see the thread it must still join.
                        # The interaction cache is dropped whether or not
                        # the worker stopped, because the ACK below is what
                        # makes the frontend drop its own: a worker that did
                        # not stop can still publish, and the two sides
                        # disagreeing about what has been sent is exactly the
                        # protocol violation the frontend refuses loudly.
                        last_interaction.pop(host_id, None)
                        if stopped:
                            hosts.pop(host_id, None)
                            last_front_sequence.pop(host_id, None)
                        closing_hosts.discard(host_id)
                        closer_threads.discard(thread)
                        fronts.trim_free(len(hosts) * FRONT_DEPTH)
                finally:
                    # A pool cleanup failure must not suppress this existing ack;
                    # it still propagates out of the close worker.
                    try:
                        send(("host-closed", host_id))
                    except Exception:
                        pass

        thread = Thread(
            target=finish,
            name=f"zlc-render-{name}-close",
            daemon=True,
        )
        with state_lock:
            closer_threads.add(thread)
        thread.start()

    def call_host(
        request_id: int,
        host_id: str,
        method: str,
        args: object,
        kwargs: object,
    ) -> None:
        if method not in _REMOTE_METHODS:
            raise ValueError(f"unsupported remote host method {method!r}")
        host = hosts[host_id]
        selected_args = _resolve_inputs(args, inputs)
        selected_kwargs = _resolve_inputs(kwargs, inputs)
        answer = getattr(host, method)(*selected_args, **selected_kwargs)
        if not isinstance(answer, Future):
            raise TypeError(f"remote host method {method!r} returned no Future")
        begin(request_id, answer)

    def save_artifact(request_id: int, payload: tuple[object, ...]) -> None:
        (
            base_path,
            input_ref,
            spec,
            parameters,
            size,
            viewport,
            classifier_thresholds,
            facet_focus,
            fit,
            lineage,
            selectors,
            source,
            host_id,
        ) = payload
        plot_input = _resolve_inputs(input_ref, inputs)

        save_arguments = {
            "plot_input": plot_input,
            "spec": spec,
            "parameters": parameters,
            "size": size,
            "viewport": viewport,
            "classifier_thresholds": classifier_thresholds,
            "facet_focus": facet_focus,
            "fit": fit,
            "lineage": lineage,
            "selectors": selectors,
            "source": source,
        }
        if host_id is None:
            from .figure_artifact import save_figure_artifact

            answer = save_worker.submit(
                save_figure_artifact,
                base_path,
                **save_arguments,
            )
        else:
            from .figure_artifact import _submit_figure_artifact

            # Queue the save transaction before this service loop can accept
            # a later Refresh/configure command for the same Editor host.
            # The Host worker performs the slow archive/export work; the
            # service loop remains free to route every other host.
            answer = _submit_figure_artifact(
                hosts[str(host_id)],
                base_path,
                **save_arguments,
            )
        begin_process_result(
            request_id,
            answer,
            lambda paths: ("save-paths", *paths),
        )

    def begin_process_result(
        request_id: int,
        answer: Future,
        encode: Callable[[object], object],
    ) -> None:
        """Complete a service operation that has no Host Front result."""

        pending[request_id] = answer

        def completed(done: Future) -> None:
            pending.pop(request_id, None)
            if done.cancelled():
                send(("cancelled", request_id))
                return
            try:
                result = done.result()
                value = getattr(result, "value", result)
            except BaseException as error:
                _send_error(send, request_id, error)
                return
            reply(request_id, ("result", request_id, encode(value), None))

        answer.add_done_callback(completed)

    def save_front(
        request_id: int,
        path: str,
        width: int,
        height: int,
        pixels: bytes,
    ) -> None:
        """Write the immutable pixels selected in B without involving A."""

        def write() -> Path:
            view = memoryview(pixels).toreadonly()
            try:
                RasterBuffer(int(width), int(height), view).save(Path(path))
            finally:
                view.release()
            return Path(path)

        begin_process_result(
            request_id,
            save_worker.submit(write),
            lambda result: ("save-path", str(result)),
        )

    try:
        while True:
            if not connection.poll(_POLL_SLICE_SECONDS):
                if write_failed.is_set():
                    # The parent stopped reading: nothing drawn here can
                    # reach anyone.  Leave the way an EOF leaves.
                    break
                continue
            message = _receive_message(connection)
            last_request[0] = monotonic()
            kind = message[0]
            if kind == "input":
                token, payload, descriptors = message[1:]
                inputs[int(token)] = _load_input(payload, descriptors, schemas)
                send(("input-ack", int(token)))
                continue
            if kind == "release-front":
                with state_lock:
                    fronts.release(str(message[1]), len(hosts) * FRONT_DEPTH)
                continue
            if kind == "drop-input":
                inputs.pop(int(message[1]), None)
                continue
            if kind == "cancel":
                answer = pending.get(int(message[1]))
                if answer is not None:
                    answer.cancel()
                continue
            if kind == "close-host":
                close_host(str(message[1]))
                continue
            if kind == "shutdown":
                break
            if kind != "request":
                raise RuntimeError(f"unknown render command {kind!r}")
            request_id = int(message[1])
            action = str(message[2])
            payload = message[3:]
            try:
                if action == "create":
                    create_host(request_id, *payload)
                elif action == "call":
                    call_host(request_id, *payload)
                elif action == "subscribe":
                    start_subscription(request_id, *payload)
                elif action == "unsubscribe":
                    unsubscribe(request_id, *payload)
                elif action == "save":
                    save_artifact(request_id, payload)
                elif action == "save-front":
                    save_front(request_id, *payload)
                else:
                    raise ValueError(f"unknown render request {action!r}")
            except BaseException as error:
                _send_error(send, request_id, error)
    except (EOFError, OSError):
        pass
    finally:
        stopping.set()
        for answer in tuple(pending.values()):
            answer.cancel()
        for _host_id, release in tuple(subscriptions.values()):
            try:
                release()
            except Exception:
                pass
        for thread in tuple(closer_threads):
            thread.join(timeout=30.0)
        for release in tuple(front_releases.values()):
            try:
                release()
            except Exception:
                pass
        for host in tuple(hosts.values()):
            try:
                host.close(timeout=30.0)
            except Exception:
                pass
        save_worker.shutdown(wait=True, cancel_futures=True)
        inputs.clear()
        schemas.clear()
        fronts.close()
        # Flush what is still queued before the pipe goes: a refusal already
        # handed to the writer is the operator's only word about why, and the
        # farewell is the parent reader's clean end.  Both go THROUGH the
        # writer, so the farewell is queued before the sentinel that retires
        # it -- put after, it lands in a queue nothing drains and the parent
        # ends on EOF, which it records as a failure of a process that in
        # fact stopped exactly as asked.
        try:
            send(("stopped",))
        except Exception:
            pass
        outbox.put(_STOP_WRITER)
        writer.join(timeout=30.0)
        connection.close()


__all__ = ["RenderProcess"]
