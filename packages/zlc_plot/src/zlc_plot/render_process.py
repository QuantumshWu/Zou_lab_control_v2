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
from threading import Event, Lock, RLock, Thread
from time import monotonic
from types import SimpleNamespace
from uuid import uuid4
import weakref

import numpy as np

from .front import (
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
    shared: object,
    retirements: Queue,
) -> None:
    process._release_front(str(lease_id))
    # A weakref finalizer runs while the exporting memoryview is itself being
    # dismantled.  mmap.close at that exact instant still sees one exported
    # pointer.  Hand the mapping to one process-owned retirement lane; its
    # next turn is after memoryview teardown, and it retains the handle across
    # a rare BufferError instead of letting SharedMemory.__del__ warn.
    retirements.put((_MAPPING_RELEASED, shared))


def _retire_shared_mappings(retirements: Queue) -> None:
    waiting: list[object] = []
    outstanding = 0
    stopping = False
    while True:
        try:
            item = retirements.get(timeout=0.02 if not waiting else 0.002)
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
        defaults: object | None,
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
        defaults = self._defaults
        if defaults is None:
            raise RuntimeError("remote raster host has not published frontend defaults")
        return defaults

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

    def qt_widget(self):
        widget = self._qt_widget
        if widget is None:
            from .backends import Qt5PlotWidget

            widget = Qt5PlotWidget(self)
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

    def _set_frontend_defaults(self, selector_handle_radius_px: float) -> None:
        self._defaults = SimpleNamespace(
            interaction=SimpleNamespace(
                selector_handle_radius_px=float(selector_handle_radius_px)
            )
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


class RenderProcess:
    """One long-lived process containing any number of RasterPlotHosts."""

    #: How long a child may stay silent, with requests outstanding and after
    #: being asked, before it has stopped.  "The service is up" used to have
    #: two proxies and no owner -- the process existing and its pipe not
    #: having closed -- and a child that was alive and not answering
    #: satisfied both: every request stayed pending, every panel it served
    #: kept its last picture, and nothing anywhere said so.  The reader owns
    #: service failure, so the reader owns liveness.  The child's dispatch
    #: loop only hands work to host workers and answers a ping in
    #: microseconds; ten seconds of not even that is not slowness.
    ANSWER_DEADLINE_SECONDS = 10.0

    def __init__(
        self, name: str, *, answer_deadline_seconds: float | None = None
    ) -> None:
        selected = str(name).strip()
        if not selected:
            raise ValueError("render process name must be non-empty")
        deadline = float(
            self.ANSWER_DEADLINE_SECONDS
            if answer_deadline_seconds is None
            else answer_deadline_seconds
        )
        if not deadline > 0.0:
            raise ValueError("answer deadline must be positive")
        self.name = selected
        self._answer_deadline = deadline
        self._ping: tuple[int, float] | None = None
        self._ping_serial = 0
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
        self._closing = False
        self._owners = 1
        self._close_started: float | None = None
        self._closed = True
        self._mapping_retirements: Queue = Queue()
        self._mapping_retirement_thread = Thread(
            target=_retire_shared_mappings,
            args=(self._mapping_retirements,),
            name=f"zlc-render-{selected}-mapping-retirement",
            daemon=True,
        )
        self._mapping_retirement_thread.start()
        self._mapping_retirement_finalizer = weakref.finalize(
            self,
            self._mapping_retirements.put,
            _RETIRE_MAPPINGS,
        )
        self._spawn_child()

    def _spawn_child(self) -> None:
        """Start one fresh child after the previous reader fully retired."""

        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        stopped = Event()
        process = context.Process(
            target=_render_process_main,
            args=(child, self.name),
            name=f"zlc-render-{self.name}",
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
        self._ping = None
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
        device_pixel_ratio: float = 1.0,
    ) -> _RemoteRasterPlotHost:
        self._ensure_running()
        with self._lock:
            if (
                self._closing
                or self._closed
                or not self._process.is_alive()
            ):
                raise RuntimeError("render process is not running")
            host_id = uuid4().hex
            host = _RemoteRasterPlotHost(self, host_id, None)
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

    def _input_reference(
        self, value: object, used: set[int]
    ) -> tuple[str, int]:
        key = self._input_key(value)
        with self._lock:
            token = self._input_tokens.get(key)
            if token is not None:
                if token not in used:
                    self._input_refcounts[token] += 1
                    used.add(token)
                return _INPUT_REF, token
        # Register the token, then enqueue its input.  A concurrent host
        # call may see the token, but its request cannot overtake this
        # message: one writer per direction drains the queue in order.
        with self._lock:
            token = self._input_tokens.get(key)
            if token is not None:
                if token not in used:
                    self._input_refcounts[token] += 1
                    used.add(token)
                return _INPUT_REF, token
            if self._closing or self._closed:
                raise RuntimeError("render process is closing")
            self._input_serial += 1
            token = self._input_serial
            self._input_tokens[key] = token
            self._input_keys[token] = key
            self._input_identity_owners[token] = value
            self._input_kinds[token] = str(key[0])
            self._input_refcounts[token] = 1
            used.add(token)
        buffers: list[pickle.PickleBuffer] = []
        released_buffers = 0
        shared: list[SharedMemory] = []
        descriptors: list[tuple[str, int]] = []
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
                    block = SharedMemory(create=True, size=max(1, nbytes))
                    shared.append(block)
                    destination = block.buf
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
            with self._lock:
                self._input_uploads[token] = tuple(shared)
            self._send(("input", token, payload, tuple(descriptors)))
        except BaseException:
            for item in buffers[released_buffers:]:
                try:
                    item.release()
                except Exception:
                    pass
            buffers.clear()
            with self._lock:
                self._input_tokens.pop(key, None)
                self._input_keys.pop(token, None)
                self._input_identity_owners.pop(token, None)
                self._input_kinds.pop(token, None)
                self._input_refcounts.pop(token, None)
                self._input_uploads.pop(token, None)
            for block in shared:
                try:
                    block.close()
                    block.unlink()
                except Exception:
                    pass
            raise
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

    def _require_answer(self) -> None:
        """Silence with work outstanding is asked about, then judged.

        Runs on the reader thread whenever the pipe has said nothing for one
        slice.  With nothing pending there is nothing to wait for.  With
        requests pending, the first silent slice sends a ping; a child whose
        dispatch loop is alive answers it at once, and the ping is forgotten
        by :meth:`_answered`.  A ping still unanswered after the deadline is
        the service's failure, raised here so that the reader ends exactly
        as it ends on EOF: every pending request fails, every host is marked
        failed, the child is terminated and the console remounts on a fresh
        one -- said, this time, on the status strip.
        """

        now = monotonic()
        with self._lock:
            if not self._pending:
                self._ping = None
                return
            if self._ping is not None:
                serial, asked = self._ping
                if now - asked > self._answer_deadline:
                    raise TimeoutError(
                        f"render process {self.name!r} did not answer for "
                        f"{self._answer_deadline:g} s with "
                        f"{len(self._pending)} request(s) outstanding"
                    )
                return
            self._ping_serial += 1
            serial = self._ping_serial
            self._ping = (serial, now)
        self._send(("ping", serial))

    def _answered(self, serial: int) -> None:
        with self._lock:
            if self._ping is not None and self._ping[0] == serial:
                self._ping = None

    def _read_messages(self) -> None:
        failure: BaseException | None = None
        try:
            while True:
                if not self._connection.poll(_POLL_SLICE_SECONDS):
                    self._require_answer()
                    continue
                message = _receive_message(self._connection)
                kind = message[0]
                if kind == "pong":
                    self._answered(int(message[1]))
                elif kind == "front":
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
        interaction: RasterInteractionMap,
        lease_id: str,
        shared_name: str,
        nbytes: int,
        width: int,
        height: int,
        selector_handle_radius_px: float,
    ) -> None:
        shared = _open_shared_mapping(shared_name)
        # The retirement lane must know about this mapping before a Front can
        # escape the reader thread.  A caller may retain only an ndarray view
        # after the RenderProcess itself is gone; STOP therefore waits for the
        # matching store finalizer instead of exiting ahead of that last view.
        self._mapping_retirements.put(_MAPPING_OPENED)
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
                shared,
                self._mapping_retirements,
            )
        except BaseException:
            if pixels is not None:
                pixels.release()
            self._mapping_retirements.put((_MAPPING_RELEASED, shared))
            raise
        finalizer.atexit = False
        front = RasterFront(
            identity=identity,
            buffer=RasterBuffer(width, height, pixels),
            logical_size=logical_size,
            logical_dpi=logical_dpi,
            device_pixel_ratio=device_pixel_ratio,
            interaction=interaction,
        )
        with self._lock:
            host = self._hosts.get(str(host_id))
        if host is None:
            # The host was retired while this front crossed the pipe.  Let the
            # buffer exporter die only after both local views are gone; calling
            # its finalizer here would close SharedMemory under live exports.
            del front
            pixels.release()
            del pixels, store
            return
        host._set_frontend_defaults(selector_handle_radius_px)
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

    def _finish_input_upload(self, token: int) -> None:
        with self._lock:
            blocks = self._input_uploads.pop(token, ())
        for block in blocks:
            try:
                block.close()
                block.unlink()
            except FileNotFoundError:
                pass

    def _receive_host_closed(self, host_id: str) -> None:
        with self._lock:
            host = self._hosts.pop(host_id, None)
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


class _SharedFrontPool:
    """Child-owned shared blocks, recycled only after the frontend releases."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._leased: dict[str, SharedMemory] = {}
        # Three reusable blocks TOTAL, not three for every historical raster
        # size.  A window dragged through hundreds of pixel sizes must not
        # turn every old size into a permanent shared-memory cache entry.
        self._free: deque[SharedMemory] = deque()

    def publish(self, pixels: object) -> tuple[str, str, int]:
        source = memoryview(pixels).cast("B")
        nbytes = source.nbytes
        with self._lock:
            block = next(
                (candidate for candidate in self._free if candidate.size == nbytes),
                None,
            )
            if block is not None:
                self._free.remove(block)
            else:
                block = SharedMemory(create=True, size=nbytes)
            block.buf[:nbytes] = source
            # A Front from a crashed generation may outlive a restarted
            # service.  A process-local integer would then collide with the
            # new pool and release an unrelated live Front.
            lease_id = uuid4().hex
            self._leased[lease_id] = block
        return lease_id, block.name, nbytes

    def release(self, lease_id: str) -> None:
        retired = None
        with self._lock:
            block = self._leased.pop(str(lease_id), None)
            if block is None:
                return
            self._free.append(block)
            if len(self._free) > 3:
                retired = self._free.popleft()
        if retired is not None:
            retired.close()
            retired.unlink()

    def close(self) -> None:
        with self._lock:
            blocks = tuple(self._leased.values()) + tuple(self._free)
            self._leased.clear()
            self._free.clear()
        for block in blocks:
            try:
                block.close()
                block.unlink()
            except FileNotFoundError:
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
    from .raster import RasterPlotHost
    from .session import PlotSession

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
    save_worker = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix=f"zlc-render-{name}-save"
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

    def publish_front(host_id: str, front: RasterFront) -> None:
        sequence = int(front.identity.sequence)
        if sequence <= last_front_sequence.get(host_id, -1):
            return
        last_front_sequence[host_id] = sequence
        lease_id, shared_name, nbytes = fronts.publish(front.buffer.pixels)
        send(
            (
                "front",
                host_id,
                front.identity,
                tuple(front.logical_size),
                float(front.logical_dpi),
                float(front.device_pixel_ratio),
                front.interaction,
                lease_id,
                shared_name,
                nbytes,
                int(front.buffer.width),
                int(front.buffer.height),
                float(DEFAULTS.interaction.selector_handle_radius_px),
            )
        )

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
    ) -> None:
        plot_input = _resolve_inputs(input_ref, inputs)

        def factory() -> PlotSession:
            return PlotSession(
                plot_input,
                spec,
                size=size,
                parameters=parameters,
                defaults=DEFAULTS,
                device_pixel_ratio=device_pixel_ratio,
            )

        host = RasterPlotHost(factory, host_id=host_id)
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
                with state_lock:
                    # A worker that did NOT stop stays in the table: after the
                    # pop this is the only handle on it, and the shutdown
                    # sweep could no longer see the thread it must still join.
                    if stopped:
                        hosts.pop(host_id, None)
                        last_front_sequence.pop(host_id, None)
                    closing_hosts.discard(host_id)
                    closer_threads.discard(thread)
                # The acknowledgement goes out either way: it is what lets the
                # console finish closing, and a worker this child is still
                # holding is this child's problem, not a reason to strand the
                # operator in a window that will not close.
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
            kind = message[0]
            if kind == "ping":
                send(("pong", int(message[1])))
                continue
            if kind == "input":
                token, payload, descriptors = message[1:]
                inputs[int(token)] = _load_input(payload, descriptors, schemas)
                send(("input-ack", int(token)))
                continue
            if kind == "release-front":
                fronts.release(str(message[1]))
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
