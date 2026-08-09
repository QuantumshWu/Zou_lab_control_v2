"""Generic precomposed frozen panel surfaces for Workbench consumers.

The producer of a page owns its typed plot specification; the board owns only
the ordinary panel state and lifecycle.  This seam keeps either side from
reconstructing the other's truth and keeps all plotting on ``RasterPlotHost``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import CancelledError
from dataclasses import dataclass, field
from time import monotonic
from threading import Lock
from typing import Protocol, runtime_checkable

from zlc_plot import RasterPlotHost

from .panel_state import PanelState, compose_panel_spec
from .plot_annotations import (
    PanelPlotAnnotations,
    apply_panel_plot_annotations,
)


__all__ = [
    "PreparedPanelPageSpec",
    "PreparedPanelSurface",
    "create_prepared_panel_surface",
]


@runtime_checkable
class PreparedPanelPageSpec(Protocol):
    """Structural page contract independent of the feature that produced it."""

    key: str
    title: str
    signal: str
    plot_input: object
    spec: object
    parameters: Mapping[str, object]
    facet_thresholds: Sequence[float | None]
    fit_options: Mapping[str, object]

    def panel_state(self, *, signal: str, interval_ms: int) -> PanelState:
        """Return the one authored state used by card, editor, and host."""


@dataclass(slots=True)
class PreparedPanelSurface:
    """One exact page hosted by the ordinary owner-thread raster stack."""

    page: PreparedPanelPageSpec
    state: PanelState
    host: RasterPlotHost
    operations: tuple[object, ...]
    descriptions: tuple[object, object, object]
    annotations: PanelPlotAnnotations = field(default_factory=PanelPlotAnnotations)
    _description_values: list[object | None] = field(init=False, repr=False)
    _description_done: list[bool] = field(init=False, repr=False)
    _description_error: BaseException | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _description_lock: Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._description_values = [None] * len(self.descriptions)
        self._description_done = [False] * len(self.descriptions)
        self._description_lock = Lock()
        for index, operation in enumerate(self.descriptions):
            callback = getattr(operation, "add_done_callback", None)
            if callable(callback):
                callback(
                    lambda completed, selected=index: self._capture_description(
                        selected,
                        completed,
                    )
                )
            else:
                self._capture_description(index, operation)

    @staticmethod
    def _done(operation: object) -> bool:
        done = getattr(operation, "done", None)
        return True if not callable(done) else bool(done())

    @staticmethod
    def _completed_value(operation: object) -> object:
        result = getattr(operation, "result", None)
        resolved = result() if callable(result) else operation
        return getattr(resolved, "value", resolved)

    def _capture_description(self, index: int, operation: object) -> None:
        """Resolve only from a completion callback, never from the GUI poll."""

        try:
            value = self._completed_value(operation)
        except CancelledError:
            value = None
        except BaseException as error:
            value = None
            with self._description_lock:
                if self._description_error is None:
                    self._description_error = error
        with self._description_lock:
            self._description_values[index] = value
            self._description_done[index] = True

    @property
    def descriptions_ready(self) -> bool:
        """Whether the complete editor surface can be read without waiting."""

        with self._description_lock:
            return all(self._description_done)

    @property
    def configured(self) -> bool:
        """Whether all page-owned overlays and fits reached a terminal state."""

        return all(self._done(operation) for operation in self.operations)

    def failure(self) -> BaseException | None:
        """Return an already-finished asynchronous failure without blocking."""

        with self._description_lock:
            if self._description_error is not None:
                return self._description_error
        for operation in self.operations:
            if not self._done(operation):
                continue
            exception = getattr(operation, "exception", None)
            if not callable(exception):
                continue
            try:
                error = exception()
            except CancelledError:
                continue
            if error is not None and not isinstance(error, CancelledError):
                return error
        return None

    def description_values(self) -> tuple[object, object, object]:
        """Resolve metadata only after :attr:`descriptions_ready` is true."""

        if not self.descriptions_ready:
            raise RuntimeError("prepared panel descriptions are still rendering")
        with self._description_lock:
            if self._description_error is not None:
                raise self._description_error
            return tuple(self._description_values)  # type: ignore[return-value]

    def wait(self, timeout: float | None = None) -> None:
        """Explicit blocking boundary for export/tests, never the GUI beat."""

        deadline = None if timeout is None else monotonic() + float(timeout)
        for operation in (*self.descriptions, *self.operations):
            result = getattr(operation, "result", None)
            if callable(result):
                remaining = (
                    None
                    if deadline is None
                    else max(0.0, deadline - monotonic())
                )
                result(timeout=remaining)
        remaining = None if deadline is None else max(0.0, deadline - monotonic())
        self.host.wait_for_front(timeout=remaining)

    def close(self) -> None:
        self.host.close()


def create_prepared_panel_surface(
    page: PreparedPanelPageSpec,
    *,
    signal: str | None = None,
    interval_ms: int = 400,
    state: PanelState | None = None,
) -> PreparedPanelSurface:
    """Compose final semantics before the host's first render."""

    if not isinstance(page, PreparedPanelPageSpec):
        raise TypeError("page must follow PreparedPanelPageSpec")
    if state is None:
        state = page.panel_state(
            signal=str(page.signal if signal is None else signal),
            interval_ms=int(interval_ms),
        )
    elif not isinstance(state, PanelState):
        raise TypeError("state must be PanelState or None")
    elif signal is not None or interval_ms != 400:
        raise ValueError("state cannot be combined with signal or interval_ms")

    snapshot = getattr(page.plot_input, "snapshot", page.plot_input)
    schema = getattr(getattr(snapshot, "block", None), "schema", None)
    spec = compose_panel_spec(schema, page.spec, state)

    parameters = dict(page.parameters)
    parameters.update(state.display)
    parameters["title"] = state.title
    if state.kind == "image":
        parameters["site_overlay"] = state.site_overlay
    fit = dict(state.fit)
    model = fit.pop("model", None)
    host = RasterPlotHost.from_plot(
        page.plot_input,
        spec,
        size=state.size or None,
        parameters=parameters,
    )
    operations: list[object] = []
    try:
        annotations = PanelPlotAnnotations(tuple(page.facet_thresholds))
        operations.extend(
            apply_panel_plot_annotations(host, annotations, display=False)
        )
        if model is not None:
            operations.append(host.fit(str(model), **fit))
        descriptions = (
            host.describe_display(),
            host.describe_semantics(),
            host.fit_models(),
        )
    except Exception:
        host.close()
        raise
    return PreparedPanelSurface(
        page,
        state,
        host,
        tuple(operations),
        descriptions,
        annotations,
    )
