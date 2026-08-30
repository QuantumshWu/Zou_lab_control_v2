"""PulseTimeline kind semantic handler."""

from __future__ import annotations

from typing import Any

from ..kinds import PlotKind
from ..specs import PulseTimelinePlot
from .base import KindHandler


def render(renderer: Any, payload: Any, state: Any, *, axes: Any, key: str) -> None:
    renderer._update_pulse_timeline(axes, payload, state, key)


def build_payload(_projection: Any, _view: Any, _state: Any) -> None:
    # PulseTimelineData is already a typed non-Dataset payload; DatasetView
    # projection is never requested for this handler.
    return None


def admits(schema: Any) -> bool:
    # PulseTimelineData is deliberately outside DatasetSchema.  The session
    # exposes it through the same semantic descriptor with a schema-less
    # input, but this handler must never claim a dataset projection.
    return default_spec(schema) is not None


def label_roles(spec: Any) -> tuple[tuple[str, tuple], ...]:
    """A pulse timeline's x is always the pulse time base."""

    return (
        ("title", ("title",)),
        ("x", ("pulse-time",)),
    )


def default_spec(schema: Any) -> PulseTimelinePlot | None:
    """Pulse timelines are schema-less and require an authored pulse payload."""

    return None


# fit_target=None: a pulse timeline has no fittable projection, so the fit
# catalogue is empty rather than a target the FitTarget contract rejects.
HANDLER = KindHandler(
    PlotKind.PULSE_TIMELINE,
    "Pulse timeline",
    PulseTimelinePlot,
    None,
    render,
    build_payload,
    ("kind",),
    admits,
    default_spec,
    label_roles,
)
