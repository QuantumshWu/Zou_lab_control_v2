"""Rolling kind semantic handler."""

from __future__ import annotations

from typing import Any

from ..kinds import PlotKind
from zlc_data import DatasetSchema
from ..specs import Reduction, RollingPlot
from .base import KindHandler


def render(renderer: Any, payload: Any, state: Any) -> None:
    renderer._update_rolling(payload, state)


def build_payload(projection: Any, view: Any, state: Any) -> None:
    # The history record is a projection-layer type.  Importing it lazily keeps
    # the closed kind registry independent of FitProjection's implementation.
    from .._fit_projection import RollingHistoryPoint

    spec = projection._spec
    history = list(projection._context.rolling_history)
    if not history:
        # The first revision seeds the full shot history from the repeat
        # axis: a static snapshot is a complete rolling record, and a live
        # session starts with its seed data instead of a single point.
        history.extend(
            RollingHistoryPoint(sample, shot_index)
            for shot_index, sample in enumerate(
                view.rolling_history_samples(
                    group=spec.group,
                    aggregation=spec.reduction,
                )
            )
        )
    elif history[-1].sample.revision != projection._revision:
        history.append(
            RollingHistoryPoint(
                view.rolling_sample(
                    group=spec.group,
                    aggregation=spec.reduction,
                ),
                history[-1].shot_index + 1,
            )
        )
    window = int(state["window"])
    projection._rolling_history_cache = tuple(history[-window:])
    projection._payload = projection._rolling_payload(
        projection._rolling_history_cache,
        window=window,
    )


def admits(schema: Any) -> bool:
    return default_spec(schema) is not None


def validate(view: Any, spec: Any) -> None:
    # Rolling reduces whole revisions; only an authored group can be invalid.
    view.validate_rolling(spec.group)


def label_roles(spec: Any) -> tuple[tuple[str, tuple], ...]:
    """A rolling plot's x is the shot counter; its y names the value."""

    return (
        ("title", ("title",)),
        ("x", ("repeat",)),
        ("y", ("value",)),
    )


def default_spec(schema: Any) -> RollingPlot | None:
    """Infer the ungrouped rolling reduction for any dataset schema."""

    if not isinstance(schema, DatasetSchema):
        return None
    return RollingPlot(reduction=Reduction.MEAN)


HANDLER = KindHandler(
    PlotKind.ROLLING,
    "Rolling",
    RollingPlot,
    "series",
    render,
    build_payload,
    ("kind", "group", "reduction"),
    admits,
    default_spec,
    label_roles,
    validate,
)
