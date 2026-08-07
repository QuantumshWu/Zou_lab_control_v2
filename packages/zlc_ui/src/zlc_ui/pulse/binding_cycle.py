"""Pure binding-cycle semantics shared by pulse presenters and demos."""

from __future__ import annotations


def cycle_binding_kind(
    binding: str | None,
    *,
    field_kind: str,
) -> str | None:
    """Return the next binding kind for a clicked Pulse field dot.

    Duration and DAC-value fields use ``off -> scan -> api -> off``.  Output
    delays are not scannable and use ``off -> api -> off``.  The Qt field only
    emits the click intent; a presenter calls this function, updates its plain
    model, and projects the new state back with ``set_field_state``.
    """

    normalized_kind = str(field_kind).strip().lower()
    if normalized_kind in {"delay", "output_delay"}:
        cycle: tuple[str | None, ...] = (None, "api")
    elif normalized_kind in {"duration", "analog", "dac", "value"}:
        cycle = (None, "scan", "api")
    else:
        raise ValueError(
            "field_kind must be duration, analog, dac, value, delay, or output_delay"
        )

    normalized_text = "" if binding is None else str(binding).strip().lower()
    normalized_binding = None if not normalized_text else normalized_text
    if normalized_binding not in cycle:
        raise ValueError(
            f"binding {binding!r} is not valid for field_kind {normalized_kind!r}"
        )
    return cycle[(cycle.index(normalized_binding) + 1) % len(cycle)]


__all__ = ["cycle_binding_kind"]
