"""Grouped keyed-choice picker projection over plain names and metadata."""

from __future__ import annotations

from .fluent import FluentTreeComboBox, signals_blocked as _signals_blocked


def choice_state(
    key,
    metadata,
    sources=None,
    *,
    state_labels: tuple[str, str, str] | None = None,
) -> str:
    """Return an injected state label for one keyed choice.

    ``state_labels`` is ``(available, pending, unassigned)``.  Passing
    ``None`` omits state text entirely, leaving the picker presentation-neutral.
    """

    if state_labels is None:
        return ""
    if len(state_labels) != 3:
        raise ValueError("state_labels must contain available, pending, unassigned")
    available, pending, unassigned = (str(value) for value in state_labels)
    if sources is not None and not (sources.get(str(key)) or ()):
        return unassigned
    return available if metadata.get(str(key)) else pending


def _choice_short_label(key, labels) -> str:
    short = (labels or {}).get(str(key))
    return str(short) if short else str(key)


def grouped_choice_items(
    names,
    sources,
    metadata,
    labels=None,
    *,
    state_labels: tuple[str, str, str] | None = None,
    empty_source_label: str = "",
) -> list[tuple[str, str | None]]:
    """Build flat ``(display, opaque_key)`` rows grouped by source."""

    names = sorted(str(name) for name in (names or []))
    sources = dict(sources or {})
    metadata = dict(metadata or {})
    labels = dict(labels or {})
    empty_source_label = str(empty_source_label)
    by_source: dict[str, list[str]] = {}
    for key in names:
        for source in ([str(value) for value in (sources.get(key) or [])] or [empty_source_label]):
            by_source.setdefault(source, []).append(key)

    items: list[tuple[str, str | None]] = []
    for source in sorted(by_source, key=lambda value: (value == empty_source_label, value.lower())):
        items.append((source, None))
        for key in by_source[source]:
            short = _choice_short_label(key, labels)
            description = metadata.get(key)
            detail = f"  [{description}]" if description else ""
            state = choice_state(key, metadata, sources, state_labels=state_labels)
            suffix = f"  {state}" if state else ""
            items.append((f"    {short}{detail}{suffix}", key))
    return items


def choice_tree_groups(
    names,
    sources,
    metadata,
    labels=None,
    *,
    state_labels: tuple[str, str, str] | None = None,
    empty_source_label: str = "",
) -> list[tuple[str, list[tuple[str, str, str]]]]:
    """Build nested ``(source, [(leaf, key, full_label)])`` rows."""

    flat = grouped_choice_items(
        names,
        sources,
        metadata,
        labels,
        state_labels=state_labels,
        empty_source_label=empty_source_label,
    )
    groups: list[tuple[str, list[tuple[str, str, str]]]] = []
    current_source: str | None = None
    leaves: list[tuple[str, str, str]] = []
    for display, key in flat:
        if key is None:
            if current_source is not None:
                groups.append((current_source, leaves))
            current_source = display
            leaves = []
            continue
        if current_source is None:
            raise ValueError("choice rows must begin with a source header")
        short = _choice_short_label(key, labels)
        leaves.append((display.strip(), key, short))
    if current_source is not None:
        groups.append((current_source, leaves))
    return groups


def fill_grouped_choice_combo(
    combo,
    *,
    names,
    sources,
    metadata,
    current,
    none_label=None,
    labels=None,
    state_labels: tuple[str, str, str] | None = None,
    empty_source_label: str = "",
) -> None:
    """Populate a flat or tree combo with grouped keyed choices."""

    cur = str(current or "")
    names = list(names or [])
    if cur and cur not in {str(name) for name in names}:
        names = [*names, cur]
    if isinstance(combo, FluentTreeComboBox):
        with _signals_blocked(combo):
            combo.set_choice_tree(
                choice_tree_groups(
                    names,
                    sources,
                    metadata,
                    labels,
                    state_labels=state_labels,
                    empty_source_label=empty_source_label,
                ),
                current=cur,
                none_label=none_label,
            )
        return
    with _signals_blocked(combo):
        combo.clear()
        if none_label is not None:
            combo.addItem(none_label, "")
        for label, key in grouped_choice_items(
            names,
            sources,
            metadata,
            labels,
            state_labels=state_labels,
            empty_source_label=empty_source_label,
        ):
            if key is None:
                combo.addItem(label, None)
                item = combo.model().item(combo.count() - 1)
                if item is not None:
                    item.setEnabled(False)
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
            else:
                combo.addItem(label, key)
        index = combo.findData(cur)
        combo.setCurrentIndex(0 if index < 0 and none_label is not None else index)


def read_editable_combo(combo) -> str:
    """Read an opaque key or current free-text draft from an editable combo."""

    if isinstance(combo, FluentTreeComboBox):
        return combo.current_choice_key()
    index = combo.currentIndex()
    text = combo.currentText()
    if index >= 0 and text == combo.itemText(index):
        data = combo.itemData(index)
        if data is not None:
            return str(data).strip()
    return text.strip()


def coerce_choice_labels(provider) -> dict:
    """Normalize a callback returning ``{opaque key: short label}``."""

    if not callable(provider):
        return {}
    try:
        return {str(key): str(value) for key, value in dict(provider()).items() if value}
    except Exception:
        return {}


__all__ = [
    "choice_state",
    "choice_tree_groups",
    "coerce_choice_labels",
    "fill_grouped_choice_combo",
    "grouped_choice_items",
    "read_editable_combo",
]
