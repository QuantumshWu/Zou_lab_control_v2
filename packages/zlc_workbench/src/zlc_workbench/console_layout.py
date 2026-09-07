"""The one persisted document for a stopped TaskConsole board.

The console owns composition, so its reusable pipeline belongs here rather
than in the device, plot, or UI packages.  This module is the only JSON shell:
the presenter deals in typed entries and asks this document to parse or write
them.

A panel is written down with the identity it had on the board.  A panel is a
producer -- its Bridge publishes the ROI and fit outputs it derives under
``@logic/<panel id>/<output>`` -- and a downstream panel or logic row names
that producer by that spelling; a document that kept the downstream name and
dropped the upstream identity could not say which panel it was reading.
Loading never reuses a written identity: the console mints fresh ones and
:func:`load_layout` respells every reference in the document to them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

from zlc_durable import write_readable_json
from zlc_plot.kinds import AxisDomain, AxisRef
from zlc_plot.semantics import FATE_PREFIX, fate_field_name

from .logic import (
    LogicBinding,
    LogicDraft,
    artifact_input_specs,
    dataset_inputs,
    device_key_options,
    split_signal_key,
    stable_signal_key,
)
from .panel_state import PanelState


LAYOUT_FORMAT = "zlc.console-board"

#: How a panel's identity is spelled.  The console mints one from its
#: monotonic serial; this module reads the same spelling back to tell a
#: reference to a panel from a reference to a logic row.
PANEL_ID_PREFIX = "panel-"

#: Saved fate keys spelled in the axis vocabulary that preceded the unified
#: Dataset domains, keyed by the prefix that identifies each: the axis a key
#: names is kept as written, only its domain is respelled.
_RENAMED_FATE_PREFIXES = {
    f"{FATE_PREFIX}point_dimension:": f"{FATE_PREFIX}{AxisDomain.POINT.value}:",
    f"{FATE_PREFIX}data:": f"{FATE_PREFIX}{AxisDomain.CELL_DATA.value}:",
}
#: The one old fate that named a whole domain rather than an axis.  Which
#: Repeat axes a signal has is the data's to say, so this key survives the
#: parse and is expanded onto today's axes when the board is loaded.
REPEAT_DOMAIN_FATE = f"{FATE_PREFIX}{AxisDomain.REPEAT.value}"
_FATE_DOMAINS = frozenset(domain.value for domain in AxisDomain)


class LayoutError(ValueError):
    """A file cannot describe one current TaskConsole board."""


def panel_id_for(serial: int) -> str:
    """The identity of the panel minted from one console serial."""

    return f"{PANEL_ID_PREFIX}{int(serial)}"


def _names_a_panel(producer: str) -> bool:
    return (
        producer.startswith(PANEL_ID_PREFIX)
        and producer[len(PANEL_ID_PREFIX):].isdigit()
    )


def current_fate_key(key: str) -> str:
    """The current spelling of one saved semantic key.

    A non-fate key is the plot kind's own vocabulary and passes through (the
    projection checks it against the kind).  A fate key is respelled by its
    old prefix, kept when it already reads in today's ``fate:<domain>:<axis>``
    grammar, and refused by name otherwise: a saved fate nobody can read is
    a saved fate silently lost, and the operator would only learn that from
    a plot drawn along the wrong axis.
    """

    if not key.startswith(FATE_PREFIX) or key == REPEAT_DOMAIN_FATE:
        return key
    for old, new in _RENAMED_FATE_PREFIXES.items():
        if key.startswith(old):
            return new + key[len(old):]
    domain, _separator, axis_id = key[len(FATE_PREFIX):].partition(":")
    if domain in _FATE_DOMAINS and axis_id.strip():
        return key
    raise LayoutError(f"unknown fate key {key!r}")


@dataclass(frozen=True, slots=True)
class LogicLayoutEntry:
    node_id: str
    api_name: str
    values: Mapping[str, Any]
    source_signal: str
    device_keys: Mapping[str, str]
    artifact_inputs: Mapping[str, str]
    #: Whether Start opens this node's declared preview.  Part of the board an
    #: operator arranged, like the panels themselves -- not of the measurement.
    auto_preview: bool = True

    def __post_init__(self) -> None:
        node_id = str(self.node_id).strip()
        api_name = str(self.api_name).strip()
        if not node_id or not api_name:
            raise LayoutError("logic entries require node_id and api_name")
        object.__setattr__(self, "node_id", node_id)
        object.__setattr__(self, "api_name", api_name)
        object.__setattr__(self, "values", dict(self.values))
        object.__setattr__(self, "source_signal", str(self.source_signal).strip())
        object.__setattr__(
            self,
            "device_keys",
            {str(name): str(key) for name, key in self.device_keys.items()},
        )
        if any(
            not isinstance(name, str) or not isinstance(path, str)
            for name, path in self.artifact_inputs.items()
        ):
            raise LayoutError("artifact input names and paths must be strings")
        object.__setattr__(self, "artifact_inputs", dict(self.artifact_inputs))
        object.__setattr__(self, "auto_preview", bool(self.auto_preview))

    def to_tree(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "api_name": self.api_name,
            "values": dict(self.values),
            "source_signal": self.source_signal,
            "device_keys": dict(self.device_keys),
            "artifact_inputs": dict(self.artifact_inputs),
            "auto_preview": self.auto_preview,
        }


@dataclass(frozen=True, slots=True)
class LayoutDocument:
    """One canonical, ordered stopped pipeline and panel arrangement."""

    panels: tuple[PanelState, ...]
    logic: tuple[LogicLayoutEntry, ...]
    #: The identity each panel had on the board, in the order of ``panels``.
    #: Required, not defaulted: a writer that does not know its panels'
    #: identities would write ones that look right and are not, and the
    #: file would mis-wire on load; only the reader may make identities up
    #: (:meth:`from_tree`, for a file written before they were kept).
    panel_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        panels = tuple(self.panels)
        logic = tuple(self.logic)
        if any(not isinstance(item, PanelState) for item in panels):
            raise TypeError("layout panels must contain PanelState values")
        if any(not isinstance(item, LogicLayoutEntry) for item in logic):
            raise TypeError("layout logic must contain LogicLayoutEntry values")
        ids = tuple(item.node_id for item in logic)
        if len(set(ids)) != len(ids):
            raise LayoutError("logic node_id values must be unique")
        panel_ids = tuple(str(item).strip() for item in self.panel_ids)
        if len(panel_ids) != len(panels):
            raise LayoutError("a layout names one panel_id per panel")
        if any(not item for item in panel_ids) or len(set(panel_ids)) != len(panel_ids):
            raise LayoutError("panel_id values must be non-empty and unique")
        object.__setattr__(self, "panels", panels)
        object.__setattr__(self, "logic", logic)
        object.__setattr__(self, "panel_ids", panel_ids)

    @classmethod
    def from_tree(cls, tree: object) -> "LayoutDocument":
        document = _mapping(tree, "layout")
        if document.get("format") != LAYOUT_FORMAT:
            raise LayoutError("that file is not a saved board")
        _exact_fields(document, {"format", "panels", "logic"}, "layout")
        entries = tuple(
            _panel_from_tree(entry, index)
            for index, entry in enumerate(_sequence(document["panels"], "panels"))
        )
        named = tuple(panel_id for panel_id, _state in entries if panel_id is not None)
        if named and len(named) != len(entries):
            raise LayoutError("either every panel entry carries a panel_id or none does")
        if not named:
            # A board written before identities were kept: its panels were
            # minted in saved order on a fresh console, so the first entry is
            # the first identity that console ever minted -- which is all
            # such a file can mean, and enough for its references to resolve.
            named = tuple(panel_id_for(index + 1) for index in range(len(entries)))
        logic = tuple(
            _logic_from_tree(entry, index)
            for index, entry in enumerate(_sequence(document["logic"], "logic"))
        )
        return cls(tuple(state for _panel_id, state in entries), logic, named)

    @classmethod
    def read(cls, path: str | Path) -> "LayoutDocument":
        source = Path(path)
        return cls.from_tree(json.loads(source.read_text(encoding="utf-8")))

    def to_tree(self) -> dict[str, Any]:
        return {
            "format": LAYOUT_FORMAT,
            "panels": [
                {"panel_id": panel_id, **state.document()}
                for panel_id, state in zip(self.panel_ids, self.panels, strict=True)
            ],
            "logic": [entry.to_tree() for entry in self.logic],
        }

    def write(self, path: str | Path) -> Path:
        return write_readable_json(path, self.to_tree())


@dataclass(frozen=True, slots=True)
class ResolvedLayout:
    """All stopped row drafts after catalog/schema/contract resolution.

    The panels still carry the document's own identities and spellings;
    :func:`load_layout` puts them onto the identities of one load.
    """

    logic: tuple[LogicBinding, ...]
    panels: tuple[PanelState, ...]
    panel_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LoadedLayout:
    """A resolved board on fresh panel identities, ready to prepare."""

    logic: tuple[LogicBinding, ...]
    panels: tuple[PanelState, ...]
    panel_ids: tuple[str, ...]
    #: One sentence per connection or fate the document had and this board
    #: cannot carry, for the status strip: nothing is dropped silently.
    notes: tuple[str, ...]


def resolve_layout(
    document: LayoutDocument,
    *,
    catalog: object,
    installation: object,
    panel_kinds: Sequence[str],
    external_outputs: Sequence[tuple[str, str]] = (),
) -> ResolvedLayout:
    """Resolve every entry without changing the current board or a device.

    A named device or signal which is absent on today's apparatus remains an
    unresolved stopped draft, as a reusable layout requires.  A name which can
    be resolved and contradicts its declared capability/contract is an invalid
    document and rejects the whole candidate.
    """

    bindings: list[LogicBinding] = []
    installed = getattr(installation, "devices", {})
    installed_keys = set(installed) if isinstance(installed, Mapping) else set()
    for entry in document.logic:
        descriptor = catalog.get(entry.api_name)
        if descriptor is None:
            raise LayoutError(f"no logic node named {entry.api_name!r}")
        try:
            expected_values = set(descriptor.authoring_schema.field_names)
            missing_values = expected_values - set(entry.values)
            if missing_values:
                raise LayoutError(
                    f"{entry.node_id}: missing authoring fields "
                    f"{sorted(missing_values)!r}"
                )
            # A saved row is an editable raw draft.  Semantic projection is
            # deliberately deferred to the same finalizer that gates Start.
            values = dict(entry.values)
            options = device_key_options(descriptor, installation=installation)
        except Exception as error:
            raise LayoutError(f"{entry.node_id}: {error}") from error
        required = tuple(
            requirement.argument_name
            for requirement in descriptor.device_requirements
        )
        unknown_devices = set(entry.device_keys) - set(required)
        if unknown_devices:
            raise LayoutError(
                f"{entry.node_id}: unknown device bindings "
                f"{sorted(unknown_devices)!r}"
            )
        missing_devices = set(required) - set(entry.device_keys)
        if missing_devices:
            raise LayoutError(
                f"{entry.node_id}: missing device bindings "
                f"{sorted(missing_devices)!r}"
            )
        selected: dict[str, str] = {}
        for name in required:
            available = options[name]
            key = str(entry.device_keys[name])
            if key in installed_keys and key not in available:
                raise LayoutError(
                    f"{entry.node_id}: {key!r} is incompatible with device input "
                    f"{name!r}"
                )
            selected[name] = key
        artifact_specs = artifact_input_specs(descriptor)
        artifact_names = tuple(spec.name for spec in artifact_specs)
        unknown_artifacts = set(entry.artifact_inputs) - set(artifact_names)
        if unknown_artifacts:
            raise LayoutError(
                f"{entry.node_id}: unknown artifact inputs "
                f"{sorted(unknown_artifacts)!r}"
            )
        missing_artifacts = {
            spec.name
            for spec in artifact_specs
            if spec.required and spec.name not in entry.artifact_inputs
        }
        if missing_artifacts:
            raise LayoutError(
                f"{entry.node_id}: missing artifact inputs "
                f"{sorted(missing_artifacts)!r}"
            )
        bindings.append(
            LogicBinding(
                entry.node_id,
                descriptor,
                LogicDraft(
                    values,
                    entry.source_signal,
                    selected,
                    dict(entry.artifact_inputs),
                ),
                auto_preview=entry.auto_preview,
            )
        )

    output_contracts = {str(signal): str(contract) for signal, contract in external_outputs}
    for binding in bindings:
        for output in binding.descriptor.outputs:
            signal = stable_signal_key(binding.node_id, output.name)
            contract = str(output.contract_id)
            previous = output_contracts.get(signal)
            if previous is not None and previous != contract:
                raise LayoutError(
                    f"{signal!r} is declared with both {previous!r} and {contract!r}"
                )
            output_contracts[signal] = contract
    for binding in bindings:
        input_specs = dataset_inputs(binding.descriptor)
        source = binding.draft.source_signal
        if source and not input_specs:
            raise LayoutError(
                f"{binding.node_id}: {binding.descriptor.api_name} has no dataset input"
            )
        actual = output_contracts.get(source) if source else None
        if actual is not None and not any(
            spec.accepts(actual) for spec in input_specs
        ):
            expected = sorted(
                str(spec.contract_id)
                for spec in input_specs
                if spec.contract_id is not None
            )
            raise LayoutError(
                f"{binding.node_id}: {source!r} publishes {actual!r}, expected "
                f"{', '.join(expected) if expected else 'a compatible Dataset'}"
            )

    offered_kinds = {str(kind) for kind in panel_kinds}
    for index, state in enumerate(document.panels):
        if state.kind not in offered_kinds:
            raise LayoutError(
                f"panel {index + 1}: plot kind {state.kind!r} is not available"
            )
    return ResolvedLayout(tuple(bindings), document.panels, document.panel_ids)


def load_layout(
    resolved: ResolvedLayout,
    *,
    panel_ids: Sequence[str],
    schema_for: Callable[[str], object | None],
) -> LoadedLayout:
    """Put a resolved board onto fresh panel identities, on today's data.

    ``panel_ids`` are the identities the console minted for this load, in
    saved order.  Every ``@logic/<saved panel id>/<output>`` in the document
    -- a panel's signal, its overlay, a logic row's source -- is respelled to
    the fresh identity, because the producer behind such a signal is the
    loaded panel's own Bridge, and it publishes under the fresh one.  A
    reference to a panel the board does not carry cannot be resolved: it is
    dropped (the field left blank, the panel or row kept for rewiring) and
    said in ``notes``.

    ``schema_for(signal)`` is what that signal publishes today, or None.  A
    saved whole-domain repeat fate names an axis only the data can name, so
    it is expanded onto every Repeat axis of that schema; with no schema, or
    no Repeat axis, it is dropped and said.
    """

    fresh = tuple(str(panel_id).strip() for panel_id in panel_ids)
    if len(fresh) != len(resolved.panels):
        raise LayoutError("a load mints one fresh panel_id per saved panel")
    renamed = dict(zip(resolved.panel_ids, fresh, strict=True))
    notes: list[str] = []
    panels: list[PanelState] = []
    for state in resolved.panels:
        who = f"panel {state.title!r}"
        signal = _respelled_reference(state.signal, renamed, who, notes)
        overlay_signal = _respelled_reference(
            state.overlay_signal, renamed, f"{who} overlay", notes
        )
        panels.append(
            replace(
                state,
                signal=signal,
                overlay_signal=overlay_signal,
                semantic=_repeat_fate_on_todays_axes(
                    state.semantic, signal, schema_for, who, notes
                ),
            )
        )
    logic = tuple(
        replace(
            binding,
            draft=replace(
                binding.draft,
                source_signal=_respelled_reference(
                    binding.draft.source_signal,
                    renamed,
                    f"logic {binding.node_id}",
                    notes,
                ),
            ),
        )
        for binding in resolved.logic
    )
    return LoadedLayout(logic, tuple(panels), fresh, tuple(notes))


def _respelled_reference(
    reference: str,
    renamed: Mapping[str, str],
    who: str,
    notes: list[str],
) -> str:
    parts = split_signal_key(reference)
    if parts is None:
        return reference
    producer, output = parts
    if producer in renamed:
        return stable_signal_key(renamed[producer], output)
    if _names_a_panel(producer):
        notes.append(
            f"{who} read {reference}, a panel this board does not carry; "
            "the connection was dropped"
        )
        return ""
    return reference


def _repeat_fate_on_todays_axes(
    semantic: Mapping[str, Any],
    signal: str,
    schema_for: Callable[[str], object | None],
    who: str,
    notes: list[str],
) -> Mapping[str, Any]:
    if REPEAT_DOMAIN_FATE not in semantic:
        return semantic
    current = dict(semantic)
    fate = current.pop(REPEAT_DOMAIN_FATE)
    schema = schema_for(signal) if signal else None
    axes = () if schema is None else tuple(schema.repeat_domain.axes)
    if not axes:
        reason = (
            "it has no signal to name them"
            if not signal
            else f"nothing is publishing {signal} to name them"
            if schema is None
            else f"{signal} has no Repeat axis"
        )
        notes.append(
            f"{who}: its saved repeat fate {fate!r} names the Repeat axes and "
            f"{reason}; the fate was dropped"
        )
        return current
    for axis in axes:
        current.setdefault(fate_field_name(AxisRef.repeat(str(axis.axis_id))), fate)
    return current


def _mapping(value: object, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LayoutError(f"{where} must be an object")
    if any(not isinstance(name, str) for name in value):
        raise LayoutError(f"{where} field names must be strings")
    return value


def _sequence(value: object, where: str) -> Sequence[object]:
    if not isinstance(value, (list, tuple)):
        raise LayoutError(f"{where} must be a list")
    return value


def _exact_fields(value: Mapping[str, Any], expected: set[str], where: str) -> None:
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing:
        raise LayoutError(f"{where} is missing {sorted(missing)!r}")
    if unknown:
        raise LayoutError(f"{where} has unknown fields {sorted(unknown)!r}")


def _logic_from_tree(value: object, index: int) -> LogicLayoutEntry:
    where = f"logic entry {index + 1}"
    entry = _mapping(value, where)
    _exact_fields(
        entry,
        {
            "node_id",
            "api_name",
            "values",
            "source_signal",
            "device_keys",
            "artifact_inputs",
            "auto_preview",
        },
        where,
    )
    values = _mapping(entry["values"], f"{where} values")
    device_keys = _mapping(entry["device_keys"], f"{where} device_keys")
    artifact_inputs = _mapping(
        entry["artifact_inputs"], f"{where} artifact_inputs"
    )
    if any(not isinstance(key, str) for key in device_keys.values()):
        raise LayoutError(f"{where} device values must be strings")
    if any(not isinstance(path, str) for path in artifact_inputs.values()):
        raise LayoutError(f"{where} artifact input paths must be strings")
    return LogicLayoutEntry(
        _string(entry["node_id"], f"{where} node_id"),
        _string(entry["api_name"], f"{where} api_name"),
        dict(values),
        _string(entry["source_signal"], f"{where} source_signal"),
        dict(device_keys),
        dict(artifact_inputs),
        _boolean(entry["auto_preview"], f"{where} auto_preview"),
    )


def _panel_from_tree(value: object, index: int) -> tuple[str | None, PanelState]:
    where = f"panel entry {index + 1}"
    entry = dict(_mapping(value, where))
    panel_id = None
    if "panel_id" in entry:
        panel_id = _string(entry.pop("panel_id"), f"{where} panel_id").strip()
        if not panel_id:
            raise LayoutError(f"{where} panel_id must not be blank")
    if isinstance(entry.get("semantic"), Mapping):
        entry["semantic"] = _current_semantic(entry["semantic"], where)
    try:
        return panel_id, PanelState.from_document(entry)
    except Exception as error:
        raise LayoutError(f"{where}: {error}") from error


def _current_semantic(semantic: Mapping[str, Any], where: str) -> dict[str, Any]:
    current: dict[str, Any] = {}
    spelled: dict[str, str] = {}
    for key, value in semantic.items():
        try:
            name = current_fate_key(str(key))
        except LayoutError as error:
            raise LayoutError(f"{where}: {error}") from error
        if name in current:
            raise LayoutError(
                f"{where}: fate keys {spelled[name]!r} and {key!r} both mean {name!r}"
            )
        current[name] = value
        spelled[name] = str(key)
    return current


def _string(value: object, where: str) -> str:
    if not isinstance(value, str):
        raise LayoutError(f"{where} must be a string")
    return value


def _boolean(value: object, where: str) -> bool:
    if not isinstance(value, bool):
        raise LayoutError(f"{where} must be true or false")
    return value


__all__ = [
    "LAYOUT_FORMAT",
    "PANEL_ID_PREFIX",
    "REPEAT_DOMAIN_FATE",
    "LayoutDocument",
    "LayoutError",
    "LoadedLayout",
    "LogicLayoutEntry",
    "ResolvedLayout",
    "current_fate_key",
    "load_layout",
    "panel_id_for",
    "resolve_layout",
]
