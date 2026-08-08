"""The one persisted document for a stopped TaskConsole board.

The console owns composition, so its reusable pipeline belongs here rather
than in the device, plot, or UI packages.  This module is the only JSON shell:
the presenter deals in typed entries and asks this document to parse or write
them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from zlc_durable import write_readable_json

from .logic import (
    LogicBinding,
    LogicDraft,
    dataset_inputs,
    device_key_options,
    stable_signal_key,
)
from .panel_state import PanelState


LAYOUT_FORMAT = "zlc.console-board/v2"


class LayoutError(ValueError):
    """A file cannot describe one current TaskConsole board."""


@dataclass(frozen=True, slots=True)
class LogicLayoutEntry:
    node_id: str
    api_name: str
    values: Mapping[str, Any]
    source_signal: str
    device_keys: Mapping[str, str]

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

    def to_tree(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "api_name": self.api_name,
            "values": dict(self.values),
            "source_signal": self.source_signal,
            "device_keys": dict(self.device_keys),
        }


@dataclass(frozen=True, slots=True)
class LayoutDocument:
    """One canonical, ordered stopped pipeline and panel arrangement."""

    panels: tuple[PanelState, ...]
    logic: tuple[LogicLayoutEntry, ...]

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
        object.__setattr__(self, "panels", panels)
        object.__setattr__(self, "logic", logic)

    @classmethod
    def from_tree(cls, tree: object) -> "LayoutDocument":
        document = _mapping(tree, "layout")
        if document.get("format") != LAYOUT_FORMAT:
            raise LayoutError("that file is not a saved board")
        _exact_fields(document, {"format", "panels", "logic"}, "layout")
        panels = tuple(
            _panel_from_tree(entry, index)
            for index, entry in enumerate(_sequence(document["panels"], "panels"))
        )
        logic = tuple(
            _logic_from_tree(entry, index)
            for index, entry in enumerate(_sequence(document["logic"], "logic"))
        )
        return cls(panels, logic)

    @classmethod
    def read(cls, path: str | Path) -> "LayoutDocument":
        source = Path(path)
        return cls.from_tree(json.loads(source.read_text(encoding="utf-8")))

    def to_tree(self) -> dict[str, Any]:
        return {
            "format": LAYOUT_FORMAT,
            "panels": [state.document() for state in self.panels],
            "logic": [entry.to_tree() for entry in self.logic],
        }

    def write(self, path: str | Path) -> Path:
        return write_readable_json(path, self.to_tree())


@dataclass(frozen=True, slots=True)
class ResolvedLayout:
    """All stopped row drafts after catalog/schema/contract resolution."""

    logic: tuple[LogicBinding, ...]
    panels: tuple[PanelState, ...]


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
            values = descriptor.authoring_schema.freeze(entry.values)
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
        bindings.append(
            LogicBinding(
                entry.node_id,
                descriptor,
                LogicDraft(values, entry.source_signal, selected),
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
        expected = {str(spec.contract_id) for spec in dataset_inputs(binding.descriptor)}
        source = binding.draft.source_signal
        if source and not expected:
            raise LayoutError(
                f"{binding.node_id}: {binding.descriptor.api_name} has no dataset input"
            )
        actual = output_contracts.get(source) if source else None
        if actual is not None and actual not in expected:
            raise LayoutError(
                f"{binding.node_id}: {source!r} publishes {actual!r}, expected "
                f"{', '.join(sorted(expected))}"
            )

    offered_kinds = {str(kind) for kind in panel_kinds}
    for index, state in enumerate(document.panels):
        if state.kind not in offered_kinds:
            raise LayoutError(
                f"panel {index + 1}: plot kind {state.kind!r} is not available"
            )
    return ResolvedLayout(tuple(bindings), document.panels)


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
        {"node_id", "api_name", "values", "source_signal", "device_keys"},
        where,
    )
    values = _mapping(entry["values"], f"{where} values")
    device_keys = _mapping(entry["device_keys"], f"{where} device_keys")
    if any(not isinstance(key, str) for key in device_keys.values()):
        raise LayoutError(f"{where} device values must be strings")
    return LogicLayoutEntry(
        _string(entry["node_id"], f"{where} node_id"),
        _string(entry["api_name"], f"{where} api_name"),
        dict(values),
        _string(entry["source_signal"], f"{where} source_signal"),
        dict(device_keys),
    )


def _panel_from_tree(value: object, index: int) -> PanelState:
    where = f"panel entry {index + 1}"
    entry = _mapping(value, where)
    _exact_fields(
        entry,
        {
            "signal",
            "title",
            "kind",
            "size",
            "interval_ms",
            "semantic",
            "display",
            "fit",
            "site_overlay",
        },
        where,
    )
    try:
        return PanelState(
            signal=_string(entry["signal"], f"{where} signal"),
            title=_string(entry["title"], f"{where} title"),
            kind=_string(entry["kind"], f"{where} kind"),
            size=_string(entry["size"], f"{where} size"),
            interval_ms=_integer(entry["interval_ms"], f"{where} interval_ms"),
            semantic=dict(_mapping(entry["semantic"], f"{where} semantic")),
            display=dict(_mapping(entry["display"], f"{where} display")),
            fit=dict(_mapping(entry["fit"], f"{where} fit")),
            site_overlay=_string(entry["site_overlay"], f"{where} site_overlay"),
        )
    except Exception as error:
        raise LayoutError(f"{where}: {error}") from error


def _string(value: object, where: str) -> str:
    if not isinstance(value, str):
        raise LayoutError(f"{where} must be a string")
    return value


def _integer(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LayoutError(f"{where} must be an integer")
    return value


__all__ = [
    "LAYOUT_FORMAT",
    "LayoutDocument",
    "LayoutError",
    "LogicLayoutEntry",
    "ResolvedLayout",
    "resolve_layout",
]
