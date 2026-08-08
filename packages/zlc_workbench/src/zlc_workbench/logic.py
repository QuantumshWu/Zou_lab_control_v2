"""Running logic nodes from the console: measurements, processors, tasks.

A panel shows a signal.  A logic node is what PUBLISHES one -- the measurement
that fires the sequence and collects frames, the processor that turns frames
into occupancy, the task that calibrates.  The console had panels and no way to
start any of it, so every signal on screen had to be produced from a notebook.

Nothing about a node is decided here.  What nodes exist, what each one needs
given to it, and what a legal setting is are declared by zlc_atom, and hosting
one -- starting, cancelling, polling, publishing -- is zlc_runtime's NodeHost.
This binds one to the other and shows the result.

Binding is by declaration, not by name-guessing: a descriptor states which
devices it needs, which signal it reads, and which settings it takes, and each
Start build is handed exactly the arguments it declares out of those facts.
Before Start, the row is only an editable draft and may deliberately contain
an unresolved device, source, or artifact path.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import inspect
from typing import Any

from zlc_runtime import DatasetOutputDeclaration, NodeHost


__all__ = [
    "DeviceClaim",
    "LogicBinding",
    "LogicCandidate",
    "LogicCatalog",
    "LogicDraft",
    "artifact_input_specs",
    "build_arguments",
    "device_key_options",
    "stable_signal_key",
]


@dataclass
class LogicDraft:
    """The one editable authoring state owned by a TaskConsole row."""

    values: dict[str, Any] = field(default_factory=dict)
    source_signal: str = ""
    device_keys: dict[str, str] = field(default_factory=dict)
    artifact_inputs: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DeviceClaim:
    """One exact resolved device object claimed by a candidate run."""

    argument_name: str
    device_key: str
    device: object = field(compare=False)
    access: object


@dataclass
class LogicCandidate:
    """A fully built run waiting for any old exclusive owners to stop."""

    node: Any
    host: NodeHost
    claims: tuple[DeviceClaim, ...] = ()
    waiting_for: set[str] = field(default_factory=set)


@dataclass
class LogicBinding:
    """One stable TaskConsole row and its current or pending run.

    It used to carry its ROW as well -- a widget, held by the layer that is
    not allowed to hold one.  The row lives in the window now, and this side
    names the node instead.
    """

    node_id: str
    descriptor: Any
    draft: LogicDraft = field(default_factory=LogicDraft)
    host: NodeHost | None = None
    node: Any = None
    claims: tuple[DeviceClaim, ...] = ()
    pending: LogicCandidate | None = None
    draft_error: str = ""
    #: Successful declared artifact paths from the current host generation.
    artifact_results: tuple[Mapping[str, str], ...] = ()
    artifact_result_host: object | None = field(default=None, compare=False)
    artifact_completion_order: int = 0
    #: The last state pushed to the row, so an unchanged row is left alone.
    shown: tuple = ()
    #: Asked to go, and still stopping.  The row stays until it has: a node
    #: taken off screen while it still holds a camera is one nobody can reach.
    removing: bool = False


def stable_signal_key(node_id: str, output_name: str) -> str:
    """The stable signal spelling shared by stopped drafts and NodeHost."""

    return f"@logic/{str(node_id)}/{str(output_name)}"


def dataset_inputs(descriptor: Any) -> tuple[Any, ...]:
    """The live signals one node reads, as its descriptor declares them.

    A processor is built around a signal it consumes, and the runtime refuses
    to host a reactive node that was never told which one.  Whether to ask is
    therefore the descriptor's answer, not a guess from the node's kind.
    """

    from zlc_atom.nodes import DatasetInputSpec

    return tuple(
        spec
        for spec in getattr(descriptor, "input_specs", ())
        if isinstance(spec, DatasetInputSpec)
    )


def artifact_input_specs(descriptor: Any) -> tuple[Any, ...]:
    """Saved-file inputs one node reads, as its descriptor declares them."""

    from zlc_atom.nodes import ArtifactInputSpec

    return tuple(
        spec
        for spec in getattr(descriptor, "input_specs", ())
        if isinstance(spec, ArtifactInputSpec)
    )


def device_key_options(
    descriptor: Any,
    *,
    installation: Any,
) -> dict[str, tuple[str, ...]]:
    """Compatible installed keys for each declared build argument.

    Keys are sorted so the first option is the deterministic headless default.
    The argument name identifies where the resolved adapter goes; it is never
    assumed to be the installed device key.
    """

    devices = getattr(installation, "devices", {})
    if not isinstance(devices, Mapping):
        raise TypeError("installation.devices must be a mapping")
    options: dict[str, tuple[str, ...]] = {}
    for requirement in descriptor.device_requirements:
        compatible = tuple(
            sorted(
                str(key)
                for key, leaf in devices.items()
                if requirement.capability_token
                in getattr(leaf, "capabilities", {})
            )
        )
        options[requirement.argument_name] = compatible
    return options


def build_arguments(
    descriptor: Any,
    *,
    installation: Any,
    signal_plane: Any,
    values: Mapping[str, Any],
    source_signal: str = "",
    artifact_inputs: Mapping[str, str] | None = None,
    extras: Mapping[str, Any] | None = None,
    device_keys: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Everything one node's build asks for, out of what its descriptor declares.

    A build is handed what it names, and nothing else.  The alternative -- pass
    every fact and hope -- fails on the first build without ``**values``, which
    is most of them, and fails with a TypeError that names a keyword rather than
    the bench fact behind it.

    Raises when a declared device is not installed.  That refusal is the whole
    value of declaring it: "this bench has no sequencer" is an answer an
    operator can act on, and a row stuck at idle is not.
    """

    build = getattr(descriptor, "build", None)
    if build is None:
        raise TypeError(f"{descriptor.api_name} cannot be built")

    parameters = inspect.signature(build).parameters
    takes_anything = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    available: dict[str, Any] = {"signal_plane": signal_plane}
    selected_keys = dict(device_keys or {})
    options = device_key_options(descriptor, installation=installation)
    for requirement in descriptor.device_requirements:
        candidates = options[requirement.argument_name]
        if not candidates:
            raise LookupError(
                f"{descriptor.api_name} needs a {requirement.capability_token} "
                "and this apparatus has none"
            )
        selected = (
            str(selected_keys[requirement.argument_name])
            if requirement.argument_name in selected_keys
            else candidates[0]
        )
        if selected not in candidates:
            raise LookupError(
                f"{selected!r} does not provide {requirement.capability_token}; "
                f"choose one of {', '.join(candidates)}"
            )
        try:
            available[requirement.argument_name] = installation.capability(
                requirement.capability_token,
                key=selected,
            )
        except Exception as error:
            raise LookupError(
                f"{descriptor.api_name} could not use {selected!r} as its "
                f"{requirement.capability_token}: {error}"
            ) from error
        key_argument = f"{requirement.argument_name}_key"
        if key_argument in parameters:
            available[key_argument] = selected
    if source_signal:
        available["source_signal"] = str(source_signal)
    offered_artifacts = dict(artifact_inputs or {})
    declared_artifacts = artifact_input_specs(descriptor)
    unknown_artifacts = set(offered_artifacts) - {
        spec.name for spec in declared_artifacts
    }
    if unknown_artifacts:
        raise ValueError(
            f"{descriptor.api_name} has no artifact inputs "
            f"{sorted(unknown_artifacts)!r}"
        )
    for spec in declared_artifacts:
        if spec.name not in offered_artifacts:
            if spec.required:
                raise LookupError(
                    f"{descriptor.api_name} needs artifact input {spec.name}"
                )
            continue
        path = offered_artifacts[spec.name]
        if not isinstance(path, str):
            raise TypeError(f"artifact input {spec.name!r} must be a path string")
        if path.strip():
            available[spec.name] = path
        elif spec.required:
            raise LookupError(
                f"{descriptor.api_name} needs artifact input {spec.name}"
            )
    available.update(dict(extras or {}))

    arguments = {
        name: value for name, value in available.items() if name in parameters
    }
    authored = descriptor.authoring_schema.freeze(dict(values))
    for name, value in authored.items():
        if takes_anything or name in parameters:
            arguments[name] = value

    missing = [
        name
        for name, parameter in parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
        and name not in arguments
    ]
    if missing:
        # Read at Start, so name the repairable missing bench facts directly.
        raise LookupError(
            f"{descriptor.api_name} needs {', '.join(missing)}, "
            "which nothing on this bench has produced yet"
        )
    return arguments


class LogicCatalog:
    """What can be added, as the rows a chooser renders.

    The discovered descriptors, not a list kept here: a console offering its own
    menu of node types would drift from the ones that can actually be built, and
    the way that shows up is an operator picking something that then refuses.
    """

    def __init__(self, descriptors: Sequence[Any] | None = None) -> None:
        if descriptors is None:
            from zlc_atom.nodes import discover_logic_nodes

            descriptors = discover_logic_nodes()
        self.by_name = {item.api_name: item for item in descriptors}

    def rows(self) -> tuple[tuple[str, str, str], ...]:
        """(api_name, kind, what it publishes) for every node type."""

        return tuple(
            (
                name,
                str(getattr(item.kind, "value", item.kind)),
                ", ".join(output.name for output in item.outputs) or "nothing",
            )
            for name, item in sorted(self.by_name.items())
        )

    def get(self, api_name: str) -> Any | None:
        return self.by_name.get(str(api_name))


def make_host(
    descriptor: Any,
    node: Any,
    *,
    signal_plane: Any,
    instance_id: str,
    request_owner_wake: Callable[[], None] | None = None,
) -> NodeHost:
    """One node under the runtime's own lifecycle, named for its instance.

    The output names come from the descriptor, so what a node publishes is what
    it declared it would -- a host told a different set is a node whose signals
    nobody can find.
    """

    return NodeHost(
        node,
        signal_plane,
        request_owner_wake,
        instance_id=str(instance_id),
        dataset_output_declarations=tuple(
            DatasetOutputDeclaration(output.name, output.contract_id)
            for output in descriptor.outputs
        ),
    )
