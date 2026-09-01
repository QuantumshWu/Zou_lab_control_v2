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
from pathlib import Path
from types import MappingProxyType
from typing import Any

from zlc_runtime import NodeHost

from .device_use import DeviceClaim, DeviceLease, LogicReservation


__all__ = [
    "LogicBinding",
    "LogicCandidate",
    "LogicCatalog",
    "LogicDraft",
    "LogicDraftFinalization",
    "artifact_input_specs",
    "build_arguments",
    "device_key_options",
    "finalize_logic_draft",
    "stable_signal_key",
    "task_input_summary",
]


@dataclass
class LogicDraft:
    """The one editable authoring state owned by a TaskConsole row."""

    values: dict[str, Any] = field(default_factory=dict)
    source_signal: str = ""
    device_keys: dict[str, str] = field(default_factory=dict)
    artifact_inputs: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class LogicDraftFinalization:
    """One immutable derived answer to whether this exact raw draft may Start."""

    values: Mapping[str, Any]
    source_signal: str
    device_keys: Mapping[str, str]
    devices: Mapping[str, object]
    artifact_paths: Mapping[str, str]
    artifacts: Mapping[str, object]
    resources: Mapping[str, object]
    #: ``{field name: why not}`` for settings this draft's bound devices
    #: cannot take.  The form disables those controls and shows the reason;
    #: unavailable booleans project an effective ``False`` while other
    #: unavailable truthy values remain start issues.
    field_availability: Mapping[str, str]
    issues: tuple[str, ...]
    #: The named source signal simply is not on the plane (or has not
    #: published) yet.  For a processor that is not a start issue but a
    #: standing follow: the console waits and completes the Start when the
    #: signal appears.  An INCOMPATIBLE source -- present under another
    #: contract -- stays a hard issue; waiting would never fix it.
    source_absent: bool = False

    def __post_init__(self) -> None:
        for name in (
            "values",
            "device_keys",
            "devices",
            "artifact_paths",
            "artifacts",
            "resources",
            "field_availability",
        ):
            object.__setattr__(
                self,
                name,
                MappingProxyType(dict(getattr(self, name))),
            )
        object.__setattr__(self, "issues", tuple(self.issues))

    @property
    def can_start(self) -> bool:
        return not self.issues


@dataclass
class LogicCandidate:
    """A fully built run waiting for any old exclusive owners to stop."""

    node: Any
    host: NodeHost
    #: What this run declares it puts on screen, resolved from the same frozen
    #: values that built the node -- beside its outputs, which are resolved
    #: from those values too.
    previews: tuple = ()
    claims: tuple[DeviceClaim, ...] = ()
    reservation: LogicReservation | None = None
    run_root: Path | None = None
    input_summary: Mapping[str, object] = field(default_factory=dict)

    @property
    def waiting_for(self) -> tuple[str, ...]:
        reservation = self.reservation
        return () if reservation is None else reservation.waiting_for


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
    owner_token: object = field(default_factory=object, compare=False)
    lease: DeviceLease | None = field(default=None, compare=False)
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
    #: A started processor keeps FOLLOWING its source: absent signal means
    #: wait for it, a source restart means start again.  Only the operator's
    #: own Stop clears it -- their stop is a decision, a source restart is
    #: not.  Failures clear it too: silently retrying an error loop is how
    #: an error gets ignored.
    following: bool = False
    draft_revision: int = 0
    finalization_key: tuple = ()
    finalization: LogicDraftFinalization | None = None
    #: Open this node's declared preview panel when it starts.  A preference
    #: about THIS board, not a parameter of the measurement -- it lives here
    #: and travels with the saved layout, never in the authoring schema, so a
    #: notebook driving the same node never meets a GUI concept.
    auto_preview: bool = True
    #: Declared previews already answered for the run below, so an operator who
    #: closes one gets to keep it closed until the node is started again.
    previewed: tuple[str, ...] = ()
    preview_host: object | None = field(default=None, compare=False)
    #: The previews the RUNNING node declared, frozen when it started.  What
    #: an operator types afterwards is a draft for the next run, and a draft
    #: mid-edit is not even a valid request.
    preview_specs: tuple = ()
    #: Request already projected into a modal, preventing nested event-loop duplicates.
    operator_request_id: str = ""


def stable_signal_key(node_id: str, output_name: str) -> str:
    """The stable signal spelling shared by stopped drafts and NodeHost."""

    return f"@logic/{str(node_id)}/{str(output_name)}"


def task_input_summary(
    descriptor: Any,
    finalization: LogicDraftFinalization,
) -> dict[str, object]:
    """Descriptor-owned Start facts, without device objects or live data."""

    artifacts = {
        spec.name: {
            "contract_id": str(spec.contract_id),
            "path": str(finalization.artifacts[spec.name].path),
        }
        for spec in artifact_input_specs(descriptor)
        if spec.name in finalization.artifacts
    }
    resources = {
        spec.field_name: {
            "contract_id": str(spec.contract_id),
            "path": str(finalization.resources[spec.field_name].path),
        }
        for spec in descriptor.workspace_resources
        if spec.field_name in finalization.resources
    }
    summary = {
        "authored": dict(finalization.values),
        "source_signal": finalization.source_signal or None,
        "devices": {
            requirement.argument_name: {
                "instance_id": finalization.device_keys[requirement.argument_name],
                "capability": requirement.capability_token,
            }
            for requirement in descriptor.device_requirements
        },
        "artifacts": artifacts,
        "resources": resources,
    }
    return summary


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


def finalize_logic_draft(
    descriptor: Any,
    draft: LogicDraft,
    *,
    installation: Any,
    signal_plane: Any,
    workspace: Any,
    source_options: Sequence[str] = (),
) -> LogicDraftFinalization:
    """Resolve every Start admission fact without building or acquiring a run."""

    from zlc_atom.nodes import ResolvedArtifact, ResolvedWorkspaceResource

    if not isinstance(draft, LogicDraft):
        raise TypeError("finalize_logic_draft needs LogicDraft")
    issues: list[str] = []
    authored = True
    try:
        values = descriptor.authoring_schema.project_values(draft.values)
    except Exception as error:
        values = {}
        authored = False
        issues.append(str(error))

    options = device_key_options(descriptor, installation=installation)
    declared_device_arguments = {
        requirement.argument_name
        for requirement in descriptor.device_requirements
    }
    unknown_device_arguments = set(draft.device_keys) - declared_device_arguments
    if unknown_device_arguments:
        issues.append(
            f"{descriptor.api_name} has no device inputs "
            f"{sorted(unknown_device_arguments)!r}"
        )
    selected_devices: dict[str, str] = {}
    devices: dict[str, object] = {}
    for requirement in descriptor.device_requirements:
        argument = str(requirement.argument_name)
        candidates = options[argument]
        selected = str(draft.device_keys.get(argument, "")).strip()
        selected_devices[argument] = selected
        if not candidates:
            issues.append(
                f"{descriptor.api_name} needs a {requirement.capability_token} "
                "and this apparatus has none"
            )
            continue
        if selected not in candidates:
            issues.append(
                f"{selected or '(not selected)'!r} does not provide "
                f"{requirement.capability_token}; choose one of "
                f"{', '.join(candidates)}"
            )
            continue
        try:
            devices[argument] = installation.capability(
                requirement.capability_token,
                key=selected,
            )
        except Exception as error:
            issues.append(
                f"{descriptor.api_name} could not use {selected!r} as its "
                f"{requirement.capability_token}: {error}"
            )

    # What the bound devices cannot do, asked once the devices are known.  A
    # setting like "read this camera in photoelectrons" cannot be checked
    # against the schema alone.  An unavailable boolean has one neutral
    # effective value, False; the raw draft stays untouched so selecting a
    # capable device restores its authored/default value.
    field_availability: dict[str, str] = {}
    resolve_availability = getattr(descriptor, "resolve_field_availability", None)
    if (
        resolve_availability is not None
        and authored
        and len(devices) == len(declared_device_arguments)
    ):
        field_availability = {
            str(name): str(reason)
            for name, reason in dict(resolve_availability(devices)).items()
            if str(reason).strip()
        }
        unknown_fields = set(field_availability) - set(
            descriptor.authoring_schema.field_names
        )
        if unknown_fields:
            raise ValueError(
                f"{descriptor.api_name} resolves availability for undeclared "
                f"fields {sorted(unknown_fields)!r}"
            )
        fields = {
            field.name: field for field in descriptor.authoring_schema.fields
        }
        for name, reason in field_availability.items():
            if fields[name].value_type == "bool":
                values[name] = False
            elif values.get(name):
                issues.append(reason)

    wants_source = dataset_inputs(descriptor)
    source = str(draft.source_signal).strip()
    compatible_sources = tuple(str(value) for value in source_options)
    source_absent = False
    processor_kind = getattr(descriptor.kind, "value", None) == "processor"
    if wants_source:
        if not source:
            issues.append("source_signal must be selected")
        elif source not in compatible_sources:
            declared_elsewhere = any(
                str(row.name) == source
                for row in signal_plane.describe_signals()
            )
            if declared_elsewhere or not processor_kind:
                contracts = ", ".join(
                    str(spec.contract_id)
                    for spec in wants_source
                    if spec.contract_id is not None
                ) or "a compatible"
                issues.append(
                    f"{source!r} is not declared as {contracts} Dataset"
                )
            else:
                # Nothing on the plane under that name yet: a processor
                # follows, it does not fail.
                source_absent = True
        elif processor_kind:
            if signal_plane.latest_publication(
                source
            ) is None and not signal_plane.is_generation_live(source):
                # Nothing published and nothing armed: a processor follows,
                # it does not fail.  An ARMED silent source starts now.
                source_absent = True
        elif not signal_plane.is_generation_live(source):
            # A measurement watches FUTURE publications, so what it needs is
            # an armed producer, not an existing Dataset: an externally
            # triggered chain publishes nothing until the measurement's own
            # pulse fires the first trigger.
            issues.append(
                f"{source!r} has no armed producer -- start the measurement "
                "chain that publishes it (its pulse may stay stopped)"
            )
    elif source:
        issues.append(f"{descriptor.api_name} has no Dataset source input")

    offered_artifacts = dict(draft.artifact_inputs)
    artifact_specs = artifact_input_specs(descriptor)
    declared_artifacts = {spec.name for spec in artifact_specs}
    unknown_artifacts = set(offered_artifacts) - declared_artifacts
    if unknown_artifacts:
        issues.append(
            f"{descriptor.api_name} has no artifact inputs "
            f"{sorted(unknown_artifacts)!r}"
        )
    artifact_paths: dict[str, str] = {}
    artifacts: dict[str, ResolvedArtifact] = {}
    data_root = Path(getattr(workspace, "data", Path.cwd())).resolve()
    for spec in artifact_specs:
        raw = offered_artifacts.get(spec.name, "")
        if not isinstance(raw, str):
            issues.append(f"artifact input {spec.name!r} must be a path string")
            continue
        text = raw.strip()
        if not text:
            if spec.required:
                issues.append(
                    f"{descriptor.api_name} needs artifact input {spec.name}"
                )
            continue
        selected_path = Path(text).expanduser()
        if not selected_path.is_absolute():
            selected_path = data_root / selected_path
        try:
            resolved = spec.codec.resolve(selected_path)
        except Exception as error:
            issues.append(
                f"{spec.contract_id} artifact {text!r} is invalid: {error}"
            )
            continue
        artifact_paths[spec.name] = str(resolved.path)
        artifacts[spec.name] = resolved

    resources: dict[str, ResolvedWorkspaceResource] = {}
    workspace_root = Path(getattr(workspace, "root", Path.cwd())).resolve()
    for spec in descriptor.workspace_resources:
        directory = (workspace_root / spec.directory).resolve()
        if directory.parent != workspace_root:
            raise ValueError("workspace resource directory escaped workspace root")
        raw_value = draft.values.get(spec.field_name, "")
        selected_text = str(raw_value).strip()
        selected = Path(selected_text).expanduser()
        if not selected.is_absolute():
            selected = directory / selected
        selected = selected.resolve()
        if not selected_text or selected.parent != directory:
            issues.append(
                f"{spec.field_name} must choose a file from {directory}"
            )
            continue
        try:
            resources[spec.field_name] = spec.resolve(selected)
        except Exception as error:
            issues.append(
                f"{spec.contract_id} resource {str(selected)!r} is invalid: {error}"
            )

    return LogicDraftFinalization(
        dict(values),
        source,
        selected_devices,
        devices,
        artifact_paths,
        artifacts,
        resources,
        field_availability,
        tuple(dict.fromkeys(str(issue) for issue in issues if str(issue))),
        source_absent=source_absent,
    )


def build_arguments(
    descriptor: Any,
    *,
    signal_plane: Any,
    finalization: LogicDraftFinalization,
    extras: Mapping[str, Any] | None = None,
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
    if not isinstance(finalization, LogicDraftFinalization):
        raise TypeError("build_arguments needs LogicDraftFinalization")
    if not finalization.can_start:
        raise ValueError(
            f"{descriptor.api_name} draft is not startable: "
            f"{'; '.join(finalization.issues)}"
        )

    parameters = inspect.signature(build).parameters
    takes_anything = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    available: dict[str, Any] = {"signal_plane": signal_plane}
    for requirement in descriptor.device_requirements:
        selected = str(finalization.device_keys[requirement.argument_name])
        available[requirement.argument_name] = finalization.devices[
            requirement.argument_name
        ]
        key_argument = f"{requirement.argument_name}_key"
        if key_argument in parameters:
            available[key_argument] = selected
    if finalization.source_signal:
        available["source_signal"] = finalization.source_signal
    for spec in artifact_input_specs(descriptor):
        resolved = finalization.artifacts.get(spec.name)
        if resolved is not None:
            available[spec.argument_name] = resolved
    for spec in descriptor.workspace_resources:
        resolved = finalization.resources.get(spec.field_name)
        if resolved is not None:
            available[spec.argument_name] = resolved
    extra_values = dict(extras or {})
    extra_collisions = sorted(
        set(extra_values) & set(descriptor.build_argument_names)
    )
    if extra_collisions:
        raise ValueError(
            f"{descriptor.api_name} bench extras collide with declared build "
            f"arguments: {extra_collisions!r}"
        )
    available.update(extra_values)

    arguments = {
        name: value for name, value in available.items() if name in parameters
    }
    for name, value in finalization.values.items():
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
                ", ".join(
                    output.name for output in item.outputs
                ) or "nothing",
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
    source_signal: str | None,
    request_owner_wake: Callable[[], None] | None = None,
) -> NodeHost:
    """One node under the runtime's own lifecycle, named for its instance.

    The descriptor's frozen output declarations are the sole signal vocabulary.
    """

    inputs = dataset_inputs(descriptor)
    if len(inputs) > 1:
        raise ValueError("NodeHost supports exactly one declared Dataset input")
    kind = str(getattr(descriptor.kind, "value", descriptor.kind))
    has_input = bool(inputs)
    selected_source = (
        str(source_signal or "").strip() if has_input else None
    )
    return NodeHost(
        node,
        signal_plane,
        request_owner_wake,
        instance_id=str(instance_id),
        kind=kind,
        dataset_output_declarations=tuple(descriptor.outputs),
        input_signal=selected_source,
        input_name=(
            inputs[0].name if has_input and kind == "processor" else None
        ),
        input_siblings=(
            inputs[0].sibling_outputs
            if has_input and kind == "processor"
            else ()
        ),
        input_delivery=(
            str(inputs[0].delivery) if has_input else None
        ),
        required_artifacts={
            output.name: output.contract_id
            for output in descriptor.artifact_outputs
        },
        task_name=str(descriptor.api_name),
    )
