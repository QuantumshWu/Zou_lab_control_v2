"""Data-only logic-node declarations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable

from zlc_atom.authoring import AuthoringSchema
from zlc_atom.install.descriptors import CAPABILITY_TYPES
from zlc_runtime import DatasetOutputDeclaration, SelectionState


class NodeKind(str, Enum):
    """What a node IS to the experiment: how it is layered, not how it runs."""

    MEASUREMENT = "measurement"
    TASK = "task"
    PROCESSOR = "processor"


@dataclass(frozen=True)
class DatasetInputSpec:
    name: str
    contract_id: str | None
    delivery: str
    #: The operator selects a producer's atomic output bundle; one member
    #: remains the existing Runtime subscription anchor, not a second input.
    select_bundle: bool = False

    def __post_init__(self) -> None:
        if not self.name or (self.contract_id is not None and not self.contract_id):
            raise ValueError("dataset input requires a name and a contract or None")
        delivery = str(self.delivery).strip()
        if delivery not in {"exact", "latest"}:
            raise ValueError("dataset input delivery must be 'exact' or 'latest'")
        object.__setattr__(self, "delivery", delivery)

    def accepts(self, contract_id: str | None) -> bool:
        return contract_id is not None and (
            self.contract_id is None or contract_id == self.contract_id
        )


@dataclass(frozen=True)
class ResolvedArtifact:
    """One exact file and the typed value decoded from those exact bytes."""

    path: Path
    contract_id: str
    value: object


@dataclass(frozen=True)
class ArtifactCodec:
    """The one file contract used to choose and validate a saved artifact."""

    contract_id: str
    file_filter: str
    suffixes: tuple[str, ...]
    decode: Callable[[Path], object]

    def __post_init__(self) -> None:
        contract_id = str(self.contract_id).strip()
        file_filter = str(self.file_filter).strip()
        suffixes = tuple(str(value).strip().lower() for value in self.suffixes)
        if not contract_id or not file_filter or not suffixes:
            raise ValueError(
                "artifact codec requires contract_id, file_filter, and suffixes"
            )
        if any(not value.startswith(".") for value in suffixes):
            raise ValueError("artifact codec suffixes must begin with '.'")
        if len(set(suffixes)) != len(suffixes):
            raise ValueError("artifact codec suffixes must be unique")
        if not callable(self.decode):
            raise TypeError("artifact codec decode must be callable")
        object.__setattr__(self, "contract_id", contract_id)
        object.__setattr__(self, "file_filter", file_filter)
        object.__setattr__(self, "suffixes", suffixes)

    def resolve(self, path: str | Path) -> ResolvedArtifact:
        source = Path(path).expanduser().resolve()
        if source.suffix.lower() not in self.suffixes:
            raise ValueError(
                f"{source.name!r} must use one of {self.suffixes!r}"
            )
        if not source.is_file():
            raise FileNotFoundError(source)
        return ResolvedArtifact(source, self.contract_id, self.decode(source))


@dataclass(frozen=True)
class ArtifactInputSpec:
    """One explicit saved-artifact path consumed by a run."""

    name: str
    label: str
    codec: ArtifactCodec
    required: bool = True
    argument_name: str = ""
    #: The run may take one of these PER FRAME of its Dataset input's cycle,
    #: beside the plain path every frame falls back to: an occupancy reads a
    #: load frame and a readout frame with calibrations trained on each.  The
    #: draft keys a frame's own path ``"<name>[<frame>]"`` (frames counted
    #: from 1, as the operator sees them) and the build receives them as
    #: ``<argument_name>_by_frame``, a mapping from that frame number.
    per_frame: bool = False

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        label = str(self.label).strip()
        if not name or not label:
            raise ValueError("artifact input requires name and label")
        if not isinstance(self.codec, ArtifactCodec):
            raise TypeError("artifact input codec must be ArtifactCodec")
        argument_name = str(self.argument_name).strip() or str(self.name).strip()
        if not argument_name:
            raise ValueError("artifact input requires a build argument name")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "argument_name", argument_name)
        object.__setattr__(self, "per_frame", bool(self.per_frame))

    @property
    def contract_id(self) -> str:
        return self.codec.contract_id

    @property
    def frame_argument_name(self) -> str:
        """The build argument carrying the per-frame artifacts, by frame number."""

        return f"{self.argument_name}_by_frame"


def artifact_input_key(name: str, frame: int) -> str:
    """The draft key of one frame's artifact: ``"<name>[<frame>]"``, frames from 1."""

    number = int(frame)
    if number < 1:
        raise ValueError("artifact frames are counted from 1")
    return f"{str(name).strip()}[{number}]"


def split_artifact_input_key(key: str) -> tuple[str, int | None]:
    """A draft key back into ``(artifact name, frame number)``; a plain key has no frame.

    A bracketed key that is not a frame number from 1 is refused: a key
    either is an artifact's own name or names one of its frames.
    """

    text = str(key).strip()
    if not text.endswith("]") or "[" not in text:
        return text, None
    name, _bracket, rest = text[:-1].rpartition("[")
    if not name or not rest.isdigit() or int(rest) < 1:
        raise ValueError(f"artifact input key {text!r} does not name a frame from 1")
    return name, int(rest)


@dataclass(frozen=True)
class ResolvedWorkspaceResource:
    """One selected workspace file and its descriptor-decoded typed value."""

    path: Path
    contract_id: str
    value: object


@dataclass(frozen=True)
class WorkspaceResourceSpec:
    """One plain file chosen from a descriptor-owned workspace collection."""

    field_name: str
    contract_id: str
    directory: str
    suffixes: tuple[str, ...]
    decode: Callable[[Path], object]
    argument_name: str = ""

    def __post_init__(self) -> None:
        field_name = str(self.field_name).strip()
        contract_id = str(self.contract_id).strip()
        directory = str(self.directory).strip()
        suffixes = tuple(str(value).strip().lower() for value in self.suffixes)
        if (
            not field_name
            or not contract_id
            or not directory
            or Path(directory).name != directory
            or not suffixes
        ):
            raise ValueError(
                "workspace resource requires field, contract, plain directory, and suffixes"
            )
        if any(not value.startswith(".") for value in suffixes):
            raise ValueError("workspace resource suffixes must begin with '.'")
        if len(set(suffixes)) != len(suffixes):
            raise ValueError("workspace resource suffixes must be unique")
        if not callable(self.decode):
            raise TypeError("workspace resource decode must be callable")
        argument_name = str(self.argument_name).strip() or field_name
        if not argument_name:
            raise ValueError("workspace resource requires a build argument name")
        object.__setattr__(self, "field_name", field_name)
        object.__setattr__(self, "contract_id", contract_id)
        object.__setattr__(self, "directory", directory)
        object.__setattr__(self, "suffixes", suffixes)
        object.__setattr__(self, "argument_name", argument_name)

    def resolve(self, path: str | Path) -> ResolvedWorkspaceResource:
        source = Path(path).expanduser().resolve()
        if source.suffix.lower() not in self.suffixes or not source.is_file():
            raise ValueError(
                f"workspace resource must be an existing {self.suffixes!r} file"
            )
        return ResolvedWorkspaceResource(
            source,
            self.contract_id,
            self.decode(source),
        )


@dataclass(frozen=True)
class ArtifactOutputSpec:
    """One saved-artifact path attribute exposed by semantic contract."""

    name: str
    contract_id: str

    def __post_init__(self) -> None:
        if not self.name or not self.contract_id:
            raise ValueError("artifact output requires name and contract_id")


@dataclass(frozen=True, slots=True)
class NodePreviewSpec:
    """UI-neutral request to preview one typed output declaration.

    A node with eight outputs knows which one an operator came to watch, and
    nothing else does.  The same is true of HOW it is watched: the plotting
    package can see the shape of a dataset but not its physics.  Three frames
    of a calibration cycle are three point rows of an image to it, and
    averaging them is a perfectly sensible thing to do with three point rows
    -- but they are a long reference, a short readout and a long reference,
    and the only reason to look at them is side by side.  The node states the
    kind because the node is what knows that.  ``producer`` is empty for the
    node's own output; a plain suffix names a stable companion producer owned
    by the same run.  ``semantic`` is the existing plot projection assignment,
    shared unchanged with any estimator that consumes the same publication.
    ``overlay`` optionally names a distinct output from that same producer.
    """

    output: DatasetOutputDeclaration
    plot_kind: str
    semantic: Mapping[str, object] = field(default_factory=dict)
    producer: str = ""
    overlay: DatasetOutputDeclaration | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.output, DatasetOutputDeclaration):
            raise TypeError("node preview output must be DatasetOutputDeclaration")
        plot_kind = str(self.plot_kind).strip()
        if not plot_kind:
            raise ValueError("node preview requires a plot kind")
        if not isinstance(self.semantic, Mapping):
            raise TypeError("node preview semantic assignment must be a mapping")
        semantic = dict(self.semantic)
        if any(not isinstance(name, str) or not name.strip() for name in semantic):
            raise TypeError("node preview semantic names must be non-empty text")
        producer = str(self.producer).strip()
        if producer and ("/" in producer or "\\" in producer):
            raise ValueError("node preview producer must be a plain owner suffix")
        overlay = self.overlay
        if overlay is not None and not isinstance(
            overlay, DatasetOutputDeclaration
        ):
            raise TypeError(
                "node preview overlay must be DatasetOutputDeclaration or None"
            )
        if overlay is not None and overlay.name == self.output.name:
            raise ValueError("node preview overlay must differ from its primary output")
        object.__setattr__(self, "plot_kind", plot_kind)
        object.__setattr__(self, "semantic", MappingProxyType(semantic))
        object.__setattr__(self, "producer", producer)


@dataclass(frozen=True)
class DeviceRequirement:
    capability_token: str
    argument_name: str
    #: The device fields this node drives for the run, which nothing else may
    #: move meanwhile.  Named, when the node knows them; ``None`` when the
    #: run freezes every tunable field the bound device declares -- a waveform
    #: capture takes the whole instrument as it stands, whatever knobs that
    #: instrument happens to have.
    protected_fields: tuple[str, ...] | None = ()

    def __post_init__(self) -> None:
        token = str(self.capability_token).strip()
        argument = str(self.argument_name).strip()
        if not token or not argument:
            raise ValueError(
                "device requirement token and build argument name must be non-empty"
            )
        object.__setattr__(self, "capability_token", token)
        object.__setattr__(self, "argument_name", argument)
        if self.protected_fields is None:
            return
        protected = tuple(str(value).strip() for value in self.protected_fields)
        if any(not value for value in protected):
            raise ValueError("protected device fields must be non-empty text")
        if len(set(protected)) != len(protected):
            raise ValueError("protected device fields must be unique")
        object.__setattr__(self, "protected_fields", protected)

    def fields_frozen_by(self, device: object) -> tuple[str, ...]:
        """The fields a run on ``device`` freezes: the named ones, or all it declares."""

        if self.protected_fields is not None:
            return self.protected_fields
        declare = getattr(device, "tunable_fields", None)
        if not callable(declare):
            return ()
        return tuple(str(field.metadata.name) for field in declare())


@dataclass(frozen=True)
class ResolvedDeviceClaim:
    """One runtime-selected device and the fields this node will drive."""

    device_key: str
    device: object = field(compare=False)
    protected_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        key = str(self.device_key).strip()
        protected = tuple(str(value).strip() for value in self.protected_fields)
        if not key or not protected or any(not value for value in protected):
            raise ValueError(
                "resolved device claim needs a key and protected fields"
            )
        if len(set(protected)) != len(protected):
            raise ValueError("resolved device claim fields must be unique")
        object.__setattr__(self, "device_key", key)
        object.__setattr__(self, "protected_fields", protected)


@dataclass(frozen=True)
class SelectionMapping:
    """Data-only translation from one semantic selection to a draft patch."""

    plot_kind: str
    selector_kind: str
    draft_fields: tuple[str, ...]
    #: Answers None when the region names nothing this producer can be set
    #: from -- a box drawn off the sensor is a place, not a crop.  "Nothing
    #: to set" is an answer; expressed as an exception it escaped a worker
    #: mid-gesture and the gesture never replied.
    map_patch: Callable[
        [SelectionState, Mapping[str, Any], Mapping[str, Any]],
        Mapping[str, Any] | None,
    ]

    def __post_init__(self) -> None:
        fields = tuple(self.draft_fields)
        if not self.plot_kind or not self.selector_kind:
            raise ValueError("selection mapping requires plot_kind and selector_kind")
        if not fields or any(not isinstance(value, str) or not value for value in fields):
            raise ValueError("selection mapping requires non-empty draft fields")
        if len(set(fields)) != len(fields):
            raise ValueError("selection mapping draft fields must be unique")
        if not callable(self.map_patch):
            raise TypeError("selection mapping map_patch must be callable")
        object.__setattr__(self, "draft_fields", fields)


@dataclass(frozen=True)
class LogicNodeDescriptor:
    """Closed declaration discovered from one leaf module."""

    api_name: str
    kind: NodeKind
    authoring_schema: AuthoringSchema
    input_specs: tuple[DatasetInputSpec | ArtifactInputSpec, ...] = ()
    outputs: tuple[DatasetOutputDeclaration, ...] = ()
    #: For a node whose outputs are named by what it is asked to compute or
    #: by the instrument it is bound to -- a derive publishes the lines of
    #: its program, a waveform measurement the quantities its source carries
    #: -- the declarations of one authored draft, called with the draft's
    #: values and its resolved devices by argument name (a device the draft
    #: has not bound yet is absent).  ``outputs`` is then empty: a node
    #: declares its outputs once, by name or by draft.
    declare_outputs: (
        Callable[
            [Mapping[str, Any], Mapping[str, object]],
            tuple[DatasetOutputDeclaration, ...],
        ]
        | None
    ) = None
    device_requirements: tuple[DeviceRequirement, ...] = ()
    build: Callable[..., object] | None = None
    node_previews: tuple[NodePreviewSpec, ...] | None = None
    #: The previews of one authored draft, for a node whose outputs are
    #: declared by draft: which of them an operator came to watch, and how,
    #: is then also the draft's answer.  Called like ``declare_outputs``.
    declare_previews: (
        Callable[
            [Mapping[str, Any], Mapping[str, object]],
            tuple[NodePreviewSpec, ...],
        ]
        | None
    ) = None
    artifact_outputs: tuple[ArtifactOutputSpec, ...] = ()
    ui_contributions: tuple[object, ...] = ()
    selection_mappings: tuple[SelectionMapping, ...] = ()
    workspace_resources: tuple[WorkspaceResourceSpec, ...] = ()
    #: ``{field name: why not}`` for the settings this bench cannot take --
    #: a fact about the DEVICES this draft has bound, not about the node.
    #: Called with the resolved devices by argument name once a draft is
    #: finalized.  One answer disables the control; an unavailable boolean is
    #: effectively False, while a truthy unavailable non-boolean refuses Start.
    resolve_field_availability: (
        Callable[[Mapping[str, object]], Mapping[str, str]] | None
    ) = None
    #: Values the node would choose for authoring fields the operator left
    #: empty, given the workspace resources the draft resolved to -- a
    #: calibration's three API fields, from the pulse it was given.  Called
    #: with the raw draft values and the resolved resources by field name.
    #: Only EMPTY fields take what it returns, so a choice the operator made
    #: stands; and it is asked at every finalization, so the defaults follow
    #: the resource rather than being copied into the draft once.
    resolve_defaults: (
        Callable[[Mapping[str, object], Mapping[str, object]], Mapping[str, object]]
        | None
    ) = None
    #: Optional text authoring field naming a readiness-reporting Logic instance.
    acquisition_input: str = ""
    #: The node reports acquisition readiness through its host execution context.
    reports_ready: bool = False
    build_argument_names: tuple[str, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )

    @property
    def offers_a_preview(self) -> bool:
        """Whether Start can put a declared output on screen."""

        return bool(self.node_previews) or self.declare_previews is not None

    def outputs_for(
        self, values: Mapping[str, Any], devices: Mapping[str, object]
    ) -> tuple[DatasetOutputDeclaration, ...]:
        """What one authored draft publishes, on the devices it has bound."""

        if self.declare_outputs is None:
            return self.outputs
        declared = tuple(self.declare_outputs(values, devices))
        if any(not isinstance(value, DatasetOutputDeclaration) for value in declared):
            raise TypeError("declare_outputs must return DatasetOutputDeclaration values")
        if len({value.name for value in declared}) != len(declared):
            raise ValueError("output names must be unique")
        return declared

    def previews_for(
        self, values: Mapping[str, Any], devices: Mapping[str, object]
    ) -> tuple[NodePreviewSpec, ...]:
        """What one authored draft puts on screen when it starts."""

        if self.declare_previews is None:
            return tuple(self.node_previews or ())
        previews = tuple(self.declare_previews(values, devices))
        if any(not isinstance(value, NodePreviewSpec) for value in previews):
            raise TypeError("declare_previews must return NodePreviewSpec values")
        declared = {output.name for output in self.outputs_for(values, devices)}
        unknown = {
            declaration.name
            for value in previews
            if not value.producer
            for declaration in (value.output, value.overlay)
            if declaration is not None and declaration.name not in declared
        }
        if unknown:
            raise ValueError(f"draft previews use undeclared outputs: {sorted(unknown)}")
        return previews

    def __post_init__(self) -> None:
        if not self.api_name or not isinstance(self.kind, NodeKind):
            raise ValueError("logic node requires api_name and a valid kind")
        if not isinstance(self.authoring_schema, AuthoringSchema):
            raise TypeError("authoring_schema must be AuthoringSchema")
        if not isinstance(self.acquisition_input, str):
            raise TypeError("acquisition_input must be a field name string")
        if type(self.reports_ready) is not bool:
            raise TypeError("reports_ready must be bool")
        if self.acquisition_input:
            selected = next((field for field in self.authoring_schema.fields
                             if field.name == self.acquisition_input), None)
            if selected is None or selected.value_type not in {"str", "text"}:
                raise ValueError("acquisition_input must name an existing text authoring field")
        inputs = tuple(self.input_specs)
        outputs = tuple(self.outputs)
        if self.kind is NodeKind.TASK and self.node_previews is None:
            raise ValueError(
                "a Task must explicitly declare node_previews, using () when it has none"
            )
        node_previews = (
            () if self.node_previews is None else tuple(self.node_previews)
        )
        artifact_outputs = tuple(self.artifact_outputs)
        requirements = tuple(self.device_requirements)
        selection_mappings = tuple(self.selection_mappings)
        workspace_resources = tuple(self.workspace_resources)
        if any(not isinstance(value, (DatasetInputSpec, ArtifactInputSpec)) for value in inputs):
            raise TypeError("input_specs contain an unsupported input type")
        if any(not isinstance(value, DatasetOutputDeclaration) for value in outputs):
            raise TypeError("outputs must contain DatasetOutputDeclaration values")
        if any(not isinstance(value, NodePreviewSpec) for value in node_previews):
            raise TypeError("node_previews must contain NodePreviewSpec values")
        if self.resolve_field_availability is not None and not callable(
            self.resolve_field_availability
        ):
            raise TypeError("resolve_field_availability must be callable or None")
        if self.resolve_defaults is not None and not callable(self.resolve_defaults):
            raise TypeError("resolve_defaults must be callable or None")
        if any(not isinstance(value, ArtifactOutputSpec) for value in artifact_outputs):
            raise TypeError("artifact_outputs must contain ArtifactOutputSpec values")
        if any(not isinstance(value, DeviceRequirement) for value in requirements):
            raise TypeError("device_requirements must contain DeviceRequirement values")
        if any(not isinstance(value, SelectionMapping) for value in selection_mappings):
            raise TypeError("selection_mappings must contain SelectionMapping values")
        if any(
            not isinstance(value, WorkspaceResourceSpec)
            for value in workspace_resources
        ):
            raise TypeError(
                "workspace_resources must contain WorkspaceResourceSpec values"
            )
        unknown_requirements = {value.capability_token for value in requirements} - set(CAPABILITY_TYPES)
        if unknown_requirements:
            raise ValueError(f"logic node uses unknown capability tokens: {sorted(unknown_requirements)}")
        if len({value.name for value in inputs}) != len(inputs):
            raise ValueError("input names must be unique")
        if len({value.name for value in outputs}) != len(outputs):
            raise ValueError("output names must be unique")
        if self.declare_outputs is not None:
            if not callable(self.declare_outputs):
                raise TypeError("declare_outputs must be callable or None")
            if outputs:
                raise ValueError(
                    "a node declares its outputs once: by name, or by what "
                    "its draft is asked to compute"
                )
        if self.declare_previews is not None:
            if not callable(self.declare_previews):
                raise TypeError("declare_previews must be callable or None")
            if self.declare_outputs is None or node_previews:
                raise ValueError(
                    "a node declares its previews once, and by draft only "
                    "when its outputs are"
                )
        preview_keys = tuple(
            (value.producer, value.output.name) for value in node_previews
        )
        if len(set(preview_keys)) != len(preview_keys):
            raise ValueError("node preview producer/output pairs must be unique")
        unknown_previews = {
            declaration.name
            for value in node_previews
            if not value.producer
            for declaration in (value.output, value.overlay)
            if declaration is not None
            and not any(declaration is output for output in outputs)
        }
        if unknown_previews:
            raise ValueError(
                f"node previews use undeclared outputs: {sorted(unknown_previews)}"
            )
        if len({value.name for value in artifact_outputs}) != len(artifact_outputs):
            raise ValueError("artifact output names must be unique")
        if len(
            {(value.plot_kind, value.selector_kind) for value in selection_mappings}
        ) != len(selection_mappings):
            raise ValueError("selection mapping plot/selector pairs must be unique")
        unknown_draft_fields = {
            field
            for mapping in selection_mappings
            for field in mapping.draft_fields
        } - set(self.authoring_schema.field_names)
        if unknown_draft_fields:
            raise ValueError(
                "selection mappings use unknown draft fields: "
                f"{sorted(unknown_draft_fields)}"
            )
        resource_fields = tuple(value.field_name for value in workspace_resources)
        if len(set(resource_fields)) != len(resource_fields):
            raise ValueError("workspace resource fields must be unique")
        unknown_resource_fields = set(resource_fields) - set(
            self.authoring_schema.field_names
        )
        if unknown_resource_fields:
            raise ValueError(
                "workspace resources use unknown authoring fields: "
                f"{sorted(unknown_resource_fields)}"
            )
        declared_resource_fields = {
            field.name
            for field in self.authoring_schema.fields
            if field.value_type == "resource"
        }
        if set(resource_fields) != declared_resource_fields:
            raise ValueError(
                "workspace resource specs and resource authoring fields must "
                "match exactly"
            )
        dataset_inputs = tuple(
            value for value in inputs if isinstance(value, DatasetInputSpec)
        )
        build_argument_names = (
            *(value.name for value in self.authoring_schema.fields),
            *(value.argument_name for value in requirements),
            *(f"{value.argument_name}_key" for value in requirements),
            *(
                value.argument_name
                for value in inputs
                if isinstance(value, ArtifactInputSpec)
            ),
            *(
                value.frame_argument_name
                for value in inputs
                if isinstance(value, ArtifactInputSpec) and value.per_frame
            ),
            *(value.argument_name for value in workspace_resources),
            *(("source_signal",) if dataset_inputs else ()),
            "signal_plane",
        )
        duplicate_build_arguments = sorted(
            {
                name
                for name in build_argument_names
                if build_argument_names.count(name) > 1
            }
        )
        if duplicate_build_arguments:
            raise ValueError(
                "logic build argument namespace has collisions: "
                f"{duplicate_build_arguments!r}"
            )
        if self.kind is NodeKind.PROCESSOR and len(dataset_inputs) != 1:
            raise ValueError("a processor requires exactly one DatasetInputSpec")
        object.__setattr__(self, "input_specs", inputs)
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "node_previews", node_previews)
        object.__setattr__(self, "artifact_outputs", artifact_outputs)
        object.__setattr__(self, "device_requirements", requirements)
        object.__setattr__(self, "selection_mappings", selection_mappings)
        object.__setattr__(self, "workspace_resources", workspace_resources)
        object.__setattr__(self, "build_argument_names", build_argument_names)
        if self.build is not None and not callable(self.build):
            raise TypeError("build must be callable or None")

    def instantiate(self, **kwargs: Any) -> object:
        if self.build is None:
            return self
        return self.build(**kwargs)

    def selection_patch(
        self,
        selection: SelectionState,
        *,
        draft: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Map a supported committed selection; unsupported kinds do nothing."""

        if not isinstance(selection, SelectionState):
            raise TypeError("selection must be SelectionState")
        if not isinstance(draft, Mapping) or not isinstance(context, Mapping):
            raise TypeError("selection draft and context must be mappings")
        for mapping in self.selection_mappings:
            if (
                selection.plot_kind == mapping.plot_kind
                and selection.selector_kind == mapping.selector_kind
            ):
                mapped = mapping.map_patch(selection, draft, context)
                if mapped is None:
                    return None
                result = dict(mapped)
                if set(result) != set(mapping.draft_fields):
                    raise ValueError(
                        "selection mapping must return its declared draft fields"
                    )
                return result
        return None

__all__ = [
    "ArtifactCodec",
    "ArtifactInputSpec",
    "DatasetInputSpec",
    "DeviceRequirement",
    "LogicNodeDescriptor",
    "NodeKind",
    "NodePreviewSpec",
    "ResolvedArtifact",
    "ResolvedWorkspaceResource",
    "SelectionMapping",
    "WorkspaceResourceSpec",
]
