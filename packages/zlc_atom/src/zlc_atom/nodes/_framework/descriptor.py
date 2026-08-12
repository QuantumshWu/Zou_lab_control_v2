"""Data-only logic-node declarations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from zlc_atom.authoring import AuthoringSchema
from zlc_atom.install.descriptors import CAPABILITY_TYPES
from zlc_runtime import SelectionState


class NodeKind(str, Enum):
    """What a node IS to the experiment: how it is layered, not how it runs."""

    MEASUREMENT = "measurement"
    TASK = "task"
    PROCESSOR = "processor"


class DeviceAccess(str, Enum):
    """How one logic run may use a named device instance."""

    OBSERVE = "observe"
    EXCLUSIVE = "exclusive"


@dataclass(frozen=True)
class DatasetInputSpec:
    name: str
    contract_id: str | None
    required: bool = True

    def __post_init__(self) -> None:
        if not self.name or (self.contract_id is not None and not self.contract_id):
            raise ValueError("dataset input requires a name and a contract or None")

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

    @property
    def contract_id(self) -> str:
        return self.codec.contract_id


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
class OutputSpec:
    name: str
    contract_id: str

    def __post_init__(self) -> None:
        if not self.name or not self.contract_id:
            raise ValueError("output requires name and contract_id")


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
    """UI-neutral request to preview one declared output of this node.

    A node with eight outputs knows which one an operator came to watch, and
    nothing else does.  The KIND is normally left empty: the plotting package
    reads it off the data, exactly as it does for a hand-added panel, and a
    node that pins one is claiming its data would otherwise be read wrong.
    """

    output_name: str
    plot_kind: str = ""

    def __post_init__(self) -> None:
        if not self.output_name:
            raise ValueError("node preview requires an output_name")


@dataclass(frozen=True)
class DeviceRequirement:
    capability_token: str
    argument_name: str
    access: DeviceAccess

    def __post_init__(self) -> None:
        if not self.capability_token or not self.argument_name:
            raise ValueError(
                "device requirement token and build argument name must be non-empty"
            )
        if not isinstance(self.access, DeviceAccess):
            raise TypeError("device requirement access must be DeviceAccess")


@dataclass(frozen=True)
class SelectionMapping:
    """Data-only translation from one semantic selection to a draft patch."""

    plot_kind: str
    selector_kind: str
    draft_fields: tuple[str, ...]
    map_patch: Callable[
        [SelectionState, Mapping[str, Any], Mapping[str, Any]],
        Mapping[str, Any],
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
    outputs: tuple[OutputSpec, ...] = ()
    resolve_outputs: Callable[[Mapping[str, object]], tuple[OutputSpec, ...]] | None = None
    device_requirements: tuple[DeviceRequirement, ...] = ()
    build: Callable[..., object] | None = None
    node_previews: tuple[NodePreviewSpec, ...] = ()
    artifact_outputs: tuple[ArtifactOutputSpec, ...] = ()
    ui_contributions: tuple[object, ...] = ()
    selection_mappings: tuple[SelectionMapping, ...] = ()
    workspace_resources: tuple[WorkspaceResourceSpec, ...] = ()
    build_argument_names: tuple[str, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not self.api_name or not isinstance(self.kind, NodeKind):
            raise ValueError("logic node requires api_name and a valid kind")
        if not isinstance(self.authoring_schema, AuthoringSchema):
            raise TypeError("authoring_schema must be AuthoringSchema")
        inputs = tuple(self.input_specs)
        outputs = tuple(self.outputs)
        node_previews = tuple(self.node_previews)
        artifact_outputs = tuple(self.artifact_outputs)
        requirements = tuple(self.device_requirements)
        selection_mappings = tuple(self.selection_mappings)
        workspace_resources = tuple(self.workspace_resources)
        if any(not isinstance(value, (DatasetInputSpec, ArtifactInputSpec)) for value in inputs):
            raise TypeError("input_specs contain an unsupported input type")
        if any(not isinstance(value, OutputSpec) for value in outputs):
            raise TypeError("outputs must contain OutputSpec values")
        if self.resolve_outputs is not None and not callable(self.resolve_outputs):
            raise TypeError("resolve_outputs must be callable or None")
        if outputs and self.resolve_outputs is not None:
            raise ValueError("static outputs and resolve_outputs are exclusive")
        if any(not isinstance(value, NodePreviewSpec) for value in node_previews):
            raise TypeError("node_previews must contain NodePreviewSpec values")
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
        if len({value.output_name for value in node_previews}) != len(node_previews):
            raise ValueError("node preview output names must be unique")
        # Only a node whose outputs are fixed at declaration can be held to
        # them here; one that resolves them from its values answers later.
        unknown_previews = (
            set()
            if self.resolve_outputs is not None
            else {value.output_name for value in node_previews}
            - {value.name for value in outputs}
        )
        if unknown_previews:
            raise ValueError(
                f"node previews use undeclared outputs: {sorted(unknown_previews)}"
            )
        if node_previews and self.kind is NodeKind.PROCESSOR:
            # A preview is what STARTING this node offers to show, and a
            # processor is not started to be watched: it exists to answer
            # someone else's signal, and which of its answers an operator
            # wants on screen is a decision about that board, made by adding
            # the panel.  A measurement or a Task owns the shot it opens.
            raise ValueError("a processor node declares no preview")
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

    def outputs_for(self, values: Mapping[str, object]) -> tuple[OutputSpec, ...]:
        """Resolve the outputs declared by one exact authored request."""

        if self.resolve_outputs is None:
            return self.outputs
        authored = self.authoring_schema.project_values(values)
        outputs = tuple(self.resolve_outputs(authored))
        if any(not isinstance(value, OutputSpec) for value in outputs):
            raise TypeError("resolve_outputs must return OutputSpec values")
        if len({value.name for value in outputs}) != len(outputs):
            raise ValueError("resolved output names must be unique")
        return outputs

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
                result = dict(mapping.map_patch(selection, draft, context))
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
    "DeviceAccess",
    "DeviceRequirement",
    "LogicNodeDescriptor",
    "NodeKind",
    "NodePreviewSpec",
    "OutputSpec",
    "ResolvedArtifact",
    "ResolvedWorkspaceResource",
    "SelectionMapping",
    "WorkspaceResourceSpec",
]
