"""Data-only logic-node declarations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
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
    contract_id: str
    required: bool = True

    def __post_init__(self) -> None:
        if not self.name or not self.contract_id:
            raise ValueError("dataset input requires name and contract_id")


@dataclass(frozen=True)
class ArtifactInputSpec:
    """One explicit saved-artifact path consumed by a run."""

    name: str
    contract_id: str
    required: bool = True

    def __post_init__(self) -> None:
        if not self.name or not self.contract_id:
            raise ValueError("artifact input requires name and contract_id")


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
    device_requirements: tuple[DeviceRequirement, ...] = ()
    build: Callable[..., object] | None = None
    task_previews: tuple[object, ...] = ()
    artifact_outputs: tuple[ArtifactOutputSpec, ...] = ()
    ui_contributions: tuple[object, ...] = ()
    selection_mappings: tuple[SelectionMapping, ...] = ()

    def __post_init__(self) -> None:
        if not self.api_name or not isinstance(self.kind, NodeKind):
            raise ValueError("logic node requires api_name and a valid kind")
        if not isinstance(self.authoring_schema, AuthoringSchema):
            raise TypeError("authoring_schema must be AuthoringSchema")
        inputs = tuple(self.input_specs)
        outputs = tuple(self.outputs)
        artifact_outputs = tuple(self.artifact_outputs)
        requirements = tuple(self.device_requirements)
        selection_mappings = tuple(self.selection_mappings)
        if any(not isinstance(value, (DatasetInputSpec, ArtifactInputSpec)) for value in inputs):
            raise TypeError("input_specs contain an unsupported input type")
        if any(not isinstance(value, OutputSpec) for value in outputs):
            raise TypeError("outputs must contain OutputSpec values")
        if any(not isinstance(value, ArtifactOutputSpec) for value in artifact_outputs):
            raise TypeError("artifact_outputs must contain ArtifactOutputSpec values")
        if any(not isinstance(value, DeviceRequirement) for value in requirements):
            raise TypeError("device_requirements must contain DeviceRequirement values")
        if any(not isinstance(value, SelectionMapping) for value in selection_mappings):
            raise TypeError("selection_mappings must contain SelectionMapping values")
        unknown_requirements = {value.capability_token for value in requirements} - set(CAPABILITY_TYPES)
        if unknown_requirements:
            raise ValueError(f"logic node uses unknown capability tokens: {sorted(unknown_requirements)}")
        if len({value.name for value in inputs}) != len(inputs):
            raise ValueError("input names must be unique")
        if len({value.name for value in outputs}) != len(outputs):
            raise ValueError("output names must be unique")
        if len({value.name for value in artifact_outputs}) != len(artifact_outputs):
            raise ValueError("artifact output names must be unique")
        if len({value.argument_name for value in requirements}) != len(requirements):
            raise ValueError("device requirement argument names must be unique")
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
        if self.kind is NodeKind.PROCESSOR and len(tuple(value for value in inputs if isinstance(value, DatasetInputSpec))) != 1:
            raise ValueError("a processor requires exactly one DatasetInputSpec")
        object.__setattr__(self, "input_specs", inputs)
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "artifact_outputs", artifact_outputs)
        object.__setattr__(self, "device_requirements", requirements)
        object.__setattr__(self, "selection_mappings", selection_mappings)
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
                result = dict(mapping.map_patch(selection, draft, context))
                if set(result) != set(mapping.draft_fields):
                    raise ValueError(
                        "selection mapping must return its declared draft fields"
                    )
                return result
        return None


__all__ = [
    "ArtifactInputSpec",
    "DatasetInputSpec",
    "DeviceAccess",
    "DeviceRequirement",
    "LogicNodeDescriptor",
    "NodeKind",
    "OutputSpec",
    "SelectionMapping",
]
