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


@dataclass(frozen=True, slots=True)
class TaskPreviewSpec:
    """UI-neutral request to preview one declared Task Dataset output."""

    output_name: str
    plot_kind: str

    def __post_init__(self) -> None:
        if not self.output_name or not self.plot_kind:
            raise ValueError("task preview requires output_name and plot_kind")


@dataclass(frozen=True, slots=True)
class TaskReportSpec:
    """UI-neutral terminal report request over one exact output sibling bundle."""

    adapter_key: str
    output_names: tuple[str, ...]
    trigger_output_name: str

    def __post_init__(self) -> None:
        adapter_key = str(self.adapter_key).strip()
        output_names = tuple(str(value).strip() for value in self.output_names)
        trigger = str(self.trigger_output_name).strip()
        if not adapter_key or not output_names or any(not value for value in output_names):
            raise ValueError("task report requires adapter_key and output_names")
        if len(set(output_names)) != len(output_names):
            raise ValueError("task report output_names must be unique")
        if trigger not in output_names:
            raise ValueError("task report trigger must belong to its output_names")
        object.__setattr__(self, "adapter_key", adapter_key)
        object.__setattr__(self, "output_names", output_names)
        object.__setattr__(self, "trigger_output_name", trigger)


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
    task_previews: tuple[TaskPreviewSpec, ...] = ()
    task_reports: tuple[TaskReportSpec, ...] = ()
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
        task_previews = tuple(self.task_previews)
        task_reports = tuple(self.task_reports)
        artifact_outputs = tuple(self.artifact_outputs)
        requirements = tuple(self.device_requirements)
        selection_mappings = tuple(self.selection_mappings)
        if any(not isinstance(value, (DatasetInputSpec, ArtifactInputSpec)) for value in inputs):
            raise TypeError("input_specs contain an unsupported input type")
        if any(not isinstance(value, OutputSpec) for value in outputs):
            raise TypeError("outputs must contain OutputSpec values")
        if any(not isinstance(value, TaskPreviewSpec) for value in task_previews):
            raise TypeError("task_previews must contain TaskPreviewSpec values")
        if any(not isinstance(value, TaskReportSpec) for value in task_reports):
            raise TypeError("task_reports must contain TaskReportSpec values")
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
        if len({value.output_name for value in task_previews}) != len(task_previews):
            raise ValueError("task preview output names must be unique")
        unknown_previews = {
            value.output_name for value in task_previews
        } - {value.name for value in outputs}
        if unknown_previews:
            raise ValueError(
                f"task previews use undeclared outputs: {sorted(unknown_previews)}"
            )
        if task_previews and self.kind is not NodeKind.TASK:
            raise ValueError("only Task nodes may declare task previews")
        if len({value.adapter_key for value in task_reports}) != len(task_reports):
            raise ValueError("task report adapter keys must be unique")
        unknown_report_outputs = {
            name for report in task_reports for name in report.output_names
        } - {value.name for value in outputs}
        if unknown_report_outputs:
            raise ValueError(
                "task reports use undeclared outputs: "
                f"{sorted(unknown_report_outputs)}"
            )
        if task_reports and self.kind is not NodeKind.TASK:
            raise ValueError("only Task nodes may declare task reports")
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
        object.__setattr__(self, "task_previews", task_previews)
        object.__setattr__(self, "task_reports", task_reports)
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
    "TaskPreviewSpec",
    "TaskReportSpec",
]
