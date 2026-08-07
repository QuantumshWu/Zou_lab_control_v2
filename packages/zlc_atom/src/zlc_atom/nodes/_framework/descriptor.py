"""Data-only logic-node declarations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from zlc_atom.authoring import AuthoringSchema
from zlc_atom.install.descriptors import CAPABILITY_TYPES


class NodeKind(str, Enum):
    """What a node IS to the experiment: how it is layered, not how it runs."""

    MEASUREMENT = "measurement"
    TASK = "task"
    PROCESSOR = "processor"


#: How each domain layer runs, in the runtime's vocabulary.
#:
#: These are two different questions -- what a node is to the experiment, and
#: how the host drives it -- but the answer to the second follows entirely from
#: the first: something that acquires or orchestrates runs to completion, and
#: something that derives reacts to each new value.  Deriving it here rather
#: than letting every node declare both keeps the two from ever disagreeing.
_RUNTIME_KIND = {
    NodeKind.MEASUREMENT: "finite",
    NodeKind.TASK: "finite",
    NodeKind.PROCESSOR: "reactive",
}


def runtime_kind(node_kind: NodeKind) -> str:
    """The zlc_runtime execution kind for a domain layer."""

    try:
        return _RUNTIME_KIND[NodeKind(node_kind)]
    except (KeyError, ValueError) as error:
        raise ValueError(f"no runtime kind is defined for {node_kind!r}") from error


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
    name: str
    contract_id: str
    allow_saved_reference: bool = False
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
class DeviceRequirement:
    capability_token: str
    device_key: str | None = None

    def __post_init__(self) -> None:
        if not self.capability_token:
            raise ValueError("device requirement token must be non-empty")


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
    artifact_outputs: tuple[object, ...] = ()
    ui_contributions: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        if not self.api_name or not isinstance(self.kind, NodeKind):
            raise ValueError("logic node requires api_name and a valid kind")
        if not isinstance(self.authoring_schema, AuthoringSchema):
            raise TypeError("authoring_schema must be AuthoringSchema")
        inputs = tuple(self.input_specs)
        outputs = tuple(self.outputs)
        requirements = tuple(self.device_requirements)
        if any(not isinstance(value, (DatasetInputSpec, ArtifactInputSpec)) for value in inputs):
            raise TypeError("input_specs contain an unsupported input type")
        if any(not isinstance(value, OutputSpec) for value in outputs):
            raise TypeError("outputs must contain OutputSpec values")
        if any(not isinstance(value, DeviceRequirement) for value in requirements):
            raise TypeError("device_requirements must contain DeviceRequirement values")
        unknown_requirements = {value.capability_token for value in requirements} - set(CAPABILITY_TYPES)
        if unknown_requirements:
            raise ValueError(f"logic node uses unknown capability tokens: {sorted(unknown_requirements)}")
        if len({value.name for value in inputs}) != len(inputs):
            raise ValueError("input names must be unique")
        if len({value.name for value in outputs}) != len(outputs):
            raise ValueError("output names must be unique")
        if self.kind is NodeKind.PROCESSOR and len(tuple(value for value in inputs if isinstance(value, DatasetInputSpec))) != 1:
            raise ValueError("a processor requires exactly one DatasetInputSpec")
        object.__setattr__(self, "input_specs", inputs)
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "device_requirements", requirements)
        if self.build is not None and not callable(self.build):
            raise TypeError("build must be callable or None")

    @property
    def definition(self) -> "LogicNodeDescriptor":
        """Compatibility-free projection used by generic host tests."""

        return self

    @property
    def name(self) -> str:
        return self.api_name

    def instantiate(self, **kwargs: Any) -> object:
        if self.build is None:
            return self
        return self.build(**kwargs)


__all__ = [
    "ArtifactInputSpec",
    "DatasetInputSpec",
    "DeviceRequirement",
    "LogicNodeDescriptor",
    "NodeKind",
    "OutputSpec",
]
