"""Rglob-based logic-node discovery."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

from .descriptor import LogicNodeDescriptor


def _leaf_modules() -> tuple[tuple[str, Path], ...]:
    root = Path(__file__).resolve().parents[1]
    found: list[tuple[str, Path]] = []
    for path in root.rglob("logic_node.py"):
        relative = path.relative_to(root).with_suffix("")
        if relative.parts[0] == "_framework":
            continue
        module_name = "zlc_atom.nodes." + ".".join(relative.parts)
        found.append((module_name, path))
    return tuple(sorted(found))


def _load(module_name: str) -> LogicNodeDescriptor:
    module: ModuleType = importlib.import_module(module_name)
    value = getattr(module, "LOGIC_NODE", None)
    if not isinstance(value, LogicNodeDescriptor):
        raise TypeError(f"{module_name} must export a LogicNodeDescriptor named LOGIC_NODE")
    return value


def discover_logic_nodes() -> tuple[LogicNodeDescriptor, ...]:
    leaves = _leaf_modules()
    descriptors = tuple(_load(module_name) for module_name, _path in leaves)
    by_name = tuple(sorted(value.api_name for value in descriptors))
    if len(set(by_name)) != len(by_name):
        raise ValueError("logic node api_name values must be unique")
    return tuple(sorted(descriptors, key=lambda value: value.api_name))


__all__ = ["discover_logic_nodes"]
