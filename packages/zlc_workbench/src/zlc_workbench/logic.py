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
build is handed exactly the arguments it declares out of those facts.  A node
that asks for something the bench does not have is refused with the reason,
because the alternative -- a row that says "idle" forever -- is the failure this
console has been audited for twice.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import inspect
from typing import Any

from zlc_runtime import NodeHost


__all__ = ["LogicBinding", "LogicCatalog", "build_arguments"]


@dataclass
class LogicBinding:
    """One hosted node: what it is and what is running it.

    It used to carry its ROW as well -- a widget, held by the layer that is
    not allowed to hold one.  The row lives in the window now, and this side
    names the node instead.
    """

    node_id: str
    descriptor: Any
    host: NodeHost
    #: The node object itself.  The console built it, so it keeps it: a task
    #: that has run carries the artifact a later node is built ON, and reaching
    #: into the host's private field for it would be reading someone else's
    #: implementation to recover something this already had.
    node: Any = None
    #: What the operator set, as its schema froze it.  Kept so Edit can show
    #: what is in effect rather than the defaults again.
    values: Mapping[str, Any] = field(default_factory=dict)
    #: Which signal a processor reads, when it reads one.
    source_signal: str = ""
    #: The last state pushed to the row, so an unchanged row is left alone.
    shown: tuple = ()
    #: Asked to go, and still stopping.  The row stays until it has: a node
    #: taken off screen while it still holds a camera is one nobody can reach.
    removing: bool = False


def dataset_inputs(descriptor: Any) -> tuple[Any, ...]:
    """The live signals one node reads, as its descriptor declares them.

    A processor is built around a signal it consumes, and the runtime refuses
    to host a reactive node that was never told which one.  Whether to ask is
    therefore the descriptor's answer, not a guess from the node's kind.
    """

    from zlc_atom.nodes._framework.descriptor import DatasetInputSpec

    return tuple(
        spec
        for spec in getattr(descriptor, "input_specs", ())
        if isinstance(spec, DatasetInputSpec)
    )


def build_arguments(
    descriptor: Any,
    *,
    installation: Any,
    signal_plane: Any,
    values: Mapping[str, Any],
    source_signal: str = "",
    artifacts: Mapping[str, Any] | None = None,
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

    available: dict[str, Any] = {"signal_plane": signal_plane}
    for requirement in descriptor.device_requirements:
        key = requirement.device_key or requirement.capability_token
        try:
            available[str(key)] = installation.capability(
                requirement.capability_token, key=requirement.device_key
            )
        except Exception as error:
            raise LookupError(
                f"{descriptor.api_name} needs a {requirement.capability_token} "
                f"and this apparatus has none: {error}"
            ) from error
    if source_signal:
        available["source_signal"] = str(source_signal)
    # Artifacts arrive keyed by CONTRACT, and the descriptor's own input specs
    # say which contract fills which build argument.  Matching on the argument
    # name instead would be a coincidence between two files that never agreed
    # to it; the contract id is what both sides actually declare.
    offered = dict(artifacts or {})
    for spec in getattr(descriptor, "input_specs", ()):
        contract = getattr(spec, "contract_id", None)
        if contract is not None and contract in offered:
            available[spec.name] = offered[contract]
    available.update(dict(extras or {}))

    parameters = inspect.signature(build).parameters
    takes_anything = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
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
        # Read by an operator in the Add Logic list, so it names what is
        # missing rather than reprs a list of strings at them.
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
            from zlc_atom.nodes._framework.discovery import discover_logic_nodes

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
        output_names=tuple(output.name for output in descriptor.outputs),
    )
