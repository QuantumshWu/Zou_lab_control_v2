"""Discoverable derive processor descriptor."""

from __future__ import annotations

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.nodes._framework.descriptor import (
    DatasetInputSpec,
    LogicNodeDescriptor,
    NodeKind,
)

from .expression import referenced_outputs
from .processor import DERIVE_OUTPUTS, DeriveProcessor

DERIVE_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "expression",
            "text",
            "Expression",
            "",
            required=True,
        ),
    )
)


def _build(*, signal_plane: object, source_signal: str, **values: object) -> DeriveProcessor:
    """The processor for one expression over the bound producer's outputs.

    The plane says which outputs the bound producer has, so a name the
    expression reads that the producer never publishes is refused here, by
    name, before anything runs -- and the bound signal is recognised among
    them by the plane's own resolution, never by taking its name apart.
    """

    authored = DERIVE_SCHEMA.project_values(values)
    expression = str(authored["expression"])
    selected_source = str(source_signal).strip()
    if not selected_source:
        raise ValueError("source_signal must be non-empty")
    referenced = referenced_outputs(expression)
    resolved = signal_plane.resolve_sibling_signals(selected_source, referenced)
    primary = next(
        (name for name, qualified in zip(referenced, resolved) if qualified == selected_source),
        None,
    )
    return DeriveProcessor(expression=expression, primary_output=primary)


LOGIC_NODE = LogicNodeDescriptor(
    "derive",
    NodeKind.PROCESSOR,
    DERIVE_SCHEMA,
    input_specs=(DatasetInputSpec("a", None, "exact"),),
    outputs=DERIVE_OUTPUTS,
    build=_build,
)


__all__ = ["DERIVE_SCHEMA", "LOGIC_NODE"]
