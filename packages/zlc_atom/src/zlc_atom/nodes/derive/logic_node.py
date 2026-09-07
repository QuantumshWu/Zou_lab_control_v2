"""Discoverable derive processor descriptor."""

from __future__ import annotations

from collections.abc import Mapping

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.nodes._framework.descriptor import (
    DatasetInputSpec,
    LogicNodeDescriptor,
    NodeKind,
)
from zlc_runtime import DatasetOutputDeclaration

from .expression import ExpressionError, referenced_outputs
from .processor import DeriveProcessor, declared_outputs


def _readable(values: Mapping[str, object]) -> None:
    """A complete draft is one whose program can be read.

    Refused here, by what was written, while the draft is being edited --
    not at Start, where the same words would arrive as a failure to run.
    """

    referenced_outputs(str(values["expressions"]))


DERIVE_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "expressions",
            "multiline",
            "Expressions",
            "",
            required=True,
            description="one per line: name = expression over a.<output>",
        ),
    ),
    validator=_readable,
)


def _draft_outputs(
    values: Mapping[str, object],
) -> tuple[DatasetOutputDeclaration, ...]:
    """What one draft publishes: the names of its lines.

    A draft that cannot be read publishes nothing yet; the schema's
    validator is what says why, to the editor, in the program's own words.
    """

    try:
        return declared_outputs(str(values.get("expressions") or ""))
    except ExpressionError:
        return ()


def _build(*, signal_plane: object, source_signal: str, **values: object) -> DeriveProcessor:
    """The processor for one program over the bound producer's outputs.

    The plane says which outputs the bound producer has, so a name the
    program reads that the producer never publishes is refused here, by
    name, before anything runs -- and the bound signal is recognised among
    them by the plane's own resolution, never by taking its name apart.
    """

    authored = DERIVE_SCHEMA.project_values(values)
    program = str(authored["expressions"])
    selected_source = str(source_signal).strip()
    if not selected_source:
        raise ValueError("source_signal must be non-empty")
    referenced = referenced_outputs(program)
    resolved = signal_plane.resolve_sibling_signals(selected_source, referenced)
    primary = next(
        (name for name, qualified in zip(referenced, resolved) if qualified == selected_source),
        None,
    )
    return DeriveProcessor(expressions=program, primary_output=primary)


LOGIC_NODE = LogicNodeDescriptor(
    "derive",
    NodeKind.PROCESSOR,
    DERIVE_SCHEMA,
    input_specs=(DatasetInputSpec("a", None, "exact"),),
    declare_outputs=_draft_outputs,
    build=_build,
)


__all__ = ["DERIVE_SCHEMA", "LOGIC_NODE"]
