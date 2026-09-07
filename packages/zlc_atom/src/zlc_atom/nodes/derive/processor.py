"""Publish one expression over the outputs of one publication."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from zlc_data import DatasetSchema, owned_snapshot_from_arrays
from zlc_runtime import (
    DatasetCoverage,
    DatasetOutputDeclaration,
    LiveDatasetOutput,
    MonitorCoverage,
    SignalValue,
)

from .expression import Operand, evaluate, referenced_outputs

DERIVE_OUTPUT = DatasetOutputDeclaration("value", "derive.value", index_by_source=True)
DERIVE_OUTPUTS = (DERIVE_OUTPUT,)


class DeriveProcessor:
    """Evaluate one expression on every publication of the bound producer.

    The inputs are the producer's outputs the expression names: the bound
    signal itself and its siblings from the same publication, so they are
    aligned by construction -- one event, one answer, one exact causal
    parent.  An expression over two producers is a join, which this
    runtime's lineage does not carry; it is not something this node quietly
    approximates.
    """

    def __init__(
        self,
        *,
        expression: str,
        primary_output: str | None,
        producer: str = "derive",
    ) -> None:
        self.instance_id = str(producer).strip()
        if not self.instance_id:
            raise ValueError("producer must be non-empty")
        self.expression = str(expression).strip()
        referenced = referenced_outputs(self.expression)
        if primary_output is not None and primary_output not in referenced:
            raise ValueError(
                f"the bound output {primary_output!r} is not one the expression reads"
            )
        self.primary_output = primary_output
        #: The producer outputs the expression reads beside the bound one:
        #: what the host must fetch from the same publication.
        self.dataset_input_siblings = tuple(
            name for name in referenced if name != primary_output
        )
        self.dataset_output_declarations = DERIVE_OUTPUTS

    def evaluate(self, source: SignalValue) -> dict[str, LiveDatasetOutput]:
        return self.evaluate_inputs({"a": source})

    def evaluate_inputs(
        self,
        inputs: Mapping[str, SignalValue],
    ) -> dict[str, LiveDatasetOutput]:
        primary = inputs.get("a")
        if not isinstance(primary, SignalValue):
            raise TypeError("derive needs its bound signal as input 'a'")
        by_output: dict[str, SignalValue] = {}
        if self.primary_output is not None:
            by_output[self.primary_output] = primary
        for name in self.dataset_input_siblings:
            sibling = inputs.get(name)
            if not isinstance(sibling, SignalValue):
                raise TypeError(f"derive publication lost sibling input {name!r}")
            by_output[name] = sibling
        operands = {
            name: Operand(
                value.schema,
                np.asarray(value.values),
                np.asarray(value.snapshot.expanded_validity(), dtype=bool),
            )
            for name, value in by_output.items()
        }
        result = evaluate(self.expression, operands)
        assert result.values is not None and result.valid is not None
        snapshot = owned_snapshot_from_arrays(
            result.schema,
            result.values,
            primary.snapshot.block.revision,
            validity=result.valid,
            stream_generation=primary.snapshot.ref.stream_generation,
        )
        run_record = {
            "node": self.instance_id,
            "parameters": {
                "expression": self.expression,
                "source_signal": primary.name,
                "inputs": {name: value.name for name, value in by_output.items()},
            },
        }
        coverage, canonical, origin = self._placement(primary, by_output, result.schema)
        return {
            DERIVE_OUTPUT.name: LiveDatasetOutput(
                DERIVE_OUTPUT,
                snapshot,
                coverage,
                run_record,
                canonical,
                origin,
            )
        }

    def _placement(
        self,
        primary: SignalValue,
        by_output: Mapping[str, SignalValue],
        schema: DatasetSchema,
    ) -> tuple[
        DatasetCoverage | MonitorCoverage,
        DatasetSchema | None,
        tuple[int, int] | None,
    ]:
        """Where the result sits in its run, from where the bound signal sits.

        The expression may collapse the point domain (``.frame(k)``); the
        run's cell count shrinks by the same factor, and the canonical
        (complete-run) schema is the same expression typed over the bound
        signal's canonical schema.
        """

        source_schema = primary.schema
        cycles = source_schema.repeat_domain.size
        scale = source_schema.point_domain.size // schema.point_domain.size
        if isinstance(primary.coverage, DatasetCoverage):
            if primary.canonical_schema is None or primary.cell_origin is None:
                raise ValueError("finite derive input lacks canonical placement")
            if (
                primary.coverage.written_cells % scale
                or primary.coverage.total_cells % scale
            ):
                raise ValueError(
                    "the bound signal's coverage is not whole cycles; derive "
                    "cannot keep exact bookkeeping"
                )
            canonical = evaluate(
                self.expression,
                {
                    name: Operand(
                        value.canonical_schema
                        if value.canonical_schema is not None
                        else value.schema
                    )
                    for name, value in by_output.items()
                },
            ).schema
            coverage: DatasetCoverage | MonitorCoverage = DatasetCoverage(
                primary.coverage.written_cells // scale,
                primary.coverage.total_cells // scale,
            )
            origin = (
                primary.cell_origin[0],
                primary.cell_origin[1] if scale == 1 else 0,
            )
            return coverage, canonical, origin
        if primary.coverage is None:
            return DatasetCoverage(cycles, cycles), schema, (0, 0)
        return (
            MonitorCoverage(
                min(cycles, primary.coverage.written_cells // scale),
                cycles,
            ),
            None,
            None,
        )


__all__ = ["DERIVE_OUTPUT", "DERIVE_OUTPUTS", "DeriveProcessor"]
