"""Publish every named line of one program over one publication."""

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

from .expression import Operand, evaluate, published_names, referenced_outputs


def declared_outputs(program: str) -> tuple[DatasetOutputDeclaration, ...]:
    """What one program publishes: one output per named line, in order.

    The name is the line's own; the contract says a derive computed it.  A
    panel leasing an output's history is what makes it a bounded dataset,
    so every line is indexed by its source, as the bound signal is.
    """

    return tuple(
        DatasetOutputDeclaration(name, f"derive.{name}", index_by_source=True)
        for name in published_names(program)
    )


class DeriveProcessor:
    """Evaluate one program on every publication of the bound producer.

    The inputs are the producer's outputs the program names: the bound
    signal itself and its siblings from the same publication, so they are
    aligned by construction -- one event, one answer, one exact causal
    parent.  A program over two producers is a join, which this runtime's
    lineage does not carry; it is not something this node quietly
    approximates.
    """

    def __init__(
        self,
        *,
        expressions: str,
        primary_output: str | None,
        producer: str = "derive",
    ) -> None:
        self.instance_id = str(producer).strip()
        if not self.instance_id:
            raise ValueError("producer must be non-empty")
        self.expressions = str(expressions).strip()
        referenced = referenced_outputs(self.expressions)
        if primary_output is not None and primary_output not in referenced:
            raise ValueError(
                f"the bound output {primary_output!r} is not one the program reads"
            )
        self.primary_output = primary_output
        #: The producer outputs the program reads beside the bound one:
        #: what the host must fetch from the same publication.
        self.dataset_input_siblings = tuple(
            name for name in referenced if name != primary_output
        )
        self.dataset_output_declarations = declared_outputs(self.expressions)

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
        results = evaluate(self.expressions, operands)
        run_record = {
            "node": self.instance_id,
            "parameters": {
                "expressions": self.expressions,
                "source_signal": primary.name,
                "inputs": {name: value.name for name, value in by_output.items()},
            },
        }
        canonical = self._canonical_schemas(primary, by_output)
        outputs: dict[str, LiveDatasetOutput] = {}
        for declaration in self.dataset_output_declarations:
            result = results[declaration.name]
            assert result.values is not None and result.valid is not None
            snapshot = owned_snapshot_from_arrays(
                result.schema,
                result.values,
                primary.snapshot.block.revision,
                validity=result.valid,
                stream_generation=primary.snapshot.ref.stream_generation,
            )
            coverage, canonical_schema, origin = self._placement(
                primary, result.schema, canonical.get(declaration.name)
            )
            outputs[declaration.name] = LiveDatasetOutput(
                declaration,
                snapshot,
                coverage,
                run_record,
                canonical_schema,
                origin,
            )
        return outputs

    def _canonical_schemas(
        self,
        primary: SignalValue,
        by_output: Mapping[str, SignalValue],
    ) -> dict[str, DatasetSchema]:
        """The complete-run schema of every line -- the same program typed
        over the bound signal's canonical schema.  Only a finite run has one."""

        if not isinstance(primary.coverage, DatasetCoverage):
            return {}
        if primary.canonical_schema is None or primary.cell_origin is None:
            raise ValueError("finite derive input lacks canonical placement")
        typed = evaluate(
            self.expressions,
            {
                name: Operand(
                    value.canonical_schema
                    if value.canonical_schema is not None
                    else value.schema
                )
                for name, value in by_output.items()
            },
        )
        return {name: operand.schema for name, operand in typed.items()}

    @staticmethod
    def _placement(
        primary: SignalValue,
        schema: DatasetSchema,
        canonical: DatasetSchema | None,
    ) -> tuple[
        DatasetCoverage | MonitorCoverage,
        DatasetSchema | None,
        tuple[int, int] | None,
    ]:
        """Where one result sits in its run, from where the bound signal sits.

        A line may collapse the point domain (``.frame(k)``); the run's cell
        count shrinks by the same factor.
        """

        source_schema = primary.schema
        cycles = source_schema.repeat_domain.size
        scale = source_schema.point_domain.size // schema.point_domain.size
        if isinstance(primary.coverage, DatasetCoverage):
            assert canonical is not None and primary.cell_origin is not None
            if (
                primary.coverage.written_cells % scale
                or primary.coverage.total_cells % scale
            ):
                raise ValueError(
                    "the bound signal's coverage is not whole cycles; derive "
                    "cannot keep exact bookkeeping"
                )
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


__all__ = ["DeriveProcessor", "declared_outputs"]
