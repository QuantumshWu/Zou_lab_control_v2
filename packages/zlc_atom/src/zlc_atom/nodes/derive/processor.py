"""One current Dataset estimate per output, evaluated over Runtime-owned inputs."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from zlc_data import owned_snapshot_from_arrays
from zlc_runtime import DatasetOutputDeclaration, LiveDatasetOutput, MonitorCoverage, SignalValue

from .expression import Operand, compact_row, compiled, execute, output_rows


def declared_outputs(rows: Sequence[Mapping[str, object]]) -> tuple[DatasetOutputDeclaration, ...]:
    return tuple(DatasetOutputDeclaration(row["name"], f"derive.{row['name']}",
                                          index_by_source=True)
                 for row in output_rows(rows))


class DeriveProcessor:
    """Numerical code owns no input history, placement table or parallel schema.

    Each publication of the bound bundle is one shot: every output is
    evaluated over that publication's members, in row order.
    """

    def __init__(self, *, outputs, primary_output: str, input_outputs: Sequence[str]):
        self._programs = compiled(outputs)
        self.outputs = tuple(compact_row(row) for row, _program in self._programs)
        self.primary_output = str(primary_output)
        names = tuple(input_outputs)
        if self.primary_output not in names:
            raise ValueError("the subscription anchor must belong to the input bundle")
        self.dataset_input_siblings = tuple(name for name in names if name != self.primary_output)
        self.dataset_output_declarations = declared_outputs(self.outputs)

    def evaluate(self, source: SignalValue):
        return self.evaluate_inputs({"a": source})

    def evaluate_inputs(self, inputs: Mapping[str, SignalValue]):
        primary = inputs.get("a")
        if not isinstance(primary, SignalValue):
            raise TypeError("derive needs its bound input bundle")
        by_output = {self.primary_output: primary}
        for name in self.dataset_input_siblings:
            value = inputs.get(name)
            if not isinstance(value, SignalValue):
                raise RuntimeError(f"input publication lost {name!r}")
            by_output[name] = value
        results = execute(self._programs, {
            name: Operand(value.schema, value.snapshot.materialize().block.values, value.snapshot.expanded_validity())
            for name, value in by_output.items()
        })
        outputs = {}
        for declaration in self.dataset_output_declarations:
            result = results[declaration.name]
            schema = replace(result.schema, value_schema=replace(
                result.schema.value_schema, name=declaration.name,
            ))
            snapshot = owned_snapshot_from_arrays(
                schema, result.values, primary.snapshot.block.revision,
                validity=result.valid, stream_generation=primary.snapshot.ref.stream_generation,
            )
            cells = result.schema.repeat_domain.size * result.schema.point_domain.size
            # An arbitrary reduction is a current estimate, not an input shot
            # pasted into a guessed finite placement. Runtime alone retains it
            # and builds history only for consumers that request a lease.
            outputs[declaration.name] = LiveDatasetOutput(
                declaration, snapshot, MonitorCoverage(cells, cells),
                event_record=primary.event_record,
            )
        return outputs

    def describe_run(self, inputs: Mapping[str, SignalValue]) -> dict[str, object]:
        primary = inputs["a"]
        return {
            "parameters": {
                "outputs": self.outputs,
                "source_signal": primary.name,
                "inputs": {
                    self.primary_output if name == "a" else name: value.name
                    for name, value in inputs.items()
                },
            },
        }


__all__ = ["DeriveProcessor", "declared_outputs"]
