"""One current Dataset estimate per published variable, evaluated over Runtime-owned inputs."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from zlc_data import owned_snapshot_from_arrays
from zlc_runtime import DatasetOutputDeclaration, LiveDatasetOutput, MonitorCoverage, SignalValue

from .expression import Operand, compact_variable, evaluate, published


def declared_outputs(rows: Sequence[Mapping[str, object]]) -> tuple[DatasetOutputDeclaration, ...]:
    """One signal per published variable, in card order; raises a name problem."""

    return tuple(DatasetOutputDeclaration(name, f"derive.{name}", index_by_source=True)
                 for name in published(rows))


class DeriveProcessor:
    """Numerical code owns no input history, placement table or parallel schema.

    Each publication of the bound bundle is evaluated once: the LIVE
    variables over that publication's members, in dependency order, and
    each published variable is a complete current estimate.
    """

    def __init__(self, *, variables, primary_output: str, input_outputs: Sequence[str]):
        self._variables = tuple(variables)
        self.primary_output = str(primary_output)
        names = tuple(input_outputs)
        if self.primary_output not in names:
            raise ValueError("the subscription anchor must belong to the input bundle")
        self.dataset_input_siblings = tuple(name for name in names if name != self.primary_output)
        self.dataset_output_declarations = declared_outputs(self._variables)

    def evaluate(self, source: SignalValue):
        return self.evaluate_inputs({"bundle": source})

    def evaluate_inputs(self, inputs: Mapping[str, SignalValue]):
        primary = inputs.get("bundle")
        if not isinstance(primary, SignalValue):
            raise TypeError("derive needs its bound input bundle")
        by_output = {self.primary_output: primary}
        for name in self.dataset_input_siblings:
            value = inputs.get(name)
            if not isinstance(value, SignalValue):
                raise RuntimeError(f"input publication lost {name!r}")
            by_output[name] = value
        results = evaluate(self._variables, {
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
        """Every variable as written -- internal and parked ones too -- and
        which signal fed each input name."""

        primary = inputs["bundle"]
        return {
            "parameters": {
                "variables": tuple(compact_variable(row) for row in self._variables),
                "source_signal": primary.name,
                "inputs": {
                    self.primary_output if name == "bundle" else name: value.name
                    for name, value in inputs.items()
                },
            },
        }


__all__ = ["DeriveProcessor", "declared_outputs"]
