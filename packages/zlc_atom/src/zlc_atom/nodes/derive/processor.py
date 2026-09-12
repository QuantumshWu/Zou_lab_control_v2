"""One current Dataset estimate per output, evaluated over Runtime-owned inputs."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from zlc_data import owned_snapshot_from_arrays
from zlc_runtime import DatasetOutputDeclaration, LiveDatasetOutput, MonitorCoverage, SignalValue

from .expression import Operand, compiled_rows, execute, signal_rows


def declared_outputs(rows: Sequence[Mapping[str, object]]) -> tuple[DatasetOutputDeclaration, ...]:
    return tuple(DatasetOutputDeclaration(row["name"], f"derive.{row['name']}", index_by_source=True)
                 for row in signal_rows(rows))


class DeriveProcessor:
    """Numerical code owns no input history, placement table or parallel schema."""

    def __init__(
        self, *, expressions, primary_output: str, input_outputs: Sequence[str],
        input_view: str = "event", window: int = 50, producer: str = "derive",
    ):
        self.instance_id = str(producer).strip()
        if not self.instance_id:
            raise ValueError("producer must be non-empty")
        self.expressions = signal_rows(expressions)
        self._programs = compiled_rows(self.expressions)
        self.primary_output = str(primary_output)
        names = tuple(input_outputs)
        if self.primary_output not in names:
            raise ValueError("the subscription anchor must belong to the input bundle")
        self.dataset_input_siblings = tuple(name for name in names if name != self.primary_output)
        if input_view not in ("event", "run", "window"):
            raise ValueError("input_view must be event, run or window")
        if isinstance(window, bool) or int(window) < 1:
            raise ValueError("window must be a positive event count")
        self.dataset_input_view = input_view
        self.dataset_input_window = int(window)
        self.dataset_output_declarations = declared_outputs(self.expressions)

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
            name: Operand(value.schema, value.values, value.snapshot.expanded_validity())
            for name, value in by_output.items()
        })
        record = {
            "node": self.instance_id,
            "parameters": {
                "expressions": self.expressions, "input_view": self.dataset_input_view,
                "window": self.dataset_input_window if self.dataset_input_view == "window" else None,
                "source_signal": primary.name,
                "inputs": {name: value.name for name, value in by_output.items()},
            },
        }
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
                declaration, snapshot, MonitorCoverage(cells, cells), record,
                event_record=primary.event_record,
            )
        return outputs


__all__ = ["DeriveProcessor", "declared_outputs"]
