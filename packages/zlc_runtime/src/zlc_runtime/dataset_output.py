"""Exact named Dataset outputs published by neutral-atom applications.

Output owners freeze the catalog-visible bare name together with the immutable
Dataset.  Desktop shells route these values; they do not
reconstruct arrays, axes, coverage, or lineage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from zlc_data import DatasetSchema, OwnedSnapshot
from .dataset import (
    DatasetCoverage,
    MonitorCoverage,
)
from zlc_data import canonical_text


def _bare_output_name(value: str, *, kind: str = "output") -> str:
    name = canonical_text(value, f"{kind} name")
    if "/" in name or name.startswith("@"):
        raise ValueError(f"{kind} name must be bare, not namespaced")
    return name


@dataclass(frozen=True, slots=True)
class DatasetOutputDeclaration:
    """One owner-paired public output and its live-history semantics.

    A Monitor normally means exactly what its producer publishes: the latest
    complete event.  Some display derivations (for example fit parameters used
    by a Rolling plot) instead need a bounded Dataset containing one cell for
    every parent event.  ``index_by_source`` only declares that capability;
    Runtime retains nothing beyond latest until a real consumer acquires a
    bounded history lease.  It must never infer that extra scientific axis
    merely because an output came from a Processor.
    """

    name: str
    contract_id: str
    index_by_source: bool = field(default=False, kw_only=True)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _bare_output_name(self.name, kind="dataset output"),
        )
        object.__setattr__(
            self,
            "contract_id",
            canonical_text(self.contract_id, "dataset output contract id"),
        )
        if type(self.index_by_source) is not bool:
            raise TypeError("dataset output index_by_source must be bool")


@dataclass(frozen=True, slots=True)
class LiveDatasetOutput:
    """One immutable event chunk and its canonical run placement.

    ``snapshot`` is the newly committed event.  A finite run also supplies
    the fixed schema of the complete run and the event's ``(repeat, point)``
    origin in that schema.  Runtime can therefore accumulate and seal the
    run without asking the plugin to keep or re-materialize its history.

    A monitor is intentionally different: it retains only its latest event,
    so it has neither a canonical run schema nor a placement.
    """

    declaration: DatasetOutputDeclaration
    snapshot: OwnedSnapshot
    coverage: DatasetCoverage | MonitorCoverage
    run_record: Mapping[str, object] | None = None
    canonical_schema: DatasetSchema | None = None
    cell_origin: tuple[int, int] | None = None
    event_record: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.declaration, DatasetOutputDeclaration):
            raise TypeError("declaration must be DatasetOutputDeclaration")
        if not isinstance(self.snapshot, OwnedSnapshot):
            raise TypeError("snapshot must be OwnedSnapshot")
        if not isinstance(self.coverage, (DatasetCoverage, MonitorCoverage)):
            raise TypeError("coverage must be DatasetCoverage or MonitorCoverage")
        if self.run_record is not None and not isinstance(self.run_record, Mapping):
            raise TypeError("run_record must be a mapping or None")
        if self.event_record is not None and not isinstance(
            self.event_record, Mapping
        ):
            raise TypeError("event_record must be a mapping or None")
        if (self.canonical_schema is None) != (self.cell_origin is None):
            raise ValueError(
                "canonical_schema and cell_origin must be supplied together"
            )
        if self.canonical_schema is None:
            if not isinstance(self.coverage, MonitorCoverage):
                raise ValueError(
                    "a finite Dataset event requires canonical placement"
                )
            total = (
                self.snapshot.block.schema.repeat_axis.size
                * self.snapshot.block.schema.point_table.row_count
            )
            if self.coverage.total_cells != total:
                raise ValueError(
                    "live coverage differs from projected Dataset geometry"
                )
            return
        if not isinstance(self.coverage, DatasetCoverage):
            raise ValueError("only a finite Dataset event has canonical placement")
        if not isinstance(self.canonical_schema, DatasetSchema):
            raise TypeError("canonical_schema must be DatasetSchema or None")
        try:
            origin = tuple(self.cell_origin)
        except TypeError as error:
            raise TypeError("cell_origin must be a two-integer tuple or None") from error
        if (
            len(origin) != 2
            or any(type(value) is not int for value in origin)
            or any(value < 0 for value in origin)
        ):
            raise ValueError("cell_origin must contain two non-negative integers")
        object.__setattr__(self, "cell_origin", origin)
        canonical_total = (
            self.canonical_schema.repeat_axis.size
            * self.canonical_schema.point_table.row_count
        )
        if self.coverage.total_cells != canonical_total:
            raise ValueError("finite coverage differs from canonical Dataset geometry")
        event_schema = self.snapshot.block.schema
        if event_schema.cell_schema != self.canonical_schema.cell_schema:
            raise ValueError("event cell schema differs from canonical Dataset schema")
        repeat_origin, point_origin = origin
        if (
            repeat_origin + event_schema.repeat_axis.size
            > self.canonical_schema.repeat_axis.size
            or point_origin + event_schema.point_table.row_count
            > self.canonical_schema.point_table.row_count
        ):
            raise ValueError("event placement lies outside canonical Dataset geometry")

    @property
    def name(self) -> str:
        return self.declaration.name

    @property
    def contract_id(self) -> str:
        return self.declaration.contract_id


__all__ = [
    "DatasetOutputDeclaration",
    "LiveDatasetOutput",
]
