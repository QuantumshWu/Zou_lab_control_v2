"""Exact named Dataset outputs published by neutral-atom applications.

Output owners freeze the catalog-visible bare name together with the immutable
Dataset.  Desktop shells route these values; they do not
reconstruct arrays, axes, coverage, or lineage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from zlc_data import OwnedSnapshot
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
    """One owner-paired public output name and semantic contract identity."""

    name: str
    contract_id: str

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


@dataclass(frozen=True, slots=True)
class FinalDatasetOutput:
    """One owner-materialized FINAL Dataset under one bare output name."""

    declaration: DatasetOutputDeclaration
    snapshot: OwnedSnapshot
    run_record: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.declaration, DatasetOutputDeclaration):
            raise TypeError("declaration must be DatasetOutputDeclaration")
        if not isinstance(self.snapshot, OwnedSnapshot):
            raise TypeError("snapshot must be OwnedSnapshot")
        if self.run_record is not None and not isinstance(self.run_record, Mapping):
            raise TypeError("run_record must be a mapping or None")

    @property
    def name(self) -> str:
        return self.declaration.name

    @property
    def contract_id(self) -> str:
        return self.declaration.contract_id


@dataclass(frozen=True, slots=True)
class LiveDatasetOutput:
    """One owner-projected live Dataset with coverage in its own geometry."""

    declaration: DatasetOutputDeclaration
    snapshot: OwnedSnapshot
    coverage: DatasetCoverage | MonitorCoverage
    run_record: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.declaration, DatasetOutputDeclaration):
            raise TypeError("declaration must be DatasetOutputDeclaration")
        if not isinstance(self.snapshot, OwnedSnapshot):
            raise TypeError("snapshot must be OwnedSnapshot")
        if not isinstance(self.coverage, (DatasetCoverage, MonitorCoverage)):
            raise TypeError("coverage must be DatasetCoverage or MonitorCoverage")
        if self.run_record is not None and not isinstance(self.run_record, Mapping):
            raise TypeError("run_record must be a mapping or None")
        total = (
            self.snapshot.block.schema.repeat_axis.size
            * self.snapshot.block.schema.point_table.row_count
        )
        if self.coverage.total_cells != total:
            raise ValueError("live coverage differs from projected Dataset geometry")

    @property
    def name(self) -> str:
        return self.declaration.name

    @property
    def contract_id(self) -> str:
        return self.declaration.contract_id


__all__ = [
    "DatasetOutputDeclaration",
    "FinalDatasetOutput",
    "LiveDatasetOutput",
]
