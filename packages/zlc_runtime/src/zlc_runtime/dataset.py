"""Coverage facts for live Runtime Dataset publications."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DatasetCoverage:
    """Written cells in one finite Dataset geometry."""

    written_cells: int
    total_cells: int

    def __post_init__(self) -> None:
        _validate_cell_counts(self)

    @property
    def complete(self) -> bool:
        return self.written_cells == self.total_cells


@dataclass(frozen=True, slots=True)
class MonitorCoverage:
    """Written cells in the currently retained live Dataset geometry."""

    written_cells: int
    total_cells: int

    def __post_init__(self) -> None:
        _validate_cell_counts(self)

    @property
    def complete(self) -> bool:
        return self.written_cells == self.total_cells


def _validate_cell_counts(coverage: DatasetCoverage | MonitorCoverage) -> None:
    if type(coverage.written_cells) is not int:
        raise TypeError("written_cells must be an integer")
    if type(coverage.total_cells) is not int:
        raise TypeError("total_cells must be an integer")
    if coverage.written_cells < 0:
        raise ValueError("written_cells must be non-negative")
    if coverage.total_cells <= 0:
        raise ValueError("total_cells must be positive")
    if coverage.written_cells > coverage.total_cells:
        raise ValueError("written_cells cannot exceed total_cells")


__all__ = ["DatasetCoverage", "MonitorCoverage"]
