"""Lightweight immutable raster values shared with frontend processes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

import numpy as np
from ._axis_transform import AxisTransform
from ._validation import finite_real, integer, optional_nonempty_text, readonly_copy, text
from .selectors import NumericRange, SelectorSnapshot, SelectorState


ValueT = TypeVar("ValueT")


@dataclass(frozen=True, slots=True)
class RasterBuffer:
    """One tightly packed RGBA8888 image, immutable while it is held."""

    width: int
    height: int
    pixels: Any

    def __post_init__(self) -> None:
        width = integer(self.width, "raster width", minimum=1)
        height = integer(self.height, "raster height", minimum=1)
        try:
            view = memoryview(self.pixels)
        except TypeError as error:
            raise TypeError("raster pixels must be a buffer") from error
        if not view.readonly:
            raise TypeError("raster pixels must be read-only")
        if view.nbytes != width * height * 4:
            raise ValueError("RGBA8888 byte length does not match raster dimensions")

    def as_rgba(self, *, copy: bool = False) -> np.ndarray:
        if not isinstance(copy, bool):
            raise TypeError("copy must be a boolean")
        if (
            isinstance(self.pixels, np.ndarray)
            and self.pixels.dtype == np.dtype(np.uint8)
            and self.pixels.size == self.width * self.height * 4
        ):
            # Preserve a shared-memory ndarray subclass: its Python base owns
            # the remote lease, while np.frombuffer would retain only a raw
            # exported pointer and could outlive the released segment.
            rgba = self.pixels.reshape(self.height, self.width, 4)
        else:
            rgba = np.frombuffer(self.pixels, dtype=np.uint8).reshape(
                self.height, self.width, 4
            )
        return readonly_copy(rgba) if copy else rgba

    def encode(self, format: str = "PNG", **options: object) -> bytes:
        from io import BytesIO

        from PIL import Image

        selected_format = text(format, "image format").upper()
        output = BytesIO()
        Image.frombytes(
            "RGBA", (self.width, self.height), self.pixels
        ).save(output, format=selected_format, **options)
        return output.getvalue()

    def save(
        self,
        path: str | Path,
        *,
        format: str | None = None,
        **options: object,
    ) -> None:
        from zlc_durable import atomic_write_bytes

        target = Path(path)
        selected = target.suffix.removeprefix(".") if format is None else format
        atomic_write_bytes(
            target, self.encode(text(selected, "image format"), **options)
        )


@dataclass(frozen=True, slots=True)
class RasterIdentity:
    """Exact session state represented by one accepted raster."""

    host_id: str
    sequence: int
    data_generation: str | None
    data_revision: int
    image_overlay_revision: int | None
    display_revision: int
    layout_revision: int
    kind: str
    preset: str

    def __post_init__(self) -> None:
        for name in (
            "sequence", "data_revision", "display_revision", "layout_revision"
        ):
            value = integer(getattr(self, name), name, minimum=0)
            assert value is not None
            object.__setattr__(self, name, value)
        for name in ("host_id", "kind", "preset"):
            object.__setattr__(self, name, text(getattr(self, name), name))
        object.__setattr__(
            self,
            "data_generation",
            optional_nonempty_text(self.data_generation, "data_generation"),
        )
        object.__setattr__(
            self,
            "image_overlay_revision",
            integer(
                self.image_overlay_revision,
                "image_overlay_revision",
                minimum=0,
                optional=True,
            ),
        )

    def same_surface(self, other: object) -> bool:
        return isinstance(other, RasterIdentity) and (
            self.host_id,
            self.display_revision,
            self.layout_revision,
            self.kind,
            self.preset,
        ) == (
            other.host_id,
            other.display_revision,
            other.layout_revision,
            other.kind,
            other.preset,
        )


@dataclass(frozen=True, slots=True)
class RasterInteractionMap:
    """Everything pointer handling may read from the exact painted front."""

    axes: tuple[AxisTransform, ...]
    selectors: tuple[SelectorState, ...]
    color_limits: NumericRange | None = None
    facet_focus_index: int | None = None

    def __post_init__(self) -> None:
        axes = tuple(self.axes)
        if not axes or any(not isinstance(value, AxisTransform) for value in axes):
            raise ValueError("interaction map requires AxisTransform values")
        selectors = SelectorSnapshot(tuple(self.selectors)).committed
        color_limits = self.color_limits
        if color_limits is not None and not isinstance(color_limits, NumericRange):
            raise TypeError("color_limits must be NumericRange or None")
        focus = integer(
            self.facet_focus_index,
            "facet_focus_index",
            minimum=0,
            optional=True,
        )
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "selectors", selectors)
        object.__setattr__(self, "color_limits", color_limits)
        object.__setattr__(self, "facet_focus_index", focus)


@dataclass(frozen=True, slots=True)
class RasterFront:
    """One complete immutable frontend value promoted atomically."""

    identity: RasterIdentity
    buffer: RasterBuffer
    logical_size: tuple[int, int]
    logical_dpi: float
    device_pixel_ratio: float
    interaction: RasterInteractionMap

    def __post_init__(self) -> None:
        if not isinstance(self.identity, RasterIdentity):
            raise TypeError("front identity must be RasterIdentity")
        if not isinstance(self.buffer, RasterBuffer):
            raise TypeError("front buffer must be RasterBuffer")
        width, height = tuple(self.logical_size)
        object.__setattr__(
            self,
            "logical_size",
            (
                integer(width, "logical width", minimum=1),
                integer(height, "logical height", minimum=1),
            ),
        )
        logical_dpi = finite_real(self.logical_dpi, "logical dpi")
        if logical_dpi <= 0.0:
            raise ValueError("logical dpi must be positive")
        ratio = finite_real(self.device_pixel_ratio, "device pixel ratio")
        if ratio <= 0.0:
            raise ValueError("device pixel ratio must be positive")
        if not isinstance(self.interaction, RasterInteractionMap):
            raise TypeError("front interaction must be RasterInteractionMap")


@dataclass(frozen=True, slots=True)
class RasterOperation(Generic[ValueT]):
    """A worker result and the exact front painted after that result."""

    value: ValueT
    front: RasterFront


__all__ = [
    "RasterBuffer",
    "RasterFront",
    "RasterIdentity",
    "RasterInteractionMap",
    "RasterOperation",
]
