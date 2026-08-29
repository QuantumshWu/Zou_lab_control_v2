"""Everything this checkout compiles, and the work that compiles it.

ONE OWNER for "what must be warm".  ``bin/warm_numba_cache`` used to call
the 3D scan-line module's own warmer, which knows about the five kernels in
that module and nothing about the nine in ``_raster_kernels`` -- so the nine
that draw every camera frame, every histogram and every uncertainty band
were compiled on the operator's first render of each, every time the cache
was cleared.  A tool that says it warms the kernel cache has to mean all of
it.

Two rules keep it that way, because a hand-written list of kernels is
exactly what went stale:

* the kernels are FOUND, not listed -- :func:`kernel_dispatchers` walks the
  modules and picks out numba's own dispatcher objects;
* the warmer CHECKS ITS OWN COVERAGE and names anything still cold, so a
  kernel nobody warmed is reported the first time anyone runs the tool
  rather than paid for silently on every fresh checkout.

The work itself is production work: real snapshots through real sessions,
because a warmup that renders something the product cannot is a warmup that
compiles a signature nothing uses.  That is not a hypothetical -- the 3D
warmer had drifted exactly that way, calling a renderer whose colour
contract had changed underneath it.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import sys
from typing import Any

import numpy as np

from . import _kernel_cache

#: The modules that define compiled kernels.  Import them here so discovery
#: sees every dispatcher; adding a third module means adding it here, and
#: :func:`kernel_dispatchers` will then report its kernels cold until the
#: work below asks for them.
_KERNEL_MODULE_NAMES = ("_raster_kernels", "_height3d_scanline")


def kernel_modules() -> tuple[Any, ...]:
    """The modules that define compiled kernels, imported."""

    import importlib  # noqa: PLC0415

    return tuple(
        importlib.import_module(f".{name}", __package__)
        for name in _KERNEL_MODULE_NAMES
    )


def kernel_dispatchers() -> dict[str, Any]:
    """Every compiled kernel this package defines -- found, never listed."""

    try:
        from numba.core.dispatcher import Dispatcher  # noqa: PLC0415
    except Exception:  # pragma: no cover - no numba, nothing to compile
        return {}
    found: dict[str, Any] = {}
    for module in kernel_modules():
        short = module.__name__.rsplit(".", 1)[-1]
        for name, value in vars(module).items():
            if isinstance(value, Dispatcher):
                found[f"{short}.{name}"] = value
    return found


def cold_kernels() -> tuple[str, ...]:
    """The kernels that have compiled nothing yet, by name."""

    return tuple(
        sorted(
            name
            for name, kernel in kernel_dispatchers().items()
            if not kernel.signatures
        )
    )


# ------------------------------------------------------------ the work
def _image_snapshot(
    height: int, width: int, dtype: Any, *, holes: bool = False
) -> Any:
    """One dense frame, built the way a camera producer builds one."""

    from zlc_data import (  # noqa: PLC0415
        COMPONENT,
        REPEAT,
        AxisId,
        AxisSpec,
        DatasetSchema,
        PointTable,
        ValidityContract,
        ValueSchema,
        owned_snapshot_from_arrays,
    )

    axes = (
        AxisSpec(AxisId("y"), "y", COMPONENT, height,
                 tuple(float(index) for index in range(height))),
        AxisSpec(AxisId("x"), "x", COMPONENT, width,
                 tuple(float(index) for index in range(width))),
    )
    schema = DatasetSchema(
        AxisSpec(AxisId("warm.repeat"), "repeat", REPEAT, 1, (0,)),
        PointTable(1),
        None,
        ValueSchema(axes, ValidityContract.value(), np.dtype(dtype), None),
    )
    generator = np.random.default_rng(0)
    if np.dtype(dtype).kind in "ui":
        values = generator.integers(0, 4000, (1, 1, height, width)).astype(dtype)
    else:
        values = generator.normal(0.0, 1.0, (1, 1, height, width)).astype(dtype)
        if holes:
            values[generator.random(values.shape) < 0.25] = np.nan
    return owned_snapshot_from_arrays(schema=schema, values=values, revision=1)


def _series_snapshot(repeats: int, points: int) -> Any:
    """One scalar series with repeats, the shape a band is formed over."""

    from zlc_data import (  # noqa: PLC0415
        REPEAT,
        SCAN_POINT,
        AxisId,
        AxisSpec,
        DatasetSchema,
        PointColumn,
        PointTable,
        ValueSchema,
        owned_snapshot_from_arrays,
    )

    coordinates = np.linspace(0.0, 1.0, points)
    schema = DatasetSchema(
        AxisSpec(AxisId("warm.repeat"), "repeat", REPEAT, repeats,
                 tuple(range(repeats))),
        PointTable(
            points,
            (
                PointColumn(
                    AxisId("x"),
                    "x",
                    SCAN_POINT,
                    PointColumn.NUMERIC,
                    tuple(float(value) for value in coordinates),
                ),
            ),
        ),
        None,
        ValueSchema.scalar(np.dtype(np.float64), None),
    )
    generator = np.random.default_rng(1)
    values = np.sin(coordinates)[None, :] + generator.normal(
        0.0, 0.1, (repeats, points)
    )
    # A scalar cell is still a cell: the block carries its trailing axis.
    values = values[..., None]
    return owned_snapshot_from_arrays(schema=schema, values=values, revision=1)


def _render(snapshot: Any, spec: Any, parameters: dict | None = None) -> None:
    from . import PlotSession  # noqa: PLC0415

    session = PlotSession(snapshot, spec)
    try:
        session.set_size("2x2")
        if parameters:
            session.set_parameters(dict(parameters))
        session.rgba()
    finally:
        session.close()


def representative_work() -> None:
    """Render what production renders, until every kernel has been asked.

    Each case names the kernels it is here for.  They are not asserted
    individually -- :func:`cold_kernels` checks the whole set afterwards,
    which is the check that keeps working when a kernel moves between
    cases.
    """

    from . import AxisRef, CurvePlot, HistogramPlot, ImagePlot  # noqa: PLC0415
    from . import _height3d_scanline  # noqa: PLC0415

    image = ImagePlot(AxisRef.data("x"), AxisRef.data("y"))

    # A raw unsigned frame small enough to draw a pixel per pixel: the
    # direct colour table and the pixel gather.
    _render(_image_snapshot(96, 96, np.uint16), image)

    # Oversampled frames, which reduce.  The SHAPES matter as much as the
    # dtypes: a source that halves evenly is served by the mip pyramid and
    # never reaches the area mean at all, so a power-of-two frame exercises
    # the unsigned kernel and a ragged one the floating kernel -- which is
    # the shape its own docstring measured, and the shape a real camera has.
    _render(_image_snapshot(2048, 2048, np.uint16), image)
    _render(_image_snapshot(1200, 1920, np.float32), image)
    _render(_image_snapshot(1200, 1920, np.float64), image)

    # The same, with holes: the masked block sum, which also counts.  A
    # masked source bypasses the pyramid, so its shape is free.
    _render(_image_snapshot(1200, 1920, np.float64, holes=True), image)

    series = _series_snapshot(8, 400)
    # The centred second moment behind an uncertainty band.
    _render(series, CurvePlot(AxisRef.point("x")), {"uncertainty": True})
    # Uniform binning and the masked extrema that choose its domain.
    _render(series, HistogramPlot())

    # The scan-line renderer's own five.  Its bare render reaches three of
    # them; the other two belong to the SCENE -- the edge-occlusion sampler
    # that decides which bar outlines are hidden, and the rim stroke -- and
    # only a real 3D panel draws those, which is why the module's own
    # warmer had been leaving them cold since it was written.
    _height3d_scanline.representative_render()
    _render(
        _image_snapshot(24, 32, np.float64),
        image,
        {"presentation": "height_bars"},
    )


# ------------------------------------------------------------ the warmer
def _fingerprint() -> str:
    """Toolchain plus the source of every module that defines a kernel."""

    import numba  # noqa: PLC0415

    parts = [sys.version.split()[0], np.__version__, numba.__version__]
    for module in sorted(kernel_modules(), key=lambda item: item.__name__):
        source = pathlib.Path(module.__file__).read_bytes()
        parts.append(f"{module.__name__}:{hashlib.sha256(source).hexdigest()}")
    return "|".join(parts)


def warm(force: bool = False) -> str:
    """Compile (or verify) every kernel's disk cache; returns the outcome."""

    from . import _height3d_raster, _raster_kernels  # noqa: PLC0415

    if not _raster_kernels.HAVE_NUMBA:
        return "numba is not installed; the numpy reference engines run"

    cache_dir = pathlib.Path(
        os.environ.get("NUMBA_CACHE_DIR") or _kernel_cache.kernel_cache_dir()
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    marker = cache_dir / "zlc_kernels.marker"
    fingerprint = _fingerprint()
    populated = any(cache_dir.glob("**/*.nbc"))
    if (
        not force
        and populated
        and marker.exists()
        and marker.read_text(encoding="utf-8") == fingerprint
    ):
        return "cache is current; nothing to do"

    previous_plot = _raster_kernels.ENGINE
    previous_h3d = _height3d_raster._ENGINE
    _raster_kernels.ENGINE = "numba"
    _height3d_raster._ENGINE = "numba"
    try:
        representative_work()
    finally:
        _raster_kernels.ENGINE = previous_plot
        _height3d_raster._ENGINE = previous_h3d

    total = len(kernel_dispatchers())
    cold = cold_kernels()
    if cold:
        # Reported, not written off: a marker written now would tell the
        # next run there is nothing to do, and the kernels named here would
        # go on being compiled during experiments forever.
        raise RuntimeError(
            f"{len(cold)} of {total} kernels were not warmed: "
            + ", ".join(cold)
            + ".  The representative work in _kernel_warm does not reach "
            "them; add the render that does."
        )
    marker.write_text(fingerprint, encoding="utf-8")
    return f"{total} kernels compiled and cached"


def main() -> int:
    """``warm_numba_cache``: compile-or-verify every kernel, say which.

    A MISSING DEPENDENCY IS NOT A FAILURE HERE -- ``warm`` says so and
    returns, because the numpy reference engines still draw.  So anything
    that reaches this handler is a defect in the warmer or a kernel, and the
    operator is told that rather than told to install something they have.
    """

    try:
        print(warm())
    except Exception as error:  # noqa: BLE001 -- this is a command-line front
        import traceback  # noqa: PLC0415

        traceback.print_exc()
        print(f"\nwarmup failed: {type(error).__name__}: {error}")
        print(
            "This is a defect in the warmer or a kernel, not a missing "
            "package: numba's absence is reported, never raised."
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
