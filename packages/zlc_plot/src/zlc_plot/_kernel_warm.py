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


def duplicate_signatures() -> tuple[str, ...]:
    """Kernels compiled more than once for one type, by name.

    Numba types an array's MUTABILITY, so a kernel handed a writable plane
    and the same kernel handed a read-only one are two compilations of one
    piece of code -- and which one a plane is, is an accident of whether
    something upstream had to copy it.  Sealing every input at the boundary
    is what makes this list empty; see ``_raster_kernels.readable``.
    """

    import re  # noqa: PLC0415

    flags = re.compile(
        r"Array\((\w+), (\d+), '(\w)', (?:True|False), aligned=(?:True|False)\)"
    )
    named: list[str] = []
    for name, kernel in kernel_dispatchers().items():
        seen: set[str] = set()
        for signature in kernel.signatures:
            shape = flags.sub(
                lambda match: f"Array({match.group(1)},{match.group(2)},{match.group(3)})",
                str(signature),
            )
            if shape in seen:
                named.append(name)
                break
            seen.add(shape)
    return tuple(sorted(named))


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


def _mixed_snapshot(
    *, repeats: int = 4, points: int = 12, sites: int = 5,
    holes: bool = False,
) -> Any:
    """One point-coordinate × data-axis block for joint-axis kernels."""

    from zlc_data import (  # noqa: PLC0415
        COMPONENT,
        REPEAT,
        SCAN_POINT,
        AxisId,
        AxisSpec,
        DatasetSchema,
        PointColumn,
        PointTable,
        ValidityContract,
        ValueSchema,
        owned_snapshot_from_arrays,
    )

    schema = DatasetSchema(
        AxisSpec(
            AxisId("warm.repeat"), "repeat", REPEAT,
            repeats, tuple(range(repeats)),
        ),
        PointTable(
            points,
            (
                PointColumn(
                    AxisId("x"), "x", SCAN_POINT, PointColumn.NUMERIC,
                    tuple(float(index % 4) for index in range(points)),
                ),
                PointColumn(
                    AxisId("group"), "group", SCAN_POINT, PointColumn.NUMERIC,
                    tuple(float(index // 4) for index in range(points)),
                ),
            ),
        ),
        None,
        ValueSchema(
            (AxisSpec(AxisId("site"), "site", COMPONENT, sites,
                      tuple(float(index) for index in range(sites))),),
            ValidityContract.value(),
            np.dtype(np.float64),
            None,
        ),
    )
    values = np.random.default_rng(2).normal(size=(repeats, points, sites))
    if holes:
        values[1::3, 2::11, :] = np.nan
    return owned_snapshot_from_arrays(schema=schema, values=values, revision=1)


def _render(
    snapshot: Any,
    spec: Any,
    parameters: dict | None = None,
    *,
    zoom_steps: int = 0,
) -> None:
    from . import PlotSession  # noqa: PLC0415
    from .selectors import NumericRange  # noqa: PLC0415

    session = PlotSession(snapshot, spec)
    try:
        session.set_size("2x2")
        if parameters:
            session.set_parameters(dict(parameters))
        session.rgba()
        if not zoom_steps:
            return
        # A ZOOM IS NOT THE SAME WORK.  Cropping the viewport changes the
        # reduction ratio, so a frame that was reducing starts drawing
        # pixel for pixel through the direct colour table instead -- and a
        # cropped view is strided, so making it contiguous COPIES, which
        # is where a writable plane came from before every input was
        # sealed.  Warming only the opening view left an operator's first
        # wheel notch compiling.
        height, width = _plane_shape(snapshot)
        span = float(width)
        for _ in range(zoom_steps):
            span /= 1.7
            half = span / 2.0
            session.set_viewport(
                NumericRange(width / 2.0 - half, width / 2.0 + half),
                NumericRange(
                    height / 2.0 - half * height / width,
                    height / 2.0 + half * height / width,
                ),
            )
            session.rgba()
    finally:
        session.close()


def _plane_shape(snapshot: Any) -> tuple[int, int]:
    shape = np.asarray(snapshot.block.values).shape
    return int(shape[-3]), int(shape[-2])


def representative_work() -> None:
    """Render what production renders, until every kernel has been asked.

    Each case names the kernels it is here for.  They are not asserted
    individually -- :func:`cold_kernels` checks the whole set afterwards,
    which is the check that keeps working when a kernel moves between
    cases.
    """

    from . import (  # noqa: PLC0415
        AxisRef,
        CurvePlot,
        FacetGridPlot,
        HistogramPlot,
        ImagePlot,
    )
    from . import _height3d_scanline  # noqa: PLC0415

    image = ImagePlot(AxisRef.data("x"), AxisRef.data("y"))

    # EVERY DTYPE A PRODUCER PUBLISHES IS ANOTHER COMPILE.  A camera is
    # unsigned and may be either width; a derived plane is floating and may
    # be either width; a signed or wide integer plane is neither.  Warming
    # one of them leaves the others to the operator's first frame of each.
    #
    # Frame SIZE is not a type -- numba does not see a shape -- but it does
    # decide which kernel runs at all: a frame small enough to draw pixel
    # for pixel takes the direct colour table, an oversampled one reduces
    # and is then coloured from the float mean.  A zoom crosses between the
    # two, which is the wheel notch that used to compile mid-gesture.
    #
    # The narrow unsigned dtypes are the ones whose block sums are provably
    # exact, so they alone take the integer kernel; everything else reduces
    # through the floating one.
    for dtype in (np.uint8, np.uint16):
        _render(_image_snapshot(96, 96, dtype), image)
        _render(_image_snapshot(1200, 1920, dtype), image, zoom_steps=5)
    for dtype in (np.uint32, np.int16, np.int32, np.float32, np.float64):
        _render(_image_snapshot(1200, 1920, dtype), image, zoom_steps=5)
    for dtype in (np.float32, np.float64):
        # With holes: the masked block sum, which also counts.
        _render(_image_snapshot(1200, 1920, dtype, holes=True), image)

    series = _series_snapshot(8, 400)
    # The centred second moment behind an uncertainty band.
    _render(series, CurvePlot(AxisRef.point("x")), {"uncertainty": True})
    # Uniform binning and the masked extrema that choose its domain.
    _render(series, HistogramPlot())
    _render(
        _image_snapshot(24, 32, np.float64),
        FacetGridPlot(AxisRef.data("y"), HistogramPlot()),
    )
    mixed = _mixed_snapshot()
    _render(mixed, ImagePlot(AxisRef.point("x"), AxisRef.data("site")))
    _render(
        mixed,
        CurvePlot(AxisRef.data("site"), group=AxisRef.point("group")),
        {"uncertainty": True},
    )
    # The fused value+count leading reduction exists only for a genuinely
    # holey, C-laid-out floating tensor; an all-valid curve takes NumPy's
    # plain reduction and a transposed tensor deliberately stays on its exact
    # NumPy reference instead of being copied merely to reach the kernel.
    _render(
        _mixed_snapshot(repeats=8, points=1024, sites=8, holes=True),
        CurvePlot(AxisRef.data("site")),
        {"uncertainty": False},
    )

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
    twins = duplicate_signatures()
    if twins:
        # Two compilations of one kernel that differ only in whether their
        # input was writable is not coverage, it is waste -- and it means an
        # input reached a kernel without being sealed.  See
        # ``_raster_kernels.readable``.
        raise RuntimeError(
            "these kernels compiled twice for the same code, differing only "
            "in an input's mutability: " + ", ".join(twins)
            + ".  An input reached them without going through "
            "_raster_kernels.readable."
        )
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
    signatures = sum(len(kernel.signatures) for kernel in kernel_dispatchers().values())
    return f"{total} kernels, {signatures} signatures compiled and cached"


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
