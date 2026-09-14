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
import threading
import traceback
import tempfile
from typing import Any, Callable

import numpy as np

from . import _kernel_cache

#: The modules that define compiled kernels.  Import them here so discovery
#: sees every dispatcher; adding another kernel module means adding it here, and
#: :func:`kernel_dispatchers` will then report its kernels cold until the
#: work below asks for them.
_KERNEL_MODULE_NAMES = (
    "_raster_kernels",
    "_height3d_scanline",
    "_fit_compiled",
    "_fit_radial",
)


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
        provider = getattr(module, "production_dispatchers", None)
        values = (
            (
                (getattr(value, "py_func", value).__name__, value)
                for value in provider()
            )
            if callable(provider)
            else vars(module).items()
        )
        for name, value in values:
            if not isinstance(value, Dispatcher):
                continue
            # A function compiled INLINE has no machine code of its own:
            # it is copied into every kernel that calls it, and it never
            # gets a signature to be warm or cold with.  A stroke's edge
            # coverage is such a helper; listing it would make a warmer
            # that reaches every kernel report one it can never reach.
            if getattr(value, "targetoptions", {}).get("inline") == "always":
                continue
            found[f"{short}.{name}"] = value
    return found


def cached_signatures(kernel: Any) -> tuple[tuple[Any, ...], ...]:
    """Every signature of ``kernel``: this process's, and the disk cache's.

    A dispatcher only knows the signatures it compiled or loaded ITSELF, and
    a cache directory is written by many processes -- an experiment sealed
    its planes, a notebook did not, and each left its own machine code
    behind.  A check that looks only at the process in hand is blind to the
    pair on disk, which is exactly the pair that matters.  The index numba
    keeps beside its machine code lists what is cached for the current
    source; a stale index (the source moved on) lists nothing.
    """

    found: list[tuple[Any, ...]] = list(kernel.signatures)
    cache_file = getattr(getattr(kernel, "_cache", None), "_cache_file", None)
    load_index = getattr(cache_file, "_load_index", None)
    if callable(load_index):
        try:
            overloads = load_index()
        except (OSError, EOFError, ValueError, pickle_error()):
            overloads = {}
        found.extend(
            key[0] for key, filename in overloads.items()
            if pathlib.Path(cache_file._data_path(filename)).is_file()
        )
    return tuple(found)


def pickle_error() -> type[Exception]:
    import pickle  # noqa: PLC0415

    return pickle.UnpicklingError


def _argument_types(signature: Any) -> tuple[Any, ...]:
    """The argument types of an overload, whichever way it was listed.

    A dispatcher lists a lazily compiled overload as a tuple of argument
    types and an exactly compiled one (the fit callbacks, compiled to their
    ABI up front) as a ``Signature`` object; the disk index lists every
    overload as the tuple.  One overload can therefore appear in two
    spellings, and only its argument types say whether two entries are one.
    """

    return tuple(getattr(signature, "args", signature))


def _exact(signature: Any) -> tuple[str, ...]:
    """One overload's identity: its argument types, mutability included."""

    return tuple(str(kind) for kind in _argument_types(signature))


def _mutability_erased(signature: Any) -> tuple[Any, ...]:
    """The signature with every array's read-only flag forgotten."""

    from numba import types  # noqa: PLC0415

    return tuple(
        (str(kind.dtype), int(kind.ndim), str(kind.layout))
        if isinstance(kind, types.Array)
        else str(kind)
        for kind in _argument_types(signature)
    )


def duplicate_signatures(
    dispatchers: dict[str, Any] | None = None,
) -> tuple[str, ...]:
    """Kernels compiled more than once for one type, by name.

    Numba types an array's MUTABILITY, so a kernel handed a writable plane
    and the same kernel handed a read-only one are two compilations of one
    piece of code -- and which one a plane is, is an accident of whether
    something upstream had to copy it.  Sealing every input at the boundary
    is what makes this list empty; see ``_raster_kernels.readable``.

    Judged on every signature the kernel has, in this process or on disk
    (:func:`cached_signatures`): the regular-image promotion sat in the
    cache twice per dtype for as long as only the warmer's own process was
    consulted, because the warmer's inputs were all writable and the
    experiment's all sealed, and no single process ever saw both.
    """

    named: list[str] = []
    for name, kernel in (dispatchers or kernel_dispatchers()).items():
        exact_by_shape: dict[tuple[Any, ...], set[tuple[str, ...]]] = {}
        for signature in cached_signatures(kernel):
            exact_by_shape.setdefault(_mutability_erased(signature), set()).add(
                _exact(signature)
            )
        if any(len(exact) > 1 for exact in exact_by_shape.values()):
            named.append(name)
    return tuple(sorted(named))


def cold_kernels() -> tuple[str, ...]:
    """Kernels with neither a loaded signature nor a current disk cache.

    Loading a cached parent does not populate its compiled helpers' Python
    dispatchers; their empty signature lists do not mean they need compiling.
    """

    return tuple(
        sorted(
            name
            for name, kernel in kernel_dispatchers().items()
            if not cached_signatures(kernel)
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
        DomainSpec,
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
        DomainSpec(
            (1,),
            (AxisSpec(AxisId("warm.repeat"), "repeat", REPEAT, 1, (0,)),),
            ((0,),),
        ),
        DomainSpec((1,), (), ()),
        DomainSpec((height, width), axes),
        ValueSchema(ValidityContract.value(), np.dtype(dtype), None),
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
        DomainSpec,
        SCALAR_DOMAIN,
        ValueSchema,
        owned_snapshot_from_arrays,
    )

    coordinates = np.linspace(0.0, 1.0, points)
    schema = DatasetSchema(
        DomainSpec(
            (repeats,),
            (AxisSpec(AxisId("warm.repeat"), "repeat", REPEAT, repeats,
                      tuple(range(repeats))),),
            (tuple(range(repeats)),),
        ),
        DomainSpec(
            (points,),
            (AxisSpec(
                    AxisId("x"),
                    "x",
                    SCAN_POINT,
                    points,
                    tuple(float(value) for value in coordinates),
                ),),
            (tuple(range(points)),),
        ),
        SCALAR_DOMAIN,
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
        DomainSpec,
        ValidityContract,
        ValueSchema,
        owned_snapshot_from_arrays,
    )

    schema = DatasetSchema(
        DomainSpec(
            (repeats,),
            (AxisSpec(
                AxisId("warm.repeat"), "repeat", REPEAT,
                repeats, tuple(range(repeats)),
            ),),
            (tuple(range(repeats)),),
        ),
        DomainSpec(
            (points,),
            (
                AxisSpec(
                    AxisId("x"), "x", SCAN_POINT, 4,
                    tuple(float(index) for index in range(4)),
                ),
                AxisSpec(
                    AxisId("group"), "group", SCAN_POINT, points // 4,
                    tuple(float(index) for index in range(points // 4)),
                ),
            ),
            (
                tuple(index % 4 for index in range(points)),
                tuple(index // 4 for index in range(points)),
            ),
        ),
        DomainSpec(
            (sites,),
            (AxisSpec(AxisId("site"), "site", COMPONENT, sites,
                      tuple(float(index) for index in range(sites))),),
        ),
        ValueSchema(
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
    size: str = "2x2",
    fit: bool = False,
) -> None:
    from . import PlotSession  # noqa: PLC0415
    from .selectors import NumericRange  # noqa: PLC0415

    session = PlotSession(snapshot, spec, size=size, parameters=parameters)
    try:
        session.rgba()
        if fit:
            # THE MODEL THE PRODUCT WOULD OFFER, not one named here: the
            # registry sorts a target's default first, which is what a panel
            # opens on, and a name typed here would go stale the day the
            # catalogue changed.  A plot with no valid model for its target
            # simply has no fit to warm.
            #
            # And THROUGH ``configure``, which is the call a panel makes.
            # Warming with ``session.fit`` instead left the live request
            # machinery -- arming, the facet batch a grid takes, accepting
            # and presenting a result -- cold, and a child's first
            # configure(fit=) still cost 140 ms after everything else was
            # hot.
            models = session.fit_models
            if models:
                session.configure(fit={"model": str(models[0].model_id)})
        if not zoom_steps:
            return
        # A ZOOM IS NOT THE SAME WORK.  Cropping the viewport changes the
        # reduction ratio, so a frame that was reducing starts drawing
        # pixel for pixel through the direct colour table instead -- and a
        # cropped view is strided, so making it contiguous COPIES, which
        # is where a writable plane came from before every input was
        # sealed.  Warming only the opening view left an operator's first
        # wheel notch compiling.  The picture is the cell's trailing two
        # dimensions of the (repeat, point, ..., y, x) block.
        height, width = (
            int(size) for size in np.asarray(snapshot.block.values).shape[-2:]
        )
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


def _save(
    snapshot: Any,
    spec: Any,
    parameters: dict | None = None,
    *,
    zoom_steps: int = 0,
) -> None:
    """Render through the export path, which materializes what native leaves lazy.

    A native draw is the pixel consumer of a live image and rasterizes no
    fallback picture for it: the block reductions, the colour tables and the
    view-filling gather that turn a prepared front into RGBA answer only when
    the scene is MATERIALIZED -- a Save, or a facet overview.  Those are
    production renders too, and an operator's first Save compiling for a
    minute is the wheel-notch compile in another place, so the warmer asks
    for them the way Save does.
    """

    from . import PlotSession  # noqa: PLC0415
    from .selectors import NumericRange  # noqa: PLC0415

    session = PlotSession(snapshot, spec, size="2x2", parameters=parameters)
    try:
        with tempfile.TemporaryDirectory() as folder:
            target = pathlib.Path(folder) / "warm.png"
            session.save(target)
            if not zoom_steps:
                return
            height, width = (
                int(size) for size in np.asarray(snapshot.block.values).shape[-2:]
            )
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
                session.save(target)
    finally:
        session.close()


def representative_work(
    *, include_render: bool = True, include_compiled_fit: bool = True,
) -> None:
    """Render what production renders, until every kernel has been asked.

    Each case names the kernels it is here for.  They are not asserted
    individually -- :func:`cold_kernels` checks the whole set afterwards,
    which is the check that keeps working when a kernel moves between
    cases.
    """

    if include_compiled_fit:
        from . import _fit_compiled, _fit_radial  # noqa: PLC0415

        # Regular-image work shares the compiled solver and model callbacks.
        # Keep their complete samples together, including all storage dtypes.
        _fit_compiled.warm_production_cache()
        # Compiling is not the same as being RIGHT.  The warm above starts
        # every model AT its true parameters; this one solves a known
        # Gaussian from the model's own initializer and checks the numbers
        # that come back, which is the only thing here that would catch a
        # kernel that converges to the wrong answer.  It was written to be
        # called from the repository warmer and never was.
        _fit_compiled.self_check()
        _fit_radial.warm_production_cache()
    if not include_render:
        return

    from . import (  # noqa: PLC0415
        AxisRef,
        CurvePlot,
        FacetGridPlot,
        HistogramPlot,
        ImagePlot,
        RollingPlot,
    )
    from . import (  # noqa: PLC0415
        _height3d_scanline,
        _raster_kernels,
    )

    image = ImagePlot(AxisRef.cell_data("x"), AxisRef.cell_data("y"))

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
    # Every dtype now shares the wide-accumulating block-mean kernel, but
    # its input dtype remains part of the compiled signature. Small unsigned
    # fronts also exercise the direct colour-table path before reduction.
    for dtype in (np.uint8, np.uint16):
        _render(_image_snapshot(96, 96, dtype), image)
        _render(_image_snapshot(1200, 1920, dtype), image, zoom_steps=5)
    for dtype in (np.uint32, np.int16, np.int32, np.float32, np.float64):
        _render(_image_snapshot(1200, 1920, dtype), image, zoom_steps=5)
    for dtype in (np.float32, np.float64):
        # With holes: the masked block sum, which also counts.
        _render(_image_snapshot(1200, 1920, dtype, holes=True), image)
    # The same pictures materialized, as a Save materializes them: the
    # exact unsigned block sum and the direct colour table for a narrow
    # unsigned frame, the counting block mean and the float colour table
    # for a floating one, and the view-filling gather of a zoomed front.
    _save(_image_snapshot(96, 96, np.uint16), image)
    _save(_image_snapshot(1200, 1920, np.uint16), image, zoom_steps=2)
    _save(_image_snapshot(1200, 1920, np.float64, holes=True), image, zoom_steps=2)

    series = _series_snapshot(8, 400)
    # The centred second moment and fused curve validity/bounds pass.
    _render(series, CurvePlot(AxisRef.point("x")), {"uncertainty": True})
    # Uniform binning and the masked extrema that choose its domain -- and
    # the fit a histogram panel opens on, whose lines the kernel strokes.
    # The batch transform of the fit lines' vertices is asked only by a
    # frame painted after a fit has landed; every render without one left
    # it to compile on the operator's first fitted frame.
    _render(series, HistogramPlot(), fit=True)
    _render(
        _image_snapshot(24, 32, np.float64),
        FacetGridPlot(AxisRef.cell_data("y"), HistogramPlot()),
    )
    mixed = _mixed_snapshot()
    _render(mixed, ImagePlot(AxisRef.point("x"), AxisRef.cell_data("site")))
    _render(
        mixed,
        CurvePlot(AxisRef.cell_data("site"), group=AxisRef.point("group")),
        {"uncertainty": True},
    )
    # Dense grouped Cell data exposes a strided y view to the serial summary
    # scan. Warm its actual A-layout signature without copying it to C.
    _render(
        _image_snapshot(8, 16, np.float64),
        CurvePlot(AxisRef.cell_data("x"), group=AxisRef.cell_data("y")),
    )
    _render(
        _mixed_snapshot(repeats=8, points=8, sites=8),
        RollingPlot(group=AxisRef.cell_data("site")),
    )
    # The fused value+count leading reduction exists only for a genuinely
    # holey, C-laid-out floating tensor; an all-valid curve takes NumPy's
    # plain reduction and a transposed tensor deliberately stays on its exact
    # NumPy reference instead of being copied merely to reach the kernel.
    _render(
        _mixed_snapshot(repeats=8, points=1024, sites=8, holes=True),
        CurvePlot(AxisRef.cell_data("site")),
        {"uncertainty": False},
    )

    # Fit overview ellipse rings are born only after an async fit result lands.
    # Text remains on the shared Matplotlib MathText owner.
    front = np.full((32, 32, 4), 255, dtype=np.uint8)
    _raster_kernels.raster_fit_ellipses(
        _raster_kernels.readable(
            np.asarray(((16.0, 16.0, 6.0, 4.0),), dtype=np.float64)
        ),
        _raster_kernels.readable(
            np.asarray(((255, 128, 0, 180),), dtype=np.uint8)
        ),
        _raster_kernels.readable(np.asarray((2.0,), dtype=np.float64)),
        _raster_kernels.readable(
            np.asarray(((255, 128, 0, 255),), dtype=np.uint8)
        ),
        _raster_kernels.readable(np.asarray((2.0,), dtype=np.float64)),
        _raster_kernels.readable(
            np.asarray(((0, 0, 32, 32),), dtype=np.int32)
        ),
        front,
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


# ------------------------------------------------------ a fresh process
def warm_process(proceed: Callable[[], bool] = lambda: True) -> None:
    """The work a fresh process pays once, done before anyone asks for it.

    The disk cache spares a process the COMPILE; it does not spare it the
    rest of a first render.  Matplotlib's figure, axes and text modules
    import on first use, the first text measured loads a font, numba
    refreshes its typing context and reads each kernel's machine code off
    the disk the first time that kernel is called.  Together that was two
    thirds of a second on the operator's first panel -- every console,
    every day -- and none of it depends on what that panel shows.  A
    render child calls this the moment it starts, on a thread of its own,
    so the first panel finds the process as warm as the second.

    A short slice of :func:`representative_work`, and a cheap one: a
    request that arrives while this runs shares the process with it, so
    every second here is a second that request may wait.  The pictures a
    panel most often opens on -- a camera frame drawn larger than it is
    and one reduced, a floating derived plane, a histogram, a curve with
    its band, a grid of cells -- on frames just big enough to take each
    path; not the zooms, saves or 3D scene, which have first uses of their
    own.

    ORDER IS THE WHOLE DESIGN, because this gets cut off.  ``proceed`` is
    asked before every step and answers False from the moment a panel is
    built here, so a child taken early runs only the front of this list --
    and the front had better hold what every renderer pays once, cheapest
    first.  Measured on this machine, in the order they now run:

    * THE SOLVER IMPORTS, on a thread of their own, started first and
      never asked ``proceed``.  They buy the most: without them the
      operator's FIRST fit costs 0.6 s against one or two milliseconds
      after, and a live fit has a one-second deadline to expire against.
      They are also the only part of this that is pure module loading, so
      they overlap with the drawing below rather than queueing behind it
      -- 1.65 s for the pair against 2.08 s in turn -- and, being off this
      thread, they finish even when a panel cuts the rest of this short.
    * the grid of camera frames, 0.5 s: Matplotlib's own import, the first
      text measured loading the font, and the raster kernels' first
      dispatch.  It is what a console opens on.
    * the other grids -- histograms, curves over a data axis -- and then
      A GRID'S WORTH OF EMPTY CELLS, 0.16 s, kept in :data:`CELL_RESERVE`.
      The reserve is single-shot and any grid takes it, so it must follow
      every grid drawn here; and nothing else, because sixty-four cells were
      the larger half of a grid's mount and a child taken at two seconds
      used to be taken before this ran -- it sat behind the fits and the
      picture variety.  Measured, a sixty-four cell mount on a child warmed
      three seconds was 693 ms with the reserve still ahead and 368 once
      it had run.
    * a curve's fit and a histogram's fit: numba's first dispatch of the
      fit kernels, which needs the imports above and so waits for them.
    * the remaining picture variety, which is the only part that is about
      what a panel happens to show rather than what every panel pays.
    * the 3D scene, last: 72 ms, and only for a panel presented as height
      bars.

    Listed last, as the solvers were, none of it ran at all: a child is
    taken about a second into its warming, and the operator paid the fit
    on the first shot of a running experiment.
    """

    from . import (  # noqa: PLC0415
        AxisRef,
        CurvePlot,
        FacetGridPlot,
        HistogramPlot,
        ImagePlot,
    )

    def load_solvers() -> None:
        """Everything a first fit costs that has nothing to do with drawing.

        The engine's modules, and then one fit per target family through
        the engine rather than through a plot: numba's first dispatch of
        the fit kernels is the other half of what a first fit costs, and it
        needs no figure, no font and no Matplotlib.  Doing it here is what
        makes it survive the panel that cuts the drawing below short.

        No scipy.  The solvers are compiled; the classifier threshold is a
        quadratic's root; the doublet seed finds its own peaks; the camera
        seed filters its own medians; the site rings measure their own
        distances.  Importing scipy.optimize and scipy.signal here for the
        one scalar fallback no registered model reaches cost every child
        0.85 s of this thread and 45 MB it kept.
        """

        from .fit import (  # noqa: PLC0415
            FitEngine,
            FitTarget,
            RegularImageFitInput,
            default_fit_registry,
        )

        engine = FitEngine()
        registry = default_fit_registry()
        rows, columns = 24, 32
        y_grid, x_grid = np.mgrid[0:rows, 0:columns].astype(np.float64)
        series_x = np.linspace(-6.0, 6.0, 256)
        samples = np.concatenate(
            [
                np.linspace(8.0, 16.0, 600),
                np.linspace(34.0, 46.0, 200),
            ]
        )
        counts, edges = np.histogram(samples, bins=40)
        inputs = {
            FitTarget.SERIES: (
                (series_x,),
                5.0 * np.exp(-0.5 * ((series_x - 0.4) / 1.3) ** 2) + 0.8,
            ),
            FitTarget.HISTOGRAM: (
                (0.5 * (edges[:-1] + edges[1:]),),
                counts.astype(np.float64),
            ),
            FitTarget.IMAGE: (
                (y_grid.ravel(), x_grid.ravel()),
                (
                    120.0
                    * np.exp(
                        -0.5
                        * (
                            ((x_grid - 16.0) / 4.0) ** 2
                            + ((y_grid - 12.0) / 5.0) ** 2
                        )
                    )
                    + 4.0
                ).ravel(),
            ),
        }
        for target, (coordinates, observations) in inputs.items():
            models = registry.models_for(target)
            if not models:
                continue
            try:
                engine.fit(models[0], coordinates, observations)
            except Exception:  # noqa: BLE001 -- warming, never fatal
                traceback.print_exc()

    def load_batch_solvers(proceed: Callable[[], bool]) -> None:
        """The solver entries a GRID takes, once the drawing is warm.

        A single fit takes the compiled routine's serial wrapper and a
        grid's cells the same routine under prange; a camera fit hands the
        engine a whole image and takes the separable stripe solver.  Three
        dispatches, and warming the first leaves the operator the other
        two: measured, a four-cell grid's first fit cost 104 ms in a fresh
        process against 26 in the same one afterwards, and an image
        panel's first fit 15 ms over its second.

        They are LAST and they ask ``proceed``, unlike the imports above,
        because compiling a parallel entry saturates the machine: run on
        the uninterruptible thread it delayed a mounting panel's picture by
        six seconds and let a live fit hit its one-second deadline.  A
        child taken before it gets here pays the dispatch on its first grid
        fit, which is what it paid before this existed; a spare child that
        is left alone does not.
        """

        from .fit import (  # noqa: PLC0415
            FitEngine,
            FitTarget,
            RegularImageFitInput,
            default_fit_registry,
        )

        engine = FitEngine()
        registry = default_fit_registry()
        rows, columns = 24, 32
        y_grid, x_grid = np.mgrid[0:rows, 0:columns].astype(np.float64)
        series_x = np.linspace(-6.0, 6.0, 256)
        series_y = 5.0 * np.exp(-0.5 * ((series_x - 0.4) / 1.3) ** 2) + 0.8
        samples = np.concatenate(
            [np.linspace(8.0, 16.0, 600), np.linspace(34.0, 46.0, 200)]
        )
        counts, edges = np.histogram(samples, bins=40)
        batches = {
            FitTarget.SERIES: ((series_x,), series_y),
            FitTarget.HISTOGRAM: (
                (0.5 * (edges[:-1] + edges[1:]),),
                counts.astype(np.float64),
            ),
        }
        for target, (coordinates, observations) in batches.items():
            if not proceed():
                return
            models = registry.models_for(target)
            if not models:
                continue
            try:
                engine.fit_batch(
                    models[0],
                    (coordinates, coordinates),
                    (observations, observations),
                )
            except Exception:  # noqa: BLE001 -- warming, never fatal
                traceback.print_exc()
        image_models = registry.models_for(FitTarget.IMAGE)
        if image_models and proceed():
            frame = (
                120.0
                * np.exp(
                    -0.5
                    * (
                        ((x_grid - 16.0) / 4.0) ** 2
                        + ((y_grid - 12.0) / 5.0) ** 2
                    )
                )
                + 4.0
            )
            regular = RegularImageFitInput(
                np.arange(columns, dtype=np.float64),
                np.arange(rows, dtype=np.float64),
                frame,
            )
            try:
                engine.fit(image_models[0], regular)
                if proceed():
                    engine.fit_batch(
                        image_models[0], (regular, regular), (None, None)
                    )
            except Exception:  # noqa: BLE001 -- warming, never fatal
                traceback.print_exc()

    solvers = threading.Thread(
        target=load_solvers, name="zlc-warm-solvers", daemon=True
    )
    solvers.start()

    if not proceed():
        return
    image = ImagePlot(AxisRef.cell_data("x"), AxisRef.cell_data("y"))
    camera = _image_snapshot(96, 128, np.uint16)
    _render(camera, FacetGridPlot(None, image), size="4x4")
    if not proceed():
        return
    # The same grid over a floating frame: a grid's cells colour a small
    # frame straight from the table, one kernel per dtype, and a grid of
    # small float images still loaded the float one on its first frame.
    _render(
        _image_snapshot(96, 128, np.float32), FacetGridPlot(None, image), size="4x4"
    )
    if not proceed():
        return
    _render(
        _image_snapshot(24, 32, np.float64),
        FacetGridPlot(AxisRef.cell_data("y"), HistogramPlot()),
        size="2x2",
    )
    if not proceed():
        return
    # A grid of curves over a data axis: each cell reads its y through a
    # strided view, which is the summary scan's A-layout signature -- and
    # a frame grid still loaded it on its first frame after everything
    # else here had run.
    mixed = _mixed_snapshot(holes=True)
    _render(
        mixed,
        FacetGridPlot(AxisRef.cell_data("site"), CurvePlot(AxisRef.point("x"))),
        size="2x2",
    )
    if not proceed():
        return
    # A GRID'S WORTH OF CELLS, RIGHT AFTER THE LAST GRID DRAWN HERE.  The
    # reserve is single-shot and any grid takes it, so it must follow
    # every grid this warming draws -- and nothing else: a grid's cells
    # are the larger half of mounting one, none of it depends on the data,
    # and a child taken at two seconds used to be taken before this ran,
    # because it sat behind the fits and the picture variety.  Measured,
    # a sixty-four cell mount on a child warmed three seconds was 693 ms
    # with the reserve still ahead of it and 368 once it had run.
    from .config import DEFAULTS  # noqa: PLC0415
    from .rendering import CELL_RESERVE  # noqa: PLC0415

    CELL_RESERVE.fill(DEFAULTS.style, int(DEFAULTS.layout.facet_max_cells))
    if not proceed():
        return
    # The fits below solve, so they need what that thread was loading.
    solvers.join()
    series = _series_snapshot(8, 400)
    _render(
        series,
        CurvePlot(AxisRef.point("x")),
        {"uncertainty": True},
        size="2x2",
        fit=True,
    )
    if not proceed():
        return
    _render(series, HistogramPlot(), size="2x2", fit=True)
    if not proceed():
        return
    _render(camera, image, size="4x4", fit=True)
    for dtype in (np.uint16, np.float32, np.float64):
        if not proceed():
            return
        _render(_image_snapshot(600, 800, dtype), image, size="2x2")
    # THE SIGNATURES A PANEL'S SHAPE SELECTS, not its kind.  A curve over
    # dense cell data reads its y through a strided view; a tensor group
    # strides its validity too; an image over a point axis and a data axis
    # aggregates axis codes.  Measured after everything above, a
    # tensor-group panel and a mixed-axes image each still loaded one
    # kernel on their first frame -- 10 to 20 ms from a cache that has
    # been read before, and a Windows first read of a fresh one: 270 ms
    # against 34 on the frame grid that led to this list.
    if not proceed():
        return
    _render(
        _image_snapshot(8, 16, np.float64),
        CurvePlot(AxisRef.cell_data("x"), group=AxisRef.cell_data("y")),
        size="2x2",
    )
    if not proceed():
        return
    _render(
        mixed,
        CurvePlot(AxisRef.cell_data("site"), group=AxisRef.point("group")),
        size="2x2",
    )
    if not proceed():
        return
    _render(mixed, ImagePlot(AxisRef.point("x"), AxisRef.cell_data("site")), size="2x2")
    load_batch_solvers(proceed)
    if proceed():
        # And the 3D scene, which was left out of the list above because it
        # is a presentation only some panels open on.  It costs the process
        # 72 ms the first time and nothing after -- measured, a second 3D
        # panel in the same process opens in 98 ms against the first one's
        # 170 -- and a render child hosts ONE panel, so without this every
        # 3D panel there has ever been paid it.
        _render(
            _image_snapshot(24, 32, np.float64),
            image,
            {"presentation": "height_bars"},
            size="2x2",
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
    current = fingerprint.split("|")
    previous = marker.read_text(encoding="utf-8").split("|") if marker.exists() else []
    needed = (
        set(_KERNEL_MODULE_NAMES)
        if force or current[:3] != previous[:3]
        else {
            entry.split(":", 1)[0].rsplit(".", 1)[-1]
            for entry in current[3:] if entry not in previous[3:]
        }
    )
    # A matching marker is not proof that the individual caches still exist.
    # Conversely, one signature is not proof that a changed module's full
    # dtype/layout sample set has run: its source fingerprint still selects it.
    needed.update(name.split(".", 1)[0] for name in cold_kernels())
    if not needed:
        return "cache is current; nothing to do"

    dispatchers = kernel_dispatchers()
    before = {
        name: (sum(kernel.stats.cache_misses.values()), sum(kernel.stats.cache_hits.values()))
        for name, kernel in dispatchers.items()
    }
    previous_plot = _raster_kernels.ENGINE
    previous_h3d = _height3d_raster._ENGINE
    _raster_kernels.ENGINE = "numba"
    _height3d_raster._ENGINE = "numba"
    try:
        representative_work(
            include_render=bool(needed & {"_raster_kernels", "_height3d_scanline"}),
            include_compiled_fit=bool(needed & {"_fit_compiled", "_fit_radial"}),
        )
    finally:
        _raster_kernels.ENGINE = previous_plot
        _height3d_raster._ENGINE = previous_h3d

    total = len(dispatchers)
    twins = duplicate_signatures(dispatchers)
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
    compiled = sum(
        sum(kernel.stats.cache_misses.values()) - before[name][0]
        for name, kernel in dispatchers.items()
    )
    loaded = sum(
        sum(kernel.stats.cache_hits.values()) - before[name][1]
        for name, kernel in dispatchers.items()
    )
    return (
        f"{total} kernels verified; {compiled} production signatures compiled, "
        f"{loaded} loaded from disk"
    )


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
