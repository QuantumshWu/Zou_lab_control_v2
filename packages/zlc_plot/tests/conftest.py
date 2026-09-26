from __future__ import annotations

import os
import time

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def close_matplotlib_figures() -> None:
    yield
    plt.close("all")


@pytest.fixture
def logical_shape():
    """The (height, width, 4) a rendered raster of a preset must have.

    A fixture rather than an importable helper: with importlib import mode
    "from conftest import ..." resolves to whichever conftest the path finds
    first, which was a sibling package's.
    """

    def _shape(preset: str = "2x2") -> tuple[int, int, int]:
        height, width = _reference_logical_shape(preset)
        return (height, width, 4)

    return _shape


@pytest.fixture
def error_bars():
    """The error-bar artists a panel draws, as public artists.

    A native prepared scene rasters the bars without artists; materializing
    it builds back the artists these tests read.  ``visible_only`` drops a
    bar artist the scene keeps hidden.
    """

    from matplotlib.collections import PolyCollection

    def _bars(session, *, visible_only: bool = False) -> list:
        session._renderer._materialize_prepared_curve()
        return [
            artist
            for axes in session._renderer.figure.axes
            for artist in axes.collections
            if isinstance(artist, PolyCollection)
            and hasattr(artist, "_zlc_segment_buffer")
            and (artist.get_visible() or not visible_only)
        ]

    return _bars


@pytest.fixture
def qt_app():
    """The one QApplication the GUI tests drive, offscreen.

    PyQt5 is a dependency of this package, so a Qt that cannot start is a
    failure, not a skip.
    """

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from zlc_plot import ensure_qt5_application

    return ensure_qt5_application([])


@pytest.fixture
def pump_until(qt_app):
    """Process Qt events until ``predicate()`` holds or ``timeout`` seconds pass.

    Returns whether the predicate held; the caller asserts what it means.
    ``predicate=None`` pumps for the whole ``timeout``.
    """

    def _pump(predicate, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            qt_app.processEvents()
            if predicate is not None and predicate():
                return True
            if time.monotonic() >= deadline:
                return predicate is None
            time.sleep(0.002)

    return _pump


@pytest.fixture(scope="session")
def warm_kernel_cache() -> None:
    """The disk cache a render child's timed bound was measured on.

    A child that finds no machine code for a kernel compiles it: seconds,
    against the tens of milliseconds such a bound allows.  A fresh worktree,
    or any edit to a kernel module, therefore turned those tests red as
    timeouts that said nothing about the code -- so a cold cache is named
    here instead of surfacing as one.
    """

    from zlc_plot import _kernel_warm

    cold = _kernel_warm.cold_kernels()
    if cold:
        pytest.fail(
            f"{len(cold)} numba kernels have no current disk cache "
            f"({', '.join(cold[:3])}{', ...' if len(cold) > 3 else ''}): "
            "run bin\\warm_numba_cache.bat (zlc warm_numba) before timing a child",
            pytrace=False,
        )


def _reference_logical_shape(preset: str = "2x2") -> tuple[int, int]:
    """The (height, width) a plan of this preset produces, DERIVED.

    Restating a derived size in a test turns a geometry change into a puzzle:
    when the panel margins were corrected to the reference figure's own
    numbers, this read 357x480 and said only that something was different.
    Deriving it means the test asserts the ONE thing it is for -- that a
    rendered raster matches its plan -- and the plan stays the single source
    of the number.
    """

    from zlc_plot.config import DEFAULTS
    from zlc_plot.layout import resolve_surface

    plan = resolve_surface(
        preset,
        "curve",
        layout=DEFAULTS.layout,
        style=DEFAULTS.style,
    )
    width, height = plan.logical_size
    return int(height), int(width)
