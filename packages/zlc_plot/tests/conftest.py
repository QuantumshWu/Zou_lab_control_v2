from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gui: requires an offscreen Qt backend")


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
