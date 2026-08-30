"""The warmer warms ALL of it, and knows when it does not.

``bin/warm_numba_cache`` called the 3D scan-line module's own warmer, which
knew about that module and nothing about ``_raster_kernels`` -- so the nine
kernels that draw every camera frame, every histogram and every uncertainty
band were compiled during experiments, every time the cache was cleared.
And the 3D warmer reached only three of its own five, because two belong to
the SCENE and it never drew one.

Nothing here compiles: these assert the two properties that let the warmer
notice such a gap by itself -- that it finds every kernel, and that the work
it runs is work the product can actually do.
"""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg", force=True)

import pytest

from zlc_plot import _kernel_warm


def _package_root() -> pathlib.Path:
    import zlc_plot

    return pathlib.Path(zlc_plot.__file__).parent


def test_every_module_that_defines_a_kernel_is_looked_in() -> None:
    """Discovery is derived from the sources, not from memory.

    A hand-written list is what went stale.  This one is written down --
    modules must be imported before their dispatchers exist -- so the list
    is checked against the tree instead of trusted.
    """

    declared = set(_kernel_warm._KERNEL_MODULE_NAMES)
    found = {
        path.stem
        for path in _package_root().glob("*.py")
        if "@njit" in path.read_text(encoding="utf-8")
    }
    assert found, "this test is meaningless if it cannot see any kernels"
    assert found == declared, (
        f"modules defining @njit kernels: {sorted(found)}; "
        f"the warmer looks in: {sorted(declared)}"
    )


def test_the_warmer_finds_every_kernel_it_looks_for() -> None:
    """Every declared production dispatcher is visible to the warmer."""

    pytest.importorskip("numba")
    found = _kernel_warm.kernel_dispatchers()
    declared = 0
    for module in _kernel_warm.kernel_modules():
        provider = getattr(module, "production_dispatchers", None)
        declared += (
            len(provider())
            if callable(provider)
            else sum(
                1
                for line in pathlib.Path(module.__file__)
                .read_text(encoding="utf-8")
                .splitlines()
                if line.startswith("@njit")
            )
        )
    assert declared > 0
    assert len(found) == declared, sorted(found)


def test_the_work_the_warmer_runs_is_work_the_product_can_do() -> None:
    """On the reference engines, so contract drift is caught without a compile.

    The 3D warmer had drifted exactly that way -- calling a renderer whose
    colour contract had changed underneath it -- and raised for every
    operator who ran the tool, under a message about a missing dependency.
    """

    from zlc_plot import _height3d_raster, _raster_kernels

    previous_plot = _raster_kernels.ENGINE
    previous_h3d = _height3d_raster._ENGINE
    _raster_kernels.ENGINE = "numpy"
    _height3d_raster._ENGINE = "numpy"
    try:
        _kernel_warm.representative_work(include_compiled_fit=False)
    finally:
        _raster_kernels.ENGINE = previous_plot
        _height3d_raster._ENGINE = previous_h3d


def test_a_missing_numba_is_reported_not_raised() -> None:
    """The one cause the launcher used to name is the one that cannot fail."""

    from zlc_plot import _raster_kernels

    previous = _raster_kernels.HAVE_NUMBA
    _raster_kernels.HAVE_NUMBA = False
    try:
        assert "numba is not installed" in _kernel_warm.warm()
    finally:
        _raster_kernels.HAVE_NUMBA = previous
