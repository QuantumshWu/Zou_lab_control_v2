"""Where compiled kernels are cached on disk.  One answer, one place.

Numba writes machine code beside a fingerprint of the source and the
toolchain that produced it.  It lives in the checkout, under a plainly
named ``numba_cache`` folder: it belongs to this checkout's sources, a
second checkout should build its own rather than share one, and a folder
an operator can see is a folder they can delete when they want it rebuilt.

Three places used to spell this path out -- both kernel modules and
``bin/warm_numba_cache.bat`` -- so moving it needed all three edited
together, or the warmer filled a directory nothing read.  Now they ask
here.

``NUMBA_CACHE_DIR`` remains the override.  Set it and nothing here applies
-- which is what a sandbox, a CI runner or a read-only checkout needs.
"""

from __future__ import annotations

import os
import pathlib

#: The folder, at the repository root.  No leading dot: this is not a
#: private dotfile, it is a build product of the checkout it sits in, and
#: hiding it only makes it harder to find and clear.
CACHE_DIRECTORY_NAME = "numba_cache"


def _checkout_root() -> pathlib.Path:
    """The repository root: this module is packages/zlc_plot/src/zlc_plot/."""

    return pathlib.Path(__file__).resolve().parents[4]


def kernel_cache_dir() -> pathlib.Path:
    """The directory compiled kernels cache in, whether or not it exists."""

    return _checkout_root() / CACHE_DIRECTORY_NAME


def install() -> str:
    """Point numba at that directory unless the caller already chose one.

    Called for its side effect at the top of every module that defines
    kernels, BEFORE numba is imported -- numba reads the variable when its
    dispatcher is built, so a later assignment is simply ignored and the
    cache silently lands wherever the default put it.
    """

    chosen = os.environ.get("NUMBA_CACHE_DIR")
    if chosen:
        return chosen
    path = kernel_cache_dir()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        # A read-only checkout is not a reason to fail to import: numba
        # falls back to compiling every time, which is slow, not wrong.
        return ""
    os.environ["NUMBA_CACHE_DIR"] = str(path)
    return str(path)
