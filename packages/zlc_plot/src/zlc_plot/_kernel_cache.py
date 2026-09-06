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

Installed as a wheel there is no checkout: the module sits in
site-packages, and the folder four levels above it is whatever holds the
interpreter -- a drive root, a user profile -- and belongs to nobody.
There numba's own default applies, a ``__pycache__`` beside the module
with numba's fallback to a per-user cache when that is read-only, and
nothing here is set.
"""

from __future__ import annotations

import os
import pathlib

#: The folder, at the repository root.  No leading dot: this is not a
#: private dotfile, it is a build product of the checkout it sits in, and
#: hiding it only makes it harder to find and clear.
CACHE_DIRECTORY_NAME = "numba_cache"


def _checkout_root() -> pathlib.Path | None:
    """The repository root, when this module lives in a checkout.

    In a checkout this module is packages/zlc_plot/src/zlc_plot/, and the
    root is the folder that holds ``packages``; a module anywhere else --
    site-packages -- is not in a checkout, and ``None`` says so.
    """

    here = pathlib.Path(__file__).resolve()
    root = here.parents[4]
    if here.parent == root / "packages" / "zlc_plot" / "src" / "zlc_plot":
        return root
    return None


def kernel_cache_dir() -> pathlib.Path:
    """The directory compiled kernels cache in, whether or not it exists.

    The checkout's ``numba_cache``, or, installed, numba's own default: the
    ``__pycache__`` beside this module.
    """

    root = _checkout_root()
    if root is None:
        return pathlib.Path(__file__).resolve().parent / "__pycache__"
    return root / CACHE_DIRECTORY_NAME


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
    if _checkout_root() is None:
        # Numba's default already is the directory kernel_cache_dir names,
        # and naming it explicitly would only take away numba's fallback
        # when a package directory is read-only.
        return ""
    path = kernel_cache_dir()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        # A read-only checkout is not a reason to fail to import: numba
        # falls back to compiling every time, which is slow, not wrong.
        return ""
    os.environ["NUMBA_CACHE_DIR"] = str(path)
    return str(path)
