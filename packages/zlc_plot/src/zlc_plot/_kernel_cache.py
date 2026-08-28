"""Where compiled kernels are cached on disk.  One answer, one place.

Numba writes machine code beside a fingerprint of the source and the
toolchain that produced it.  Two facts decide where that belongs, and
neither of them is "next to the source":

  * It is per-MACHINE.  The bytes are compiled for this CPU, this numba and
    this LLVM; carrying them anywhere else is at best waste and at worst a
    stale hit.  So it goes in the user's LOCAL (non-roaming) cache area, the
    place every platform reserves for exactly this.

  * A checkout is often inside a synced folder -- this one lives under
    Dropbox -- and a directory of machine code that is rewritten on every
    compile is then uploaded continuously, forever, for nothing.  That is
    what a cache in the working tree costs, and it is why the tree is the
    wrong home even though the folder was already git-ignored.

Numba keys each entry by the source file's own path, so several checkouts
or git worktrees on one machine share this directory without colliding:
each simply gets its own entries.

``NUMBA_CACHE_DIR`` remains the override.  Set it and nothing here applies
-- which is what a sandbox, a CI runner or a read-only home needs.
"""

from __future__ import annotations

import os
import pathlib
import tempfile

#: The folder name under the platform's cache root.  The product's name, so
#: an operator finding it knows what wrote it.
CACHE_NAMESPACE = "zou_lab_control"


def _user_cache_root() -> pathlib.Path:
    """The platform's per-user, non-roaming cache directory."""

    if os.name == "nt":
        # LOCALAPPDATA, not APPDATA: the roaming one would carry machine
        # code between machines on a domain profile.
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP")
        if not base:
            profile = os.environ.get("USERPROFILE")
            base = (
                str(pathlib.Path(profile) / "AppData" / "Local")
                if profile
                else tempfile.gettempdir()
            )
        return pathlib.Path(base)
    base = os.environ.get("XDG_CACHE_HOME")
    if base:
        return pathlib.Path(base)
    home = os.environ.get("HOME")
    return pathlib.Path(home) / ".cache" if home else pathlib.Path(
        tempfile.gettempdir()
    )


def kernel_cache_dir() -> pathlib.Path:
    """The directory compiled kernels cache in, whether or not it exists."""

    return _user_cache_root() / CACHE_NAMESPACE / "numba"


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
        # An unwritable cache root is not a reason to fail to import: numba
        # falls back to compiling every time, which is slow, not wrong.
        return ""
    os.environ["NUMBA_CACHE_DIR"] = str(path)
    return str(path)
