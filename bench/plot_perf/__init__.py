"""Plot performance benches, each run as ``python -m bench.plot_perf.<runner>``.

A render child of such a run is not the product's child unless this package
says so first.  A spawned child re-runs its parent's main module as
``__mp_main__`` before it unpickles its Process -- only a package
``__main__`` is skipped, which is how the product starts -- so every runner's
top-level imports ran in every render child, numpy among them, before
``zlc_plot.render_process`` could bound the child to one BLAS thread: each
child committed the parent's OpenBLAS team, and every child memory, spawn and
warm-pool number a runner reported was a bench child's.  The runner's package
is imported ahead of its module, in the child too, so the checkout's bootstrap
and then the render-child rule go here, before anything numerical.
"""

import zou_lab_control  # noqa: F401  -- the bootstrap, before any zlc import
import zlc_plot.render_process  # noqa: F401  -- a render child's one BLAS thread
