# The tutorials, in the order they build on each other

These notebooks are product tutorials, not mirrors of every exported symbol.
Only APIs that belong to a real tutorial flow should appear; an unused export
is deleted rather than kept alive by documentation coverage.

| read | to learn |
|---|---|
| `packages/zlc_plot/notebooks/usage.ipynb` | drawing and fitting, and what a panel kind means |
| `packages/zlc_pulse/notebooks/usage.ipynb` | the sequence model, the compiler, and driving a board (the last cells want real hardware) |
| `packages/zlc_atom/notebooks/usage.ipynb` | the physics layer: nodes, devices, calibration |
| `packages/zlc_ui/notebooks/usage.ipynb` | opening each window and driving it through its handle |
| `packages/zlc_workbench/notebooks/usage.ipynb` | the whole thing end to end: apparatus, pulse, one shot, saved and reopened |

Start with the Workbench notebook for the product flow. The lower-level Plot,
Pulse, Atom, and UI notebooks remain focused technical tutorials.
