# The tutorials, in the order they build on each other

Each layer teaches its own surface, and every name on that surface has to
appear in its notebook — a guard in each layer fails otherwise, so a facade
cannot grow a name nobody was shown how to use.

| read | to learn |
|---|---|
| `packages/zlc_data/notebooks/usage.ipynb` | what a dataset is: axes, validity, snapshots, and how one is written down |
| `packages/zlc_runtime/notebooks/usage.ipynb` | generations and publication: who produced this, and when |
| `packages/zlc_plot/notebooks/usage.ipynb` | drawing and fitting, and what a panel kind means |
| `packages/zlc_pulse/notebooks/usage.ipynb` | the sequence model, the compiler, and driving a board (the last cells want real hardware) |
| `packages/zlc_atom/notebooks/usage.ipynb` | the physics layer: nodes, devices, calibration |
| `packages/zlc_ui/notebooks/usage.ipynb` | opening each window and driving it through its handle |
| `packages/zlc_workbench/notebooks/usage.ipynb` | the whole thing end to end: apparatus, pulse, one shot, saved and reopened |

Start with the last one if you want to see the experiment run; start with the
first if you want to understand what the numbers are.
