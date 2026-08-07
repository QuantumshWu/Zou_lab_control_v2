# Zou lab control

Neutral-atom experiment control: one repository, one distribution, eight layers.

```
bin\experiment.bat        the apparatus, then the console, then the pulse editor
bin\pulse_editor.bat      the pulse window on its own
```

Run them from your experiment folder — the one holding `pulses\`, `data\` and
`apparatus.json`. They are deliberately not passed a workspace: a
double-clicked launcher starts in `bin\`, which is nobody's experiment.

## The eight layers, and what each is allowed to know

Each lives under `packages/` with its own `src/`, `tests/`, `docs/` and
`notebooks/`. The order is the dependency order — a layer may know the ones
above it and must not know the ones below.

| layer | owns | may not |
|---|---|---|
| `zlc_data` | what a dataset IS: axes, validity, snapshots, manifests | everything else |
| `zlc_durable` | writing a file so a crash cannot half-write it | domain, plots, Qt |
| `zlc_runtime` | generations, publication, board-coherent ticks | what any signal MEANS |
| `zlc_plot` | drawing, fitting, panel layout | scheduling, devices |
| `zlc_ui` | windows and controls | data, plots, domain — **and no Qt escapes it** |
| `zlc_pulse` | the sequence model, the compiler, the wire | anything above |
| `zlc_atom` | the physics: nodes, devices, calibration | how any of it is shown |
| `zlc_workbench` | composition and wiring, and nothing else | inventing domain or UI |

Two rules are mechanically enforced rather than remembered, because both were
broken before they were checked:

* **Every window is one call and one handle.** `zlc_ui` hands out
  `open_pulse_editor()`, `open_task_console()`, `open_figure_viewer()`,
  `open_device_manager()`; each answers with a handle carrying signals and
  `set_*` methods. Nothing outside `zlc_ui` holds a QWidget, and
  `packages/zlc_workbench/tests/test_gui_seam.py` fails if anything tries.
  Whatever the outside can hold, the outside will assemble — and then what is
  on screen has two owners.
* **Every package is reached through its front door.** No layer imports
  another's submodule; if a name is needed, it goes on that layer's facade with
  its contract and its tutorial updated. The count went 45 → 0.

## Running the tests

```
python -m pytest
```

All eight suites, from the top: a change that crosses two layers is checked by
both in one run, which is the point of merging them.

## Where the code lives

This tree is the product. The eight standalone repositories beside it
(`Github/zlc_data`, `Github/zlc_ui`, …) are where it was built and are kept for
their history; they are not a second place to edit. If `import zlc_data`
resolves to one of them, this checkout is not the one running — which
`packages/zlc_data/tests/test_package_guards.py` will say out loud.
