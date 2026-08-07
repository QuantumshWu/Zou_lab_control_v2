# Semantic edit surface — 2026-08-04

The semantic edit contract is now registry-derived and shared by the direct
session API, `RasterPlotHost`, `NotebookView`, and `Qt5ParameterPanel`.

- `_kinds/*` declares `admits(schema)` and `semantic_fields`; no GUI branch
  owns a plot-kind semantic catalogue.
- `describe_semantics(schema, spec, *, layout=DEFAULTS.layout)` returns the
  current values and admissible `AxisRef`/kind/reduction/facet domains.
- `SemanticField.rebuild` is true for every semantic field. `replace_spec` is
  the only semantic mutation path and retains revalidated display state,
  compatible viewport and fixed size while clearing selectors and fit.
- `FacetGridPlot` uses `facet_rows` plus optional `facet_cols`; `DataView` and
  the geometry resolver preserve row-major two-dimensional shape. Two-axis
  facets expose independent row/column display-unit parameters.
- Qt semantic controls are in a separate Semantics group and emit
  `semanticEdited`; display controls remain on the cheap parameter lane.

The repository currently contains 25,825 non-blank Python lines under
`src/zlc_plot` and 113 passing tests. (The corresponding physical line count,
including blank lines, is 28,627.) The original 143 kind-specific dispatch
lines present at the V6 baseline remain a documented follow-up; the current
source grep reports 154 lines after the facet/semantic additions. The S5
registry proof ensures a newly
declared semantic field is carried mechanically without adding frontend
branches.
