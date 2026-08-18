# zlc_runtime

`zlc_runtime` owns ZLC node execution, signal publication, causal lineage,
future-publication following, coherent fronts, presentation scheduling, and
selection-derived signals. It is Qt-free and contains no plugin physics.

The retained implementation is the path used by `NodeHost`,
`SignalDataPlane`, and Workbench. The unused exact-reservation, dataset-builder,
monitor-dataset, legacy live-port, preview-port, and RunHandle compatibility
frameworks were removed instead of being preserved as a parallel runtime.

The installed product is the repository-root ZLC distribution; this directory
is an internal dependency layer. Target invariants and current implementation
status are recorded in the root Architecture and Implementation Plan.
