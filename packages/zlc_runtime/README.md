# zlc-runtime

`zlc-runtime` is the Qt-free runtime layer for signal data, exact/follow/latest
consumption, node hosting, and presentation scheduling.  It is a
small contract package: domain-specific device identity and UI integration
remain in their owning packages.

## Charter

These decisions are part of the public contract and are not optional
implementation details:

1. Derived signal families are same-shot automatically.  Every member of a
   `SignalFront` must trace to the same root publication.  If a family is not
   complete, the whole family falls back to the previous complete front.  No
   global shot counter or cross-producer guarantee is invented.
2. Device-return accounting protocols are excluded.  Association request and
   arm/bind/next/finish cursor machinery do not belong in this package.  The
   immutable `EventRef(stream, generation, sequence)` lineage identity does.
3. The runtime is Qt-free.  Wakeups are injected callbacks; Qt adapters belong
   in `zlc_ui`.
4. Notebooks are first-class acceptance fixtures, while the top-level facade
   remains finite and allow-listed. The current package API is recorded in
   [docs/contract.md](docs/contract.md); that contract is not a repository Goal.

## Package map

The implementation is organized around `streams.py` and `dataset.py`, the
lineage-aware `plane.py` signal surface, `host.py` node lifecycles, and `presentation.py` scheduling.  `live_dataset.py`,
`owner_mailbox.py`, `dataset_output.py`, and the small helper modules provide
the ownership seams; `window_runtime.py` contains the dependency-free compute
and export helpers.  Domain packages own hardware identity, device execution,
and UI/plot adapters.

`zlc_runtime.host.NodeHost` is deliberately a lifecycle seam.  Domain-side
descriptors, application contexts, device requirements, and operation binding
stay outside this package. The host chooses a worker when no source signal is
bound and a processor when one source is bound. Source extent then selects the
processor path: finite `FollowTap`, one retained final snapshot, or infinite
latest. Measurement/Task/Processor role does not select a runtime kind.

## Dependency boundary

Runtime code depends on the standard library, NumPy, and `zlc-data`.  It must
not import Qt, plotting, storage, workbench, or neutral-atom domain modules.
`zlc-data` owns the canonical validation functions plus immutable data blocks,
schemas, snapshots, and transform primitives consumed by this package.

## Deliberate exclusions

The runtime does not carry `run.py` or `ports.py`; those modules are the
hardware safety/execution closure and remain domain-owned.  `resources.py`
contains only the generic in-process resource arbiter.  Physical-device
identity and binding-stamp types remain with the device domain.  Device-return
association cursors and the arm/bind/next/finish accounting protocol are also
excluded; a future scan coordinator may count points using the lossless exact
stream contract, but this package does not define a scan engine.

The signal plane likewise does not expose preemption-generation binding APIs or
event-derived generations.  Application code performs explicit stop-then-start
retirement, while `EventRef` remains the lineage identity used for same-shot
derived-family coherence.

The notebook and `examples/demo_signal_flow.py` remain headless and do not
depend on `zlc_plot`. The integrated plot-panel and TaskConsole path lives in
`zlc_workbench`, which composes this runtime with `zlc_plot` without moving plot
or device semantics into this package.

The migration size and comparison to the read-only reference are recorded in
[docs/loc-report-2026-08-03.md](docs/loc-report-2026-08-03.md).
