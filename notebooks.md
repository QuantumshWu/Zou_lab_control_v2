# Product notebook

The one supported tutorial is
`packages/zlc_workbench/notebooks/usage.ipynb`. It runs only public installed
product paths: a temporary workspace, virtual apparatus, canonical Camera
Measurement publication, and the normal Plot `NotebookView`.

Release evidence loads that tracked document into memory and executes every
cell from a temporary working directory in a fresh kernel from the built wheel,
with `PYTHONPATH` empty, `QT_QPA_PLATFORM=offscreen`, and `MPLBACKEND=Agg`. The
tracked notebook keeps no outputs or execution counts; the executed in-memory
document is discarded after the evidence checks.

Real-screen inspection and FPGA/camera/SLM hardware acceptance are separate
runbooks and are never implied by this offline notebook lane.
