# zlc_data

`zlc_data` owns the immutable scientific data model used by ZLC: typed axes,
point metadata, value schema, validity, snapshots, selections, projection, and
strict dataset/figure codecs.

It does not own runtime generations, plotting, Qt, devices, workspace paths,
or experiment policy. The installed product is the repository-root ZLC
distribution; this directory is an internal dependency layer, not a separately
supported product.

Current public code lives under `src/zlc_data/`. The approved cross-layer
contract is defined by the root `ARCHITECTURE_DESIGN.md`; historical package
contracts and tutorial-shaped API lists were removed rather than kept as a
second authority.

`zlc_data.figure_archive.write_figure_archive(stream, ...)` is the sole Figure
format encoder. It validates the complete member namespace and metadata before
streaming NPZ content to caller-owned binary IO; durable path publication
belongs to the composition layer.
