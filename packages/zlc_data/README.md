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
belongs to `zlc_durable`.

The reader also migrates the exact historical `zlc.figure/v1` envelope into
the current v2 representation. It derives member dtype/shape from the NPY
members and supplies only the two formerly absent Dataset label fields; unknown
legacy fields and every other schema/version remain errors.

Likewise, `save_npz(stream, snapshot)` only encodes a Dataset to caller-owned
writable binary IO. A path consumer publishes it with
`zlc_durable.atomic_write_file(path, lambda stream: save_npz(stream, snapshot))`;
the codec never opens or truncates a destination path itself.
