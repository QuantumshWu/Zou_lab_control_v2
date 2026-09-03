# zlc_data

`zlc_data` owns the immutable scientific data model used by ZLC: typed axes,
Repeat/Point/Cell-data `DomainSpec` mappings, value schema, validity, snapshots,
selections, projection, and strict dataset/figure codecs. All three domains
own logical axes through the same contract. Repeat and Point use axis-major
codes over one flat carrier; Cell-data uses implicit dense dimensions, so an
image never materializes a pixel-sized coordinate plane.

It does not own runtime generations, plotting, Qt, devices, workspace paths,
or experiment policy. The installed product is the repository-root ZLC
distribution; this directory is an internal dependency layer, not a separately
supported product.

Current public code lives under `src/zlc_data/`. The approved cross-layer
contract is defined by the root `ARCHITECTURE_DESIGN.md`; this README describes
only the current product surface.

`zlc_data.figure_archive.write_figure_archive(stream, ...)` is the sole Figure
format encoder. It validates the complete member namespace and metadata before
streaming NPZ content to caller-owned binary IO; durable path publication
belongs to `zlc_durable`.

Figure uses stable `schema="zlc.figure"` with no numeric version. The reader
accepts only the current complete grammar and rejects missing, extra or unknown
fields. Existing workspace files are not converted; there is no alternate alias.
The Figure NPZ is the primary typed artifact. Exact Plot recipe, overlay,
viewport and lineage metadata travel beside its Dataset members; PNG rendering
is a derived preview owned by `zlc_plot`.
Each member remains an ordinary NPZ `.npy` entry. The single Figure writer
Deflates small or materially compressible members and stores large camera-like
noise members when Deflate would save less than 20%; readers need no alternate
format or compatibility path.

Likewise, `save_npz(stream, snapshot)` only encodes a Dataset to caller-owned
writable binary IO. A path consumer publishes it with
`zlc_durable.atomic_write_file(path, lambda stream: save_npz(stream, snapshot))`;
the codec never opens or truncates a destination path itself.
