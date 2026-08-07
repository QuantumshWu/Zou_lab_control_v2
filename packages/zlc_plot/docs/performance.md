# Plot performance

## Source-size audit (2026-08-03)

The current `src/zlc_plot` tree contains 25,825 non-blank Python lines (28,627
physical lines including blanks).  This
is above the aspirational 15k target because the package still carries four
independent responsibilities that are part of its public contract: (1) the
typed `(R, P, *data_dim)` projection and unit-aware DataView, (2) the complete
Matplotlib artist renderer and fixed-preset/DPR layout, (3) the Qt5 raster host
and controls, and (4) the notebook DOM adapter plus bounded live/fit transport.
The fit engine and regular-image solver are shared by every plot kind, while
the `_kinds` registry owns only the closed semantic dispatch; deleting those
modules would reintroduce duplicated kind branches rather than reduce the
system.

The retained size is therefore deliberate and itemized here instead of being
hidden by generated code or compatibility shims.  The next safe reduction
boundary is a responsibility-level extraction into separately installable
packages; collapsing those public boundaries inside this package would make
the GUI/notebook parity and the single raster authority less explicit.

## Scope

`examples/camera_live_profile.py` is a headless internal instrumentation utility,
not an application integration example. It subclasses protected session/host
hooks only to time stage boundaries; the workload itself uses the same public
composition as a GUI camera view:

```text
immutable zlc_data.OwnedSnapshot
    -> LivePlotController (capacity one, latest only)
    -> PlotSession ImagePlot
    -> RasterPlotHost
    -> immutable promoted RasterFront
```

The primary camera contract is `(R=1, P=1, camera_y, camera_x)`.  Both spatial
axes are declared dense `data_dim` axes and the `ImagePlot` refers to them with
`AxisRef.data(...)`.  Every public live camera frame keeps the fixed
`(1, 1, *frame)` geometry, while acquisition history remains private.  The
capacity-one ingress replaces only a pending revision, so acquisition can keep
publishing without building a presentation queue.

The profiler can run a flattened `P=H*W` plus `GridTopology` comparison, but
that is not the recommended camera representation:

```powershell
python examples/camera_live_profile.py --resolution 1024 --geometry data-dim
python examples/camera_live_profile.py --resolution 1024 --geometry point-topology
```

The producer freezes a new immutable `uint16` snapshot for every revision.  It
publishes faster than the presentation cadence, waits for the final revision,
checks the initial and final rasters for painted data, and then closes both
workers.  Timings cover these boundaries:

- `projection_prepare`: build the `DataView` and `ImageData` on the preparation
  worker; no fit is active in this profile.
- `artist_render_commit`: accept the prepared revision, update Matplotlib
  artists, and draw/blit the surface.
- `rgba_handoff`: obtain the already-painted RGBA canvas view.
- `front_capture_and_pack`: RGBA handoff plus owned `bytes`, axes/interaction
  metadata, and construction of the immutable `RasterFront`.
- `front_promotion_and_callbacks`: atomic front replacement and observers.
- `publish_to_promoted_front`: producer ingress to the callback observing that
  exact revision.  This includes cadence waiting and any older active frame.

One-time schema, Figure, font and initial-front construction is excluded from
the steady-state stage distributions.  RSS is whole-process peak RSS, so it
includes NumPy/Matplotlib allocator retention and is not an object-level memory
attribution.

## Ordinary interactive workload

The following measurements use the `2x2` preset at DPR 1. Curve, Image,
Histogram, Rolling and FacetGrid share a `(64, 93, 3)` floating-point snapshot;
PulseTimeline contains digital, analog, scan-region and DAC-segment artists.
Construction, representative hot edits and PNG export are the preceding
baseline measurements; the label column was rerun after the transaction
refactor with 30 alternating title/x/y/value edits. Times are milliseconds.

| Kind | Representative hot edit | Session construction median | Hot edit p50 / p95 | Title + x/y/value labels p50 / p95 | PNG export median |
|---|---|---:|---:|---:|---:|
| Curve | grid visibility | 29.05 | 7.54 / 8.81 | 1.20 / 1.81 | 39.21 |
| Image | colormap | 54.08 | 27.04 / 31.74 | 1.60 / 2.26 | 77.66 |
| Histogram | bin count | 21.11 | 12.62 / 14.17 | 1.11 / 1.63 | 36.24 |
| Rolling | visible window | 36.41 | 12.63 / 15.42 | 1.50 / 2.21 | 45.50 |
| FacetGrid | grid visibility | 85.48 | 15.38 / 16.90 | 2.02 / 2.52 | 54.66 |
| PulseTimeline | scan-region visibility | 22.75 | 5.20 / 6.09 | 1.09 / 1.35 | 39.61 |

Hot edits retain the Figure, layout and existing artists. Selector geometry,
fit overlays and text all finish through the same complete-frame draw, so a new
fit can never be painted over a selector background from an older revision.
Histogram bins and rolling windows update existing artist data. Grid visibility
is intentionally a chrome update, while colormap changes mutate the existing
image mappable and colorbar. Image payload updates may use the renderer's
per-axis image path when no layout change is involved; it still publishes one
complete RGBA front. Pure text transactions mutate only title, axis-label and
colorbar-label objects before that final draw. Grid and colorbar visibility are
separate chrome effects because they change axes presentation. Image colormap
changes update the existing mappable and colorbar, which explains their higher
cost.
The first plot in a fresh Python process can additionally pay font discovery
and Matplotlib initialization; that one-time cost is excluded from the warm
medians.

## Dense-image data and display paths

Dense image data and display pixels have different responsibilities.  The
immutable `zlc_data.OwnedSnapshot` remains authoritative for selector queries and
fit input.  The Matplotlib `AxesImage` only needs enough scalar samples for the
physical pixels in its current axes box.

`DataView.image` therefore keeps two exact paths:

- regular `AxisRef.data` images reduce `R`, `P`, and other data dimensions in
  vectorized NumPy operations;
- the common camera geometry `(R=1, P=1, y, x)` retains a read-only view of the
  producer's native image dtype and the snapshot validity plane.  It does not
  create full-image `float64` values or an unused `int64` count plane.

The renderer then prepares a display-only raster for each accepted frame.  It
crops the source to the final viewport, resolves equal-aspect geometry from the
canonical x/y physical-unit scale, reads
the actual physical `axes.bbox` after DPR scaling, and block-averages only when
the visible source has more samples than that box can display.  The cache key
contains the source revision, final viewport and physical pixel shape, so
colormap and color-limit edits reuse the same scalar raster.  Resetting the
viewport restores the full source extent; selectors and fits never read the
decimated raster.

The general PointTable/GridTopology projection remains authoritative for
irregular coordinates, masks and facet subsets.  No topology is inferred from
repeated PointTable values.

## Measured results

The recorded reference measurements were taken on Windows 11, Python 3.13.12, NumPy 2.4.2 and
Matplotlib 3.10.8 on a 16-logical-CPU machine.  Each row is a fresh process
running `examples/camera_live_profile.py` with a `2048 x 2048 uint16` camera.
Values are milliseconds.

| Preset / cadence | DPR / front raster | Prepared image | Promoted / published | Projection p50 / p95 | Artist render p50 / p95 | Complete pipeline p50 / p95 | Peak RSS delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| `2x2`, 100 ms | 1 / 480x357 | 256x256 | 11 / 30 | 0.446 / 0.729 | 22.163 / 23.023 | 23.047 / 24.046 | 98.17 MiB |
| `8x8`, 400 ms | 1 / 1488x1113 | 1024x1024 | 4 / 30 | 0.681 / 0.750 | 117.746 / 121.831 | 122.980 / 126.850 | 225.52 MiB |

Both runs reached the latest revision, painted the first and final fronts,
reported no update failure and closed without a timeout.  The `2x2` run
sustained 9.91 Hz at the 100 ms render budget.  The `8x8` run remained below
its 400 ms budget while rendering roughly sixteen times as many prepared scalar
samples.  The producer ran at 30 Hz; capacity-one ingress intentionally
coalesced 19 and 26 intermediate revisions respectively instead of building a
queue.

The final presentation-transaction pass was also rerun at DPR 2 with a `2x2`
surface. A 1024² source at 100 ms promoted 20 of 24 published revisions; its
projection, artist render and complete pipeline measured 0.474 / 0.718,
22.232 / 24.141 and 25.389 / 27.255 ms p50/p95. A 2048² source at 400 ms
promoted 4 of 12 revisions and measured 0.583 / 0.880, 43.609 / 46.421 and
47.078 / 49.312 ms respectively. Both cases painted their initial front,
reached the final revision, reported no failure or timeout and closed both
workers. Their peak RSS deltas were 51.9 MiB and 112.5 MiB. Intermediate
revisions were coalesced by the capacity-one mailbox; no stale frame was later
replayed.

A separate `2x2` color-limit pointer profile retained the 256x256 raster.
Selector geometry and its numeric labels continue to follow pointer cadence,
while raster color remapping and front promotion are admitted at most once per
100 ms; pointer release always commits the exact final limits. The raster
frontend excludes its externally composited selector scene before the complete
front is published; it never exposes an intermediate Matplotlib buffer.
Before display-sized raster preparation, the same 2048² path spent about
206 ms per preview and about 269 ms per live commit because Matplotlib
normalized, color-mapped and resampled the entire source image on each blit.

DPR changes the preparation budget rather than stretching a low-resolution
front.  For the same `2x2` source, DPR 1.0, 1.5 and 2.0 produced 480x357,
720x536 and 960x714 fronts; the corresponding image axes were approximately
252, 378 and 504 physical pixels wide, with 256, 409 and 512 sample prepared
rasters. The recorded DPR 2 run promoted the final revision without error at
10.06 Hz observed cadence; artist render was 48.878 / 51.402 ms p50/p95.

## Selector and fit paths

The backend-neutral `SelectorScene` builds the same immutable geometry, live
numeric labels and z-order for Matplotlib and the Qt candidate painter. A scene
build measured about 5.08 µs; painting a candidate into an off-screen Qt image
measured about 120 µs. The expensive part of a color-limit gesture is therefore
the image presentation above, not selector geometry or text generation.

Every plot kind with a numerical fit meaning shares one request and
presentation lifecycle: `FitSelection -> _solve_fit_selection -> FitEngine ->
FitOverlay`. Curve and Rolling select their painted series, Histogram selects
its painted bin centers/counts, Image selects its painted scalar field, and a
FacetGrid delegates only that projection step to its focused Curve, Histogram
or Image cell. PulseTimeline deliberately advertises no fit models because it
has no single numerical dependent variable. There is no raw-snapshot or
frontend-specific fit authority beside this pipeline.

Curve, Rolling and one-dimensional Facet fits consume the first immutable
series already produced by the plot's `DataView`; selector and viewport masks
are applied to that same series. A direct warm run on the documented
`(64, 93, 3)` example measured 0.022 ms to form an 18-sample Curve Area fit,
0.023 ms for a 16-sample Facet Area fit and 0.077 ms for a 15-sample Rolling
Area fit. Complete synchronous fit transactions, including the accepted
overlay presentation, measured 7.65, 6.39 and 9.11 ms respectively. Built-in
SciPy solver modules are loaded with the package rather than on the first fit
gesture, so the first click does not pay a several-hundred-millisecond lazy
import inside the fit transaction.

A separate 20,000-point complete-session run measured a 72.0 ms cold and
19.4 ms warm Curve Gaussian transaction, including the accepted overlay draw.
The same source measured 6.36 ms warm for Rolling's visible 100 samples and
12.29 ms warm for a 60-bin Histogram Gaussian. These plot kinds all enter the
same fit engine with the one projected selection supplied by `FitProjection`;
there is no plot-kind-specific second reduction or raw-data fit path.

The built-in radial Gaussian path for a regular dense Image never reads the
display-decimated raster. Its selection retains the native 2-D image and
represents an all-image selection by authority plus `sample_count`, without
allocating a full index vector first. A 2048² all-image selection reported 4,194,304 samples with
`selected_indices=None` and no measurable steady RSS increase.

The radial Gaussian solver uses a bounded regular grid of at most 257×257 for
its seed and primary parameter search, then checks the full-resolution
objective and performs a bounded full-resolution convergence refinement when
needed. The final fitted values, residuals, RSS and selected indices are always
materialised from every selected source pixel. For an all-valid linear-loss
image, the final full-resolution covariance uses the exact separable Gaussian
information matrix; masked and robust-loss fits retain the bounded-row exact
information pass. Thus optimization avoids repeatedly sweeping millions of
pixels without turning a display-decimated raster into fit authority.

The `soft_l1` objective uses the stable equivalent
`2*squared/(sqrt(1+squared)+1)`, avoiding cancellation close to an exact fit.
Isolated warm solver medians were 53.2 ms at 1024² and 211.8 ms at 2048² for an
exact Gaussian, versus 274 ms and 1.901 s before the bounded-search change;
with 0.02 standard-deviation noise they were 84.2 ms and 339.4 ms. The largest
old/new parameter difference across noisy linear, masked linear and `soft_l1`
comparisons was 1.2e-7; the separable and striped full-resolution information
matrices differed by 1.1e-15 relatively. Complete `PlotSession` transactions,
including overlay presentation, measured 204 / 151 ms cold/warm at 1024² and
332 / 325 ms at 2048² on the reference machine. Both returned the exact
all-pixel sample count, full fitted/residual/index arrays, finite parameter
uncertainties and a current overlay. If a live fit exceeds the display cadence,
the current complete data+fit front remains visible and incoming revisions
continue to coalesce; there is no half-frame or catch-up burst. Custom Image
models remain on the general coordinate-expansion solver path unless they
provide a specialization.

## GridTopology comparison

The 1024² flattened comparison used `(R=1, P=1,048,576)` with explicit
`GridTopology`.  It still took 2.606 / 2.610 s p50/p95 to project because that
representation correctly remains on the general grouping path.  Reusing the
same immutable schema object makes ingress validation constant-time:
`publish()` measured 0.169 / 0.194 ms p50/p95.  Snapshot freezing still took
46.52 / 66.43 ms and projection took 2.657 / 2.665 s.  The run remained bounded
(16 published, two promoted, 14 coalesced) and reached revision 16, but this is
further evidence that camera pixels belong in dense data axes rather than
flattened point rows.

## Reproduction

Install the optional process-metrics dependency with
`python -m pip install -e ".[profile]"` before collecting RSS/CPU results.
Run each case in a fresh process so its RSS baseline is independent:

```powershell
python examples/camera_live_profile.py --resolution 2048 --frames 30 --producer-hz 30 --refresh-ms 100 --size 2x2 --dpr 1 --timeout 20
python examples/camera_live_profile.py --resolution 2048 --frames 30 --producer-hz 30 --refresh-ms 400 --size 8x8 --dpr 1 --timeout 20
python examples/camera_live_profile.py --resolution 1024 --geometry point-topology --frames 16 --producer-hz 20 --refresh-ms 100 --size 8x8 --dpr 1
```

The command prints machine-readable JSON.  `--output report.json` writes the
same report for later comparisons.
