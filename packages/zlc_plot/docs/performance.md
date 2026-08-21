# Plot performance

## Current M4 performance closure (2026-08-20)

- One `PlotSession` owns one serial analysis executor. Frame preparation,
  manual fit and live fit submit to that same lane; there is no package-global
  fit/stripe pool and no second prepare executor.
- A complete PanelState target is one idempotent transaction. A no-op performs
  zero solve, zero render and zero front promotion; a changed target merges its
  effects into one final render. Startup always promotes the initial front
  first, and a startup no-op reuses it instead of creating a ghost front.
- Indexed-derived data preserves every Measurement primary index as value or
  invalid. A busy same-shot Surface group never queues another full frame; it
  keeps Plane latest and stages it on completion. `RasterPlotHost` likewise
  owns only one active pair and one latest input. Its existing worker Condition
  enforces the 1 s active deadline without waiting for another frame: timeout
  publishes one loud invalid result, cancels that solve, then releases latest.
  Cadence/backpressure skips remain ordinary invalid Dataset cells.
- Selector Off means the plot consumes no pointer gesture. Its wheel belongs to
  the outer page. Selector On permits only double-click focus in a FacetGrid
  overview; area/pan/zoom begin only on a focused cell or non-grid surface.
- A low-disturbance formal 96x128 Camera -> 26x26 Area fit + parallel ROI
  image -> Rolling harness uses the real 100 ms TaskConsole timer. Across
  three fresh 60-revision runs, clean HEAD -> candidate same-shot joint
  P50/P95 was 76.46/99.95 -> 74.85/93.10 ms; pooled all-three Rolling was
  93.11/113.03 -> 87.83/96.15 ms. Of 180 source indices, 174 produced valid
  fit values and six solver-invalid cells; there were zero busy misses, FIFO
  entries or panel errors. The 100 ms value is a profiling warning, not a hard
  acceptance deadline.
- Timeline accounting observed all 300/300 expected occurrences of each
  stage exactly once. Admission P50 was about 3.3 ms (sampled P95 about 54 ms),
  FitEvent -> Rolling was 0.52/about 38.7 ms, and last promote -> owner accept
  was 0.34/1.33 ms. One generation-2 GC per run took 78--81 ms and affected
  only maxima, not the reported percentiles. A live `FitEvent` is published
  after exact solve and before the slower owner raster; the main Panel still
  presents only an atomic data/fit pair. Manual fit retains accepted-overlay-
  then-event ordering.
- Direct-host resource gates separated contention from TaskConsole latency.
  One Image+fit host completed at 36.90/40.03 ms P50/P95 and four mixed hosts
  at 71.02/78.47 ms, both with 90/90 complete revisions. Eight mixed hosts
  saturated one Python core at 145.18/192.51 ms and completed 88/90 main-fit
  revisions; every other host reached 90/90, FIFO stayed zero, and final latest
  healed without an error. Supporting eight 10 Hz Matplotlib surfaces would
  require a materially different renderer/process architecture and is not a
  hidden same-shot scheduling defect.
- Isolated renderer P50 A/B in milliseconds was: Image+fit 34.37 -> 32.49,
  Curve 2.74 -> 2.40, Histogram 3.18 -> 2.82, Rolling 16.55 -> 15.18 and a
  four-cell FacetGrid 11.84 -> 9.59. DPR1/2 parity across Image, Curve,
  Histogram, Rolling and FacetGrid remained pixel-exact, including Area/fit
  and tight colorbar updates. Series/Histogram unbeatable warm seeds reduced
  warm P50 by about 62--91%; cold starts and Image fits did not regress.
- Three larger alternatives were measured and rejected. A third foreground
  layer offered only about 1.5--4 ms theoretically, increased pixel-order risk,
  and its prototype left zero useful residual. A separate wall-deadline
  admission scheduler adds state despite no admission miss or owner bottleneck. Full backend ingress
  would add roughly 300--500 lines despite zero observed busy misses. None has
  enough benefit to justify its complexity.

The older tables below remain dated numeric references for projection and
render costs. Their former queue/lifecycle policies are not product contracts;
the closure above is authoritative.

## Overhaul reference measurements (2026-08-10)

Everything below this section predates the 2026-08-10 performance overhaul
and is retained only as a dated numeric record, not as lifecycle or queue
policy. Measured on the reference machine (Windows 11, Python 3.13.12,
numpy 2.4.2, matplotlib 3.10.8), 2048^2 uint16 camera contract, warm p50,
via direct session/host probes:

| Case | Before | After |
|---|---:|---:|
| live 2x2: projection / render / pipeline | 154 / 47 / 202 ms | 0.8 / 33 / 34 ms |
| live 2x2: observed cadence | 3.3 Hz | 9.9 Hz (meets the 100 ms budget) |
| live 8x8: pipeline @100 ms cadence | 297 ms, 2.5 Hz | 93 ms, 5.9 Hz |
| Qt widget update -> paint | 207 (2x2) / 300 (8x8) ms | 29 / 91 ms |
| wheel step | 21 (2x2) / 78 (8x8) ms | 18 / 50 ms |
| 10-notch wheel flick | one zoom step survived | all ten compound, one render |
| radial image fit transaction | 625 ms | 58 ms (31 ms warm-started) |
| radial fit with Area selector | 3375 ms | 43 ms |
| anisotropic image fit, 1024^2 | ~5.5 s (general path) | 24 ms (separable path) |
| FacetGrid 35x300 update, classifier on | 226 ms | 137 ms (solve is same-frame) |
| virtual MOT frame synthesis | 166 / 301 ms | 18 / 33 ms, uint8 |

The structural changes behind these numbers: lazy flat-plane projection, the
chrome background cache with bit-identical composition, the prepared-front
LRU plus per-revision mip pyramid, reshape-mean decimation, coalesced
compounding wheel ticks, BLAS-backed separable fit objectives with
Gauss-Newton/multigrid refinement and lazy result arrays, the same-frame
exact paired live fit, and the separable windowed MOT synthesis.

A same-day follow-up pass tightened the remaining hot paths (same machine,
same harnesses):

| Case | Overhaul | Follow-up |
|---|---:|---:|
| live 2x2: commit / pipeline | 33 / 34 ms | 14 / 15 ms (10 Hz held) |
| live 8x8: commit / pipeline @100 ms | 93 / 96 ms, 5.9 Hz | 54 / 57 ms, 10 Hz |
| Qt update -> paint | 29 / 91 ms | 16 / 58 ms |
| FacetGrid 35x300 update, classifier off | ~100 ms | ~36 ms |
| wheel step 8x8 / pan 8x8 | 50 / 56 ms | 41 / 51 ms |

The follow-up levers: pyramid levels decimate straight to the deepest evenly
divisible power of two with at least one source sample per display pixel
(the residual oversample goes to Matplotlib's resample stage, as fractional
DPR always has); nearest-compatible image fronts precompose uint8 RGBA
through a cached colormap table (unsigned raw fronts through one direct
domain table) while the artist's cmap/clim remain the authority selector
geometry reads; a 1:1 integer-aligned RGBA front blits straight into the
Agg buffer during composition; and pooled facet limits hold steady under
shot-to-shot jitter, expanding with margin only when data actually breaches
the held bound.  The same pass fixed the live warm-start regression (stale
seeds now compete against the cold moment search instead of short-circuiting
it), made the rolling window a pure display selector over retained history,
routed the live-fit budget through the controller's actual cadence, and
stopped reporting superseded render futures as panel errors.

A later pass fixed the color-limit drag: the preview no longer rewrites the
colorbar's endpoint labels (chrome whose edit forced a full background
recapture on every drag step) nor pokes the proxy mappable (whose `set_clim`
triggered a complete `colorbar._draw_all`).  The committed path reapplies
colorbar state once on release through its `colorbar_state` comparison.
One drag step (preview + overlay render + capture) went from 17.7 ms to
6.8 ms median at bench size, and the per-step cost no longer scales with a
full-figure Agg redraw at larger panels.  With the step that cheap, the
color preview's own 100 ms throttle lane became the drag lag it once
guarded against — the recolor now rides the same 30 ms pointer cadence as
pan (`raster_preview_interval_ms` is gone), with same-key pointer-motion
coalescing as flow control.  Measured through the live pointer path under
concurrent 10 Hz 2048² updates: 7–8 ms per recolor for either handle.

The projection layer's dense fast paths were completed in the same pass.
`curve()` over a declared dense data axis (the console's 1D-vector panel on
a camera frame) reduced through the generic sample-materializing
aggregator: flatten 4.2 M (position, value) pairs, sort-unique the
positions, then a Python loop of per-bucket reductions — 290 ms per
revision at 2048², which saturated the render worker and stuttered every
panel scheduled beside it.  `_dense_data_curve` — the narrow twin of the
image projection's `_dense_data_image` — is a straight tensor reduction,
bit-identical across all six reductions: 5.3 ms at 2048², ~25 ms for the
whole 1D panel update including its in-commit Gaussian fit.  The same
audit gave `_aggregate_by_codes` a sortless single-bucket branch (a rolling
trace pointed at the frame itself was paying a stable argsort of millions
of identical codes); histograms were already one vectorized
`np.histogram`, and facet cells and scan-point curves are small by
construction.

## Scope

The measured workload used the same direct composition as a GUI camera view:

```text
immutable zlc_data.OwnedSnapshot
    -> PlotSession ImagePlot
    -> RasterPlotHost
    -> immutable promoted RasterFront
```

The primary camera contract is `(R=1, P=1, camera_y, camera_x)`.  Both spatial
axes are declared dense `data_dim` axes and the `ImagePlot` refers to them with
`AxisRef.data(...)`. Every public live camera frame keeps the fixed
`(1, 1, *frame)` event geometry. Generic indexed-derived signals separately
materialize a bounded ordinary Dataset over their source primary index; the
Surface host never retains a FIFO of full camera frames.

The comparison also covered flattened `P=H*W` plus `GridTopology`, but that is
not the recommended camera representation.

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

## Dated virtual-camera and classifier measurements (2026-08-09)

This pass used the current checkout on Windows 11 / Python 3.13.12 with the
public `PlotSession` and `RasterPlotHost` APIs. Ten fresh
processes constructed one `96 x 128` Image session at `2x2`; the cold session
median/p95 was 190.83 / 194.77 ms. A 30-frame, 30 Hz `128 x 128` camera run at
the 100 ms display cadence measured projection 0.77 / 1.06 ms, artist render
15.21 / 17.24 ms, complete render pipeline 16.35 / 18.42 ms, and observed
front interval 101.11 / 111.26 ms median/p95. It observed 11 fronts,
reached revision 30, and reported no error or timeout. These render-stage
numbers predate current active+latest admission and are not policy acceptance.

The fit/classifier cases used ten warm complete session transactions. The Image
case was an anisotropic Gaussian over a regular `96 x 128` field. Distribution
used 300 bimodal samples; the grid used the same 300 samples for each of 35
sites. Classifier measurements time only Off -> On (the untimed reset prepares
the next sample).

| Operation | Median / p95 ms |
|---|---:|
| 2-D anisotropic Image fit + overlay front | 43.47 / 45.74 |
| one Distribution generic bimodal fit | 10.55 / 10.88 |
| one Distribution threshold classifier enable | 14.03 / 15.43 |
| 35-cell FacetGrid[Histogram] threshold classifier enable | 217.05 / 220.15 |
| 35-cell FacetGrid[Histogram] generic bimodal fit | 177.19 / 183.14 |

Histogram fitting consumed the painted histogram centers/counts through the
existing `FitSelection`; it did not submit the 300 raw samples to a second fit
path. Calibration uses only the independent threshold classifier, not the
generic 35-cell fit row.

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
using a direct `2048 x 2048 uint16` session/host probe.
Values are milliseconds.

| Preset / cadence | DPR / front raster | Prepared image | Projection p50 / p95 | Artist render p50 / p95 | Complete pipeline p50 / p95 | Peak RSS delta |
|---|---:|---:|---:|---:|---:|---:|
| `2x2`, 100 ms | 1 / 480x357 | 256x256 | 0.446 / 0.729 | 22.163 / 23.023 | 23.047 / 24.046 | 98.17 MiB |
| `8x8`, 400 ms | 1 / 1488x1113 | 1024x1024 | 0.681 / 0.750 | 117.746 / 121.831 | 122.980 / 126.850 | 225.52 MiB |

Both runs painted the first and final observed fronts, reported no update
failure and closed without a timeout. The `2x2` render stages sustained the
100 ms budget; the `8x8` stages remained below 400 ms while rendering roughly
sixteen times as many prepared scalar samples. Promotion counts from that old
cadence harness are intentionally omitted because they do not describe the
current active+latest contract.

The final presentation-transaction pass was also rerun at DPR 2 with a `2x2`
surface. A 1024² source measured projection, artist render and complete
pipeline at 0.474 / 0.718, 22.232 / 24.141 and 25.389 / 27.255 ms p50/p95. A
2048² source measured 0.583 / 0.880, 43.609 / 46.421 and 47.078 / 49.312 ms.
Both cases painted an initial and final observed front, reported no render
failure and closed cleanly. Their peak RSS deltas were 51.9 MiB and 112.5 MiB.

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
uncertainties and a current overlay. An armed live fit makes every data
frame a pair. One Session-owned serial analysis executor performs prepare and
solve in order while the render worker remains free; commit then accepts the
overlay into the same front as its data. The host retains one active pair and
one latest complete input, never a full-frame FIFO. Its worker wakes on the
1 second active deadline even when no successor arrives, publishes an invalid
fit for that source index, cancels the stale analysis generation and continues
from latest. The regular-image objective evaluates bounded
stripes serially inside that analysis task; there is no global stripe pool to
oversubscribe 4/8-panel runs or outlive sessions. Reference 2048² stage costs
remain paint 15–25 ms, projection 10–25 ms and warm radial solve about 33 ms.
Custom Image models stay on the general coordinate-expansion solver path unless
they provide a specialization.

## GridTopology comparison

The 1024² flattened comparison used `(R=1, P=1,048,576)` with explicit
`GridTopology`.  It still took 2.606 / 2.610 s p50/p95 to project because that
representation correctly remains on the general grouping path.  Reusing the
same immutable schema object makes ingress validation constant-time:
`publish()` measured 0.169 / 0.194 ms p50/p95.  Snapshot freezing still took
46.52 / 66.43 ms and projection took 2.657 / 2.665 s. This remains evidence
that camera pixels belong in dense data axes rather than flattened point rows;
the old cadence harness's promotion counts are not a current queue contract.

## Grouped reduction is vectorised (2026-08-11)

A FacetGrid over a scan of camera frames reduces one group PER PIXEL: a
`(1, 9, 1200, 1920)` scan faceted on one axis with image cells reduces
2.3 million single-pixel groups per cell.  `_aggregate_by_codes` looped
those groups through Python, one `np.mean` call each -- 6.9 million calls
and ~30 s of a 39.7 s session build; the whole facet-image path measured
22.5 s through the console's raster host.

Three fixes, all in `data_view.py` and all measured on that same scan:

* `_reduce_segments` reduces every code-sorted segment in one ufunc pass
  (`add/minimum/maximum.reduceat`, segment-start gather for FIRST, one
  in-segment lexsort for MEDIAN).  Sums accumulate in the output dtype
  because camera bytes wrap at 256 under their own arithmetic.
* Groups with at most one member (the dense image case) skip the sort
  entirely: `bincount` proves the multiplicity and the reduction is
  identity for every `Reduction`.
* `_domain` on a declared, all-finite domain takes codes straight off the
  index plane (bincount + remap) instead of `np.unique` over one value per
  element, and skips the per-element canonical gather + isfinite pass.

Session construct for the facet-image case: 39.7 s -> 3.3 s in-process;
the console acceptance probe's first rendered front: 22.5 s -> 3.0 s.
Curve and histogram cells on the same dataset build in 1.7 s / 1.4 s.
`tests/test_aggregate_by_codes.py` pins every vectorised branch against
the plain per-group loop, including the uint8 non-wrapping sums.

Still open: the remaining ~3 s is the position-projection machinery
itself (per-cell domain derivation and several full-size passes over the
20.7 M-element expansion); a dense fast lane that recognises "x/y are the
data axes, reduce the rest" as a reshape+mean would take it to tens of
milliseconds, at the cost of a second projection path that must prove
equivalence.

### Superseded: the facet now reuses the dense projections (2026-08-11, later)

The "still open" paragraph above is resolved the right way round: instead
of a faster generic aggregator, `facet()` now routes through the SAME dense
mechanism the single kinds already had.  A facet over the repeat axis or a
point-domain axis slices whole rows, which preserves the regularity
`_dense_data_image`/`_dense_data_curve` rely on -- so each cell reduces
through the one shared kernel (`_masked_leading_reduce`, now also the single
implementation behind both single-kind dense paths, which had duplicated
the reduction chain).  Facets over DATA axes and grouped curve cells keep
the generic algorithm.

Measured on the same `(1, 9, 1200, 1920)` scan: session construct
3.3 s -> 0.87 s (projection itself 0.39 s; the rest is Matplotlib
composition).  Image cells 0.72 s, histogram cells 0.77 s; curve cells over
a scan dimension stay generic (1.66 s) because their x is not a data axis.
`tests/test_facet_dense_equivalence.py` proves the dense path ENGAGES for
scan-of-frames cells and agrees with the generic path cell for cell --
image (mean/median/sum), repeat facets, curve cells, histogram cells,
invalid cells included.
