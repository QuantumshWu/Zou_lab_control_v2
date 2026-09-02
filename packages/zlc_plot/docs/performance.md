# Plot performance

## Current performance baseline (2026-08-20)

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

## SmartOffset steady-frame memo (2026-08-21)

The locator now memoizes a completed layout only when its complete input is
unchanged: directed limits, axis/figure identity, drawn point extent/orientation,
locator policy, label size, measurement function and font family. Range direction,
resize, font or policy changes remain misses. The A/B used two clean
`55d6ee7` worktrees, with only the 15-line locator cut applied to the candidate;
each side ran three independent processes in alternating order and 60 steady
updates per case (Rolling first warmed 110 revisions). Each table entry is the
median of the three runs' own P50/P95, in milliseconds.

| Case | Complete update, old → memoized | Compose, old → memoized |
|---|---:|---:|
| Image 96×128 `uint16` | 12.47955/14.41273 → 11.91770/13.73864 | 8.55010/10.225685 → 7.92430/10.122235 |
| ROI Image 26×26 `uint16` | 11.47305/13.118625 → 10.76450/12.59024 | 7.83830/9.770585 → 6.93355/9.05635 |
| Curve, 120 scalars | 2.52675/3.54965 → 1.92755/3.474205 | 1.65015/2.54599 → 1.07115/2.088035 |
| Histogram, 1000 scalars | 12.63590/25.625325 → 9.08785/23.71979 | 11.42575/23.980615 → 8.17395/20.862275 |
| Rolling, window 100 | 15.27245/25.151425 → 12.51480/25.565915 | 13.79970/23.572395 → 11.08160/24.294440 |
| FacetGrid ×4, 26×26 cells | 10.22625/21.88321 → 7.53110/14.39510 | 6.31940/15.82574 → 3.75645/7.25682 |

Independent 20-frame instrumentation counted the outer `tick_values` call on
every draw; `_unit` and `_lay_out` counted only real cache misses. Values below
are calls/frame and cumulative milliseconds/frame.

| Case | `tick_values`, old → memoized | `_unit`, old → memoized | `_lay_out`, old → memoized | Hits/misses per frame |
|---|---:|---:|---:|---:|
| Image | 4/0.585545 → 4/0.026715 | 4/0.450570 → 0/0 | 42/0.329300 → 0/0 | 0/4 → 4/0 |
| ROI Image | 4/0.696065 → 4/0.025945 | 4/0.540975 → 0/0 | 40/0.382150 → 0/0 | 0/4 → 4/0 |
| Curve | 4/0.580430 → 4/0.019165 | 4/0.442020 → 0/0 | 42/0.338165 → 0/0 | 0/4 → 4/0 |
| Histogram | 16/2.598680 → 16/0.288865 | 16/1.931810 → 1/0.161680 | 152/1.488330 → 9/0.115840 | 0/16 → 15/1 |
| Rolling | 16/2.282030 → 16/0.244830 | 16/1.679395 → 1/0.126180 | 144/1.348515 → 8/0.091440 | 0/16 → 15/1 |
| FacetGrid ×4 | 16/2.010510 → 16/0.059955 | 16/1.504710 → 0/0 | 144/1.049140 → 0/0 | 0/16 → 16/0 |

Every case improved P50 in every process. Rolling alone had a small median P95
tail change of +0.41449 ms complete / +0.722045 ms compose while its independent
process tails ranged 22--28 ms; its P50 improved by 2.75765 / 2.71810 ms. The
style-only Image fit change was separately rendered as standalone and four-cell
Facet surfaces at DPR 1/2 and `2x2`/`8x8`: the occupied-style hollow ring and
`2.25 pt²` center remained visible, and the existing DPR1/2 compose check stayed
pixel-identical to a full draw.

## Histogram representation edits and settled-tick bound (2026-08-21)

`bin_count` changes scientific edges/counts, so each accepted edit performs
exactly one necessary payload projection and one render. `density` and
`cumulative` change only the representation of the already projected
`HistogramData`; they reuse the same counts/edges identity, perform zero payload
projections and exactly one render. They still refit the axes and invalidate fit
selection as before. This is one shared Histogram lifecycle, not a second
plot-kind lane.

The A/B compared a clean detached `9571cd6` worktree with the frozen candidate.
It used fixed-seed continuous `float32` samples, `2048×2048` qCMOS and
`1200×1920` MOT payloads, the `2x2` preset, fresh `PlotSession` instances and
three rounds. Bin edits used `32 → 64 → 128` in one session per round;
density/cumulative each used three fresh sessions. Values below are aggregate
primitive wall P50/P95 in milliseconds; profiling and representation/payload
inspection were outside the timed region.

| Payload / edit | Baseline | Candidate |
|---|---:|---:|
| qCMOS bins | 71.539/73.763 | 71.957/74.487 |
| qCMOS density | 73.049/73.128 | 7.448/9.520 |
| qCMOS cumulative | 71.301/71.973 | 7.709/7.880 |
| MOT bins | 45.057/46.206 | 44.355/45.149 |
| MOT density | 46.257/46.429 | 8.508/9.354 |
| MOT cumulative | 44.607/45.628 | 9.212/10.026 |

An instrumented qCMOS density edit changed from one payload projection at
62.469/64.193 ms to zero calls, while retaining one render at
8.716/10.264 → 9.270/9.330 ms. A 120-bin edit retained exactly one projection
(65.609/66.796 → 64.251/64.708 ms) and one render. The change therefore
removes the redundant O(N) sample materialization/rebin only from
representation edits; no memory percentage is claimed. Baseline and candidate
RGBA hashes matched for base, bins, density, cumulative and combined modes at
DPR 1 and 2 (ten exact comparisons).

The settled tick unit is only a hysteresis candidate, not permission to
enumerate an arbitrarily dense old lattice. Before this cut, the real 2048²
RasterHost sequence `density On → cumulative On → density Off` remained in
the `Decimal` lattice loop after five seconds. The candidate checks the implied
count before enumeration; a count above `max_ticks=8` enumerates zero old-unit
ticks and falls back to the existing unit search. Every actual `range` in the
old-red proof was at most `max_ticks + 1 = 9`, and final ticks were exactly equal
to a fresh locator. The same full host sequence promoted fronts 1 through 7
without timeout or error; after the initial/bin work, representation edits were
8.08--9.85 ms.

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

The primary live camera-cycle contract is
`(R=1, P=frames_per_cycle, camera_y, camera_x)`.  The frame identity is one
`READOUT_EVENT` point column; both spatial axes are declared dense `data_dim`
axes and the `ImagePlot` refers to them with `AxisRef.data(...)`. Every public
live camera cycle keeps that fixed event geometry. A capable derived output
materializes a bounded ordinary Dataset over its source primary index only
while a real window consumer holds a lease; before that it retains latest only.
The Surface host never retains a FIFO of full camera frames.

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

For regular images, the default lower radius bound is half the finest native
coordinate spacing, not a fraction of the noise-sensitive moment seed. The
moment initializer still supplies the upper radius and every other model
bound; an explicitly authored bound remains authoritative. Radial and
anisotropic Gaussian kernels use the same metadata-driven rule. When the
bounded proxy is already the full image, a successful exact solve is also not
invalidated merely because a retry without material cost improvement returns
an unsuccessful status. A materially better retry is still retained and must
pass its own status. This prevents a valid FitEvent from becoming a false gap
without weakening the full-resolution objective or adding a model-specific
acceptance path.

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
oversubscribe 4/8-panel runs or outlive sessions. Current large-component
projection and complete-update measurements are recorded below; the former
10–25 ms blanket projection estimate is no longer current.
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

## Large-component projection matrix (2026-08-21)

The frozen baseline was compared with `e8e5517` using the same script and one
fresh process per case. Each process warmed one replace/capture, measured the
listed number of complete `session.update_data` plus front captures, profiled
three separate frames, and measured one separate allocation frame. Times are
P50/P95 milliseconds; allocation is the `tracemalloc` peak in MiB. The source
shape, dtype and frame count were identical on both sides.

| Case | Shape / frames | Complete update, old → current | Projection, old → current | Allocation peak, old → current |
|---|---|---:|---:|---:|
| qCMOS virtual Image | 96×128 `uint16` / 20 | 10.43/12.15 → 10.14/11.57 | 0.202/0.268 → 0.163/0.229 | 0.87 → 0.86 |
| qCMOS Image | 2048² `uint16` / 12 | 35.42/38.05 → 21.60/22.39 | 15.89/16.78 → 0.319/0.725 | 44.13 → 5.37 |
| MOT Image | 1200×1920 `uint8` / 12 | 26.81/28.83 → 19.13/20.98 | 8.84/9.49 → 0.311/0.451 | 24.27 → 4.15 |
| MOT FacetGrid ×1 | 1200×1920 `uint8` / 10 | 49.97/51.55 → 10.33/11.76 | 38.42/39.74 → 0.363/0.417 | 81.40 → 2.21 |
| MOT FacetGrid ×4 | 1200×1920 `uint8` / 10 | 183.48/191.71 → 24.05/27.13 | 153.85/158.19 → 0.489/0.921 | 281.35 → 1.99 |
| MOT FacetGrid ×8 | 1200×1920 `uint8` / 10 | 366.48/377.69 → 45.24/46.98 | 309.99/315.49 → 0.821/0.970 | 562.60 → 2.55 |
| qCMOS Curve | 2048² `uint16` / 12 | 9.05/10.10 → 8.97/10.34 | 5.75/6.46 → 5.39/6.77 | 0.94 → 0.94 |
| qCMOS Histogram | 2048² `uint16` / 12 | 175.79/178.02 → 33.13/34.80 | 163.65/165.54 → 21.90/22.62 | 72.00 → 32.13 |
| qCMOS Rolling | 2048² `uint16` / 12 | 111.80/112.77 → 15.14/17.30 | 97.30/99.08 → 2.42/3.19 | 148.00 → 0.72 |
| indexed Rolling | 1×100 `float64` / 20 | 14.79/17.14 → 15.13/16.90 | 1.71/3.18 → 1.73/2.93 | 0.81 → 0.81 |

The large gains come from geometry- and dtype-based shared kernels, not
plot-kind scheduling lanes. A singleton dense Image retains the producer's
native values and boolean validity. Contiguous FacetGrid rows remain views;
multi-sample cells reduce through the common dense reducer, while irregular,
grouped and DATA-faceted cases retain the generic grouping fallback. Sparse
domain queries gather only the requested positions, whereas a full-domain
generic projection retains its flat-coordinate cache.

Histogram domain discovery takes native integer min/max without first making
a full `float64` pool. Strictly integer-aligned edges use one bounded
`bincount`; floats, non-uniform or nearly aligned edges, sparse integer spans
and values outside the safe `int64` range fall back to `np.histogram` exactly.
Rolling reduces one cached valid pool directly when it is ungrouped; grouped,
repeat-seeded and primary-index history retain their authored index semantics.
Randomized comparison covered 194 dense/sparse/grid/validity/group/reduction
projections, 954 Histogram fast/fallback cases and 21 Fit cases without a
numeric mismatch.

The final residual sweep also removed two first-frame copies. For an R=1
2048² initial history, pool projection fell from 11.02/12.79 to 0.117/0.130 ms
P50/P95 and its allocation peak from 32.01 MiB to 0.005 MiB; complete initial
Rolling session construction fell from 44.24/44.53 to 31.97/34.00 ms. Four
1200×1920 Histogram facet cells now pass their contiguous values and validity
views directly: projection fell from 50.43/51.13 to 40.15/41.26 ms and peak
allocation from 19.79 MiB to 17.59 MiB, with exact edges and counts.

## Facet, fit and kernel round (September 2026)

Session-layer medians at 2x2, offscreen (`bench/plot_perf/run_session`),
before -> after: facet64_curve 29.3 -> 17.8, facet64_histogram 45.6 -> 34.2,
facet64_image 22.6 -> 17.2, facet34_mixed 48.1 -> 37.1, rolling_2M
18.6 -> 14.4, image_camera_4M 16.2 -> 14.7, fit_facet10 solve 20.3 -> 13.7 ms;
deep indexed rolling (35 sites x 5000 window) update 34.9 -> 27.9 ms.
Isolated MOT-ROI cases (`run_mot_roi_isolated`): camera-grid render
29.1 -> 18.2, facet-curve-fit-40 total 47.6 -> 34.2, facet-curve-40
10.9 -> 6.5 ms.

What changed: `raster_polylines` strokes disjoint clip lanes in parallel
(the error-bar kernel's own grouping), and the facet caller applies a
linear cell's transData as the affine it is; the whole-pixel cell-box
searches are memoized on (plan box, figure size, ratio); the centred
square-sum kernel accepts size-1 axes inside its kept block, which had
been sending every facet-curve sem down a full centred-copy einsum;
`block_sum_float` folded into `block_sum_valid` behind a loop-invariant
flag (measured faster on both faces); the uncertainty band's two edges
ride one `transform_curve_batch` launch; batch fit proves each shared
coordinate object once (digest and finiteness).  The damped-sine seeder's
frequency scan swapped its per-sample trig calls for Goertzel
(60.6 -> 31.9 ms solve; a measured-phase seed was tried and reverted --
it made the solve basin-sensitive).  Refused with numbers:
`uniform_histogram` into `uniform_facet_histograms` (+18% on the single
histogram hot loop even with a hoisted single-facet branch).

Four-panel contention, measured on the MOT-ROI chain: worker-thread
masks 2/4/8 move the causal critical path 90.0/84.5/79.2 ms (mild
under-parallelization, no oversubscription); `OMP_WAIT_POLICY=ACTIVE`
is not faster (wake latency is not the wall-cpu gap); with
`ZLC_PLOT_KERNELS=numpy` the same chain's composes inflate ~6-10x with
cpu~wall -- the nogil kernels are what keep four panels affordable.
The residual console/isolated render ratio (camera-grid ~2x) is
scheduler-level sharing among four worker threads, their kernel teams
and the GUI thread; it shrinks only by making composes cheaper.

## Compose and stroke round (September 2026, second pass)

The question was what the four-panel chain still paid per panel after the
facet/fit/kernel round, measured on the same MOT-ROI chain
(`run_mot_roi_chain`), the isolated same-source cases
(`run_mot_roi_isolated`) and the 2x2 layout (`run_console`), all at DPR 3.
Archives: `bench/results/final_dceb240` (before) and `bench/results/round2`
(after, plus the band A/B runs).

What the probes found, per frame of the focused camera cell (isolated
compose 10.6 ms): 29 dynamic artists repainted through matplotlib, of
which three whole Axis objects -- the colour scale's long axis (2.7 ms)
and the distribution rail's two axes (1.3 each) -- and the ROI annotation
text (0.8) were 5.3 ms; inside the colour scale's axis, one rotated label
cost 1.04 ms, 0.8 of it Agg's bilinear rotation of a bitmap that never
changes.  The curve panel's forty grouped series stroked on one serial
lane: 6 ms of error bars and 3.3 ms of lines while the pool idled.  The
isolated facet-curve case had been drawn without uncertainty bars while
the console draws them by default, so its "same source" number was not
the same picture (fixed in the harness; its before/after is therefore not
comparable).

What changed:

* Text raster memo on the Agg renderer (`_prepare_renderer`): a string's
  raster is a function of string, font, angle and antialiasing; it is
  taken once through Agg's own machinery and blitted back at every later
  position with Agg's rounding to the anchor pixel.  The rotated label
  went 1.04 -> 0.04 ms.  Rotated strings replay only when opaque and
  unclipped -- Agg's rotated path quantizes alpha twice and clips through
  the rasterizer, which the blit does not reproduce -- and mathtext keeps
  the original route.
* Dynamic axes replay their recorded renderer calls while the facts their
  draw is a function of hold (axes box, view limits, locator and formatter
  with fixed values, label, pad, tick parameters); a key seen twice
  running is recorded, from then on replayed, and a key that changes every
  frame (a colour scale under TIGHT re-fitting per shot) is drawn plainly
  and never recorded.  Recordings are forgotten with the background.
* The two stroke kernels cut a lane with fewer peers than the pool has
  threads into column bands, every band replaying every primitive in
  painter order over its own columns -- bit-identical to the serial
  kernel on 64 captured product frames at every band count; the polyline
  envelope's square roots are one table per line; the band count comes
  from Python (`stroke_bands`), because a thread query inside the kernel
  made it uncacheable.

Refused or reverted, with numbers: walking a rectangle row-first (same
time as column-first: the blend arithmetic is the cost, not the cache
line); a unit-coverage interior fast path (same time); the text memo for
translucent rotated strings (one channel off by one: Agg rounds alpha
twice on that path); full-pool bands in the console (see below).

Results.  Isolated render medians: camera-grid 18.2 -> 16.5,
standalone-image 17.6 -> 15.3, curve-40 13.5 -> 6.6 ms; facet-curve-fit-40
now measures 27.3 render / 46.0 total with its bars on.  Layout, four
panels at 9.1 fps, render per frame: image 34.6 -> 25.8 wall (17.6 ->
11.7 cpu), rolling 14.6 -> 9.2, curve 12.0 -> 5.9, histogram 3.0 -> 3.7
(noise).  Chain, per source revision: camera-grid commit 47.6 -> 29.7
wall (compose 27.3 -> 7.0), curve-40 commit 16.7 -> 13.0, histogram
unchanged, facet-curve-fit-40 37.6 -> 35.6; source 7.5/s, no stalls.

The four-panel critical path did not move: 82.7 -> 83.1 ms median.  It
is the facet-curve-fit panel end to end (route 1.6 + projection 12.3 +
fit 25.2 + commit 35.6), and nothing in this round touched its floors:
the batch fit itself (21.6 ms of numeric solve for forty cells), twenty
thousand error bars whose caps at 0.3 px spacing are 9.9 Mpx of blending
per frame (a presentation fact -- caps at every point -- not a kernel
fact), and forty mathtext annotations parsed per frame under the GIL
(5-10 ms).  The other three panels now finish well inside it.

Band count is a shared-pool decision, measured: the default worker mask
is the whole pool (16 threads here), and cutting a lone lane into sixteen
bands from four workers at once moved the critical path from 83 to
105 ms -- every panel's projection, fit and compose inflated under the
oversubscription.  Bands forced to one restored 83; four bands held 84
and still took the curve panel's compose from 15.1 to 11.9 ms.  The cap
is four.  The previous round's mask sweep (2/4/8 -> 90/84.5/79.2 ms)
suggests a smaller default worker mask would help the chain; that is a
product decision and was not taken here.

Two measurement lessons worth keeping: kernel inputs captured by
reference belong to a later frame (the renderer reuses its geometry
buffers), so a golden capture copies at capture time; and a same-source
comparison is only a comparison when both sides ask for the same picture.

## Indexed history at depth (September 2026, third pass)

The scenario: a qCMOS camera, the occupancy processor's `counts`, a
FacetGrid of histogram cells faceted by site, a 3000-shot window and a
live bimodal fit in every cell.  The bench for it is
`bench/plot_perf/run_indexed_history.py --kind facet_histogram --site-axis
cell --fit bimodal_gaussian` (`--site-axis point` is the per-site point
table whose history rows multiply by the site count).  Medians per shot,
`update_data` of the isolated session (projection + fit + compose, no Qt),
pushed baseline -> this tree:

| geometry                         | window | before  | after |
|----------------------------------|-------:|--------:|------:|
| 35 sites in the cell, fit        |   3000 | 176 ms  | 69 ms |
| 35 sites in the cell, fit        |     40 | 104 ms  | 70 ms |
| 35 sites in the cell, no fit     |   3000 |         | 20 ms |
| 35 sites in the cell, no fit     |      1 |         | 17 ms |
| 64 sites in the cell, fit        |   3000 | 266 ms  | 118 ms |
| 64 sites in the cell, no fit     | 1/3000 |         | 31 / 31 ms |
| 35 sites as point rows, fit      |   3000 | 2636 ms | 77 ms |
| rolling, 35 sites                | 1/3000 |  7 / 15 | 7 / 15 ms |

What changed:

* ONE reading of the history's layout.  "Which rows belong to the last N
  shots" was answered four times: the rolling trace through the domain
  machinery, every histogram and facet path by a Python walk over the
  rows plus `np.isin` over dtype=object (1.2 s per call at 105k rows,
  twice a shot), the compatibility gate by another Python walk per shot,
  and the title by a set over the column.  `zlc_data.snapshot_projection.
  indexed_history_layout` now derives shots, rows per shot and the
  repeating event once, vectorised, cached on the schema; the window
  mask, the rolling shot codes and source indices,
  `indexed_schemas_compatible` and `schema_structure` all read it.
* A cell's fit label is two artists: the catalogue's mathtext symbol,
  constant while the parameter is chosen and parse-cached by Matplotlib,
  and a plain value placed after it by the symbol's measured width.  One
  Text carrying the whole live line parsed it as MathText on every shot
  (thirty-five pyparsing runs a frame, half of the update).  The
  batched-mask blit that existed for this ran only behind the native
  curve and image rasters and painted masks that were not the full
  draw's pixels; it is gone with its cache and every exclusion it needed.
* A cell title is replayed above the chrome it may touch: the full draw
  paints the title after the spines, the composed frame had held it in
  the background under the replayed spine -- one level, six pixels, in
  every histogram grid.  Replayed text is faded rather than hidden for
  the background draw, because Matplotlib positions a title from its own
  extent during the draw and a hidden Text reports a unit box.
* A labelled point coordinate names its distinct values from their first
  rows instead of a dict over every row (105k `_python_scalar` calls a
  shot for a 35-value domain).

Where the remaining 69 ms go (35 cells, profile of one shot): drawing
the cells through Matplotlib's Agg path pipeline -- thirty-five bar
collections rebuilt and stroked (~15 ms), a hundred and five fit
polylines (~10 ms), seventy label texts laid out and blitted (~8 ms);
the fit batch itself 7 ms (4 ms of numeric solve); projection, history
mask and binning under 3 ms.  The window depth no longer matters (40
and 3000 shots draw alike); the cost is proportional to cells x artists.
The next floor would be a native raster for histogram bars and their
fit curves, as the curve cells already have, which is a larger project
and was not taken here.

Two observations, not changes: a bimodal fit over cells holding one or
a few samples (window 1) spends 75-90 ms a shot converging to nothing,
ten times its cost on a real histogram -- a fit that lacks the samples
for its parameters would be cheaper refused than solved; and at window 1
the p90 of any kind is dominated by the plane's ramp, not the plot.

## Poisson-Gaussian histogram models (2026-09-02)

Two histogram models joined the catalogue: `histogram_poisson_gaussian` and
`bimodal_poisson_gaussian`, the exact convolution of a Poisson photon count on
the integer lattice with Gaussian read noise (parameters `A, lambda, sigma`
per state; the bimodal is parameterised by `lambda_L` and the splitting
`delta`, its headline).  The lattice sum has one implementation, the compiled
kernel: one Poisson table per objective evaluation (exact at the mode, walked
outward), three exponentials per bin, a five-multiply recursion per lattice
term, and the sum restricted to the intersection of the +-8 sigma Gaussian
window with the rate's own window (`rate - 10 sqrt(rate) - 10` to
`rate + 12 sqrt(rate) + 20`, outside which a term is under e^-50 of the
mode).  The SciPy path and the overlays call the same kernel; the frozen
anchors hold it to independent arithmetic.  Seeds come from the histogram's
weighted median and quartile range, not its moments: a hot-pixel spike far
from the peak put the moment seed in a flat valley (rate on its zero bound
under a ten-photon-wide bump).

`python -m bench.plot_perf.run_fit_models --rounds 8`, `session.fit` wall
on the sweep's own two-normal histogram (states at 8 +- 2 and 30 +- 4, so
the read-noise widths here are two and four lattice steps -- wide for this
model, whose regime is under one), median ms:

| Model | Before this round | After |
|---|---:|---:|
| histogram_gaussian | 3.03 | 3.04 |
| bimodal_gaussian | 3.82 | 3.50 |
| histogram_poisson_gaussian | -- | 5.45 |
| bimodal_poisson_gaussian | -- | 3.91 |

The single Poisson-Gaussian row is a misfit by construction (one state
asked to cover two, so its width runs to twelve lattice steps and the solver
iterates longer); the bimodal row is the model on data shaped for it, 1.1x
the Gaussian bimodal.  On a photon-count histogram (states at 0.7 and 6
photons, read noise 0.45, 64 bins) the compiled solves are 0.7 / 1.0 ms for
the Poisson-Gaussian single / bimodal against 0.6 / 0.7 ms for the Gaussian
pair, and the compiled and SciPy optima agree to 3e-15.

`python -m bench.plot_perf.run_mot_roi_chain --panel3 histogram --fit-model
<model> --seconds 12` (new `--panel3 histogram`: forty source-index
Histogram cells of the 40x500 MOT ROI, live bimodal fit per cell; pixel
values, so the read-noise widths are tens of lattice steps -- the worst
regime for the lattice sum):

| panel3 facet-histogram-fit-40 | bimodal_gaussian | bimodal_poisson_gaussian |
|---|---:|---:|
| fit_total wall / cpu ms per call | 18.6 / 8.0 | 22.3 / 13.9 |
| numeric_fit_batch wall / cpu | 13.7 / 3.7 | 13.0 / 4.3 |
| four-panel critical path median | 94.4 ms | 97.2 ms |
| all four panels | 7.7 fps | 7.6 fps |

The first Poisson-Gaussian cut measured 259 ms `fit_total` here: not the
solve (15 ms) but forty cells' overlays evaluated through a NumPy twin of
the lattice sum over hundreds of terms per bin.  Deleting the twin and
routing the evaluator through the compiled kernel is what the table shows.
