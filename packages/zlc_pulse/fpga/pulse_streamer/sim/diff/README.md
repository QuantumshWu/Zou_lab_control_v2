# Differential simulation: does a new engine play the same pins as the old one?

`run_diff.py` takes two BUILT checkouts -- one with the edge-table streamer
(`zlc_edge_streamer.v`, last built at `codex/fpga-ttl25-20260921` = 0f7dd53c) and
one with the period-table streamer (`zlc_period_streamer.v`) -- authors a set of
short pulses with the new checkout's model (`make_diff_pulses.py`, both file
formats), lets EACH side's own host code compile and pack them
(`pack_diff.py`), plays each image through that side's REAL
`zlc_pulse_streamer_top` with its generated BRAM IP models in xsim
(`tb_diff.v`: SAFE, upload, LOAD, run count, BANK_READY, FIRE), dumps every
running clock's 25 TTL pins, 4x10 DAC data bits and 4 DAC clocks, and compares
the two dumps clock for clock after aligning on the first TTL activity.

```sh
python run_diff.py --old C:/path/to/edge-table-checkout --new C:/path/to/period-table-checkout
```

Outputs land in `out/` (git-ignored): per-side `work_*/` with the xsim logs,
images and dumps, and `report.json`.  Vivado's `xsim.bat` splits an argument at
`=`, so the bench reads its parameters from `diff_args.txt` in the work
directory rather than from plusargs.

## What was compared on 2026-09-23 (period table merged into master)

Fifteen variants, all within what the old engine could express (at most one
bracket): one-tick rows next to 100- and 250-tick rows; DAC edges and Bresenham
ramps on three buses; a whole-pulse bracket x3 with Run repeats; partial
brackets starting on a plain row, on an edge row and on a ramp row; TTL delays
(3 and 1 ticks) and a DAC bus delay (2 ticks) across the run seam; a burst of
one-tick rows x3 with and without a bracket; a three-point scan table driving
a duration slot and a DAC slot, alone, with a bracket, and with delays.

| variant | result |
|---|---|
| v1 plain, v2 DAC edge+ramp, v3 whole bracket x3 run 2, v5 delays run 2, v6 partial bracket + delays, v7 one-tick burst x3, v9 bracket starting on an edge row, v12 scan, v14 scan + bracket + run 2, v15 scan + delays | identical, every clock |
| v4, v8, v10, v11, v13: a bracket whose LAST row ends with a ramp | differ only on that ramp's DAC bus, from the rewind clock on; TTL and DAC clocks identical |

The five differences are one behaviour of the OLD engine: a ramp's target is
output on the first tick of the next row, and when that next row is reached by
a bracket rewind the old engine never output it -- the DAC stayed one Bresenham
step short (384 instead of the programmed 312 in v4; 910 instead of 912 in
v10, which then made the next iteration's ramp start one step low) until the
next DAC action.  Its own delayed path (v6) did output the target at the same
rewind, so the old engine disagreed with itself.  The period-table engine
outputs the target on the rewind clock exactly as on a linear row change, in
the direct and the delayed path alike.  The scan variants show that the old
base/delta-times-scale slot encoding and the new absolute slot values play the
same pins.  FIRE-to-first-running-clock latency was the same on both sides.
