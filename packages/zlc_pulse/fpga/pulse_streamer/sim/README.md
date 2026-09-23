# Current xsim checks

These optional Vivado `xsim` benches exercise the checked-in frozen RTL. They
are hardware-development evidence, not part of normal experiment startup and
not permission to rebuild or program a board.

`test_command_strobe.py::test_vivado_rtl_matrix_requires_each_numeric_oracle`
automatically compiles and runs the 10 self-checking engine benches plus both
UART benches when Vivado and generated BRAM models are present, otherwise it
reports an explicit skip. It requires each exact success marker and rejects
`Fatal`/`FAIL`/`BAD`/`LATE` text because Vivado 2019.1 may still return process
code zero after `$fatal`.

The maintained benches are self-contained except where explicitly noted:

- `tb_1tick.v`, `tb_gapsweep.v`, and `tb_loop.v` cover
  exact 1/2-row finite one-shots, dense one-tick rows around a prefetch
  bubble, gap-dependent complete-Pulse Run repeats, and a non-zero-start
  finite nested bracket (outer x3 with an inner x2) with distinct
  preamble/body/tail on TTL and DAC, then one-tick rows through four nested
  brackets that share start rows and end rows (every tick's mask against the
  hand-expanded play order).
- `tb_scan_wrap.v` covers a whole-timeline bracket, per-row Run repeats, finite
  Scan repeats, a streamed three-chunk table, cumulative row cursor, and the
  cyclic two-bank wrap through the scan-point prefetcher.
- `tb_delay_sched.v`, `tb_delay_compact.v`, `tb_evt_depth.v`, and
  `tb_bus_delay.v` cover the current 32-bit TTL event and DAC action delay
  schedulers.
- `tb_ramp_scan.v` and `tb_da_ttl_align.v` cover slot-targeted DAC ramps
  (gentle and steep Bresenham), a late bank held with UNDERFLOW and resumed,
  and TTL/DAC alignment.
- `tb_real_engine.v` uses the generated row BRAM simulation model.
- `../tb_uart_pipeline.v` and `../tb_uart_read_tap.v` require exact write
  commit counts/address order while covering pipelining, watchdog/bounds,
  last-word delivery, readback, and the layout identifier.
- `tb_t_ff.v` is the retained full-top consecutive-FIRE regression. Its
  committed `replay_t.vh` and `replay_t_frame.vh` are a literal current-layout
  host image (the 9-period, 116-tick Pulse described in the bench header,
  packed by `zlc_pulse.wire.pack_program` and written as one `wr(...)` line
  per sparse word; regenerate them the same way after any layout change); the
  bench checks that repeated runs produce identical first frames through the
  CTRL decoder, loop/delay registers, engine, and output mapping.
- `diff/` is the differential harness: two BUILT checkouts (the last edge-table
  build and the period-table build) play the same authored pulses through their
  own real tops in xsim and the per-clock pin dumps are compared; its README
  records what was compared when the period table replaced the edge table.

The engine benches other than `tb_real_engine.v` model the row and scan BRAMs
behaviorally (a registered address plus three pipeline stages, the RD_LAT+2
issue-to-data latency of the generated IP), so they run without a build.

## Running a bench

Build artifacts for Xilinx IP simulation models are required. A build is only
performed inside the separately approved evidence-driven hardware workflow.
With those artifacts already available, invoke `xvlog`, `xelab`, and `xsim`
from this directory, for example:

```sh
VIV=/c/Xilinx/Vivado/2019.1/bin
IPR=../../build/ps/ps.srcs/sources_1/ip
"$VIV/xvlog" -i .. ../zlc_period_streamer.v   "$IPR/blk_mem_gen_rows/sim/blk_mem_gen_rows.v"   "$IPR/blk_mem_gen_rows/simulation/blk_mem_gen_v8_4.v"   tb_real_engine.v
"$VIV/xelab" work.tb_real_engine -s sreal
"$VIV/xsim" sreal -runall
```

Every bench `include`s `../zlc_geometry.vh`, so `xvlog` needs `-i ..` (the
directory of the RTL) whatever the working directory is.  The full-top bench
uses `$isunknown` and therefore compiles in SystemVerilog mode; with the
generated IP models it replays the frozen host image through the real BRAMs,
and with `-d ZLC_IVERILOG` its IP-free `tb_safe_gate` proves the SAFE pin gate:

```sh
"$VIV/xvlog" -sv -i .. ../zlc_uart_bridge.v ../zlc_period_streamer.v ../zlc_pulse_streamer_top.v   "$IPR/blk_mem_gen_rows/simulation/blk_mem_gen_v8_4.v" "$IPR/blk_mem_gen_rows/sim/blk_mem_gen_rows.v"   "$IPR/blk_mem_gen_scan/sim/blk_mem_gen_scan.v" tb_t_ff.v
"$VIV/xelab" work.tb_t_ff -s stff && "$VIV/xsim" stff -runall      # T-FF-OK, RESIDENT-REPLAY-SAFE-INTERRUPT-DEDUP-OK
"$VIV/xvlog" -sv -d ZLC_IVERILOG -i .. ../zlc_uart_bridge.v ../zlc_period_streamer.v ../zlc_pulse_streamer_top.v tb_t_ff.v
"$VIV/xelab" work.tb_safe_gate -s sgate && "$VIV/xsim" sgate -runall  # TOP-SAFE-PIN-GATE-OK
```

This particular `tb_real_engine.v` example prints seven diagnostic emCCD pulse
widths (each exactly 2000 ticks) and a final `DONE` note but has no self-checking
PASS marker. Its transcript must be inspected against those widths; it is not a
stand-alone pass oracle. For self-checking benches, absence of the exact success
marker or any failure token is failure even when `xsim` itself exits successfully.

The real-IP/full-top benches use the frozen deployment geometry.  Focused
behavioral benches may instantiate a deliberately narrower geometry to isolate
one scheduler or scan rule; they are semantic unit oracles, not resource or
deployment-timing evidence.
