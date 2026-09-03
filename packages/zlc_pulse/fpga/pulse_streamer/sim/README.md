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
  exact 1/2-tick finite one-shots, dense edge prefetch, gap-dependent
  complete-Pulse Run repeats, and a non-zero-start finite `PulseBracket` with
  distinct preamble/body/tail.
- `tb_scan_wrap.v` covers PulseBracket, per-row Run repeats, finite Scan repeats,
  a streamed three-chunk table, cumulative row cursor, and cyclic two-bank wrap.
- `tb_delay_sched.v`, `tb_delay_compact.v`, `tb_evt_depth.v`, and
  `tb_bus_delay.v` cover the current 32-bit TTL event and DAC segment delay
  schedulers.
- `tb_ramp_scan.v` and `tb_da_ttl_align.v` cover DAC ramps, affine boundary
  cache ownership (including late-bank recovery), and TTL/DAC alignment.
- `tb_real_engine.v` uses the generated edge BRAM simulation models.
- `../tb_uart_pipeline.v` and `../tb_uart_read_tap.v` require exact write
  commit counts/address order while covering pipelining, watchdog/bounds,
  last-word delivery, readback, and the layout identifier.
- `tb_t_ff.v` is the retained full-top consecutive-FIRE regression. Its
  committed `replay_t.vh` and `replay_t_frame.vh` are a literal current-layout
  host image; the bench checks that repeated runs produce identical first
  frames through the CTRL decoder, bus loader, engine, and output mapping.

The former captured-session replay (`tb_full_top.v`/`replay_image.vh`) was
removed because it encoded an obsolete register map and had no current
generator. A stale historical trace is not valid deployment evidence.

## Running a bench

Build artifacts for Xilinx IP simulation models are required. A build is only
performed inside the separately approved evidence-driven hardware workflow.
With those artifacts already available, invoke `xvlog`, `xelab`, and `xsim`
from this directory, for example:

```sh
VIV=/c/Xilinx/Vivado/2019.1/bin
IPT=../../build/ps/ps.srcs/sources_1/ip/blk_mem_gen_edge_tick
IPM=../../build/ps/ps.srcs/sources_1/ip/blk_mem_gen_edge_mask
"$VIV/xvlog" ../zlc_edge_streamer.v \
  "$IPT/sim/blk_mem_gen_edge_tick.v" \
  "$IPT/simulation/blk_mem_gen_v8_4.v" \
  "$IPM/sim/blk_mem_gen_edge_mask.v" \
  "$IPM/simulation/blk_mem_gen_v8_4.v" \
  tb_real_engine.v
"$VIV/xelab" work.tb_real_engine -s sreal
"$VIV/xsim" sreal -runall
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
