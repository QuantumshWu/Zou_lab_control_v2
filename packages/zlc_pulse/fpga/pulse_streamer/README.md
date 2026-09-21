# ZLC FPGA Pulse Streamer

Vivado Tcl and HDL sources for the neutral-atom runtime pulse streamer. The
user-facing Windows entry points live at repository root in
`bin\build_and_program.bat` and `pulse_server`.

This is a short subsystem pointer. The root `ARCHITECTURE_DESIGN.md` and
`IMPLEMENTATION_PLAN.md` are the active architecture and evidence authorities;
hardware acceptance remains the runbook in `fpga\README.md`.

## Files

- `zlc_period_streamer.v`: the engine. A PERIOD TABLE held in one block RAM
  (one 128-bit row per authored period: duration or duration slot, TTL levels,
  one DAC action per bus; forced `READ_LATENCY_B=2`), a loop table of nested
  brackets (`MAX_LOOPS` entries, `LOOP_DEPTH` levels) followed by a stack
  walker, a depth-`FIFO_DEPTH` (=`RD_LAT`+4=6) continuous row prefetch that
  hides the BRAM latency so back-to-back 1-tick (20 ns) rows play one per
  clock, a scan-point prefetcher over the 2-bank continuous cyclic ping-pong
  window (`BANK_SIZE`=2048) for autonomous streamed scans, the Bresenham
  DAC ramp stepper, and the output delays -- per-channel (TTL)
  event-scheduler FIFOs + per-bus (DAC) action-descriptor FIFOs
  (`out[t]=in[t-d]`, popped against a free-running 48-bit `g_time`).
- `zlc_pulse_streamer_top.v`: top wrapper. Region-decoded BRAMs behind an
  `axi_bram_ctrl` (row table + scan window) plus register regions (the loop
  table, the per-channel/per-bus DELAY words) and a CTRL
  register file (the COMMAND/STATUS mailbox, the resident-bank `CURSOR`/`BANK_READY`/
  `BANK*_CHUNK` handshake, the CLK_ENABLE
  mask, and the hardwired `LAYOUT_ID` readback word used by the host
  register-layout handshake), driving the engine and the board output pins /
  four 10-bit DAC buses.  LOAD only marks the uploaded image resident; the
  engine reads every table directly.
- `create_project.tcl`: create project (jtag_axi + axi_bram_ctrl + 2 BRAMs),
  `zlc_force_latency2` forces both BRAMs to `READ_LATENCY_B=2`, synth,
  implement, write bitstream + probes.
- `program_fpga.tcl`: program the device with the generated `.bit`/`.ltx`.
- `diagnose_hw_target.tcl`: non-destructive hardware-target diagnostic.
- The host-side wire contract lives in `src/zlc_pulse/wire.py`. The old `host/`
  tree is deliberately absent so there is one Python owner for packing.
- `sim/`: xsim (Vivado simulator) testbenches that run the REAL RTL --- and,
  where it matters, the real block-RAM IP netlists --- covering the prefetch
  pipeline, seamless scan wrap, event-scheduler delays, ramp scans, and a
  full-chain first-frame regression (`tb_t_ff.v`).  See
  `sim/README.md` for how to run them.

## Contract Summary

Target FPGA is the Artix-7 35T `xc7a35tfgg484-2`. The default board XDC is
`fpga\board_config\board.xdc` (see `fpga/board_config/README.md`; override with
`ZLC_PS_XDC` for the Vivado build). Explicit `streamer_config.board.lanes`
indices own lane identity; XDC and top-level ports are unordered validated
projections. The bitstream is fixed; every
`On Pulse` packs a fresh program image and uploads it over the server's
transport into `axi_bram_ctrl`, then drives the CTRL mailbox. One row means "hold these outputs for this
many ticks, then enter the next row"; brackets are loop-table entries.

The server's `auto` backend selects the transport: it enumerates COM ports,
tries USB VID/PID descriptors first, and takes a UART only after the word-63
fingerprint matches; when no port matches it falls back to JTAG-to-AXI (see
`../README.md`). The UART encoder splits every request to at most 256 words,
and RTL rejects zero/oversize counts and address overflow before commit. Both
are trusted-laboratory transports rather than an authentication or
authorization boundary.

Scans use named slots: a scan point is one vector of `NUM_SLOTS` 32-bit
values, and a row reads its duration (a tick count) or a DAC action reads its
code from the slot its selector names.  The scan window is a 2-bank
ping-pong.  Prepare uploads the first two chunks before FIRE; during the run
the sole host observer uses `BANK_READY` and `BANK*_CHUNK` to refill each
released bank.  The FPGA clocks scan points autonomously through its own
scan-point prefetcher, while the host only transfers chunks.  A late or
missing refill produces `UNDERFLOW`, and the run is rejected.  Analog buses
need no separate table: every row carries one action per bus (hold, edge to
a code, or ramp from the carried level to a code over the row's whole
duration), so a ramp costs nothing beyond the row it belongs to.

One-tick rows are ordinary rows, at the start, the end or a seam of a Pulse:
the row prefetch keeps `FIFO_DEPTH` rows resident, the loop walker and the
scan-point prefetcher run ahead of the executor, and a seam (bracket rewind,
Run repeat, scan point or sweep wrap) costs no tick.  The host therefore has
no seam-margin rule to validate; it validates only capacity (rows, loops,
loop depth, delayed events in flight).

Expansion profile: `CHANNEL_COUNT=69` physical pins, 25 TTL control bits,
`NUM_SLOTS=4`, `MAX_ROWS=512`, `MAX_LOOPS=8`, `LOOP_DEPTH=4`, `BANK_SIZE=2048`
(4096 bank-local resident rows), `TICK_WIDTH=32`, `RD_LAT=2`, `FIFO_DEPTH=6`,
`EVT_FIFO_DEPTH=32`, `BUS_EVT_FIFO_DEPTH=64`, `CLOCK_HZ=50 MHz` (20 ns tick).
The affine edge-table build (2026-09-21, 4096 edges) used 19645/20800 LUTs,
14590/41600 registers, 76/90 DSPs and 37/50 BRAM tiles.  The period-table
engine drops the twelve affine evaluators, the bus segment LUTRAM and the
bus-image loader; its routed numbers are recorded in `wire.estimate_resources`
and `test_fpga_assets` after each build-only run.  No hardware was connected
or programmed; experimental pin/timing acceptance remains required.  Migration
itself never builds or programs a bitstream.

## CTRL Register-File Mailbox

The host never bit-bangs probes; it reads and writes a small CTRL register file
over `axi_bram_ctrl`. The mailbox words (see `zlc_pulse.wire.CtrlWords`):

```text
COMMAND     host -> top   rising-edge LOAD(1) / FIRE(2) / RESET(4) / SAFE(8)
STATUS      top -> host   LOADED(1) / RUNNING(2) / DONE(4) / ENGINE_ERROR(8) / UNDERFLOW(16) / LINK_ERROR(32)
PROG_COUNT                number of period rows
SCAN_COUNT                unique scan rows N in one table sweep
SCAN_ENABLE
RUN_REPEAT_COUNT          complete Pulse executions per row; 0 = infinite
SCAN_REPEAT_COUNT         complete table sweeps; 0 = infinite
LOOP_TABLE_COUNT          loop-table entries in use (nested brackets, outermost first)
BANK_SIZE / SLOT_COUNT
CURSOR      top -> host   cumulative row-visit ordinal; unchanged by Run repeats
                          current table row is CURSOR modulo SCAN_COUNT
BANK_READY  host -> top   bit b = bank b is loaded and ready
BANK0_CHUNK / BANK1_CHUNK host -> top   sweep-chunk index resident in each bank
CLK_ENABLE  host -> top   per-channel mask: output the 50 MHz clock instead of data
LAYOUT_ID   top -> host   hardwired register-layout ID (word 63); the host refuses
                          to drive a bitstream whose layout differs from its own
```

Period rows upload through the ROWS region (`ROW_WORDS` words per row), the
loop table through the LOOP region (`first | last << 16`, then the count, per
entry), and per-channel TTL delays and per-bus DA delays through the dedicated
DELAY register region (one 32-bit word per channel and per bus; see
`zlc_pulse.wire.region_bases`).

Lifecycle: `prepare` (SAFE, upload the static image and first two scan chunks,
arm both banks, LOAD) / `fire` (FIRE) / `wait_done` (the sole observer polls and
refills released banks) / `safe_state`.
`STATUS_UNDERFLOW` is fatal evidence that seamless timing was not achieved; the
run is rejected. Current Python tests compare bounded prefetch, streamed-scan,
stale-seed, TTL-delay, and DAC-delay cases against reference models; when Vivado
and generated IP are present, the same suite also runs the exact-marker xsim
matrix. Neither software lane replaces the on-board acceptance in `fpga\README.md`.
