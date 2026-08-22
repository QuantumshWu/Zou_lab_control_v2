# ZLC FPGA Pulse Streamer

Vivado Tcl and HDL sources for the neutral-atom runtime pulse streamer. The
user-facing Windows entry points live at repository root in
`bin\build_and_program.bat` and `bin\run_server.bat`.

This is a short subsystem pointer. The root `ARCHITECTURE_DESIGN.md` and
`IMPLEMENTATION_PLAN.md` are the active architecture and evidence authorities;
hardware acceptance remains the runbook in `fpga\README.md`.

## Files

- `zlc_edge_streamer.v`: the engine. A global edge table held in three parallel
  block RAMs (tick 32b / coeff 64b / mask 62b, forced `READ_LATENCY_B=2`), a
  depth-`FIFO_DEPTH` (=`RD_LAT`+3=5) continuous edge prefetch that hides the BRAM
  latency so back-to-back 1-tick (20 ns) edges fire one per clock, a 2-bank
  continuous cyclic ping-pong scan window (`BANK_SIZE`=2048, 4096 bank-local
  resident slots at one time) for autonomous streamed scans, the affine effective-tick MAC + analog-bus
  DAC engine, and the output delays -- per-channel (TTL) event-scheduler FIFOs +
  per-bus (DAC) segment-descriptor FIFOs (`out[t]=in[t-d]`, popped against a
  free-running 48-bit `g_time`).
- `zlc_pulse_streamer_top.v`: top wrapper. Region-decoded BRAMs behind an
  `axi_bram_ctrl` (edge tables + scan window + bus image) plus a CTRL
  register file (the COMMAND/STATUS mailbox, the resident-bank `CURSOR`/`BANK_READY`/
  `BANK*_CHUNK` handshake, the per-channel/per-bus DELAY words, the CLK_ENABLE
  mask, and the hardwired `LAYOUT_ID` readback word used by the host
  register-layout handshake), driving the engine and the board output pins /
  four 10-bit DAC buses.  A mini-loader copies the bus image into the engine
  LUTRAM at LOAD.
- `create_project.tcl`: create project (jtag_axi + axi_bram_ctrl + 5 BRAMs),
  `zlc_force_latency2` forces the edge BRAMs to `READ_LATENCY_B=2`, synth,
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
`On Pulse` packs a fresh program image and uploads it over JTAG-to-AXI through
`axi_bram_ctrl`, then drives the CTRL mailbox. One edge row means "at this
absolute FPGA tick, set all outputs to this mask".

JTAG-to-AXI is the current default transport. The optional UART path is for a
controlled repository host: its encoder splits every request to at most 256
words, and RTL rejects zero/oversize counts and address overflow before commit.
It is still a trusted-laboratory transport rather than an authentication or
authorization boundary.

Scans use named slots: each edge row stores a base tick plus `NUM_SLOTS`
fixed-point coefficients, and the FPGA computes
`effective_tick = base + (sum_j coeff_j * slot_j) >>> COEFF_FRAC_BITS` while
iterating the scan-point table. The scan window is a 2-bank ping-pong. Prepare
uploads the first two chunks before FIRE; during the run the sole host observer
uses `BANK_READY` and `BANK*_CHUNK` to refill each released bank. The FPGA clocks
scan points autonomously, while the host only transfers chunks. A late or
missing refill produces `UNDERFLOW`, and the run is rejected. Analog buses upload through a separate
LUTRAM segment table (`bus_id, start_tick, stop_tick, start_value, stop_value,
mode`, plus dual `value_select` for scanned endpoints) so a ramp costs one
segment, not hundreds of TTL edge rows.

The edge FIFO still supports adjacent 1-tick edges, and a finite one-shot may
end after only 1 or 2 ticks.  The registered affine boundary cache is scheduled
two clocks before a seam.  Therefore every outer cycle which has a successor
needs at least 3 ticks after its final restart; a two-pass `RepeatRegion` needs
its single boundary at or after absolute tick 3, and three or more passes need
at least a 3-tick loop span.  The host validates only the rows that actually
cross such a seam before FIRE/LOAD, rather than allowing a stale-cache cycle or
misreporting the deterministic scheduling error as a scan-refill underflow.

Frozen profile (from `zlc_pulse.wire.StreamerParams` and the deployed manifest's
explicit 98% ceiling; the generic 90% solver correctly rejects this tight 35T):
`CHANNEL_COUNT=62`, `NUM_SLOTS=4`, `MAX_EDGES=4096`, `BANK_SIZE=2048` (4096
bank-local resident slots), `TICK_WIDTH=32`, `COEFF_WIDTH=16`, `COEFF_FRAC_BITS=8`,
`RD_LAT=2`, `FIFO_DEPTH=5`, `EVT_FIFO_DEPTH=64`, `BUS_EVT_FIFO_DEPTH=64`,
`CLOCK_HZ=50 MHz` (20 ns tick). Vivado `report_utilization` is the final
resource authority; the routed 2026-08-21 calibration used by
`bin\estimate_resources.bat` is block-RAM tiles 82%, LUT 96.51%, FF 33.78%, DSP 84.44%.
This frozen 35T deployment is tight on LUTs; proposed geometry changes must be
re-estimated and routed rather than inferred from the old 2026-06 profile.

## CTRL Register-File Mailbox

The host never bit-bangs probes; it reads and writes a small CTRL register file
over `axi_bram_ctrl`. The mailbox words (see `zlc_pulse.wire.CtrlWords`):

```text
COMMAND     host -> top   rising-edge LOAD(1) / FIRE(2) / RESET(4) / SAFE(8)
STATUS      top -> host   LOADED(1) / RUNNING(2) / DONE(4) / ERROR(8) / UNDERFLOW(16)
PROG_COUNT                number of edge rows
SCAN_COUNT                TOTAL scan points N (independent of bank-local window depth)
SCAN_ENABLE / REPEAT_FOREVER
LOOP_START / LOOP_COUNT / LOOP_END_TICK / LOOP_END_LO / LOOP_END_HI
BUS_COUNTS                packed per-bus segment counts
BANK_SIZE / SLOT_COUNT
CURSOR      top -> host   scan points consumed so far (progress/terminal evidence)
BANK_READY  host -> top   bit b = bank b is loaded and ready
BANK0_CHUNK / BANK1_CHUNK host -> top   sweep-chunk index resident in each bank
CLK_ENABLE  host -> top   per-channel mask: output the 50 MHz clock instead of data
LAYOUT_ID   top -> host   hardwired register-layout ID (word 63); the host refuses
                          to drive a bitstream whose layout differs from its own
```

Per-channel TTL delays and per-bus DA delays upload through the dedicated
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
