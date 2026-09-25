# fpga/board_config — board / platform configuration

This directory contains the explicit board manifest, its XDC projection, and
the frozen deployment geometry. `streamer_config.json` `board.lanes` is the
single lane-mapping authority. Every lane has an explicit index, logical
signal, RTL port, package pin, and electrical role; file declaration order has
no meaning.

## `streamer_config.json`

This file describes the approved board and frozen deployment geometry:

| field | meaning |
|---|---|
| `fpga_part` | Vivado part used by the build and capacity estimate |
| `clock_hz` | sequencer clock; the shipped deployment is 50 MHz |
| `target_pct` | resource-budget target for the estimate |
| `board.id` | explicit board-manifest identity |
| `board.lanes` | indexed host/RTL/XDC lane and pin mapping |
| `params.channel_count` | physical raw lanes, including TTL, DAC data and latch clocks |
| `params.bus_count` / `bus_width` | DAC bus geometry |
| remaining `params.*` fields | row, loop, scan-slot, delay and FIFO limits |

`zlc_pulse.pulse_target_from_xdc()` generates the complete `PulseTarget` from
the explicit manifest indices, then requires both checked-in projections to
match: the XDC port/pin set and the RTL top assignments. The XDC does not create
lane identity.

The returned target keeps the manifest package pin for every raw lane in
`target.package_pins`. Any missing, extra, or differently pinned pulse port in
the XDC fails immediately. A top-level RTL assignment to the wrong TTL
`out_final[index]`, DAC bit or `bus_clk_final[bus]` also fails validation.

The expansion has 69 physical lanes: 25 TTL, four 10-bit DAC buses and four
latch clocks. The board plays a period table: one row per period (up to
`max_rows`), holding its tick count or the scan slot that supplies it, the 25
TTL levels, and what each DAC bus does when the row is entered; Brackets are a
separate loop table (`max_loops`, nested up to `loop_depth`). CTRL word 20
holds four bus-clock enable bits. Delay words contain the 25 TTL delays
followed by four DAC delays. The TTL event FIFO is 32 deep; the DAC event FIFO
64 deep. The register ABI version is `zlc_pulse.wire.LAYOUT_STRUCT_VERSION`
and the layout fingerprint a bitstream must report on word 63 is the
`ZLC_LAYOUT_FINGERPRINT` in the generated `pulse_streamer/zlc_geometry.vh`;
neither is restated here, so this page cannot fall behind a rebuild.

## Upgrading existing Pulse files

Run `bin\migrate_pulses.bat --dry-run` first, then the same command without
`--dry-run`. Double-click uses the discovered workspace's `pulses` and
`config_values`; passing files/folders explicitly restricts the migration.
Each changed file is validated, backed up beside its original and atomically
replaced. Running the tool again does not modify an already migrated file.

Existing waveforms stay on their physical pins: the former F13 `cooling_pgc`
is named `shutter_420`, while the added J14 `cooling_pgc` starts low. R17 `trig`
is unchanged. All six new TTL outputs start low. DAC bit order, scan rows,
timing, saved Config names/values and authored Pulse parameters are retained.
The normal loader does not silently migrate old wiring. Build/program the new
FPGA design and update client/server software together before using it; never
use a capacity estimate as a routed/timing acceptance report.

Editing JSON cannot alter a programmed FPGA. A hardware change needs an
approved rebuild and qualification. The layout fingerprint proves geometry
identity match; it is not by itself a receipt for a particular board or
qualified bitstream.

## `board.xdc`

`board.xdc` is the Vivado package-pin/electrical projection. Its
`PACKAGE_PIN` declarations may be reordered without changing `ch00`, `ch01`,
… semantics. Vivado scripts may use `ZLC_PS_XDC` only for an approved build;
the selected XDC must still equal the explicit manifest.

The validator accepts explicit XDC/config paths for tests. The normal notebook
and `pulse_server` use the checked-in manifest and projections; a remote
client does not read them.

## Who reads this directory

- `zlc_pulse.manifest` generates the runtime target from `board.lanes` and
  validates the XDC and RTL top projections.
- `zlc_pulse.wire` reads the deployment geometry from the same JSON file.
- `zlc_pulse.remote` validates both before it accepts clients.
- Vivado project and programming scripts consume the XDC for an approved build.

Run `bin\estimate_resources.bat` for the capacity report. It is an estimator,
not permission to replace the frozen bitstream.
