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
| `params.channel_count` | number of raw pulse lanes |
| `params.bus_count` / `bus_width` | DAC bus geometry |
| remaining `params.*` fields | edge, scan, coefficient, delay, and FIFO limits |

`zlc_pulse.pulse_target_from_xdc()` generates the complete `PulseTarget` from
the explicit manifest indices, then requires both checked-in projections to
match: the XDC port/pin set and the RTL top assignments. The public function
name is historical; the XDC no longer creates lane identity.

The returned target keeps the manifest package pin for every raw lane in
`target.package_pins`. Any missing, extra, or differently pinned pulse port in
the XDC fails immediately. A top-level RTL assignment to the wrong
`out_final[index]` or DAC bit also fails validation.

Editing JSON cannot alter a programmed FPGA. A hardware change needs an
approved rebuild and qualification. The layout fingerprint proves geometry
compatibility; it is not by itself a receipt for a particular board or
qualified bitstream.

## `board.xdc`

`board.xdc` is the Vivado package-pin/electrical projection. Its
`PACKAGE_PIN` declarations may be reordered without changing `ch00`, `ch01`,
… semantics. Vivado scripts may use `ZLC_PS_XDC` only for an approved build;
the selected XDC must still equal the explicit manifest.

The validator accepts explicit XDC/config paths for tests. The normal notebook
and `bin\run_server.bat` use the checked-in manifest and projections; a remote
client does not read them.

## Who reads this directory

- `zlc_pulse.manifest` generates the runtime target from `board.lanes` and
  validates the XDC and RTL top projections.
- `zlc_pulse.wire` reads the deployment geometry from the same JSON file.
- `zlc_pulse.remote` validates both before it accepts clients.
- Vivado project and programming scripts consume the XDC for an approved build.

Run `bin\estimate_resources.bat` for the capacity report. It is an estimator,
not permission to replace the frozen bitstream.
