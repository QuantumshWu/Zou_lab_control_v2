# fpga/board_config — board / platform configuration

This directory contains the board description and the frozen deployment
geometry. The checked-in `board.xdc` is the single runtime mapping source: the
Python package does not contain a second table of board signal names.

## `board.xdc`

`zlc_pulse.pulse_target_from_xdc()` reads the XDC and returns the complete
`PulseTarget` used by the notebook and server-side validation. The parser:

- preserves the order of `PACKAGE_PIN` declarations as `ch00`, `ch01`, …;
- treats a bare output as one digital TTL port;
- groups every contiguous `name[index]` family as one DAC port, ordered by
  numeric bit index;
- pairs each DAC family with its numbered latch-clock output; and
- excludes clock, UART, reset/status, LED, ground, and other control-only
  outputs from the pulse lanes.

The returned target keeps the physical package pin for every raw lane in
`target.package_pins`. This is diagnostic metadata; the pulse ABI remains the
ordered raw-lane and logical-port description.

Loading also validates the XDC against `streamer_config.json`. A mismatch in
lane count, DAC bus count, or DAC width raises immediately and reports the XDC
value and the configuration value. Therefore the notebook, server, and
diagnostic tools all consume the same board mapping and cannot silently use
different geometry.

The parser accepts an explicit `path=` for tests or a separately managed board
description. The normal notebook and `fpga/run_server.bat` use the checked-in
`fpga/board_config/board.xdc`; a client machine never needs a local XDC.

The XDC is also consumed by the Vivado project scripts. Those scripts may use
`ZLC_PS_XDC` when an approved build is intentionally directed at another board
file, but that build-time selection does not create a client-side mapping or a
runtime environment-variable switch.

## `streamer_config.json`

This file describes the approved frozen deployment geometry:

| field | meaning |
|---|---|
| `fpga_part` | Vivado part used by the build and capacity estimate |
| `clock_hz` | sequencer clock; the shipped deployment is 50 MHz |
| `target_pct` | resource-budget target for the estimate |
| `params.channel_count` | number of raw pulse lanes |
| `params.bus_count` / `bus_width` | DAC bus geometry |
| remaining `params.*` fields | edge, scan, coefficient, delay, and FIFO limits |

The host reads this file to validate the deployed geometry and to estimate
capacity. Editing JSON cannot alter a programmed FPGA; a hardware change needs
an approved rebuild and qualification.

Run `estimate_resources.bat` from the repository root for the capacity report.
It is an estimator, not permission to replace the frozen bitstream.

## Who reads this directory

- `zlc_pulse.manifest` derives the runtime target from `board.xdc`.
- `zlc_pulse.wire` reads the deployment geometry.
- `zlc_pulse.remote` validates both before it accepts clients.
- Vivado project and programming scripts consume the XDC for an approved build.

The remote server publishes the usable client address after the hardware
handshake. A separated client uses that server-published address and port; it
does not read this directory.
