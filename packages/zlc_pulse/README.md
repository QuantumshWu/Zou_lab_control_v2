# zlc-pulse

`zlc-pulse` is the small host-side package for the frozen pulse-streamer
register image. It models a target and sequence, compiles one edge program,
loads one complete application, and executes an explicit number of cycles:

```python
from zlc_pulse import PulseStreamer, compile_sequence, load_streamer_config

config = load_streamer_config()
program = compile_sequence(sequence, config["params"], config["clock_hz"])
streamer.load(program, source=sequence, rows=((12,),))
streamer.fire(cycles=1)
report = streamer.wait_done(2.0)
state = streamer.applied()       # passive last-application echo for GUI sync
```

`rows` is the complete value table for a slotted program. A slotted program
requires one or more integer rows; an unslotted program omits `rows`. There is
no later slot-write or scan-table-write phase. `fire(cycles=N)` executes `N`
cycles, and `fire(cycles=None)` is the explicit forever form. `RepeatRegion`
means only a loop inside one compiled timeline; it does not express cycles,
scan sweeps, or Dataset repeats.

The package has no measurement, GUI, or run-planning layer. `applied()` is only
the device's saved passive echo of the last program, source, rows, and cycle
count; it is not trigger scheduling, expected-frame accounting, or
point-by-point reconciliation. `trigger_times()` is a pure host-side projection
and is not sent to the device.

For a separated FPGA machine, `bin\run_server.bat` starts the thin
length-prefixed-JSON facade. The server process is the only hardware-transport
owner. The first valid RPC claims the board; a newer valid client takes over
only after the old physical state reaches verified SAFE. A real disconnect or
server shutdown also drives SAFE. There is no normal-connection idle timeout,
authentication, or TLS in this trusted-lab protocol.

The launcher distinguishes the listen bind from client addresses. With the
default bind `0.0.0.0:18861`, use `127.0.0.1:18861` from the same computer or a
printed LAN address from another computer; never use `0.0.0.0` as a client
host. `RemotePulseStreamer` mirrors the local device surface and adds only its
TCP connection lifecycle. `wait_done()` uses short client-side polls, so SAFE
can interrupt `fire(cycles=None)` over the same connection.

The default `auto` policy enumerates COM ports, tries USB VID/PID descriptors
first, and accepts a UART only after the deployed word-63 geometry fingerprint
matches. Every failed probe is closed before the next port. To restrict the
probe to one known port, use:

```powershell
.\bin\run_server.bat --backend uart --uart-port COM6
```

If no enumerated UART matches, `auto` falls back to JTAG-to-AXI.
`--backend jtag-axi` and the offline `--backend memory` mode are explicit
alternatives. An explicitly requested UART failure is an error and never
silently falls back.

The one product notebook uses only the virtual sequencer and contains no
hardware section. Pulse model, transport and FPGA acceptance details live in
this README and `fpga/README.md`; continuous `cycles=None` hardware use remains
an explicit operator workflow whose runbook requires `try/finally`, verified
SAFE, and close.

This repository tracks the RTL, board description, Vivado Tcl, and simulations
under `packages\zlc_pulse\fpga\`. `bin\build_and_program.bat` is the explicit
build/recovery entry; its default action builds only. Programming or flashing
requires a separately approved explicit target. Generated
Vivado products, the deployed `.bit`/`.ltx`, and the FPGA's volatile or flash
programmed state are external machine artifacts, not Python package data.
Normal experiment startup uses the already deployed bitstream and never builds
or programs hardware.
