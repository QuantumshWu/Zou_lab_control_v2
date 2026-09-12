# zlc-pulse

`zlc-pulse` is the small host-side package for the pulse-streamer register
image. It models a target and sequence, compiles one edge program, loads one
complete application, and executes the three explicit repeat layers:

```python
from zlc_pulse import PulseStreamer, compile_sequence, load_streamer_config

config = load_streamer_config()
program = compile_sequence(sequence, config["params"], config["clock_hz"])
streamer.load(program, source=sequence, rows=((12,),))
streamer.fire(run_repeats=1, scan_repeats=1)
report = streamer.wait_done(2.0)
state = streamer.applied()       # passive last-application echo for GUI sync
```

`rows` is the complete value table for a slotted program. A slotted program
requires one or more integer rows; an unslotted program omits `rows`. There is
no later slot-write or scan-table-write phase. The execution order is
`scan_repeats -> scan row -> run_repeats -> Pulse timeline -> PulseBracket`.
Each row remains current for `run_repeats` complete Pulse runs, then the cursor
advances; after the last row, `scan_repeats` controls complete table sweeps.
`0` means infinite for either hardware repeat count. With no scan table,
`scan_repeats` is exactly `1`.

`PulseBracket` is the optional single continuous interval inside the timeline.
Its count is at least two, and it compiles only to the program's `LOOP_*`
metadata. Even a bracket spanning the whole Pulse does not become or alter
`run_repeats`. The sequence's authored `run_repeats` defaults to `0`; a task
may explicitly override it for one execution without changing the saved Pulse.

A pulse field's value comes from one of three places, and which one is the
whole meaning of its binding. A SCAN slot is filled by the board, one value per
point, out of the hardware's four. An API parameter is a hole a caller fills
once per run; `compile_sequence` refuses one that is still open. A CONFIG
parameter is neither: its default IS the field's own number. Config files use
the displayed Config numbers `1`, `2`, ... in declaration/click order, never
local period/slot/parameter IDs. Scan and API bindings do not affect that order.
Unmatched numbers keep the pulse's defaults; matched values and units are validated.
`compile_pulse` is pure authoring compilation. Device `load`/`fire` share the
override rule, retain the authored source separately from the actual source,
and record the program actually loaded. Load also reads the bound file before
preparing its upload, avoiding an obsolete upload immediately before Fire.
Deleting an override restores its
authored default, not a value left behind by an earlier override.

Pulse Editor's **Save config** exports the current document's Config fields
through `authored_config_entries`, including their declared units and an empty
set when none are declared. It works offline and neither runs a pulse nor
modifies the sequencer. **Load config** binds the file with `load_config_file`.
Every device `fire()` rereads that file before execution; unchanged effective
values require neither recompilation nor a new hardware load. Changed values
are recompiled/reloaded before firing, preserving scan rows, tick scales and
the requested repeats. Invalid or missing bound files refuse that Fire, not
silently execute old values. There is no background watcher or preview file I/O.
`load_config_values(entries)` remains the explicit in-memory API and detaches
any file binding; its `source` label is not guessed to be a path. Config files
with old field-name keys must be re-saved as numbered files, not guessed across
pulses. Local/Virtual/Remote devices share this behavior; a Remote client owns
its own file binding, not a server-wide path.
A field a run needs to vary is an API parameter, which is the
whole difference between the two, so a field carries at most one binding and
all three share one id namespace.

The package has no measurement, GUI, or run-planning layer. `applied()` is only
the device's saved passive echo of the last program, source, rows, and repeat
counts; it is not trigger scheduling, expected-frame accounting, or
point-by-point reconciliation. `trigger_times()` and the other schedule queries
are pure finite host-side projections. They take finite `run_repeats` and
`scan_repeats` values (both at least one) and are not sent to the device.

Pulse documents use the stable strict root `zlc.pulse` with no numeric format
version. Their sequence root contains `bracket` and `run_repeats`; the removed
`repeat` field is not accepted. The codec accepts only the current complete
grammar, so unsupported workspace files are refused.

For a separated FPGA machine, the bench serves the board in-process (the
`sequencer.local` device type), or the headless `pulse_server` command
starts the same thin length-prefixed-JSON facade. The server process is the only hardware-transport
owner. The first valid control RPC claims the board; a newer valid client takes over
only after the old physical state reaches verified SAFE. A real disconnect or
server shutdown also drives SAFE. There is no normal-connection idle timeout,
authentication, or TLS in this trusted-lab protocol.

The launcher distinguishes the listen bind from client addresses. With the
default bind `0.0.0.0:18861`, use `127.0.0.1:18861` from the same computer or a
printed LAN address from another computer; never use `0.0.0.0` as a client
host. `RemotePulseStreamer` mirrors the local device surface and adds only its
TCP connection lifecycle. `wait_done()` waits on a separate, non-owning connection
bound to the current owner's token and FIRE command ID. The server waits on the
run's completion event and replies immediately when it completes; status queries
and SAFE retain the control connection. A timeout does not consume the result,
and Stop, disconnect or a replacement run cannot hand an old waiter a new result.
Update and restart both client and server for this completion protocol; no FPGA
rebuild is needed. The former client-side `poll_interval` option is removed.

The default `auto` policy enumerates COM ports, tries USB VID/PID descriptors
first, and accepts a UART only after the deployed word-63 geometry fingerprint
matches. Every failed probe is closed before the next port. To restrict the
probe to one known port, use:

```powershell
pulse_server --backend uart --uart-port COM6
```

If no enumerated UART matches, `auto` falls back to JTAG-to-AXI.
`--backend jtag-axi` and the offline `--backend memory` mode are explicit
alternatives. An explicitly requested UART failure is an error and never
silently falls back.

The one product notebook uses only the virtual sequencer and contains no
hardware section. Pulse model, transport and FPGA acceptance details live in
this README and `fpga/README.md`; continuous hardware use (`run_repeats=0`, or
`scan_repeats=0` with a scan table) remains an explicit operator workflow whose
runbook requires `try/finally`, verified SAFE, and close.

This repository tracks the RTL, board description, Vivado Tcl, and simulations
under `packages\zlc_pulse\fpga\`. `bin\build_and_program.bat` is the explicit
build/recovery entry; its default action builds only. Programming or flashing
requires a separately approved explicit target. Generated
Vivado products, the deployed `.bit`/`.ltx`, and the FPGA's volatile or flash
programmed state are external machine artifacts, not Python package data.
Normal experiment startup uses the already deployed bitstream and never builds
or programs hardware.
