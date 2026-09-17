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

Each physical field has one default value and at most one `PulseBinding`.
The binding stores independent Scan capability and a Default/API/Config source.
Scan may coexist with API or Config; API and Config are mutually exclusive.
The editor always permits editing the default. A Plan activates only its explicit
Scan fields; omitted Scan capabilities become constants using their selected base
source. An explicit scan point overrides the base source for that execution.

Scan and API are addressed by stable physical field identity internally and by
readable paths such as `MOT.duration` in authoring. They have no user-assigned
numbers or aliases. A Config binding instead references a named entry shared
across Pulses; multiple fields can reference the same name.

Config files contain only `format: zlc.pulse.config_values` and `values`:
each name maps to a numeric `value` and `unit`. Unassigned or missing entries
leave field defaults unchanged; supplied entries use the shared unit checks.
Runtime readers reject old numbered/source/export formats. The explicit offline
`bin/migrate_pulses.bat` tool converts known old Pulse/Config files with original
byte backups; it is not an automatic load-time compatibility path.

The Config tab edits that independent values file. Only explicit Save/Save as
writes it; unsaved rows never reach execution. Pulse Save stores the Pulse and
its field references, not a copy of the Config values. Device `load`/`fire`
reread the selected saved file and update the executable only when effective
values change. The authored source remains separate from the applied source;
an override never becomes the next default. Running output is not modified by
editing or saving. Local/Virtual/Remote share the same resolution owner.

`compile_pulse` remains pure. API resolution, Config resolution and selecting
an explicit Scan row use the common field writer. A field selected as an explicit
Hold/Step scan point cannot be overwritten again by Config at Fire.

`fire()` returns the already confirmed `AppliedState`: consumers do not need a
second `snapshot()`/`applied()` query. Unchanged execution counts and scan rows
are reused. A changed Config still goes through normal LOAD validation before
FIRE; the remote client and server must run the same protocol implementation.

The package has no measurement, GUI, or run-planning layer. `applied()` is only
the device's saved passive echo of the last program, source, rows, and repeat
counts; it is not trigger scheduling, expected-frame accounting, or
point-by-point reconciliation. `trigger_windows_by_channel()` and the other
schedule queries are pure finite host-side projections. They take finite
`run_repeats` and `scan_repeats` values (both at least one) and are not sent to
the device.

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
