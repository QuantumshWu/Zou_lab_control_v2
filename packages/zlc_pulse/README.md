# zlc-pulse

`zlc-pulse` is the small host-side package for the frozen pulse-streamer
register image. It models a target and sequence, compiles one edge program,
packs that program for the existing bitstream, and exposes the two writable
slot-table paths:

```python
from zlc_pulse import PulseStreamer, compile_sequence, load_streamer_config

config = load_streamer_config()
program = compile_sequence(sequence, config["params"], config["clock_hz"])
streamer.load(program, source=sequence)
streamer.write_slots((12,))
streamer.fire()
report = streamer.wait_done(2.0)
state = streamer.applied()       # passive last-application echo for GUI sync
```

The package has no measurement, GUI, or run-planning layer. `applied()` is only
the device's saved passive echo of the last `load`/table/fire inputs; it is not
trigger scheduling, expected-frame accounting, or point-by-point reconciliation.
`trigger_times()` is a pure host-side projection and is not sent to the device.
`write_slots()` updates one row, while `write_scan_table()` preloads a multi-row
table and lets the observer refill the frozen ping-pong banks.

For a separated FPGA machine, `fpga\run_server.bat` starts the thin
length-prefixed-JSON façade. The launcher prints the listen bind and the local
client endpoint before starting Python. The Python process then prints
`HARDWARE CONNECTED` only after the geometry handshake and SAFE readback pass,
followed by `RPC LISTENING`, explicit `SERVER ADDRESS` lines, and copyable
`CLIENT CONNECT EXAMPLE` lines. With
the default bind `0.0.0.0:18861`, use `127.0.0.1:18861` from a notebook on the
same computer, or use one of the listed LAN IPs from another computer; never
pass `0.0.0.0` as the `RemotePulseStreamer` host. Client connect and disconnect
messages are also printed; disconnect releases SAFE but preserves the
server-process applied echo. Every server event is timestamped and includes
the client endpoint where relevant: `LOAD` shows a compact program summary,
`FIRE` shows `ONCE`/`FOREVER`, `DONE` marks completion, `STOP/SAFE` records the
stable readback, and a dropped connection is shown as `AUTO-SAFE`. All eleven
device RPC methods are logged with the client endpoint; `SNAPSHOT`, `CURSOR`,
`APPLIED`, and pending `WAIT DONE` entries use compact state summaries rather
than payload dumps. `RemotePulseStreamer`
has the same eleven device methods plus only `disconnect()` and context-manager
helpers (`__enter__`/`__exit__`); `wait_done()` is a client-side short poll,
so `safe()` can interrupt a forever fire over the same connection.

The server backend defaults to `auto`: it enumerates serial ports (or uses
`--uart-port` when supplied), probes each at 3 Mbaud through the existing
word63 geometry handshake, and selects UART only after a matching fingerprint.
If every probe fails, it records the per-port reason and falls back to
JTAG-to-AXI. Use `fpga\run_server.bat --backend jtag-axi|uart|memory` for an
explicit choice; an explicitly requested UART failure is an error and never
silently falls back. The JTAG path keeps a resident Vivado process (roughly
1–2 GB), so UART is preferred on memory-constrained hosts.
When another COM instrument is connected, pass `--uart-port COMx` explicitly;
omitting it intentionally enumerates and probes every listed serial port.

For an entirely offline walk-through, run the notebook cells before the real
hardware section in [`notebooks/usage.ipynb`](notebooks/usage.ipynb). They build
the target from XDC, model the sequence, compile it, and show the local
transport choices without opening hardware.

The notebook ends with an explicit real-hardware all-channel loop. Start
`fpga\run_server.bat` on the same machine first, then run the final cell: it
connects directly to `127.0.0.1:18861`, loads the program, and starts all 18 TTL
outputs alternating high/low every 1 µs while all four 10-bit DAC buses ramp
between signed codes −512 and +511. Stop it later with
`remote.safe(); remote.close()` in a new cell. An idle client is automatically
SAFE-released after five minutes by default; override it with
`--client-idle-timeout SECONDS` when needed.

The RTL and bitstream are external frozen artifacts. This repository does not
build or program them.
