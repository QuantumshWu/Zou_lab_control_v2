# FPGA Submodule

`fpga/` is the standalone hardware side of the ZLC pulse-streamer path. It can
be copied to the Vivado computer with the Python package and run without an
experiment configuration.

For the current frozen-bitstream operating and evidence policy, see
`docs/REAL_HARDWARE_BRINGUP_zh.md`.  Capacity invariants and implementation
notes live in `docs/MAINTAINER_NOTES.md`; the authoritative architecture is
`docs/SYSTEM_ARCHITECTURE_DESIGN_zh.md`.

## Layout

- `build_and_program.bat`: evidence-driven recovery tool for an established RTL
  defect or design mismatch; it is not part of normal experiment startup.
- `run_server.bat`: start the thin length-prefixed-JSON pulse server. It owns
  one `zlc_pulse.PulseStreamer` on the FPGA machine before accepting clients.
- `pulse_streamer/`: frozen HDL (`zlc_edge_streamer.v`,
  `zlc_pulse_streamer_top.v`), Vivado Tcl, and simulation testbenches. The
  Python image/capacity projection is `zlc_pulse.fpga`.
- Generated Vivado projects and server state default to `fpga\build`. The
  default project is `fpga\build\ps`; the short name `ps` (-> `ps.runs`) keeps
  Vivado's deep run/.Xil temp path under the Windows MAX_PATH limit while the
  build stays in-repo. The default server state dir is `fpga\build\state`.

The root-level `pulse_gui.bat` is the direct frontend entry point. The
sequencer-only `remote_pulse` installation and the complete `hardware`
installation (remote FPGA + qCMOS DCAM + Pylon MOT camera) both use this same
server contract; camera qualification remains on the client installation
machine. The FPGA batch files stay here so hardware setup is separate from the
GUI and camera SDKs.

## Runtime Chain

```text
Pulse GUI / Zou_lab_control.api
  -> RemotePulseStreamer.load/fire/wait_done/applied
  -> zlc_pulse.remote on the FPGA computer (uart or injected axi backend)
  -> zlc_pulse.PulseStreamer
  -> zlc_pulse_streamer_top.bit on the FPGA
```

The host packs the compiled program into a BRAM image and writes it over
JTAG-to-AXI (`axi_bram_ctrl`), then drives the CTRL register-file mailbox
(`COMMAND`/`STATUS` + the resident-bank `BANK_READY`/`BANK*_CHUNK` handshake).
The server uses the frozen package geometry and transport; it does not compile
against GUI channels or know measurement/run state. `applied()` is a passive
last-application echo for GUI sync, not trigger scheduling or point accounting.

## Normal Use

```powershell
.\fpga\run_server.bat --check-config    # print the deployed geometry and RPC policy
.\fpga\run_server.bat                    # start the thin server (host 0.0.0.0, port 18861)
```

The startup window distinguishes the listen bind from client addresses. With
the default `0.0.0.0:18861` bind, a notebook on the same FPGA computer uses
`127.0.0.1:18861`; a notebook on another computer uses one of the LAN
addresses printed by the server under `SERVER ADDRESS`. The Python process
also prints a copyable `RemotePulseStreamer(...)` example after the hardware
handshake. `0.0.0.0` means “listen on all interfaces” and must not be passed
as the client host.

The single client owner is released after five minutes without an RPC request
by an automatic SAFE; use `--client-idle-timeout SECONDS` to change that limit.

The default backend policy is `auto`: enumerate/probe UART at 3 Mbaud using the
word63 geometry handshake, select it only on a matching fingerprint, and then
fall back to JTAG-to-AXI with per-port reasons if no UART matches. Override it
explicitly on the command line, for example
`.\fpga\run_server.bat --backend uart --uart-port COM6` or
`.\fpga\run_server.bat --backend jtag-axi`. There is no
`ZLC_PS_SERVER_BACKEND` environment-variable switch. The JTAG backend keeps a
resident Vivado process (roughly 1–2 GB); prefer UART on memory-constrained
hosts.

Without `--uart-port`, every server start asks
`serial.tools.list_ports.comports()` for the ports currently present. Ports
with a USB VID/PID are probed first, while every remaining port stays in the
stable candidate order: this is sorting, not a whitelist. Thus COM3/COM4/COM5/
COM6 on one machine are runtime Windows assignments, not hardcoded choices;
COM1, COM12, or COM27 on another machine work without a code change.

Supplying `--uart-port COM6` selects only that port and skips enumeration; the
startup banner still reports the selected candidate. If other COM instruments
are connected, always supply this option so auto-probe does not open and query
those unrelated ports.

`build_and_program.bat` is not a normal-use command. It is retained only for a
separately approved, evidence-driven recovery after an actual RTL/deployment
defect has been established; that workflow must close routed setup/hold timing
and requalify the image before it may replace the frozen bitstream.

Default clock is 50 MHz (20 ns tick); the minimal pulse width and resolution are
1 tick. The qualified deployment has 4096 edge rows and two 2048-point scan
banks, so 4096 bank-local scan slots are resident at one time. A run may contain
more points: preparation preloads the first two chunks, then the sole host
observer refills each released bank through `BANK_READY` / `BANK*_CHUNK`. The
FPGA still clocks every point autonomously; the host moves chunks, never drives
individual point timing. Any observed `UNDERFLOW` invalidates the run. The
deployed clock is fixed at 50 MHz. `ZLC_PS_XDC` selects the approved
board-description input and `ZLC_PS_VIVADO_BIN` selects the Vivado executable;
neither changes the running bitstream.

## Path Rules

Vivado 2019 debug cores are path-length sensitive. Keep the checkout short
(`D:\ZLC`). The batch files print `ZLC build root` / `ZLC project dir`; those
printed paths are the source of truth for the generated
`impl_1\zlc_pulse_streamer_top.{bit,ltx}`. All FPGA launchers share the same
resolver: an explicit `ZLC_FPGA_PYTHON` override wins, then the repository
`.venv`, `.zlc_python_path`, and finally PATH; Vivado comes from
`ZLC_PS_VIVADO_BIN`, known installation roots, then PATH. Set
`ZLC_NO_PAUSE=1` for automation.
