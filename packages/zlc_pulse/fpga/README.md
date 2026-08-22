# FPGA Submodule

`fpga/` is the standalone hardware side of the ZLC pulse-streamer path. It can
be copied to the Vivado computer with the Python package and run without an
experiment configuration.

The root `ARCHITECTURE_DESIGN.md` is the authority. This page contains the
operating and hardware-acceptance boundary; no deleted package-local design
document is a second source of truth.

## Layout

- `..\..\..\bin\build_and_program.bat`: evidence-driven recovery tool for an established RTL
  defect or design mismatch; it is not part of normal experiment startup.
- `..\..\..\bin\run_server.bat`: start the thin length-prefixed-JSON pulse server. It owns
  one `zlc_pulse.PulseStreamer` on the FPGA machine before accepting clients.
- `pulse_streamer/`: frozen HDL (`zlc_edge_streamer.v`,
  `zlc_pulse_streamer_top.v`), Vivado Tcl, and simulation testbenches. The
  Python image/capacity projection is `zlc_pulse.fpga`.
- Generated Vivado projects and server state default to `fpga\build`. The
  default project is `fpga\build\ps`; the short name `ps` (-> `ps.runs`) keeps
  Vivado's deep run/.Xil temp path under the Windows MAX_PATH limit while the
  build stays in-repo. The default server state dir is `fpga\build\state`.

The root `bin\pulse_editor.bat` is the direct frontend entry point. The
sequencer-only `remote_pulse` installation and the complete `hardware`
installation (remote FPGA + qCMOS DCAM + Pylon MOT camera) both use this same
server contract; camera qualification remains on the client installation
machine. Human launchers stay together in root `bin\`; the hardware sources and
generated build tree remain owned here, separate from GUI and camera SDKs.

## Runtime Chain

```text
Pulse Editor / Workbench
  -> RemotePulseStreamer.load(..., rows=...)/fire(cycles=...)/wait_done/applied
  -> zlc_pulse.remote on the FPGA computer (uart or injected axi backend)
  -> zlc_pulse.PulseStreamer
  -> zlc_pulse_streamer_top.bit on the FPGA
```

The host packs the compiled program into a BRAM image and writes it over
JTAG-to-AXI (`axi_bram_ctrl`), then drives the CTRL register-file mailbox
(`COMMAND`/`STATUS` + the resident-bank `BANK_READY`/`BANK*_CHUNK` handshake).
The server uses the frozen package geometry and the explicit manifest lane
indices; XDC/RTL are validated projections, not ordering authorities. It does
not compile against GUI channels or know measurement/run state. `applied()` is
a passive echo of the loaded program, source, rows, and cycle count for GUI
sync, not trigger scheduling or point accounting.

## Normal Use

```powershell
.\bin\run_server.bat --check-config    # print the deployed geometry and RPC policy
.\bin\run_server.bat                    # start the thin server (host 0.0.0.0, port 18861)
```

The startup window distinguishes the listen bind from client addresses. With
the default `0.0.0.0:18861` bind, a notebook on the same FPGA computer uses
`127.0.0.1:18861`; a notebook on another computer uses one of the LAN
addresses printed by the server under `SERVER ADDRESS`. The Python process
also prints a copyable `RemotePulseStreamer(...)` example after the hardware
handshake. `0.0.0.0` means “listen on all interfaces” and must not be passed
as the client host.

There is no idle timeout for a healthy client. The first valid RPC owns the
board; a newer valid client takes over only after verified physical SAFE. A
real disconnect or server shutdown also drives SAFE.

The `auto` backend considers UART only when the one intended port is supplied;
it never enumerates or probes unrelated serial devices. For example, use
`.\bin\run_server.bat --backend uart --uart-port COM6`. Without `--uart-port`,
`auto` skips UART and uses JTAG-to-AXI. An explicit UART failure is loud and
does not fall back. `--backend jtag-axi` and offline `--backend memory` are
explicit alternatives. There is no `ZLC_PS_SERVER_BACKEND` environment switch.

`bin\build_and_program.bat` is not a normal-use command. It is retained only for a
separately approved, evidence-driven recovery after an actual RTL/deployment
defect has been established; that workflow must close routed setup/hold timing
and requalify the image before it may replace the frozen bitstream. Its default
action builds only; program/flash requires one explicit target.

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

## Hardware acceptance runbook (not executed by software milestones)

M5 validates RTL in simulation but does not program or flash a board. A human
hardware acceptance must use the following order and retain one receipt:

1. Record the repository commit, `streamer_config.json` path, `board.id`, FPGA
   part, 50 MHz constraint, and generated project directory.
2. Run the build-only action. Preserve timing status and the exact `.bit` and
   `.ltx` paths; do not program or flash as a side effect of building.
3. Diagnose the cable and require exactly one target, device, and JTAG AXI.
   Verify the hardware part against the config before an explicit program.
4. After programming, read word 63 and require the expected layout fingerprint,
   then open the server and record `describe()` geometry, clock, and lane count.
5. With instruments disconnected or otherwise made safe, verify reset/SAFE pin
   levels and clock gates, one finite pulse with delayed DAC tail, DONE only
   after physical drain, and loud sticky underflow/overflow/protocol errors.
6. Verify UART truncated/invalid frames, client disconnect, failed SAFE, and
   second-client takeover all leave no owner until a stable physical SAFE.
7. Flash only as a separate, explicitly requested action after all volatile
   programming checks pass.

The receipt records: operator and time; repository commit; config path and
`board.id`; FPGA part; Vivado version; build/project/bit/ltx paths; selected
target/device/transport; word-63 layout fingerprint; SAFE status and clock-word
readback; timing result; each check above as pass/fail with raw log path; and
whether the image was volatile-programmed, flashed, or not written. A manifest
ID or layout fingerprint alone is not a board/build receipt.
