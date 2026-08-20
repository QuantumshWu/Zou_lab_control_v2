@echo off
setlocal EnableExtensions EnableDelayedExpansion

if /I not "%~1"=="--inner" (
  set "ZLC_FORWARD_ARGS=%*"
  call "%~f0" --inner %*
  set "ZLC_STATUS=!ERRORLEVEL!"
  echo.
  if "!ZLC_STATUS!"=="0" (
    echo ZLC pulse server stopped normally.
  ) else (
    echo ZLC pulse server command failed with code !ZLC_STATUS!.
  )
  if "%ZLC_NO_PAUSE%"=="" pause
  exit /b !ZLC_STATUS!
)
shift /1

rem Thin separated-machine launcher.  The Python server owns one PulseStreamer;
rem this shell never builds or programs the frozen bitstream.
rem This launcher lives in bin\ -- one folder for everything a human clicks --
rem while what it drives is zlc_pulse's own fpga tree: its RTL, its board
rem config, its build.  The path is derived from the repository root rather
rem than from %~dp0, so moving the launcher does not move the board.
set "FPGA_DIR=%~dp0..\packages\zlc_pulse\fpga\"
for %%I in ("%~dp0..") do set "ZLC_HOME=%%~fI"
for %%I in ("%FPGA_DIR%..") do set "REPO_ROOT=%%~fI"

if /I "%~1"=="/?" goto zlc_help

call "%FPGA_DIR%_resolve_tools.bat" python "%REPO_ROOT%"
if errorlevel 1 exit /b 1

rem Restore launcher-side Vivado discovery for an explicit JTAG request.  Auto
rem fallback performs the same search in transport.axi before starting Vivado.
if /I "%~1"=="--backend" if /I "%~2"=="jtag-axi" (
  call "%FPGA_DIR%_resolve_tools.bat" vivado "%REPO_ROOT%" /optional
)

if not defined ZLC_PS_HOST set "ZLC_PS_HOST=0.0.0.0"
if not defined ZLC_PS_PORT set "ZLC_PS_PORT=18861"
if not defined ZLC_PS_STATE_DIR set "ZLC_PS_STATE_DIR=%FPGA_DIR%build\state"
if not defined ZLC_PS_CONFIG set "ZLC_PS_CONFIG=%REPO_ROOT%\fpga\board_config\streamer_config.json"

rem THIS checkout, through its one entry -- the same path a window takes.
rem Naming packages\zlc_pulse\src directly reached the package but not the
rem tree around it, so a layer it depends on could still come from a
rem standalone repository that happened to be installed.
set "PYTHONPATH=%ZLC_HOME%;%PYTHONPATH%"
pushd "%REPO_ROOT%"
if /I "%~1"=="--help" (
  %ZLC_PY_CMD% -m zou_lab_control_v2 pulse_server %ZLC_FORWARD_ARGS%
  set "ZLC_STATUS=!ERRORLEVEL!"
  call :zlc_help
  popd
  endlocal & exit /b !ZLC_STATUS!
)
if /I "%~1"=="--check-config" (
  if /I "%ZLC_PS_HOST%"=="0.0.0.0" (
    echo remote_server same-computer endpoint: 127.0.0.1:%ZLC_PS_PORT%
    echo remote_server other-computer endpoint: printed below by Python as SERVER ADDRESS
  ) else (
    echo remote_server endpoint: %ZLC_PS_HOST%:%ZLC_PS_PORT%
  )
  %ZLC_PY_CMD% -m zou_lab_control_v2 pulse_server --host "%ZLC_PS_HOST%" --port "%ZLC_PS_PORT%" %ZLC_FORWARD_ARGS%
  set "ZLC_STATUS=!ERRORLEVEL!"
  popd
  endlocal & exit /b !ZLC_STATUS!
)

echo.
echo ============================================================
echo ZLC PULSE SERVER
echo Host:                %ZLC_PS_HOST%:%ZLC_PS_PORT%
echo Port:                %ZLC_PS_PORT%
echo Server listen bind: %ZLC_PS_HOST%:%ZLC_PS_PORT%
echo Server port:        %ZLC_PS_PORT%
echo Default backend policy: auto ^(JTAG-to-AXI unless --uart-port names one port^)
echo Explicit backend override: --backend jtag-axi ^| uart ^| memory
if /I "%ZLC_PS_HOST%"=="0.0.0.0" (
  echo remote_server same-computer endpoint: 127.0.0.1:%ZLC_PS_PORT%
  echo remote_server other-computer endpoint: printed below by Python as SERVER ADDRESS
) else (
  echo remote_server address: %ZLC_PS_HOST%:%ZLC_PS_PORT%
)
echo NOTE: 0.0.0.0 is listen-only; use 127.0.0.1 locally or a Python LAN endpoint remotely.
echo Geometry: %REPO_ROOT%\fpga\board_config\streamer_config.json
echo Status:   CONNECTING to the deployed FPGA streamer...
echo ============================================================
echo The Python process will report HARDWARE CONNECTED and RPC LISTENING
echo only after the geometry handshake and SAFE readback succeed.
echo After that it prints exact SERVER ADDRESS and copyable CLIENT CONNECT EXAMPLE lines.

%ZLC_PY_CMD% -m zou_lab_control_v2 pulse_server --state-dir "%ZLC_PS_STATE_DIR%" --host "%ZLC_PS_HOST%" --port "%ZLC_PS_PORT%" %ZLC_FORWARD_ARGS%
set "ZLC_STATUS=!ERRORLEVEL!"
popd
endlocal & exit /b !ZLC_STATUS!

:zlc_help
echo Start the thin zlc_pulse JSON server on the FPGA/control machine.
echo.
echo Usage:
echo   bin\run_server.bat
echo   bin\run_server.bat --check-config
echo.
echo Environment:
echo   ZLC_FPGA_PYTHON=path\to\python.exe         (optional interpreter override)
echo   ZLC_PS_VIVADO_BIN=path\to\vivado(.bat)          (auto-discovered for jtag-axi)
echo   ZLC_PS_HOST=0.0.0.0
echo   ZLC_PS_PORT=18861
echo   ZLC_PS_STATE_DIR=path\to\state
echo.
echo CLI backend overrides:
echo   bin\run_server.bat --backend jtag-axi
echo   bin\run_server.bat --backend uart --uart-port COM6
echo   bin\run_server.bat --backend memory
echo   bin\run_server.bat --uart-port COM6 --uart-baud 3000000
echo   UART is considered only when --uart-port names exactly one configured port.
echo.
echo Client address rules:
echo   same computer: 127.0.0.1:18861
echo   another computer: use the LAN IP shown under SERVER ADDRESS
echo   0.0.0.0 is a listen bind address, not a client address
echo.
echo The server uses length-prefixed JSON, one client at a time, and a client-side
echo short-poll implementation of wait_done.  It never builds or programs hardware.
exit /b 0

:zlc_help_error
call :zlc_help
exit /b 2
