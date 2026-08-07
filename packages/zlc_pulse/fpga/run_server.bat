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
set "FPGA_DIR=%~dp0"
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

set "PYTHONPATH=%REPO_ROOT%\src;%REPO_ROOT%;%PYTHONPATH%"
pushd "%REPO_ROOT%"
if /I "%~1"=="--help" (
  %ZLC_PY_CMD% -m zlc_pulse.remote %ZLC_FORWARD_ARGS%
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
  %ZLC_PY_CMD% -m zlc_pulse.remote --host "%ZLC_PS_HOST%" --port "%ZLC_PS_PORT%" %ZLC_FORWARD_ARGS%
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
echo Default backend policy: auto ^(UART word63 probe, then JTAG-to-AXI fallback^)
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

%ZLC_PY_CMD% -m zlc_pulse.remote --state-dir "%ZLC_PS_STATE_DIR%" --host "%ZLC_PS_HOST%" --port "%ZLC_PS_PORT%" %ZLC_FORWARD_ARGS%
set "ZLC_STATUS=!ERRORLEVEL!"
popd
endlocal & exit /b !ZLC_STATUS!

:zlc_help
echo Start the thin zlc_pulse JSON server on the FPGA/control machine.
echo.
echo Usage:
echo   fpga\run_server.bat
echo   fpga\run_server.bat --check-config
echo.
echo Environment:
echo   ZLC_FPGA_PYTHON=path\to\python.exe         (optional interpreter override)
echo   ZLC_PS_VIVADO_BIN=path\to\vivado(.bat)          (auto-discovered for jtag-axi)
echo   ZLC_PS_HOST=0.0.0.0
echo   ZLC_PS_PORT=18861
echo   ZLC_PS_STATE_DIR=path\to\state
echo.
echo CLI backend overrides:
echo   fpga\run_server.bat --backend jtag-axi
echo   fpga\run_server.bat --backend uart --uart-port COM6
echo   fpga\run_server.bat --backend memory
echo   fpga\run_server.bat --uart-baud 3000000
echo   fpga\run_server.bat --client-idle-timeout 300
echo   If other COM instruments are connected, pass --uart-port explicitly to skip port enumeration.
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
