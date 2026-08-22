@echo off
rem Start the installed pulse server. This wrapper never builds, programs or
rem flashes the FPGA. The Python command keeps the deployed auto policy:
rem enumerate UART candidates, require the word-63 fingerprint, then use JTAG.
setlocal EnableExtensions DisableDelayedExpansion

for %%I in ("%~dp0..") do set "ZLC_HOME=%%~fI"
set "FPGA_DIR=%ZLC_HOME%\packages\zlc_pulse\fpga"
if not defined ZLC_PS_HOST set "ZLC_PS_HOST=0.0.0.0"
if not defined ZLC_PS_PORT set "ZLC_PS_PORT=18861"
if not defined ZLC_PS_STATE_DIR set "ZLC_PS_STATE_DIR=%FPGA_DIR%\build\state"

rem An explicit JTAG request may need Vivado added to the environment. Auto
rem fallback performs the same discovery in the Python transport owner.
if /I "%~1"=="--backend" if /I "%~2"=="jtag-axi" (
  call "%FPGA_DIR%\_resolve_tools.bat" vivado "%ZLC_HOME%" /optional
)

set "ZLC_COMMAND=pulse_server"
call "%~dp0_launch.bat" --state-dir "%ZLC_PS_STATE_DIR%" --host "%ZLC_PS_HOST%" --port "%ZLC_PS_PORT%" %*
exit /b %ERRORLEVEL%
