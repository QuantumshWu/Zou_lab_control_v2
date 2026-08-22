@echo off
setlocal EnableExtensions DisableDelayedExpansion

rem ============================================================================
rem  Double-click this file to check whether the FPGA part configured in
rem  fpga\board_config\streamer_config.json has enough resources (LUT / FF / DSP
rem  / BRAM) for the configured pulse-streamer geometry.  Edit that JSON to change
rem  the part, edge count, delay depth, etc., then re-run this.  The window stays
rem  open with the report.
rem ============================================================================

if /I "%~1"=="--inner" goto zlc_inner
call "%~f0" --inner %*
set "ZLC_STATUS=%ERRORLEVEL%"
echo.
if "%ZLC_STATUS%"=="0" (
  echo ZLC resource estimate: the configured part HAS enough resources.
) else if "%ZLC_STATUS%"=="1" (
  echo ZLC resource estimate: INSUFFICIENT -- see the OVER BUDGET lines above.
) else (
  echo ZLC resource estimate failed with code %ZLC_STATUS% -- read the messages above.
)
echo You can close this window, or press any key to exit.
if "%ZLC_NO_PAUSE%"=="" pause
exit /b %ZLC_STATUS%

:zlc_inner
shift /1

for %%I in ("%~dp0..") do set "ZLC_HOME=%%~fI"
set "FPGA_DIR=%ZLC_HOME%\packages\zlc_pulse\fpga"

call "%FPGA_DIR%\_resolve_tools.bat" python "%ZLC_HOME%"
if errorlevel 1 exit /b 2

if "%ZLC_PS_CONFIG%"=="" set "ZLC_PS_CONFIG=%FPGA_DIR%\board_config\streamer_config.json"

echo Reading config: %ZLC_PS_CONFIG%
echo.
%ZLC_PY_CMD% -m zou_lab_control_v2 fpga --config "%ZLC_PS_CONFIG%"
set "ZLC_RC=%ERRORLEVEL%"
exit /b %ZLC_RC%
