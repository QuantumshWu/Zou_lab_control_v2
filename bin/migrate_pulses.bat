@echo off
rem Offline, one-shot migration. Double-click for workspace pulses/config_values;
rem drag files/folders here to restrict the run to those paths. --dry-run previews.
rem Expands the old 63-lane board; F13 cooling_pgc becomes shutter_420 on the same pin.
setlocal EnableExtensions DisableDelayedExpansion
for %%I in ("%~dp0..") do set "ZLC_HOME=%%~fI"
call "%ZLC_HOME%\packages\zlc_pulse\fpga\_resolve_tools.bat" python "%ZLC_HOME%"
if errorlevel 1 (
  echo Pulse migration cannot find Python.
  if "%ZLC_NO_PAUSE%"=="" pause
  exit /b 1
)
%ZLC_PY_CMD% -c "import zou_lab_control; from zlc_workbench.tools import migrate_pulses; print('ZLC root:', zou_lab_control.__file__); print('Migration:', migrate_pulses.__file__); raise SystemExit(migrate_pulses.main())" %*
set "ZLC_RC=%ERRORLEVEL%"
echo.
if "%ZLC_NO_PAUSE%"=="" pause
exit /b %ZLC_RC%
