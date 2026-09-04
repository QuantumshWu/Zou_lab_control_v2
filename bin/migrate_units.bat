@echo off
rem Rewrite this machine's saved pulses in one unit spelling.
rem
rem A ONE-SHOT.  A unit used to have several names -- us, the micro sign, and
rem whichever of them the layer that wrote the file happened to prefer, which
rem was not the same layer twice.  There is one registry now and one symbol
rem per unit, so a saved pulse holds that symbol.
rem
rem Nothing here is required to OPEN an old pulse: "us" is still an accepted
rem spelling and always will be, because it is what a keyboard has.  This is
rem so that what is on disk matches what the editor would save, and a Save
rem that changes nothing shows no diff.
rem
rem Double-click it to rewrite the workspace the Pulse Editor opens, or drag
rem pulse files or a folder onto it to do those instead.  Originals are kept
rem beside each file as <name>.json.pre-units, and a file that was already
rem current keeps no backup at all.
rem
rem A TOOL, not a product command: it is imported from the checkout this
rem launcher sits in, which _resolve_tools.bat puts on PYTHONPATH, so it runs
rem on a bench machine without reinstalling anything first.
setlocal EnableExtensions DisableDelayedExpansion

for %%I in ("%~dp0..") do set "ZLC_HOME=%%~fI"
call "%ZLC_HOME%\packages\zlc_pulse\fpga\_resolve_tools.bat" python "%ZLC_HOME%"
if errorlevel 1 (
  echo.
  echo Unit migration cannot start: this machine has no Python this launcher can run.
  if "%ZLC_NO_PAUSE%"=="" pause
  exit /b 1
)

%ZLC_PY_CMD% -c "from zlc_workbench.tools.migrate_units import main; raise SystemExit(main())" %*
set "ZLC_RC=%ERRORLEVEL%"
echo.
rem Always pause: this window IS the report, and it is double-clicked.
if "%ZLC_NO_PAUSE%"=="" pause
exit /b %ZLC_RC%
