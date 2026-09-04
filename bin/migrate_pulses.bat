@echo off
rem Bring this machine's pulse files up to the current document schema.
rem
rem A ONE-SHOT.  A pulse outlives the build that wrote it, and the reader is
rem a strict whitelist, so a renamed field turns every pulse already on disk
rem into "cannot open <name>.json: unknown pulse field(s): ...".  The product
rem carries no readers for old shapes on purpose; this is the other half of
rem that choice.  Run it once here, then it and its module get deleted.
rem
rem Double-click it to migrate the workspace the Pulse Editor opens, or drag
rem pulse files or a folder onto it to migrate those instead.  Originals are
rem kept beside each file as <name>.json.pre-migration, and nothing is written
rem until the migrated pulse has been read back by the editor's own reader.
rem
rem A TOOL, not a product command: it is imported from the checkout this
rem launcher sits in, which _resolve_tools.bat puts on PYTHONPATH, so it runs
rem on a bench machine without reinstalling anything first.
setlocal EnableExtensions DisableDelayedExpansion

for %%I in ("%~dp0..") do set "ZLC_HOME=%%~fI"
call "%ZLC_HOME%\packages\zlc_pulse\fpga\_resolve_tools.bat" python "%ZLC_HOME%"
if errorlevel 1 (
  echo.
  echo Pulse migration cannot start: this machine has no Python this launcher can run.
  if "%ZLC_NO_PAUSE%"=="" pause
  exit /b 1
)

%ZLC_PY_CMD% -c "from zlc_workbench.tools.migrate_pulses import main; raise SystemExit(main())" %*
set "ZLC_RC=%ERRORLEVEL%"
echo.
rem Always pause: this window IS the report, and it is double-clicked.
if "%ZLC_NO_PAUSE%"=="" pause
exit /b %ZLC_RC%
