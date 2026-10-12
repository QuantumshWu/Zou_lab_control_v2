@echo off
rem One-click SLM rearrangement benchmark on this machine's GPU (the lab's T400):
rem this checkout's solver, and commit de73d721 for the sequence comparison, on
rem run 14 of 2026-10-10 and the central 3x3 target (bench\slm_t400\run.py).
rem Close the SLM task console first. It takes about 10-20 minutes and writes
rem bench\results\slm_t400_<host>_<time>\ and a .zip of it: send the .zip.
setlocal EnableExtensions DisableDelayedExpansion

for %%I in ("%~dp0..") do set "ZLC_HOME=%%~fI"
call "%ZLC_HOME%\packages\zlc_pulse\fpga\_resolve_tools.bat" python "%ZLC_HOME%"
if errorlevel 1 (
  echo.
  echo SLM benchmark cannot start: this machine has no Python this launcher can run.
  if "%ZLC_NO_PAUSE%"=="" pause
  exit /b 2
)

echo.
echo ============================================================
echo SLM rearrangement benchmark - close the SLM task console first
echo Interpreter: %ZLC_PY_CMD%
echo ============================================================
%ZLC_PY_CMD% "%ZLC_HOME%\bench\slm_t400\run.py" %*
set "ZLC_RC=%ERRORLEVEL%"
if not "%ZLC_RC%"=="0" (
  echo.
  echo The benchmark exited with code %ZLC_RC% -- see the output above.
)
if "%ZLC_NO_PAUSE%"=="" pause
exit /b %ZLC_RC%
