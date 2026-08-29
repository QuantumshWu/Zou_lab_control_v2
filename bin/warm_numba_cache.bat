@echo off
rem Warm the numba kernel cache so experiment runs never compile.
rem ALL of it: the warmer renders what production renders and then
rem checks that every kernel it can find has compiled, naming any it
rem could not reach.  It used to call the 3D module's own warmer,
rem which knew about that module's kernels and nothing about the nine
rem that draw every camera frame, histogram and uncertainty band.
rem A cache-warming TOOL, not a product app: it needs numba and the
rem zlc_plot and zlc_data sources, so it runs from the checkout with
rem no product install.  Where the cache lives has ONE owner, in
rem zlc_plot/_kernel_cache.py (numba_cache at the checkout root) -- this
rem used to guess the same path a second time, so moving it needed both
rem edited or the warmer filled a directory nothing read.  When it holds
rem machine code for the current toolchain and kernel source (a
rem fingerprint marker checks both), this exits in milliseconds.
setlocal EnableExtensions DisableDelayedExpansion

for %%I in ("%~dp0..") do set "ZLC_HOME=%%~fI"
call "%ZLC_HOME%\packages\zlc_pulse\fpga\_resolve_tools.bat" python "%ZLC_HOME%"
if errorlevel 1 exit /b 2

set "PYTHONPATH=%ZLC_HOME%\packages\zlc_plot\src;%ZLC_HOME%\packages\zlc_data\src;%PYTHONPATH%"
%ZLC_PY_CMD% -c "from zlc_plot._kernel_warm import main; raise SystemExit(main())"
set "ZLC_RC=%ERRORLEVEL%"
if not "%ZLC_RC%"=="0" (
  echo.
  echo Warmup failed -- see the traceback above.  A missing numba is NOT
  echo a failure here: it is reported and the numpy reference engine runs.
  if "%ZLC_NO_PAUSE%"=="" pause
)
exit /b %ZLC_RC%
