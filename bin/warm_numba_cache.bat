@echo off
rem Warm the numba kernel cache so experiment runs never compile.
rem ALL of it: the warmer renders what production renders and then
rem checks that every kernel it can find has compiled, naming any it
rem could not reach.  It used to call the 3D module's own warmer,
rem which knew about that module's kernels and nothing about the nine
rem that draw every camera frame, histogram and uncertainty band.
rem The warmer is the product's "warm_numba" command, entered from this
rem checkout through the same dispatcher as every other launcher: no
rem product install, and no layer list of its own -- the bootstrap binds
rem the layers.  Where the cache lives has ONE owner, in
rem zlc_plot/_kernel_cache.py (numba_cache at the checkout root).  When it
rem holds machine code for the current toolchain and kernel source (a
rem fingerprint marker checks both), this exits in milliseconds.
setlocal EnableExtensions DisableDelayedExpansion

for %%I in ("%~dp0..") do set "ZLC_HOME=%%~fI"
call "%ZLC_HOME%\packages\zlc_pulse\fpga\_resolve_tools.bat" python "%ZLC_HOME%"
if errorlevel 1 exit /b 2

%ZLC_PY_CMD% -m zou_lab_control warm_numba
set "ZLC_RC=%ERRORLEVEL%"
if not "%ZLC_RC%"=="0" (
  echo.
  echo Warmup failed -- see the traceback above.  A missing numba is NOT
  echo a failure here: it is reported and the numpy reference engine runs.
  if "%ZLC_NO_PAUSE%"=="" pause
)
exit /b %ZLC_RC%
