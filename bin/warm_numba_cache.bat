@echo off
rem Warm the numba kernel cache so experiment runs never compile.
rem A cache-warming TOOL, not a product app: it needs numba and the
rem zlc_plot sources alone, so it runs from the checkout with no
rem product install.  Where the cache lives has ONE owner, in
rem zlc_plot/_kernel_cache.py (numba_cache at the checkout root) -- this
rem used to guess the same path a second time, so moving it needed both
rem edited or the warmer filled a directory nothing read.  When it holds
rem machine code for the current toolchain and kernel source (a
rem fingerprint marker checks both), this exits in milliseconds.
setlocal EnableExtensions DisableDelayedExpansion

for %%I in ("%~dp0..") do set "ZLC_HOME=%%~fI"
call "%ZLC_HOME%\packages\zlc_pulse\fpga\_resolve_tools.bat" python "%ZLC_HOME%"
if errorlevel 1 exit /b 2

set "PYTHONPATH=%ZLC_HOME%\packages\zlc_plot\src;%PYTHONPATH%"
%ZLC_PY_CMD% -c "from zlc_plot._height3d_scanline import main; raise SystemExit(main())"
set "ZLC_RC=%ERRORLEVEL%"
if not "%ZLC_RC%"=="0" (
  echo.
  echo Warmup failed -- it needs numba and numpy on this interpreter
  echo ^(pip install -c constraints.txt numba^).
  if "%ZLC_NO_PAUSE%"=="" pause
)
exit /b %ZLC_RC%
