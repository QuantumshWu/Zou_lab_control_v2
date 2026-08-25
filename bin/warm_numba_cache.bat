@echo off
rem Warm the numba kernel cache so experiment runs never compile.
rem A cache-warming TOOL, not a product app: it needs numba and the
rem zlc_plot sources alone, so it runs from the checkout with no
rem product install.  The cache lives in the repository .numba_cache;
rem when it already holds machine code for the current toolchain and
rem kernel source (a fingerprint marker checks both), this exits in
rem milliseconds.
setlocal EnableExtensions DisableDelayedExpansion

for %%I in ("%~dp0..") do set "ZLC_HOME=%%~fI"
if "%NUMBA_CACHE_DIR%"=="" set "NUMBA_CACHE_DIR=%ZLC_HOME%\.numba_cache"
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
