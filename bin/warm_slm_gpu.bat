@echo off
rem Warm the SLM rearrangement GPU code on this machine, like warm_numba_cache:
rem compile the CUDA module (CuPy keeps the binary) and measure, for every band
rem count, whether cuBLAS or a warp kernel computes the analysis product faster
rem on this GPU (tensor cores make cuBLAS faster; the lab's T400 has none). The
rem choices go to slm_gpu_cache\ at the checkout root. Without this, the first
rem preparation on the machine measures what it needs and is slower once.
rem Run it again after a driver or GPU change; close the SLM task console first.
setlocal EnableExtensions DisableDelayedExpansion

for %%I in ("%~dp0..") do set "ZLC_HOME=%%~fI"
call "%ZLC_HOME%\packages\zlc_pulse\fpga\_resolve_tools.bat" python "%ZLC_HOME%"
if errorlevel 1 exit /b 2

%ZLC_PY_CMD% -m zou_lab_control warm_slm_gpu
set "ZLC_RC=%ERRORLEVEL%"
if not "%ZLC_RC%"=="0" (
  echo.
  echo SLM GPU warmup failed -- see the message above.
)
if "%ZLC_NO_PAUSE%"=="" pause
exit /b %ZLC_RC%
