@echo off
rem Shared installed-product launcher. The wrapper sets ZLC_COMMAND and this
rem file forwards the caller's original argument vector exactly once.
setlocal EnableExtensions DisableDelayedExpansion

if not defined ZLC_COMMAND (
  echo ZLC launcher error: wrapper did not select a product command.
  exit /b 2
)

for %%I in ("%~dp0..") do set "ZLC_HOME=%%~fI"
call "%ZLC_HOME%\packages\zlc_pulse\fpga\_resolve_tools.bat" python "%ZLC_HOME%"
if errorlevel 1 (
  echo.
  echo %ZLC_COMMAND% cannot start: this machine has no Python this launcher can run.
  if "%ZLC_NO_PAUSE%"=="" pause
  exit /b 1
)

echo.
echo ============================================================
echo ZLC - %ZLC_COMMAND%
echo Interpreter: %ZLC_PY_CMD%
echo Started in:  %CD%
echo ============================================================
%ZLC_PY_CMD% -m zou_lab_control_v2 %ZLC_COMMAND% %*
set "ZLC_STATUS=%ERRORLEVEL%"
if not "%ZLC_STATUS%"=="0" (
  echo.
  echo %ZLC_COMMAND% exited with code %ZLC_STATUS%.
  if "%ZLC_NO_PAUSE%"=="" pause
)
exit /b %ZLC_STATUS%
