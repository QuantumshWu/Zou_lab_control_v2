@echo off
rem Install one editable root distribution under the single constraints file.
setlocal EnableExtensions DisableDelayedExpansion

for %%I in ("%~dp0..") do set "ZLC_HOME=%%~fI"
call "%ZLC_HOME%\packages\zlc_pulse\fpga\_resolve_tools.bat" python "%ZLC_HOME%"
if errorlevel 1 (
  echo Nothing can be installed until this machine has Python.
  if "%ZLC_NO_PAUSE%"=="" pause
  exit /b 1
)
if not exist "%ZLC_HOME%\constraints.txt" (
  echo Missing product constraints: %ZLC_HOME%\constraints.txt
  if "%ZLC_NO_PAUSE%"=="" pause
  exit /b 2
)

echo.
echo ============================================================
echo ZOU LAB CONTROL - install one product
echo Interpreter: %ZLC_PY_CMD%
echo Product:     %ZLC_HOME%
echo ============================================================
%ZLC_PY_CMD% -m pip install --constraint "%ZLC_HOME%\constraints.txt" --editable "%ZLC_HOME%[notebook]"
set "ZLC_STATUS=%ERRORLEVEL%"
if not "%ZLC_STATUS%"=="0" goto zlc_failed
%ZLC_PY_CMD% -m pip check
set "ZLC_STATUS=%ERRORLEVEL%"
if not "%ZLC_STATUS%"=="0" goto zlc_failed
%ZLC_PY_CMD% -m zou_lab_control_v2 check
set "ZLC_STATUS=%ERRORLEVEL%"
if not "%ZLC_STATUS%"=="0" goto zlc_failed
echo.
echo Installed. Run bin\experiment.bat from the experiment workspace.
if "%ZLC_NO_PAUSE%"=="" pause
exit /b 0

:zlc_failed
echo.
echo Product installation failed with code %ZLC_STATUS%.
if "%ZLC_NO_PAUSE%"=="" pause
exit /b %ZLC_STATUS%
