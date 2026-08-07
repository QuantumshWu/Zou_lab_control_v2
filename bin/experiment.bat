@echo off
rem Start an experiment: check the apparatus, then open the two windows you work in.
rem
rem   bin\experiment.bat
rem   bin\experiment.bat --pulse calibration
rem   bin\experiment.bat --devices          (stop after the apparatus editor)
rem
rem Run it from your experiment folder -- the one holding pulses\, data\ and
rem apparatus.json.  It is NOT passed from here on purpose: a double-clicked
rem launcher starts in bin\, which is nobody's experiment directory, and telling
rem a physicist their pulses are missing from bin\ is how that mistake reads.
rem
rem The order is the order the work happens in: the apparatus first, because a
rem console cannot open devices that are not written down; then the console,
rem which owns the session and the beat; then the pulse editor beside it.
setlocal EnableExtensions EnableDelayedExpansion

if not defined ZLC_PY_CMD set "ZLC_PY_CMD=python"

echo.
echo ============================================================
echo ZOU LAB CONTROL - experiment
echo Interpreter: %ZLC_PY_CMD%
echo Started in:  %CD%
echo ============================================================

set "STOP_AFTER_DEVICES="
set "FORWARD="
:zlc_collect
if "%~1"=="" goto zlc_devices
if /I "%~1"=="--devices" (
  set "STOP_AFTER_DEVICES=1"
) else (
  set "FORWARD=!FORWARD! %1"
)
shift /1
goto zlc_collect

:zlc_devices
echo.
echo [1/3] apparatus -- add the devices this bench has, Test devices, Save.
%ZLC_PY_CMD% -m zlc_workbench.apps.device_manager
set "ZLC_STATUS=!ERRORLEVEL!"
if not "!ZLC_STATUS!"=="0" goto zlc_failed
if defined STOP_AFTER_DEVICES goto zlc_done

echo.
echo [2/3] task console -- opens the session, holds the beat, owns the devices.
start "ZLC task console" %ZLC_PY_CMD% -m zlc_workbench.apps.task_console !FORWARD!

echo.
echo [3/3] pulse editor -- edit and fire the sequence beside it.
%ZLC_PY_CMD% -m zlc_workbench.apps.pulse_editor !FORWARD!
set "ZLC_STATUS=!ERRORLEVEL!"
if not "!ZLC_STATUS!"=="0" goto zlc_failed

:zlc_done
endlocal & exit /b 0

:zlc_failed
echo.
echo experiment stopped with code !ZLC_STATUS!.
if "%ZLC_NO_PAUSE%"=="" pause
endlocal & exit /b %ZLC_STATUS%
