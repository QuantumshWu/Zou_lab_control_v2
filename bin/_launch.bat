@echo off
rem Shared launcher body for the windows this product ships.
rem
rem   %1  the app module under zlc_workbench.apps
rem   %2+ arguments forwarded to it
rem
rem Run through zou_lab_control_v2, never straight at the app: importing that
rem package first is what makes THIS checkout the code that runs.  Eight of the
rem layers are also installed as editable packages pointing elsewhere, and a
rem launcher that skipped this would open a window from those -- the failure
rem with no symptom, where the change you just made simply does not appear.
rem
rem The workspace -- pulses\, data\, apparatus.json -- is NOT forced from here.
rem A double-clicked launcher starts in the folder holding the launcher, which is
rem never anyone's experiment directory; passing that as --workspace is how this
rem told a physicist their pulses were missing from bin\.  The app searches for
rem one at and above the working directory, and prints what it found.
setlocal EnableExtensions EnableDelayedExpansion

set "APP=%~1"
shift /1

if not defined ZLC_PY_CMD set "ZLC_PY_CMD=python"

rem The checkout itself, so "python -m zou_lab_control_v2" resolves no matter
rem which folder the launcher was started from -- and it is started from the
rem experiment folder on purpose.  Prepended, so this tree wins over anything
rem installed under the same names.
set "ZLC_HOME=%~dp0.."
if defined PYTHONPATH (
  set "PYTHONPATH=%ZLC_HOME%;%PYTHONPATH%"
) else (
  set "PYTHONPATH=%ZLC_HOME%"
)

set "FORWARD="
:zlc_collect
if "%~1"=="" goto zlc_run
set "FORWARD=!FORWARD! %1"
shift /1
goto zlc_collect

:zlc_run
echo.
echo ============================================================
echo ZLC WORKBENCH - %APP%
echo Interpreter: %ZLC_PY_CMD%
echo Started in:  %CD%
echo ============================================================
%ZLC_PY_CMD% -m zou_lab_control_v2 %APP% !FORWARD!
set "ZLC_STATUS=!ERRORLEVEL!"
if not "!ZLC_STATUS!"=="0" (
  echo.
  echo %APP% exited with code !ZLC_STATUS!.
  if "%ZLC_NO_PAUSE%"=="" pause
)
endlocal & exit /b %ZLC_STATUS%
