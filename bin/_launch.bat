@echo off
rem Shared launcher body for the windows this product ships.
rem
rem   %1  the app module under zlc_workbench.apps
rem   %2+ arguments forwarded to it
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
%ZLC_PY_CMD% -m zlc_workbench.apps.%APP% !FORWARD!
set "ZLC_STATUS=!ERRORLEVEL!"
if not "!ZLC_STATUS!"=="0" (
  echo.
  echo %APP% exited with code !ZLC_STATUS!.
  if "%ZLC_NO_PAUSE%"=="" pause
)
endlocal & exit /b %ZLC_STATUS%
