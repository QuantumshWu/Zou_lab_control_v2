@echo off
rem Bring this machine up to date: pull, then check it still runs.
rem
rem   bin\update.bat
rem
rem Nothing is "installed" by this and nothing needs to be.  The code is
rem reached by PATH -- every launcher here puts this checkout on it -- so a
rem pull IS the update.  What this adds is the two things worth doing after
rem one: making sure the dependency list did not grow, and proving the windows
rem still open before you find out during a run.
setlocal EnableExtensions EnableDelayedExpansion

if not defined ZLC_PY_CMD set "ZLC_PY_CMD=python"
set "ZLC_HOME=%~dp0.."
pushd "%ZLC_HOME%"

echo.
echo ============================================================
echo ZOU LAB CONTROL - update
echo Checkout:    %CD%
echo Interpreter: %ZLC_PY_CMD%
echo ============================================================

echo.
echo [1/3] pulling...
git rev-parse --is-inside-work-tree >nul 2>&1
if errorlevel 1 (
  echo   this folder is not a git checkout; skipping the pull.
) else (
  rem --ff-only on purpose: a merge on the experiment machine is somebody
  rem editing here, and that wants a person looking at it, not a launcher.
  git pull --ff-only
  if errorlevel 1 (
    echo.
    echo   The pull did not fast-forward.  Something was changed on this
    echo   machine, or the branch has diverged.  Sort that out first --
    echo   "git status" says what is different.
    popd
    if "%ZLC_NO_PAUSE%"=="" pause
    endlocal & exit /b 1
  )
)

echo.
echo [2/3] dependencies...
call "%~dp0install_requirements.bat"
if errorlevel 1 (
  popd
  if "%ZLC_NO_PAUSE%"=="" pause
  endlocal & exit /b 1
)

echo.
echo [3/3] checking the windows still open...
%ZLC_PY_CMD% -c "import zou_lab_control_v2, zlc_workbench, inspect; print('   code in use:', inspect.getfile(zlc_workbench))"
if errorlevel 1 goto zlc_broken
%ZLC_PY_CMD% -m zou_lab_control_v2 check
if errorlevel 1 goto zlc_broken

echo.
echo up to date.  Run bin\experiment.bat from your experiment folder.
popd
if "%ZLC_NO_PAUSE%"=="" pause
endlocal & exit /b 0

:zlc_broken
echo.
echo The checkout pulled but does not import cleanly.  Nothing was installed,
echo so the previous commit still works: "git log --oneline -5" and
echo "git checkout <previous>" puts the machine back while this is looked at.
popd
if "%ZLC_NO_PAUSE%"=="" pause
endlocal & exit /b 1
