@echo off
rem Install what this product needs from PyPI -- and only that.
rem
rem   bin\install_requirements.bat
rem
rem It does NOT install this checkout as a package, on purpose.  The code is
rem reached by PATH: a launcher, a notebook or a script imports
rem zou_lab_control_v2 first, and that puts THIS tree on sys.path.  Installing
rem it as well would create a second copy under the same names, which is the
rem failure this project has already paid for -- a change that appears to do
rem nothing because the copy being imported is somewhere else.
rem
rem The list is read from pyproject.toml, so there is one place that says what
rem the dependencies are.
setlocal EnableExtensions EnableDelayedExpansion

set "ZLC_HOME=%~dp0.."

rem Which interpreter runs this product is ONE question, and it has an owner --
rem the same resolver the FPGA launchers ask.  Answering it here with the bare
rem name "python" is what made a double-clicked window die with "exited with
rem code 9009" and nothing else on a machine where that name does not resolve.
call "%ZLC_HOME%\packages\zlc_pulse\fpga\_resolve_tools.bat" python "%ZLC_HOME%"
if errorlevel 1 (
  echo.
  echo Nothing can be installed until this machine has a Python to install with.
  if "%ZLC_NO_PAUSE%"=="" pause
  endlocal & exit /b 1
)

echo.
echo ============================================================
echo ZOU LAB CONTROL - install requirements
echo Interpreter: %ZLC_PY_CMD%
echo ============================================================

rem Asked the way every other line here calls Python.  Inside for /f the
rem command is re-parsed by cmd, which eats the quotes around an
rem interpreter given as a full path -- and a full path is exactly what a
rem machine without Python on PATH resolves to.  The capture came back
rem empty and this said the dependency list could not be read.
set "ZLC_DEPS_FILE=%TEMP%\zlc_dependencies.txt"
%ZLC_PY_CMD% -c "import tomllib;print(' '.join(tomllib.load(open(r'%ZLC_HOME%\pyproject.toml','rb'))['project']['dependencies']))" > "%ZLC_DEPS_FILE%"
if exist "%ZLC_DEPS_FILE%" set /p ZLC_DEPS=<"%ZLC_DEPS_FILE%"
del "%ZLC_DEPS_FILE%" >nul 2>&1

if not defined ZLC_DEPS (
  echo could not read the dependency list from pyproject.toml
  if "%ZLC_NO_PAUSE%"=="" pause
  endlocal & exit /b 2
)

echo.
echo installing: !ZLC_DEPS!
echo.
%ZLC_PY_CMD% -m pip install !ZLC_DEPS!
set "ZLC_STATUS=!ERRORLEVEL!"

if not "!ZLC_STATUS!"=="0" (
  echo.
  echo pip exited with code !ZLC_STATUS!.
) else (
  echo.
  echo done.  Run bin\experiment.bat from your experiment folder.
)
if "%ZLC_NO_PAUSE%"=="" pause
endlocal & exit /b %ZLC_STATUS%
