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

if not defined ZLC_PY_CMD set "ZLC_PY_CMD=python"
set "ZLC_HOME=%~dp0.."

echo.
echo ============================================================
echo ZOU LAB CONTROL - install requirements
echo Interpreter: %ZLC_PY_CMD%
echo ============================================================

for /f "usebackq delims=" %%D in (`%ZLC_PY_CMD% -c "import tomllib;print(' '.join(tomllib.load(open(r'%ZLC_HOME%\pyproject.toml','rb'))['project']['dependencies']))"`) do set "ZLC_DEPS=%%D"

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
