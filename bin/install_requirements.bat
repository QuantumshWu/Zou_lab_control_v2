@echo off
rem Install the product, or only its GPU dependency group, using one manifest.
setlocal EnableExtensions DisableDelayedExpansion

set "ZLC_INSTALL_EXTRAS=notebook"
if /I "%~1"=="slm-gpu" set "ZLC_INSTALL_EXTRAS=slm-gpu"
if not "%~1"=="" if /I not "%~1"=="slm-gpu" (
  echo Usage: install_requirements.bat [slm-gpu]
  if "%ZLC_NO_PAUSE%"=="" pause
  exit /b 2
)
set "ZLC_INSTALL_LOG="
if /I "%ZLC_INSTALL_EXTRAS%"=="slm-gpu" set "ZLC_INSTALL_LOG=%TEMP%\zlc-slm-gpu-install.log"

for %%I in ("%~dp0..") do set "ZLC_HOME=%%~fI"
call "%ZLC_HOME%\packages\zlc_pulse\fpga\_resolve_tools.bat" python "%ZLC_HOME%" /installed
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
echo Extras:      %ZLC_INSTALL_EXTRAS%
if defined ZLC_INSTALL_LOG echo Log:         %ZLC_INSTALL_LOG%
echo ============================================================
if /I "%ZLC_INSTALL_EXTRAS%"=="slm-gpu" goto zlc_gpu_install
echo [1/3] Installing product dependencies...
set "ZLC_INSTALL_COMMAND=%ZLC_PY_CMD% -m pip install --constraint "%ZLC_HOME%\constraints.txt" --editable "%ZLC_HOME%[%ZLC_INSTALL_EXTRAS%]""
call :zlc_run_step
set "ZLC_STATUS=%ERRORLEVEL%"
if not "%ZLC_STATUS%"=="0" goto zlc_failed
echo [2/3] Checking dependency consistency...
set "ZLC_INSTALL_COMMAND=%ZLC_PY_CMD% -m pip check"
call :zlc_run_step
set "ZLC_STATUS=%ERRORLEVEL%"
if not "%ZLC_STATUS%"=="0" goto zlc_failed
pushd "%TEMP%"
echo [3/3] Checking the installed product...
set "ZLC_INSTALL_COMMAND=%ZLC_PY_CMD% -m zou_lab_control check"
call :zlc_run_step
set "ZLC_STATUS=%ERRORLEVEL%"
popd
if not "%ZLC_STATUS%"=="0" goto zlc_failed

:zlc_installed
echo.
echo Installed. Run bin\experiment.bat from the experiment workspace.
if defined ZLC_INSTALL_LOG echo GPU installation log: %ZLC_INSTALL_LOG%
if "%ZLC_NO_PAUSE%"=="" pause
exit /b 0

:zlc_failed
echo.
echo Installation failed with code %ZLC_STATUS%.
if defined ZLC_INSTALL_LOG echo Original error and progress log: %ZLC_INSTALL_LOG%
if "%ZLC_NO_PAUSE%"=="" pause
exit /b %ZLC_STATUS%

:zlc_gpu_install
echo [GPU] Installing only the manifest's GPU dependencies and their requirements...
echo This does not reinstall the product or change Notebook/JupyterLab.
echo This does not install or update the NVIDIA display driver.
set "ZLC_INSTALL_COMMAND=%ZLC_PY_CMD% -c "import pathlib,subprocess,sys,tomllib; root=pathlib.Path(sys.argv[1]); deps=tomllib.loads((root/'pyproject.toml').read_text(encoding='utf-8'))['project']['optional-dependencies']['slm-gpu']; raise SystemExit(subprocess.call([sys.executable,'-m','pip','install','--constraint',str(root/'constraints.txt'),*deps]))" "%ZLC_HOME%""
call :zlc_run_step
set "ZLC_STATUS=%ERRORLEVEL%"
if not "%ZLC_STATUS%"=="0" goto zlc_failed
echo [GPU] Checking imports, the selected device, and a real dot product and FFT...
pushd "%TEMP%"
set "ZLC_INSTALL_COMMAND=%ZLC_PY_CMD% -c "import zou_lab_control; from zlc_atom.devices.slm import solver; import sys; print('Interpreter:', sys.executable); print('Root:', zou_lab_control.__file__); print('SLM:', solver.__file__); import cupy as cp; from cuda.bindings import runtime; print('CuPy:', cp.__version__, cp.__file__); print('CUDA bindings:', runtime.__file__); info=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id); print('GPU:', info['name']); x=cp.arange(4, dtype=cp.float32); value=float(cp.dot(x, x).item()); spectrum=cp.asnumpy(cp.fft.fft(x)); assert value == 14.0 and float(spectrum[0].real) == 6.0; print('GPU dot + FFT: PASS', value, spectrum)""
call :zlc_run_step
set "ZLC_STATUS=%ERRORLEVEL%"
popd
if not "%ZLC_STATUS%"=="0" goto zlc_failed
goto zlc_installed

:zlc_run_step
if defined ZLC_INSTALL_LOG goto zlc_logged_step
%ZLC_INSTALL_COMMAND%
exit /b %ERRORLEVEL%

:zlc_logged_step
rem Tee keeps pip progress and original Python tracebacks visible and saved.
powershell.exe -NoLogo -NoProfile -Command "& cmd.exe /d /c $env:ZLC_INSTALL_COMMAND 2>&1 | Tee-Object -FilePath $env:ZLC_INSTALL_LOG -Append; exit $LASTEXITCODE"
exit /b %ERRORLEVEL%
