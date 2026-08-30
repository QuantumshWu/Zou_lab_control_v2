@echo off

rem Single owner for the Python and Vivado executables every launcher uses --
rem the windows in bin\ as much as the FPGA scripts.  Which interpreter runs
rem this product is one question, and a launcher that answered it itself with
rem the bare name "python" is how a double-clicked window died with nothing
rem but "exited with code 9009" on a machine where that name is not on PATH.
rem This file intentionally has no SETLOCAL: resolved variables must return to
rem the caller, whose launcher already owns the enclosing local environment.

if /I "%~1"=="python" goto zlc_find_python
if /I "%~1"=="vivado" goto zlc_find_vivado
echo ERROR: _resolve_tools.bat expects python or vivado.
exit /b 2

:zlc_find_python
if "%~2"=="" (
  echo ERROR: Python resolution requires the repository root.
  exit /b 2
)
for %%I in ("%~2") do set "ZLC_TOOL_REPO_ROOT=%%~fI"
if defined ZLC_PY_CMD (
  if not defined ZLC_PY_PATH set "ZLC_PY_PATH=%ZLC_PY_CMD%"
  goto zlc_python_found
)
if defined ZLC_FPGA_PYTHON (
  if exist "%ZLC_FPGA_PYTHON%" (
    set "ZLC_PY_PATH=%ZLC_FPGA_PYTHON%"
    goto zlc_python_executable
  ) else (
    set "ZLC_PY_CMD=%ZLC_FPGA_PYTHON%"
    set "ZLC_PY_PATH=%ZLC_FPGA_PYTHON%"
  )
  goto zlc_python_found
)
rem NO venv preference here on purpose.  Preferring a repo .venv is what made the
rem server fall back to JTAG forever: that interpreter had no pyserial, so port
rem enumeration raised ModuleNotFoundError, the UART probe never ran, and auto
rem resolved to jtag-axi whether or not the USB-UART cable was plugged in.  This
rem project installs its dependencies globally; the global interpreter is the
rem one that has them.
if exist "%ZLC_TOOL_REPO_ROOT%\.zlc_python_path" (
  set /p "ZLC_STORED_PY="<"%ZLC_TOOL_REPO_ROOT%\.zlc_python_path"
  if exist "!ZLC_STORED_PY!" (
    call :zlc_python_try "!ZLC_STORED_PY!"
    if defined ZLC_PY_PATH goto zlc_python_executable
  )
  echo Ignoring stale .zlc_python_path: !ZLC_STORED_PY!
)
rem A name on PATH is not an interpreter.  Windows ships a python.exe under
rem WindowsApps that only opens the Store and answers 9009 -- the very code a
rem launcher then reports with nothing an operator can act on.  Every candidate
rem is asked to run before it is believed.
for /f "delims=" %%I in ('where python 2^>nul') do call :zlc_python_try "%%I"
if defined ZLC_PY_PATH goto zlc_python_executable
for /f "delims=" %%I in ('where py 2^>nul') do call :zlc_python_try "%%I"
if defined ZLC_PY_PATH goto zlc_python_executable

rem PATH is where an interpreter ANNOUNCES itself, not where it lives.  The
rem Anaconda installer recommends leaving PATH alone and ships no py.exe, so
rem on a machine that has Python through conda both searches above come back
rem empty and the launcher used to report 9009 for a machine that is in fact
rem fully equipped.  Ask conda where its base is, then look where installers
rem actually put things.
if defined CONDA_PREFIX call :zlc_python_try "%CONDA_PREFIX%\python.exe"
if defined ZLC_PY_PATH goto zlc_python_executable
if defined CONDA_EXE for %%I in ("%CONDA_EXE%\..\..") do call :zlc_python_try "%%~fI\python.exe"
if defined ZLC_PY_PATH goto zlc_python_executable
for /f "delims=" %%I in ('conda info --base 2^>nul') do call :zlc_python_try "%%I\python.exe"
if defined ZLC_PY_PATH goto zlc_python_executable
for %%R in (
  "%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%USERPROFILE%\miniforge3"
  "%LOCALAPPDATA%\anaconda3" "%LOCALAPPDATA%\miniconda3"
  "%LOCALAPPDATA%\Continuum\anaconda3" "%LOCALAPPDATA%\Continuum\miniconda3"
  "%ProgramData%\Anaconda3" "%ProgramData%\Miniconda3" "%ProgramData%\miniforge3"
  "%ProgramFiles%\Anaconda3" "%ProgramFiles%\Miniconda3"
  "C:\Anaconda3" "C:\Miniconda3"
) do call :zlc_python_try "%%~R\python.exe"
if defined ZLC_PY_PATH goto zlc_python_executable
for /d %%R in (
  "%LOCALAPPDATA%\Programs\Python\Python3*"
  "%ProgramFiles%\Python3*" "%ProgramFiles(x86)%\Python3*" "C:\Python3*"
) do call :zlc_python_try "%%~R\python.exe"
if defined ZLC_PY_PATH goto zlc_python_executable

echo ERROR: no working Python was found on this machine.
echo        PATH was searched (python.exe, py.exe), then conda's base and the
echo        usual install folders for Anaconda, Miniconda and python.org.
echo        Install Python 3, or point ZLC_PY_CMD at the interpreter to use:
echo            setx ZLC_PY_CMD "C:\Users\you\anaconda3\python.exe"
set "ZLC_TOOL_REPO_ROOT="
exit /b 1

:zlc_python_try
rem One candidate: keep it only if it actually starts.  A name that exists is
rem not an interpreter -- Windows ships a python.exe under WindowsApps that
rem only opens the Store and answers 9009.
if defined ZLC_PY_PATH exit /b 0
if not exist "%~1" exit /b 0
"%~1" -c "import sys" >nul 2>&1
if errorlevel 1 exit /b 0
set "ZLC_PY_PATH=%~1"
exit /b 0

:zlc_python_conda_activate
rem A conda interpreter found by path still needs its environment: PyQt5 and
rem numpy load DLLs from Library\bin, which is what activation puts on PATH.
rem Without this the window dies with "DLL load failed" although python.exe
rem itself started perfectly.
if not exist "%ZLC_PY_PATH%" exit /b 0
for %%I in ("%ZLC_PY_PATH%") do set "ZLC_CONDA_ROOT=%%~dpI"
if not exist "%ZLC_CONDA_ROOT%conda-meta\" (
  set "ZLC_CONDA_ROOT="
  exit /b 0
)
set "PATH=%ZLC_CONDA_ROOT%;%ZLC_CONDA_ROOT%Library\mingw-w64\bin;%ZLC_CONDA_ROOT%Library\usr\bin;%ZLC_CONDA_ROOT%Library\bin;%ZLC_CONDA_ROOT%Scripts;%ZLC_CONDA_ROOT%bin;%ZLC_CONDA_ROOT%condabin;%PATH%"
echo ZLC conda environment: %ZLC_CONDA_ROOT%
set "ZLC_CONDA_ROOT="
exit /b 0

:zlc_python_executable
for %%I in ("%ZLC_PY_PATH%") do set "ZLC_PY_EXT=%%~xI"
if /I "%ZLC_PY_EXT%"==".bat" set "ZLC_PY_CMD=call "%ZLC_PY_PATH%""
if /I not "%ZLC_PY_EXT%"==".bat" set "ZLC_PY_CMD="%ZLC_PY_PATH%""

:zlc_python_found
call :zlc_python_conda_activate
call :zlc_python_product_environment "%~3"
echo ZLC Python: %ZLC_PY_PATH%
set "ZLC_TOOL_REPO_ROOT="
exit /b 0

:zlc_python_product_environment
rem Every human-facing command launched from a checkout must import THAT
rem checkout, independently of the caller's working directory and of editable
rem installs left in the selected interpreter.  The product dispatcher lives at
rem the root, while each layer deliberately lives under packages/<layer>/src;
rem injecting only the root leaves those layers to whichever unrelated editable
rem install happens to be registered in site-packages.
rem The installer is the sole exception: /installed clears source injection so
rem its final check proves the installed distribution rather than the checkout.
if /I "%~1"=="/installed" (
  set "PYTHONPATH="
  set "ZLC_CHECKOUT_PRODUCT_ROOT="
  exit /b 0
)
if defined ZLC_CHECKOUT_PRODUCT_ROOT if /I "%ZLC_CHECKOUT_PRODUCT_ROOT%"=="%ZLC_TOOL_REPO_ROOT%" exit /b 0
set "ZLC_CHECKOUT_PYTHONPATH=%ZLC_TOOL_REPO_ROOT%;%ZLC_TOOL_REPO_ROOT%\packages\zlc_data\src;%ZLC_TOOL_REPO_ROOT%\packages\zlc_durable\src;%ZLC_TOOL_REPO_ROOT%\packages\zlc_runtime\src;%ZLC_TOOL_REPO_ROOT%\packages\zlc_plot\src;%ZLC_TOOL_REPO_ROOT%\packages\zlc_ui\src;%ZLC_TOOL_REPO_ROOT%\packages\zlc_pulse\src;%ZLC_TOOL_REPO_ROOT%\packages\zlc_atom\src;%ZLC_TOOL_REPO_ROOT%\packages\zlc_workbench\src"
if defined PYTHONPATH (
  set "PYTHONPATH=%ZLC_CHECKOUT_PYTHONPATH%;%PYTHONPATH%"
) else (
  set "PYTHONPATH=%ZLC_CHECKOUT_PYTHONPATH%"
)
set "ZLC_CHECKOUT_PYTHONPATH="
set "ZLC_CHECKOUT_PRODUCT_ROOT=%ZLC_TOOL_REPO_ROOT%"
exit /b 0

:zlc_find_vivado
if defined ZLC_PS_VIVADO_BIN goto zlc_vivado_found
for %%V in (2019.1 2019.2 2020.1 2020.2 2021.1 2021.2 2022.1 2022.2 2023.1 2023.2 2024.1 2024.2 2025.1 2025.2 2026.1 2026.2) do (
  if exist "C:\Xilinx\Vivado\%%V\bin\vivado.bat" set "ZLC_PS_VIVADO_BIN=C:\Xilinx\Vivado\%%V\bin\vivado.bat"
  if exist "D:\Xilinx\Vivado\%%V\bin\vivado.bat" set "ZLC_PS_VIVADO_BIN=D:\Xilinx\Vivado\%%V\bin\vivado.bat"
  if exist "C:\AMD\Vivado\%%V\bin\vivado.bat" set "ZLC_PS_VIVADO_BIN=C:\AMD\Vivado\%%V\bin\vivado.bat"
  if exist "D:\AMD\Vivado\%%V\bin\vivado.bat" set "ZLC_PS_VIVADO_BIN=D:\AMD\Vivado\%%V\bin\vivado.bat"
)
rem Preserve default-root discovery for releases newer than the explicit list.
for /d %%V in ("C:\Xilinx\Vivado\*" "D:\Xilinx\Vivado\*" "C:\AMD\Vivado\*" "D:\AMD\Vivado\*") do (
  if exist "%%~V\bin\vivado.bat" set "ZLC_PS_VIVADO_BIN=%%~V\bin\vivado.bat"
)
if defined ZLC_PS_VIVADO_BIN goto zlc_vivado_found
for /f "delims=" %%I in ('where vivado.bat 2^>nul') do if not defined ZLC_PS_VIVADO_BIN set "ZLC_PS_VIVADO_BIN=%%I"
if defined ZLC_PS_VIVADO_BIN goto zlc_vivado_found
where vivado >nul 2>nul
if not errorlevel 1 set "ZLC_PS_VIVADO_BIN=vivado"
if defined ZLC_PS_VIVADO_BIN goto zlc_vivado_found
if /I "%~3"=="/optional" (
  echo NOTE: Python will report a JTAG startup failure if no Vivado installation is available.
  exit /b 0
)
echo ERROR: Vivado was not found. Set ZLC_PS_VIVADO_BIN.
exit /b 1

:zlc_vivado_found
echo ZLC Vivado: %ZLC_PS_VIVADO_BIN%
exit /b 0
