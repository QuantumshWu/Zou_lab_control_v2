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
echo ERROR: no working Python was found on this machine.
echo        Every python.exe and py.exe on PATH was tried and none of them ran.
echo        Install Python 3 with "Add python.exe to PATH" ticked, or point
echo        ZLC_PY_CMD at the interpreter to use, for example:
echo            set ZLC_PY_CMD=C:\Python313\python.exe
set "ZLC_TOOL_REPO_ROOT="
exit /b 1

:zlc_python_try
rem One candidate: keep it only if it actually starts.
if defined ZLC_PY_PATH exit /b 0
"%~1" -c "import sys" >nul 2>&1
if errorlevel 1 exit /b 0
set "ZLC_PY_PATH=%~1"
exit /b 0

:zlc_python_executable
for %%I in ("%ZLC_PY_PATH%") do set "ZLC_PY_EXT=%%~xI"
if /I "%ZLC_PY_EXT%"==".bat" set "ZLC_PY_CMD=call "%ZLC_PY_PATH%""
if /I not "%ZLC_PY_EXT%"==".bat" set "ZLC_PY_CMD="%ZLC_PY_PATH%""

:zlc_python_found
echo ZLC Python: %ZLC_PY_PATH%
set "ZLC_TOOL_REPO_ROOT="
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
