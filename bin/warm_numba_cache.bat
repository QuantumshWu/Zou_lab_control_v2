@echo off
rem Warm the numba kernel cache so experiment runs never compile.
rem The cache lives in the repository's .numba_cache; when it already
rem holds machine code for the current toolchain and kernel source
rem (a fingerprint marker checks both), this exits in milliseconds.
setlocal DisableDelayedExpansion
for %%I in ("%~dp0..") do set "ZLC_HOME=%%~fI"
if "%NUMBA_CACHE_DIR%"=="" set "NUMBA_CACHE_DIR=%ZLC_HOME%\.numba_cache"
set "ZLC_COMMAND=warm_numba"
call "%~dp0_launch.bat" %*
exit /b %ERRORLEVEL%
