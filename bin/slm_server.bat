@echo off
rem Start the sole installed SLM output owner. The proven DVI transport stays
rem the default; --transport usb opts into vendor-DLL discovery explicitly.
rem The Python owner prints the local and LAN host/port values clients enter.
setlocal EnableExtensions DisableDelayedExpansion
set "ZLC_COMMAND=slm_server"
call "%~dp0_launch.bat" %*
exit /b %ERRORLEVEL%
