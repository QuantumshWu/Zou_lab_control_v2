@echo off
rem Open the installed Pulse Editor; connection choice remains operator-owned.
setlocal DisableDelayedExpansion
set "ZLC_COMMAND=pulse_editor"
call "%~dp0_launch.bat" %*
exit /b %ERRORLEVEL%
