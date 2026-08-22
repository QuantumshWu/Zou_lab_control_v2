@echo off
rem Device Manager Init opens the Task Console in one ExperimentSession.
setlocal DisableDelayedExpansion
set "ZLC_COMMAND=task_console"
call "%~dp0_launch.bat" %*
exit /b %ERRORLEVEL%
