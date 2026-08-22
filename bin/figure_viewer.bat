@echo off
rem Open one saved Figure archive with the installed product.
setlocal DisableDelayedExpansion
set "ZLC_COMMAND=figure_viewer"
call "%~dp0_launch.bat" %*
exit /b %ERRORLEVEL%
