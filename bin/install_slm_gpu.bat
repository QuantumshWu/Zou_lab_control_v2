@echo off
rem One-click GPU dependencies through the same product installer/resolver.
setlocal EnableExtensions DisableDelayedExpansion
call "%~dp0install_requirements.bat" slm-gpu
exit /b %ERRORLEVEL%
