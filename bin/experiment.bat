@echo off
rem The experiment entry: Device Manager Init, then TaskConsole and PulseGUI
rem over one shared ExperimentSession in one Python process.
call "%~dp0_launch.bat" task_console %*
