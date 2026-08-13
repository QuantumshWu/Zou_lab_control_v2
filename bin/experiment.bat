@echo off
rem The experiment entry: Device Manager Init opens TaskConsole; each loaded
rem device card opens its on-demand Control in the same ExperimentSession.
call "%~dp0_launch.bat" task_console %*
