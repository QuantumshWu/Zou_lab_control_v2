@echo off
rem The live experiment window: devices, panels, the display beat.
rem
rem   bin\task_console.bat
rem   bin\task_console.bat --template virtual
rem   bin\task_console.bat --workspace D:\experiment --pulse calibration
rem
rem Without --template it reads apparatus.json from the workspace.  Start from
rem --template virtual the first time: it needs no hardware at all.
call "%~dp0_launch.bat" task_console %*
