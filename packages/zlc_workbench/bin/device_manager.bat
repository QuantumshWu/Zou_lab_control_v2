@echo off
rem The apparatus editor: which devices this bench has, and how each is set up.
rem
rem   bin\device_manager.bat
rem
rem It opens no device.  Writing down that the bench has a camera at index 2 is
rem a different act from reaching for it, so an apparatus can be authored on a
rem laptop with no hardware attached.
call "%~dp0_launch.bat" device_manager %*
