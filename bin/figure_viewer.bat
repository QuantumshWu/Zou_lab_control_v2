@echo off
rem The saved-figure window: open an archive and read what it was.
rem
rem   bin\figure_viewer.bat
rem   bin\figure_viewer.bat --path data\2026_08_05\mot-loading.npz
rem
rem It needs no session, no devices and no apparatus file, so an archive from
rem the bench opens on any machine that has this checkout.
call "%~dp0_launch.bat" figure_viewer %*
