@echo off
rem The pulse window: edit a sequence, preview it, fire it.
rem
rem   bin\pulse_editor.bat
rem   bin\pulse_editor.bat --pulse imaging_template.json
rem   bin\pulse_editor.bat --connect virtual
rem   bin\pulse_editor.bat --connect remote:127.0.0.1:18861
rem
rem A remote connection needs the pulse server running on the machine wired to
rem the board -- that is zlc_pulse\fpga\run_server.bat, which prints the exact
rem endpoint to type here.  Without --connect the window opens offline and the
rem Connection panel can dial it later.
call "%~dp0_launch.bat" pulse_editor %*
