@echo off
rem Start the sole USB SLM owner.  A local client uses 127.0.0.1; another
rem control computer uses this machine's LAN address.  Device browsing and
rem target editing stay client-local; only phase commands cross this boundary.
call "%~dp0_launch.bat" slm_server %*
