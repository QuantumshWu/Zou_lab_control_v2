@echo off
rem Start the sole SLM output owner. Default transport is the proven DVI
rem display path; pass --transport usb only when the vendor SDK is installed.
rem A local client uses 127.0.0.1; another
rem control computer uses this machine's LAN address.  Device browsing and
rem target editing stay client-local; only phase commands cross this boundary.
call "%~dp0_launch.bat" slm_server %*
