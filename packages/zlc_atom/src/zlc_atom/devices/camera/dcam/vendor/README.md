# Hamamatsu qCMOS vendor files

Put `dcamapi.dll` (the Hamamatsu DCAM-API runtime, 64-bit) into THIS folder,
or write its absolute path into `vendor.json` beside it:

    {"dcamapi.dll": "C:\\Windows\\System32\\dcamapi.dll"}

`camera.dcam` looks here and nowhere else.  Nothing is committed: the folder
keeps only this README and a `.gitignore`.
