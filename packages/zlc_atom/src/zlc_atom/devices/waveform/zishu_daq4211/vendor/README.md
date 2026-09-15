# ZishuTech DAQ vendor files

Put `daqlib2.dll` (the libdaq2 SDK, 64-bit — the `MS64` folder of the
vendor's release archive) into THIS folder, or write its absolute path into
`vendor.json` beside it:

    {"daqlib2.dll": "C:\zishutech\libdaq2\MS64\daqlib2.dll"}

`waveform.zishu_daq4211` looks here and nowhere else.  Nothing is committed:
the folder keeps only this README and a `.gitignore`.
