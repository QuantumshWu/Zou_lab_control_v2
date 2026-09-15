# SLM vendor files

For USB control of the Hamamatsu X15213 (`transport = usb`), copy the
Hamamatsu SDK DLL `hpkSLMdaLV.dll` (with the rest of its SDK folder's
DLLs beside it) into THIS folder, or create `vendor.json` here with
`{"hpkSLMdaLV.dll": "C:/absolute/path/to/hpkSLMdaLV.dll"}`; the SDK's
other DLLs are then loaded from that file's folder.  Nowhere else is
searched, and a missing library is reported as exactly this instruction.

The DVI transport needs no vendor file.  Vendor binaries are per-machine
and never committed; git ignores everything here except this README.
