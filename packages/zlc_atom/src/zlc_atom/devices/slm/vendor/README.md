# SLM vendor files

For USB control of the Hamamatsu X15213 (`transport = usb`), copy the
Hamamatsu SDK DLL `hpkSLMdaLV.dll` (with the rest of its SDK folder's
DLLs beside it) into THIS folder, or author the SDK directory in the
device form ("Hamamatsu SDK directory").

The DVI transport needs no vendor file.  Vendor binaries are per-machine
and never committed; git ignores everything here except this README.
