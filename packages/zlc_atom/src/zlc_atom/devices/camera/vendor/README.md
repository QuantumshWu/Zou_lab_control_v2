# Camera vendor files

Put the vendor artifacts the camera drivers need into THIS folder:

- **Hamamatsu qCMOS (camera.dcam)**: copy `dcamapi.dll` here (the DCAM-API
  runtime; the 64-bit build to match the bench Python).  Alternatively
  create `vendor.json` here with
  `{"dcamapi.dll": "C:/absolute/path/to/dcamapi.dll"}` -- an absolute path,
  for example the copy the Hamamatsu installer placed under
  `C:/Windows/System32`.  The driver looks nowhere else: `Scan hardware`
  and opening the camera both report this instruction when the file is
  missing.
- **Basler (camera.pylon)**: reached through `pypylon`, which pip installs
  together with the pylon runtime; no file goes here.

Vendor binaries are per-machine and never committed; git ignores
everything in this folder except this README.
