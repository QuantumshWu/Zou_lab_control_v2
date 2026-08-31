# RF vendor files

Put the vendor artifacts the RF drivers need into THIS folder:

- **Vaunix Lab Brick (rf.vaunix_lms)**: copy `vnx_fmsynth.dll` here
  (from the Vaunix LMS SDK, 64-bit build to match the bench Python).
  Alternatively create `vendor.json` here with
  `{"vnx_fmsynth.dll": "C:/absolute/path/to/vnx_fmsynth.dll"}`.
- **Rigol DG4000 (rf.rigol_dg4000)**: speaks VISA; install NI-VISA
  system-wide or `pip install pyvisa-py` -- no file goes here.

Vendor binaries are per-machine and never committed; git ignores
everything in this folder except this README.
