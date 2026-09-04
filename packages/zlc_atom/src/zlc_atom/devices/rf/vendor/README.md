# RF vendor files

Put the vendor artifacts the RF drivers need into THIS folder:

- **Vaunix Lab Brick (rf.vaunix_lms)**: copy `vnx_fmsynth.dll` here
  (from the Vaunix LMS SDK, 64-bit build to match the bench Python).
  Alternatively create `vendor.json` here with
  `{"vnx_fmsynth.dll": "C:/absolute/path/to/vnx_fmsynth.dll"}`.
- **Rigol DG4000 (rf.rigol_dg4000)**: speaks VISA, and no file goes here --
  but VISA is not all pip: the product ships PyVISA and the PyVISA-py
  fallback backend, while getting the instrument to APPEAR is per-machine.
  Over **USB** it must have a USB-TMC driver bound to it, which is what
  installing **NI-VISA** (or Rigol UltraSigma) does; until then VISA does
  not list it and `Scan hardware` cannot find it. Over **LAN** it is listed
  only once added in NI MAX -- or skip listing entirely and type its
  `TCPIP0::<address>::INSTR` into the device's VISA resource field, which
  needs no vendor install at all.

Vendor binaries are per-machine and never committed; git ignores
everything in this folder except this README.
