# Vaunix Lab Brick vendor files

Put `vnx_fmsynth.dll` (the Vaunix LMS SDK, 64-bit) into THIS folder, or write
its absolute path into `vendor.json` beside it:

    {"vnx_fmsynth.dll": "C:\\Vaunix\\vnx_fmsynth.dll"}

`rf.vaunix_lms` looks here and nowhere else.  Nothing is committed: the folder
keeps only this README and a `.gitignore`.
