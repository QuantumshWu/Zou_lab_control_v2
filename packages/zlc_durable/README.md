# zlc_durable

Write it atomically, land it where you meant to.

Every package in this workspace that saves anything depends on this one, and on
nothing else of ours. It is deliberately the bottom of the stack: standard
library only, no numpy, no `zlc_*`.

```python
from datetime import date
from zlc_durable import atomic_write_text, day_folder, unique_path

folder = day_folder("D:/data", date.today())      # D:/data/2026_08_05/
path = unique_path(folder, "MOT loading", ".npz")  # never an occupied name
atomic_write_text(path.with_suffix(".json"), "{}") # temp file, fsync, replace
```

## What is here

| | |
|---|---|
| `atomic_write_file` / `_bytes` / `_text` | temp file in the same directory, fsync, `os.replace`, then flush the directory — a reader never sees a half-written file |
| `durable_mkdir` / `durable_makedirs` | create a directory and flush its parent, so the entry survives a crash |
| `flush_directory` | the platform-specific directory flush the above rely on |
| `resolve_under` | resolve a relative path under a root and refuse anything that escapes it |
| `day_folder` / `day_folder_name` / `unique_path` | the save layout: one folder per calendar day, non-colliding names within it |

## Where things live

Saved work groups **by date**, with no per-run subdirectory:

```
<save_root>/2026_08_05/mot-loading.npz
<save_root>/2026_08_05/mot-loading-2.npz
<save_root>/2026_08_06/...
```

You look for today's data under today's date. `save_root` is yours to choose;
this package only creates the day folder beneath it, and refuses a root that
does not exist so a typo cannot scatter data into a new tree.

Date routing lives here rather than in the composition root because deciding
where a file goes and putting it there safely are one concern — and because a
notebook calls these functions directly, with no application in the process.

## What is deliberately not here

**No canonical encoder and no content digests.** The pre-split tree had a
701-line `canonical.py` producing framed, digest-stamped records. Archives in
this project are plain JSON and plain arrays, readable without importing any of
our packages, so an old file still opens after we refactor.

## Provenance

`durability.py` (224 lines) and `paths.py` (33 lines) are byte-identical
migrations of the pre-split `zlc_storage`; `tests/test_durability.py` is its
test file with only the import lines changed. `workspace.py` is new.
