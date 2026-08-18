# zlc_durable

Write it atomically, land it where you meant to.

Every package in this workspace that saves anything depends on this one, and on
nothing else of ours. It is deliberately the bottom of the stack: standard
library only, no numpy, no `zlc_*`.

```python
from datetime import date
from zlc_durable import day_folder, unique_path

folder = day_folder("D:/data", date.today())
path = unique_path(
    folder,
    "MOT loading",
    ".json",
    writer=lambda temporary: temporary.write_text("{}", encoding="utf-8"),
)
```

## What is here

| | |
|---|---|
| `atomic_write_file` / `_bytes` / `_text` | temp file in the same directory, fsync, `os.replace`, then flush the directory — a reader never sees a half-written file |
| `durable_mkdir` / `durable_makedirs` | create a directory and flush its parent, so the entry survives a crash |
| `flush_directory` | the platform-specific directory flush the above rely on |
| `resolve_under` | resolve a relative path under a root and refuse anything that escapes it |
| `day_folder` / `day_folder_name` | the save layout: one folder per calendar day |
| `unique_path` | atomically publish a complete file at the first free numbered name, or exclusively create a uniquely named run directory |

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

For a file, `unique_path` requires a `writer`. The writer receives a hidden
same-directory temporary path with the requested suffix, so path-based encoders
can use it directly. The complete flushed file is then published without
replacement. Concurrent processes therefore cannot select the same final name,
and a failed writer publishes no partial artifact. An empty suffix instead
performs an exclusive `mkdir`; it takes no writer.

## What is deliberately not here

**No format encoder and no content digests.** Each artifact owner validates its
own declared format and version. This package only makes the resulting bytes or
arrays land durably at an explicit or uniquely allocated path.
