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

The top-level facade contains only these names:

| Public name | Purpose |
|---|---|
| `atomic_write_file`, `atomic_write_bytes`, `atomic_write_text` | publish complete content by same-directory temporary, fsync, replace, and directory flush |
| `readable_json_bytes`, `write_readable_json` | validate a plain JSON tree and encode or durably write its readable UTF-8 representation |
| `durable_makedirs` | durably create every missing level of a directory tree |
| `day_folder` | create or open one calendar-day folder beneath an existing save root |
| `unique_path` | atomically publish a complete file at the first free numbered name, or exclusively create a uniquely named run directory |
| `DirectoryDurabilityError` | report that a directory entry could not be made crash-durable |

## Where things live

Saved work groups first **by date**. Ordinary explicit saves may be files in the
day folder; hosted long-running Tasks allocate one exclusive directory per run:

```
<save_root>/2026_08_05/mot-loading.npz
<save_root>/2026_08_05/calibration/run.json
<save_root>/2026_08_05/calibration-2/run.json
<save_root>/2026_08_06/...
```

You look for today's data under today's date. `save_root` is yours to choose;
this package only creates the day folder beneath it, and refuses a root that
does not exist so a typo cannot scatter data into a new tree.

For a file, `unique_path` requires a `writer`. The writer receives a hidden
same-directory temporary path with the requested suffix, so path-based encoders
can use it directly. The complete flushed file is then published without
replacement. Concurrent processes therefore cannot select the same final name,
and a failed writer publishes no partial artifact. An empty suffix instead
performs an exclusive `mkdir`; it takes no writer.

## What is deliberately not here

**No format encoder and no content digests.** Each artifact owner validates its
own stable format identity and strict current grammar. This package only makes
the resulting bytes or arrays land durably at an explicit or uniquely allocated
path.
