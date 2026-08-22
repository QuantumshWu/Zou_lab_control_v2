from __future__ import annotations

import os
import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
if os.environ.get("ZLC_TEST_INSTALLED") != "1" and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
