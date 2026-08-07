"""The fit numeric contract is one document living in two repos.

zlc_plot produces the numeric table and zlc_runtime consumes it.  The two
copies of docs/fit-numeric-contract.md must stay byte-identical: a one-sided
edit is exactly the accident that produced the value/unit-system and
revision-semantics bugs earlier.  Pinning the digest makes a one-sided edit
fail here instead of on the bench.

To change the contract: edit BOTH repos' copies, recompute the digest in both,
and name the sibling commit in each commit message.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


CONTRACT = Path(__file__).resolve().parents[1] / "docs" / "fit-numeric-contract.md"
ANCHOR = re.compile(r"<!-- CONTRACT-SHA256: ([0-9a-f]{64}) -->")
AGREED = "582bfee5bea7e13153067bdb20d5770f1c31431cbd52cf1f5175238251973da3"


def _body_and_anchor() -> tuple[str, str]:
    text = CONTRACT.read_text(encoding="utf-8")
    match = ANCHOR.search(text)
    assert match, "the contract lost its digest anchor"
    return text[: match.start()], match.group(1)


def test_contract_matches_its_pinned_digest() -> None:
    body, anchor = _body_and_anchor()
    assert hashlib.sha256(body.encode("utf-8")).hexdigest() == anchor, (
        "docs/fit-numeric-contract.md changed without updating its digest -- "
        "update the sibling repo's copy in the same breath"
    )


def test_pinned_digest_is_the_agreed_one() -> None:
    """Both repos assert the same literal, so a one-sided edit cannot pass."""

    assert _body_and_anchor()[1] == AGREED
