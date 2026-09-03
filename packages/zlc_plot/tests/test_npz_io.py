from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest

from data_factory import (
    make_dataset_schema,
    make_snapshot,
    mapped_domain_from_columns,
    repeat_domain,
)

from zlc_data import OwnedSnapshot
from zlc_data import NPZFormatError, load_npz, save_npz

def _snapshot() -> OwnedSnapshot:
    schema = make_dataset_schema(
        repeat_domain(size=2),
        mapped_domain_from_columns({"x": [0.0, 1.0]}),
        dtype=np.float32,
    )
    values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    validity = np.array([[True, False], [True, True]], dtype=np.bool_)
    return make_snapshot(schema, values, revision=12, validity=validity)

def _encoded(snapshot: OwnedSnapshot) -> bytes:
    stream = BytesIO()
    save_npz(stream, snapshot)
    return stream.getvalue()

def test_npz_round_trip_preserves_snapshot_and_encoding() -> None:
    snapshot = _snapshot()
    first = _encoded(snapshot)
    second = _encoded(snapshot)
    assert first == second
    restored = load_npz(BytesIO(first))
    assert restored.ref == snapshot.ref
    assert restored.block.schema.fingerprint == snapshot.block.schema.fingerprint
    assert np.array_equal(restored.block.values, snapshot.block.values)
    assert np.array_equal(restored.expanded_validity(), snapshot.expanded_validity())
    assert restored.block.values.dtype == np.dtype(np.float32)
    assert restored.expanded_validity().dtype == np.dtype(np.bool_)

def test_npz_missing_manifest_is_rejected() -> None:
    stream = BytesIO()
    np.savez_compressed(stream, **{"snapshot.values": np.zeros(1, dtype=np.float32)})
    with pytest.raises(NPZFormatError, match="manifest"):
        load_npz(BytesIO(stream.getvalue()))

def test_npz_extra_member_is_rejected() -> None:
    encoded = _encoded(_snapshot())
    with np.load(BytesIO(encoded), allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["unexpected"] = np.asarray(1, dtype=np.int64)
    malformed = BytesIO()
    np.savez_compressed(malformed, **arrays)
    with pytest.raises(NPZFormatError, match="members mismatch"):
        load_npz(BytesIO(malformed.getvalue()))
