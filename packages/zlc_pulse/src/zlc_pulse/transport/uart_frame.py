"""CRC-framed UART codec for the word-addressed register map."""

from __future__ import annotations

import binascii
import struct
from typing import Sequence

SYNC0 = 0x5A
SYNC1 = 0xA5
OP_WRITE = 0x01
OP_READ = 0x02
RESP = 0x81
ST_OK = 0x00
ST_CRC_FAIL = 0x01
ST_BAD_OP = 0x02
ST_ADDR_RANGE = 0x03
OFF_OP = 2
CRC_LEN = 2
MASK32 = 0xFFFFFFFF
MAX_FRAME_WORDS = 256


class FrameError(ValueError):
    pass


def crc16_ccitt(data: bytes) -> int:
    return binascii.crc_hqx(data, 0xFFFF)


def _unsigned(value: int, maximum: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise ValueError(f"{name} is outside its wire range")
    return value


def _span(address: int, count: int) -> tuple[int, int]:
    address = _unsigned(address, MASK32, "word_addr")
    count = _unsigned(count, MAX_FRAME_WORDS, "count")
    if count == 0 or address + count - 1 > MASK32:
        raise ValueError("invalid UART frame span")
    return address, count


def _frame(body: bytes) -> bytes:
    return bytes((SYNC0, SYNC1)) + body + crc16_ccitt(body).to_bytes(2, "little")


def encode_write(word_addr: int, values: Sequence[int], *, seq: int = 0) -> bytes:
    address, count = _span(word_addr, len(values))
    sequence = _unsigned(seq, 0xFF, "seq")
    body = bytes((OP_WRITE, sequence)) + address.to_bytes(4, "little") + count.to_bytes(2, "little")
    body += struct.pack(f"<{count}I", *[_unsigned(int(value), MASK32, "value") for value in values])
    return _frame(body)


def encode_read(word_addr: int, count: int, *, seq: int = 0) -> bytes:
    address, count = _span(word_addr, count)
    sequence = _unsigned(seq, 0xFF, "seq")
    return _frame(bytes((OP_READ, sequence)) + address.to_bytes(4, "little") + count.to_bytes(2, "little"))


def encode_reply(seq: int, status: int, words: Sequence[int] = ()) -> bytes:
    sequence = _unsigned(seq, 0xFF, "seq")
    status = _unsigned(status, 0xFF, "status")
    if len(words) > MAX_FRAME_WORDS:
        raise ValueError("reply is too long")
    body = bytes((RESP, sequence, status)) + len(words).to_bytes(2, "little")
    if words:
        body += struct.pack(f"<{len(words)}I", *[_unsigned(int(value), MASK32, "word") for value in words])
    return _frame(body)


def decode_reply(frame: bytes) -> tuple[int, int, list[int]]:
    if len(frame) < 9 or frame[:2] != bytes((SYNC0, SYNC1)) or frame[2] != RESP:
        raise FrameError("invalid UART reply header")
    count = int.from_bytes(frame[5:7], "little")
    length = 2 + 3 + 2 + 4 * count + CRC_LEN
    if len(frame) != length:
        raise FrameError("invalid UART reply length")
    body = frame[OFF_OP:-CRC_LEN]
    if crc16_ccitt(body) != int.from_bytes(frame[-CRC_LEN:], "little"):
        raise FrameError("UART reply CRC mismatch")
    words = [int.from_bytes(frame[7 + index * 4:11 + index * 4], "little") for index in range(count)]
    return frame[3], frame[4], words


def reply_frame_len(count: int) -> int:
    _unsigned(count, MAX_FRAME_WORDS, "count")
    return 2 + 3 + 2 + 4 * count + CRC_LEN


def coalesce_runs(pairs: Sequence[tuple[int, int]], *, max_words: int = MAX_FRAME_WORDS) -> list[tuple[int, list[int]]]:
    maximum = _unsigned(int(max_words), MAX_FRAME_WORDS, "max_words")
    if maximum == 0:
        raise ValueError("max_words must be positive")
    result: list[tuple[int, list[int]]] = []
    index = 0
    while index < len(pairs):
        base, value = pairs[index]
        values = [value]
        cursor = index + 1
        while cursor < len(pairs) and len(values) < maximum and pairs[cursor][0] == base + len(values):
            values.append(pairs[cursor][1])
            cursor += 1
        result.append((int(base), values))
        index = cursor
    return result


__all__ = [
    "FrameError",
    "MAX_FRAME_WORDS",
    "OP_READ",
    "OP_WRITE",
    "RESP",
    "ST_OK",
    "SYNC0",
    "SYNC1",
    "coalesce_runs",
    "crc16_ccitt",
    "decode_reply",
    "encode_read",
    "encode_reply",
    "encode_write",
    "reply_frame_len",
]
