"""Tier 3b: Asynchronous Data Channels (Channels 1..8) & Jameson's Alexandrian Vault.

Conforms strictly to Principles of Operation (Rev. 1.5), Chapter 6 (§6.1–§6.2):
  - 8 autonomous DMA Data Channels invoked by C-EXEC via privileged `CHIO As1, #ch`
  - Exact 6-word Channel Control Block (CCB):
      CCB[0] : device_type(3t, 35..33) | unit(6t, 32..27) | op(4t, 26..23) | flags(23t, 22..0)
      CCB[1] : mem_addr(18t, 35..18) | word_count(18t, 17..0)
      CCB[2] : media_pos(36t) (disc sector/word offset, tape record, or silo cartridge ID)
      CCB[3] : provenance_tag(36t) (WEAR compartment & Jameson archival provenance tag)
      CCB[4] : chain_ptr(18t) (0 = end of chain; else address of next 6-word CCB)
      CCB[5] : status(3t, 35..33) | words_transferred(18t, 17..0)
               (+1 OK, 0 BUSY, -1 MEDIA_ERR, -2 PROVENANCE_DENIED)
  - Native Ternary Media Devices:
      +1 : Online Ternary Disc Array (32 UE-modified 3330 spindles, 5-trit-per-8-flux coding)
      +2 : Nine-Track Ternary Tape (32 drives, 1 trit/track, 4 frames/T36 word, READ_BACKWARD)
      +3 : Alexandrian Archival Silo (Ampex TBM + IBM 1360 vault, WEAR provenance enforcement)
      -1 : Oliver SETI Baseband Ring (Channel 8 dedicated ring, 5.0M TBFP-9C samples/sec)
      -2 : Level -8 Digitisation Chute (Jameson microform/OCR OBB-729 vertical feed)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .memory import TernaryMemory
from .ternary import (
    POW3,
    WORD_TRITS,
    char_to_obb729,
    clamp,
    extract_field,
    int_to_trits,
    obb729_to_char,
    pack_trytes_to_words,
    trits_to_int,
    unpack_words_to_trytes,
    wrap,
)
from .tfloat import pack_nonets_to_word, quantize_tbfp
from .vcomplex import VComplex

# Device Type Codes (3-trit field in CCB[0] trits 35..33)
DEV_DISC = 1
DEV_TAPE = 2
DEV_SILO = 3
DEV_SETI_RING = -1
DEV_L8_CHUTE = -2

# Channel Operation Codes (4-trit field in CCB[0] trits 26..23)
OP_READ = 1
OP_WRITE = 2
OP_READ_BACKWARD = 3
OP_SEEK = 4
OP_REWIND = 5
OP_MARK = 6
OP_VERIFY = 7

# Completion Status Codes (3-trit field in CCB[5] trits 35..33)
CH_OK = 1
CH_BUSY = 0
CH_MEDIA_ERR = -1
CH_PROVENANCE_DENIED = -2

# Hardware Geometry & Timing Constants (PoO §6.2)
NUM_CHANNELS = 8
SETI_DEDICATED_CHANNEL = 8
DISC_SPINDLES = 32
DISC_WORDS_PER_SPINDLE = 24_000_000   # 768M T36 words across 32 spindles
DISC_SEEK_NS = 30_000_000             # 30 ms avg seek
DISC_ROT_NS = 8_350_000               # half of 16.7 ms rotation
DISC_NS_PER_WORD = 2_850              # ~350k words/s/channel (~20 MB/s equiv across 8 ch)

TAPE_DRIVES = 32
TAPE_WORDS_PER_SEC = 16_875           # 600 fpi @ 112.5 ips, 4 frames/word
TAPE_NS_PER_WORD = 59_259             # ~59.3 us/word

SILO_MOUNT_NS = 12_000_000_000        # 12 s cartridge mount
SILO_NS_PER_WORD = 4_760              # ~1.5 MB/s equiv (~210k words/s)

SETI_SAMPLES_PER_SEC = 5_000_000      # 5.0M complex TBFP-9C samples/s = 2.5M T36 words/s
SETI_NS_PER_WORD = 400                # 400 ns/word

L8_CHUTE_NS_PER_WORD = 2_000          # shielded vertical optical/tape conduit

MAX_CCB_CHAIN = 64


# ---------------------------------------------------------------------------
# Native Ternary Media Group-Coding Helpers (PoO §6.2)
# ---------------------------------------------------------------------------

def encode_5trit_to_8flux(trits5: Sequence[int]) -> int:
    """Encode 5 balanced trits (243 states) into an 8-bit flux group (0..242).

    Achieves log2(243)/8 = 99.06% recording efficiency on UE-modified 3330 disc spindles.
    """
    if len(trits5) != 5:
        raise ValueError("Expected exactly 5 trits")
    val = trits_to_int(trits5)  # [-121 .. +121]
    return val + 121            # [0 .. 242] fits cleanly inside 8 flux transitions (0..255)


def decode_8flux_to_5trit(flux_byte: int) -> List[int]:
    """Decode an 8-bit flux group (0..242) back into 5 balanced trits."""
    if not (0 <= flux_byte <= 242):
        raise ValueError(f"Invalid 5-trit flux group {flux_byte} (must be 0..242)")
    return int_to_trits(flux_byte - 121, 5)


def encode_word_to_4tape_frames(word: int) -> Tuple[List[int], List[int], List[int], List[int]]:
    """Split one 36-trit word into 4 nine-track tape frames (1 trit/track, 9 trits/frame)."""
    ts = int_to_trits(wrap(word, WORD_TRITS), WORD_TRITS)
    return (ts[0:9], ts[9:18], ts[18:27], ts[27:36])


def decode_4tape_frames_to_word(frames: Sequence[Sequence[int]]) -> int:
    """Reassemble 4 nine-track tape frames (9 trits each) into one T36 word."""
    if len(frames) != 4 or any(len(f) != 9 for f in frames):
        raise ValueError("Expected 4 frames of 9 trits each")
    all_trits: List[int] = []
    for f in frames:
        all_trits.extend(f)
    return trits_to_int(all_trits)


# ---------------------------------------------------------------------------
# 6-Word Channel Control Block (CCB, PoO §6.1)
# ---------------------------------------------------------------------------

@dataclass
class ChannelControlBlock:
    """Exact 6-word (6 x T36) Channel Control Block per PoO §6.1."""
    device_type: int = DEV_DISC
    unit: int = 0
    op: int = OP_READ
    flags: int = 0
    mem_addr: int = 0
    word_count: int = 0
    media_pos: int = 0
    provenance_tag: int = 0
    chain_ptr: int = 0
    status: int = CH_BUSY
    words_transferred: int = 0

    def pack_6_words(self) -> List[int]:
        w0 = (
            wrap(self.device_type, 3) * POW3[33]
            + wrap(self.unit, 6) * POW3[27]
            + wrap(self.op, 4) * POW3[23]
            + wrap(self.flags, 23)
        )
        w1 = wrap(self.mem_addr, 18) * POW3[18] + wrap(self.word_count, 18)
        w2 = wrap(self.media_pos, WORD_TRITS)
        w3 = wrap(self.provenance_tag, WORD_TRITS)
        w4 = wrap(self.chain_ptr, 18)
        w5 = self.pack_status(self.status, self.words_transferred)
        return [w0, w1, w2, w3, w4, w5]

    @staticmethod
    def pack_status(status: int, words_transferred: int) -> int:
        return wrap(status, 3) * POW3[33] + clamp(words_transferred, 18)

    @staticmethod
    def unpack_status(word5: int) -> Tuple[int, int]:
        return (
            extract_field(word5, 33, 3, WORD_TRITS),
            extract_field(word5, 0, 18, WORD_TRITS),
        )

    @classmethod
    def unpack_6_words(cls, words: Sequence[int]) -> "ChannelControlBlock":
        if len(words) != 6:
            raise ValueError("CCB requires exactly 6 T36 words")
        w0, w1, w2, w3, w4, w5 = words
        status, transferred = cls.unpack_status(w5)
        return cls(
            device_type=extract_field(w0, 33, 3, WORD_TRITS),
            unit=extract_field(w0, 27, 6, WORD_TRITS),
            op=extract_field(w0, 23, 4, WORD_TRITS),
            flags=extract_field(w0, 0, 23, WORD_TRITS),
            mem_addr=extract_field(w1, 18, 18, WORD_TRITS),
            word_count=extract_field(w1, 0, 18, WORD_TRITS),
            media_pos=wrap(w2, WORD_TRITS),
            provenance_tag=wrap(w3, WORD_TRITS),
            chain_ptr=extract_field(w4, 0, 18, WORD_TRITS),
            status=status,
            words_transferred=transferred,
        )


# ---------------------------------------------------------------------------
# Tier 3b Device Subsystems
# ---------------------------------------------------------------------------

class TernaryDiscArray:
    """32 UE-modified 3330-class ternary disc spindles (768M T36 words total)."""

    def __init__(self, num_spindles: int = DISC_SPINDLES) -> None:
        self.num_spindles = num_spindles
        self._storage: List[Dict[int, int]] = [{} for _ in range(num_spindles)]
        self.head_pos: List[int] = [0] * num_spindles
        self.total_reads: int = 0
        self.total_writes: int = 0

    def _valid_unit(self, unit: int) -> bool:
        return 0 <= unit < self.num_spindles

    def execute(
        self, ccb: ChannelControlBlock, mem: TernaryMemory
    ) -> Tuple[int, int, int]:
        """Return (status_code, words_transferred, duration_ns)."""
        if not self._valid_unit(ccb.unit):
            return CH_MEDIA_ERR, 0, 0
        pos = ccb.media_pos
        count = ccb.word_count
        if pos < 0 or count < 0 or pos + count > DISC_WORDS_PER_SPINDLE:
            return CH_MEDIA_ERR, 0, 0

        seek_ns = (DISC_SEEK_NS + DISC_ROT_NS) if pos != self.head_pos[ccb.unit] else 0
        spindle = self._storage[ccb.unit]

        if ccb.op == OP_SEEK:
            self.head_pos[ccb.unit] = pos
            return CH_OK, 0, seek_ns

        if ccb.op == OP_WRITE:
            for i in range(count):
                w, _ = mem.read(ccb.mem_addr + i, supervisor=True)
                spindle[pos + i] = wrap(w, WORD_TRITS)
            self.head_pos[ccb.unit] = pos + count
            self.total_writes += count
            return CH_OK, count, seek_ns + count * DISC_NS_PER_WORD

        if ccb.op == OP_READ:
            for i in range(count):
                mem.write(ccb.mem_addr + i, spindle.get(pos + i, 0), supervisor=True)
            self.head_pos[ccb.unit] = pos + count
            self.total_reads += count
            return CH_OK, count, seek_ns + count * DISC_NS_PER_WORD

        if ccb.op == OP_VERIFY:
            for i in range(count):
                w, _ = mem.read(ccb.mem_addr + i, supervisor=True)
                if wrap(w, WORD_TRITS) != spindle.get(pos + i, 0):
                    self.head_pos[ccb.unit] = pos + i
                    return CH_MEDIA_ERR, i, seek_ns + (i + 1) * DISC_NS_PER_WORD
            self.head_pos[ccb.unit] = pos + count
            return CH_OK, count, seek_ns + count * DISC_NS_PER_WORD

        return CH_MEDIA_ERR, 0, 0


@dataclass
class TapeDriveState:
    records: List[Optional[List[int]]] = field(default_factory=list)  # None = Tape Mark
    position: int = 0
    write_protected: bool = False


class NineTrackTernaryTape:
    """32 nine-track ternary tape drives with Setun / IBM 2401 READ_BACKWARD support."""

    def __init__(self, num_drives: int = TAPE_DRIVES) -> None:
        self.num_drives = num_drives
        self.drives: List[TapeDriveState] = [TapeDriveState() for _ in range(num_drives)]
        self.total_words_read: int = 0
        self.total_words_written: int = 0

    def execute(
        self, ccb: ChannelControlBlock, mem: TernaryMemory
    ) -> Tuple[int, int, int]:
        if not (0 <= ccb.unit < self.num_drives):
            return CH_MEDIA_ERR, 0, 0
        drv = self.drives[ccb.unit]

        if ccb.op == OP_REWIND:
            dur = drv.position * 500_000
            drv.position = 0
            return CH_OK, 0, dur

        if ccb.op == OP_SEEK:
            target = ccb.media_pos
            if target < 0 or target > len(drv.records):
                return CH_MEDIA_ERR, 0, 0
            delta = abs(target - drv.position)
            drv.position = target
            return CH_OK, 0, delta * 1_000_000

        if ccb.op == OP_MARK:
            if drv.write_protected:
                return CH_MEDIA_ERR, 0, 0
            if drv.position < len(drv.records):
                drv.records = drv.records[: drv.position]
            drv.records.append(None)
            drv.position += 1
            return CH_OK, 0, 200_000

        if ccb.op == OP_WRITE:
            if drv.write_protected or ccb.word_count < 0:
                return CH_MEDIA_ERR, 0, 0
            rec: List[int] = []
            for i in range(ccb.word_count):
                w, _ = mem.read(ccb.mem_addr + i, supervisor=True)
                rec.append(wrap(w, WORD_TRITS))
            if drv.position < len(drv.records):
                drv.records = drv.records[: drv.position]
            drv.records.append(rec)
            drv.position += 1
            self.total_words_written += len(rec)
            return CH_OK, len(rec), len(rec) * TAPE_NS_PER_WORD

        if ccb.op == OP_READ:
            if drv.position >= len(drv.records):
                return CH_MEDIA_ERR, 0, 0
            rec = drv.records[drv.position]
            drv.position += 1
            if rec is None:
                # Tape mark encountered: 0 words transferred cleanly
                return CH_OK, 0, 200_000
            n = min(len(rec), ccb.word_count) if ccb.word_count > 0 else len(rec)
            for i in range(n):
                mem.write(ccb.mem_addr + i, rec[i], supervisor=True)
            self.total_words_read += n
            return CH_OK, n, n * TAPE_NS_PER_WORD

        if ccb.op == OP_READ_BACKWARD:
            # Authentic Setun / IBM 2401 reverse-direction read:
            # Steps back one record and transfers its words in reverse order into T-DRAM!
            if drv.position <= 0:
                return CH_MEDIA_ERR, 0, 0
            drv.position -= 1
            rec = drv.records[drv.position]
            if rec is None:
                return CH_OK, 0, 200_000
            rev = list(reversed(rec))
            n = min(len(rev), ccb.word_count) if ccb.word_count > 0 else len(rev)
            for i in range(n):
                mem.write(ccb.mem_addr + i, rev[i], supervisor=True)
            self.total_words_read += n
            return CH_OK, n, n * TAPE_NS_PER_WORD

        if ccb.op == OP_VERIFY:
            if drv.position >= len(drv.records):
                return CH_MEDIA_ERR, 0, 0
            rec = drv.records[drv.position]
            drv.position += 1
            if rec is None:
                return CH_MEDIA_ERR, 0, 200_000
            n = min(len(rec), ccb.word_count) if ccb.word_count > 0 else len(rec)
            for i in range(n):
                w, _ = mem.read(ccb.mem_addr + i, supervisor=True)
                if wrap(w, WORD_TRITS) != rec[i]:
                    return CH_MEDIA_ERR, i, (i + 1) * TAPE_NS_PER_WORD
            return CH_OK, n, n * TAPE_NS_PER_WORD

        return CH_MEDIA_ERR, 0, 0


@dataclass
class SiloCartridge:
    cartridge_id: int
    label: str
    words: List[int]
    provenance_compartment: int = 0   # 0 = unrestricted; non-zero requires matching CCB[3]
    read_only: bool = False


class AlexandrianArchivalSilo:
    """Jameson's Alexandrian Vault (~100 GB equiv Ampex TBM + IBM 1360) with WEAR audit."""

    def __init__(self, vcomplex: Optional[VComplex] = None) -> None:
        self.vcomplex = vcomplex
        self.cartridges: Dict[int, SiloCartridge] = {}
        self.mounted_id: Optional[int] = None
        self.audit_log: List[str] = []
        self.provenance_denials: int = 0

    def mount_cartridge(
        self,
        cartridge_id: int,
        words: Sequence[int],
        label: str = "",
        provenance_compartment: int = 0,
        read_only: bool = False,
    ) -> None:
        self.cartridges[cartridge_id] = SiloCartridge(
            cartridge_id=cartridge_id,
            label=label or f"CART-{cartridge_id:04d}",
            words=[wrap(w, WORD_TRITS) for w in words],
            provenance_compartment=provenance_compartment,
            read_only=read_only,
        )

    def execute(
        self, ccb: ChannelControlBlock, mem: TernaryMemory, timestamp_ns: int = 0
    ) -> Tuple[int, int, int]:
        cid = ccb.media_pos
        mount_ns = 0 if self.mounted_id == cid else SILO_MOUNT_NS

        if ccb.op == OP_WRITE:
            existing = self.cartridges.get(cid)
            if existing is not None:
                if existing.read_only:
                    return CH_MEDIA_ERR, 0, 0
                if existing.provenance_compartment != 0 and ccb.provenance_tag != existing.provenance_compartment:
                    self._deny(cid, ccb.provenance_tag, existing.provenance_compartment, "WRITE", timestamp_ns)
                    return CH_PROVENANCE_DENIED, 0, 0
            words = [wrap(mem.read(ccb.mem_addr + i, supervisor=True)[0], WORD_TRITS) for i in range(ccb.word_count)]
            self.cartridges[cid] = SiloCartridge(
                cartridge_id=cid,
                label=existing.label if existing else f"CART-{cid:04d}",
                words=words,
                provenance_compartment=ccb.provenance_tag,
                read_only=False,
            )
            self.mounted_id = cid
            self.audit_log.append(f"WRITE cart={cid} words={len(words)} tag={ccb.provenance_tag} OK")
            return CH_OK, len(words), mount_ns + len(words) * SILO_NS_PER_WORD

        cart = self.cartridges.get(cid)
        if cart is None:
            return CH_MEDIA_ERR, 0, 0

        if cart.provenance_compartment != 0 and ccb.provenance_tag != cart.provenance_compartment:
            self._deny(cid, ccb.provenance_tag, cart.provenance_compartment, "READ/VERIFY", timestamp_ns)
            return CH_PROVENANCE_DENIED, 0, 0

        self.mounted_id = cid
        if ccb.op == OP_SEEK:
            return CH_OK, 0, mount_ns

        n = min(len(cart.words), ccb.word_count) if ccb.word_count > 0 else len(cart.words)
        if ccb.op == OP_READ:
            for i in range(n):
                mem.write(ccb.mem_addr + i, cart.words[i], supervisor=True)
            self.audit_log.append(f"READ cart={cid} ({cart.label}) words={n} tag={ccb.provenance_tag} OK")
            return CH_OK, n, mount_ns + n * SILO_NS_PER_WORD

        if ccb.op == OP_VERIFY:
            for i in range(n):
                w, _ = mem.read(ccb.mem_addr + i, supervisor=True)
                if wrap(w, WORD_TRITS) != cart.words[i]:
                    return CH_MEDIA_ERR, i, mount_ns + (i + 1) * SILO_NS_PER_WORD
            return CH_OK, n, mount_ns + n * SILO_NS_PER_WORD

        return CH_MEDIA_ERR, 0, 0

    def _deny(self, cid: int, presented: int, required: int, op_name: str, timestamp_ns: int) -> None:
        self.provenance_denials += 1
        msg = f"DENIED {op_name} cart={cid}: tag={presented} != required={required}"
        self.audit_log.append(msg)
        if self.vcomplex is not None:
            self.vcomplex.record_provenance_denial(cid, presented, required, timestamp_ns=timestamp_ns)


class SETIBasebandRing:
    """Oliver SETI Baseband Annex dedicated Channel 8 ring (5.0M TBFP-9C samples/sec)."""

    def __init__(self) -> None:
        self._ring: Deque[int] = deque()
        self.total_words_streamed: int = 0
        self.last_block_exponent: int = 8

    def ingest_words(self, words: Sequence[int]) -> None:
        for w in words:
            self._ring.append(wrap(w, WORD_TRITS))

    def ingest_complex_samples(self, signal_complex: np.ndarray) -> int:
        """Quantize complex baseband samples into TBFP-9C (2 samples/T36 word) & enqueue.

        Returns shared 6-trit block exponent E_blk.
        """
        sig = np.asarray(signal_complex, dtype=np.complex128).ravel()
        stacked = np.stack([sig.real, sig.imag], axis=-1)
        q, e_blk = quantize_tbfp(stacked, mant_trits=9)
        self.last_block_exponent = e_blk
        n = sig.size
        for idx in range(0, n, 2):
            r0, i0 = int(q[idx, 0]), int(q[idx, 1])
            r1, i1 = (int(q[idx + 1, 0]), int(q[idx + 1, 1])) if idx + 1 < n else (0, 0)
            self._ring.append(pack_nonets_to_word(r0, i0, r1, i1))
        return e_blk

    @property
    def buffered_words(self) -> int:
        return len(self._ring)

    def execute(
        self, ccb: ChannelControlBlock, mem: TernaryMemory
    ) -> Tuple[int, int, int]:
        if ccb.op != OP_READ or ccb.word_count <= 0:
            return CH_MEDIA_ERR, 0, 0
        n = min(ccb.word_count, len(self._ring))
        if n == 0:
            return CH_MEDIA_ERR, 0, 0
        for i in range(n):
            mem.write(ccb.mem_addr + i, self._ring.popleft(), supervisor=True)
        self.total_words_streamed += n
        return CH_OK, n, n * SETI_NS_PER_WORD


class Level8DigitisationChute:
    """Jameson's Level -8 Microform/OCR vertical feed streaming OBB-729 tryte records."""

    def __init__(self) -> None:
        self._queue: Deque[Tuple[List[int], int]] = deque()  # (words, provenance_tag)
        self.total_records_ingested: int = 0

    def feed_text_record(self, text: str, provenance_tag: int = 0) -> int:
        """Pack an OBB-729 text record (6 trytes per T36 word, 0-terminated) into the chute."""
        trytes = [char_to_obb729(ch) for ch in text] + [0]
        words = pack_trytes_to_words(trytes)
        self._queue.append((words, provenance_tag))
        return len(words)

    @staticmethod
    def decode_words_to_text(words: Sequence[int]) -> str:
        chars: List[str] = []
        for t in unpack_words_to_trytes(words):
            if t == 0:
                return "".join(chars)
            chars.append(obb729_to_char(t))
        return "".join(chars)

    @property
    def pending_records(self) -> int:
        return len(self._queue)

    def execute(
        self, ccb: ChannelControlBlock, mem: TernaryMemory
    ) -> Tuple[int, int, int]:
        if ccb.op != OP_READ or not self._queue:
            return CH_MEDIA_ERR, 0, 0
        words, rec_tag = self._queue.popleft()
        n = min(len(words), ccb.word_count) if ccb.word_count > 0 else len(words)
        for i in range(n):
            mem.write(ccb.mem_addr + i, words[i], supervisor=True)
        # Stamp record provenance tag back into CCB[3] if present
        if rec_tag != 0 and ccb.provenance_tag == 0:
            ccb.provenance_tag = rec_tag
        self.total_records_ingested += 1
        return CH_OK, n, n * L8_CHUTE_NS_PER_WORD


# ---------------------------------------------------------------------------
# 8-Channel Autonomous DMA Controller
# ---------------------------------------------------------------------------

@dataclass
class ChannelState:
    channel_id: int          # 1..8
    busy_until_ns: int = 0
    last_status: int = CH_OK
    last_ccb_addr: int = 0
    total_ops: int = 0
    total_words: int = 0


@dataclass
class PendingChannelJob:
    channel_id: int
    ccb_addr: int
    ready_ns: int
    status: int
    words: int
    commit_fn: Callable[[], None]


class IOChannelSubsystem:
    """Tier 3b 8-Channel DMA Controller linking C-Complex, T-DRAM, and Alexandrian Vault."""

    def __init__(
        self,
        vcomplex: Optional[VComplex] = None,
        async_dma: bool = False,
    ) -> None:
        self.vcomplex = vcomplex
        self.async_dma = async_dma

        self.disc = TernaryDiscArray()
        self.tape = NineTrackTernaryTape()
        self.silo = AlexandrianArchivalSilo(vcomplex=vcomplex)
        self.seti_ring = SETIBasebandRing()
        self.l8_chute = Level8DigitisationChute()

        self.channels: Dict[int, ChannelState] = {
            ch: ChannelState(channel_id=ch) for ch in range(1, NUM_CHANNELS + 1)
        }
        self._pending: List[PendingChannelJob] = []

    def is_busy(self, channel: int, now_ns: int) -> bool:
        st = self.channels.get(channel)
        if st is None:
            return False
        return now_ns < st.busy_until_ns or any(p.channel_id == channel for p in self._pending)

    def advance_to_ns(self, now_ns: int) -> int:
        """Complete any asynchronous DMA jobs whose transfer duration has elapsed."""
        completed = 0
        still_pending: List[PendingChannelJob] = []
        for job in self._pending:
            if job.ready_ns <= now_ns:
                job.commit_fn()
                completed += 1
            else:
                still_pending.append(job)
        self._pending = still_pending
        return completed

    def flush_all(self) -> int:
        """Force completion of all pending async DMA jobs immediately."""
        count = len(self._pending)
        for job in self._pending:
            job.commit_fn()
        self._pending.clear()
        return count

    def channel_status_word(self, channel: int, now_ns: int = 0) -> int:
        """Pack channel status word for MMIO aperture (+100,000,010..017)."""
        st = self.channels.get(channel)
        if st is None:
            return 0
        status = CH_BUSY if self.is_busy(channel, now_ns) else st.last_status
        return ChannelControlBlock.pack_status(status, st.total_words)

    def execute_chio(
        self,
        channel: int,
        ccb_addr: int,
        mem: TernaryMemory,
        now_ns: int = 0,
    ) -> Tuple[int, int, int]:
        """Execute a (possibly chained) 6-word CCB program on Data Channel `1..8`.

        Returns (final_status, total_words_transferred, total_duration_ns).
        """
        if channel not in self.channels:
            return CH_MEDIA_ERR, 0, 0

        ch_state = self.channels[channel]
        cur_addr = ccb_addr
        visited = set()
        total_words = 0
        total_dur_ns = 0
        final_status = CH_OK
        # Check if caller requested async mode via subsystem flag or CCB[0] flag bit 0 (trit 0 == +1)
        cur_time = max(now_ns, ch_state.busy_until_ns)

        for _ in range(MAX_CCB_CHAIN):
            if cur_addr == 0 or cur_addr in visited:
                break
            visited.add(cur_addr)
            raw = [mem.read(cur_addr + i, supervisor=True)[0] for i in range(6)]
            ccb = ChannelControlBlock.unpack_6_words(raw)

            # Enforce Channel 8 dedicated ring for Oliver SETI Baseband Annex (PoO §6.2)
            if ccb.device_type == DEV_SETI_RING and channel != SETI_DEDICATED_CHANNEL:
                status, words, dur_ns = CH_MEDIA_ERR, 0, 0
                mem.write(cur_addr + 5, ChannelControlBlock.pack_status(status, 0), supervisor=True)
                ch_state.last_status = status
                ch_state.last_ccb_addr = cur_addr
                return status, total_words, total_dur_ns

            use_async = self.async_dma or (extract_field(ccb.flags, 0, 1, 23) == 1)

            if use_async:
                # Mark CCB[5] = BUSY immediately; schedule completion when clock advances
                mem.write(cur_addr + 5, ChannelControlBlock.pack_status(CH_BUSY, 0), supervisor=True)
                # Compute duration & stage execution into a deferred commit closure
                status, words, dur_ns = self._dispatch_device(ccb, mem, cur_time, dry_run=False)
                # Re-mark BUSY until ready_ns so polling observes BUSY -> OK transition!
                mem.write(cur_addr + 5, ChannelControlBlock.pack_status(CH_BUSY, 0), supervisor=True)
                ready_ns = cur_time + max(1, dur_ns)
                ch_state.busy_until_ns = ready_ns
                ch_state.last_ccb_addr = cur_addr
                ch_state.last_status = CH_BUSY

                def _make_commit(
                    addr: int = cur_addr,
                    st_code: int = status,
                    w_cnt: int = words,
                    cs: ChannelState = ch_state,
                    m: TernaryMemory = mem,
                ) -> Callable[[], None]:
                    def _commit() -> None:
                        m.write(addr + 5, ChannelControlBlock.pack_status(st_code, w_cnt), supervisor=True)
                        cs.last_status = st_code
                        cs.total_ops += 1
                        cs.total_words += w_cnt
                    return _commit

                self._pending.append(
                    PendingChannelJob(
                        channel_id=channel,
                        ccb_addr=cur_addr,
                        ready_ns=ready_ns,
                        status=status,
                        words=words,
                        commit_fn=_make_commit(),
                    )
                )
                total_words += words
                total_dur_ns += dur_ns
                cur_time = ready_ns
                final_status = CH_BUSY
                if status < 0:
                    break
                cur_addr = ccb.chain_ptr
                continue

            status, words, dur_ns = self._dispatch_device(ccb, mem, cur_time)
            if ccb.device_type == DEV_L8_CHUTE and ccb.provenance_tag != 0:
                mem.write(cur_addr + 3, ccb.provenance_tag, supervisor=True)
            mem.write(cur_addr + 5, ChannelControlBlock.pack_status(status, words), supervisor=True)
            total_words += words
            total_dur_ns += dur_ns
            cur_time += dur_ns
            ch_state.busy_until_ns = cur_time
            ch_state.last_status = status
            ch_state.last_ccb_addr = cur_addr
            ch_state.total_ops += 1
            ch_state.total_words += words
            final_status = status

            if status < 0:
                break
            cur_addr = ccb.chain_ptr

        return final_status, total_words, total_dur_ns

    def _dispatch_device(
        self,
        ccb: ChannelControlBlock,
        mem: TernaryMemory,
        timestamp_ns: int,
        dry_run: bool = False,
    ) -> Tuple[int, int, int]:
        if ccb.device_type == DEV_DISC:
            return self.disc.execute(ccb, mem)
        if ccb.device_type == DEV_TAPE:
            return self.tape.execute(ccb, mem)
        if ccb.device_type == DEV_SILO:
            return self.silo.execute(ccb, mem, timestamp_ns=timestamp_ns)
        if ccb.device_type == DEV_SETI_RING:
            return self.seti_ring.execute(ccb, mem)
        if ccb.device_type == DEV_L8_CHUTE:
            return self.l8_chute.execute(ccb, mem)
        return CH_MEDIA_ERR, 0, 0

    def telemetry_summary(self) -> Dict[str, object]:
        return {
            "channels": {
                ch: {
                    "last_status": st.last_status,
                    "total_ops": st.total_ops,
                    "total_words": st.total_words,
                }
                for ch, st in self.channels.items()
            },
            "disc_reads": self.disc.total_reads,
            "disc_writes": self.disc.total_writes,
            "tape_words_read": self.tape.total_words_read,
            "tape_words_written": self.tape.total_words_written,
            "silo_cartridges": len(self.silo.cartridges),
            "silo_provenance_denials": self.silo.provenance_denials,
            "seti_words_streamed": self.seti_ring.total_words_streamed,
            "l8_records_ingested": self.l8_chute.total_records_ingested,
        }
