"""Tier 3a: Duwamish-I Main Ternary Store (81+9 Bank T-DRAM + Ferrite Vault).

Conforms strictly to Principles of Operation (Rev. 1.5), Chapter 3:
  - 18-trit symmetric physical address space:
      [-1,594,323 .. -177,148] : Supervisor T-DRAM (1,417,176 words)
      [-177,147   .. -1]       : Non-Volatile Dual-Core Ferrite Vault (3^11 words, 600 ns)
      [0]                      : Hardware Null-Guard Word (write trap / user trap)
      [+1         .. +41,629,544] : User & Tensor T-DRAM (41,629,544 words)
      [+100,000,000 .. +100,999,999] : Memory-Mapped Control Apertures (privileged)
  - Total T-DRAM = 3^16 = 43,046,721 T36 words across 81 active + 9 hot-spare banks.
  - Sparse dict-backed storage so a 43M-word address space runs with zero host bloat,
    while optionally storing full 42-trit ECC codewords & injecting trit/erasure faults
    on any physical bank!
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .ecc import ECCDecodeResult, decode_word_42, encode_word_42
from .ternary import WORD_TRITS, wrap

FERRITE_WORDS: int = 3**11              # 177,147
FERRITE_MIN_ADDR: int = -FERRITE_WORDS  # -177,147
FERRITE_MAX_ADDR: int = -1

SUP_TDRAM_MIN: int = -(3**13)           # -1,594,323
SUP_TDRAM_MAX: int = -FERRITE_WORDS - 1 # -177,148
USER_TDRAM_MIN: int = 1
USER_TDRAM_MAX: int = 41_629_544        # 3^16 - (3^13 - 3^11) - 1
TOTAL_TDRAM_WORDS: int = 3**16          # 43,046,721

MMIO_MIN: int = 100_000_000
MMIO_MAX: int = 100_999_999

TDRAM_BANKS_ACTIVE: int = 81            # 3^4
TDRAM_BANKS_SPARE: int = 9              # 3^2
TDRAM_BANKS_TOTAL: int = 90
TDRAM_BANK_WORDS: int = 3**12           # 531,441

TDRAM_CYCLE_NS: int = 240
TDRAM_BURST_NS: int = 60
FERRITE_CYCLE_NS: int = 600


class MemoryProtectionError(Exception):
    """Raised on Trap 2 (PROT) memory protection violation."""


class MemoryDEDError(Exception):
    """Raised on Trap 14 (MEMDED) uncorrectable double-trit or >=4-erasure fault."""


@dataclass
class BankHealth:
    physical_bank: int
    sec_count: int = 0
    erasure_count: int = 0
    ded_count: int = 0
    retired: bool = False


class TernaryMemory:
    """Cycle-approximate, ECC-verified 18-trit physical store for the UE-1000."""

    def __init__(self, strict_ecc: bool = False) -> None:
        # Fast sparse data map: addr -> T36 int (for clean untouched words)
        self._words: Dict[int, int] = {}
        # Full 42-trit ECC storage for words with active faults or when strict_ecc=True
        self._ecc_words: Dict[int, List[int]] = {}
        self._erasures: Dict[int, Set[int]] = {}
        self.strict_ecc: bool = strict_ecc

        # 81 logical banks -> 90 physical banks (0..80 active, 81..89 hot spares)
        self.bank_lut: List[int] = list(range(TDRAM_BANKS_ACTIVE))
        self.spare_banks: List[int] = list(range(TDRAM_BANKS_ACTIVE, TDRAM_BANKS_TOTAL))
        self.bank_health: Dict[int, BankHealth] = {
            b: BankHealth(physical_bank=b) for b in range(TDRAM_BANKS_TOTAL)
        }
        self.mmio: Dict[int, int] = {}

        # Telemetry counters
        self.reads: int = 0
        self.writes: int = 0
        self.ferrite_accesses: int = 0
        self.bank_cutovers: int = 0
        self.last_ecc_result: Optional[ECCDecodeResult] = None

    def classify_address(self, addr: int) -> str:
        if addr == 0:
            return "ZERO_GUARD"
        if FERRITE_MIN_ADDR <= addr <= FERRITE_MAX_ADDR:
            return "FERRITE"
        if SUP_TDRAM_MIN <= addr <= SUP_TDRAM_MAX:
            return "SUP_TDRAM"
        if USER_TDRAM_MIN <= addr <= USER_TDRAM_MAX:
            return "USER_TDRAM"
        if MMIO_MIN <= addr <= MMIO_MAX:
            return "MMIO"
        return "UNMAPPED"

    def logical_bank(self, addr: int) -> int:
        """Map any T-DRAM address to its interleaved logical bank 0..80."""
        return (addr - SUP_TDRAM_MIN) % TDRAM_BANKS_ACTIVE

    def physical_bank(self, addr: int) -> int:
        return self.bank_lut[self.logical_bank(addr)]

    def _check_access(
        self,
        addr: int,
        is_write: bool,
        supervisor: bool,
        base: int = 0,
        limit: int = USER_TDRAM_MAX,
    ) -> Tuple[str, int]:
        """Resolve virtual/relocated user address and enforce protection."""
        if not supervisor:
            if addr <= 0:
                raise MemoryProtectionError(f"User access to protected/null address {addr}")
            phys = base + addr
            if phys > base + limit or phys > USER_TDRAM_MAX:
                raise MemoryProtectionError(
                    f"User address {addr} (phys {phys}) exceeds LIMIT {limit}"
                )
            return "USER_TDRAM", phys

        region = self.classify_address(addr)
        if region == "UNMAPPED":
            raise MemoryProtectionError(f"Unmapped physical address {addr}")
        if region == "ZERO_GUARD" and is_write:
            raise MemoryProtectionError("Write to hardware Zero-Guard M[0] forbidden")
        return region, addr

    def read(
        self,
        addr: int,
        supervisor: bool = True,
        base: int = 0,
        limit: int = USER_TDRAM_MAX,
    ) -> Tuple[int, int]:
        """Read T36 word. Returns (value, latency_ns)."""
        region, phys = self._check_access(addr, is_write=False, supervisor=supervisor, base=base, limit=limit)
        self.reads += 1

        if region == "ZERO_GUARD":
            return 0, TDRAM_CYCLE_NS
        if region == "MMIO":
            return self.mmio.get(phys, 0), TDRAM_CYCLE_NS
        if region == "FERRITE":
            self.ferrite_accesses += 1
            return self._words.get(phys, 0), FERRITE_CYCLE_NS

        # T-DRAM read with [42, 36, 4]_3 ECC + erasure checking
        pbank = self.physical_bank(phys)
        bh = self.bank_health[pbank]

        if phys in self._ecc_words or phys in self._erasures:
            cw = self._ecc_words.get(phys)
            if cw is None:
                cw = encode_word_42(self._words.get(phys, 0))
            er = self._erasures.get(phys, set())
            res = decode_word_42(cw, er)
            self.last_ecc_result = res
            if res.status == "SEC_CORRECTED":
                bh.sec_count += 1
                return res.data_word, TDRAM_CYCLE_NS
            if res.status == "ERASURE_CORRECTED":
                bh.erasure_count += res.erasure_count
                return res.data_word, TDRAM_CYCLE_NS
            if res.status == "UNCORRECTABLE_DED":
                bh.ded_count += 1
                raise MemoryDEDError(
                    f"MEMDED at phys={phys} (bank {pbank}): erasures={res.erasure_count}, syn={res.syndrome}"
                )
            return res.data_word, TDRAM_CYCLE_NS

        return self._words.get(phys, 0), TDRAM_CYCLE_NS

    def write(
        self,
        addr: int,
        value: int,
        supervisor: bool = True,
        base: int = 0,
        limit: int = USER_TDRAM_MAX,
    ) -> int:
        """Write T36 word. Returns latency_ns."""
        region, phys = self._check_access(addr, is_write=True, supervisor=supervisor, base=base, limit=limit)
        self.writes += 1
        val = wrap(value, WORD_TRITS)

        if region == "MMIO":
            self.mmio[phys] = val
            return TDRAM_CYCLE_NS
        if region == "FERRITE":
            self.ferrite_accesses += 1
            self._words[phys] = val
            return FERRITE_CYCLE_NS

        self._words[phys] = val
        if self.strict_ecc or phys in self._ecc_words:
            self._ecc_words[phys] = encode_word_42(val)
        self._erasures.pop(phys, None)
        return TDRAM_CYCLE_NS

    # ------------------------------------------------------------------
    # Fault Injection & Ashby Hot-Spare Bank Scrub-Cutover
    # ------------------------------------------------------------------

    def inject_trit_flip(self, phys: int, codeword_pos: int, delta: int = 1) -> None:
        """Inject a physical trit error (+1 or -1 mod 3) into a 42-trit T-DRAM word."""
        cw = list(self._ecc_words.get(phys, encode_word_42(self._words.get(phys, 0))))
        r = (cw[codeword_pos] + delta) % 3
        cw[codeword_pos] = -1 if r == 2 else r
        self._ecc_words[phys] = cw

    def inject_erasure(self, phys: int, codeword_pos: int) -> None:
        """Inject a UE-TD2000 (1,1) forbidden-state hardware erasure at codeword_pos (0..41)."""
        if phys not in self._ecc_words:
            self._ecc_words[phys] = encode_word_42(self._words.get(phys, 0))
        self._erasures.setdefault(phys, set()).add(codeword_pos)

    def scrub_and_cutover_bank(self, logical_bank: int) -> Tuple[int, int, int]:
        """Ashby Hot-Spare Bank Cutover: scrub all stored words in logical_bank onto a spare bank.

        Returns (old_phys_bank, new_phys_bank, words_scrubbed).
        """
        if not self.spare_banks:
            raise RuntimeError("All 9 hot-spare T-DRAM banks exhausted!")
        old_phys = self.bank_lut[logical_bank]
        new_phys = self.spare_banks.pop(0)

        scrubbed = 0
        # Re-encode every resident word belonging to logical_bank through ECC onto clean new_phys
        addrs = [a for a in set(self._words) | set(self._ecc_words) if self.classify_address(a) in ("SUP_TDRAM", "USER_TDRAM") and self.logical_bank(a) == logical_bank]
        for a in addrs:
            val, _ = self.read(a, supervisor=True)
            self._words[a] = val
            self._ecc_words[a] = encode_word_42(val)
            self._erasures.pop(a, None)
            scrubbed += 1

        self.bank_health[old_phys].retired = True
        self.bank_lut[logical_bank] = new_phys
        self.bank_cutovers += 1
        return old_phys, new_phys, scrubbed
