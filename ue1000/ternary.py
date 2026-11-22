"""Balanced-ternary primitives, UE Heptavintimal notation, Kleene logic & OBB-729 codec.

Normative constants per Principles of Operation (Rev. 1.5), Chapter 2 & Appendix A:
  - Trit values: -1 ('T'), 0 ('0'), +1 ('1')
  - Triad: 3 trits -> 1 UE Heptavintimal digit ("NOPQRSTUVWXYZ0ABCDEFGHIJKLM")
  - Tryte: 6 trits (±364, 729 states -> OBB-729 character set)
  - Nonet: 9 trits (±9,841, 19,683 states -> TBFP-9 mantissa & SAEW vocab)
  - Half-word (T18): 18 trits (±193,710,244 -> Setun word & short instruction parcel)
  - Accumulator (T27): 27 trits (±3,812,798,742,493)
  - Full word (T36): 36 trits (±75,047,317,648,499,560)
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

TRIT_VALUES: Tuple[int, int, int] = (-1, 0, 1)
TRYTE_TRITS: int = 6
NONET_TRITS: int = 9
HALF_TRITS: int = 18
ACC_TRITS: int = 27
WORD_TRITS: int = 36
ECC_CHECK_TRITS: int = 6
MEM_WORD_TRITS: int = 42

POW3: Tuple[int, ...] = tuple(3**i for i in range(WORD_TRITS * 2 + 4))

WMAX_6: int = (POW3[6] - 1) // 2      # 364
WMAX_7: int = (POW3[7] - 1) // 2      # 1,093
WMAX_9: int = (POW3[9] - 1) // 2      # 9,841
WMAX_12: int = (POW3[12] - 1) // 2    # 265,720
WMAX_18: int = (POW3[18] - 1) // 2    # 193,710,244
WMAX_22: int = (POW3[22] - 1) // 2    # 15,690,529,804
WMAX_27: int = (POW3[27] - 1) // 2    # 3,812,798,742,493
WMAX_36: int = (POW3[36] - 1) // 2    # 75,047,317,648,499,560

TRIT_TO_CHAR = {-1: "T", 0: "0", 1: "1"}
CHAR_TO_TRIT = {"T": -1, "t": -1, "-": -1, "0": 0, "1": 1, "+": 1}

# Index = triad_val + 13  (-13 -> 'N', ..., -1 -> 'Z', 0 -> '0', +1 -> 'A', ..., +13 -> 'M')
HEPT_ALPHABET: str = "NOPQRSTUVWXYZ0ABCDEFGHIJKLM"
HEPT_TO_VAL = {ch: idx - 13 for idx, ch in enumerate(HEPT_ALPHABET)}


def max_val(n: int) -> int:
    """Maximum balanced signed integer in n trits: (3^n - 1) / 2."""
    return (POW3[n] - 1) // 2


def wrap(v: int, n: int = WORD_TRITS) -> int:
    """Reduce integer v modulo 3^n into balanced range [-max_val(n), +max_val(n)]."""
    m = POW3[n]
    h = (m - 1) // 2
    return (v + h) % m - h


def clamp(v: int, n: int = WORD_TRITS) -> int:
    """Saturate integer v into balanced n-trit range."""
    h = (POW3[n] - 1) // 2
    if v > h:
        return h
    if v < -h:
        return -h
    return v


def in_range(v: int, n: int = WORD_TRITS) -> bool:
    h = (POW3[n] - 1) // 2
    return -h <= v <= h


def sign_trit(v: int) -> int:
    """Return balanced sign trit: -1, 0, or +1."""
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


def int_to_trits(v: int, n: int = WORD_TRITS) -> List[int]:
    """Convert signed integer v to n balanced trits, LSB first (index 0 = 3^0)."""
    v = wrap(v, n)
    out: List[int] = []
    for _ in range(n):
        r = v % 3
        if r == 2:
            r = -1
        out.append(r)
        v = (v - r) // 3
    return out


def trits_to_int(ts: Sequence[int]) -> int:
    """Convert LSB-first sequence of balanced trits {-1, 0, +1} to signed integer."""
    v = 0
    for t in reversed(ts):
        v = v * 3 + t
    return v


def int_to_trit_str(v: int, n: int = WORD_TRITS) -> str:
    """Format integer v as MSB-first trit string ('T', '0', '1')."""
    ts = int_to_trits(v, n)
    return "".join(TRIT_TO_CHAR[t] for t in reversed(ts))


def trit_str_to_int(s: str) -> int:
    """Parse MSB-first trit string ('T', '0', '1', underscores ignored) to integer."""
    v = 0
    for ch in s:
        if ch in ("_", " "):
            continue
        v = v * 3 + CHAR_TO_TRIT[ch]
    return v


def int_to_hept(v: int, n: int = WORD_TRITS) -> str:
    """Format integer v in UE Heptavintimal (MSB-first, n must be divisible by 3)."""
    if n % 3 != 0:
        raise ValueError("Heptavintimal width must be a multiple of 3 trits")
    ts = int_to_trits(v, n)
    digits: List[str] = []
    for i in range(n - 3, -1, -3):
        triad = ts[i + 2] * 9 + ts[i + 1] * 3 + ts[i]
        digits.append(HEPT_ALPHABET[triad + 13])
    return "".join(digits)


def hept_to_int(s: str) -> int:
    """Parse MSB-first UE Heptavintimal string to signed integer."""
    v = 0
    for ch in s:
        if ch in ("_", " "):
            continue
        v = v * 27 + HEPT_TO_VAL[ch]
    return v


def hept_to_trits_msb(s: str) -> List[int]:
    """Decode MSB-first UE Heptavintimal string directly to MSB-first trit list."""
    out: List[int] = []
    for ch in s:
        if ch in ("_", " "):
            continue
        val = HEPT_TO_VAL[ch]
        t_lsb = int_to_trits(val, 3)
        out.extend(reversed(t_lsb))
    return out


def shift_trits(v: int, k: int, n: int = WORD_TRITS) -> int:
    """Balanced-ternary arithmetic shift: multiply by 3^k.

    For k < 0, dropping |k| low trits rounds EXACTLY and UNBIASEDLY to the
    nearest integer (ties cannot occur because 3^|k| is odd).
    """
    if k == 0:
        return wrap(v, n)
    if k > 0:
        if k >= n:
            return 0
        return wrap(v * POW3[k], n)
    s = -k
    if s >= n + 1:
        return 0
    p = POW3[s]
    # Exact round-to-nearest integer division by odd p = 3^s:
    return (v + (p - 1) // 2) // p


def extract_field(v: int, pos: int, length: int, width: int = WORD_TRITS) -> int:
    """Extract signed balanced value of trits [pos .. pos+length-1] from v."""
    if pos < 0 or length < 1 or pos + length > width:
        raise ValueError(f"Invalid field [{pos}..{pos+length-1}] for width {width}")
    if pos:
        p = POW3[pos]
        v = (v + (p - 1) // 2) // p
    return wrap(v, length)


def set_field(word: int, pos: int, length: int, value: int, width: int = WORD_TRITS) -> int:
    """Replace trits [pos .. pos+length-1] of word with balanced `value`."""
    ts = int_to_trits(word, width)
    vs = int_to_trits(value, length)
    ts[pos : pos + length] = vs
    return trits_to_int(ts)


def get_trit(v: int, pos: int, width: int = WORD_TRITS) -> int:
    """Return single balanced trit at index `pos` (0 = LSB)."""
    if not (0 <= pos < width):
        return 0
    return extract_field(v, pos, 1, width)


# ---------------------------------------------------------------------------
# Kleene Strong Three-Valued Logic (Tritwise across T36)
# ---------------------------------------------------------------------------

def kleene_and(a: int, b: int, n: int = WORD_TRITS) -> int:
    """Tritwise Kleene strong AND: min(t_a, t_b)."""
    ta = int_to_trits(a, n)
    tb = int_to_trits(b, n)
    return trits_to_int([x if x < y else y for x, y in zip(ta, tb)])


def kleene_or(a: int, b: int, n: int = WORD_TRITS) -> int:
    """Tritwise Kleene strong OR: max(t_a, t_b)."""
    ta = int_to_trits(a, n)
    tb = int_to_trits(b, n)
    return trits_to_int([x if x > y else y for x, y in zip(ta, tb)])


def kleene_eqv(a: int, b: int, n: int = WORD_TRITS) -> int:
    """Tritwise Kleene equivalence / product: t_a * t_b."""
    ta = int_to_trits(a, n)
    tb = int_to_trits(b, n)
    return trits_to_int([x * y for x, y in zip(ta, tb)])


def kleene_not(a: int, n: int = WORD_TRITS) -> int:
    """Tritwise Kleene negation: -t_a (identical to balanced integer negation)."""
    return wrap(-a, n)


def kleene_sel(cond_trit: int, table_val: int) -> int:
    """SEL instruction: return trit index (cond_trit + 1) in {0, 1, 2} of table_val."""
    idx = cond_trit + 1
    return get_trit(table_val, idx, WORD_TRITS)


# ---------------------------------------------------------------------------
# OBB-729 Tryte Character Encoding (Chapter 2.7: -364..+364)
#   0      : NUL
#   +1..+95: Printable ASCII-1968 (' ' = +1 .. '~' = +95)
#   +96..+163: Cyrillic (GOST-1967 subset)
#   +164..+224: Greek & Mathematical operators
#   -1..-32: Control characters (CR=-13, LF=-10, TAB=-9, ETX=-3, etc.)
# ---------------------------------------------------------------------------

_OBB_SPECIAL_TO_TRYTE = {
    "\x00": 0,
    "\n": -10,
    "\r": -13,
    "\t": -9,
    "\x0c": -12,
    "\x07": -7,
    "\x03": -3,
    # Common Greek / Math symbols (+164..+185)
    "α": 164, "β": 165, "γ": 166, "δ": 167, "ε": 168, "λ": 169, "μ": 170,
    "π": 171, "σ": 172, "τ": 173, "ω": 174, "Δ": 175, "Σ": 176, "Ω": 177,
    "∂": 178, "∇": 179, "∫": 180, "≤": 181, "≥": 182, "≠": 183, "∧": 184,
    "∨": 185, "¬": 186, "⊤": 187, "⊥": 188, "∈": 189, "∞": 190, "∑": 191, "∏": 192,
    # APL\36 Iverson array & tensor glyphs (+193..+224)
    "←": 193, "→": 194, "×": 195, "÷": 196, "⌈": 197, "⌊": 198, "⍴": 199, "⍳": 200,
    "⌽": 201, "⊖": 202, "⍉": 203, "⍋": 204, "⍒": 205, "○": 206, "⍟": 207, "⌹": 208,
    "∘": 209, "↑": 210, "↓": 211, "∼": 212, "∣": 213, "≡": 214, "⍲": 215, "⍱": 216,
    "⌿": 217, "⍀": 218, "⍎": 219, "⍕": 220, "⍝": 221, "¯": 222, "⎕": 223, "⋄": 224,
    # Extended math & technical typography (+225..+230)
    "±": 225, "√": 226, "·": 227, "—": 228, "–": 229, "°": 230,
}
# Cyrillic uppercase А..Я (+96..+127) and lowercase а..я (+128..+159)
for _i in range(32):
    _OBB_SPECIAL_TO_TRYTE[chr(0x0410 + _i)] = 96 + _i
    _OBB_SPECIAL_TO_TRYTE[chr(0x0430 + _i)] = 128 + _i

_TRYTE_TO_OBB_SPECIAL = {v: k for k, v in _OBB_SPECIAL_TO_TRYTE.items()}


def char_to_obb729(ch: str) -> int:
    """Encode a single character to a signed 6-trit OBB-729 tryte (-364..+364)."""
    if ch in _OBB_SPECIAL_TO_TRYTE:
        return _OBB_SPECIAL_TO_TRYTE[ch]
    o = ord(ch)
    if 32 <= o <= 126:
        return (o - 32) + 1  # ' ' -> +1 .. '~' -> +95
    if 1 <= o <= 31:
        return -o            # control codes -> negative trytes
    raise ValueError(f"Character {ch!r} (U+{o:04X}) not mapped in OBB-729")


def obb729_to_char(tryte: int) -> str:
    """Decode a signed 6-trit OBB-729 tryte to a Python character."""
    if tryte in _TRYTE_TO_OBB_SPECIAL:
        return _TRYTE_TO_OBB_SPECIAL[tryte]
    if 1 <= tryte <= 95:
        return chr((tryte - 1) + 32)
    if -31 <= tryte <= -1:
        return chr(-tryte)
    return f"<OBB:{tryte:+d}>"


def pack_trytes_to_words(trytes: Sequence[int]) -> List[int]:
    """Pack 6-trit trytes (6 per T36 word, tryte 5 MSB .. tryte 0 LSB) into T36 words."""
    words: List[int] = []
    for i in range(0, len(trytes), 6):
        chunk = list(trytes[i : i + 6])
        while len(chunk) < 6:
            chunk.append(0)
        w = 0
        for tr in chunk:
            w = w * POW3[6] + wrap(tr, 6)
        words.append(w)
    return words


def unpack_words_to_trytes(words: Iterable[int]) -> List[int]:
    """Unpack T36 words into 6-trit trytes (tryte 5 MSB .. tryte 0 LSB)."""
    out: List[int] = []
    for w in words:
        for slot in range(5, -1, -1):
            out.append(extract_field(w, slot * 6, 6, WORD_TRITS))
    return out
