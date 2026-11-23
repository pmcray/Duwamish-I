"""TRIAD-36 Two-Pass Symbolic Assembler for the Duwamish-I (UE-1000).

Supports:
  - Full Rev. 1.5 ISA: 36-trit Long instructions and 18-trit Dual-Parcel packing (`op1 | op2`)
  - Automatic `| NOP` padding when a single 18-trit parcel occupies a word
  - Rejects illegal Long | Short or Short | Long packing at assembly time
  - Strict range checking on 7-trit `simm` (±1,093) and 22-trit `disp/imm` (±15.69G)
  - Centered bijective XTR `#p, #n` encoding and PC-relative word branch resolution
  - Directives: `.ORG addr`, `.WORD val...`, `.TTD w0..w8`, `.EQU name, val`
  - UE Heptavintimal listing output (`int_to_hept`)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .isa import (
    LONG_OPS,
    MODE_DIRECT,
    MODE_H0,
    MODE_H1,
    MODE_IMM,
    MODE_INDIRECT,
    MODE_NONET,
    MODE_TRYTE,
    MODE_VSTRIDE,
    MODE_VUNIT,
    SHORT0_OPS,
    SHORTT_OPS,
    encode_long,
    encode_short,
    encode_xtr_simm,
    pack_dual_parcel,
)
from .ternary import WORD_TRITS, int_to_hept, wrap

_REG_RE = re.compile(r"^([SAV])([0-8])$", re.IGNORECASE)
_MEM_RE = re.compile(r"^(@)?([^()]+)\(([Aa][0-8])\)$")


class AssemblyError(ValueError):
    """Raised on syntax, range, or illegal parcel-packing errors."""


@dataclass
class AssembledProgram:
    origin: int
    words: Dict[int, int]
    labels: Dict[str, int]
    listing: List[str] = field(default_factory=list)


def _strip_comment(line: str) -> str:
    idx = line.find(";")
    return line[:idx] if idx >= 0 else line


def _parse_reg(tok: str, expected_prefix: Optional[str] = None) -> Optional[int]:
    m = _REG_RE.match(tok.strip())
    if not m:
        return None
    if expected_prefix and m.group(1).upper() != expected_prefix.upper():
        return None
    return int(m.group(2))


def _eval_expr(expr: str, symbols: Dict[str, int]) -> int:
    """Evaluate integer literal, symbol, or simple +/- offset expression."""
    s = expr.strip()
    if s.startswith("#"):
        s = s[1:].strip()
    # Handle +/- expressions cleanly (e.g. LABEL+1, WALD_0-1, -1620, +1)
    tokens = re.findall(r"[+-]?[^+-]+", s)
    if not tokens:
        raise AssemblyError(f"Empty expression: {expr!r}")
    total = 0
    for tok in tokens:
        t = tok.strip()
        sign = 1
        if t.startswith("+"):
            t = t[1:].strip()
        elif t.startswith("-"):
            sign = -1
            t = t[1:].strip()
        if t in symbols:
            val = symbols[t]
        else:
            try:
                val = int(t, 0)
            except ValueError as exc:
                raise AssemblyError(f"Undefined symbol or invalid literal {t!r} in {expr!r}") from exc
        total += sign * val
    return total


def _split_operands(rest: str) -> List[str]:
    if not rest.strip():
        return []
    return [p.strip() for p in rest.split(",") if p.strip()]


def _parse_instr_text(text: str) -> Tuple[str, Optional[str], List[str]]:
    """Parse 'MNEM[.MOD] op1, op2, ...' -> (mnem, suffix, operands)."""
    s = text.strip()
    if not s:
        return "", None, []
    parts = s.split(None, 1)
    head = parts[0].upper()
    rest = parts[1] if len(parts) > 1 else ""
    if "." in head:
        mn, suf = head.split(".", 1)
    else:
        mn, suf = head, None
    return mn, suf, _split_operands(rest)


def _assemble_short_parcel(
    text: str, pc_word: int, symbols: Dict[str, int]
) -> int:
    mn, suf, ops = _parse_instr_text(text)
    if mn in SHORT0_OPS:
        fmt = 0
        sop = SHORT0_OPS[mn]
    elif mn in SHORTT_OPS:
        fmt = -1
        sop = SHORTT_OPS[mn]
    else:
        raise AssemblyError(f"Instruction {mn!r} is not an 18-trit short parcel (cannot pack)")

    rd = rs1 = rs2 = simm = 0

    if mn == "NOP":
        pass
    elif mn in ("ADD", "SUB", "SHF", "AADD", "ASUB"):
        rd = _parse_reg(ops[0])
        rs1 = _parse_reg(ops[1])
        if rd is None or rs1 is None:
            raise AssemblyError(f"{mn} expects register operands: {text}")
        if len(ops) == 3:
            r2 = _parse_reg(ops[2])
            if r2 is not None:
                rs2 = r2
            else:
                simm = _eval_expr(ops[2], symbols)
        elif len(ops) == 4:
            rs2 = _parse_reg(ops[2])
            if rs2 is None:
                raise AssemblyError(f"{mn} 3rd operand must be register: {text}")
            simm = _eval_expr(ops[3], symbols)
        else:
            raise AssemblyError(f"{mn} expects 3 or 4 operands: {text}")
    elif mn in ("MUL", "MULH", "DIV", "MOD", "AND", "OR", "EQV",
                "FAD", "FSB", "FMP", "FDV", "FMA", "F18PK", "FAD18", "FMP18",
                "DADD", "DSUB", "DLSE"):
        rd, rs1, rs2 = _parse_reg(ops[0]), _parse_reg(ops[1]), _parse_reg(ops[2])
        if None in (rd, rs1, rs2):
            raise AssemblyError(f"{mn} expects 3 registers: {text}")
        if len(ops) >= 4:
            simm = _eval_expr(ops[3], symbols)
    elif mn in ("VADD", "VSUB", "VMUL", "VFMA", "VF18OP", "VKLEEN"):
        rd, rs1, rs2 = _parse_reg(ops[0]), _parse_reg(ops[1]), _parse_reg(ops[2])
        if None in (rd, rs1, rs2):
            raise AssemblyError(f"{mn} expects 3 vector registers: {text}")
        if len(ops) >= 4:
            simm = _eval_expr(ops[3], symbols)
    elif mn in ("NEG", "ABS", "A2S", "S2A", "FNEG", "FABS", "FLT", "FIX", "DEXP"):
        rd, rs1 = _parse_reg(ops[0]), _parse_reg(ops[1])
        if None in (rd, rs1):
            raise AssemblyError(f"{mn} expects 2 registers: {text}")
    elif mn in ("F18UP", "VRED"):
        rd, rs1 = _parse_reg(ops[0]), _parse_reg(ops[1])
        if None in (rd, rs1):
            raise AssemblyError(f"{mn} expects 2 registers: {text}")
        if len(ops) >= 3:
            simm = _eval_expr(ops[2], symbols)
    elif mn == "XTR":
        rd, rs1 = _parse_reg(ops[0]), _parse_reg(ops[1])
        if None in (rd, rs1) or len(ops) != 4:
            raise AssemblyError(f"XTR expects Sd, Ss1, #p, #n: {text}")
        p = _eval_expr(ops[2], symbols)
        n = _eval_expr(ops[3], symbols)
        simm = encode_xtr_simm(p, n)
    elif mn in ("TRT", "CMP"):
        rs1 = _parse_reg(ops[0])
        if rs1 is None:
            raise AssemblyError(f"{mn} expects register first operand: {text}")
        if len(ops) == 2:
            r2 = _parse_reg(ops[1])
            if r2 is not None:
                rs2 = r2
            else:
                simm = _eval_expr(ops[1], symbols)
        elif len(ops) == 3:
            rs2 = _parse_reg(ops[1])
            simm = _eval_expr(ops[2], symbols)
    elif mn == "SEL":
        rd, rs1 = _parse_reg(ops[0]), _parse_reg(ops[1])
        if None in (rd, rs1):
            raise AssemblyError(f"SEL expects Sd, Ss1, [#simm]: {text}")
        if len(ops) >= 3:
            simm = _eval_expr(ops[2], symbols)
    elif mn == "TST":
        rs1 = _parse_reg(ops[0])
        if rs1 is None:
            raise AssemblyError(f"TST expects register: {text}")
    elif mn == "FCM":
        rs1, rs2 = _parse_reg(ops[0]), _parse_reg(ops[1])
        if None in (rs1, rs2):
            raise AssemblyError(f"FCM expects 2 registers: {text}")
    elif mn == "DWALD":
        rs1 = _parse_reg(ops[0])
        if rs1 is None:
            raise AssemblyError(f"DWALD expects register first operand: {text}")
        if len(ops) == 2:
            r2 = _parse_reg(ops[1])
            if r2 is not None:
                rs2 = r2
            else:
                simm = _eval_expr(ops[1], symbols)
        elif len(ops) == 3:
            rs2 = _parse_reg(ops[1])
            if rs2 is None:
                raise AssemblyError(f"DWALD 2nd operand must be register: {text}")
            simm = _eval_expr(ops[2], symbols)
        else:
            raise AssemblyError(f"DWALD expects 2 or 3 operands: {text}")
    elif mn in ("BR", "B3", "BN", "BZ", "BP"):
        target = _eval_expr(ops[0], symbols)
        simm = target - pc_word
    elif mn == "BCT":
        rd = _parse_reg(ops[0])
        if rd is None:
            raise AssemblyError(f"BCT expects Address register: {text}")
        target = _eval_expr(ops[1], symbols)
        simm = target - pc_word
    elif mn == "PUSH":
        rs1 = _parse_reg(ops[0])
        if rs1 is None:
            raise AssemblyError(f"PUSH expects register: {text}")
    elif mn == "POP":
        rd = _parse_reg(ops[0])
        if rd is None:
            raise AssemblyError(f"POP expects register: {text}")
    elif mn == "RET":
        simm = _eval_expr(ops[0], symbols) if ops else 0
    elif mn == "VSVL":
        r1 = _parse_reg(ops[0])
        if r1 is not None:
            rs1 = r1
            if len(ops) >= 2:
                simm = _eval_expr(ops[1], symbols)
        else:
            simm = _eval_expr(ops[0], symbols)
    else:
        raise AssemblyError(f"Unhandled short instruction {mn}")

    return encode_short(fmt, sop, rd=rd, rs1=rs1, rs2=rs2, simm=simm)


def _parse_mem_operand(tok: str, symbols: Dict[str, int]) -> Tuple[int, int, int]:
    """Parse 'disp(As2)', '@disp(As2)', or bare 'label' -> (disp, rs2, mode)."""
    s = tok.strip()
    m = _MEM_RE.match(s)
    if m:
        indirect = m.group(1) == "@"
        disp = _eval_expr(m.group(2), symbols)
        rs2 = int(m.group(3)[1])
        return disp, rs2, (MODE_INDIRECT if indirect else MODE_DIRECT)
    if s.startswith("#"):
        return _eval_expr(s, symbols), 0, MODE_IMM
    if s.startswith("@"):
        return _eval_expr(s[1:], symbols), 0, MODE_INDIRECT
    return _eval_expr(s, symbols), 0, MODE_DIRECT


def _assemble_long_word(text: str, symbols: Dict[str, int]) -> int:
    mn, suf, ops = _parse_instr_text(text)
    if mn not in LONG_OPS:
        raise AssemblyError(f"Unknown long instruction {mn!r}")
    op = LONG_OPS[mn]
    rd = rs1 = rs2 = mode = imm = 0

    suf_modes = {
        "H1": MODE_H1, "H0": MODE_H0, "T6": MODE_TRYTE, "T9": MODE_NONET,
        "VUNIT": MODE_VUNIT, "VSTR": MODE_VSTRIDE,
    }

    if mn in ("LD", "LDA", "LEA"):
        rd = _parse_reg(ops[0])
        if rd is None:
            raise AssemblyError(f"{mn} expects destination register: {text}")
        imm, rs2, mode = _parse_mem_operand(ops[1], symbols)
        if suf in suf_modes:
            mode = suf_modes[suf]
            if len(ops) >= 3:
                rs1 = _parse_reg(ops[2]) or 0
    elif mn in ("ST", "STA"):
        rs1 = _parse_reg(ops[0])
        if rs1 is None:
            raise AssemblyError(f"{mn} expects source register: {text}")
        imm, rs2, mode = _parse_mem_operand(ops[1], symbols)
        if suf in suf_modes:
            mode = suf_modes[suf]
            if len(ops) >= 3:
                rd = _parse_reg(ops[2]) or 0
    elif mn == "LDI":
        rd = _parse_reg(ops[0])
        if rd is None:
            raise AssemblyError(f"LDI expects destination register: {text}")
        mode = MODE_IMM
        imm = _eval_expr(ops[1], symbols)
    elif mn in ("VLD", "VST"):
        rd = _parse_reg(ops[0])
        if rd is None:
            raise AssemblyError(f"{mn} expects vector register: {text}")
        imm, rs2, _ = _parse_mem_operand(ops[1], symbols)
        if len(ops) >= 3:
            rs1 = _parse_reg(ops[2])
            if rs1 is None:
                raise AssemblyError(f"{mn} stride must be Address register: {text}")
            mode = MODE_VSTRIDE
        else:
            mode = MODE_VUNIT
    elif mn in ("VGTH", "VSCT"):
        rd = _parse_reg(ops[0])
        imm, rs2, _ = _parse_mem_operand(ops[1], symbols)
        rs1 = _parse_reg(ops[2])
        if None in (rd, rs1):
            raise AssemblyError(f"{mn} expects vector registers: {text}")
    elif mn in ("JMP", "J3"):
        imm, rs2, mode = _parse_mem_operand(ops[0], symbols)
    elif mn == "CALL":
        if len(ops) == 1:
            rd = 0
            imm, rs2, mode = _parse_mem_operand(ops[0], symbols)
        else:
            rd = _parse_reg(ops[0])
            if rd is None:
                raise AssemblyError(f"CALL link operand must be Address register: {text}")
            imm, rs2, mode = _parse_mem_operand(ops[1], symbols)
    elif mn == "TEXEC":
        rs1 = _parse_reg(ops[0])
        rs2 = _parse_reg(ops[1])
        if None in (rs1, rs2):
            raise AssemblyError(f"TEXEC expects As1, Ss2: {text}")
    elif mn == "TWAIT":
        rs2 = _parse_reg(ops[0])
        if rs2 is None:
            raise AssemblyError(f"TWAIT expects Ss2, [#timeout]: {text}")
        imm = _eval_expr(ops[1], symbols) if len(ops) >= 2 else 0
    elif mn == "TSTAT":
        rd = _parse_reg(ops[0])
        rs1 = _parse_reg(ops[1])
        if None in (rd, rs1):
            raise AssemblyError(f"TSTAT expects Sd, As1: {text}")
    elif mn == "SVC":
        imm = _eval_expr(ops[0], symbols)
    elif mn in ("RTI", "HLT"):
        pass
    elif mn == "CHIO":
        rs1 = _parse_reg(ops[0])
        if rs1 is None:
            raise AssemblyError(f"CHIO expects Address register As1, #ch: {text}")
        imm = _eval_expr(ops[1], symbols)
    elif mn == "ALGD":
        rd = _parse_reg(ops[0])
        imm = _eval_expr(ops[1], symbols) if len(ops) >= 2 else 0
    else:
        raise AssemblyError(f"Unhandled long instruction {mn}")

    return encode_long(op, rd=rd, rs1=rs1, rs2=rs2, mode=mode, imm=imm)


def assemble(source: str, origin: int = 100) -> AssembledProgram:
    """Two-pass TRIAD-36 assembler returning word image, symbol table & heptavintimal listing."""
    pc = origin
    symbols: Dict[str, int] = {}
    parsed_items: List[Tuple[int, str, str]] = []  # (addr, kind, payload)

    # Pass 1: Assign addresses & resolve labels / .EQU / .ORG
    for raw_line in source.splitlines():
        line = _strip_comment(raw_line).strip()
        if not line:
            continue

        # Handle one or more labels on a line (e.g. "LOOP: LD ...")
        while ":" in line:
            before, after = line.split(":", 1)
            lbl = before.strip()
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", lbl):
                break
            if lbl in symbols:
                raise AssemblyError(f"Duplicate label {lbl!r}")
            symbols[lbl] = pc
            line = after.strip()
        if not line:
            continue

        upper = line.upper()
        if upper.startswith(".ORG"):
            parts = line.split(None, 1)
            pc = _eval_expr(parts[1], symbols)
            continue
        if upper.startswith(".EQU"):
            parts = line.split(None, 1)[1].split(",", 1)
            symbols[parts[0].strip()] = _eval_expr(parts[1], symbols)
            continue
        if upper.startswith(".WORD") or upper.startswith(".DATA") or upper.startswith(".TTD") or upper.startswith(".CCB"):
            vals = _split_operands(line.split(None, 1)[1])
            for v_str in vals:
                parsed_items.append((pc, "DATA", v_str))
                pc += 1
            continue

        parsed_items.append((pc, "INSTR", line))
        pc += 1

    # Pass 2: Encode words and generate UE heptavintimal listing
    words: Dict[int, int] = {}
    listing: List[str] = []

    for addr, kind, payload in parsed_items:
        if kind == "DATA":
            w = wrap(_eval_expr(payload, symbols), WORD_TRITS)
            words[addr] = w
            listing.append(f"{addr:+10d}  {int_to_hept(w, 36)}  .WORD {payload}")
            continue

        if "|" in payload:
            left, right = [p.strip() for p in payload.split("|", 1)]
            pa = _assemble_short_parcel(left, addr, symbols)
            pb = _assemble_short_parcel(right, addr, symbols)
            w = pack_dual_parcel(pa, pb)
        else:
            mn, _, _ = _parse_instr_text(payload)
            if mn in LONG_OPS:
                w = _assemble_long_word(payload, symbols)
            elif mn in SHORT0_OPS or mn in SHORTT_OPS:
                pa = _assemble_short_parcel(payload, addr, symbols)
                pb = encode_short(0, SHORT0_OPS["NOP"])
                w = pack_dual_parcel(pa, pb)
            else:
                raise AssemblyError(f"Unknown instruction mnemonic {mn!r} at addr {addr}: {payload}")

        words[addr] = w
        listing.append(f"{addr:+10d}  {int_to_hept(w, 36)}  {payload}")

    return AssembledProgram(origin=origin, words=words, labels=symbols, listing=listing)
