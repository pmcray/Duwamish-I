"""REFAL-36 — Valentin Turchin's Recursive Functions Algorithmic Language for Duwamish-I.

Historically grounded in Turchin's 1968–1971 Moscow Academy publications (Plenum's 1969
English translation in *Cybernetics* Vol. 5, pp. 307–312, and IPM Preprints Nos. 41, 43,
44, 48, 49 brought home from the September 1971 Byurakan CETI Conference) and Refal-5
where-clause & metasystem transition semantics.

Hardware-Backed Execution on UE-1000 (Rev. 1.5):
  - Pre-loads native TRIAD-36 arithmetic, 3-way compare (`CMP` + `SEL`), Kleene logic
    (`AND`/`OR`/`EQV`/`NEG`), and Good Deciban (`DLSE`/`DWALD`) micro-kernels onto
    `machine.cpu0` (origin 2000).
  - Mirrors burial-store (`<Br>`/`<Dg>`/`<Cp>`) and tagged term descriptors into Tier 3a
    `TernaryMemory` (`REFAL_STORE_BASE = 150_000`) using exact 1-trit tags.
  - Full Refal-5 backtracking pattern matching (`s.X`, `t.X`, `e.X`, repeated variables,
    nested structure brackets `( ... )`, where-conditions `, expr : pat` with full
    backtracking into outer `e.` variables, and block conditions `, expr : { ... }`).
  - Constant-stack tail-call optimization (TCO) for loops, sieves, Futamura mixed
    computation (`Mix`), and Byurakan CETI self-describing grammar decoding!
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

from .machine import UE1000Machine
from .ternary import (
    WORD_TRITS,
    char_to_obb729,
    int_to_hept,
    int_to_trits,
    obb729_to_char,
    shift_trits,
    sign_trit,
    trit_str_to_int,
    wrap,
)
from .tfloat import db_to_ticks, ticks_to_db
from .triad36 import AssembledProgram, assemble


class Refal36Error(Exception):
    """Parse, pattern, or runtime error in REFAL-36."""


# ---------------------------------------------------------------------------
# Pre-Assembled Native TRIAD-36 Hardware ALU Kernel for REFAL-36
# ---------------------------------------------------------------------------

_REFAL_KERNEL_ASM = """
; === REFAL-36 Native TRIAD-36 Hardware ALU, Kleene & Good-Wald Kernel ===
_REF_ADD:
    ADD S1, S1, S2
    HLT
_REF_SUB:
    SUB S1, S1, S2
    HLT
_REF_MUL:
    MUL S1, S1, S2
    HLT
_REF_DIV:
    DIV S1, S1, S2
    HLT
_REF_MOD:
    MOD S1, S1, S2
    HLT
_REF_CMP:
    CMP S1, S2
    SEL S1, S0, #8
    HLT
_REF_KAND:
    AND S1, S1, S2
    HLT
_REF_KOR:
    OR S1, S1, S2
    HLT
_REF_KEQV:
    EQV S1, S1, S2
    HLT
_REF_KNOT:
    NEG S1, S1
    HLT
_REF_DLSE:
    DLSE S1, S1, S2
    HLT
_REF_DWALD:
    DWALD S1, S2, #0
    SEL S1, S0, #8
    HLT
"""

_REFAL_KERNEL_ORIGIN = 2000
_REFAL_STORE_BASE = 150_000


# ---------------------------------------------------------------------------
# Refal Ground Data Model (Immutable Value Terms & Expressions)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CharSym:
    ch: str


@dataclass(frozen=True)
class WordSym:
    name: str


@dataclass(frozen=True)
class NumSym:
    val: int


@dataclass(frozen=True)
class Bracket:
    items: Tuple["RefalTerm", ...]


RefalTerm = Union[CharSym, WordSym, NumSym, Bracket]
RefalExpr = Tuple[RefalTerm, ...]


def format_refal_expr(expr: RefalExpr, readable: bool = True) -> str:
    """Format a Refal ground expression for `<Prout>` / `<Print>` output."""
    parts: List[str] = []
    i = 0
    n = len(expr)
    while i < n:
        t = expr[i]
        if isinstance(t, CharSym):
            chars = []
            while i < n and isinstance(expr[i], CharSym):
                chars.append(expr[i].ch)  # type: ignore[union-attr]
                i += 1
            s = "".join(chars)
            parts.append(s if readable else f"'{s}'")
            continue
        if isinstance(t, WordSym):
            parts.append(t.name)
        elif isinstance(t, NumSym):
            parts.append(str(t.val))
        elif isinstance(t, Bracket):
            parts.append("(" + format_refal_expr(t.items, readable=readable) + ")")
        i += 1
    # Join cleanly: adjacent chars already coalesced; separate words/nums with space when needed
    out: List[str] = []
    for idx, p in enumerate(parts):
        if idx > 0:
            prev_t = _last_term_before(expr, idx)
            # Add space between word/num tokens, keep clean around chars/brackets
            if out and not out[-1].endswith((" ", "(", "\n")) and not p.startswith((" ", ")", ",", ";", ":", ".", "!", "?")):
                # Check if either side is a raw CharSym run (which already carries its own spaces!)
                out.append(" ")
        out.append(p)
    # Better exact formatting matching classic Refal-5 Prout:
    return _format_refal_exact(expr)


def _last_term_before(expr: RefalExpr, idx: int) -> Optional[RefalTerm]:
    return expr[idx - 1] if 0 < idx <= len(expr) else None


def _format_refal_exact(expr: RefalExpr) -> str:
    out: List[str] = []
    prev_kind = ""  # "C", "W", "N", "B"
    for t in expr:
        if isinstance(t, CharSym):
            out.append(t.ch)
            prev_kind = "C"
        elif isinstance(t, WordSym):
            if prev_kind in ("W", "N", "B"):
                out.append(" ")
            out.append(t.name)
            prev_kind = "W"
        elif isinstance(t, NumSym):
            if prev_kind in ("W", "N", "B"):
                out.append(" ")
            out.append(str(t.val))
            prev_kind = "N"
        elif isinstance(t, Bracket):
            if prev_kind in ("W", "N"):
                out.append(" ")
            out.append("(" + _format_refal_exact(t.items) + ")")
            prev_kind = "B"
    return "".join(out)


# ---------------------------------------------------------------------------
# AST Nodes for Patterns, Result Expressions, Conditions & Functions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PatVar:
    vtype: str      # 's', 't', 'e'
    name: str


@dataclass(frozen=True)
class PatBracket:
    items: Tuple["PatNode", ...]


PatNode = Union[CharSym, WordSym, NumSym, PatVar, PatBracket]


@dataclass(frozen=True)
class ResBracket:
    items: Tuple["ResNode", ...]


@dataclass(frozen=True)
class ResCall:
    func: str
    args: Tuple["ResNode", ...]


ResNode = Union[CharSym, WordSym, NumSym, PatVar, ResBracket, ResCall]


@dataclass(frozen=True)
class CondMatch:
    expr: Tuple[ResNode, ...]
    pat: Tuple[PatNode, ...]


@dataclass(frozen=True)
class CondBlock:
    expr: Tuple[ResNode, ...]
    sentences: Tuple["RefalSentence", ...]


@dataclass(frozen=True)
class RefalSentence:
    pat: Tuple[PatNode, ...]
    conds: Tuple[Union[CondMatch, CondBlock], ...]
    rhs: Optional[Tuple[ResNode, ...]]  # None if terminated by a CondBlock


@dataclass
class RefalFunction:
    name: str
    is_entry: bool
    sentences: Tuple[RefalSentence, ...]


@dataclass
class RefalProgram:
    functions: Dict[str, RefalFunction]
    entry_func: str
    data_cards: List[str]


# ---------------------------------------------------------------------------
# Tokenizer & Recursive-Descent Parser (Classic Refal-5 + $END Card Deck)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RTok:
    kind: str
    val: object
    line: int


def tokenize_refal(source: str) -> Tuple[List[RTok], List[str]]:
    """Tokenize REFAL-36 source; splits off card deck after `$END`."""
    code_lines: List[str] = []
    data_cards: List[str] = []
    in_data = False
    for raw in source.splitlines():
        if in_data:
            data_cards.append(raw.rstrip("\r\n"))
            continue
        if raw.strip().upper().startswith("$END"):
            in_data = True
            continue
        # Lines starting with '*' in column 1 (or after leading spaces) are comments
        if raw.startswith("*"):
            code_lines.append("")
            continue
        code_lines.append(raw)

    text = "\n".join(code_lines)
    toks: List[RTok] = []
    i = 0
    n = len(text)
    line = 1

    while i < n:
        ch = text[i]
        if ch == "\n":
            line += 1
            i += 1
            continue
        if ch in " \t\r":
            i += 1
            continue
        # Block comment /* ... */
        if text[i : i + 2] == "/*":
            end = text.find("*/", i + 2)
            if end < 0:
                raise Refal36Error(f"Line {line}: Unterminated /* ... */ comment")
            line += text[i : end + 2].count("\n")
            i = end + 2
            continue
        # Line comment starting with * at line beginning (indented)
        if ch == "*" and (i == 0 or text[i - 1] == "\n"):
            nl = text.find("\n", i)
            i = n if nl < 0 else nl
            continue
        # Directive: $ENTRY, $EXTERN, $EXTRN
        if ch == "$":
            j = i + 1
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            toks.append(RTok("DIR", text[i:j].upper(), line))
            i = j
            continue
        # Character string '...' (sequence of CharSym)
        if ch == "'":
            j = i + 1
            chars: List[str] = []
            while j < n:
                c = text[j]
                if c == "\\" and j + 1 < n:
                    nxt = text[j + 1]
                    esc = {"n": "\n", "t": "\t", "r": "\r", "'": "'", "\\": "\\", '"': '"'}.get(nxt, nxt)
                    chars.append(esc)
                    j += 2
                    continue
                if c == "'":
                    if j + 1 < n and text[j + 1] == "'":
                        chars.append("'")
                        j += 2
                        continue
                    break
                if c == "\n":
                    line += 1
                chars.append(c)
                j += 1
            if j >= n:
                raise Refal36Error(f"Line {line}: Unterminated character string '...'")
            toks.append(RTok("CHARS", "".join(chars), line))
            i = j + 1
            continue
        # Compound symbol "..." (single WordSym!)
        if ch == '"':
            j = i + 1
            chars = []
            while j < n:
                c = text[j]
                if c == "\\" and j + 1 < n:
                    nxt = text[j + 1]
                    chars.append({"n": "\n", "t": "\t", '"': '"', "\\": "\\"}.get(nxt, nxt))
                    j += 2
                    continue
                if c == '"':
                    break
                if c == "\n":
                    line += 1
                chars.append(c)
                j += 1
            if j >= n:
                raise Refal36Error(f"Line {line}: Unterminated compound symbol \"...\"")
            toks.append(RTok("WORD", "".join(chars), line))
            i = j + 1
            continue
        # Signed / balanced-ternary integer literal: -42, 42, 3t1T01
        if ch.isdigit() or (ch == "-" and i + 1 < n and text[i + 1].isdigit()):
            j = i
            neg = False
            if text[j] == "-":
                neg = True
                j += 1
            if text[j : j + 2].lower() == "3t":
                j += 2
                k = j
                while k < n and text[k] in "10Tt_":
                    k += 1
                v = trit_str_to_int(text[j:k])
                toks.append(RTok("NUM", wrap(-v if neg else v, WORD_TRITS), line))
                i = k
                continue
            k = j
            while k < n and text[k].isdigit():
                k += 1
            v = int(text[j:k])
            toks.append(RTok("NUM", wrap(-v if neg else v, WORD_TRITS), line))
            i = k
            continue
        # Variable s.X, t.X, e.X
        if ch in "steSTE" and i + 1 < n and text[i + 1] == ".":
            vt = ch.lower()
            j = i + 2
            while j < n and (text[j].isalnum() or text[j] in "_-"):
                j += 1
            if j == i + 2:
                raise Refal36Error(f"Line {line}: Empty variable name after `{vt}.`")
            toks.append(RTok("VAR", (vt, text[i + 2 : j]), line))
            i = j
            continue
        # Word / identifier (supports Latin, digits, `_`, `-`, and Cyrillic!)
        if ch.isalpha() or ch == "_":
            j = i + 1
            while j < n and (text[j].isalnum() or text[j] in "_-"):
                j += 1
            toks.append(RTok("WORD", text[i:j], line))
            i = j
            continue
        if ch in "{}()<>:,;=":
            toks.append(RTok("PUNCT", ch, line))
            i += 1
            continue
        raise Refal36Error(f"Line {line}: Unexpected character {ch!r}")

    toks.append(RTok("EOF", None, line))
    return toks, data_cards


class _RefalParser:
    def __init__(self, toks: List[RTok], data_cards: List[str]) -> None:
        self.toks = toks
        self.pos = 0
        self.data_cards = data_cards

    def peek(self) -> RTok:
        return self.toks[self.pos]

    def match(self, kind: str, val: object = None) -> Optional[RTok]:
        t = self.peek()
        if t.kind == kind and (val is None or t.val == val):
            self.pos += 1
            return t
        return None

    def expect(self, kind: str, val: object = None) -> RTok:
        t = self.peek()
        if t.kind != kind or (val is not None and t.val != val):
            raise Refal36Error(f"Line {t.line}: Expected {kind} {val or ''}, got {t.kind}({t.val!r})")
        self.pos += 1
        return t

    def parse_program(self) -> RefalProgram:
        funcs: Dict[str, RefalFunction] = {}
        first_entry: Optional[str] = None

        while self.peek().kind != "EOF":
            is_entry = False
            if self.peek().kind == "DIR":
                d = str(self.peek().val)
                self.pos += 1
                if d in ("$EXTERN", "$EXTRN"):
                    while self.peek().kind != "EOF" and not (self.peek().kind == "PUNCT" and self.peek().val == ";"):
                        self.pos += 1
                    self.match("PUNCT", ";")
                    continue
                if d == "$ENTRY":
                    is_entry = True
                else:
                    raise Refal36Error(f"Line {self.peek().line}: Unknown directive {d}")

            name_tok = self.expect("WORD")
            fname = str(name_tok.val)
            self.expect("PUNCT", "{")
            sents = self._parse_sentences()
            self.expect("PUNCT", "}")
            self.match("PUNCT", ";")  # optional trailing semicolon
            funcs[fname] = RefalFunction(name=fname, is_entry=is_entry, sentences=tuple(sents))
            if is_entry and first_entry is None:
                first_entry = fname

        entry = "Go" if "Go" in funcs else (first_entry or next(iter(funcs.keys()), ""))
        if not entry:
            raise Refal36Error("REFAL-36 program defines no functions")
        return RefalProgram(functions=funcs, entry_func=entry, data_cards=self.data_cards)

    def _parse_sentences(self) -> List[RefalSentence]:
        sents: List[RefalSentence] = []
        while self.peek().kind != "EOF" and not (self.peek().kind == "PUNCT" and self.peek().val == "}"):
            pat = self._parse_pattern(stop_punct={"=", ",", ";", "}"})
            conds: List[Union[CondMatch, CondBlock]] = []
            rhs: Optional[Tuple[ResNode, ...]] = None

            while self.match("PUNCT", ","):
                cond_expr = self._parse_result(stop_punct={":"})
                self.expect("PUNCT", ":")
                if self.match("PUNCT", "{"):
                    sub_sents = self._parse_sentences()
                    self.expect("PUNCT", "}")
                    conds.append(CondBlock(expr=cond_expr, sentences=tuple(sub_sents)))
                    # Block terminates sentence!
                    break
                cond_pat = self._parse_pattern(stop_punct={"=", ",", ";", "}"})
                conds.append(CondMatch(expr=cond_expr, pat=cond_pat))

            if self.match("PUNCT", "="):
                rhs = self._parse_result(stop_punct={";", "}"})

            self.match("PUNCT", ";")
            sents.append(RefalSentence(pat=pat, conds=tuple(conds), rhs=rhs))
        return sents

    def _parse_pattern(self, stop_punct: set) -> Tuple[PatNode, ...]:
        nodes: List[PatNode] = []
        while self.peek().kind != "EOF":
            t = self.peek()
            if t.kind == "PUNCT" and t.val in stop_punct:
                break
            if t.kind == "CHARS":
                self.pos += 1
                for ch in str(t.val):
                    nodes.append(CharSym(ch))
            elif t.kind == "WORD":
                self.pos += 1
                nodes.append(WordSym(str(t.val)))
            elif t.kind == "NUM":
                self.pos += 1
                nodes.append(NumSym(int(t.val)))
            elif t.kind == "VAR":
                self.pos += 1
                vt, vn = t.val  # type: ignore[misc]
                nodes.append(PatVar(vt, vn))
            elif t.kind == "PUNCT" and t.val == "(":
                self.pos += 1
                inner = self._parse_pattern(stop_punct={")"})
                self.expect("PUNCT", ")")
                nodes.append(PatBracket(inner))
            else:
                raise Refal36Error(f"Line {t.line}: Unexpected token {t.kind}({t.val!r}) in pattern")
        return tuple(nodes)

    def _parse_result(self, stop_punct: set) -> Tuple[ResNode, ...]:
        nodes: List[ResNode] = []
        while self.peek().kind != "EOF":
            t = self.peek()
            if t.kind == "PUNCT" and t.val in stop_punct:
                break
            if t.kind == "CHARS":
                self.pos += 1
                for ch in str(t.val):
                    nodes.append(CharSym(ch))
            elif t.kind == "WORD":
                self.pos += 1
                nodes.append(WordSym(str(t.val)))
            elif t.kind == "NUM":
                self.pos += 1
                nodes.append(NumSym(int(t.val)))
            elif t.kind == "VAR":
                self.pos += 1
                vt, vn = t.val  # type: ignore[misc]
                nodes.append(PatVar(vt, vn))
            elif t.kind == "PUNCT" and t.val == "(":
                self.pos += 1
                inner = self._parse_result(stop_punct={")"})
                self.expect("PUNCT", ")")
                nodes.append(ResBracket(inner))
            elif t.kind == "PUNCT" and t.val == "<":
                self.pos += 1
                fn_tok = self.expect("WORD")
                arg = self._parse_result(stop_punct={">"})
                self.expect("PUNCT", ">")
                nodes.append(ResCall(str(fn_tok.val), arg))
            else:
                raise Refal36Error(f"Line {t.line}: Unexpected token {t.kind}({t.val!r}) in result")
        return tuple(nodes)


def parse_refal(source: str) -> RefalProgram:
    toks, data_cards = tokenize_refal(source)
    return _RefalParser(toks, data_cards).parse_program()


# ---------------------------------------------------------------------------
# Backtracking Pattern Matcher
# ---------------------------------------------------------------------------

Env = Dict[Tuple[str, str], RefalExpr]


def _match_seq(
    pat: Tuple[PatNode, ...],
    p_idx: int,
    expr: RefalExpr,
    e_idx: int,
    env: Env,
) -> Iterator[Env]:
    """Yield every valid variable binding matching pat[p_idx:] against expr[e_idx:]."""
    if p_idx == len(pat):
        if e_idx == len(expr):
            yield env
        return

    node = pat[p_idx]

    # Literal symbol match
    if isinstance(node, (CharSym, WordSym, NumSym)):
        if e_idx < len(expr) and expr[e_idx] == node:
            yield from _match_seq(pat, p_idx + 1, expr, e_idx + 1, env)
        return

    # Nested structure bracket match
    if isinstance(node, PatBracket):
        if e_idx < len(expr) and isinstance(expr[e_idx], Bracket):
            inner_expr = expr[e_idx].items  # type: ignore[union-attr]
            for env2 in _match_seq(node.items, 0, inner_expr, 0, env):
                yield from _match_seq(pat, p_idx + 1, expr, e_idx + 1, env2)
        return

    # Variable match: s.X, t.X, e.X
    if isinstance(node, PatVar):
        key = (node.vtype, node.name)
        if node.vtype == "s":
            if e_idx < len(expr) and not isinstance(expr[e_idx], Bracket):
                term = (expr[e_idx],)
                if key in env:
                    if env[key] == term:
                        yield from _match_seq(pat, p_idx + 1, expr, e_idx + 1, env)
                else:
                    nenv = dict(env)
                    nenv[key] = term
                    yield from _match_seq(pat, p_idx + 1, expr, e_idx + 1, nenv)
            return

        if node.vtype == "t":
            if e_idx < len(expr):
                term = (expr[e_idx],)
                if key in env:
                    if env[key] == term:
                        yield from _match_seq(pat, p_idx + 1, expr, e_idx + 1, env)
                else:
                    nenv = dict(env)
                    nenv[key] = term
                    yield from _match_seq(pat, p_idx + 1, expr, e_idx + 1, nenv)
            return

        # Open e-variable: shortest-first backtracking (or deterministic if already bound!)
        if key in env:
            bound = env[key]
            blen = len(bound)
            if e_idx + blen <= len(expr) and expr[e_idx : e_idx + blen] == bound:
                yield from _match_seq(pat, p_idx + 1, expr, e_idx + blen, env)
            return

        # Optimization: if e.X is the LAST node in pattern, it must consume the rest!
        if p_idx == len(pat) - 1:
            nenv = dict(env)
            nenv[key] = expr[e_idx:]
            yield nenv
            return

        max_len = len(expr) - e_idx
        for length in range(0, max_len + 1):
            nenv = dict(env)
            nenv[key] = expr[e_idx : e_idx + length]
            yield from _match_seq(pat, p_idx + 1, expr, e_idx + length, nenv)
        return


# ---------------------------------------------------------------------------
# REFAL-36 Virtual Machine (Backed by UE-1000 CComplexCPU & TernaryMemory)
# ---------------------------------------------------------------------------

@dataclass
class RefalRunResult:
    result: RefalExpr
    result_str: str
    output: List[str]
    steps: int
    cpu_cycles: int
    hw_alu_calls: int
    store_ops: int


class Refal36VM:
    """Executes REFAL-36 programs with native UE-1000 T36 hardware ALU & T-DRAM store mirror."""

    def __init__(
        self,
        program: RefalProgram,
        machine: Optional[UE1000Machine] = None,
        cards: Optional[Sequence[str]] = None,
        max_steps: int = 200_000,
    ) -> None:
        self.program = program
        self.machine = machine if machine is not None else UE1000Machine()
        self.kernels: AssembledProgram = assemble(_REFAL_KERNEL_ASM, origin=_REFAL_KERNEL_ORIGIN)
        self.machine.load_program(self.kernels, cpu_id=0)

        self.cards: List[str] = list(cards) if cards is not None else list(program.data_cards)
        self.card_pos: int = 0
        self.max_steps: int = max_steps

        self.steps: int = 0
        self.hw_alu_calls: int = 0
        self.store_ops: int = 0
        self.output: List[str] = []
        # Refal burial store: key_expr -> stack of buried values (LIFO per Refal-5!)
        self.store: Dict[RefalExpr, List[RefalExpr]] = {}
        self._mem_cursor: int = _REFAL_STORE_BASE

    def _hw_alu(self, entry: str, a: int, b: int = 0) -> int:
        cpu = self.machine.cpu0
        cpu.pc = self.kernels.labels[entry]
        cpu.halted = False
        cpu.s[1] = wrap(a, WORD_TRITS)
        cpu.s[2] = wrap(b, WORD_TRITS)
        cpu.run(max_steps=100)
        self.hw_alu_calls += 1
        return cpu.s[1]

    def _mirror_store_write(self, tag: int, payload: int) -> None:
        """Record tagged burial/heap word into Tier 3a TernaryMemory."""
        self.machine.mem.write(self._mem_cursor, wrap(3 * payload + tag, WORD_TRITS), supervisor=True)
        self._mem_cursor += 1
        self.store_ops += 1

    def run(self, entry: Optional[str] = None, arg: RefalExpr = ()) -> RefalRunResult:
        fn = entry or self.program.entry_func
        res = self._call_func(fn, arg)
        return RefalRunResult(
            result=res,
            result_str=format_refal_expr(res),
            output=list(self.output),
            steps=self.steps,
            cpu_cycles=self.machine.cpu0.cycles,
            hw_alu_calls=self.hw_alu_calls,
            store_ops=self.store_ops,
        )

    def _eval_res_nodes(self, nodes: Tuple[ResNode, ...], env: Env) -> RefalExpr:
        out: List[RefalTerm] = []
        for nd in nodes:
            if isinstance(nd, (CharSym, WordSym, NumSym)):
                out.append(nd)
            elif isinstance(nd, PatVar):
                key = (nd.vtype, nd.name)
                if key not in env:
                    raise Refal36Error(f"Unbound variable {nd.vtype}.{nd.name} in result")
                out.extend(env[key])
            elif isinstance(nd, ResBracket):
                out.append(Bracket(self._eval_res_nodes(nd.items, env)))
            elif isinstance(nd, ResCall):
                arg_val = self._eval_res_nodes(nd.args, env)
                out.extend(self._call_func(nd.func, arg_val))
        return tuple(out)

    def _match_conds(
        self,
        conds: Tuple[Union[CondMatch, CondBlock], ...],
        c_idx: int,
        env: Env,
    ) -> Iterator[Tuple[Env, Optional[Tuple[ResNode, ...]]]]:
        """Evaluate conditions in order; backtracks into earlier matches if any fails!"""
        if c_idx == len(conds):
            yield env, None
            return
        cond = conds[c_idx]
        if isinstance(cond, CondMatch):
            val = self._eval_res_nodes(cond.expr, env)
            for env2 in _match_seq(cond.pat, 0, val, 0, env):
                yield from self._match_conds(conds, c_idx + 1, env2)
            return
        if isinstance(cond, CondBlock):
            val = self._eval_res_nodes(cond.expr, env)
            for sent in cond.sentences:
                for env2 in _match_seq(sent.pat, 0, val, 0, env):
                    for env3, blk_rhs in self._match_conds(sent.conds, 0, env2):
                        yield env3, (blk_rhs if blk_rhs is not None else sent.rhs)
                        return
            return

    def _call_func(self, fname: str, arg: RefalExpr) -> RefalExpr:
        """Execute `<fname arg>` with constant-stack tail-call optimization (TCO)."""
        cur_fn = fname
        cur_arg = arg

        while True:
            self.steps += 1
            if self.steps > self.max_steps:
                raise Refal36Error(f"Step limit exceeded ({self.max_steps:,} steps) in <{cur_fn} ...>")

            # Check built-in functions first (or if not overridden by user)
            if cur_fn not in self.program.functions:
                return self._call_builtin(cur_fn, cur_arg)

            fn_def = self.program.functions[cur_fn]
            matched_env: Optional[Env] = None
            matched_rhs: Optional[Tuple[ResNode, ...]] = None

            for sent in fn_def.sentences:
                found = False
                for env in _match_seq(sent.pat, 0, cur_arg, 0, {}):
                    for env_final, blk_rhs in self._match_conds(sent.conds, 0, env):
                        matched_env = env_final
                        matched_rhs = blk_rhs if blk_rhs is not None else sent.rhs
                        found = True
                        break
                    if found:
                        break
                if found:
                    break

            if matched_env is None or matched_rhs is None:
                raise Refal36Error(
                    f"Recognition impossible (no matching sentence) in <{cur_fn} {format_refal_expr(cur_arg)}>"
                )

            # Tail-call optimization: RHS is a single <NextFn arg>!
            if len(matched_rhs) == 1 and isinstance(matched_rhs[0], ResCall):
                tc = matched_rhs[0]
                cur_arg = self._eval_res_nodes(tc.args, matched_env)
                cur_fn = tc.func
                continue

            return self._eval_res_nodes(matched_rhs, matched_env)

    # ------------------------------------------------------------------
    # Standard Refal-5 + Duwamish-I Hardware Built-in Functions
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_two_nums(arg: RefalExpr, fn: str) -> Tuple[int, int]:
        """Accept classic Refal `s.1 s.2` OR `(s.1) s.2` / `s.1 (s.2)` macrodigit parens."""
        nums: List[int] = []
        for t in arg:
            if isinstance(t, NumSym):
                nums.append(t.val)
            elif isinstance(t, Bracket) and len(t.items) == 1 and isinstance(t.items[0], NumSym):
                nums.append(t.items[0].val)
            elif isinstance(t, Bracket) and len(t.items) == 2 and t.items[0] == CharSym("-") and isinstance(t.items[1], NumSym):
                nums.append(-t.items[1].val)
            elif t == CharSym("-") and len(nums) < 2:
                nums.append(None)  # type: ignore[arg-type]  # pending minus sign
            else:
                raise Refal36Error(f"<{fn}> expects two numbers, got {format_refal_expr(arg)!r}")
        # Collapse any pending unary minus markers
        clean: List[int] = []
        neg_next = False
        for x in nums:
            if x is None:
                neg_next = True
            else:
                clean.append(-x if neg_next else x)
                neg_next = False
        if len(clean) != 2:
            raise Refal36Error(f"<{fn}> expects 2 numbers, got {len(clean)}")
        return clean[0], clean[1]

    @staticmethod
    def _to_trit(term: RefalTerm, fn: str) -> int:
        if isinstance(term, NumSym) and term.val in (-1, 0, 1):
            return term.val
        if isinstance(term, WordSym):
            u = term.name.upper()
            if u in ("TRUE", "T", "+"):
                return 1
            if u in ("FALSE", "F", "-"):
                return -1
            if u in ("UNDECIDED", "UNKNOWN", "U", "0"):
                return 0
        if isinstance(term, CharSym):
            if term.ch in ("+", "1"):
                return 1
            if term.ch in ("-", "T"):
                return -1
            if term.ch == "0":
                return 0
        raise Refal36Error(f"<{fn}> expects Kleene trit (True/Undecided/False or +1/0/-1), got {term}")

    @staticmethod
    def _trit_to_word(t: int) -> WordSym:
        return WordSym("True") if t > 0 else (WordSym("False") if t < 0 else WordSym("Undecided"))

    def _call_builtin(self, fn: str, arg: RefalExpr) -> RefalExpr:
        if fn == "Prout":
            self.output.append(format_refal_expr(arg))
            return ()
        if fn == "Print":
            self.output.append(format_refal_expr(arg))
            return arg
        if fn == "Card":
            if self.card_pos >= len(self.cards):
                return (NumSym(0),)
            line = self.cards[self.card_pos]
            self.card_pos += 1
            return tuple(CharSym(ch) for ch in line)

        # Hardware T36 arithmetic & 3-way compare on C-Complex CPU0!
        if fn in ("Add", "Sub", "Mul", "Div", "Mod"):
            a, b = self._extract_two_nums(arg, fn)
            if fn in ("Div", "Mod") and b == 0:
                raise Refal36Error(f"<{fn}> division by zero")
            entry = {"Add": "_REF_ADD", "Sub": "_REF_SUB", "Mul": "_REF_MUL", "Div": "_REF_DIV", "Mod": "_REF_MOD"}[fn]
            return (NumSym(self._hw_alu(entry, a, b)),)
        if fn == "Compare":
            a, b = self._extract_two_nums(arg, fn)
            c = self._hw_alu("_REF_CMP", a, b)
            return (CharSym("+" if c > 0 else ("-" if c < 0 else "0")),)

        # Conversions & reflection
        if fn == "Numb":
            s = "".join(t.ch for t in arg if isinstance(t, CharSym)).strip()
            if not s:
                return (NumSym(0),)
            try:
                return (NumSym(wrap(int(s), WORD_TRITS)),)
            except ValueError as exc:
                raise Refal36Error(f"<Numb> invalid integer string {s!r}") from exc
        if fn == "Symb":
            if len(arg) != 1 or not isinstance(arg[0], NumSym):
                raise Refal36Error(f"<Symb> expects 1 number, got {arg}")
            return tuple(CharSym(ch) for ch in str(arg[0].val))
        if fn == "Ord":
            return tuple(NumSym(ord(t.ch)) if isinstance(t, CharSym) else t for t in arg)
        if fn == "Chr":
            return tuple(CharSym(chr(t.val)) if isinstance(t, NumSym) else t for t in arg)
        if fn == "ObbOrd":
            return tuple(NumSym(char_to_obb729(t.ch)) if isinstance(t, CharSym) else t for t in arg)
        if fn == "ObbChr":
            return tuple(CharSym(obb729_to_char(t.val)) if isinstance(t, NumSym) else t for t in arg)
        if fn == "Explode":
            if len(arg) != 1 or not isinstance(arg[0], WordSym):
                raise Refal36Error(f"<Explode> expects 1 word symbol, got {arg}")
            return tuple(CharSym(ch) for ch in arg[0].name)
        if fn == "Implode":
            s = "".join(t.ch for t in arg if isinstance(t, CharSym))
            return (WordSym(s),)
        if fn == "Lenw":
            return (NumSym(len(arg)),) + arg
        if fn == "Type":
            if not arg:
                return (CharSym("*"), CharSym("0"))
            t0 = arg[0]
            if isinstance(t0, Bracket):
                code = "B"
            elif isinstance(t0, NumSym):
                code = "N"
            elif isinstance(t0, WordSym):
                code = "W"
            elif isinstance(t0, CharSym):
                code = "L" if t0.ch.isalpha() else ("D" if t0.ch.isdigit() else "O")
            else:
                code = "O"
            return (CharSym(code),) + arg
        if fn == "First":
            if not arg or not isinstance(arg[0], NumSym):
                raise Refal36Error("<First s.N e.X> expects count first")
            k = max(0, arg[0].val)
            rest = arg[1:]
            return (Bracket(rest[:k]),) + rest[k:]
        if fn == "Last":
            if not arg or not isinstance(arg[0], NumSym):
                raise Refal36Error("<Last s.N e.X> expects count first")
            k = max(0, arg[0].val)
            rest = arg[1:]
            split = max(0, len(rest) - k)
            return (Bracket(rest[:split]),) + rest[split:]

        # Burial Store (<Br>, <Dg>, <Cp>) mirrored into UE-1000 TernaryMemory!
        if fn == "Br":
            eq_idx = next((i for i, t in enumerate(arg) if t == CharSym("=")), -1)
            if eq_idx < 0:
                raise Refal36Error("<Br e.Key '=' e.Val> missing '=' separator")
            key = arg[:eq_idx]
            val = arg[eq_idx + 1 :]
            self.store.setdefault(key, []).append(val)
            self._mirror_store_write(-1, hash(key) & 0xFFFFF)
            return ()
        if fn == "Cp":
            stack = self.store.get(arg)
            self.store_ops += 1
            return stack[-1] if stack else ()
        if fn == "Dg":
            stack = self.store.get(arg)
            self.store_ops += 1
            return stack.pop() if stack else ()

        # Meta-call <Mu s.Fn e.Arg>
        if fn == "Mu":
            if not arg:
                raise Refal36Error("<Mu> requires function name")
            head = arg[0]
            if isinstance(head, WordSym):
                target = head.name
            elif isinstance(head, CharSym):
                target = head.ch
            else:
                raise Refal36Error(f"<Mu> invalid function designator {head}")
            return self._call_func(target, arg[1:])

        if fn == "Step":
            return (NumSym(self.steps),)

        # Duwamish-I Native Balanced-Ternary, Kleene & Good-Wald Built-ins!
        if fn in ("KleeneAnd", "KleeneOr", "KleeneEqv"):
            if len(arg) != 2:
                raise Refal36Error(f"<{fn}> expects 2 Kleene arguments")
            a = self._to_trit(arg[0], fn)
            b = self._to_trit(arg[1], fn)
            entry = {"KleeneAnd": "_REF_KAND", "KleeneOr": "_REF_KOR", "KleeneEqv": "_REF_KEQV"}[fn]
            return (self._trit_to_word(self._hw_alu(entry, a, b)),)
        if fn == "KleeneNot":
            if len(arg) != 1:
                raise Refal36Error("<KleeneNot> expects 1 Kleene argument")
            a = self._to_trit(arg[0], fn)
            return (self._trit_to_word(self._hw_alu("_REF_KNOT", a, 0)),)
        if fn == "Trits":
            if len(arg) != 1 or not isinstance(arg[0], NumSym):
                raise Refal36Error("<Trits s.N> expects integer")
            v = arg[0].val
            if v == 0:
                return (CharSym("0"),)
            ts = int_to_trits(v, WORD_TRITS)
            while len(ts) > 1 and ts[-1] == 0:
                ts.pop()
            cmap = {-1: "T", 0: "0", 1: "1"}
            return tuple(CharSym(cmap[t]) for t in reversed(ts))
        if fn == "Untrits":
            s = "".join(t.ch for t in arg if isinstance(t, CharSym)).strip()
            return (NumSym(trit_str_to_int(s)),)
        if fn == "Hept":
            if len(arg) != 1 or not isinstance(arg[0], NumSym):
                raise Refal36Error("<Hept s.N> expects integer")
            return tuple(CharSym(ch) for ch in int_to_hept(arg[0].val, WORD_TRITS))
        if fn == "Wald":
            # <Wald evidence_ticks threshold_ticks> -> True / Undecided / False via hardware DWALD!
            ev, th = self._extract_two_nums(arg, fn)
            res = self._hw_alu("_REF_DWALD", ev, th)
            return (self._trit_to_word(res),)
        if fn == "Dlse":
            a, b = self._extract_two_nums(arg, fn)
            return (NumSym(self._hw_alu("_REF_DLSE", a, b)),)

        raise Refal36Error(f"Unknown REFAL-36 function <{fn}>")


def run_refal(
    source: str,
    machine: Optional[UE1000Machine] = None,
    cards: Optional[Sequence[str]] = None,
    max_steps: int = 200_000,
) -> RefalRunResult:
    prog = parse_refal(source)
    return Refal36VM(prog, machine=machine, cards=cards, max_steps=max_steps).run()
