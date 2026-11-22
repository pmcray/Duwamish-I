"""SAL-3 — Knuth's Strongly-Typed Balanced-Ternary Systems & Vector Compiler for UE-1000.

Targets TRIAD-36 assembly (Principles of Operation, Rev. 1.5), featuring:
  - Strong static typing: INT (T36), FLOAT (TF36), TRIT (-1/0/+1), DECIBAN (DB36), ADDR (T18)
  - Automatic compiler-enforced safety for TF36 UNDECIDED (NaN): `sign` and comparisons on
    FLOAT always emit `FCM` rather than raw scalar `TST` (Rev. 1.5 §2.3 & §4.3A!)
  - Native three-valued `sign expr of - : ... 0 : ... + : ... end` compiling to single `B3`
  - Hardware vector slice statements (`dst[lo..hi] := a[lo..hi] + b[lo..hi]`, `-`, `*`, `fma`,
    `&`, `|`, `eqv`) automatically strip-mined into C-Complex VL<=27 `VLD/VADD/VFMA/VST` loops
  - Native T-Fabric dispatch intrinsics (`tfabric_gemm`, `tfabric_fft`) building 9-word TTDs
    and emitting `TEXEC A1, S0` + `TWAIT S0, #0` returning status TRIT
  - Optional Hofstadter `bloop` termination certification (acyclic call graph, no `while`)
  - Peephole optimizer + Cray Dual-Parcel 18-trit instruction packer (`opt=True`)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .isa import LONG_OPS, SHORT0_OPS, SHORTT_OPS
from .machine import UE1000Machine
from .ternary import (
    WMAX_7,
    WMAX_22,
    WORD_TRITS,
    char_to_obb729,
    hept_to_int,
    in_range,
    obb729_to_char,
    trit_str_to_int,
    wrap,
)
from .tfabric import MODE_RADIX3_FFT, MODE_TBFP9_GEMM, TileTaskDescriptor
from .tfloat import (
    DECIBAN_TICKS_PER_DB,
    TF36,
    db_to_ticks,
    float_to_tfloat,
    tfloat_to_float,
    ticks_to_db,
)
from .triad36 import AssembledProgram, assemble

# ---------------------------------------------------------------------------
# Errors & Types
# ---------------------------------------------------------------------------

class SAL3Error(Exception):
    """Compile-time lexical, syntax, type, or BlooP certification error."""


TYPES = {"INT", "FLOAT", "TRIT", "DECIBAN", "ADDR"}

# Comparison SEL truth tables (low triad of SEL operand: trits [C=-1, C=0, C=+1])
# Value = trit[+1]*9 + trit[0]*3 + trit[-1]
_CMP_SEL_TABLE: Dict[str, int] = {
    "<":  (-1) * 9 + (-1) * 3 + (1),   # -11: true iff C == -1
    "<=": (-1) * 9 + (1) * 3 + (1),    #  -5: true iff C in {-1, 0}
    "=":  (-1) * 9 + (1) * 3 + (-1),   #  -7: true iff C == 0
    "<>": (1) * 9 + (-1) * 3 + (1),    #  +7: true iff C != 0
    ">=": (1) * 9 + (1) * 3 + (-1),    # +11: true iff C in {0, +1}
    ">":  (1) * 9 + (-1) * 3 + (-1),   #  +5: true iff C == +1
}


# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Token:
    kind: str
    value: object
    line: int


KEYWORDS = {
    "bloop", "const", "global", "var", "proc", "begin", "end",
    "if", "then", "else", "while", "do", "for", "to", "by",
    "sign", "of", "return", "break", "next",
    "and", "or", "not",
    "true", "false", "undecided", "unknown", "undecided_f",
    "neg", "zero", "pos",
    "INT", "FLOAT", "TRIT", "DECIBAN", "ADDR",
    "mod",
}

_TWO_CHAR_OPS = {":=", "+:=", "-:=", "<=", ">=", "<>", "<<", ">>", "->", ".."}
_SINGLE_CHAR_OPS = set("+-*/&|=<>,:;()[]@~")


def tokenize(source: str) -> List[Token]:
    tokens: List[Token] = []
    lines = source.splitlines()
    for lineno, raw in enumerate(lines, start=1):
        # Strip -- comments (respecting string literals)
        i = 0
        n = len(raw)
        while i < n:
            ch = raw[i]
            if ch in " \t\r":
                i += 1
                continue
            if ch == "-" and i + 1 < n and raw[i + 1] == "-":
                break
            # Three-char compound assignment (+:=, -:=)
            if raw[i : i + 3] in ("+:=", "-:="):
                tokens.append(Token("OP", raw[i : i + 3], lineno))
                i += 3
                continue
            # Two-char operators
            if raw[i : i + 2] in _TWO_CHAR_OPS:
                tokens.append(Token("OP", raw[i : i + 2], lineno))
                i += 2
                continue
            # Character literal 'a' / '\n'
            if ch == "'":
                j = i + 1
                if j >= n:
                    raise SAL3Error(f"Line {lineno}: Unterminated character literal")
                if raw[j] == "\\":
                    esc = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", "'": "'", "0": "\x00"}
                    if j + 1 >= n or raw[j + 1] not in esc:
                        raise SAL3Error(f"Line {lineno}: Invalid escape in char literal")
                    val = char_to_obb729(esc[raw[j + 1]])
                    j += 2
                else:
                    val = char_to_obb729(raw[j])
                    j += 1
                if j >= n or raw[j] != "'":
                    raise SAL3Error(f"Line {lineno}: Expected closing quote for char literal")
                tokens.append(Token("INT_LIT", val, lineno))
                i = j + 1
                continue
            # String literal
            if ch == '"':
                j = i + 1
                chars: List[str] = []
                while j < n and raw[j] != '"':
                    if raw[j] == "\\":
                        esc = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"'}
                        if j + 1 >= n or raw[j + 1] not in esc:
                            raise SAL3Error(f"Line {lineno}: Invalid string escape")
                        chars.append(esc[raw[j + 1]])
                        j += 2
                    else:
                        chars.append(raw[j])
                        j += 1
                if j >= n:
                    raise SAL3Error(f"Line {lineno}: Unterminated string literal")
                tokens.append(Token("STR_LIT", "".join(chars), lineno))
                i = j + 1
                continue
            # Numeric literals: 0t1T0 (balanced ternary), 0hAM (heptavintimal), float, deciban, int
            if ch.isdigit():
                if raw[i : i + 2] in ("0t", "0T"):
                    j = i + 2
                    while j < n and raw[j] in "01Tt_":
                        j += 1
                    tokens.append(Token("INT_LIT", trit_str_to_int(raw[i + 2 : j]), lineno))
                    i = j
                    continue
                if raw[i : i + 2] in ("0h", "0H"):
                    j = i + 2
                    while j < n and (raw[j] in "0ABCDEFGHIJKLMNOPQRSTUVWXYZ_"):
                        j += 1
                    tokens.append(Token("INT_LIT", hept_to_int(raw[i + 2 : j]), lineno))
                    i = j
                    continue
                j = i
                is_float = False
                while j < n and raw[j].isdigit():
                    j += 1
                # Check decimal point (careful NOT to swallow `..` range operator!)
                if j < n and raw[j] == "." and not (j + 1 < n and raw[j + 1] == "."):
                    is_float = True
                    j += 1
                    while j < n and raw[j].isdigit():
                        j += 1
                if j < n and raw[j] in "eE":
                    is_float = True
                    j += 1
                    if j < n and raw[j] in "+-":
                        j += 1
                    while j < n and raw[j].isdigit():
                        j += 1
                num_str = raw[i:j]
                if raw[j : j + 2].lower() == "db":
                    tokens.append(Token("DB_LIT", float(num_str), lineno))
                    i = j + 2
                elif is_float:
                    tokens.append(Token("FLOAT_LIT", float(num_str), lineno))
                    i = j
                else:
                    tokens.append(Token("INT_LIT", int(num_str), lineno))
                    i = j
                continue
            # Identifiers & keywords
            if ch.isalpha() or ch == "_":
                j = i
                while j < n and (raw[j].isalnum() or raw[j] == "_"):
                    j += 1
                word = raw[i:j]
                if word in KEYWORDS:
                    tokens.append(Token("KW", word, lineno))
                else:
                    tokens.append(Token("IDENT", word, lineno))
                i = j
                continue
            if ch in _SINGLE_CHAR_OPS:
                tokens.append(Token("OP", ch, lineno))
                i += 1
                continue
            raise SAL3Error(f"Line {lineno}: Unexpected character {ch!r}")
    tokens.append(Token("EOF", None, len(lines) + 1))
    return tokens


# ---------------------------------------------------------------------------
# AST Nodes
# ---------------------------------------------------------------------------

@dataclass
class ConstDecl:
    name: str
    typ: str
    expr: object
    line: int


@dataclass
class GlobalDecl:
    name: str
    typ: str
    size: Optional[int]      # None -> scalar; int -> static array of `size` words
    init: object             # Expr or List[Expr] or None
    line: int


@dataclass
class Param:
    name: str
    typ: str


@dataclass
class ProcDecl:
    name: str
    params: List[Param]
    ret_type: str
    body: List[object]
    line: int


@dataclass
class Program:
    bloop: bool
    consts: List[ConstDecl]
    globals: List[GlobalDecl]
    procs: List[ProcDecl]


# Statements
@dataclass
class VarStmt:
    name: str
    typ: str
    size: Optional[int]
    init: object
    line: int


@dataclass
class AssignStmt:
    target: object           # VarExpr or IndexExpr
    op: str                  # ":=", "+:=", "-:="
    value: object
    line: int


@dataclass
class VecSliceStmt:
    """Hardware C-Complex vector slice operation: dst[lo..hi] := a[lo..hi] op b[lo..hi]."""
    dst_name: str
    lo: object
    hi: object
    op: str                  # "+", "-", "*", "fma", "&", "|", "eqv", "copy"
    a_name: str
    b_name: Optional[str]
    line: int


@dataclass
class IfStmt:
    cond: object
    then_body: List[object]
    else_body: Optional[List[object]]
    line: int


@dataclass
class WhileStmt:
    cond: object
    body: List[object]
    line: int


@dataclass
class ForStmt:
    var: str
    start: object
    stop: object
    step: Optional[object]
    body: List[object]
    line: int


@dataclass
class SignStmt:
    expr: object
    neg_body: Optional[List[object]]
    zero_body: Optional[List[object]]
    pos_body: Optional[List[object]]
    line: int


@dataclass
class ReturnStmt:
    expr: Optional[object]
    line: int


@dataclass
class BreakStmt:
    line: int


@dataclass
class NextStmt:
    line: int


@dataclass
class ExprStmt:
    expr: object
    line: int


# Expressions
@dataclass
class LitExpr:
    typ: str
    value: object
    line: int


@dataclass
class VarExpr:
    name: str
    line: int


@dataclass
class AddrOfExpr:
    name: str
    line: int


@dataclass
class IndexExpr:
    base: object
    index: object
    line: int


@dataclass
class UnaryExpr:
    op: str
    operand: object
    line: int


@dataclass
class BinaryExpr:
    op: str
    left: object
    right: object
    line: int


@dataclass
class CallExpr:
    name: str
    args: List[object]
    line: int


# ---------------------------------------------------------------------------
# Recursive-Descent Parser
# ---------------------------------------------------------------------------

class Parser:
    def __init__(self, tokens: List[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self, offset: int = 0) -> Token:
        idx = min(self.pos + offset, len(self.tokens) - 1)
        return self.tokens[idx]

    def match(self, kind: str, val: object = None) -> Optional[Token]:
        tok = self.peek()
        if tok.kind == kind and (val is None or tok.value == val):
            self.pos += 1
            return tok
        return None

    def expect(self, kind: str, val: object = None) -> Token:
        tok = self.peek()
        if tok.kind != kind or (val is not None and tok.value != val):
            raise SAL3Error(
                f"Line {tok.line}: Expected {kind} {val or ''}, got {tok.kind}({tok.value!r})"
            )
        self.pos += 1
        return tok

    def parse_type(self) -> str:
        tok = self.peek()
        if tok.kind == "KW" and tok.value in TYPES:
            self.pos += 1
            return str(tok.value)
        raise SAL3Error(f"Line {tok.line}: Expected type in {sorted(TYPES)}, got {tok.value!r}")

    def parse_program(self) -> Program:
        bloop = False
        consts: List[ConstDecl] = []
        globs: List[GlobalDecl] = []
        procs: List[ProcDecl] = []

        while self.peek().kind != "EOF":
            if self.match("KW", "bloop"):
                bloop = True
            elif self.match("KW", "const"):
                consts.append(self.parse_const())
            elif self.match("KW", "global"):
                globs.append(self.parse_global())
            elif self.match("KW", "proc"):
                procs.append(self.parse_proc())
            elif self.match("OP", ";"):
                continue
            else:
                tok = self.peek()
                raise SAL3Error(f"Line {tok.line}: Unexpected top-level token {tok.value!r}")

        return Program(bloop=bloop, consts=consts, globals=globs, procs=procs)

    def parse_const(self) -> ConstDecl:
        tok = self.expect("IDENT")
        self.expect("OP", ":")
        typ = self.parse_type()
        self.expect("OP", "=")
        expr = self.parse_expr()
        self.match("OP", ";")
        return ConstDecl(str(tok.value), typ, expr, tok.line)

    def parse_global(self) -> GlobalDecl:
        tok = self.expect("IDENT")
        self.expect("OP", ":")
        typ = self.parse_type()
        size = None
        if self.match("OP", "["):
            sz_tok = self.expect("INT_LIT")
            size = int(sz_tok.value)
            self.expect("OP", "]")
        init = None
        if self.match("OP", ":="):
            if self.match("OP", "["):
                items = []
                if not self.match("OP", "]"):
                    while True:
                        items.append(self.parse_expr())
                        if not self.match("OP", ","):
                            break
                    self.expect("OP", "]")
                init = items
                if size is None:
                    size = len(items)
            else:
                init = self.parse_expr()
        self.match("OP", ";")
        return GlobalDecl(str(tok.value), typ, size, init, tok.line)

    def parse_proc(self) -> ProcDecl:
        name_tok = self.expect("IDENT")
        self.expect("OP", "(")
        params: List[Param] = []
        if not self.match("OP", ")"):
            while True:
                p_tok = self.expect("IDENT")
                self.expect("OP", ":")
                p_typ = self.parse_type()
                params.append(Param(str(p_tok.value), p_typ))
                if not self.match("OP", ","):
                    break
            self.expect("OP", ")")
        ret_type = "INT"
        if self.match("OP", "->"):
            ret_type = self.parse_type()
        if self.match("OP", "="):
            expr = self.parse_expr()
            self.match("OP", ";")
            body: List[object] = [ReturnStmt(expr, name_tok.line)]
        else:
            body = self.parse_block()
        return ProcDecl(str(name_tok.value), params, ret_type, body, name_tok.line)

    def parse_block(self) -> List[object]:
        self.expect("KW", "begin")
        stmts: List[object] = []
        while not self.match("KW", "end"):
            if self.peek().kind == "EOF":
                raise SAL3Error("Unexpected EOF inside begin..end block")
            if self.match("OP", ";"):
                continue
            stmts.append(self.parse_stmt())
        return stmts

    def parse_stmt_or_block(self) -> List[object]:
        if self.peek().kind == "KW" and self.peek().value == "begin":
            return self.parse_block()
        return [self.parse_stmt()]

    def parse_stmt(self) -> object:
        tok = self.peek()
        if tok.kind == "KW" and tok.value == "begin":
            # Inline nested block wrapped in `if true`? No, flatten via simple IfStmt(true)
            body = self.parse_block()
            return IfStmt(LitExpr("TRIT", 1, tok.line), body, None, tok.line)
        if self.match("KW", "var"):
            vtok = self.expect("IDENT")
            self.expect("OP", ":")
            typ = self.parse_type()
            size = None
            if self.match("OP", "["):
                size = int(self.expect("INT_LIT").value)
                self.expect("OP", "]")
            init = None
            if self.match("OP", ":="):
                init = self.parse_expr()
            self.match("OP", ";")
            return VarStmt(str(vtok.value), typ, size, init, vtok.line)
        if self.match("KW", "if"):
            cond = self.parse_expr()
            self.expect("KW", "then")
            then_b = self.parse_stmt_or_block()
            else_b = None
            if self.match("KW", "else"):
                else_b = self.parse_stmt_or_block()
            self.match("OP", ";")
            return IfStmt(cond, then_b, else_b, tok.line)
        if self.match("KW", "while"):
            cond = self.parse_expr()
            self.expect("KW", "do")
            body = self.parse_stmt_or_block()
            self.match("OP", ";")
            return WhileStmt(cond, body, tok.line)
        if self.match("KW", "for"):
            vtok = self.expect("IDENT")
            self.expect("OP", ":=")
            start = self.parse_expr()
            self.expect("KW", "to")
            stop = self.parse_expr()
            step = None
            if self.match("KW", "by"):
                step = self.parse_expr()
            self.expect("KW", "do")
            body = self.parse_stmt_or_block()
            self.match("OP", ";")
            return ForStmt(str(vtok.value), start, stop, step, body, tok.line)
        if self.match("KW", "sign"):
            expr = self.parse_expr()
            self.expect("KW", "of")
            neg_b = zero_b = pos_b = None
            while not self.match("KW", "end"):
                arm_tok = self.peek()
                if arm_tok.kind == "OP" and arm_tok.value == "-":
                    self.pos += 1
                    arm = -1
                elif arm_tok.kind == "OP" and arm_tok.value == "+":
                    self.pos += 1
                    arm = 1
                elif arm_tok.kind == "INT_LIT" and arm_tok.value == 0:
                    self.pos += 1
                    arm = 0
                elif arm_tok.kind == "KW" and arm_tok.value in ("neg", "false"):
                    self.pos += 1
                    arm = -1
                elif arm_tok.kind == "KW" and arm_tok.value in ("zero", "undecided", "unknown"):
                    self.pos += 1
                    arm = 0
                elif arm_tok.kind == "KW" and arm_tok.value in ("pos", "true"):
                    self.pos += 1
                    arm = 1
                else:
                    raise SAL3Error(f"Line {arm_tok.line}: Invalid sign arm label {arm_tok.value!r}")
                self.expect("OP", ":")
                arm_body = self.parse_stmt_or_block()
                if arm == -1:
                    neg_b = arm_body
                elif arm == 0:
                    zero_b = arm_body
                else:
                    pos_b = arm_body
                self.match("OP", ";")
            self.match("OP", ";")
            return SignStmt(expr, neg_b, zero_b, pos_b, tok.line)
        if self.match("KW", "return"):
            nxt = self.peek()
            if nxt.kind in ("EOF",) or (nxt.kind == "KW" and nxt.value in ("end", "else")) or (nxt.kind == "OP" and nxt.value == ";"):
                self.match("OP", ";")
                return ReturnStmt(None, tok.line)
            expr = self.parse_expr()
            self.match("OP", ";")
            return ReturnStmt(expr, tok.line)
        if self.match("KW", "break"):
            self.match("OP", ";")
            return BreakStmt(tok.line)
        if self.match("KW", "next"):
            self.match("OP", ";")
            return NextStmt(tok.line)

        # Lookahead for Hardware Vector Slice Statement: dst[lo..hi] := ...
        if tok.kind == "IDENT" and self.peek(1).kind == "OP" and self.peek(1).value == "[":
            saved = self.pos
            dst_name = str(self.expect("IDENT").value)
            self.expect("OP", "[")
            lo_expr = self.parse_expr()
            if self.match("OP", ".."):
                hi_expr = self.parse_expr()
                self.expect("OP", "]")
                self.expect("OP", ":=")
                # Check RHS: `fma(a[..], b[..])` or `a[..] op b[..]` or `a[..]`
                if self.peek().kind == "IDENT" and self.peek().value == "fma" and self.peek(1).value == "(":
                    self.pos += 2  # fma(
                    a_name = str(self.expect("IDENT").value)
                    self._consume_slice_brackets()
                    self.expect("OP", ",")
                    b_name = str(self.expect("IDENT").value)
                    self._consume_slice_brackets()
                    self.expect("OP", ")")
                    self.match("OP", ";")
                    return VecSliceStmt(dst_name, lo_expr, hi_expr, "fma", a_name, b_name, tok.line)
                a_name = str(self.expect("IDENT").value)
                self._consume_slice_brackets()
                op_tok = self.peek()
                if op_tok.kind == "OP" and op_tok.value in ("+", "-", "*", "&", "|"):
                    self.pos += 1
                    b_name = str(self.expect("IDENT").value)
                    self._consume_slice_brackets()
                    self.match("OP", ";")
                    return VecSliceStmt(dst_name, lo_expr, hi_expr, str(op_tok.value), a_name, b_name, tok.line)
                self.match("OP", ";")
                return VecSliceStmt(dst_name, lo_expr, hi_expr, "copy", a_name, None, tok.line)
            else:
                self.pos = saved  # Normal scalar/array assignment

        expr = self.parse_expr()
        if self.peek().kind == "OP" and self.peek().value in (":=", "+:=", "-:="):
            op = str(self.peek().value)
            self.pos += 1
            rhs = self.parse_expr()
            self.match("OP", ";")
            if not isinstance(expr, (VarExpr, IndexExpr)):
                raise SAL3Error(f"Line {tok.line}: Invalid assignment target")
            return AssignStmt(expr, op, rhs, tok.line)
        self.match("OP", ";")
        return ExprStmt(expr, tok.line)

    def _consume_slice_brackets(self) -> None:
        self.expect("OP", "[")
        _ = self.parse_expr()
        self.expect("OP", "..")
        _ = self.parse_expr()
        self.expect("OP", "]")

    # Precedence climbing: or -> and -> not -> cmp -> | -> & -> shifts -> +- -> */mod -> unary -> postfix
    def parse_expr(self) -> object:
        return self.parse_or()

    def parse_or(self) -> object:
        left = self.parse_and()
        while self.match("KW", "or"):
            left = BinaryExpr("or", left, self.parse_and(), self.peek().line)
        return left

    def parse_and(self) -> object:
        left = self.parse_not()
        while self.match("KW", "and"):
            left = BinaryExpr("and", left, self.parse_not(), self.peek().line)
        return left

    def parse_not(self) -> object:
        tok = self.peek()
        if self.match("KW", "not") or self.match("OP", "~"):
            return UnaryExpr("not", self.parse_not(), tok.line)
        return self.parse_cmp()

    def parse_cmp(self) -> object:
        left = self.parse_kor()
        tok = self.peek()
        if tok.kind == "OP" and tok.value in ("=", "<>", "<", "<=", ">", ">="):
            self.pos += 1
            return BinaryExpr(str(tok.value), left, self.parse_kor(), tok.line)
        return left

    def parse_kor(self) -> object:
        left = self.parse_kand()
        while self.match("OP", "|"):
            left = BinaryExpr("|", left, self.parse_kand(), self.peek().line)
        return left

    def parse_kand(self) -> object:
        left = self.parse_shift()
        while self.match("OP", "&"):
            left = BinaryExpr("&", left, self.parse_shift(), self.peek().line)
        return left

    def parse_shift(self) -> object:
        left = self.parse_add()
        while self.peek().kind == "OP" and self.peek().value in ("<<", ">>"):
            op = str(self.peek().value)
            self.pos += 1
            left = BinaryExpr(op, left, self.parse_add(), self.peek().line)
        return left

    def parse_add(self) -> object:
        left = self.parse_mul()
        while (
            self.peek().kind == "OP"
            and self.peek().value in ("+", "-")
            and not (self.peek(1).kind == "OP" and self.peek(1).value == ":")
        ):
            op = str(self.peek().value)
            self.pos += 1
            left = BinaryExpr(op, left, self.parse_mul(), self.peek().line)
        return left

    def parse_mul(self) -> object:
        left = self.parse_unary()
        while (self.peek().kind == "OP" and self.peek().value in ("*", "/")) or (
            self.peek().kind == "KW" and self.peek().value == "mod"
        ):
            op = str(self.peek().value)
            self.pos += 1
            left = BinaryExpr(op, left, self.parse_unary(), self.peek().line)
        return left

    def parse_unary(self) -> object:
        tok = self.peek()
        if self.match("OP", "-"):
            return UnaryExpr("-", self.parse_unary(), tok.line)
        if self.match("OP", "+"):
            return self.parse_unary()
        if self.match("OP", "@"):
            id_tok = self.expect("IDENT")
            return AddrOfExpr(str(id_tok.value), tok.line)
        return self.parse_postfix()

    def parse_postfix(self) -> object:
        expr = self.parse_primary()
        while self.match("OP", "["):
            idx = self.parse_expr()
            self.expect("OP", "]")
            expr = IndexExpr(expr, idx, self.peek().line)
        return expr

    def parse_primary(self) -> object:
        tok = self.peek()
        if self.match("INT_LIT"):
            return LitExpr("INT", int(tok.value), tok.line)
        if self.match("FLOAT_LIT"):
            return LitExpr("FLOAT", float(tok.value), tok.line)
        if self.match("DB_LIT"):
            return LitExpr("DECIBAN", float(tok.value), tok.line)
        if self.match("STR_LIT"):
            return LitExpr("ADDR", str(tok.value), tok.line)
        if self.match("KW", "true"):
            return LitExpr("TRIT", 1, tok.line)
        if self.match("KW", "false"):
            return LitExpr("TRIT", -1, tok.line)
        if self.match("KW", "undecided") or self.match("KW", "unknown"):
            return LitExpr("TRIT", 0, tok.line)
        if self.match("KW", "undecided_f"):
            return LitExpr("FLOAT", float("nan"), tok.line)
        if self.match("OP", "("):
            e = self.parse_expr()
            self.expect("OP", ")")
            return e
        if self.match("IDENT"):
            name = str(tok.value)
            if self.match("OP", "("):
                args: List[object] = []
                if not self.match("OP", ")"):
                    while True:
                        args.append(self.parse_expr())
                        if not self.match("OP", ","):
                            break
                    self.expect("OP", ")")
                return CallExpr(name, args, tok.line)
            return VarExpr(name, tok.line)
        raise SAL3Error(f"Line {tok.line}: Unexpected expression token {tok.kind}({tok.value!r})")


# ---------------------------------------------------------------------------
# Code Generator, Type Checker, BlooP Certifier & Dual-Parcel Optimizer
# ---------------------------------------------------------------------------

@dataclass
class VarInfo:
    name: str
    typ: str
    is_array: bool
    scope: str               # "GLOBAL" | "PARAM" | "LOCAL"
    offset: int              # FP offset for LOCAL/PARAM; 0 for GLOBAL
    size: int = 1            # array size if is_array


@dataclass
class CompileResult:
    assembly: str
    assembled: AssembledProgram
    bloop_certified: bool
    peephole_edits: int
    parcels_packed: int


class SAL3Compiler:
    """Strongly-typed SAL-3 -> TRIAD-36 compiler targeting Duwamish-I (UE-1000)."""

    def __init__(self, prog: Program, opt: bool = True, origin: int = 100) -> None:
        self.prog = prog
        self.opt = opt
        self.origin = origin

        self.consts: Dict[str, Tuple[str, int]] = {}  # name -> (typ, raw_t36_word)
        self.globals: Dict[str, VarInfo] = {}
        self.procs: Dict[str, ProcDecl] = {}
        self.const_pool: Dict[int, str] = {}           # raw_t36_word -> label `_K0`
        self.string_pool: List[Tuple[str, str]] = []   # (label, text)
        self.ttd_slots: List[str] = []                 # 9-word static TTD buffers
        self.code: List[str] = []
        self._lbl_seq = 0

        # Per-procedure state
        self._locals: Dict[str, VarInfo] = {}
        self._local_words = 0
        self._loop_stack: List[Tuple[str, str]] = []   # (continue_lbl, break_lbl)
        self._cur_proc: Optional[ProcDecl] = None

    def _new_label(self, prefix: str = "L") -> str:
        self._lbl_seq += 1
        return f"_{prefix}{self._lbl_seq}"

    def _emit(self, line: str) -> None:
        self.code.append(line)

    def _eval_const(self, expr: object) -> Tuple[str, int]:
        if isinstance(expr, LitExpr):
            if expr.typ == "INT":
                return "INT", wrap(int(expr.value), WORD_TRITS)
            if expr.typ == "TRIT":
                return "TRIT", int(expr.value)
            if expr.typ == "FLOAT":
                return "FLOAT", float_to_tfloat(float(expr.value), TF36)
            if expr.typ == "DECIBAN":
                return "DECIBAN", db_to_ticks(float(expr.value))
            raise SAL3Error(f"Line {expr.line}: Non-scalar constant literal")
        if isinstance(expr, VarExpr) and expr.name in self.consts:
            return self.consts[expr.name]
        if isinstance(expr, UnaryExpr) and expr.op == "-":
            t, v = self._eval_const(expr.operand)
            if t == "FLOAT":
                return "FLOAT", float_to_tfloat(-tfloat_to_float(v, TF36), TF36)
            return t, wrap(-v, WORD_TRITS)
        if isinstance(expr, BinaryExpr):
            tl, vl = self._eval_const(expr.left)
            tr, vr = self._eval_const(expr.right)
            if tl != tr:
                raise SAL3Error(f"Line {expr.line}: Constant type mismatch {tl} vs {tr}")
            if tl == "INT":
                if expr.op == "+":
                    return "INT", wrap(vl + vr)
                if expr.op == "-":
                    return "INT", wrap(vl - vr)
                if expr.op == "*":
                    return "INT", wrap(vl * vr)
                if expr.op == "/":
                    return "INT", int(vl / vr)
        raise SAL3Error("Constant initializer must be a compile-time constant expression")

    def _load_const_into(self, reg: str, raw_word: int) -> None:
        """Emit minimal instruction to load a 36-trit balanced constant into `reg`."""
        if in_range(raw_word, 7):
            self._emit(f"ADD {reg}, S0, #{raw_word}")
        elif in_range(raw_word, 22):
            self._emit(f"LDI {reg}, #{raw_word}")
        else:
            if raw_word not in self.const_pool:
                self.const_pool[raw_word] = f"_K{len(self.const_pool)}"
            lbl = self.const_pool[raw_word]
            self._emit(f"LD {reg}, {lbl}(A0)")

    # ------------------------------------------------------------------
    # Hofstadter BlooP Static Termination Certification
    # ------------------------------------------------------------------

    def _certify_bloop(self) -> None:
        """Verify Hofstadter's BlooP restrictions across all procedures reachable from main."""
        if "main" not in self.procs:
            raise SAL3Error("BlooP certification requires proc main()")

        call_graph: Dict[str, Set[str]] = {name: set() for name in self.procs}

        def scan_stmts(proc_name: str, stmts: Sequence[object], loop_vars: Set[str]) -> None:
            for st in stmts:
                if isinstance(st, WhileStmt):
                    raise SAL3Error(f"BlooP violation in {proc_name} (line {st.line}): unbounded `while` loop forbidden")
                elif isinstance(st, ForStmt):
                    scan_expr(proc_name, st.start)
                    scan_expr(proc_name, st.stop)
                    if st.step:
                        scan_expr(proc_name, st.step)
                    scan_stmts(proc_name, st.body, loop_vars | {st.var})
                elif isinstance(st, AssignStmt):
                    if isinstance(st.target, VarExpr) and st.target.name in loop_vars:
                        raise SAL3Error(f"BlooP violation in {proc_name} (line {st.line}): loop variable {st.target.name!r} reassigned")
                    scan_expr(proc_name, st.value)
                elif isinstance(st, IfStmt):
                    scan_expr(proc_name, st.cond)
                    scan_stmts(proc_name, st.then_body, loop_vars)
                    if st.else_body:
                        scan_stmts(proc_name, st.else_body, loop_vars)
                elif isinstance(st, SignStmt):
                    scan_expr(proc_name, st.expr)
                    for b in (st.neg_body, st.zero_body, st.pos_body):
                        if b:
                            scan_stmts(proc_name, b, loop_vars)
                elif isinstance(st, ReturnStmt) and st.expr:
                    scan_expr(proc_name, st.expr)
                elif isinstance(st, VarStmt) and st.init:
                    scan_expr(proc_name, st.init)
                elif isinstance(st, ExprStmt):
                    scan_expr(proc_name, st.expr)

        def scan_expr(proc_name: str, e: object) -> None:
            if isinstance(e, CallExpr):
                if e.name in self.procs:
                    call_graph[proc_name].add(e.name)
                for a in e.args:
                    scan_expr(proc_name, a)
            elif isinstance(e, BinaryExpr):
                scan_expr(proc_name, e.left)
                scan_expr(proc_name, e.right)
            elif isinstance(e, UnaryExpr):
                scan_expr(proc_name, e.operand)
            elif isinstance(e, IndexExpr):
                scan_expr(proc_name, e.base)
                scan_expr(proc_name, e.index)

        for name, proc in self.procs.items():
            scan_stmts(name, proc.body, set())

        # Check acyclicity from main via DFS 3-color (white/gray/black)
        state: Dict[str, int] = {}

        def dfs(u: str) -> None:
            state[u] = 1
            for v in call_graph[u]:
                if state.get(v, 0) == 1:
                    raise SAL3Error(f"BlooP violation: recursion cycle detected involving {u!r} -> {v!r}")
                if state.get(v, 0) == 0:
                    dfs(v)
            state[u] = 2

        dfs("main")

    # ------------------------------------------------------------------
    # Variable Resolution & Type Checking
    # ------------------------------------------------------------------

    def _lookup_var(self, name: str, line: int) -> VarInfo:
        if name in self._locals:
            return self._locals[name]
        if name in self.globals:
            return self.globals[name]
        raise SAL3Error(f"Line {line}: Undeclared identifier {name!r}")

    def _require_type(self, got: str, expected: str, line: int, ctx: str) -> None:
        if got == expected:
            return
        # Allow ADDR <-> INT interop for pointer arithmetic only where explicitly requested
        raise SAL3Error(f"Line {line}: Type error in {ctx}: expected {expected}, got {got}")

    # ------------------------------------------------------------------
    # Main Compilation Pipeline
    # ------------------------------------------------------------------

    def compile(self) -> CompileResult:
        # 1. Register constants, globals & procedures
        for c in self.prog.consts:
            t, v = self._eval_const(c.expr)
            if t != c.typ:
                raise SAL3Error(f"Line {c.line}: Const {c.name} declared {c.typ}, got {t}")
            self.consts[c.name] = (c.typ, v)

        for g in self.prog.globals:
            self.globals[g.name] = VarInfo(
                name=g.name, typ=g.typ, is_array=(g.size is not None), scope="GLOBAL",
                offset=0, size=(g.size or 1),
            )

        for p in self.prog.procs:
            if p.name in self.procs:
                raise SAL3Error(f"Line {p.line}: Duplicate procedure {p.name!r}")
            self.procs[p.name] = p

        if "main" not in self.procs:
            raise SAL3Error("SAL-3 program must define `proc main()`")

        if self.prog.bloop:
            self._certify_bloop()

        # 2. Emit startup stub: initialize SP (S8) to +200000, call main(), HLT!
        self._emit("; === SAL-3 Compiled TRIAD-36 Image (UE-1000 Rev. 1.5) ===")
        if self.prog.bloop:
            self._emit("; [BLOOP CERTIFIED: TERMINATION STATICALLY PROVEN]")
        self._emit("LDI S8, #200000")
        self._emit("CALL A0, _P_main")
        self._emit("HLT")

        # 3. Compile each procedure
        for p in self.prog.procs:
            self._compile_proc(p)

        # 4. Emit static globals, constant pool, string pool, and TTD scratch slots
        self._emit("; === Static Data & Constant Pool ===")
        for g in self.prog.globals:
            lbl = f"_G_{g.name}"
            if g.size is not None:
                vals: List[int] = [0] * g.size
                if isinstance(g.init, list):
                    for idx, item in enumerate(g.init):
                        t, v = self._eval_const(item)
                        self._require_type(t, g.typ, g.line, f"global array {g.name}")
                        vals[idx] = v
                self._emit(f"{lbl}:")
                # Emit in chunks of 9 words per .WORD line
                for i in range(0, len(vals), 9):
                    chunk = ", ".join(str(x) for x in vals[i : i + 9])
                    self._emit(f".WORD {chunk}")
            else:
                val = 0
                if g.init is not None:
                    t, val = self._eval_const(g.init)
                    self._require_type(t, g.typ, g.line, f"global {g.name}")
                self._emit(f"{lbl}: .WORD {val}")

        for raw_w, lbl in self.const_pool.items():
            self._emit(f"{lbl}: .WORD {raw_w}")

        for lbl, text in self.string_pool:
            trytes = [len(text)] + [char_to_obb729(ch) for ch in text]
            self._emit(f"{lbl}: .WORD " + ", ".join(str(x) for x in trytes))

        for ttd_lbl in self.ttd_slots:
            self._emit(f"{ttd_lbl}: .WORD 0, 0, 0, 0, 0, 0, 0, 0, 0")

        # 5. Peephole optimization & Cray Dual-Parcel packing
        peep_edits = 0
        packed = 0
        lines = self.code
        if self.opt:
            lines, peep_edits = _peephole_optimize(lines)
            lines, packed = _pack_dual_parcels(lines)

        asm_text = "\n".join(lines) + "\n"
        assembled = assemble(asm_text, origin=self.origin)
        return CompileResult(
            assembly=asm_text,
            assembled=assembled,
            bloop_certified=self.prog.bloop,
            peephole_edits=peep_edits,
            parcels_packed=packed,
        )

    def _count_locals(self, stmts: Sequence[object]) -> int:
        total = 0
        for st in stmts:
            if isinstance(st, VarStmt):
                total += st.size if st.size is not None else 1
            elif isinstance(st, IfStmt):
                total += self._count_locals(st.then_body)
                if st.else_body:
                    total += self._count_locals(st.else_body)
            elif isinstance(st, WhileStmt):
                total += self._count_locals(st.body)
            elif isinstance(st, ForStmt):
                # 3 hidden slots: loop var (if local), limit, step
                total += 3 + self._count_locals(st.body)
            elif isinstance(st, SignStmt):
                for b in (st.neg_body, st.zero_body, st.pos_body):
                    if b:
                        total += self._count_locals(b)
        return total

    def _alloc_local(self, name: str, typ: str, size: Optional[int] = None) -> VarInfo:
        words = size if size is not None else 1
        self._local_words += words
        # Base address of scalar or local array = FP - self._local_words
        info = VarInfo(
            name=name, typ=typ, is_array=(size is not None), scope="LOCAL",
            offset=-self._local_words, size=words,
        )
        self._locals[name] = info
        return info

    def _compile_proc(self, p: ProcDecl) -> None:
        self._cur_proc = p
        self._locals = {}
        self._local_words = 0
        self._loop_stack = []

        n_params = len(p.params)
        # Stack frame after PUSH S7; S7 <- S8:
        #   FP + 0: saved FP (S7)
        #   FP + 1: return PC (pushed by CALL A0)
        #   FP + 2 + (n_params - 1 - i): param i (pushed left-to-right!)
        for idx, param in enumerate(p.params):
            off = 2 + (n_params - 1 - idx)
            self._locals[param.name] = VarInfo(param.name, param.typ, False, "PARAM", off)

        frame_slots = self._count_locals(p.body)
        self._emit(f"_P_{p.name}:")
        self._emit("PUSH S7")
        self._emit("ADD S7, S8, #0")
        self._emit("S2A A7, S7")
        if frame_slots > 0:
            self._load_const_into("S2", frame_slots)
            self._emit("SUB S8, S8, S2")

        for st in p.body:
            self._compile_stmt(st)

        # Default fallthrough return 0
        self._emit("ADD S1, S0, #0")
        self._emit_epilogue(n_params)

    def _emit_epilogue(self, n_params: int) -> None:
        self._emit("ADD S8, S7, #0")
        self._emit("POP S7")
        self._emit("S2A A7, S7")
        self._emit(f"RET #{n_params}")

    def _compile_stmt(self, st: object) -> None:
        if isinstance(st, VarStmt):
            info = self._alloc_local(st.name, st.typ, st.size)
            if st.init is not None:
                got_t = self._compile_expr(st.init)
                self._require_type(got_t, st.typ, st.line, f"var {st.name} init")
                self._emit(f"ST S1, {info.offset}(A7)")
            elif not info.is_array:
                self._emit(f"ST S0, {info.offset}(A7)")
            return

        if isinstance(st, AssignStmt):
            if isinstance(st.target, VarExpr):
                info = self._lookup_var(st.target.name, st.line)
                if info.is_array:
                    raise SAL3Error(f"Line {st.line}: Cannot assign directly to array {info.name!r}")
                if st.op == ":=":
                    got_t = self._compile_expr(st.value)
                    self._require_type(got_t, info.typ, st.line, f"assign {info.name}")
                else:
                    # Compound +:= / -:=
                    self._load_var(info)
                    self._emit("PUSH S1")
                    got_t = self._compile_expr(st.value)
                    self._require_type(got_t, info.typ, st.line, f"compound {st.op}")
                    self._emit("POP S2")
                    self._emit_typed_binop(st.op[0], info.typ, st.line)
                self._store_var(info)
            elif isinstance(st.target, IndexExpr):
                elem_t = self._compile_index_addr(st.target)  # address in S1
                self._emit("PUSH S1")
                if st.op == ":=":
                    got_t = self._compile_expr(st.value)
                    self._require_type(got_t, elem_t, st.line, "array element assign")
                    self._emit("POP S2")
                    self._emit("S2A A2, S2")
                    self._emit("ST S1, 0(A2)")
                else:
                    self._emit("S2A A2, S1")
                    self._emit("LD S2, 0(A2)")
                    self._emit("PUSH S2")
                    got_t = self._compile_expr(st.value)
                    self._require_type(got_t, elem_t, st.line, f"array compound {st.op}")
                    self._emit("POP S2")
                    self._emit_typed_binop(st.op[0], elem_t, st.line)
                    self._emit("POP S2")
                    self._emit("S2A A2, S2")
                    self._emit("ST S1, 0(A2)")
            return

        if isinstance(st, VecSliceStmt):
            self._compile_vec_slice(st)
            return

        if isinstance(st, IfStmt):
            l_else = self._new_label("ELSE")
            l_end = self._new_label("ENDIF")
            ct = self._compile_expr(st.cond)
            if ct == "FLOAT":
                self._emit("FCM S1, S0")
            else:
                self._emit("TST S1")
            # Three-valued rule: taken ONLY when C == +1 (TRUE); 0 (UNDECIDED) & -1 (FALSE) go to else!
            target_no = l_else if st.else_body is not None else l_end
            self._emit(f"BN {target_no}")
            self._emit(f"BZ {target_no}")
            for s in st.then_body:
                self._compile_stmt(s)
            if st.else_body is not None:
                self._emit(f"BR {l_end}")
                self._emit(f"{l_else}:")
                for s in st.else_body:
                    self._compile_stmt(s)
            self._emit(f"{l_end}:")
            return

        if isinstance(st, WhileStmt):
            l_top = self._new_label("WTOP")
            l_end = self._new_label("WEND")
            self._loop_stack.append((l_top, l_end))
            self._emit(f"{l_top}:")
            ct = self._compile_expr(st.cond)
            self._emit("FCM S1, S0" if ct == "FLOAT" else "TST S1")
            self._emit(f"BN {l_end}")
            self._emit(f"BZ {l_end}")
            for s in st.body:
                self._compile_stmt(s)
            self._emit(f"BR {l_top}")
            self._emit(f"{l_end}:")
            self._loop_stack.pop()
            return

        if isinstance(st, ForStmt):
            # Allocate or look up loop variable + hidden limit & step slots
            if st.var in self._locals or st.var in self.globals:
                v_info = self._lookup_var(st.var, st.line)
                self._require_type(v_info.typ, "INT", st.line, "for loop variable")
            else:
                v_info = self._alloc_local(st.var, "INT")
            lim_info = self._alloc_local(f"__lim_{self._lbl_seq}", "INT")
            stp_info = self._alloc_local(f"__stp_{self._lbl_seq}", "INT")

            # Evaluate start, stop, step once on entry (fixed bounds -> BlooP safe!)
            self._require_type(self._compile_expr(st.start), "INT", st.line, "for start")
            self._store_var(v_info)
            self._require_type(self._compile_expr(st.stop), "INT", st.line, "for stop")
            self._store_var(lim_info)
            if st.step is not None:
                self._require_type(self._compile_expr(st.step), "INT", st.line, "for step")
            else:
                self._emit("ADD S1, S0, #1")
            self._store_var(stp_info)

            l_top = self._new_label("FTOP")
            l_next = self._new_label("FNEXT")
            l_end = self._new_label("FEND")
            self._loop_stack.append((l_next, l_end))

            self._emit(f"{l_top}:")
            self._load_var(v_info)
            self._emit(f"LD S2, {lim_info.offset}(A7)")
            self._emit("CMP S1, S2")
            self._emit(f"BP {l_end}")  # stop when var > limit (for positive step)
            for s in st.body:
                self._compile_stmt(s)
            self._emit(f"{l_next}:")
            self._load_var(v_info)
            self._emit(f"LD S2, {stp_info.offset}(A7)")
            self._emit("ADD S1, S1, S2")
            self._store_var(v_info)
            self._emit(f"BR {l_top}")
            self._emit(f"{l_end}:")
            self._loop_stack.pop()
            return

        if isinstance(st, SignStmt):
            # Native 3-way balanced-ternary branch compiling directly to hardware B3!
            et = self._compile_expr(st.expr)
            # Rev. 1.5 conformance: FLOAT uses FCM S1, S0 so UNDECIDED maps cleanly to C = 0!
            if et == "FLOAT":
                self._emit("FCM S1, S0")
            else:
                self._emit("TST S1")
            l_tbl = self._new_label("S0")
            l_neg = self._new_label("SNEG")
            l_zero = self._new_label("SZERO")
            l_pos = self._new_label("SPOS")
            l_end = self._new_label("SEND")

            self._emit(f"B3 {l_tbl}")
            # Exact 3-word branch table (each slot padded to 1 full 36-trit word!)
            self._emit(f"BR {l_neg if st.neg_body is not None else l_end} | NOP")
            self._emit(f"{l_tbl}: BR {l_zero if st.zero_body is not None else l_end} | NOP")
            self._emit(f"BR {l_pos if st.pos_body is not None else l_end} | NOP")

            if st.neg_body is not None:
                self._emit(f"{l_neg}:")
                for s in st.neg_body:
                    self._compile_stmt(s)
                self._emit(f"BR {l_end}")
            if st.zero_body is not None:
                self._emit(f"{l_zero}:")
                for s in st.zero_body:
                    self._compile_stmt(s)
                self._emit(f"BR {l_end}")
            if st.pos_body is not None:
                self._emit(f"{l_pos}:")
                for s in st.pos_body:
                    self._compile_stmt(s)
            self._emit(f"{l_end}:")
            return

        if isinstance(st, ReturnStmt):
            assert self._cur_proc is not None
            if st.expr is not None:
                rt = self._compile_expr(st.expr)
                self._require_type(rt, self._cur_proc.ret_type, st.line, f"return in {self._cur_proc.name}")
            else:
                self._emit("ADD S1, S0, #0")
            self._emit_epilogue(len(self._cur_proc.params))
            return

        if isinstance(st, BreakStmt):
            if not self._loop_stack:
                raise SAL3Error(f"Line {st.line}: `break` outside loop")
            self._emit(f"BR {self._loop_stack[-1][1]}")
            return

        if isinstance(st, NextStmt):
            if not self._loop_stack:
                raise SAL3Error(f"Line {st.line}: `next` outside loop")
            self._emit(f"BR {self._loop_stack[-1][0]}")
            return

        if isinstance(st, ExprStmt):
            self._compile_expr(st.expr)
            return

        raise SAL3Error(f"Unhandled statement node {type(st).__name__}")

    def _compile_vec_slice(self, st: VecSliceStmt) -> None:
        """Compile `dst[lo..hi] := a[lo..hi] op b[lo..hi]` into strip-mined C-Complex vector ops!"""
        dst_info = self._lookup_var(st.dst_name, st.line)
        a_info = self._lookup_var(st.a_name, st.line)
        b_info = self._lookup_var(st.b_name, st.line) if st.b_name else None
        if dst_info.typ != a_info.typ or (b_info and b_info.typ != dst_info.typ):
            raise SAL3Error(f"Line {st.line}: Vector slice type mismatch")

        # Evaluate lo -> S5 (index), len = hi - lo + 1 -> S6 (remaining count)
        self._require_type(self._compile_expr(st.lo), "INT", st.line, "slice lo")
        self._emit("PUSH S1")
        self._require_type(self._compile_expr(st.hi), "INT", st.line, "slice hi")
        self._emit("POP S5")         # S5 = lo
        self._emit("SUB S6, S1, S5")
        self._emit("ADD S6, S6, #1")  # S6 = remaining length

        # Compute base pointers + lo into A3 (dst), A4 (a), A5 (b)
        self._emit_array_base("A3", dst_info)
        self._emit("S2A A1, S5")
        self._emit("AADD A3, A3, A1")
        self._emit_array_base("A4", a_info)
        self._emit("AADD A4, A4, A1")
        if b_info:
            self._emit_array_base("A5", b_info)
            self._emit("AADD A5, A5, A1")

        # Strip-mine loop in chunks of VL <= 27 (3^3 C-Complex hardware vector length!)
        l_vloop = self._new_label("VLOOP")
        l_vend = self._new_label("VEND")
        l_chunk_ok = self._new_label("VOK")

        self._emit(f"{l_vloop}:")
        self._emit("CMP S6, S0")
        self._emit(f"BN {l_vend}")
        self._emit(f"BZ {l_vend}")
        # S4 = min(S6, 27)
        self._emit("ADD S4, S6, #0")
        self._emit("CMP S4, #27")
        self._emit(f"BN {l_chunk_ok}")
        self._emit(f"BZ {l_chunk_ok}")
        self._emit("ADD S4, S0, #27")
        self._emit(f"{l_chunk_ok}:")
        self._emit("VSVL S4, #0")
        self._emit("VLD V1, 0(A4)")
        if b_info:
            self._emit("VLD V2, 0(A5)")
        if st.op == "fma":
            self._emit("VLD V3, 0(A3)")
            self._emit("VFMA V3, V1, V2")
            self._emit("VST V3, 0(A3)")
        elif st.op == "copy":
            self._emit("VST V1, 0(A3)")
        elif st.op in ("+", "-", "*"):
            # simm: 0=TF36, +1=T36 int, -1=DB36
            tmode = 0 if dst_info.typ == "FLOAT" else (-1 if dst_info.typ == "DECIBAN" else 1)
            vmn = {"+": "VADD", "-": "VSUB", "*": "VMUL"}[st.op]
            self._emit(f"{vmn} V3, V1, V2, #{tmode}")
            self._emit("VST V3, 0(A3)")
        elif st.op in ("&", "|"):
            kmode = -1 if st.op == "&" else 1
            self._emit(f"VKLEEN V3, V1, V2, #{kmode}")
            self._emit("VST V3, 0(A3)")
        else:
            raise SAL3Error(f"Line {st.line}: Unsupported vector slice op {st.op!r}")

        # Advance pointers & decrement remaining count S6
        self._emit("S2A A1, S4")
        self._emit("AADD A3, A3, A1")
        self._emit("AADD A4, A4, A1")
        if b_info:
            self._emit("AADD A5, A5, A1")
        self._emit("SUB S6, S6, S4")
        self._emit(f"BR {l_vloop}")
        self._emit(f"{l_vend}:")

    def _emit_array_base(self, areg: str, info: VarInfo) -> None:
        if info.scope == "GLOBAL":
            if info.is_array:
                self._emit(f"LEA {areg}, _G_{info.name}(A0)")
            else:
                self._emit(f"LDA {areg}, _G_{info.name}(A0)")
        elif info.is_array:
            self._emit(f"LEA {areg}, {info.offset}(A7)")
        else:
            self._emit(f"LDA {areg}, {info.offset}(A7)")

    def _load_var(self, info: VarInfo, reg: str = "S1") -> None:
        if info.scope == "GLOBAL":
            if info.is_array:
                self._emit(f"LEA A1, _G_{info.name}(A0)")
                self._emit(f"A2S {reg}, A1")
            else:
                self._emit(f"LD {reg}, _G_{info.name}(A0)")
        else:
            if info.is_array:
                self._emit(f"LEA A1, {info.offset}(A7)")
                self._emit(f"A2S {reg}, A1")
            else:
                self._emit(f"LD {reg}, {info.offset}(A7)")

    def _store_var(self, info: VarInfo, reg: str = "S1") -> None:
        if info.scope == "GLOBAL":
            self._emit(f"ST {reg}, _G_{info.name}(A0)")
        else:
            self._emit(f"ST {reg}, {info.offset}(A7)")

    def _compile_index_addr(self, expr: IndexExpr) -> str:
        """Compute element address into S1; returns element type."""
        if isinstance(expr.base, VarExpr):
            info = self._lookup_var(expr.base.name, expr.line)
            self._require_type(self._compile_expr(expr.index), "INT", expr.line, "array index")
            self._emit("S2A A1, S1")
            self._emit_array_base("A2", info)
            self._emit("AADD A1, A2, A1")
            self._emit("A2S S1, A1")
            return info.typ if info.is_array else "INT"
        bt = self._compile_expr(expr.base)
        self._emit("PUSH S1")
        self._require_type(self._compile_expr(expr.index), "INT", expr.line, "array index")
        self._emit("POP S2")
        self._emit("ADD S1, S2, S1")
        return "INT"

    def _emit_typed_binop(self, op: str, typ: str, line: int) -> None:
        """Combine S2 (left) and S1 (right) -> result in S1."""
        if op == "+":
            if typ == "FLOAT":
                self._emit("FAD S1, S2, S1")
            elif typ == "DECIBAN":
                self._emit("DADD S1, S2, S1")
            else:
                self._emit("ADD S1, S2, S1")
        elif op == "-":
            if typ == "FLOAT":
                self._emit("FSB S1, S2, S1")
            elif typ == "DECIBAN":
                self._emit("DSUB S1, S2, S1")
            else:
                self._emit("SUB S1, S2, S1")
        elif op == "*":
            if typ == "FLOAT":
                self._emit("FMP S1, S2, S1")
            elif typ == "INT":
                self._emit("MUL S1, S2, S1")
            else:
                raise SAL3Error(f"Line {line}: Operator `*` not defined on {typ}")
        elif op == "/":
            if typ == "FLOAT":
                self._emit("FDV S1, S2, S1")
            elif typ == "INT":
                self._emit("DIV S1, S2, S1")
            else:
                raise SAL3Error(f"Line {line}: Operator `/` not defined on {typ}")
        elif op == "mod":
            self._require_type(typ, "INT", line, "mod")
            self._emit("MOD S1, S2, S1")
        else:
            raise SAL3Error(f"Line {line}: Unknown arithmetic op {op!r}")

    def _compile_expr(self, expr: object) -> str:
        """Emit code leaving expression value in S1; returns its static SAL-3 type."""
        if isinstance(expr, LitExpr):
            if expr.typ == "INT":
                self._load_const_into("S1", wrap(int(expr.value), WORD_TRITS))
            elif expr.typ == "TRIT":
                self._emit(f"ADD S1, S0, #{int(expr.value)}")
            elif expr.typ == "FLOAT":
                self._load_const_into("S1", float_to_tfloat(float(expr.value), TF36))
            elif expr.typ == "DECIBAN":
                self._load_const_into("S1", db_to_ticks(float(expr.value)))
            elif expr.typ == "ADDR":
                lbl = f"_STR{len(self.string_pool)}"
                self.string_pool.append((lbl, str(expr.value)))
                self._emit(f"LEA A1, {lbl}(A0)")
                self._emit("A2S S1, A1")
            return expr.typ

        if isinstance(expr, VarExpr):
            if expr.name in self.consts:
                t, val = self.consts[expr.name]
                self._load_const_into("S1", val)
                return t
            info = self._lookup_var(expr.name, expr.line)
            self._load_var(info, "S1")
            return "ADDR" if info.is_array else info.typ

        if isinstance(expr, AddrOfExpr):
            info = self._lookup_var(expr.name, expr.line)
            if info.scope == "GLOBAL":
                self._emit(f"LEA A1, _G_{info.name}(A0)")
            else:
                self._emit(f"LEA A1, {info.offset}(A7)")
            self._emit("A2S S1, A1")
            return "ADDR"

        if isinstance(expr, IndexExpr):
            elem_t = self._compile_index_addr(expr)
            self._emit("S2A A1, S1")
            self._emit("LD S1, 0(A1)")
            return elem_t

        if isinstance(expr, UnaryExpr):
            t = self._compile_expr(expr.operand)
            if expr.op == "-":
                if t == "FLOAT":
                    self._emit("FNEG S1, S1")
                else:
                    self._emit("NEG S1, S1")
                return t
            if expr.op == "not":
                if t not in ("TRIT", "INT"):
                    raise SAL3Error(f"Line {expr.line}: `not` expects TRIT or INT, got {t}")
                self._emit("NEG S1, S1")
                return t

        if isinstance(expr, BinaryExpr):
            # McCarthy short-circuit sequential `and` / `or` preserving UNDECIDED (0)
            if expr.op in ("and", "or"):
                lt = self._compile_expr(expr.left)
                self._require_type(lt, "TRIT", expr.line, f"sequential {expr.op}")
                l_done = self._new_label("SC")
                self._emit("TST S1")
                if expr.op == "and":
                    # If left <= 0 (-1 or 0), short-circuit with left!
                    self._emit(f"BN {l_done}")
                    self._emit(f"BZ {l_done}")
                else:
                    # If left >= 0 (+1 or 0), short-circuit with left!
                    self._emit(f"BP {l_done}")
                    self._emit(f"BZ {l_done}")
                rt = self._compile_expr(expr.right)
                self._require_type(rt, "TRIT", expr.line, f"sequential {expr.op}")
                self._emit(f"{l_done}:")
                return "TRIT"

            lt = self._compile_expr(expr.left)
            self._emit("PUSH S1")
            rt = self._compile_expr(expr.right)
            self._emit("POP S2")  # S2 = left, S1 = right

            if expr.op in ("+", "-", "*", "/", "mod"):
                if lt != rt and not ({lt, rt} == {"ADDR", "INT"} and expr.op in ("+", "-")):
                    raise SAL3Error(f"Line {expr.line}: Type mismatch {lt} {expr.op} {rt}")
                res_t = "ADDR" if "ADDR" in (lt, rt) else lt
                self._emit_typed_binop(expr.op, res_t, expr.line)
                return res_t

            if expr.op in ("<<", ">>"):
                self._require_type(lt, "INT", expr.line, "ternary shift lhs")
                self._require_type(rt, "INT", expr.line, "ternary shift amount")
                if expr.op == ">>":
                    self._emit("NEG S1, S1")
                self._emit("SHF S1, S2, S1")
                return "INT"

            if expr.op in ("&", "|"):
                if lt != rt or lt not in ("TRIT", "INT"):
                    raise SAL3Error(f"Line {expr.line}: Kleene `{expr.op}` requires matching TRIT or INT")
                self._emit(("AND" if expr.op == "&" else "OR") + " S1, S2, S1")
                return lt

            if expr.op in _CMP_SEL_TABLE:
                if lt != rt and not ({lt, rt} == {"ADDR", "INT"}):
                    raise SAL3Error(f"Line {expr.line}: Cannot compare {lt} with {rt}")
                if lt == "FLOAT":
                    self._emit("FCM S2, S1")
                else:
                    self._emit("CMP S2, S1")
                tbl = _CMP_SEL_TABLE[expr.op]
                self._emit(f"SEL S1, S0, #{tbl}")
                return "TRIT"

        if isinstance(expr, CallExpr):
            return self._compile_call(expr)

        raise SAL3Error(f"Unhandled expression {type(expr).__name__}")

    def _compile_call(self, call: CallExpr) -> str:
        name = call.name
        args = call.args
        line = call.line

        # ---------------- Native UE-1000 Hardware Intrinsics ----------------
        if name == "eqv":
            lt = self._compile_expr(args[0])
            self._emit("PUSH S1")
            rt = self._compile_expr(args[1])
            self._emit("POP S2")
            if lt != rt:
                raise SAL3Error(f"Line {line}: eqv type mismatch {lt} vs {rt}")
            self._emit("EQV S1, S2, S1")
            return lt

        if name == "abs":
            t = self._compile_expr(args[0])
            self._emit("FABS S1, S1" if t == "FLOAT" else "ABS S1, S1")
            return t

        if name == "flt":
            self._require_type(self._compile_expr(args[0]), "INT", line, "flt")
            self._emit("FLT S1, S1")
            return "FLOAT"

        if name == "fix":
            self._require_type(self._compile_expr(args[0]), "FLOAT", line, "fix")
            self._emit("FIX S1, S1")
            return "INT"

        if name == "fma":
            # fma(acc, a, b) -> single-round TF36 FMA!
            self._require_type(self._compile_expr(args[0]), "FLOAT", line, "fma acc")
            self._emit("PUSH S1")
            self._require_type(self._compile_expr(args[1]), "FLOAT", line, "fma a")
            self._emit("PUSH S1")
            self._require_type(self._compile_expr(args[2]), "FLOAT", line, "fma b")
            self._emit("POP S2")
            self._emit("POP S3")  # acc in S3
            self._emit("FMA S3, S2, S1")
            self._emit("ADD S1, S3, #0")
            return "FLOAT"

        if name == "lse":
            self._require_type(self._compile_expr(args[0]), "DECIBAN", line, "lse")
            self._emit("PUSH S1")
            self._require_type(self._compile_expr(args[1]), "DECIBAN", line, "lse")
            self._emit("POP S2")
            self._emit("DLSE S1, S2, S1")
            return "DECIBAN"

        if name == "dexp":
            self._require_type(self._compile_expr(args[0]), "DECIBAN", line, "dexp")
            self._emit("DEXP S1, S1")
            return "FLOAT"

        if name == "wald":
            # wald(evidence: DECIBAN, threshold: DECIBAN) -> TRIT via hardware DWALD + SEL!
            self._require_type(self._compile_expr(args[0]), "DECIBAN", line, "wald ev")
            self._emit("PUSH S1")
            self._require_type(self._compile_expr(args[1]), "DECIBAN", line, "wald thr")
            self._emit("POP S2")  # S2 = evidence, S1 = threshold ticks
            self._emit("DWALD S2, S1, #0")
            self._emit("SEL S1, S0, #8")  # Identity truth table (T->-1, 0->0, 1->+1)
            return "TRIT"

        if name == "xtr":
            self._require_type(self._compile_expr(args[0]), "INT", line, "xtr val")
            _, p = self._eval_const(args[1])
            _, n = self._eval_const(args[2])
            self._emit(f"XTR S1, S1, #{p}, #{n}")
            return "INT"

        if name == "trit":
            self._require_type(self._compile_expr(args[0]), "INT", line, "trit val")
            self._emit("PUSH S1")
            self._require_type(self._compile_expr(args[1]), "INT", line, "trit pos")
            self._emit("POP S2")
            self._emit("TRT S2, S1")
            self._emit("SEL S1, S0, #8")
            return "TRIT"

        if name in ("vsum", "vmax", "vmin", "vlse"):
            # Strip-mined C-Complex hardware VLD + VRED over up to 27 elements!
            if not isinstance(args[0], VarExpr):
                raise SAL3Error(f"Line {line}: {name} first arg must be array name")
            info = self._lookup_var(args[0].name, line)
            self._require_type(self._compile_expr(args[1]), "INT", line, f"{name} length")
            self._emit("VSVL S1, #0")
            self._emit_array_base("A1", info)
            self._emit("VLD V1, 0(A1)")
            rmode = {"vsum": 0, "vmax": 1, "vmin": -1, "vlse": 2}[name]
            self._emit(f"VRED S1, V1, #{rmode}")
            return "DECIBAN" if name == "vlse" else info.typ

        if name == "tfabric_gemm":
            # tfabric_gemm(A_addr, B_addr, C_addr, M, N, K, eA, eB, eC) -> TRIT status!
            if len(args) != 9:
                raise SAL3Error(f"Line {line}: tfabric_gemm expects 9 args (A, B, C, M, N, K, eA, eB, eC)")
            ttd_lbl = f"_TTD{len(self.ttd_slots)}"
            self.ttd_slots.append(ttd_lbl)
            # Constant-fold dimensions & exponents to build exact 9-word TTD cleanly at runtime!
            vals = []
            for a_expr in args:
                self._compile_expr(a_expr)
                self._emit("PUSH S1")
            # Pop in reverse into stack-allocated slots? Better: write W0..W8 directly!
            # Actually evaluate M,N,K,eA,eB,eC as compile-time constants or runtime shifts:
            # Pop eC, eB, eA, K, N, M, C, B, A cleanly via shifts!
            self._emit(f"LEA A6, {ttd_lbl}(A0)")
            # W0 = mode(+1)*3^33 + zgate(+1)*3^32 + abft(+1)*3^31 + micro_gran(1)*3^23
            w0_val = TileTaskDescriptor(mode=MODE_TBFP9_GEMM, zgate=1, abft=1, micro_gran=1).pack_9_words()[0]
            self._load_const_into("S2", w0_val)
            self._emit("ST S2, 0(A6)")
            # Pop eC, eB, eA -> pack into W5 = eA*3^30 + eB*3^24 + eC*3^18
            self._emit("POP S3")          # eC
            self._emit("SHF S3, S3, #18")
            self._emit("POP S2")          # eB
            self._emit("SHF S2, S2, #24")
            self._emit("ADD S3, S3, S2")
            self._emit("POP S2")          # eA
            self._emit("SHF S2, S2, #30")
            self._emit("ADD S3, S3, S2")
            self._emit("ST S3, 5(A6)")
            # Pop K, N, M -> pack into W4 = M*3^24 + N*3^12 + K
            self._emit("POP S3")          # K
            self._emit("POP S2")          # N
            self._emit("SHF S2, S2, #12")
            self._emit("ADD S3, S3, S2")
            self._emit("POP S2")          # M
            self._emit("SHF S2, S2, #24")
            self._emit("ADD S3, S3, S2")
            self._emit("ST S3, 4(A6)")
            # Pop C, B, A -> pack into W3, W2, W1 (base << 18)
            for w_idx in (3, 2, 1):
                self._emit("POP S2")
                self._emit("SHF S2, S2, #18")
                self._emit(f"ST S2, {w_idx}(A6)")
            # Dispatch across ALL 27 active tiles using Rev. 1.5 S0 shorthand!
            self._emit("TEXEC A6, S0")
            self._emit("TWAIT S0, #0")
            self._emit("SEL S1, S0, #8")  # Return C status trit (+1 OK, 0 TIMEOUT, -1 RECOVERED)
            return "TRIT"

        if name == "clock":
            self._emit("ALGD S1, #0")
            return "INT"

        if name == "abft_count":
            self._emit("TSTAT S1, A0")
            return "INT"

        if name == "peek":
            self._compile_expr(args[0])
            self._emit("S2A A1, S1")
            self._emit("LD S1, 0(A1)")
            return "INT"

        if name == "poke":
            self._compile_expr(args[0])
            self._emit("PUSH S1")
            self._compile_expr(args[1])
            self._emit("POP S2")
            self._emit("S2A A1, S2")
            self._emit("ST S1, 0(A1)")
            return "INT"

        if name in ("print_int", "print_float", "print_trit", "print_db", "putc"):
            svc_map = {"print_int": 1, "print_float": 2, "print_trit": 3, "print_db": 4, "putc": 5}
            self._compile_expr(args[0])
            self._emit(f"SVC #{svc_map[name]}")
            return "INT"

        # ---------------- User Procedure Call ----------------
        if name not in self.procs:
            raise SAL3Error(f"Line {line}: Undeclared procedure {name!r}")
        proc = self.procs[name]
        if len(args) != len(proc.params):
            raise SAL3Error(f"Line {line}: {name} expects {len(proc.params)} args, got {len(args)}")
        for arg, param in zip(args, proc.params):
            at = self._compile_expr(arg)
            self._require_type(at, param.typ, line, f"arg {param.name} of {name}")
            self._emit("PUSH S1")
        self._emit(f"CALL A0, _P_{name}")
        return proc.ret_type


# ---------------------------------------------------------------------------
# Peephole Optimizer & Cray Dual-Parcel 18-Trit Packer (SAL-3/O)
# ---------------------------------------------------------------------------

_SHORT_MNEMS = set(SHORT0_OPS) | set(SHORTT_OPS)
_BRANCH_MNEMS = {"BR", "B3", "BN", "BZ", "BP", "BCT", "RET", "JMP", "J3", "CALL", "HLT", "SVC"}


def _line_mnemonic(line: str) -> str:
    s = line.strip()
    if not s or s.startswith(";") or s.startswith(".") or ":" in s or "|" in s:
        return ""
    return s.split(None, 1)[0].upper().split(".")[0]


def _is_pure_s1_load(instr: str) -> bool:
    """True if `instr` loads S1 without reading S2, S1, or modifying the stack."""
    if instr.startswith("ADD S1, S0, ") or instr.startswith("LDI S1, "):
        return True
    if instr.startswith("LD S1, ") and ("(A7)" in instr or "(A0)" in instr):
        return True
    return False


def _peephole_optimize(lines: List[str]) -> Tuple[List[str], int]:
    """Eliminate stack-temp PUSH/POP pairs, redundant reloads, and dead jumps."""
    out: List[str] = []
    edits = 0
    i = 0
    n = len(lines)
    while i < n:
        cur = lines[i].strip()
        nxt = lines[i + 1].strip() if i + 1 < n else ""
        nxt2 = lines[i + 2].strip() if i + 2 < n else ""

        # Pattern 1a: PUSH S1 ; POP S2 -> ADD S2, S1, #0
        if cur == "PUSH S1" and nxt == "POP S2":
            out.append("ADD S2, S1, #0")
            edits += 1
            i += 2
            continue
        # Pattern 1b (SALISH/O stack-temp elimination):
        # PUSH S1 ; <pure load into S1> ; POP S2  ->  ADD S2, S1, #0 ; <pure load into S1>
        if cur == "PUSH S1" and _is_pure_s1_load(nxt) and nxt2 == "POP S2":
            out.append("ADD S2, S1, #0")
            out.append(lines[i + 1])
            edits += 1
            i += 3
            continue
        # Pattern 2: ST S1, X ; LD S1, X -> ST S1, X (drop redundant reload)
        if cur.startswith("ST S1, ") and nxt == "LD S1, " + cur[7:]:
            out.append(lines[i])
            edits += 1
            i += 2
            continue
        # Pattern 3: BR L ; L: -> L: (drop jump to immediately following label)
        if cur.startswith("BR ") and nxt == cur[3:].strip() + ":":
            out.append(lines[i + 1])
            edits += 1
            i += 2
            continue
        out.append(lines[i])
        i += 1
    return out, edits


def _strip_asm_comment(line: str) -> str:
    idx = line.find(";")
    return (line[:idx] if idx >= 0 else line).strip()


def _pack_dual_parcels(lines: List[str]) -> Tuple[List[str], int]:
    """Cray Dual-Parcel Packer: pair adjacent short 18-trit instructions into `A | B`."""
    out: List[str] = []
    packed = 0
    i = 0
    n = len(lines)
    while i < n:
        s1 = _strip_asm_comment(lines[i])
        m1 = _line_mnemonic(s1)
        # Can pack s1 with s2 if s1 is a non-branching short parcel and s2 is a short parcel
        if m1 in _SHORT_MNEMS and m1 not in _BRANCH_MNEMS and i + 1 < n:
            s2 = _strip_asm_comment(lines[i + 1])
            m2 = _line_mnemonic(s2)
            # Do NOT pack a branch into Parcel B if it is followed by an adjacent B3 jump table
            if m2 in _SHORT_MNEMS and m2 != "B3":
                out.append(f"{s1:<26s} | {s2}")
                packed += 1
                i += 2
                continue
        out.append(lines[i])
        i += 1
    return out, packed


# ---------------------------------------------------------------------------
# Public Compile & Execute API
# ---------------------------------------------------------------------------

def compile_sal3(source: str, opt: bool = True, origin: int = 100) -> CompileResult:
    """Compile SAL-3 source code to TRIAD-36 assembly and assembled UE-1000 image."""
    tokens = tokenize(source)
    prog = Parser(tokens).parse_program()
    return SAL3Compiler(prog, opt=opt, origin=origin).compile()


@dataclass
class SAL3RunResult:
    return_value: int
    return_trit: int
    stdout: List[str]
    cycles: int
    instructions: int
    istack_hits: int
    compile_result: CompileResult


def run_sal3(
    source: str,
    machine: Optional[UE1000Machine] = None,
    opt: bool = True,
    max_steps: int = 500_000,
) -> SAL3RunResult:
    """Compile a SAL-3 program, load it onto UE-1000, and execute `main()` with SVC I/O."""
    comp = compile_sal3(source, opt=opt, origin=100)
    m = machine if machine is not None else UE1000Machine()
    m.load_program(comp.assembled, cpu_id=0)

    stdout: List[str] = []

    def _svc_monitor(cpu, code: int) -> bool:
        val = cpu.s[1]
        if code == 1:
            stdout.append(str(val))
            return True
        if code == 2:
            stdout.append(f"{tfloat_to_float(val, TF36):.6g}")
            return True
        if code == 3:
            stdout.append({1: "TRUE", 0: "UNDECIDED", -1: "FALSE"}.get(val, str(val)))
            return True
        if code == 4:
            stdout.append(f"{ticks_to_db(val):+.2f} dB")
            return True
        if code == 5:
            stdout.append(obb729_to_char(val))
            return True
        return False

    cpu = m.cpu0
    cpu.svc_handler = _svc_monitor
    cpu.run(max_steps=max_steps)
    return SAL3RunResult(
        return_value=cpu.s[1],
        return_trit=cpu.c,
        stdout=stdout,
        cycles=cpu.cycles,
        instructions=cpu.instructions,
        istack_hits=cpu.istack_hits,
        compile_result=comp,
    )
