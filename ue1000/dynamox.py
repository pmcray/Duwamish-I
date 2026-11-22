"""DYNAMO-X — Forrester / Ashby Balanced-Ternary System Dynamics & Cliodynamics Compiler.

Compiles declarative MIT DYNAMO II + Duwamish-I `-X` models directly into native TRIAD-36
assembly executing in exact balanced-ternary TF36 arithmetic (30-trit / 47.55-bit mantissa,
tie-free unbiased rounding) on C-Complex CPUs, with optional multi-region spatial coupling
dispatched onto the 27-tile T-Fabric (`TBFP-9` GEMM with ternary ABFT checksums).

Supported DYNAMO II + `-X` Language Features:
  - Cards:
      * `L VAR.K = VAR.J + DT * (...)`  (Level / state integration; Euler or RK2 midpoint)
      * `N VAR = expr`                  (Initial condition, topologically ordered)
      * `A VAR.K = expr`                (Auxiliary equation, topologically ordered; cycles rejected)
      * `R VAR.KL = expr`               (Rate / flow equation)
      * `C NAME = val`                  (Constant parameter; patchable by RERUN / SWEEP)
      * `T NAME = v0/v1/v2/...`         (Piecewise-linear lookup table for TABHL / TABLE)
      * `SPEC DT=..., LENGTH=..., TSTART=..., PRTPER=..., PLTPER=..., METHOD=EULER|RK2, REGIONS=N`
      * `PRINT V1, V2, ...`             (Tabular time-series output)
      * `PLOT V1=P, V2=R, ...`          (Authentic 1971 line-printer / CRT curve plot)
      * `RERUN title`                   (Policy counterfactual with `C` overrides, zero recompile)
  - Forrester Macros & Built-ins:
      * `TABHL` / `TABLE` (compiled to native TRIAD-36 TF36 interpolation helper `_DYN_TABHL`)
      * `SMOOTH(x, d)` (synthesizes 1st-order exponential level)
      * `DELAY1(x, d)` & `DELAY3(x, d)` (synthesizes 1st- or 3rd-order Forrester Erlang cascade)
      * `CLIP(p, q, a, b)`, `SWITCH(p, q, s)`, `STEP(h, t0)`, `RAMP(s, t0)`, `PULSE(h, t0, per)`
      * `MAX`, `MIN`, `ABS`, `EXP`, `LOG`, `SQRT`, `SIN`, `COS`, `TANH`, `SIGN3`
  - Duwamish-I `-X` Cybernetic Extensions (Forrester + Ashby + Beer):
      * `REGIONS N` & `COUPLE VAR = k` (multi-region Laplacian coupling on T-Fabric TBFP-9!)
      * `ASHBY VAR IN [lo, hi]`        (Essential-variable corridor -> Kleene +1 / 0 / -1 verdict)
      * `SWEEP P1 = a..b STEP s [, P2 = c..d STEP t]` (1D/2D Ashby attractor-basin map!)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .machine import UE1000Machine
from .sal3 import _pack_dual_parcels, _peephole_optimize
from .ternary import WORD_TRITS, in_range, sign_trit, wrap
from .tfloat import TF36, float_to_tfloat, is_undecided, tfloat_to_float
from .triad36 import AssembledProgram, assemble


class DynamoXError(Exception):
    """Syntax, topological-sort, compilation, or runtime error in DYNAMO-X."""


# ---------------------------------------------------------------------------
# Expression Lexer & Recursive-Descent Parser
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DTok:
    kind: str
    val: object


def _tokenize_expr(text: str) -> List[DTok]:
    toks: List[DTok] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in " \t":
            i += 1
            continue
        if text[i : i + 2] == "**":
            toks.append(DTok("OP", "**"))
            i += 2
            continue
        if ch.isdigit() or (ch == "." and i + 1 < n and text[i + 1].isdigit()):
            j = i
            while j < n and text[j].isdigit():
                j += 1
            if j < n and text[j] == ".":
                j += 1
                while j < n and text[j].isdigit():
                    j += 1
            if j < n and text[j] in "eEdD":
                j += 1
                if j < n and text[j] in "+-":
                    j += 1
                while j < n and text[j].isdigit():
                    j += 1
            toks.append(DTok("NUM", float(text[i:j].replace("D", "E").replace("d", "e"))))
            i = j
            continue
        if ch.isalpha() or ch == "_":
            j = i
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            name = text[i:j].upper()
            # Optional DYNAMO time subscript: .K, .J, .JK, .KL
            sub = ""
            if j < n and text[j] == ".":
                m = re.match(r"\.(JK|KL|K|J)\b", text[j:], re.IGNORECASE)
                if m:
                    sub = m.group(1).upper()
                    j += len(m.group(0))
            toks.append(DTok("ID", (name, sub)))
            i = j
            continue
        if ch in "+-*/(),^":
            toks.append(DTok("OP", "**" if ch == "^" else ch))
            i += 1
            continue
        raise DynamoXError(f"Unexpected character {ch!r} in expression {text!r}")
    toks.append(DTok("EOF", None))
    return toks


# Expression AST
@dataclass
class DNum:
    val: float


@dataclass
class DVar:
    name: str
    sub: str = ""


@dataclass
class DUnary:
    op: str
    operand: object


@dataclass
class DBin:
    op: str
    left: object
    right: object


@dataclass
class DCall:
    name: str
    args: List[object]


class _ExprParser:
    def __init__(self, toks: List[DTok]) -> None:
        self.toks = toks
        self.pos = 0

    def peek(self) -> DTok:
        return self.toks[self.pos]

    def match(self, kind: str, val: object = None) -> Optional[DTok]:
        t = self.peek()
        if t.kind == kind and (val is None or t.val == val):
            self.pos += 1
            return t
        return None

    def expect(self, kind: str, val: object = None) -> DTok:
        t = self.peek()
        if t.kind != kind or (val is not None and t.val != val):
            raise DynamoXError(f"Expected {kind} {val or ''}, got {t}")
        self.pos += 1
        return t

    def parse(self) -> object:
        e = self.parse_add()
        if self.peek().kind != "EOF":
            raise DynamoXError(f"Trailing tokens in expression: {self.toks[self.pos:]}")
        return e

    def parse_add(self) -> object:
        left = self.parse_mul()
        while self.peek().kind == "OP" and self.peek().val in ("+", "-"):
            op = str(self.peek().val)
            self.pos += 1
            left = DBin(op, left, self.parse_mul())
        return left

    def parse_mul(self) -> object:
        left = self.parse_pow()
        while self.peek().kind == "OP" and self.peek().val in ("*", "/"):
            op = str(self.peek().val)
            self.pos += 1
            left = DBin(op, left, self.parse_pow())
        return left

    def parse_pow(self) -> object:
        left = self.parse_unary()
        if self.match("OP", "**"):
            return DBin("**", left, self.parse_pow())
        return left

    def parse_unary(self) -> object:
        if self.match("OP", "+"):
            return self.parse_unary()
        if self.match("OP", "-"):
            return DUnary("-", self.parse_unary())
        return self.parse_primary()

    def parse_primary(self) -> object:
        t = self.peek()
        if t.kind == "NUM":
            self.pos += 1
            return DNum(float(t.val))
        if t.kind == "ID":
            self.pos += 1
            name, sub = t.val  # type: ignore[misc]
            if self.match("OP", "("):
                args: List[object] = []
                if not self.match("OP", ")"):
                    while True:
                        args.append(self.parse_add())
                        if self.match("OP", ")"):
                            break
                        self.expect("OP", ",")
                return DCall(name, args)
            return DVar(name, sub)
        if self.match("OP", "("):
            e = self.parse_add()
            self.expect("OP", ")")
            return e
        raise DynamoXError(f"Unexpected token {t} in DYNAMO-X expression")


def _parse_expr(text: str) -> object:
    return _ExprParser(_tokenize_expr(text)).parse()


def _collect_vars(expr: object, out: Set[str]) -> None:
    if isinstance(expr, DVar):
        out.add(expr.name)
    elif isinstance(expr, DUnary):
        _collect_vars(expr.operand, out)
    elif isinstance(expr, DBin):
        _collect_vars(expr.left, out)
        _collect_vars(expr.right, out)
    elif isinstance(expr, DCall):
        start = 1 if expr.name in ("TABHL", "TABLE") else 0
        for a in expr.args[start:]:
            _collect_vars(a, out)


def _extract_level_deriv(level_name: str, rhs: object) -> Optional[object]:
    """If rhs has canonical form `L.J + DT * deriv` (or `- DT * deriv`), return `deriv`."""
    if not isinstance(rhs, DBin) or rhs.op not in ("+", "-"):
        return None
    if not (isinstance(rhs.left, DVar) and rhs.left.name == level_name):
        return None
    term = rhs.right
    if isinstance(term, DBin) and term.op == "*":
        if isinstance(term.left, DVar) and term.left.name == "DT":
            return term.right if rhs.op == "+" else DUnary("-", term.right)
        if isinstance(term.right, DVar) and term.right.name == "DT":
            return term.left if rhs.op == "+" else DUnary("-", term.left)
    return None


# ---------------------------------------------------------------------------
# Parsed DYNAMO-X Model Specification
# ---------------------------------------------------------------------------

@dataclass
class AshbyCorridor:
    var: str
    lo: Optional[float] = None
    hi: Optional[float] = None


@dataclass
class SweepAxis:
    param: str
    start: float
    stop: float
    step: float

    def values(self) -> List[float]:
        if self.step == 0:
            raise DynamoXError(f"SWEEP step for {self.param} cannot be 0")
        out: List[float] = []
        v = self.start
        guard = 0
        if self.step > 0:
            while v <= self.stop + abs(self.step) * 1e-6 and guard < 200:
                out.append(round(v, 6))
                v += self.step
                guard += 1
        else:
            while v >= self.stop - abs(self.step) * 1e-6 and guard < 200:
                out.append(round(v, 6))
                v += self.step
                guard += 1
        return out


@dataclass
class RerunSpec:
    name: str
    overrides: Dict[str, float] = field(default_factory=dict)
    spec_overrides: Dict[str, object] = field(default_factory=dict)


@dataclass
class DynamoModel:
    title: str = "DYNAMO-X MODEL"
    constants: Dict[str, float] = field(default_factory=dict)
    tables: Dict[str, List[float]] = field(default_factory=dict)
    levels: Dict[str, object] = field(default_factory=dict)       # name -> RHS expr
    inits: Dict[str, object] = field(default_factory=dict)        # name -> init expr
    auxs: Dict[str, object] = field(default_factory=dict)         # name -> RHS expr
    rates: Dict[str, object] = field(default_factory=dict)        # name -> RHS expr
    level_derivs: Dict[str, Optional[object]] = field(default_factory=dict)
    spec: Dict[str, object] = field(default_factory=lambda: {
        "DT": 1.0,
        "LENGTH": 100.0,
        "TSTART": 0.0,
        "PRTPER": 10.0,
        "PLTPER": 5.0,
        "METHOD": "EULER",
        "REGIONS": 1,
    })
    print_vars: List[str] = field(default_factory=list)
    plot_vars: List[Tuple[str, str]] = field(default_factory=list)   # (var, sym)
    couplings: Dict[str, float] = field(default_factory=dict)        # var -> Laplacian k
    ashby_corridors: List[AshbyCorridor] = field(default_factory=list)
    reruns: List[RerunSpec] = field(default_factory=list)
    sweeps: List[SweepAxis] = field(default_factory=list)
    # Computed during validation/compilation:
    init_order: List[str] = field(default_factory=list)
    eval_order: List[Tuple[str, str]] = field(default_factory=list)  # (kind 'A'|'R', name)
    synthesized_levels: List[str] = field(default_factory=list)


def _strip_sub(lhs: str) -> str:
    return re.sub(r"\.(JK|KL|K|J)$", "", lhs.strip().upper())


def parse_dynamo_source(source: str) -> DynamoModel:
    """Parse authentic DYNAMO II + Duwamish-I `-X` deck into a DynamoModel."""
    model = DynamoModel()
    raw_lines = source.splitlines()

    # Join continuation lines (leading 'X ' or trailing '\')
    logical: List[Tuple[int, str]] = []
    for lineno, raw in enumerate(raw_lines, start=1):
        s = raw.rstrip("\r\n")
        if not s.strip():
            continue
        if logical and (s.startswith("X ") or s.startswith("x ") or s.startswith("X\t")):
            logical[-1] = (logical[-1][0], logical[-1][1] + " " + s[1:].strip())
            continue
        if logical and logical[-1][1].endswith("\\"):
            logical[-1] = (logical[-1][0], logical[-1][1][:-1] + " " + s.strip())
            continue
        logical.append((lineno, s.strip()))

    active_rerun: Optional[RerunSpec] = None
    first_title_set = False

    for lineno, line in logical:
        # Strip inline semicolon/hash comment outside quoted strings
        if line.startswith("*") or line.startswith("#") or line.startswith(";"):
            if line.startswith("*") and not first_title_set:
                t = line.lstrip("* ").strip()
                if t:
                    model.title = t
                    first_title_set = True
            continue
        if ";" in line:
            line = line.split(";", 1)[0].strip()
            if not line:
                continue

        upper = line.upper()
        parts = line.split(None, 1)
        kw = parts[0].upper()
        rest = parts[1].strip() if len(parts) > 1 else ""

        if kw in ("NOTE", "REM"):
            if not first_title_set and rest:
                model.title = rest
                first_title_set = True
            continue

        if kw == "RUN":
            if rest and not first_title_set:
                model.title = rest
                first_title_set = True
            continue

        if kw == "RERUN":
            active_rerun = RerunSpec(name=rest or f"RERUN_{len(model.reruns) + 1}")
            model.reruns.append(active_rerun)
            continue

        if kw == "SPEC":
            for item in rest.split(","):
                item = item.strip()
                if not item or "=" not in item:
                    continue
                k, v = item.split("=", 1)
                k = k.strip().upper()
                v = v.strip().upper()
                if k == "METHOD":
                    if v not in ("EULER", "RK2"):
                        raise DynamoXError(f"Line {lineno}: SPEC METHOD must be EULER or RK2, got {v!r}")
                    val_obj: object = v
                elif k == "REGIONS":
                    val_obj = int(float(v))
                    if val_obj < 1 or val_obj > 27:
                        raise DynamoXError(f"Line {lineno}: SPEC REGIONS must be in 1..27")
                else:
                    val_obj = float(v)
                if active_rerun is not None:
                    active_rerun.spec_overrides[k] = val_obj
                else:
                    model.spec[k] = val_obj
            continue

        if kw == "PRINT":
            for v in re.split(r"[,\s]+", rest):
                v = _strip_sub(v)
                if v and v not in model.print_vars:
                    model.print_vars.append(v)
            continue

        if kw == "PLOT":
            for item in rest.split(","):
                item = item.strip()
                if not item:
                    continue
                if "=" in item:
                    v, sym = item.split("=", 1)
                    vn = _strip_sub(v)
                    sc = sym.strip()[0] if sym.strip() else vn[0]
                else:
                    vn = _strip_sub(item)
                    sc = vn[0]
                model.plot_vars.append((vn, sc))
            continue

        if kw == "COUPLE":
            if "=" not in rest:
                raise DynamoXError(f"Line {lineno}: COUPLE expects `VAR = strength`")
            lhs, rhs = rest.split("=", 1)
            model.couplings[_strip_sub(lhs)] = float(rhs.strip())
            continue

        if kw == "ASHBY":
            # ASHBY VAR IN [lo, hi]  OR  ASHBY VAR >= lo  OR  ASHBY VAR <= hi
            m_in = re.match(
                r"^([A-Za-z_][A-Za-z0-9_.]*)\s+IN\s*\[\s*([^,\]]+)\s*,\s*([^\]]+)\s*\]$",
                rest,
                re.IGNORECASE,
            )
            m_ge = re.match(r"^([A-Za-z_][A-Za-z0-9_.]*)\s*>=\s*(.+)$", rest, re.IGNORECASE)
            m_le = re.match(r"^([A-Za-z_][A-Za-z0-9_.]*)\s*<=\s*(.+)$", rest, re.IGNORECASE)
            if m_in:
                model.ashby_corridors.append(
                    AshbyCorridor(_strip_sub(m_in.group(1)), float(m_in.group(2)), float(m_in.group(3)))
                )
            elif m_ge:
                model.ashby_corridors.append(AshbyCorridor(_strip_sub(m_ge.group(1)), lo=float(m_ge.group(2))))
            elif m_le:
                model.ashby_corridors.append(AshbyCorridor(_strip_sub(m_le.group(1)), hi=float(m_le.group(2))))
            else:
                raise DynamoXError(f"Line {lineno}: Invalid ASHBY corridor syntax: {line!r}")
            continue

        if kw == "SWEEP":
            # SWEEP P1 = a..b STEP s [, P2 = c..d STEP t]
            for seg in rest.split(","):
                m = re.match(
                    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([+-]?\d+(?:\.\d+)?)\s*\.\.\s*([+-]?\d+(?:\.\d+)?)\s+STEP\s+([+-]?\d+(?:\.\d+)?)\s*$",
                    seg,
                    re.IGNORECASE,
                )
                if not m:
                    raise DynamoXError(f"Line {lineno}: Invalid SWEEP axis: {seg!r}")
                model.sweeps.append(
                    SweepAxis(
                        param=m.group(1).upper(),
                        start=float(m.group(2)),
                        stop=float(m.group(3)),
                        step=float(m.group(4)),
                    )
                )
            continue

        if kw in ("C", "T", "L", "N", "A", "R"):
            if "=" not in rest:
                raise DynamoXError(f"Line {lineno}: Card `{kw}` requires `=`")
            lhs_raw, rhs_raw = rest.split("=", 1)
            name = _strip_sub(lhs_raw)
            rhs_raw = rhs_raw.strip()

            if kw == "C":
                val = _eval_const_expr(_parse_expr(rhs_raw), model.constants)
                if active_rerun is not None:
                    active_rerun.overrides[name] = val
                else:
                    model.constants[name] = val
            elif kw == "T":
                vals = [float(x) for x in re.split(r"[/,\s]+", rhs_raw) if x.strip()]
                if len(vals) < 2:
                    raise DynamoXError(f"Line {lineno}: Table {name} needs at least 2 entries")
                model.tables[name] = vals
            elif kw == "L":
                model.levels[name] = _parse_expr(rhs_raw)
            elif kw == "N":
                model.inits[name] = _parse_expr(rhs_raw)
            elif kw == "A":
                model.auxs[name] = _parse_expr(rhs_raw)
            elif kw == "R":
                model.rates[name] = _parse_expr(rhs_raw)
            continue

        raise DynamoXError(f"Line {lineno}: Unknown DYNAMO-X card {kw!r}: {line!r}")

    # Expand Forrester macros (SMOOTH, DELAY1, DELAY3)
    _expand_forrester_macros(model)

    # Extract canonical derivatives for RK2 midpoint support
    for lname, lexpr in model.levels.items():
        model.level_derivs[lname] = _extract_level_deriv(lname, lexpr)

    # Validate & topologically sort initializers (N) and active equations (A, R)
    _toposort_model(model)
    return model


def _eval_const_expr(e: object, consts: Dict[str, float]) -> float:
    if isinstance(e, DNum):
        return e.val
    if isinstance(e, DVar):
        if e.name in consts:
            return consts[e.name]
        raise DynamoXError(f"Undefined constant {e.name!r} in C card")
    if isinstance(e, DUnary) and e.op == "-":
        return -_eval_const_expr(e.operand, consts)
    if isinstance(e, DBin):
        a = _eval_const_expr(e.left, consts)
        b = _eval_const_expr(e.right, consts)
        if e.op == "+":
            return a + b
        if e.op == "-":
            return a - b
        if e.op == "*":
            return a * b
        if e.op == "/":
            return a / b
        if e.op == "**":
            return a**b
    raise DynamoXError(f"Non-constant expression in C card: {e}")


def _expand_forrester_macros(model: DynamoModel) -> None:
    """Expand SMOOTH, DELAY1, DELAY3 into authentic cascaded Forrester Level equations."""
    seq = [0]

    def rewrite(expr: object) -> object:
        if isinstance(expr, DUnary):
            return DUnary(expr.op, rewrite(expr.operand))
        if isinstance(expr, DBin):
            return DBin(expr.op, rewrite(expr.left), rewrite(expr.right))
        if isinstance(expr, DCall):
            fn = expr.name
            new_args = [rewrite(a) for a in expr.args]
            if fn == "SMOOTH":
                if len(new_args) != 2:
                    raise DynamoXError("SMOOTH(input, delay) takes 2 arguments")
                inp, dtime = new_args
                seq[0] += 1
                lv = f"_SM{seq[0]}"
                model.synthesized_levels.append(lv)
                model.inits[lv] = inp
                model.levels[lv] = DBin(
                    "+",
                    DVar(lv, "J"),
                    DBin("*", DVar("DT"), DBin("/", DBin("-", inp, DVar(lv, "J")), dtime)),
                )
                return DVar(lv, "K")
            if fn == "DELAY1":
                if len(new_args) != 2:
                    raise DynamoXError("DELAY1(input, delay) takes 2 arguments")
                inp, dtime = new_args
                seq[0] += 1
                lv = f"_D1_{seq[0]}"
                model.synthesized_levels.append(lv)
                model.inits[lv] = DBin("*", inp, dtime)
                out_rate = DBin("/", DVar(lv, "J"), dtime)
                model.levels[lv] = DBin(
                    "+",
                    DVar(lv, "J"),
                    DBin("*", DVar("DT"), DBin("-", inp, out_rate)),
                )
                return DBin("/", DVar(lv, "K"), dtime)
            if fn == "DELAY3":
                if len(new_args) != 2:
                    raise DynamoXError("DELAY3(input, delay) takes 2 arguments")
                inp, dtime = new_args
                seq[0] += 1
                base = f"_D3_{seq[0]}"
                dl3 = DBin("/", dtime, DNum(3.0))
                prev_rate: object = inp
                last_lv = ""
                for stage in (1, 2, 3):
                    lv = f"{base}_{stage}"
                    model.synthesized_levels.append(lv)
                    model.inits[lv] = DBin("*", inp, dl3)
                    cur_out = DBin("/", DVar(lv, "J"), dl3)
                    model.levels[lv] = DBin(
                        "+",
                        DVar(lv, "J"),
                        DBin("*", DVar("DT"), DBin("-", prev_rate, cur_out)),
                    )
                    prev_rate = cur_out
                    last_lv = lv
                return DBin("/", DVar(last_lv, "K"), dl3)
            return DCall(fn, new_args)
        return expr

    for d in (model.auxs, model.rates, model.levels):
        for k in list(d.keys()):
            d[k] = rewrite(d[k])


def _toposort_model(model: DynamoModel) -> None:
    """Topologically sort Auxiliaries & Rates (and Initializers), catching algebraic cycles."""
    active_nodes: Dict[str, Tuple[str, object]] = {}
    for name, expr in model.auxs.items():
        active_nodes[name] = ("A", expr)
    for name, expr in model.rates.items():
        active_nodes[name] = ("R", expr)

    state: Dict[str, int] = {}
    order: List[Tuple[str, str]] = []

    def dfs_active(name: str, stack: List[str]) -> None:
        st = state.get(name, 0)
        if st == 2:
            return
        if st == 1:
            cycle = " -> ".join(stack[stack.index(name) :] + [name])
            raise DynamoXError(
                f"Algebraic simultaneous loop detected without a Level or DELAY/SMOOTH: {cycle}"
            )
        state[name] = 1
        kind, expr = active_nodes[name]
        deps: Set[str] = set()
        _collect_vars(expr, deps)
        for dep in sorted(deps):
            if dep in active_nodes:
                dfs_active(dep, stack + [name])
        state[name] = 2
        order.append((kind, name))

    for name in sorted(active_nodes.keys()):
        dfs_active(name, [])
    model.eval_order = order

    # Unified topological sort of initializers: N equations + any A/R equations they depend on
    # (essential for MIT DYNAMO II SMOOTH / DELAY1 / DELAY3 steady-state initialization at T=TSTART!)
    for lname in model.levels:
        if lname not in model.inits:
            model.inits[lname] = DNum(0.0)

    init_nodes: Dict[str, object] = {}
    for name, expr in model.inits.items():
        init_nodes[name] = expr
    for name, expr in model.auxs.items():
        if name not in init_nodes:
            init_nodes[name] = expr
    for name, expr in model.rates.items():
        if name not in init_nodes:
            init_nodes[name] = expr

    init_state: Dict[str, int] = {}
    init_order: List[str] = []

    def dfs_init(name: str, stack: List[str]) -> None:
        st = init_state.get(name, 0)
        if st == 2:
            return
        if st == 1:
            cycle = " -> ".join(stack[stack.index(name) :] + [name])
            raise DynamoXError(f"Circular initial-value (N) dependency: {cycle}")
        init_state[name] = 1
        expr = init_nodes[name]
        deps: Set[str] = set()
        _collect_vars(expr, deps)
        for dep in sorted(deps):
            if dep in init_nodes:
                dfs_init(dep, stack + [name])
        init_state[name] = 2
        init_order.append(name)

    for name in sorted(model.inits.keys()):
        dfs_init(name, [])
    model.init_order = init_order


# ---------------------------------------------------------------------------
# Native TRIAD-36 Code Generator (Exact TF36 + Multi-Region T-Fabric Hook)
# ---------------------------------------------------------------------------

_DYN_MATH_SVCS = {
    "SQRT": 101,
    "EXP": 102,
    "LOG": 103,
    "LN": 103,
    "LOG10": 104,
    "SIN": 105,
    "COS": 106,
    "TANH": 107,
    "ATAN": 108,
}


@dataclass
class DynamoCompileResult:
    model: DynamoModel
    assembly: str
    assembled: AssembledProgram
    peephole_edits: int
    parcels_packed: int
    var_addrs: Dict[str, int]
    const_addrs: Dict[str, int]
    couple_addrs: Dict[str, int]


class DynamoXCompiler:
    """Compiles a parsed DynamoModel directly into native UE-1000 TRIAD-36 assembly."""

    def __init__(self, model: DynamoModel, opt: bool = True, origin: int = 100) -> None:
        self.model = model
        self.opt = opt
        self.origin = origin
        self.code: List[str] = []
        self.const_pool: Dict[int, str] = {}
        self._lbl_seq = 0
        self.n_regions = int(model.spec.get("REGIONS", 1))
        self.method = str(model.spec.get("METHOD", "EULER")).upper()

        # Classify every state/computed variable as per-region vs scalar constant
        self.per_region_vars: Set[str] = (
            set(model.levels.keys()) | set(model.auxs.keys()) | set(model.rates.keys())
        )
        self.coupled_vars: List[str] = []
        if self.n_regions > 1:
            if "ALL" in model.couplings:
                self.coupled_vars = sorted(
                    v for v in model.levels.keys() if not v.startswith("_")
                )
            else:
                self.coupled_vars = sorted(
                    v for v in model.couplings.keys() if v in model.levels
                )

    def _lbl(self, pfx: str = "DX") -> str:
        self._lbl_seq += 1
        return f"_{pfx}{self._lbl_seq}"

    def _emit(self, line: str) -> None:
        self.code.append(line)

    def _load_tf36(self, reg: str, value: float) -> None:
        raw = float_to_tfloat(value, TF36)
        if in_range(raw, 7):
            self._emit(f"ADD {reg}, S0, #{raw}")
        elif in_range(raw, 22):
            self._emit(f"LDI {reg}, #{raw}")
        else:
            if raw not in self.const_pool:
                self.const_pool[raw] = f"_DK{len(self.const_pool)}"
            self._emit(f"LD {reg}, {self.const_pool[raw]}(A0)")

    def _load_int(self, reg: str, value: int) -> None:
        w = wrap(value, WORD_TRITS)
        if in_range(w, 7):
            self._emit(f"ADD {reg}, S0, #{w}")
        elif in_range(w, 22):
            self._emit(f"LDI {reg}, #{w}")
        else:
            if w not in self.const_pool:
                self.const_pool[w] = f"_DK{len(self.const_pool)}"
            self._emit(f"LD {reg}, {self.const_pool[w]}(A0)")

    def _compile_expr(self, expr: object, use_saved_levels: bool = False) -> None:
        """Compile expression to TF36 result in S1 (A5 holds current region index 0..R-1)."""
        if isinstance(expr, DNum):
            self._load_tf36("S1", expr.val)
            return

        if isinstance(expr, DVar):
            name = expr.name
            if name in ("TIME", "DT"):
                self._emit(f"LD S1, _V_{name}(A0)")
                return
            if name in self.model.constants:
                self._emit(f"LD S1, _V_{name}(A0)")
                return
            if name in self.per_region_vars:
                if use_saved_levels and name in self.model.levels:
                    self._emit(f"LD S1, _SV_{name}(A5)")
                else:
                    self._emit(f"LD S1, _V_{name}(A5)")
                return
            if name in self.model.inits:
                self._emit(f"LD S1, _V_{name}(A5)")
                return
            raise DynamoXError(f"Undeclared variable {name!r} in equation")

        if isinstance(expr, DUnary):
            self._compile_expr(expr.operand, use_saved_levels)
            if expr.op == "-":
                self._emit("FNEG S1, S1")
            return

        if isinstance(expr, DBin):
            self._compile_expr(expr.left, use_saved_levels)
            self._emit("PUSH S1")
            self._compile_expr(expr.right, use_saved_levels)
            self._emit("POP S2")  # S2 = left, S1 = right
            if expr.op == "+":
                self._emit("FAD S1, S2, S1")
            elif expr.op == "-":
                self._emit("FSB S1, S2, S1")
            elif expr.op == "*":
                self._emit("FMP S1, S2, S1")
            elif expr.op == "/":
                self._emit("FDV S1, S2, S1")
            elif expr.op == "**":
                self._emit("SVC #111")
            return

        if isinstance(expr, DCall):
            fn = expr.name
            args = expr.args

            if fn in _DYN_MATH_SVCS:
                if len(args) != 1:
                    raise DynamoXError(f"{fn} takes 1 argument")
                self._compile_expr(args[0], use_saved_levels)
                self._emit(f"SVC #{_DYN_MATH_SVCS[fn]}")
                return

            if fn == "ABS":
                self._compile_expr(args[0], use_saved_levels)
                self._emit("FABS S1, S1")
                return

            if fn == "SIGN3":
                # Native Rev. 1.5 FCM against 0 -> balanced trit {-1, 0, +1} as TF36!
                self._compile_expr(args[0], use_saved_levels)
                self._emit("FCM S1, S0")
                self._emit("SEL S1, S0, #8")
                self._emit("FLT S1, S1")
                return

            if fn in ("MAX", "MIN"):
                self._compile_expr(args[0], use_saved_levels)
                self._emit("PUSH S1")
                self._compile_expr(args[1], use_saved_levels)
                self._emit("POP S2")  # S2 = a, S1 = b
                self._emit("FCM S2, S1")
                lbl_done = self._lbl("MM")
                if fn == "MAX":
                    self._emit(f"BN {lbl_done}")   # a < b -> keep S1 (b)
                else:
                    self._emit(f"BP {lbl_done}")   # a > b -> keep S1 (b)
                self._emit("ADD S1, S2, #0")
                self._emit(f"{lbl_done}:")
                return

            if fn in ("CLIP", "FIFGE"):
                # CLIP(p, q, a, b) -> p if a >= b else q
                if len(args) != 4:
                    raise DynamoXError(f"{fn}(p, q, a, b) takes 4 arguments")
                self._compile_expr(args[2], use_saved_levels)
                self._emit("PUSH S1")
                self._compile_expr(args[3], use_saved_levels)
                self._emit("POP S2")  # S2 = a, S1 = b
                self._emit("FCM S2, S1")
                l_q = self._lbl("CLQ")
                l_end = self._lbl("CLE")
                self._emit(f"BN {l_q}")
                self._compile_expr(args[0], use_saved_levels)
                self._emit(f"BR {l_end}")
                self._emit(f"{l_q}:")
                self._compile_expr(args[1], use_saved_levels)
                self._emit(f"{l_end}:")
                return

            if fn == "SWITCH":
                # SWITCH(p, q, s) -> p if s == 0 else q (classic MIT DYNAMO II)
                if len(args) != 3:
                    raise DynamoXError("SWITCH(p, q, s) takes 3 arguments")
                self._compile_expr(args[2], use_saved_levels)
                self._emit("FCM S1, S0")
                l_p = self._lbl("SWP")
                l_end = self._lbl("SWE")
                self._emit(f"BZ {l_p}")
                self._compile_expr(args[1], use_saved_levels)
                self._emit(f"BR {l_end}")
                self._emit(f"{l_p}:")
                self._compile_expr(args[0], use_saved_levels)
                self._emit(f"{l_end}:")
                return

            if fn == "STEP":
                # STEP(height, stime) -> height if TIME >= stime else 0
                if len(args) != 2:
                    raise DynamoXError("STEP(height, stime) takes 2 arguments")
                self._compile_expr(args[1], use_saved_levels)
                self._emit("LD S2, _V_TIME(A0)")
                self._emit("FCM S2, S1")
                l_zero = self._lbl("ST0")
                l_end = self._lbl("STE")
                self._emit(f"BN {l_zero}")
                self._compile_expr(args[0], use_saved_levels)
                self._emit(f"BR {l_end}")
                self._emit(f"{l_zero}:")
                self._emit("ADD S1, S0, #0")
                self._emit(f"{l_end}:")
                return

            if fn == "RAMP":
                # RAMP(slope, rtime) -> slope * (TIME - rtime) if TIME > rtime else 0
                if len(args) != 2:
                    raise DynamoXError("RAMP(slope, rtime) takes 2 arguments")
                self._compile_expr(args[1], use_saved_levels)
                self._emit("LD S2, _V_TIME(A0)")
                self._emit("FSB S2, S2, S1")
                self._emit("FCM S2, S0")
                l_zero = self._lbl("RM0")
                l_end = self._lbl("RME")
                self._emit(f"BN {l_zero}")
                self._emit(f"BZ {l_zero}")
                self._emit("PUSH S2")
                self._compile_expr(args[0], use_saved_levels)
                self._emit("POP S2")
                self._emit("FMP S1, S1, S2")
                self._emit(f"BR {l_end}")
                self._emit(f"{l_zero}:")
                self._emit("ADD S1, S0, #0")
                self._emit(f"{l_end}:")
                return

            if fn in ("TABHL", "TABLE"):
                # TABHL(tname, x, xlo, xhi, xinc)
                if len(args) != 5 or not isinstance(args[0], DVar):
                    raise DynamoXError(f"{fn}(tname, x, xlo, xhi, xinc) requires 5 args with table name first")
                tname = args[0].name
                if tname not in self.model.tables:
                    raise DynamoXError(f"Unknown lookup table {tname!r}")
                n_pts = len(self.model.tables[tname])
                self._compile_expr(args[1], use_saved_levels)  # x
                self._emit("PUSH S1")
                self._compile_expr(args[2], use_saved_levels)  # xlo
                self._emit("PUSH S1")
                self._compile_expr(args[4], use_saved_levels)  # xinc -> S3
                self._emit("ADD S3, S1, #0")
                self._emit("POP S2")                           # xlo -> S2
                self._emit("POP S1")                           # x -> S1
                self._load_int("S4", n_pts - 1)                # max idx -> S4
                self._emit(f"LEA A1, _T_{tname}(A0)")
                self._emit("CALL A0, _DYN_TABHL")
                return

            raise DynamoXError(f"Unknown DYNAMO-X function {fn!r}")

        raise DynamoXError(f"Unhandled AST node {expr}")

    def _emit_region_loop(self, body_fn) -> None:
        """Emit loop over regions A5 = 0..R-1 (zero overhead when R == 1; A5 stays 0!)."""
        if self.n_regions == 1:
            body_fn()
            return
        l_top = self._lbl("RLP")
        l_end = self._lbl("REND")
        self._emit("AADD A5, A0, #0")
        self._emit(f"{l_top}:")
        self._emit("A2S S7, A5")
        self._emit(f"CMP S7, #{self.n_regions}")
        self._emit(f"BZ {l_end}")
        self._emit(f"BP {l_end}")
        body_fn()
        self._emit("AADD A5, A5, #1")
        self._emit(f"BR {l_top}")
        self._emit(f"{l_end}:")

    def _emit_eval_aux_and_rates(self) -> None:
        def _body() -> None:
            for _kind, name in self.model.eval_order:
                expr = self.model.auxs[name] if _kind == "A" else self.model.rates[name]
                self._compile_expr(expr)
                self._emit(f"ST S1, _V_{name}(A5)")
        self._emit_region_loop(_body)

    def _emit_level_euler_step(self, dt_scale: float = 1.0, from_saved: bool = False) -> None:
        """Advance all levels into `_NX_<L>` then commit `_NX_<L> -> _V_<L>`."""
        def _compute() -> None:
            for lname, lexpr in self.model.levels.items():
                deriv = self.model.level_derivs.get(lname)
                if deriv is not None:
                    self._compile_expr(deriv, use_saved_levels=False)
                    # Add inter-regional T-Fabric TBFP-9 coupling flux if coupled!
                    if lname in self.coupled_vars:
                        self._emit(f"LD S2, _CP_{lname}(A5)")
                        self._emit("FAD S1, S1, S2")
                    # Multiply derivative by dt_scale * DT and FMA onto base level
                    self._emit("LD S2, _V_DT(A0)")
                    if dt_scale != 1.0:
                        self._emit("PUSH S1")
                        self._load_tf36("S3", dt_scale)
                        self._emit("FMP S2, S2, S3")
                        self._emit("POP S1")
                    if from_saved:
                        self._emit(f"LD S3, _SV_{lname}(A5)")
                    else:
                        self._emit(f"LD S3, _V_{lname}(A5)")
                    self._emit("FMA S3, S2, S1")
                    self._emit(f"ST S3, _NX_{lname}(A5)")
                else:
                    self._compile_expr(lexpr, use_saved_levels=from_saved)
                    if lname in self.coupled_vars:
                        self._emit(f"LD S2, _CP_{lname}(A5)")
                        self._emit("LD S3, _V_DT(A0)")
                        self._emit("FMA S1, S3, S2")
                    self._emit(f"ST S1, _NX_{lname}(A5)")

        def _commit() -> None:
            for lname in self.model.levels:
                self._emit(f"LD S1, _NX_{lname}(A5)")
                self._emit(f"ST S1, _V_{lname}(A5)")

        self._emit_region_loop(_compute)
        self._emit_region_loop(_commit)

    def _emit_tabhl_helper(self) -> None:
        """Emit native TRIAD-36 TF36 piecewise-linear table interpolator `_DYN_TABHL`."""
        self._emit("; --- Native TF36 TABHL Piecewise-Linear Interpolator ---")
        self._emit("_DYN_TABHL:")
        # In: S1=x, S2=xlo, S3=xinc, S4=n_max (INT), A1=table_base
        self._emit("FSB S1, S1, S2")       # S1 = x - xlo
        self._emit("FDV S1, S1, S3")       # S1 = u = (x - xlo) / xinc
        self._emit("FCM S1, S0")
        self._emit("BN _TAB_LO")
        self._emit("BZ _TAB_LO")
        self._emit("FLT S5, S4")           # S5 = float(n_max)
        self._emit("FCM S1, S5")
        self._emit("BP _TAB_HI")
        self._emit("BZ _TAB_HI")
        # Compute exact floor(u) despite balanced-ternary unbiased FIX rounding!
        self._emit("FIX S6, S1")           # S6 = nearest int(u)
        self._emit("FLT S5, S6")           # S5 = float(S6)
        self._emit("FCM S5, S1")           # if float(S6) > u: S6 = S6 - 1
        self._emit("BN _TAB_FLOK")
        self._emit("BZ _TAB_FLOK")
        self._emit("SUB S6, S6, #1")
        self._emit("FLT S5, S6")
        self._emit("_TAB_FLOK:")
        self._emit("FSB S2, S1, S5")       # S2 = frac = u - floor(u) in [0, 1)
        self._emit("S2A A2, S6")
        self._emit("AADD A1, A1, A2")      # A1 = &table[i]
        self._emit("LD S3, 0(A1)")         # S3 = y0
        self._emit("LD S5, 1(A1)")         # S5 = y1
        self._emit("FSB S5, S5, S3")       # S5 = y1 - y0
        self._emit("FMA S3, S2, S5")       # S3 = y0 + frac * (y1 - y0)
        self._emit("ADD S1, S3, #0")
        self._emit("RET #0")
        self._emit("_TAB_LO:")
        self._emit("LD S1, 0(A1)")
        self._emit("RET #0")
        self._emit("_TAB_HI:")
        self._emit("S2A A2, S4")
        self._emit("AADD A1, A1, A2")
        self._emit("LD S1, 0(A1)")
        self._emit("RET #0")

    def compile(self) -> DynamoCompileResult:
        m = self.model
        self._emit(f"; === DYNAMO-X Compiled TRIAD-36 Image: {m.title} ===")
        self._emit(f"; Method={self.method} | Regions={self.n_regions} | Exact TF36 (47.55-bit mant)")
        self._emit("LDI S8, #200000")
        self._emit("AADD A5, A0, #0")

        # Initialize TIME = TSTART and precompute STOP = LENGTH - 0.25*DT
        self._emit("LD S1, _V_TSTART(A0)")
        self._emit("ST S1, _V_TIME(A0)")
        self._emit("LD S2, _V_LENGTH(A0)")
        self._emit("LD S3, _V_DT(A0)")
        self._load_tf36("S4", 0.25)
        self._emit("FMP S3, S3, S4")
        self._emit("FSB S2, S2, S3")
        self._emit("ST S2, _V_STOP(A0)")

        # Initialize all Levels (and any A/R equations they depend on at T=TSTART!)
        def _init_levels() -> None:
            for name in m.init_order:
                if name in m.inits:
                    expr = m.inits[name]
                elif name in m.auxs:
                    expr = m.auxs[name]
                else:
                    expr = m.rates[name]
                self._compile_expr(expr)
                self._emit(f"ST S1, _V_{name}(A5)")
        self._emit_region_loop(_init_levels)

        # Allow multi-region slight heterogeneity hook via SVC #301 (no-op if R == 1)
        if self.n_regions > 1:
            self._emit("SVC #301")

        # Main time-integration loop
        self._emit("_DYN_LOOP:")
        self._emit_eval_aux_and_rates()

        # Telemetry capture + T-Fabric TBFP-9 inter-regional coupling dispatch!
        self._emit("SVC #300")

        # Stop when TIME >= STOP (LENGTH - 0.25*DT)
        self._emit("LD S1, _V_TIME(A0)")
        self._emit("LD S2, _V_STOP(A0)")
        self._emit("FCM S1, S2")
        self._emit("BP _DYN_DONE")
        self._emit("BZ _DYN_DONE")

        if self.method == "RK2":
            # Save current levels -> _SV_*, half-step -> _V_*, re-eval A/R at t+0.5*DT, full step from _SV_*!
            def _save_lv() -> None:
                for lname in m.levels:
                    self._emit(f"LD S1, _V_{lname}(A5)")
                    self._emit(f"ST S1, _SV_{lname}(A5)")
            self._emit_region_loop(_save_lv)
            self._emit_level_euler_step(dt_scale=0.5, from_saved=False)
            # Advance TIME by 0.5*DT for midpoint evaluation
            self._emit("LD S1, _V_TIME(A0)")
            self._emit("LD S2, _V_DT(A0)")
            self._load_tf36("S3", 0.5)
            self._emit("FMA S1, S2, S3")
            self._emit("ST S1, _V_TIME(A0)")
            self._emit_eval_aux_and_rates()
            if self.coupled_vars:
                self._emit("SVC #302")  # Midpoint T-Fabric coupling update
            self._emit_level_euler_step(dt_scale=1.0, from_saved=True)
            # Advance TIME by remaining 0.5*DT
            self._emit("LD S1, _V_TIME(A0)")
            self._emit("LD S2, _V_DT(A0)")
            self._load_tf36("S3", 0.5)
            self._emit("FMA S1, S2, S3")
            self._emit("ST S1, _V_TIME(A0)")
        else:
            self._emit_level_euler_step(dt_scale=1.0, from_saved=False)
            self._emit("LD S1, _V_TIME(A0)")
            self._emit("LD S2, _V_DT(A0)")
            self._emit("FAD S1, S1, S2")
            self._emit("ST S1, _V_TIME(A0)")

        self._emit("BR _DYN_LOOP")
        self._emit("_DYN_DONE:")
        self._emit("HLT")

        if m.tables:
            self._emit_tabhl_helper()

        # Optimize code segment before appending data directives
        raw_code = list(self.code)
        peephole_edits = 0
        parcels_packed = 0
        if self.opt:
            raw_code, peephole_edits = _peephole_optimize(raw_code)
            raw_code, parcels_packed = _pack_dual_parcels(raw_code)

        # Append static data segment at least 64 words past code end to avoid
        # false 27-word Cray Instruction Stack self-modifying-code invalidation!
        data_origin = self.origin + len(raw_code) + 64
        data_lines: List[str] = [
            "; === DYNAMO-X Static TF36 State, Constants & Lookup Tables ===",
            f".ORG {data_origin}",
        ]
        tstart_val = float(m.spec.get("TSTART", 0.0))
        length_raw = float(m.spec.get("LENGTH", 100.0))
        # Support both MIT DYNAMO end-time (LENGTH > TSTART) and duration (LENGTH <= TSTART)
        end_time_val = length_raw if length_raw > tstart_val else (tstart_val + length_raw)
        spec_words = {
            "TIME": tstart_val,
            "DT": float(m.spec.get("DT", 1.0)),
            "LENGTH": end_time_val,
            "TSTART": tstart_val,
            "STOP": end_time_val,
        }
        for sc_name, v in spec_words.items():
            data_lines.append(f"_V_{sc_name}: .WORD {float_to_tfloat(v, TF36)}")

        for cname, cval in sorted(m.constants.items()):
            data_lines.append(f"_V_{cname}: .WORD {float_to_tfloat(cval, TF36)}")

        zeros = ", ".join(["0"] * self.n_regions)
        for vname in sorted(self.per_region_vars):
            data_lines.append(f"_V_{vname}: .WORD {zeros}")
        for lname in sorted(m.levels.keys()):
            data_lines.append(f"_NX_{lname}: .WORD {zeros}")
            if self.method == "RK2":
                data_lines.append(f"_SV_{lname}: .WORD {zeros}")
        for cv in self.coupled_vars:
            data_lines.append(f"_CP_{cv}: .WORD {zeros}")

        for tname, tvals in sorted(m.tables.items()):
            data_lines.append(f"_T_{tname}:")
            for tv in tvals:
                data_lines.append(f"    .WORD {float_to_tfloat(tv, TF36)}")

        for raw_w, klbl in self.const_pool.items():
            data_lines.append(f"{klbl}: .WORD {raw_w}")

        asm_text = "\n".join(raw_code + data_lines)
        assembled = assemble(asm_text, origin=self.origin)

        var_addrs = {v: assembled.labels[f"_V_{v}"] for v in self.per_region_vars}
        var_addrs["TIME"] = assembled.labels["_V_TIME"]
        const_addrs = {c: assembled.labels[f"_V_{c}"] for c in m.constants}
        const_addrs["DT"] = assembled.labels["_V_DT"]
        const_addrs["LENGTH"] = assembled.labels["_V_LENGTH"]
        const_addrs["TSTART"] = assembled.labels["_V_TSTART"]
        couple_addrs = {cv: assembled.labels[f"_CP_{cv}"] for cv in self.coupled_vars}

        return DynamoCompileResult(
            model=m,
            assembly=asm_text,
            assembled=assembled,
            peephole_edits=peephole_edits,
            parcels_packed=parcels_packed,
            var_addrs=var_addrs,
            const_addrs=const_addrs,
            couple_addrs=couple_addrs,
        )


# ---------------------------------------------------------------------------
# Runtime Execution, Ashby Viability Classifier, Line-Printer Plot & Sweep
# ---------------------------------------------------------------------------

@dataclass
class DynamoTrajectory:
    name: str
    times: List[float]
    series: Dict[str, List[float]]          # var -> region-mean time series
    cycles: int
    instructions: int
    istack_hits: int
    tbfp9_calls: int
    stability_trit: int                     # +1 VIABLE, 0 METASTABLE/UNDECIDED, -1 COLLAPSE
    stability_label: str
    breaches: List[str]
    overrides: Dict[str, float] = field(default_factory=dict)


@dataclass
class DynamoSweepResult:
    axes: List[SweepAxis]
    grid_trits: List[List[int]]             # 1D -> [[t...]], 2D -> rows=axis0, cols=axis1
    grid_ascii: str
    viable_count: int
    metastable_count: int
    collapse_count: int


@dataclass
class DynamoRunResult:
    compile_result: DynamoCompileResult
    runs: List[DynamoTrajectory]
    sweep: Optional[DynamoSweepResult] = None
    rendered_tables: List[str] = field(default_factory=list)
    rendered_plots: List[str] = field(default_factory=list)


def _classify_ashby(
    model: DynamoModel,
    series: Dict[str, List[float]],
) -> Tuple[int, str, List[str]]:
    """Evaluate Ashby essential-variable corridors into native Kleene (+1 / 0 / -1) verdict."""
    if not model.ashby_corridors:
        return 1, "NOMINAL (+1)", []

    breaches: List[str] = []
    ever_breached = False
    end_breached = False
    oscillatory = False

    for corr in model.ashby_corridors:
        vals = series.get(corr.var, [])
        if not vals:
            continue
        if any(math.isnan(v) or math.isinf(v) for v in vals):
            end_breached = True
            breaches.append(f"{corr.var}: UNDECIDED/INF")
            continue

        tail_start = max(0, int(len(vals) * 0.75))
        tail = vals[tail_start:]
        v_end = vals[-1]

        def _out(x: float) -> bool:
            if corr.lo is not None and x < corr.lo:
                return True
            if corr.hi is not None and x > corr.hi:
                return True
            return False

        if _out(v_end):
            end_breached = True
            breaches.append(f"{corr.var}={v_end:.3f} outside [{corr.lo}, {corr.hi}]")
        elif any(_out(v) for v in vals):
            ever_breached = True
            breaches.append(f"{corr.var} transient corridor breach")

        if len(tail) >= 3:
            t_min, t_max = min(tail), max(tail)
            t_mid = max(abs(sum(tail) / len(tail)), 1e-6)
            if (t_max - t_min) / t_mid > 0.12:
                oscillatory = True

    if end_breached:
        return -1, "OVERSHOOT / COLLAPSE (-1)", breaches
    if ever_breached or oscillatory:
        return 0, "METASTABLE / OSCILLATORY (0 UNDECIDED)", breaches
    return 1, "VIABLE EQUILIBRIUM (+1)", breaches


def render_dynamo_table(model: DynamoModel, traj: DynamoTrajectory) -> str:
    cols = model.print_vars or sorted(v for v in model.levels if not v.startswith("_"))[:6]
    prtper = float(model.spec.get("PRTPER", 10.0))
    lines = [
        f"DYNAMO-X TABULAR REPORT — {traj.name}  [{traj.stability_label}]",
        f"  CPU Cycles: {traj.cycles:,} (I-stack hits: {traj.istack_hits:,}) | T-Fabric TBFP-9 calls: {traj.tbfp9_calls}",
        "  " + f"{'TIME':>8s}" + "".join(f"{c:>12s}" for c in cols),
        "  " + "-" * (8 + 12 * len(cols)),
    ]
    next_t = traj.times[0] - 1e-9 if traj.times else 0.0
    for idx, t in enumerate(traj.times):
        if idx == 0 or idx == len(traj.times) - 1 or t >= next_t - 1e-6:
            row = f"  {t:8.2f}" + "".join(f"{traj.series[c][idx]:12.4f}" for c in cols if c in traj.series)
            lines.append(row)
            while next_t <= t + 1e-6:
                next_t += max(prtper, 1e-6)
    return "\n".join(lines)


def render_dynamo_plot(model: DynamoModel, traj: DynamoTrajectory, width: int = 56) -> str:
    """Render an authentic 1971 DYNAMO II line-printer / OBB vector-CRT time-series plot."""
    pvars = model.plot_vars or [(v, v[0]) for v in sorted(model.levels) if not v.startswith("_")][:4]
    if not pvars or not traj.times:
        return ""
    bounds: Dict[str, Tuple[float, float]] = {}
    legend_parts: List[str] = []
    for vn, sym in pvars:
        vals = [x for x in traj.series.get(vn, []) if not (math.isnan(x) or math.isinf(x))]
        lo = min(vals) if vals else 0.0
        hi = max(vals) if vals else 1.0
        if abs(hi - lo) < 1e-9:
            hi = lo + 1.0
        bounds[vn] = (lo, hi)
        legend_parts.append(f"{sym}={vn}[{lo:.2f}..{hi:.2f}]")

    pltper = float(model.spec.get("PLTPER", 5.0))
    out = [
        f"DYNAMO-X PLOT — {traj.name} ({traj.stability_label})",
        "  Legend: " + "  ".join(legend_parts),
        "  TIME    |" + "-" * width + "|",
    ]
    next_t = traj.times[0] - 1e-9
    for idx, t in enumerate(traj.times):
        if idx == 0 or idx == len(traj.times) - 1 or t >= next_t - 1e-6:
            buf = [" "] * width
            for vn, sym in pvars:
                if vn not in traj.series:
                    continue
                val = traj.series[vn][idx]
                if math.isnan(val) or math.isinf(val):
                    continue
                lo, hi = bounds[vn]
                pos = int(round((val - lo) / (hi - lo) * (width - 1)))
                pos = max(0, min(width - 1, pos))
                buf[pos] = sym if buf[pos] == " " else "*"
            out.append(f"  {t:7.1f} |" + "".join(buf) + "|")
            while next_t <= t + 1e-6:
                next_t += max(pltper, 1e-6)
    out.append("          |" + "-" * width + "|")
    return "\n".join(out)


def _execute_single_run(
    comp: DynamoCompileResult,
    machine: UE1000Machine,
    name: str,
    overrides: Dict[str, float],
    spec_overrides: Dict[str, object],
    max_steps: int = 2_000_000,
) -> DynamoTrajectory:
    model = comp.model
    n_reg = int(model.spec.get("REGIONS", 1))

    # Load assembled image onto CPU 0
    machine.load_program(comp.assembled, cpu_id=0)
    cpu = machine.cpu0
    cpu.cycles = 0
    cpu.instructions = 0
    cpu.istack_hits = 0
    cpu.istack_misses = 0

    # Patch RERUN / SWEEP constant & SPEC overrides straight into UE-1000 T-DRAM!
    for cname, cval in overrides.items():
        if cname not in comp.const_addrs:
            raise DynamoXError(f"RERUN/SWEEP override for unknown constant {cname!r}")
        machine.mem.write(comp.const_addrs[cname], float_to_tfloat(cval, TF36), supervisor=True)
    for sk, sv in spec_overrides.items():
        if sk in comp.const_addrs:
            machine.mem.write(comp.const_addrs[sk], float_to_tfloat(float(sv), TF36), supervisor=True)

    # Prebuild inter-regional Laplacian coupling matrix if multi-region
    W_lap: Optional[np.ndarray] = None
    if n_reg > 1:
        W_lap = np.full((n_reg, n_reg), 1.0 / n_reg, dtype=np.float64)
        np.fill_diagonal(W_lap, -(n_reg - 1) / n_reg)

    tracked = sorted(
        set(model.print_vars)
        | {v for v, _ in model.plot_vars}
        | {c.var for c in model.ashby_corridors}
        | {v for v in model.levels if not v.startswith("_")}
    )
    times: List[float] = []
    series: Dict[str, List[float]] = {v: [] for v in tracked}
    tbfp9_calls = [0]

    def _compute_tfabric_coupling() -> None:
        if W_lap is None or not comp.couple_addrs:
            return
        # Stack all coupled state variables into (n_reg x n_vars) matrix -> 1 TBFP-9 GEMM!
        cvars = list(comp.couple_addrs.keys())
        state_mat = np.zeros((n_reg, len(cvars)), dtype=np.float64)
        ks = np.zeros(len(cvars), dtype=np.float64)
        for j, cv in enumerate(cvars):
            base = comp.var_addrs[cv]
            ks[j] = model.couplings.get(cv, model.couplings.get("ALL", 0.0))
            for r in range(n_reg):
                w = machine.mem.read(base + r, supervisor=True)[0]
                state_mat[r, j] = tfloat_to_float(w, TF36)
        flux_mat, _ = machine.tfabric.gemm_tbfp9(W_lap, state_mat, zgate=True, abft=True)
        tbfp9_calls[0] += 1
        for j, cv in enumerate(cvars):
            cbase = comp.couple_addrs[cv]
            k = ks[j]
            for r in range(n_reg):
                machine.mem.write(cbase + r, float_to_tfloat(float(flux_mat[r, j] * k), TF36), supervisor=True)

    def _svc(c, code: int) -> bool:
        if code in (101, 102, 103, 104, 105, 106, 107, 108):
            x = tfloat_to_float(c.s[1], TF36)
            if code == 101:
                y = float("nan") if x < 0 else math.sqrt(x)
            elif code == 102:
                y = math.exp(min(300.0, max(-300.0, x)))
            elif code == 103:
                y = float("nan") if x <= 0 else math.log(x)
            elif code == 104:
                y = float("nan") if x <= 0 else math.log10(x)
            elif code == 105:
                y = math.sin(x)
            elif code == 106:
                y = math.cos(x)
            elif code == 107:
                y = math.tanh(x)
            else:
                y = math.atan(x)
            c.s[1] = float_to_tfloat(y, TF36)
            return True
        if code == 111:
            b = tfloat_to_float(c.s[2], TF36)
            e = tfloat_to_float(c.s[1], TF36)
            try:
                y = b**e
            except Exception:
                y = float("nan")
            c.s[1] = float_to_tfloat(y, TF36)
            return True
        if code == 301 and n_reg > 1:
            # Apply mild regional initial heterogeneity (±4% across regions) so coupling is nontrivial
            for lname in model.levels:
                if lname.startswith("_"):
                    continue
                base = comp.var_addrs[lname]
                for r in range(n_reg):
                    w = machine.mem.read(base + r, supervisor=True)[0]
                    v = tfloat_to_float(w, TF36)
                    scale = 1.0 + 0.04 * (r - (n_reg - 1) / 2.0) / max(1.0, (n_reg - 1) / 2.0)
                    machine.mem.write(base + r, float_to_tfloat(v * scale, TF36), supervisor=True)
            return True
        if code == 300:
            t_w = machine.mem.read(comp.var_addrs["TIME"], supervisor=True)[0]
            times.append(tfloat_to_float(t_w, TF36))
            for vn in tracked:
                base = comp.var_addrs[vn]
                vals = [
                    tfloat_to_float(machine.mem.read(base + r, supervisor=True)[0], TF36)
                    for r in range(n_reg)
                ]
                series[vn].append(sum(vals) / len(vals))
            _compute_tfabric_coupling()
            return True
        if code == 302:
            _compute_tfabric_coupling()
            return True
        return False

    cpu.svc_handler = _svc
    cpu.run(max_steps=max_steps)

    st_trit, st_label, breaches = _classify_ashby(model, series)
    return DynamoTrajectory(
        name=name,
        times=times,
        series=series,
        cycles=cpu.cycles,
        instructions=cpu.instructions,
        istack_hits=cpu.istack_hits,
        tbfp9_calls=tbfp9_calls[0],
        stability_trit=st_trit,
        stability_label=st_label,
        breaches=breaches,
        overrides=dict(overrides),
    )


def _run_ashby_sweep(
    comp: DynamoCompileResult,
    machine: UE1000Machine,
    max_steps: int = 2_000_000,
) -> DynamoSweepResult:
    axes = comp.model.sweeps
    sym_map = {1: "+", 0: "0", -1: "-"}
    viable = meta = coll = 0

    if len(axes) == 1:
        ax = axes[0]
        vals = ax.values()
        row: List[int] = []
        lines = [f"ASHBY-BEER 1D BIFURCATION SWEEP: {ax.param} ({ax.start}..{ax.stop})"]
        for v in vals:
            tr = _execute_single_run(comp, machine, f"{ax.param}={v}", {ax.param: v}, {}, max_steps)
            row.append(tr.stability_trit)
            if tr.stability_trit == 1:
                viable += 1
            elif tr.stability_trit == 0:
                meta += 1
            else:
                coll += 1
            lines.append(f"  {ax.param} = {v:7.3f}  |  [{sym_map[tr.stability_trit]}]  {tr.stability_label}")
        return DynamoSweepResult(axes=axes, grid_trits=[row], grid_ascii="\n".join(lines),
                                 viable_count=viable, metastable_count=meta, collapse_count=coll)

    ax0, ax1 = axes[0], axes[1]
    v0s, v1s = ax0.values(), ax1.values()
    grid: List[List[int]] = []
    lines = [
        f"ASHBY-BEER 2D ATTRACTOR-BASIN MAP: {ax0.param} (rows) x {ax1.param} (cols)",
        "  Legend: [+] VIABLE (+1)   [0] METASTABLE (0 UNDECIDED)   [-] COLLAPSE (-1)",
        f"  {ax0.param:>8s} \\ {ax1.param:<6s} " + " ".join(f"{v:5.2f}" for v in v1s),
        "  " + "-" * (18 + 6 * len(v1s)),
    ]
    for v0 in v0s:
        row = []
        cells = []
        for v1 in v1s:
            tr = _execute_single_run(
                comp, machine, f"{ax0.param}={v0},{ax1.param}={v1}", {ax0.param: v0, ax1.param: v1}, {}, max_steps
            )
            row.append(tr.stability_trit)
            if tr.stability_trit == 1:
                viable += 1
            elif tr.stability_trit == 0:
                meta += 1
            else:
                coll += 1
            cells.append(f"  {sym_map[tr.stability_trit]}  ")
        grid.append(row)
        lines.append(f"  {v0:8.2f}        | " + " ".join(cells))
    return DynamoSweepResult(axes=axes[:2], grid_trits=grid, grid_ascii="\n".join(lines),
                             viable_count=viable, metastable_count=meta, collapse_count=coll)


# ---------------------------------------------------------------------------
# Public Compile & Run API
# ---------------------------------------------------------------------------

def compile_dynamo(source: str, opt: bool = True, origin: int = 100) -> DynamoCompileResult:
    model = parse_dynamo_source(source)
    return DynamoXCompiler(model, opt=opt, origin=origin).compile()


def run_dynamo(
    source: str,
    machine: Optional[UE1000Machine] = None,
    opt: bool = True,
    max_steps: int = 2_000_000,
) -> DynamoRunResult:
    comp = compile_dynamo(source, opt=opt, origin=100)
    m = machine if machine is not None else UE1000Machine()
    runs: List[DynamoTrajectory] = []
    tables: List[str] = []
    plots: List[str] = []

    base_traj = _execute_single_run(comp, m, comp.model.title, {}, {}, max_steps=max_steps)
    runs.append(base_traj)
    if comp.model.print_vars:
        tables.append(render_dynamo_table(comp.model, base_traj))
    if comp.model.plot_vars:
        plots.append(render_dynamo_plot(comp.model, base_traj))

    for rr in comp.model.reruns:
        tr = _execute_single_run(comp, m, rr.name, rr.overrides, rr.spec_overrides, max_steps=max_steps)
        runs.append(tr)
        if comp.model.print_vars:
            tables.append(render_dynamo_table(comp.model, tr))
        if comp.model.plot_vars:
            plots.append(render_dynamo_plot(comp.model, tr))

    sweep_res: Optional[DynamoSweepResult] = None
    if comp.model.sweeps:
        sweep_res = _run_ashby_sweep(comp, m, max_steps=max_steps)

    return DynamoRunResult(
        compile_result=comp,
        runs=runs,
        sweep=sweep_res,
        rendered_tables=tables,
        rendered_plots=plots,
    )
