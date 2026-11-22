"""CONVERSE — Pask / Licklider / Good Interactive Procognitive & Evidence Language.

Runs on Tier 4 (V-Complex interactive vector-CRT consoles, OBB Unseen Level -9) and
dispatches real hardware operations across Tiers 1, 2, 3, and 4 of the UE-1000:
  - Alexandrian Vault Associative Search (`FIND` / `ASSOC`): executes Mode +3
    Rosenblatt-Kelvanov Pure-Trit `TB1` inner products on `TFabric.assoc_tb1` over
    J. Ulysses Jameson's curated Level -7/-8/-9 card catalogue.
  - Good Bayesian Epistemic Store (`ASSERT`, `EVIDENCE`, `WEIGH`, `DIAGNOSE`):
    accumulates exact DB36 deciban ticks (81/dB) using saturating `deciban_add`,
    `deciban_lse`, `deciban_exp_float`, and hardware `deciban_wald` (+1/0/-1).
  - Pask Conversation Theory (`ENTAIL`, `TEACHBACK`, `SIGN`): entailment meshes and
    three-valued conversational closure checks across TRUE / UNDECIDED / FALSE topics.
  - Interactive Anchor Workload Steering (`TRAIN SAEW`, `ASK SAEW`, `SCAN SETI`,
    `STEER CLIO`, `STATUS VSM`, `FAULT TILE`): drives real TBFP-9/TBFP-3 SAEW,
    Radix-3 SETI, exact TF36 Cliodynamics, and < 75 us Ashby hot-spare cutover!
"""

from __future__ import annotations

import math
import re
import shlex
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .machine import UE1000Machine
from .tfloat import (
    TF36,
    db_to_ticks,
    deciban_add,
    deciban_exp_float,
    deciban_lse,
    deciban_wald,
    float_to_tfloat,
    tfloat_to_float,
    ticks_to_db,
)
from .tritran70v import format_fortran_record  # noqa: F401 (parity)


class ConverseError(Exception):
    """Syntax or runtime error in a CONVERSE script or terminal session."""


# ---------------------------------------------------------------------------
# J. Ulysses Jameson's Alexandrian Vault Catalogue (36-Trit TB1 Concept Vectors)
# ---------------------------------------------------------------------------

# 36 canonical procognitive indexing axes (Licklider + Jameson MARC subject facets)
CONCEPT_AXES: Tuple[str, ...] = (
    "TERNARY", "SETUN", "LOGIC", "KLEENE", "UNDECIDED", "BAYES",
    "DECIBAN", "WALD", "NEURAL", "PERCEPTRON", "ATTENTION", "SAEW",
    "CYBERNETICS", "ASHBY", "HOMEOSTAT", "BEER", "VSM", "ALGEDONIC",
    "PASK", "CONVERSATION", "LIBRARY", "CATALOGUE", "ALEXANDRIAN", "LICKLIDER",
    "SETI", "BYURAKAN", "RADIO", "FFT", "RADIX3", "CLIODYNAMICS",
    "FORRESTER", "WORLD3", "STIFF", "SUPERCOMPUTER", "CRAY", "LISP",
)
assert len(CONCEPT_AXES) == 36
_AXIS_INDEX = {ax: i for i, ax in enumerate(CONCEPT_AXES)}

# Synonym expansion mapping user query words -> canonical 36-trit concept axes
_SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "TRIT": ("TERNARY", "LOGIC"),
    "TRITS": ("TERNARY", "LOGIC"),
    "TRYTE": ("TERNARY", "SUPERCOMPUTER"),
    "BALANCED": ("TERNARY", "SETUN"),
    "BRUSENTSOV": ("SETUN", "TERNARY"),
    "MOSCOW": ("SETUN", "BYURAKAN"),
    "KELVANOV": ("SETUN", "BYURAKAN", "ATTENTION", "TERNARY"),
    "ROSENBLATT": ("NEURAL", "PERCEPTRON", "ATTENTION", "BYURAKAN"),
    "ASSOCIATIVE": ("ATTENTION", "NEURAL", "SAEW"),
    "TRANSFORMER": ("SAEW", "ATTENTION", "NEURAL"),
    "AUTOREGRESSIVE": ("SAEW", "ATTENTION"),
    "GOOD": ("BAYES", "DECIBAN", "WALD", "FFT"),
    "EVIDENCE": ("BAYES", "DECIBAN", "WALD"),
    "PROBABILITY": ("BAYES", "DECIBAN"),
    "BEACON": ("SETI", "RADIO", "BYURAKAN", "WALD"),
    "PULSAR": ("SETI", "RADIO", "FFT"),
    "OLIVER": ("SETI", "RADIO", "FFT", "RADIX3"),
    "CYCLOPS": ("SETI", "RADIO", "FFT"),
    "SPECTRUM": ("FFT", "RADIX3", "RADIO"),
    "JAMESON": ("LIBRARY", "CATALOGUE", "ALEXANDRIAN"),
    "MICROFORM": ("LIBRARY", "CATALOGUE", "ALEXANDRIAN"),
    "VAULT": ("ALEXANDRIAN", "LIBRARY"),
    "PROCOGNITIVE": ("LICKLIDER", "LIBRARY", "CONVERSATION"),
    "SAKI": ("PASK", "CONVERSATION", "CYBERNETICS"),
    "ENTAILMENT": ("PASK", "CONVERSATION"),
    "ULTRASTABILITY": ("ASHBY", "HOMEOSTAT", "CYBERNETICS"),
    "VIABLE": ("BEER", "VSM", "ALGEDONIC", "CYBERNETICS"),
    "TELEMETRY": ("ALGEDONIC", "BEER", "VSM"),
    "CUTOVER": ("ASHBY", "ALGEDONIC", "VSM"),
    "COLLAPSE": ("CLIODYNAMICS", "WORLD3", "FORRESTER"),
    "DYNAMICS": ("CLIODYNAMICS", "FORRESTER", "WORLD3", "STIFF"),
    "KNUTH": ("TERNARY", "SUPERCOMPUTER", "LOGIC"),
    "BROOKS": ("SUPERCOMPUTER",),
    "SLOTNICK": ("SUPERCOMPUTER", "NEURAL"),
    "MCCARTHY": ("LISP", "LOGIC"),
    "SYMBOLIC": ("LISP", "LOGIC", "CONVERSATION"),
}


@dataclass(frozen=True)
class CatalogueRecord:
    call_no: str
    year: int
    author: str
    title: str
    vault_reel: str
    pos_axes: Tuple[str, ...]
    neg_axes: Tuple[str, ...] = ()

    def to_tb1_vector(self) -> np.ndarray:
        v = np.zeros(36, dtype=np.int8)
        for ax in self.pos_axes:
            if ax in _AXIS_INDEX:
                v[_AXIS_INDEX[ax]] = 1
        for ax in self.neg_axes:
            if ax in _AXIS_INDEX:
                v[_AXIS_INDEX[ax]] = -1
        return v


ALEXANDRIAN_CATALOGUE: Tuple[CatalogueRecord, ...] = (
    CatalogueRecord(
        "QA76.8.S4 B78", 1958, "N. P. Brusentsov",
        "Setun: Balanced-Ternary Small Digital Computer (Moscow State Univ.)",
        "L8-MF-0014", ("TERNARY", "SETUN", "LOGIC", "SUPERCOMPUTER"),
    ),
    CatalogueRecord(
        "QA76.8.S4 K45", 1971, "P. Kelvanov",
        "Byurakan Notes on Setun-70 Sparse-Trit Weights & Three-Valued Associative Retrieval",
        "L7-SC-0109", ("TERNARY", "SETUN", "BYURAKAN", "ATTENTION", "UNDECIDED", "NEURAL"),
    ),
    CatalogueRecord(
        "Q335.G66", 1965, "I. J. Good",
        "Speculations Concerning the First Ultraintelligent Machine & Weight of Evidence in Decibans",
        "L8-MF-0211", ("BAYES", "DECIBAN", "WALD", "LOGIC", "UNDECIDED", "SAEW"),
    ),
    CatalogueRecord(
        "QP363.R67", 1962, "F. Rosenblatt",
        "Principles of Neurodynamics: Perceptrons and the Theory of Brain Mechanisms",
        "L8-MF-0304", ("NEURAL", "PERCEPTRON", "ATTENTION", "SAEW", "BYURAKAN"),
    ),
    CatalogueRecord(
        "Z699.L53", 1965, "J. C. R. Licklider",
        "Libraries of the Future: Procognitive Systems and Symbiotic Consoles",
        "L8-MF-0001", ("LIBRARY", "CATALOGUE", "ALEXANDRIAN", "LICKLIDER", "CONVERSATION"),
    ),
    CatalogueRecord(
        "Z675.U5 J36", 1971, "J. U. Jameson (WEAR Archival Custody)",
        "OBB Level -7/-8 Microform Digitisation Schema & Alexandrian Vault Provenance Standard",
        "L7-SC-0002", ("LIBRARY", "CATALOGUE", "ALEXANDRIAN", "SAEW"),
    ),
    CatalogueRecord(
        "Q310.A84", 1952, "W. R. Ashby",
        "Design for a Brain: Ultrastability, Homeostat & the Law of Requisite Variety",
        "L8-MF-0410", ("CYBERNETICS", "ASHBY", "HOMEOSTAT", "CLIODYNAMICS"),
    ),
    CatalogueRecord(
        "HD38.B36", 1972, "S. Beer",
        "Brain of the Firm: Viable System Model & Algedonic Homeostasis",
        "L8-MF-0418", ("CYBERNETICS", "BEER", "VSM", "ALGEDONIC"),
    ),
    CatalogueRecord(
        "LB1028.5.P37", 1961, "G. Pask",
        "An Approach to Cybernetics: SAKI Adaptive Teaching & Conversation Theory",
        "L8-MF-0422", ("CYBERNETICS", "PASK", "CONVERSATION", "UNDECIDED"),
    ),
    CatalogueRecord(
        "QB54.O45", 1971, "B. M. Oliver & J. Billingham",
        "Project Cyclops: A Design Study of a System for Detecting Extraterrestrial Intelligent Life",
        "L8-MF-0512", ("SETI", "RADIO", "FFT", "RADIX3", "BYURAKAN", "WALD"),
    ),
    CatalogueRecord(
        "QB54.B98", 1971, "Byurakan CETI Proceedings (Soviet Armenia)",
        "First Soviet-American Conference on Communication with Extraterrestrial Intelligence",
        "L7-SC-0519", ("SETI", "BYURAKAN", "RADIO", "DECIBAN", "WALD", "TERNARY"),
    ),
    CatalogueRecord(
        "CB161.F67", 1971, "J. W. Forrester",
        "World Dynamics: Nonlinear Feedback-Loop Global Socio-Ecological Systems",
        "L8-MF-0602", ("CLIODYNAMICS", "FORRESTER", "WORLD3", "STIFF", "CYBERNETICS"),
    ),
    CatalogueRecord(
        "QA76.6.K64", 1969, "D. E. Knuth",
        "The Art of Computer Programming, Vol. 2: Seminumerical Algorithms (Balanced Ternary §4.1)",
        "L8-MF-0701", ("TERNARY", "LOGIC", "SUPERCOMPUTER", "SAEW"),
    ),
    CatalogueRecord(
        "TK7888.3.C73", 1969, "S. R. Cray",
        "Control Data 7600 Computer System: High-Density Freon-Cooled ECL Packaging",
        "L8-MF-0714", ("SUPERCOMPUTER", "CRAY", "STIFF"),
    ),
    CatalogueRecord(
        "QA76.73.L23 M33", 1962, "J. McCarthy et al.",
        "LISP 1.5 Programmer's Manual: Symbolic S-Expressions and Universal EVAL",
        "L8-MF-0750", ("LISP", "LOGIC", "CONVERSATION"),
    ),
    CatalogueRecord(
        "QA9.K54", 1952, "S. C. Kleene",
        "Introduction to Metamathematics: Strong Three-Valued Logic (True / Undecided / False)",
        "L8-MF-0788", ("LOGIC", "KLEENE", "UNDECIDED", "TERNARY"),
    ),
)


def encode_query_tb1(query_text: str) -> np.ndarray:
    """Convert free-text library query into a 36-trit balanced-ternary TB1 query vector."""
    vec = np.zeros(36, dtype=np.float64)
    tokens = re.findall(r"[A-Za-z0-9_-]+", query_text.upper())
    negate_next = False
    for tok in tokens:
        if tok in ("NOT", "WITHOUT", "EXCLUDE", "NON"):
            negate_next = True
            continue
        sign = -1.0 if negate_next else 1.0
        negate_next = False
        if tok in _AXIS_INDEX:
            vec[_AXIS_INDEX[tok]] += sign
        if tok in _SYNONYMS:
            for ax in _SYNONYMS[tok]:
                vec[_AXIS_INDEX[ax]] += sign * 0.8
    return vec


# ---------------------------------------------------------------------------
# Epistemic Fact & Pask Entailment Structures
# ---------------------------------------------------------------------------

@dataclass
class EpistemicClaim:
    name: str
    ticks: int = 0  # Exact DB36 ticks (81 ticks/dB)
    sources: List[str] = field(default_factory=list)

    def verdict(self, threshold_db: float = 20.0) -> int:
        return deciban_wald(self.ticks, reg_ticks=0, simm_db=int(round(threshold_db)))

    @property
    def decibans(self) -> float:
        return ticks_to_db(self.ticks)


@dataclass
class ConverseTranscriptEntry:
    command: str
    output: List[str]


class ConverseSession:
    """Live Tier-4 V-Complex CONVERSE procognitive & cybernetic terminal session."""

    def __init__(self, machine: Optional[UE1000Machine] = None) -> None:
        self.machine = machine if machine is not None else UE1000Machine()
        self.claims: Dict[str, EpistemicClaim] = {}
        self.entailments: Dict[str, Set[str]] = {}  # target -> set of supporting concepts/claims
        self.vars: Dict[str, object] = {}
        self.threshold_db: float = 20.0
        self.transcript: List[ConverseTranscriptEntry] = []

        # Lazy-loaded live Mini-SAEW model (trained on demand via TRAIN SAEW or ASK SAEW)
        self._saew_model = None
        self._saew_report = None

        # Precompute 16 x 36 TB1 pure-trit key matrix for Jameson's Alexandrian Catalogue
        self._cat_keys = np.stack([rec.to_tb1_vector() for rec in ALEXANDRIAN_CATALOGUE], axis=0)

    def _verdict_str(self, trit: int) -> str:
        return {1: "TRUE (+1)", 0: "UNDECIDED (0)", -1: "FALSE (-1)"}[trit]

    def execute_line(self, raw_line: str) -> List[str]:
        """Execute a single CONVERSE command line and return formatted CRT output lines."""
        # Strip -- comments outside quotes
        line = self._strip_comment(raw_line).strip()
        if not line:
            return []

        out = self._dispatch(line)
        self.transcript.append(ConverseTranscriptEntry(command=line, output=out))
        return out

    def run_script(self, source: str) -> List[str]:
        """Execute a multi-line CONVERSE script (.cv), returning all output lines."""
        all_out: List[str] = []
        for lineno, raw in enumerate(source.splitlines(), start=1):
            try:
                all_out.extend(self.execute_line(raw))
            except ConverseError as exc:
                raise ConverseError(f"Line {lineno}: {exc}") from exc
        return all_out

    @staticmethod
    def _strip_comment(line: str) -> str:
        in_q = False
        for i in range(len(line) - 1):
            if line[i] == '"':
                in_q = not in_q
            elif not in_q and line[i : i + 2] == "--":
                return line[:i]
        return line

    def _dispatch(self, line: str) -> List[str]:
        up = line.upper()

        # Conditional execution: IF <CLAIM> IS TRUE|UNDECIDED|FALSE THEN <CMD>
        m_if = re.match(
            r"^IF\s+([A-Za-z0-9_-]+)\s+IS\s+(TRUE|UNDECIDED|FALSE)\s+THEN\s+(.+)$",
            line,
            re.IGNORECASE,
        )
        if m_if:
            cname = m_if.group(1).upper()
            want = {"TRUE": 1, "UNDECIDED": 0, "FALSE": -1}[m_if.group(2).upper()]
            claim = self.claims.get(cname)
            got = claim.verdict(self.threshold_db) if claim else 0
            if got == want:
                return self._dispatch(m_if.group(3).strip())
            return []

        # Three-way SIGN <CLAIM> OF FALSE: <c1> | UNDECIDED: <c2> | TRUE: <c3>
        m_sign = re.match(r"^SIGN\s+([A-Za-z0-9_-]+)\s+OF\s+(.+)$", line, re.IGNORECASE)
        if m_sign:
            cname = m_sign.group(1).upper()
            claim = self.claims.get(cname)
            v = claim.verdict(self.threshold_db) if claim else 0
            arms: Dict[int, str] = {}
            for seg in m_sign.group(2).split("|"):
                m_arm = re.match(r"^\s*(FALSE|NEG|-|UNDECIDED|ZERO|0|TRUE|POS|\+)\s*:\s*(.+)$", seg, re.IGNORECASE)
                if m_arm:
                    lbl = m_arm.group(1).upper()
                    key = -1 if lbl in ("FALSE", "NEG", "-") else (0 if lbl in ("UNDECIDED", "ZERO", "0") else 1)
                    arms[key] = m_arm.group(2).strip()
            if v in arms:
                return self._dispatch(arms[v])
            return []

        # PRINT "..."
        if up.startswith("PRINT "):
            msg = line[6:].strip()
            if msg.startswith('"') and msg.endswith('"'):
                msg = msg[1:-1]
            return [msg]

        # THRESHOLD <val>db
        m_thr = re.match(r"^THRESHOLD\s+([0-9.]+)(?:DB)?$", up)
        if m_thr:
            self.threshold_db = float(m_thr.group(1))
            return [f"WALD THRESHOLD SET TO ±{self.threshold_db:.1f} dB ({db_to_ticks(self.threshold_db)} ticks)"]

        # FIND / ASSOC "<query>" [TOP <k>] -> Rosenblatt-Kelvanov TB1 Associative Search on T-Fabric!
        m_find = re.match(r"^(?:FIND|ASSOC)\s+\"([^\"]+)\"(?:\s+TOP\s+(\d+))?$", line, re.IGNORECASE)
        if m_find:
            return self._cmd_find(m_find.group(1), int(m_find.group(2) or 4))

        # ASSERT <CLAIM> AT <±X.X>db [SOURCE "..."]
        m_ass = re.match(
            r"^ASSERT\s+([A-Za-z0-9_-]+)\s+AT\s+([+-]?[0-9.]+)(?:DB)?(?:\s+SOURCE\s+\"([^\"]+)\")?$",
            line,
            re.IGNORECASE,
        )
        if m_ass:
            cname = m_ass.group(1).upper()
            db_val = float(m_ass.group(2))
            src = m_ass.group(3) or "OBB researcher prior"
            c = EpistemicClaim(name=cname, ticks=db_to_ticks(db_val), sources=[f"{db_val:+.2f} dB ({src})"])
            self.claims[cname] = c
            v = c.verdict(self.threshold_db)
            return [f"CLAIM {cname}: {c.decibans:+.2f} dB -> {self._verdict_str(v)} [{src}]"]

        # EVIDENCE <CLAIM> ADD <±X.X>db [SOURCE "..."]
        m_ev = re.match(
            r"^EVIDENCE\s+([A-Za-z0-9_-]+)\s+ADD\s+([+-]?[0-9.]+)(?:DB)?(?:\s+SOURCE\s+\"([^\"]+)\")?$",
            line,
            re.IGNORECASE,
        )
        if m_ev:
            cname = m_ev.group(1).upper()
            delta_db = float(m_ev.group(2))
            src = m_ev.group(3) or "observation"
            c = self.claims.setdefault(cname, EpistemicClaim(name=cname))
            c.ticks = deciban_add(c.ticks, db_to_ticks(delta_db))
            c.sources.append(f"{delta_db:+.2f} dB ({src})")
            v = c.verdict(self.threshold_db)
            return [
                f"EVIDENCE {cname}: {delta_db:+.2f} dB -> TOTAL {c.decibans:+.2f} dB "
                f"({self._verdict_str(v)}) [{src}]"
            ]

        # WEIGH <CLAIM>
        m_wg = re.match(r"^WEIGH\s+([A-Za-z0-9_-]+)$", line, re.IGNORECASE)
        if m_wg:
            cname = m_wg.group(1).upper()
            c = self.claims.get(cname)
            if c is None:
                return [f"CLAIM {cname}: 0.00 dB -> UNDECIDED (0) [no evidence recorded]"]
            v = c.verdict(self.threshold_db)
            odds = deciban_exp_float(c.ticks)
            odds_s = f"{odds:.3g}:1" if odds < 1e9 else ">1e9:1"
            out = [
                f"WEIGH {cname}: {c.decibans:+.2f} dB (odds {odds_s}, threshold ±{self.threshold_db:.1f} dB) -> {self._verdict_str(v)}"
            ]
            for s in c.sources:
                out.append(f"  + {s}")
            return out

        # COMBINE <DEST> = LSE(<C1>, <C2>)  (Good hardware 243-entry ROM DLSE!)
        m_lse = re.match(
            r"^COMBINE\s+([A-Za-z0-9_-]+)\s*=\s*LSE\(\s*([A-Za-z0-9_-]+)\s*,\s*([A-Za-z0-9_-]+)\s*\)$",
            line,
            re.IGNORECASE,
        )
        if m_lse:
            dst, c1, c2 = m_lse.group(1).upper(), m_lse.group(2).upper(), m_lse.group(3).upper()
            t1 = self.claims[c1].ticks if c1 in self.claims else 0
            t2 = self.claims[c2].ticks if c2 in self.claims else 0
            comb = deciban_lse(t1, t2)
            self.claims[dst] = EpistemicClaim(name=dst, ticks=comb, sources=[f"DLSE({c1}, {c2})"])
            return [f"COMBINE {dst} = DLSE({c1}, {c2}) -> {ticks_to_db(comb):+.2f} dB ({self._verdict_str(self.claims[dst].verdict(self.threshold_db))})"]

        # Pask Conversation Theory: ENTAIL <TARGET> FROM <A>, <B>, ...
        m_ent = re.match(r"^ENTAIL\s+([A-Za-z0-9_-]+)\s+FROM\s+(.+)$", line, re.IGNORECASE)
        if m_ent:
            target = m_ent.group(1).upper()
            deps = [d.strip().upper() for d in m_ent.group(2).split(",") if d.strip()]
            self.entailments.setdefault(target, set()).update(deps)
            return [f"ENTAILMENT MESH: {target} <- {', '.join(sorted(self.entailments[target]))}"]

        # Pask Teachback / Good Active Diagnosis: DIAGNOSE <TARGET>
        m_diag = re.match(r"^DIAGNOSE\s+([A-Za-z0-9_-]+)$", line, re.IGNORECASE)
        if m_diag:
            return self._cmd_diagnose(m_diag.group(1).upper())

        # TRAIN SAEW [STEPS <N>]
        m_tr = re.match(r"^TRAIN\s+SAEW(?:\s+STEPS\s+(\d+))?$", up)
        if m_tr:
            return self._cmd_train_saew(int(m_tr.group(1) or 60))

        # ASK SAEW "<prompt>" [MODE TBFP9|TBFP3]
        m_ask = re.match(r"^ASK\s+SAEW\s+\"([^\"]+)\"(?:\s+MODE\s+(TBFP9|TBFP3))?$", line, re.IGNORECASE)
        if m_ask:
            return self._cmd_ask_saew(m_ask.group(1), (m_ask.group(2) or "TBFP9").lower())

        # SCAN SETI [FRAMES <N>]
        m_seti = re.match(r"^SCAN\s+SETI(?:\s+FRAMES\s+(\d+))?$", up)
        if m_seti:
            return self._cmd_scan_seti(int(m_seti.group(1) or 12))

        # STEER CLIO CONSERVATION <c> ABATEMENT <a> [YEARS <Y>]
        m_clio = re.match(
            r"^STEER\s+CLIO\s+CONSERVATION\s+([0-9.]+)\s+ABATEMENT\s+([0-9.]+)(?:\s+YEARS\s+(\d+))?$",
            up,
        )
        if m_clio:
            return self._cmd_steer_clio(float(m_clio.group(1)), float(m_clio.group(2)), int(m_clio.group(3) or 150))

        # STATUS VSM
        if up == "STATUS VSM":
            return self._cmd_status_vsm()

        # FAULT TILE <k> [REASON "..."]
        m_flt = re.match(r"^FAULT\s+TILE\s+(\d+)(?:\s+REASON\s+\"([^\"]+)\")?$", line, re.IGNORECASE)
        if m_flt:
            return self._cmd_fault_tile(int(m_flt.group(1)), m_flt.group(2) or "CONSOLE_FAULT_DRILL")

        if up in ("HELP", "?"):
            return [
                "CONVERSE COMMANDS (OBB LEVEL -9 V-COMPLEX TERMINAL):",
                "  FIND \"<query>\" [TOP k]                         -- TB1 associative catalogue search",
                "  ASSERT <C> AT <±dB>db [SOURCE \"...\"]          -- Register Bayesian prior",
                "  EVIDENCE <C> ADD <±dB>db [SOURCE \"...\"]       -- Saturating Good deciban update",
                "  WEIGH <C> | COMBINE <D> = LSE(<A>,<B>)         -- Hardware DWALD / DLSE",
                "  ENTAIL <T> FROM <A>,<B> | DIAGNOSE <T>         -- Pask entailment mesh & teachback",
                "  TRAIN SAEW [STEPS n] | ASK SAEW \"...\" [MODE]   -- 1-Tile TBFP-9/TBFP-3 library AI",
                "  SCAN SETI [FRAMES n]                           -- Radix-3 FFT + TRIAD-36 Wald sweep",
                "  STEER CLIO CONSERVATION c ABATEMENT a [YEARS]  -- Live exact TF36 policy probe",
                "  STATUS VSM | FAULT TILE k                      -- Beer Algedonic bus & <75 us cutover",
                "  IF <C> IS TRUE|UNDECIDED|FALSE THEN <cmd>      -- Three-valued conditional",
                "  SIGN <C> OF FALSE: <c1> | UNDECIDED: <c2> | TRUE: <c3>",
            ]

        raise ConverseError(f"Unknown CONVERSE command: {line!r}")

    # ------------------------------------------------------------------
    # Hardware-Backed Command Implementations
    # ------------------------------------------------------------------

    def _cmd_find(self, query: str, top_k: int) -> List[str]:
        q_vec = encode_query_tb1(query)
        # Execute REAL Mode +3 Rosenblatt-Kelvanov TB1 pure-trit associative matching on T-Fabric!
        scores_mat, stats = self.machine.tfabric.assoc_tb1(q_vec[None, :], self._cat_keys)
        scores = scores_mat[0]
        order = np.argsort(-scores)

        out = [
            f"ALEXANDRIAN VAULT TB1 SEARCH \"{query}\" "
            f"({stats.ops} trit-ops, {stats.elapsed_ns} ns, Kelvanov zero-trit save {stats.zero_trit_suppressed_pct:.1f}%):"
        ]
        shown = 0
        for idx in order:
            sc = int(scores[idx])
            if sc <= 0 and shown > 0:
                break
            rec = ALEXANDRIAN_CATALOGUE[int(idx)]
            out.append(
                f"  [{sc:+d}] {rec.call_no:16s} ({rec.year}) {rec.author} — {rec.title} [{rec.vault_reel}]"
            )
            shown += 1
            if shown >= top_k:
                break
        return out

    def _cmd_diagnose(self, target: str) -> List[str]:
        deps = sorted(self.entailments.get(target, set()))
        if not deps:
            c = self.claims.get(target)
            v = c.verdict(self.threshold_db) if c else 0
            return [f"DIAGNOSE {target}: leaf concept -> {self._verdict_str(v)}"]

        # Kleene strong conjunction over entailment supports (min of support verdicts!)
        verdicts: Dict[str, int] = {}
        undecided_gaps: List[Tuple[float, str]] = []
        for d in deps:
            c = self.claims.get(d)
            v = c.verdict(self.threshold_db) if c else 0
            verdicts[d] = v
            if v == 0:
                # Expected remaining decibans needed to cross ±threshold_db (Good active inquiry!)
                cur_db = abs(c.decibans) if c else 0.0
                undecided_gaps.append((self.threshold_db - cur_db, d))

        closure_trit = min(verdicts.values())
        out = [
            f"PASK TEACHBACK / DIAGNOSE {target}: entailment closure = {self._verdict_str(closure_trit)}"
        ]
        for d in deps:
            c = self.claims.get(d)
            db_s = f"{c.decibans:+.2f} dB" if c else "0.00 dB"
            out.append(f"  - {d:24s}: {db_s:>10s} -> {self._verdict_str(verdicts[d])}")
        if undecided_gaps:
            undecided_gaps.sort()
            gap_db, best_q = undecided_gaps[0]
            out.append(
                f"  >> NEXT OPTIMAL INQUIRY (Good E[WoE]): gather {gap_db:.1f} dB on UNDECIDED premise `{best_q}`"
            )
        elif closure_trit == 1:
            out.append(f"  >> PASK CONVERSATIONAL LOOP CLOSED: `{target}` is entailed as TRUE (+1).")
        else:
            out.append(f"  >> REFUTED: at least one required premise for `{target}` is FALSE (-1).")
        return out

    def _cmd_train_saew(self, steps: int) -> List[str]:
        from workloads.saew import MiniSAEW, WOODWORTH_MINI_CARDS, WOODWORTH_MINI_CORPUS

        vocab = sorted(set(WOODWORTH_MINI_CORPUS))
        T = 27
        model = MiniSAEW(self.machine, vocab, d_model=36, d_ff=72, context_len=T, seed=1971)
        seqs = [np.array([model.stoi[ch] for ch in card[: T + 1]], dtype=np.int64) for card in WOODWORTH_MINI_CARDS]
        x_b = np.stack([s[:-1] for s in seqs])
        y_b = np.stack([s[1:] for s in seqs])

        loss0 = 0.0
        loss_last = 0.0
        for s in range(1, steps + 1):
            lr = 0.025 if s <= steps // 2 else 0.012
            loss_last = model.forward_and_backward(x_b, y_b, lr=lr, step_idx=s)
            if s == 1:
                loss0 = loss_last
        self._saew_model = model
        return [
            f"SAEW MINI-MODEL TRAINED IN NATIVE TBFP-9 ({model.param_count:,} params, {steps} steps): "
            f"loss {loss0:.3f} -> {loss_last:.3f} (ABFT checks={self.machine.tfabric.total_abft_checks:,})"
        ]

    def _cmd_ask_saew(self, prompt: str, mode: str) -> List[str]:
        if self._saew_model is None:
            self._cmd_train_saew(65)
        assert self._saew_model is not None
        reply = self._saew_model.generate(prompt.upper(), max_new=24, mode=mode)
        return [f"SAEW [{mode.upper()} · 1-TILE]: {reply}"]

    def _cmd_scan_seti(self, frames: int) -> List[str]:
        from workloads.seti import run_seti_demo

        rep = run_seti_demo(self.machine, n_fft=729, n_frames=frames)
        out = [f"SETI RADIX-3 TBFP-9C SWEEP ({rep.fft_points}-pt FFT x {frames} frames, TRIAD-36 Wald):"]
        for r in rep.results:
            # Automatically log beacon/noise/faint claims into the CONVERSE Bayesian store!
            key = re.sub(r"[^A-Z0-9]+", "_", r.label.upper()).strip("_")[:24]
            self.claims[f"SETI_{key}"] = EpistemicClaim(
                name=f"SETI_{key}",
                ticks=db_to_ticks(r.final_decibans),
                sources=[f"Oliver Radix-3 FFT + Section 8.1 TRIAD-36 ({r.cpu_cycles} cyc)"],
            )
            out.append(f"  - {r.label:46s}: {r.final_decibans:+6.2f} dB -> {r.verdict_name}")
        return out

    def _cmd_steer_clio(self, conservation: float, abatement: float, years: int) -> List[str]:
        from workloads.cliodynamics import _tf36_step_local

        n_reg = 6
        dt = 2.5
        n_steps = max(1, int(years / dt))
        const_vals = [
            "0.0", "1.0", "0.01", "0.35", "0.40", "0.60", "0.068", "0.009", "0.032", "2.60",
            "0.14", "0.055", "-0.028", "0.80", "0.015", "1.20", "0.065", "0.85", "0.09", "0.08", "0.06",
        ]
        tf_consts = {k: float_to_tfloat(float(k), TF36) for k in const_vals}
        dt_tf = float_to_tfloat(dt, TF36)
        cons_tf = float_to_tfloat(conservation, TF36)
        abat_tf = float_to_tfloat(abatement, TF36)
        W_coup = np.full((n_reg, n_reg), 0.004, dtype=np.float64)
        np.fill_diagonal(W_coup, -0.004 * (n_reg - 1))

        init_f = np.array(
            [[1.0 + 0.08 * r, 1.0 + 0.12 * r, 1.4 - 0.05 * r, 0.18 + 0.02 * r, 0.82 - 0.02 * r] for r in range(n_reg)],
            dtype=np.float64,
        )
        state_tf = [[float_to_tfloat(float(init_f[r, v]), TF36) for v in range(5)] for r in range(n_reg)]
        pop_hist: List[float] = []
        peak_pop = 0.0
        for _ in range(n_steps):
            state_f = np.array(
                [[tfloat_to_float(state_tf[r][v], TF36) for v in range(5)] for r in range(n_reg)],
                dtype=np.float64,
            )
            mp = float(np.mean(state_f[:, 0]))
            pop_hist.append(mp)
            peak_pop = max(peak_pop, mp)
            coup, _ = self.machine.tfabric.gemm_tbfp9(W_coup, state_f, zgate=True, abft=True)
            state_tf = [
                _tf36_step_local(state_tf[r], coup[r].tolist(), dt_tf, cons_tf, abat_tf, tf_consts)
                for r in range(n_reg)
            ]

        final_f = np.array(
            [[tfloat_to_float(state_tf[r][v], TF36) for v in range(5)] for r in range(n_reg)],
            dtype=np.float64,
        )
        f_pop = float(np.mean(final_f[:, 0]))
        f_res = float(np.mean(final_f[:, 2]))
        f_leg = float(np.mean(final_f[:, 4]))
        if f_res >= 0.45 and f_leg >= 0.65 and f_pop >= 0.80 * peak_pop:
            stab, lbl = 1, "VIABLE EQUILIBRIUM (+1)"
        elif f_res <= 0.18 or f_pop < 0.65 * peak_pop or f_leg < 0.45:
            stab, lbl = -1, "OVERSHOOT & COLLAPSE (-1)"
        else:
            stab, lbl = 0, "METASTABLE / UNDECIDED (0)"

        # Record policy outcome as an epistemic claim (+25 / 0 / -25 dB)
        self.claims["CLIO_POLICY_VIABLE"] = EpistemicClaim(
            name="CLIO_POLICY_VIABLE",
            ticks=db_to_ticks(25.0 * stab),
            sources=[f"TF36 Cliodynamics (cons={conservation}, abat={abatement}, {years} yrs)"],
        )

        # Render 18-char ASCII vector-CRT sparkline of Population trajectory!
        bars = "._-:=+*#@"
        hi_p = max(max(pop_hist), 1.5)
        idxs = np.linspace(0, len(pop_hist) - 1, 18).astype(int)
        spark = "".join(bars[min(len(bars) - 1, int((pop_hist[i] / hi_p) * (len(bars) - 1)))] for i in idxs)
        return [
            f"CLIODYNAMICS PROBE (cons={conservation:.2f}, abat={abatement:.2f}, {years} yrs, exact TF36 + TBFP-9):",
            f"  POP CRT TRACE [{spark}]  Final Pop={f_pop:.2f}  Res={f_res:.2f}  Legit={f_leg:.2f} -> {lbl}",
        ]

    def _cmd_status_vsm(self) -> List[str]:
        snap = self.machine.telemetry_snapshot()
        tf = snap["t_fabric"]
        mem = snap["memory"]
        vc = snap["v_complex"]
        out = [
            f"VSM ALGEDONIC STATUS (OBB Level -9, 28+2 Cells):",
            f"  T-Fabric : {tf['active_tiles']} active / {tf['hot_spares_remaining']} hot spares | "
            f"ABFT={tf['abft_checks']:,} | Zero-Trit Save={tf['avg_zero_trit_power_saving_pct']:.1f}%",
            f"  T-DRAM   : 43.05M T36 words ({mem['active_banks']}+{mem['spare_banks_remaining']} banks) | "
            f"SEC={mem['total_sec_corrections']} | Erasures={mem['total_erasure_corrections']}",
            f"  Cutovers : {len(vc['cutovers'])} total | Algedonic Alarms={vc['algedonic_alarms']}",
        ]
        for lamp in vc["gallery_lamps"][-3:]:
            out.append(f"  [LAMP] {lamp}")
        return out

    def _cmd_fault_tile(self, logical_tile: int, reason: str) -> List[str]:
        self.machine.tfabric.inject_tile_fault(logical_tile=logical_tile, reason=reason)
        # Trigger immediate 6x6 verification GEMM to exercise ABFT + Ashby < 75 us cutover!
        a = np.eye(6)
        _, stats = self.machine.tfabric.gemm_tbfp9(a, a, abft=True)
        ev = self.machine.vcomplex.cutovers[-1]
        return [
            f"ASHBY CUTOVER EXECUTED ({ev.hw_cutover_us:.1f} us < 75 us): "
            f"Logical T{ev.logical_id:02d} (Bay P{ev.old_physical_id:02d}) -> Spare P{ev.new_physical_id:02d} "
            f"| Spares left={len(self.machine.vcomplex.spare_tiles)} | Status={stats.status_code:+d}"
        ]


def run_converse_script(source: str, machine: Optional[UE1000Machine] = None) -> Tuple[List[str], ConverseSession]:
    """Run a CONVERSE script (.cv) and return (output_lines, live_session)."""
    sess = ConverseSession(machine=machine)
    out = sess.run_script(source)
    return out, sess


def run_converse_repl(machine: Optional[UE1000Machine] = None) -> None:
    """Interactive Level -9 Vector-CRT CONVERSE terminal REPL."""
    sess = ConverseSession(machine=machine)
    print("DUWAMISH-I (UE-1000) · V-COMPLEX CONVERSE TERMINAL (OBB LEVEL -9)")
    print("Type HELP for commands, QUIT to exit.")
    while True:
        try:
            raw = input("CONVERSE> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if raw.strip().upper() in ("QUIT", "EXIT", "BYE"):
            break
        try:
            for ln in sess.execute_line(raw):
                print(ln)
        except Exception as exc:
            print(f"*** ERROR: {exc}")
