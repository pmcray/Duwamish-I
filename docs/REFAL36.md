# REFAL-36 — Recursive Functions Algorithmic Language Manual

**System:** Duwamish-I (`UE-1000`, Rev. 1.5) — OBB Unseen Level -9  
**Origin:** Valentin F. Turchin (Institute of Applied Mathematics, USSR Academy of Sciences, Moscow, 1966–1971)  
**Documentation Lineage:** Turchin & Serdobol'skii (*Kibernetika* 1969, Plenum English trans. *Cybernetics* Vol. 5, pp. 307–312) and Turchin's 1971 *Programming in the Language Refal* (IPM Preprints Nos. 41, 43, 44, 48, 49 — acquired at the September 1971 Byurakan CETI Conference and archived in Jameson's Alexandrian Vault)  
**Implementation:** [ue1000/refal36.py](../ue1000/refal36.py)

---

## 1. Paradigm & Hardware Integration

Unlike LISP's binary cons-spine or FORTRAN's indexed arrays, **REFAL-36** computes by **associative Markov pattern rewriting over arbitrary parenthesized trees**:

- **Expressions:** Sequences of symbols (`'abc'` characters, `True`/`"text"` words, `T36` integers including `3t1T01` balanced ternary) and structure brackets `( ... )`. Native `OBB-729` Cyrillic (`'СЕТУНЬ'`, `'БЮРАКАН'`), Greek, and Math symbols are first-class characters.
- **Pattern Variables:** `s.X` (one symbol), `t.X` (one term — symbol or whole bracket), `e.X` (any expression, shortest-first backtracking). Repeated variables enforce equality (`s.1 e.2 s.1` recognizes palindromes in one line!).
- **Refal-5 Where-Conditions:** `, expr : pattern` evaluates `expr` and matches `pattern`; if it fails, backtracking automatically returns into the outer pattern to extend `e.` variables. Block conditions `, expr : { ... }` are also supported.
- **Native UE-1000 Hardware Backing:** Arithmetic (`Add`/`Sub`/`Mul`/`Div`/`Mod`), three-way `<Compare>` (`CMP` + `SEL` → `'-'`/`'0'`/`'+'`), Kleene logic (`KleeneAnd`/`Or`/`Eqv`/`Not`), and Good Deciban `<Dlse>`/`<Wald>` run directly on `machine.cpu0` via pre-loaded `TRIAD-36` micro-kernels, while the burial store (`<Br>`/`<Cp>`/`<Dg>`) mirrors tagged entries into `TernaryMemory`.

---

## 2. Built-in Function Reference

| Category | Built-ins |
|---|---|
| **I/O & Deck** | `<Prout e>`, `<Print e>`, `<Card>` *(reads lines after `$END`, `0` at EOF)* |
| **T36 Hardware ALU** | `<Add>`, `<Sub>`, `<Mul>`, `<Div>`, `<Mod>`, `<Compare>` *(`'-'` / `'0'` / `'+'`)* |
| **Conversion & Meta** | `<Numb>`, `<Symb>`, `<Ord>`, `<Chr>`, `<ObbOrd>`, `<ObbChr>`, `<Explode>`, `<Implode>`, `<Lenw>`, `<Type>`, `<First>`, `<Last>`, `<Mu s.F e>`, `<Step>` |
| **Burial Store** | `<Br e.Key '=' e.Val>`, `<Cp e.Key>`, `<Dg e.Key>` |
| **Duwamish-I Ternary** | `<KleeneAnd>`, `<KleeneOr>`, `<KleeneEqv>`, `<KleeneNot>`, `<Trits s.N>`, `<Untrits e>`, `<Hept s.N>`, `<Dlse a b>`, `<Wald ev thresh>` |

---

## 3. CLI & Reference Programs

```bash
python3 -m ue1000 refal programs/refal_tour.ref       # Palindromes, 1969 differentiation, 3-way Qsort, Kleene
python3 -m ue1000 refal programs/refal_mix.ref        # Futamura's 1st Projection (1971) partial evaluator!
python3 -m ue1000 refal programs/byurakan_decode.ref  # Sept 1971 Byurakan CETI trit-grammar decoder
```
