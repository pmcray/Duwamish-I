# Duwamish-I (`UE-1000`) — Balanced-Ternary Supercomputer Simulator

**Sponsor & Charter Authority:** WEAR (Woodworth's UARC/FFRDC, A. J. Woodworth III University, West Seattle)  
**Mandated Hardware Builder:** Special Systems Section, Universal Entropics  
**Site:** Unseen Level -9, Oral Bland Building (`2` Central Core cells + `28` Structural Cells)  
**Specification:** Conforms strictly to [PRINCIPLES_OF_OPERATION.md](docs/PRINCIPLES_OF_OPERATION.md) (Rev. 1.5) and [PROJECT_PLAN.md](docs/PROJECT_PLAN.md) (Rev. 2).

## Architecture at a Glance
- **Native Balanced-Ternary Hardware (`-1, 0, +1`):** 36-trit full word (`6 × 6-trit` Setun-70/Knuth trytes = `4 × 9-trit` `TBFP-9` lanes = `2 × 18-trit` Setun half-words), stored in Main `T-DRAM` as **42-trit `[42, 36, 4]_3` SEC-DED + triple-erasure** words.
- **Tier 1 (`C-Complex`):** 8 symmetric 36-trit hardwired vector–scalar CPUs (`30 ns` `UE-TECL`, concurrent multi-CPU scheduler), dual-parcel 18-trit / 36-trit ISA (including `VF18OP` 54-float packed vector ops, `2×DB18` decibans, full Ferrite Vault `RTI` trap-frame restore & watchdog `TRAP_TIMER`), exact unbiased `TF36` (`47.55-bit` mantissa) & `TF18`, Kleene three-valued logic, Good Deciban `DLSE`/`DWALD`, and 27-word Cray Instruction Stack.
- **Tier 2 (`T-Fabric`):** 30 liquid-cooled tiles (`27` active + `3` hot spares), $6 \times 6 = 36$ `TBFP-9` MACs/tile (`40 ns`, **48.6 GOPS peak / 17.0 GOPS sustained**), all 6 TTD modes (`ELEM`, `TBFP-9`, `TBFP-3` 12-triads/word, **Q4 1971 Rosenblatt–Kelvanov Byurakan ECO** `TB1` 36-trits/word associative mode, Oliver–Good radix-3 & Good–Thomas PFA radix-6 FFTs), truly asynchronous `TEXEC`/`TWAIT` overlap & timeout, and native ternary ABFT checksums.
- **Tier 3 (`Hierarchical Store & Tier 3b Channels`):** $3^{16} = 43{,}046{,}721$ `T36` words (`81` active + `9` spare `UE-TD2000` differential 2-cell banks, 81-bank crossbar contention model, `292.8 MB` equiv), $3^{11} = 177{,}147$-word non-volatile dual-core ferrite Executive vault, live MMIO apertures, and **8 autonomous `CHIO` DMA Data Channels** linking the 32-spindle Ternary Disc Array, 32 Nine-Track Ternary Tape drives (`READ_BACKWARD`), Jameson's Alexandrian Archival Silo (WEAR provenance enforcement), Channel 8 Oliver SETI Baseband Ring, and Level -8 Digitisation Chute.
- **Tier 4 (`V-Complex`):** Stafford Beer out-of-band Algedonic Telemetry Bus and W. Ross Ashby `< 75 µs` hot-spare tile & bank cutover.

## Quick Start
```bash
# Run the full test suite (hardware, ECC, ISA, workloads, and all 9 languages)
pytest

# Optional: install editable for the `ue1000` console command
pip install -e ".[dev]"

# Print locked Rev. 1.5 architecture summary
python3 -m ue1000 info

# Compile & run SAL-3 programs natively on the simulated UE-1000
python3 -m ue1000 sal3 programs/wald_seti.sal3
python3 -m ue1000 sal3 programs/vec_cliodynamics.sal3 --asm
python3 -m ue1000 sal3 programs/tfabric_gemm.sal3
python3 -m ue1000 sal3 programs/bloop_primes.sal3

# Compile & run TRI-TRAN 70V vectorized FORTRAN-66 programs
python3 -m ue1000 tritran programs/livermore70v.f70
python3 -m ue1000 tritran programs/tritran_tour.f70
python3 -m ue1000 tritran programs/oliver_fft70v.f70

# Compile & run TRILISP-36 classic LISP 1.5 programs natively on UE-1000
python3 -m ue1000 trilisp programs/trilisp_tour.lsp
python3 -m ue1000 trilisp programs/metacircular.lsp
python3 -m ue1000 trilisp programs/symbolic_diff.lsp

# Run CONVERSE procognitive query & interactive steering sessions (or --repl)
python3 -m ue1000 converse programs/obb_session.cv
python3 -m ue1000 converse programs/byurakan_inquiry.cv
python3 -m ue1000 converse --repl

# Compile & run DYNAMO-X Forrester/Ashby system-dynamics & Cliodynamics models
python3 -m ue1000 dynamo programs/world2_forrester.dyn
python3 -m ue1000 dynamo programs/ashby_sweep.dyn
python3 -m ue1000 dynamo programs/predator_delay.dyn --asm

# Run APL\36 balanced-ternary array & tensor scripts (or --repl)
python3 -m ue1000 apl programs/apl_tour.apl
python3 -m ue1000 apl programs/saew_attention.apl
python3 -m ue1000 apl programs/oliver_dft_apl.apl
python3 -m ue1000 apl --repl

# Run REFAL-36 Markov pattern-rewriting & Futamura mixed-computation programs
python3 -m ue1000 refal programs/refal_tour.ref
python3 -m ue1000 refal programs/refal_mix.ref
python3 -m ue1000 refal programs/byurakan_decode.ref

# Run FORTH-3 native TRIAD-36 indirect-threaded Forth (or --repl)
python3 -m ue1000 forth programs/forth_tour.fth
python3 -m ue1000 forth programs/oliver_dish_ctrl.fth
python3 -m ue1000 forth --repl

# Run unsanctioned TRI-INTERCAL grad-student gag scripts (Reel SS-GAG-72 — hide from Brooks!)
python3 -m ue1000 intercal programs/hello_woodworth.i3
python3 -m ue1000 intercal programs/grad_gag_add.i3
python3 -m ue1000 intercal programs/comefrom_1974.i3 --1974

# Run all three anchor workloads (SAEW + SETI + Cliodynamics) & generate the interactive VSM Gallery HTML
python3 -m ue1000 demo all
```

See [docs/SAL3.md](docs/SAL3.md), [docs/TRITRAN70V.md](docs/TRITRAN70V.md), [docs/TRILISP36.md](docs/TRILISP36.md), [docs/CONVERSE.md](docs/CONVERSE.md), [docs/DYNAMOX.md](docs/DYNAMOX.md), [docs/APL36.md](docs/APL36.md), [docs/REFAL36.md](docs/REFAL36.md), [docs/FORTH3.md](docs/FORTH3.md), and [docs/TRI_INTERCAL.md](docs/TRI_INTERCAL.md) for the complete language reference manuals.
