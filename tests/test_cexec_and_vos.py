"""Tests for Tier 1 Resident Ferrite Vault C-EXEC & Tier 4 V-OS Supervisory Shell."""

from __future__ import annotations

from pathlib import Path

import pytest

from ue1000.cexec import C_EXEC_MAGIC, CExecHypervisor
from ue1000.cpu import TRAP_PRIV, TRAP_PROT
from ue1000.machine import UE1000Machine
from ue1000.ternary import char_to_obb729
from ue1000.vos import VOSError, VOSSession, run_vos_script


def test_cexec_cold_boot_ipl_and_native_svcs_in_ferrite_vault():
    """Verify native TRIAD-36 C-EXEC boots in Ferrite Vault and serves all User SVCs."""
    m = UE1000Machine()
    cex = CExecHypervisor(m)
    stats = cex.boot()
    assert stats["booted"] == 1
    assert stats["boot_magic"] == C_EXEC_MAGIC
    assert stats["kernel_words"] > 80

    # User program exercises PUTINT (#2), PUTCHAR (#3), CLOCK (#4), HEALTH (#5), YIELD (#7), EXIT (#1)
    ch_o = char_to_obb729("O")
    ch_k = char_to_obb729("K")
    ch_nl = char_to_obb729("\n")
    user_asm = f"""
        ADD  S1, S0, #123         | NOP
        SVC  #2
        ADD  S1, S0, #{ch_o}      | NOP
        SVC  #3
        ADD  S1, S0, #{ch_k}      | NOP
        SVC  #3
        ADD  S1, S0, #{ch_nl}     | NOP
        SVC  #3
        SVC  #7
        SVC  #4
        SVC  #2
        SVC  #5
        SVC  #2
        ADD  S1, S0, #77          | NOP
        SVC  #1
    """
    job = cex.submit_assembly("SVC_TOUR", user_asm, owner="KNUTH", quantum=50)
    cex.run_scheduler(max_cpus=1)

    assert job.state == "EXITED"
    assert job.exit_code == 77
    assert job.yields == 1
    assert job.output_lines[0] == "123"
    assert job.output_lines[1] == "OK"
    assert int(job.output_lines[2]) > 0     # SYS_CLOCK > 0 cycles
    assert job.output_lines[3] == "1"       # SYS_HEALTH == +1 NOMINAL

    kstats = cex.kernel_stats()
    assert kstats["total_svcs"] == 10
    assert kstats["total_aborts"] == 0


def test_cexec_preemptive_multi_cpu_time_sharing_and_hardware_isolation():
    """Run 4 jobs at the SAME logical origin=100 across 2 CPUs with TRAP_TIMER & rogue isolation."""
    m = UE1000Machine()
    cex = CExecHypervisor(m)
    cex.boot()

    # Job 1: sum(1..25) = 325, using logical addr 500 as scratch!
    asm_j1 = """
        AADD A1, A0, #25          | ADD S1, S0, #0
    LP: A2S  S2, A1               | ADD S1, S1, S2
        ST   S1, 500(A0)
        BCT  A1, #LP              | NOP
        LD   S1, 500(A0)
        SVC  #2
        SVC  #1
    """
    # Job 2: 2 * sum(1..20) = 420, ALSO at logical origin=100 & logical addr 500!
    asm_j2 = """
        AADD A1, A0, #20          | ADD S1, S0, #0
    LP: A2S  S2, A1               | ADD S1, S1, S2
        ADD  S1, S1, S2           | NOP
        ST   S1, 500(A0)
        BCT  A1, #LP              | NOP
        LD   S1, 500(A0)
        SVC  #2
        SVC  #1
    """
    # Job 3: Rogue memory access attempting to read Ferrite Vault M[-1] -> TRAP_PROT (2)
    asm_rogue_prot = """
        LD   S1, -1(A0)
        SVC  #1
    """
    # Job 4: Rogue privileged instruction in User mode -> TRAP_PRIV (4)
    asm_rogue_priv = """
        HLT
    """

    j1 = cex.submit_assembly("SUM_25", asm_j1, owner="FORRESTER", quantum=5)
    j2 = cex.submit_assembly("DBL_20", asm_j2, owner="OLIVER", quantum=5)
    j3 = cex.submit_assembly("ROGUE_PROT", asm_rogue_prot, owner="GUEST", quantum=5)
    j4 = cex.submit_assembly("ROGUE_PRIV", asm_rogue_priv, owner="GUEST", quantum=5)

    # Disjoint physical arenas!
    assert len({j1.base, j2.base, j3.base, j4.base}) == 4

    cex.run_scheduler(max_cpus=2, max_rounds=100)

    assert j1.state == "EXITED" and j1.exit_code == 325 and j1.output_lines == ["325"]
    assert j2.state == "EXITED" and j2.exit_code == 420 and j2.output_lines == ["420"]
    assert j1.preemptions >= 3 and j2.preemptions >= 3  # Real hardware TRAP_TIMER time-slicing!
    assert j3.state == "ABORTED" and j3.abort_trap == TRAP_PROT
    assert j4.state == "ABORTED" and j4.abort_trap == TRAP_PRIV

    kstats = cex.kernel_stats()
    assert kstats["total_preemptions"] >= 6
    assert kstats["total_aborts"] == 2


def test_vos_wear_capability_security_and_alexandrian_vault():
    """Verify V-OS WEAR compartment enforcement, DISC/SILO/TAPE/CHUTE CHIO, and audit trail."""
    sess = VOSSession(user="GUEST")

    # GUEST can read open silo & disc files
    open_text = sess.execute_line("CAT SILO:OPEN_NOTES.TXT")[0]
    assert "LEVEL -9" in open_text

    # GUEST denied restricted WEAR compartment 7777, fault injection, IPL, and COMP elevation!
    with pytest.raises(VOSError, match="PROVENANCE DENIED"):
        sess.execute_line("CAT SILO:WEAR_CHARTER.SEC")
    with pytest.raises(VOSError, match="WEAR DENIAL"):
        sess.execute_line("FAULT TILE 3")
    with pytest.raises(VOSError, match="WEAR DENIAL"):
        sess.execute_line("IPL")
    with pytest.raises(VOSError, match="WEAR DENIAL"):
        sess.execute_line("COMP 7777")
    assert sess.machine.io_channels.silo.provenance_denials == 1

    # Login as Jameson (cleared for 1971 & 7777) -> Silo staging, Chute ingest, & Tape READ_BACKWARD!
    sess.execute_line("LOGIN JAMESON")
    memo = sess.execute_line("CAT SILO:BYURAKAN_71.MEMO")[0]
    assert "KELVANOV" in memo
    sess.execute_line("STAGE SILO:WEAR_CHARTER.SEC DISC:CHARTER.TXT")
    assert "1.000 G$" in sess.execute_line("CAT DISC:CHARTER.TXT")[0]

    ing = sess.execute_line("CHUTE INGEST DISC:SETUN_SCAN.TXT")[0]
    assert "SETUN-70" in ing

    sess.execute_line('TAPE WRITE "BYURAKAN REEL 1971"')
    rb = sess.execute_line("TAPE READBACK")[0]
    assert "BYURAKAN REEL 1971" in rb


def test_vos_showcase_script_and_unified_language_launcher():
    """Execute programs/level9_ops.vos end-to-end and verify unified APL/FORTH/SAL-3/CONVERSE launch."""
    script_path = Path(__file__).resolve().parent.parent / "programs" / "level9_ops.vos"
    out, sess = run_vos_script(script_path.read_text(encoding="utf-8"), user="BROOKS")
    joined = "\n".join(out)

    assert "WEAR PROVENANCE DENIED" in joined
    assert "ROSENBLATT-KELVANOV" in joined
    assert "OBB-LEVEL-9 ALEXANDRIAN CHECKPOINT 1973" in joined
    assert "Job #1 [SUM_20] -> EXITED (exit=210" in joined
    assert "Job #2 [CLOCK_CHK] -> EXITED (exit=42" in joined
    assert "Job #3 [ROGUE_JOB] -> ABORTED (ABORT TRAP 2 (PROT)" in joined
    assert "Ashby Cutover #1: Logical T7" in joined
    assert "Acknowledged" in joined

    # Unified language dispatch inside the same live V-OS session!
    apl_out = sess.execute_line("APL +/ 1 2 3 4 5")
    assert any("15" in ln for ln in apl_out)

    forth_out = sess.execute_line("FORTH 6 7 * .")
    assert any("42" in ln for ln in forth_out)

    conv_out = sess.execute_line('CONVERSE FIND "SETUN"')
    assert any("SETUN" in ln for ln in conv_out)

    sal_out = sess.execute_line("RUN PROGRAMS:bloop_primes.sal3")
    assert any("SAL-3 bloop_primes.sal3" in ln for ln in sal_out)
