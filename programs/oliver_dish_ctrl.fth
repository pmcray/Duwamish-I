\ === OLIVER-MOORE LEVEL -9 SETI ANNEX INSTRUMENT CONTROLLER (FORTH-3) ===
\ Demonstrates real-time radio-telescope beam tracking using CREATE..DOES>
\ beam descriptors, hardware Good Deciban DWALD / DLSE, 3-way TRIT-IF
\ candidate classification, and Tier 2 T-Fabric Radix-3 TBFP-9C FFT (`T.FFT3`)!

." === LEVEL -9 OLIVER SETI ANNEX: FORTH-3 BEAM CONTROLLER ===" CR

\ Define a SETI beam descriptor: [ az_deg, el_deg, evid_ticks ]
: BEAM ( az el -- )
  CREATE SWAP , , 0 ,
  DOES> ;  \ pushes base address of 3-word beam record

: BEAM-AZ   ( b -- az )   @ ;
: BEAM-EL   ( b -- el )   1+ @ ;
: BEAM-EV   ( b -- ev )   2 + @ ;
: ADD-EVID  ( ticks b -- ) 2 + +! ;

20 dB>TICKS CONSTANT WALD-THRESH  \ +20 dB = 1620 ticks!

\ Declare three target beams (Tau Ceti, Epsilon Eridani, Background Sky)
142 38 BEAM TAU-CETI
219 51 BEAM EPS-ERIDANI
 15 74 BEAM SKY-REF

\ Accumulate real-time multi-integration Good Deciban evidence on each beam
 12.0dB TAU-CETI ADD-EVID
 10.5dB TAU-CETI ADD-EVID

  6.0dB EPS-ERIDANI ADD-EVID
  3.5dB EPS-ERIDANI ADD-EVID

-14.0dB SKY-REF ADD-EVID
 -9.0dB SKY-REF ADD-EVID

: INSPECT-BEAM ( beam -- verdict )
  DUP BEAM-EV DUP .DB
  WALD-THRESH DWALD DUP
  TRIT-IF
    ." -> [-1] REJECT (NATURAL/NOISE)"
  WHEN0
    ." -> [ 0] UNDECIDED (HOLD INTEGRATION & REFINE)"
  WHEN+
    ." -> [+1] CANDIDATE ARTIFICIAL BEACON!"
  TRIT-THEN CR
  SWAP DROP ;

." Beam 1 (TAU-CETI)    : " TAU-CETI    INSPECT-BEAM
." Beam 2 (EPS-ERIDANI) : " EPS-ERIDANI INSPECT-BEAM
." Beam 3 (SKY-REF)     : " SKY-REF     INSPECT-BEAM

\ Synthesize a 9-point periodic baseband carrier in UE-1000 T-DRAM & dispatch T.FFT3!
CREATE BASEBAND 9 ALLOT
 1.0f BASEBAND 0 + !
 0.5f BASEBAND 1 + !
-0.5f BASEBAND 2 + !
-1.0f BASEBAND 3 + !
-0.5f BASEBAND 4 + !
 0.5f BASEBAND 5 + !
 1.0f BASEBAND 6 + !
 0.5f BASEBAND 7 + !
-0.5f BASEBAND 8 + !

." Tier 2 T-Fabric Radix-3 FFT (9-pt): peak bin = "
BASEBAND 9 T.FFT3 SWAP . ." | peak mag (TF36) = " F. CR
." Hardware cycles consumed: " CYCLES . CR
