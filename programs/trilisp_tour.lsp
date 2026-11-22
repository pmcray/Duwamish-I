; TRILISP-36 Showcase 1: Classic LISP 1.5 Tour on the 36-Trit Duwamish-I
; Exercises 1-trit tags, lists, DEFMACRO, lexical closures + SETQ, LABEL,
; Kleene 3-valued logic (T / U / NIL), SIGN3 (B3), Good Decibans, and native GC!

(DEFMACRO (WHEN TEST . BODY)
  (CONS 'IF (CONS TEST (LIST (CONS 'PROGN BODY)))))

; 1. One trit, three types: +1 fixnum, 0 cons, -1 symbol
(PRINT (LIST (TAG 42) (TAG '(A B)) (TAG 'SETUN) (TAG NIL) (TAG U)))

; 2. Classic recursive APPEND & MAPCAR
(DEFUN APPEND (XS YS)
  (IF (NULL XS)
      YS
      (CONS (CAR XS) (APPEND (CDR XS) YS))))

(DEFUN MAPCAR (FN XS)
  (IF (NULL XS)
      NIL
      (CONS (FN (CAR XS)) (MAPCAR FN (CDR XS)))))

(PRINT (APPEND '(SETUN DUWAMISH) '(OBB LEVEL-9)))
(PRINT (MAPCAR (LAMBDA (X) (* X X)) '(1 2 3 4 5)))

; 3. Lexical closure (FUNARG) with mutable state via SETQ on env cell!
(DEFUN MAKE-COUNTER (START)
  (LAMBDA ()
    (SETQ START (+ START 1))
    START))

(DEFINE C1 (MAKE-COUNTER 10))
(PRINT (LIST (C1) (C1) (C1)))

; 4. Classic LISP 1.5 LABEL factorial: 6! = 720
(PRINT ((LABEL FACT (LAMBDA (N)
                      (IF (= N 0) 1 (* N (FACT (- N 1))))))
        6))

; 5. Kleene three-valued logic (T, U, NIL) & hardware SIGN3 (B3)
(PRINT (LIST (KAND T U) (KOR NIL U) (KEQV NIL NIL) (KNOT U)))
(DEFUN CLASSIFY (V)
  (SIGN3 V 'NEGATIVE 'UNDECIDED-OR-ZERO 'POSITIVE))
(PRINT (LIST (CLASSIFY -9) (CLASSIFY U) (CLASSIFY 9)))

; 6. Good Deciban Bayesian evidence right inside LISP 1.5!
(DEFINE EV (DB+ 12.0db 9.5db))
(PRINT (LIST EV (WALD EV 20.0db) (WALD 5.0db 20.0db) (DLSE 10.0db 10.0db)))

; 7. Allocate temporary garbage, then invoke native TRIAD-36 mark-and-sweep (GC)!
(DEFUN MAKE-GARBAGE (N ACC)
  (IF (= N 0) ACC (MAKE-GARBAGE (- N 1) (CONS N ACC))))

(DEFINE TMP (MAKE-GARBAGE 60 NIL))
(SETQ TMP NIL)
(DEFINE FREED (GC))
(WHEN (> FREED 50)
  (PRINT (LIST 'GC-RECLAIMED FREED 'LIVE (HEAP-USED))))

'TOUR-COMPLETE
