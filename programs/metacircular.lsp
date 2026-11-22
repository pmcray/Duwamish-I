; TRILISP-36 Showcase 2: McCarthy's LISP 1.5 Metacircular Evaluator (M-EVAL)
; Interprets QUOTE, ATOM, EQ, CAR, CDR, CONS, COND, LAMBDA, and LABEL
; *inside* TRILISP-36 compiled to native TRIAD-36 running on the UE-1000 CPU!

(DEFUN LOOKUP (X ENV)
  (COND ((NULL ENV) NIL)
        ((EQ (CAAR ENV) X) (CDAR ENV))
        (T (LOOKUP X (CDR ENV)))))

(DEFUN PAIRLIS (VARS VALS ENV)
  (COND ((NULL VARS) ENV)
        (T (CONS (CONS (CAR VARS) (CAR VALS))
                 (PAIRLIS (CDR VARS) (CDR VALS) ENV)))))

(DEFUN EVCON (CLAUSES ENV)
  (COND ((NULL CLAUSES) NIL)
        ((MEVAL (CAAR CLAUSES) ENV)
         (MEVAL (CADAR CLAUSES) ENV))
        (T (EVCON (CDR CLAUSES) ENV))))

(DEFUN EVLIS (ARGS ENV)
  (COND ((NULL ARGS) NIL)
        (T (CONS (MEVAL (CAR ARGS) ENV)
                 (EVLIS (CDR ARGS) ENV)))))

(DEFUN MAPPLY (FN ARGS ENV)
  (COND ((ATOM FN)
         (COND ((EQ FN 'CAR)  (CAR (CAR ARGS)))
               ((EQ FN 'CDR)  (CDR (CAR ARGS)))
               ((EQ FN 'CONS) (CONS (CAR ARGS) (CADR ARGS)))
               ((EQ FN 'ATOM) (ATOM (CAR ARGS)))
               ((EQ FN 'EQ)   (EQ (CAR ARGS) (CADR ARGS)))
               ((EQ FN '+)    (+ (CAR ARGS) (CADR ARGS)))
               ((EQ FN '-)    (- (CAR ARGS) (CADR ARGS)))
               ((EQ FN '*)    (* (CAR ARGS) (CADR ARGS)))
               ((EQ FN '=)    (= (CAR ARGS) (CADR ARGS)))
               (T (MAPPLY (MEVAL FN ENV) ARGS ENV))))
        ((EQ (CAR FN) 'LAMBDA)
         (MEVAL (CADDR FN) (PAIRLIS (CADR FN) ARGS ENV)))
        ((EQ (CAR FN) 'LABEL)
         (MAPPLY (CADDR FN)
                 ARGS
                 (CONS (CONS (CADR FN) (CADDR FN)) ENV)))
        (T NIL)))

(DEFUN MEVAL (E ENV)
  (COND ((NUMBERP E) E)
        ((EQ E T) T)
        ((EQ E NIL) NIL)
        ((EQ E U) U)
        ((ATOM E) (LOOKUP E ENV))
        ((EQ (CAR E) 'QUOTE) (CADR E))
        ((EQ (CAR E) 'COND)  (EVCON (CDR E) ENV))
        ((EQ (CAR E) 'LAMBDA) E)
        ((EQ (CAR E) 'LABEL)  E)
        (T (MAPPLY (CAR E) (EVLIS (CDR E) ENV) ENV))))

; Run interpreted APPEND via LABEL inside the metacircular evaluator!
(DEFINE R1
  (MEVAL '((LABEL APP
                  (LAMBDA (X Y)
                    (COND ((EQ X NIL) Y)
                          (T (CONS (CAR X) (APP (CDR X) Y))))))
           '(A B)
           '(C D))
         NIL))

; Run interpreted FACTORIAL(5) = 120 via LABEL inside the metacircular evaluator!
(DEFINE R2
  (MEVAL '((LABEL FACT
                  (LAMBDA (N)
                    (COND ((= N 0) 1)
                          (T (* N (FACT (- N 1)))))))
           5)
         NIL))

(PRINT R1)
(PRINT R2)
(LIST R1 R2)
