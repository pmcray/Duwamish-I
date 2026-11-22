; TRILISP-36 Showcase 3: Classic MIT AI Symbolic Differentiation & Simplifier
; Differentiates algebraic S-expressions with respect to a variable and simplifies!

(DEFUN MAKE-SUM (A B)
  (COND ((EQUAL A 0) B)
        ((EQUAL B 0) A)
        ((AND (NUMBERP A) (NUMBERP B)) (+ A B))
        (T (LIST '+ A B))))

(DEFUN MAKE-PROD (A B)
  (COND ((OR (EQUAL A 0) (EQUAL B 0)) 0)
        ((EQUAL A 1) B)
        ((EQUAL B 1) A)
        ((AND (NUMBERP A) (NUMBERP B)) (* A B))
        (T (LIST '* A B))))

(DEFUN DIFF (EXP VAR)
  (COND ((NUMBERP EXP) 0)
        ((SYMBOLP EXP) (IF (EQ EXP VAR) 1 0))
        ((EQ (CAR EXP) '+)
         (MAKE-SUM (DIFF (CADR EXP) VAR)
                   (DIFF (CADDR EXP) VAR)))
        ((EQ (CAR EXP) '*)
         (MAKE-SUM (MAKE-PROD (CADR EXP) (DIFF (CADDR EXP) VAR))
                   (MAKE-PROD (DIFF (CADR EXP) VAR) (CADDR EXP))))
        (T (LIST 'D EXP VAR))))

; d/dx (x*x + 3*x + 5) = (x + x) + 3
(DEFINE POLY '(+ (+ (* X X) (* 3 X)) 5))
(DEFINE D1 (DIFF POLY 'X))
(DEFINE D2 (DIFF D1 'X))

(PRINT D1)
(PRINT D2)
(LIST D1 D2)
