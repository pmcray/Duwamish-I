"""Duwamish-I (UE-1000) — Native Balanced-Ternary Supercomputer Simulator.

Conforms strictly to Universal Entropics Special Systems Section Form SS-1000-01
(Principles of Operation, Rev. 1.5) under WEAR charter for OBB Unseen Level -9.
"""

from .ternary import (
    WORD_TRITS,
    HALF_TRITS,
    NONET_TRITS,
    TRYTE_TRITS,
    ECC_CHECK_TRITS,
    MEM_WORD_TRITS,
    HEPT_ALPHABET,
)

__version__ = "1.5.0"
__all__ = [
    "WORD_TRITS",
    "HALF_TRITS",
    "NONET_TRITS",
    "TRYTE_TRITS",
    "ECC_CHECK_TRITS",
    "MEM_WORD_TRITS",
    "HEPT_ALPHABET",
    "__version__",
]
