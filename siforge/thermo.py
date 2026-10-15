"""RNA/RNA duplex thermodynamics.

Nearest-neighbour parameters are the Watson-Crick RNA set of
Xia et al. (1998) Biochemistry 37:14719, as tabulated in the standard
Turner rules. Entropies are derived self-consistently from the published
dH and dG37 values via dS = (dH - dG37) / 310.15, which avoids mixing
rounding conventions between separately published dS tables.

Everything here is a reimplementation from published parameters; no
third-party source is vendored.
"""

from __future__ import annotations

import math
from functools import lru_cache

R_CAL = 1.98720425864083  # cal / (mol K)
T37 = 310.15

# dG37 (kcal/mol) for 5'-XY-3' stacked on its Watson-Crick complement.
NN_DG37: dict[str, float] = {
    "AA": -0.93, "UU": -0.93,
    "AU": -1.10,
    "UA": -1.33,
    "CU": -2.08, "AG": -2.08,
    "CA": -2.11, "UG": -2.11,
    "GU": -2.24, "AC": -2.24,
    "GA": -2.35, "UC": -2.35,
    "CG": -2.36,
    "GG": -3.26, "CC": -3.26,
    "GC": -3.42,
}

# dH (kcal/mol), same stacking convention.
NN_DH: dict[str, float] = {
    "AA": -6.82, "UU": -6.82,
    "AU": -9.38,
    "UA": -7.69,
    "CU": -10.48, "AG": -10.48,
    "CA": -10.44, "UG": -10.44,
    "GU": -11.40, "AC": -11.40,
    "GA": -12.44, "UC": -12.44,
    "CG": -10.64,
    "GG": -13.39, "CC": -13.39,
    "GC": -14.88,
}

INIT_DG = 4.09
INIT_DH = 3.61
TERM_AU_DG = 0.45
TERM_AU_DH = 3.72


@lru_cache(maxsize=1 << 16)
def duplex_dg(seq: str, include_init: bool = True) -> float:
    """dG37 (kcal/mol) of a perfectly complementary RNA duplex.

    ``seq`` is one strand 5'->3'; the other strand is its complement.
    With ``include_init=False`` only the stacking terms are summed, which
    is what you want when comparing *sub-stacks* of the same duplex (e.g.
    the four terminal base pairs at each end).
    """
    if len(seq) < 2:
        return 0.0
    dg = sum(NN_DG37.get(seq[i : i + 2], 0.0) for i in range(len(seq) - 1))
    if include_init:
        dg += INIT_DG
        if seq[0] in "AU":
            dg += TERM_AU_DG
        if seq[-1] in "AU":
            dg += TERM_AU_DG
    return dg


@lru_cache(maxsize=1 << 16)
def duplex_dh(seq: str, include_init: bool = True) -> float:
    if len(seq) < 2:
        return 0.0
    dh = sum(NN_DH.get(seq[i : i + 2], 0.0) for i in range(len(seq) - 1))
    if include_init:
        dh += INIT_DH
        if seq[0] in "AU":
            dh += TERM_AU_DH
        if seq[-1] in "AU":
            dh += TERM_AU_DH
    return dh


def duplex_tm(seq: str, strand_conc_m: float = 1e-6) -> float:
    """Melting temperature (deg C) of a perfect RNA/RNA duplex.

    ``strand_conc_m`` is the total strand concentration C_T. For the
    non-self-complementary case Tm = dH / (dS + R ln(C_T/4)).

    The 7-mer seed-target duplex Tm computed with this function is the
    quantity used by the siDirect 2.0 specificity criterion (guide
    positions 2-8; candidates are preferred when it falls below ~21.5 C).
    """
    dh = duplex_dh(seq) * 1000.0
    dg = duplex_dg(seq) * 1000.0
    ds = (dh - dg) / T37
    denom = ds + R_CAL * math.log(strand_conc_m / 4.0)
    if denom == 0:
        return float("nan")
    return dh / denom - 273.15


def end_stability(seq: str, n: int = 4, from_5_prime: bool = True) -> float:
    """dG37 of the terminal ``n`` base pairs (stacking only, no init)."""
    sub = seq[:n] if from_5_prime else seq[-n:]
    return duplex_dg(sub, include_init=False)


def ddg_asymmetry(guide: str, n: int = 4) -> float:
    """Duplex-end asymmetry, oriented so that higher is better.

    Returns dG(guide 5' end) - dG(guide 3' end). Efficient RISC loading
    requires the guide 5' terminus to be the *less* thermodynamically
    stable end, so a positive value (guide 5' end less negative, i.e.
    weaker) favours correct strand selection.
    """
    five = end_stability(guide, n, from_5_prime=True)
    three = end_stability(guide, n, from_5_prime=False)
    return five - three


def internal_hairpin_tm(seq: str) -> float:
    """Approximate stability of the most stable internal self-pairing.

    Reynolds criterion 3 ("no internal repeats") is operationalised as the
    melting temperature of the best intramolecular stem; low is good.
    Returns deg C, or -99 when no stem of >= 4 bp exists.
    """
    best = -99.0
    n = len(seq)
    from .seq import revcomp

    for stem in range(min(8, n // 2), 3, -1):
        for i in range(0, n - stem + 1):
            sub = seq[i : i + stem]
            target = revcomp(sub)
            # require a loop of >= 3 nt between the two arms
            j = seq.find(target, i + stem + 3)
            if j != -1:
                tm = duplex_tm(sub)
                best = max(best, tm)
        if best > -99.0:
            break
    return best
