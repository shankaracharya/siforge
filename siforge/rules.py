"""Rule-based siRNA scoring engines, reimplemented from the literature.

Fidelity notes (read these before citing results)
--------------------------------------------------
``reynolds2004``, ``uitei2004``, ``amarzguioui2004`` and ``katoh2007``
implement the published criteria directly; each criterion is exposed as a
separate boolean flag so downstream models can use the individual rules
rather than only the composite.

``pwm_linear`` is a *scaffold* for the position-weight linear models of
the i-Score (Ichihara et al. 2007) and DSIR (Vert et al. 2006) families.
siforge ships a surrogate coefficient matrix derived from published
consensus base preferences -- it is NOT the published i-Score or DSIR
coefficient table, and must not be reported as such. To obtain the real
models, drop the published coefficients into a TSV and point
``--pwm`` at it; see docs/PWM.md.

Each engine returns a float where higher is better.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .seq import (
    au_count,
    gc_fraction,
    longest_gc_stretch,
)
from .thermo import ddg_asymmetry, internal_hairpin_tm

DATA_DIR = Path(__file__).parent / "data"


@dataclass(slots=True)
class RuleResult:
    score: float
    flags: dict[str, bool]


# --------------------------------------------------------------------------
# Reynolds et al. 2004, Nat Biotechnol 22:326 -- eight criteria, +1 each.
# Positions are 1-based on the SENSE strand.
# --------------------------------------------------------------------------
def reynolds2004(sense: str, guide: str) -> RuleResult:
    gc = gc_fraction(sense)
    flags = {
        "reynolds_gc_30_52": 0.30 <= gc <= 0.52,
        "reynolds_au_rich_15_19": au_count(sense[14:19]) >= 3,
        "reynolds_no_internal_repeat": internal_hairpin_tm(sense) < 20.0,
        "reynolds_A_at_19": sense[18] == "A",
        "reynolds_A_at_3": sense[2] == "A",
        "reynolds_U_at_10": sense[9] == "U",
        "reynolds_not_GC_at_19": sense[18] not in "GC",
        "reynolds_not_G_at_13": sense[12] != "G",
    }
    return RuleResult(float(sum(flags.values())), flags)


# --------------------------------------------------------------------------
# Ui-Tei et al. 2004, NAR 32:936 -- four criteria. This is the functional
# core of the siDirect selection step (the specificity half of siDirect 2.0
# lives in gates.py / offtarget.py).
# --------------------------------------------------------------------------
def uitei2004(sense: str, guide: str) -> RuleResult:
    flags = {
        "uitei_guide5p_AU": guide[0] in "AU",
        "uitei_sense5p_GC": sense[0] in "GC",
        "uitei_guide_au_rich_1_7": au_count(guide[:7]) >= 4,
        "uitei_no_gc_stretch_ge9": longest_gc_stretch(sense) < 9,
    }
    return RuleResult(float(sum(flags.values())), flags)


# --------------------------------------------------------------------------
# Amarzguioui & Prydz 2004, BBRC 316:1050 -- asymmetry-weighted positional
# model. Implemented as the published signed criteria; the asymmetry term
# uses the nearest-neighbour duplex-end difference rather than the original
# A/U count proxy, which is a documented deviation.
# --------------------------------------------------------------------------
def amarzguioui2004(sense: str, guide: str) -> RuleResult:
    asym = ddg_asymmetry(guide, n=3)
    flags = {
        "amarz_asymmetry_positive": asym > 0.0,
        "amarz_A_or_U_at_sense19": sense[18] in "AU",
        "amarz_G_or_C_at_sense1": sense[0] in "GC",
        "amarz_A_at_6": sense[5] == "A",
        "amarz_no_U_at_1": sense[0] != "U",
        "amarz_no_G_at_19": sense[18] != "G",
    }
    weights = {
        "amarz_asymmetry_positive": 2.0,
        "amarz_A_or_U_at_sense19": 1.0,
        "amarz_G_or_C_at_sense1": 1.0,
        "amarz_A_at_6": 0.5,
        "amarz_no_U_at_1": 0.5,
        "amarz_no_G_at_19": 0.5,
    }
    score = sum(weights[k] for k, v in flags.items() if v)
    return RuleResult(score, flags)


# --------------------------------------------------------------------------
# Katoh & Suzuki 2007, NAR 35:e27 -- residues at every third position of the
# guide strand shape RNAi activity. Positions 1,4,7,10,13,16,19 of the guide.
# --------------------------------------------------------------------------
_KATOH_POSITIONS = (1, 4, 7, 10, 13, 16, 19)


def katoh2007(sense: str, guide: str) -> RuleResult:
    flags = {}
    score = 0.0
    for p in _KATOH_POSITIONS:
        if p > len(guide):
            continue
        base = guide[p - 1]
        ok = base in "AU"
        flags[f"katoh_pos{p}_AU"] = ok
        score += 1.0 if ok else 0.0
    return RuleResult(score, flags)


# --------------------------------------------------------------------------
# Position-weight linear model scaffold (i-Score / DSIR family).
# --------------------------------------------------------------------------
_BASES = ("A", "C", "G", "U")


def load_pwm(path: str | Path | None = None) -> dict[tuple[int, str], float]:
    """Load a position-weight table.

    TSV format: ``position<TAB>base<TAB>coefficient`` with 1-based
    positions on the SENSE strand. Lines beginning with '#' are ignored.
    """
    p = Path(path) if path else DATA_DIR / "pwm_surrogate.tsv"
    table: dict[tuple[int, str], float] = {}
    with open(p) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            pos, base, coef = line.split("\t")[:3]
            table[(int(pos), base.strip().upper())] = float(coef)
    return table


def pwm_linear(
    sense: str, guide: str, table: dict[tuple[int, str], float] | None = None
) -> RuleResult:
    """Linear position-weight score. Higher is better.

    Mean-centres per position so the score is comparable across tables with
    different intercept conventions.
    """
    tbl = table if table is not None else load_pwm()
    score = 0.0
    for i, base in enumerate(sense, start=1):
        coefs = [tbl.get((i, b)) for b in _BASES]
        present = [c for c in coefs if c is not None]
        if not present:
            continue
        mean = sum(present) / len(present)
        score += tbl.get((i, base), mean) - mean
    return RuleResult(score, {})


RULE_ENGINES = {
    "reynolds": reynolds2004,
    "uitei": uitei2004,
    "amarzguioui": amarzguioui2004,
    "katoh": katoh2007,
    "pwm": pwm_linear,
}
