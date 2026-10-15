"""Stage 0 -- hard gates.

A gate failure eliminates a candidate outright; it is never traded off
against a high potency score. Each gate records a machine-readable reason
so the report can show near-misses and the user can relax a specific gate
deliberately rather than by loosening a weight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .seq import (
    Candidate,
    gc_fraction,
    has_g_quadruplex,
    immunostim_hits,
    longest_run,
)
from .thermo import duplex_tm


@dataclass(slots=True)
class GateConfig:
    gc_min: float = 0.30
    gc_max: float = 0.60
    max_homopolymer: int = 3  # reject runs of 4 or more
    forbid_g_quadruplex: bool = True
    forbid_immunostim: bool = True
    seed_tm_max: float | None = 21.5  # siDirect 2.0 criterion; None disables
    strand_conc_m: float = 1e-6
    max_common_snp_af: float | None = 0.001
    forbid_offtarget_mismatch_le: int | None = 2  # full-length near-identity


GateFn = Callable[[Candidate, GateConfig], str | None]


def gate_gc(c: Candidate, cfg: GateConfig) -> str | None:
    gc = gc_fraction(c.sense)
    c.features["gc_fraction"] = gc
    if not (cfg.gc_min <= gc <= cfg.gc_max):
        return f"gc_out_of_range({gc:.2f})"
    return None


def gate_homopolymer(c: Candidate, cfg: GateConfig) -> str | None:
    run = longest_run(c.sense)
    c.features["longest_run"] = float(run)
    if run > cfg.max_homopolymer:
        return f"homopolymer_run({run})"
    return None


def gate_quadruplex(c: Candidate, cfg: GateConfig) -> str | None:
    if not cfg.forbid_g_quadruplex:
        return None
    if has_g_quadruplex(c.sense):
        return "g_quadruplex_motif"
    return None


def gate_immunostim(c: Candidate, cfg: GateConfig) -> str | None:
    if not cfg.forbid_immunostim:
        return None
    hits = immunostim_hits(c.sense) + immunostim_hits(c.guide)
    c.features["immunostim_hits"] = float(len(hits))
    if hits:
        return f"immunostimulatory_motif({','.join(sorted(set(hits)))})"
    return None


def gate_seed_tm(c: Candidate, cfg: GateConfig) -> str | None:
    tm = duplex_tm(c.seed7, cfg.strand_conc_m)
    c.features["seed_tm"] = tm
    if cfg.seed_tm_max is None:
        return None
    if tm >= cfg.seed_tm_max:
        return f"seed_tm_high({tm:.1f}C)"
    return None


def gate_snp(c: Candidate, cfg: GateConfig) -> str | None:
    """Requires ``snp_af`` to have been annotated upstream."""
    if cfg.max_common_snp_af is None:
        return None
    af = c.features.get("max_snp_af")
    if af is not None and af > cfg.max_common_snp_af:
        return f"common_snp_in_guide(af={af:.4f})"
    return None


def gate_offtarget_identity(c: Candidate, cfg: GateConfig) -> str | None:
    """Requires ``min_offtarget_mismatches`` from the off-target stage."""
    if cfg.forbid_offtarget_mismatch_le is None:
        return None
    mm = c.features.get("min_offtarget_mismatches")
    if mm is not None and mm <= cfg.forbid_offtarget_mismatch_le:
        return f"near_identical_offtarget(mm={int(mm)})"
    return None


DEFAULT_GATES: tuple[tuple[str, GateFn], ...] = (
    ("gc", gate_gc),
    ("homopolymer", gate_homopolymer),
    ("quadruplex", gate_quadruplex),
    ("immunostim", gate_immunostim),
    ("seed_tm", gate_seed_tm),
    ("snp", gate_snp),
    ("offtarget_identity", gate_offtarget_identity),
)


def apply_gates(
    candidates: list[Candidate],
    cfg: GateConfig,
    gates: tuple[tuple[str, GateFn], ...] = DEFAULT_GATES,
) -> list[Candidate]:
    """Annotate every candidate with its gate failures. Returns the same list.

    Note that all gates run even after the first failure: the report shows
    *why* a near-miss missed, which is far more useful than a single reason.
    """
    for c in candidates:
        c.gate_failures = []
        for _name, fn in gates:
            reason = fn(c, cfg)
            if reason:
                c.gate_failures.append(reason)
    return candidates
