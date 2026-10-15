"""End-to-end design pipeline."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .accessibility import HAVE_VIENNA, annotate_accessibility
from .engines import (
    NATIVE_ENGINES,
    Engine,
    build_external_engines,
    engine_families,
    load_external_specs,
    run_engines,
)
from .gates import GateConfig, apply_gates
from .offtarget import SeedTable, TranscriptomeIndex, annotate_offtargets
from .rules import load_pwm
from .scoring import (
    Calibration,
    ReferenceDistribution,
    ScoredCandidate,
    UnifiedScorer,
    consensus_index,
    select_portfolio,
)
from .seq import Candidate, enumerate_candidates, read_fasta, to_rna

PROFILE_DIR = Path(__file__).parent / "profiles"


def load_profile(name_or_path: str) -> dict:
    p = Path(name_or_path)
    if not p.exists():
        p = PROFILE_DIR / f"{name_or_path}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"profile not found: {name_or_path}")
    return yaml.safe_load(p.read_text())


@dataclass
class DesignResult:
    transcript_id: str
    scored: list[ScoredCandidate]
    portfolio: list[ScoredCandidate]
    n_enumerated: int
    n_gated: int
    engines_run: list[str]
    consensus_w: float
    calibrated: bool
    accessibility_source: str
    gate_reasons: dict[str, int] = field(default_factory=dict)


def design(
    fasta: str | Path,
    profile: dict,
    transcriptome: str | Path | None = None,
    utr3: str | Path | None = None,
    seed_table_json: str | Path | None = None,
    calibration: str | Path | None = None,
    reference: str | Path | None = None,
    pwm: str | Path | None = None,
    duplex_len: int = 19,
    start_offset: int = 75,
    top_k: int = 4,
    max_mm: int = 3,
    force_fallback_accessibility: bool = False,
) -> list[DesignResult]:
    """Run the full v1.0 stack over every record in ``fasta``."""
    if pwm:
        load_pwm(pwm)  # validate early; engines pick it up via profile

    gate_cfg = GateConfig(**profile.get("gates", {}))

    # --- specificity resources ---------------------------------------
    table: SeedTable | None = None
    if seed_table_json:
        table = SeedTable.from_json(seed_table_json)
    elif utr3:
        from .offtarget import build_seed_table

        table = build_seed_table(utr3)
    else:
        warnings.warn(
            "No 3'UTR source given: the POTS seed-promiscuity axis will be "
            "unavailable and the specificity objective will fall back to its "
            "`missing` default. Pass --utr3 or --seed-table.",
            RuntimeWarning,
        )

    index: TranscriptomeIndex | None = None
    if transcriptome:
        index = TranscriptomeIndex.from_fasta(transcriptome)

    calib = Calibration.load(calibration) if calibration else None
    ref = ReferenceDistribution.load(reference) if reference else None

    specs = load_external_specs(profile)
    engines: list[Engine] = list(NATIVE_ENGINES) + build_external_engines(specs)
    families = engine_families(engines)

    results: list[DesignResult] = []
    for name, seq in read_fasta(str(fasta)):
        transcript = to_rna(seq)
        cands = enumerate_candidates(
            name, transcript, duplex_len=duplex_len, start_offset=start_offset
        )
        if not cands:
            warnings.warn(f"{name}: no candidates after tiling (sequence too short?)")
            continue

        annotate_accessibility(
            transcript, cands, force_fallback=force_fallback_accessibility
        )
        annotate_offtargets(
            cands, index, table, on_target_prefix=name, max_mm=max_mm
        )
        apply_gates(cands, gate_cfg)

        reasons: dict[str, int] = {}
        for c in cands:
            for r in c.gate_failures:
                key = r.split("(")[0]
                reasons[key] = reasons.get(key, 0) + 1

        passing = [c for c in cands if not c.gated]
        if not passing:
            results.append(
                DesignResult(
                    name, [], [], len(cands), len(cands), [], float("nan"),
                    False, "none", reasons,
                )
            )
            warnings.warn(f"{name}: every candidate was gated out")
            continue

        engine_scores, ran = run_engines(engines, passing)
        scorer = UnifiedScorer(profile, calib, ref, families)
        scored = scorer.score(cands, engine_scores)
        w = consensus_index(engine_scores, [c.candidate_id for c in passing])
        port = select_portfolio(
            scored,
            k=top_k,
            **profile.get("portfolio", {}),
        )
        src = "ViennaRNA" if (HAVE_VIENNA and not force_fallback_accessibility) else "fallback"
        results.append(
            DesignResult(
                transcript_id=name,
                scored=scored,
                portfolio=port,
                n_enumerated=len(cands),
                n_gated=len(cands) - len(passing),
                engines_run=ran,
                consensus_w=w,
                calibrated=bool(scored and scored[0].calibrated),
                accessibility_source=src,
                gate_reasons=reasons,
            )
        )
    return results


def results_to_rows(results: list[DesignResult]) -> list[dict]:
    rows = []
    for res in results:
        in_port = {s.candidate_id for s in res.portfolio}
        for rank, s in enumerate(res.scored, start=1):
            c = s.candidate
            row = {
                "transcript": res.transcript_id,
                "candidate_id": c.candidate_id,
                "rank": rank,
                "position": c.start + 1,
                "guide_5to3": c.guide,
                "sense_5to3": c.sense,
                "guide_oligo": c.guide_oligo,
                "sense_oligo": c.sense_oligo,
                "USS": round(s.uss, 2),
                "USS_LCB": round(s.uss_lcb, 2),
                "potency": round(s.potency, 4),
                "potency_pi_low": round(s.pi_low, 4),
                "potency_pi_high": round(s.pi_high, 4),
                "n_engines": s.n_engines,
                "calibrated": s.calibrated,
                "in_portfolio": c.candidate_id in in_port,
            }
            for k, v in s.desirability.items():
                row[f"d_{k}"] = round(v, 4)
            for k in (
                "gc_fraction", "seed_tm", "ddg_asymmetry", "accessibility",
                "pots", "offtarget_burden", "n_offtarget_hits",
                "min_offtarget_mismatches",
            ):
                if k in c.features:
                    row[k] = round(float(c.features[k]), 4)
            for e, v in s.per_engine_kd.items():
                row[f"kd_{e}"] = round(v, 4)
            rows.append(row)
    return rows


def gated_rows(results: list[DesignResult], candidates_by_tx: dict[str, list[Candidate]]
               ) -> list[dict]:
    rows = []
    for tx, cands in candidates_by_tx.items():
        for c in cands:
            if c.gated:
                rows.append(
                    {
                        "transcript": tx,
                        "candidate_id": c.candidate_id,
                        "position": c.start + 1,
                        "guide_5to3": c.guide,
                        "gate_failures": ";".join(c.gate_failures),
                    }
                )
    return rows
