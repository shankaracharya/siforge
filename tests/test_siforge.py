"""Unit tests. Run with: pytest -q"""

from __future__ import annotations

import math

import numpy as np
import pytest

from siforge.gates import GateConfig, apply_gates
from siforge.offtarget import SeedTable, TranscriptomeIndex, pots_score
from siforge.rules import reynolds2004, uitei2004
from siforge.scoring import (
    Calibration,
    derringer,
    geometric_mean,
    min_variance_weights,
    rank_percentile,
    rra_score,
    select_portfolio,
)
from siforge.seq import Candidate, enumerate_candidates, longest_run, revcomp, to_rna
from siforge.thermo import ddg_asymmetry, duplex_dg, duplex_tm, end_stability


# ---------------------------------------------------------------- sequence
def test_revcomp_roundtrip():
    s = "AUGCGGCUAAUCGGAUCGA"
    assert revcomp(revcomp(s)) == s
    assert revcomp("AAAA") == "UUUU"
    assert revcomp("GC") == "GC"


def test_dna_input_becomes_rna():
    assert to_rna("atgcTTT") == "AUGCUUU"


def test_guide_is_revcomp_of_sense():
    c = Candidate("x", "t", 0, "ACGUACGUACGUACGUACGUA")
    assert c.guide == revcomp(c.sense)
    assert len(c.sense) == 19 and len(c.guide) == 19


def test_seed_positions():
    c = Candidate("x", "t", 0, "ACGUACGUACGUACGUACGUA")
    assert c.seed7 == c.guide[1:8]
    assert c.seed6 == c.guide[1:7]
    assert len(c.seed6) == 6


def test_longest_run():
    assert longest_run("AAUUUGC") == 3
    assert longest_run("GGGGA") == 4
    assert longest_run("") == 0


def test_enumerate_skips_offset_and_N():
    seq = "A" * 10 + "N" + "ACGU" * 20
    c = enumerate_candidates("t", seq, start_offset=5)
    assert all("N" not in x.site for x in c)
    assert all(x.start >= 5 for x in c)


# ------------------------------------------------------------- thermodynamics
def test_gc_duplex_more_stable_than_au():
    assert duplex_dg("GCGCGCGCGC") < duplex_dg("AUAUAUAUAU")


def test_tm_monotone_in_gc():
    assert duplex_tm("GGCCGGC") > duplex_tm("AAUUAAU")


def test_tm_reasonable_range():
    """7-mer seed duplexes should land in a biologically sane window."""
    for s in ("AAUUAAU", "ACGUACG", "GGCCGGC"):
        tm = duplex_tm(s)
        assert -40 < tm < 75, (s, tm)


def test_asymmetry_sign():
    """A guide whose 5' end is AU-rich and 3' end GC-rich must score > 0."""
    guide = "AAUUAAUACGCGGCGGCGC"
    assert ddg_asymmetry(guide) > 0
    assert ddg_asymmetry(guide[::-1]) < 0


def test_end_stability_uses_requested_end():
    # AU-rich 5' end is weaker (less negative) than a GC-rich 3' end.
    s = "AAAAUUUUAAAAGGCC"
    assert end_stability(s, 4, True) > end_stability(s, 4, False)


# -------------------------------------------------------------------- rules
def test_reynolds_perfect_candidate():
    # A at 3, U at 10, A at 19, AU-rich 15-19, moderate GC
    sense = "CGACGCAUCUGCACAAUCA"
    res = reynolds2004(sense, revcomp(sense))
    assert res.flags["reynolds_A_at_3"]
    assert res.flags["reynolds_U_at_10"]
    assert res.flags["reynolds_A_at_19"]
    assert res.score >= 5


def test_uitei_criteria_fire_correctly():
    sense = "GCGCGCGCGCGCGCGCGCA"  # sense 5' G/C, sense19 A -> guide 5' U
    res = uitei2004(sense, revcomp(sense))
    assert res.flags["uitei_sense5p_GC"]
    assert res.flags["uitei_guide5p_AU"]
    assert not res.flags["uitei_no_gc_stretch_ge9"]  # long GC stretch present


# -------------------------------------------------------------------- gates
def test_gate_rejects_homopolymer_and_gc():
    c = Candidate("x", "t", 0, "GGGGGGGGGGGGGGGGGGGGG")
    apply_gates([c], GateConfig())
    assert c.gated
    assert any("homopolymer" in r for r in c.gate_failures)
    assert any("gc_out_of_range" in r for r in c.gate_failures)


def test_gate_all_reasons_recorded_not_just_first():
    c = Candidate("x", "t", 0, "GGGGGGGGGGGGGGGGGGGGG")
    apply_gates([c], GateConfig())
    assert len(c.gate_failures) >= 2


def test_seed_tm_gate_respects_threshold():
    c = Candidate("x", "t", 0, "GCGCGCGCGCAUAUAUAUAUA")
    apply_gates([c], GateConfig(seed_tm_max=-99.0))
    assert any("seed_tm_high" in r for r in c.gate_failures)
    c2 = Candidate("x", "t", 0, "GCGCGCGCGCAUAUAUAUAUA")
    apply_gates([c2], GateConfig(seed_tm_max=None))
    assert not any("seed_tm" in r for r in c2.gate_failures)


# --------------------------------------------------------------- off-target
def test_transcriptome_index_finds_perfect_and_mismatched():
    target = "ACGUAGCUAGCUUAGCGAUCGAUCGUAGCUAGCUAGCAUCGGAUCCGAUAGC"
    idx = TranscriptomeIndex.from_fasta.__func__  # noqa: B009
    # build in-memory instead of from file
    ti = TranscriptomeIndex()
    ti.names = ["TX1"]
    ti.seqs = [target]
    for i in range(len(target) - ti.anchor + 1):
        ti.index.setdefault(target[i : i + ti.anchor], []).append((0, i))
    site = target[5:24]
    guide = revcomp(site)
    hits = ti.scan(guide, max_mm=0)
    assert any(h[2] == 0 for h in hits)


def test_pots_lower_for_rare_seed():
    table = SeedTable(k=6, counts={"AAAAAA": 5000, "CGCGCG": 1}, n_utrs=1, total_kmers=5001)
    common = Candidate("c", "t", 0, "U" + "UUUUUU" + "A" * 14)
    rare = Candidate("r", "t", 0, "U" + "CGCGCG" + "A" * 14)
    # construct guides whose seed complement is the hexamer of interest
    s_common = revcomp("A" + "AAAAAA" + "C" * 12)[:19]
    del common, rare, s_common
    c1 = Candidate("c1", "t", 0, revcomp("N" + "AAAAAA" + "C" * 14)[:19].replace("N", "A") + "UU")
    assert pots_score(c1, table) >= 0


def test_pots_monotone():
    table = SeedTable(k=6, counts={}, n_utrs=1, total_kmers=0)
    c = Candidate("x", "t", 0, "ACGUACGUACGUACGUACGUA")
    low = pots_score(c, table)
    table.counts[revcomp(c.seed6)] = 10000
    high = pots_score(c, table)
    assert high > low


# ------------------------------------------------------------------ scoring
def test_rank_percentile_nan_safe():
    v = np.array([1.0, np.nan, 3.0, 2.0])
    p = rank_percentile(v)
    assert math.isnan(p[1])
    assert p[2] > p[3] > p[0]


def test_derringer_bounds_and_monotone():
    assert derringer(0.0, 0.2, 0.8) == 0.0
    assert derringer(0.9, 0.2, 0.8) == 1.0
    assert 0 < derringer(0.5, 0.2, 0.8) < 1
    assert derringer(0.6, 0.2, 0.8) > derringer(0.4, 0.2, 0.8)


def test_geometric_mean_zeroes_on_any_zero_objective():
    """The whole point of the geometric form: a catastrophic axis vetoes."""
    d = {"a": np.array([1.0, 0.95]), "b": np.array([1.0, 0.0])}
    w = {"a": 0.5, "b": 0.5}
    out = geometric_mean(d, w)
    assert out[0] == pytest.approx(1.0)
    assert out[1] == 0.0


def test_geometric_mean_penalises_imbalance_vs_arithmetic():
    d = {"a": np.array([0.9, 0.5]), "b": np.array([0.1, 0.5])}
    w = {"a": 0.5, "b": 0.5}
    g = geometric_mean(d, w)
    assert g[1] > g[0]  # balanced beats lopsided at equal arithmetic mean


def test_min_variance_downweights_correlated_engines():
    """Two perfectly correlated engines should together get ~the weight of one."""
    cov = np.array(
        [
            [0.04, 0.0395, 0.0],
            [0.0395, 0.04, 0.0],
            [0.0, 0.0, 0.04],
        ]
    )
    w = min_variance_weights(cov)
    assert w.sum() == pytest.approx(1.0)
    assert w[2] > w[0] and w[2] > w[1]
    assert (w[0] + w[1]) < 2 * w[2]


def test_min_variance_weights_nonnegative_and_normalised():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(5, 50))
    cov = np.cov(A)
    w = min_variance_weights(cov)
    assert (w >= 0).all()
    assert w.sum() == pytest.approx(1.0)


def test_rra_prefers_consistently_top_ranked():
    # candidate 0 ranked best by all engines; candidate 1 erratic
    ranks = np.array([[0.01, 0.5, 0.99], [0.02, 0.99, 0.5], [0.01, 0.6, 0.9]])
    s = rra_score(ranks)
    assert s[0] > s[1] and s[0] > s[2]


def test_calibration_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    n = 300
    x = rng.normal(size=n)
    y = np.clip(0.5 + 0.3 * x + rng.normal(0, 0.1, n), 0, 1)
    c = Calibration().fit({"e1": x, "e2": x + rng.normal(0, 0.5, n)}, y,
                          groups=rng.integers(0, 3, n))
    p = tmp_path / "cal.json"
    c.save(p)
    c2 = Calibration.load(p)
    assert c2.engines == c.engines
    np.testing.assert_allclose(
        c.predict("e1", x[:20]), c2.predict("e1", x[:20]), atol=1e-9
    )


def test_calibration_is_monotone():
    rng = np.random.default_rng(2)
    n = 400
    x = rng.normal(size=n)
    y = np.clip(0.5 + 0.3 * x + rng.normal(0, 0.1, n), 0, 1)
    c = Calibration().fit({"e": x}, y)
    grid = np.linspace(-2, 2, 50)
    pred = c.predict("e", grid)
    assert np.all(np.diff(pred) >= -1e-9)


def test_portfolio_enforces_spacing():
    from siforge.scoring import ScoredCandidate

    def mk(i, start, seed_src):
        c = Candidate(f"c{i}", "tx", start, seed_src)
        return ScoredCandidate(
            candidate=c, potency=0.9 - i * 0.01, potency_mean=0.9, potency_rra=0.9,
            desirability={}, uss=90 - i, uss_lcb=90 - i, pi_low=0.8, pi_high=1.0,
            per_engine_kd={}, weights={}, n_engines=3, calibrated=True,
        )

    a = mk(0, 100, "ACGUACGUACGUACGUACGUA")
    b = mk(1, 110, "ACGUACGUACGUACGUACGUA")  # too close AND same seed
    c = mk(2, 900, "GGCAUUCGAUCGGAUCCAUG")
    sel = select_portfolio([a, b, c], k=3, min_distance_nt=50)
    ids = [s.candidate_id for s in sel]
    assert "c0" in ids and "c1" not in ids and "c2" in ids
