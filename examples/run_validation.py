"""Validation example runs for the siforge v1.0 stack.

Run: python examples/run_validation.py --data demo_data

Four experiments, each checking something a reviewer would ask about:

  E1  Ablation          -- does fusing engines beat any single engine?
  E2  Generalisation    -- grouped-CV, strict holdout, calibration coverage
  E3  Specificity veto  -- does a potent-but-promiscuous guide actually lose?
  E4  Portfolio spacing -- are the picks diverse, or four views of one site?
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from siforge.engines import NATIVE_ENGINES, run_engines
from siforge.offtarget import SeedTable
from siforge.pipeline import design, load_profile
from siforge.scoring import (
    Calibration,
    ReferenceDistribution,
    UnifiedScorer,
    min_variance_weights,
)
from siforge.seq import Candidate, revcomp, to_rna

BAR = "=" * 74


REQUIRED = {
    "benchmark.tsv": "python examples/make_demo_data.py -o {d}",
    "benchmark_heldout.tsv": "python examples/make_demo_data.py -o {d}",
    "targets.fa": "python examples/make_demo_data.py -o {d}",
    "transcriptome.fa": "python examples/make_demo_data.py -o {d}",
    "utr3.fa": "python examples/make_demo_data.py -o {d}",
    "seedtable.json": "siforge build-seedtable {d}/utr3.fa -o {d}/seedtable.json",
    "calibration.json": (
        "siforge calibrate {d}/benchmark.tsv -o {d}/calibration.json "
        "--reference-out {d}/reference.json"
    ),
    "reference.json": (
        "siforge calibrate {d}/benchmark.tsv -o {d}/calibration.json "
        "--reference-out {d}/reference.json"
    ),
}


def check_inputs(data: Path) -> None:
    """Fail fast with the exact commands needed, not a pandas traceback."""
    missing = [f for f in REQUIRED if not (data / f).exists()]
    if not missing:
        return
    print(f"Missing {len(missing)} input file(s) in {data}/:", file=sys.stderr)
    for f in missing:
        print(f"  - {f}", file=sys.stderr)
    cmds = []
    for f in missing:
        c = REQUIRED[f].format(d=data)
        if c not in cmds:
            cmds.append(c)
    print("\nRun these first, in order:\n", file=sys.stderr)
    for c in cmds:
        print(f"  {c}", file=sys.stderr)
    if not data.exists():
        print(f"\n(Note: {data}/ does not exist at all. If you generated the "
              f"demo data elsewhere, pass --data <path>.)", file=sys.stderr)
    raise SystemExit(1)



def _load_bench(path: Path) -> tuple[list[Candidate], np.ndarray, np.ndarray]:
    df = pd.read_csv(path, sep="\t")
    cands = []
    for i, row in df.iterrows():
        site = to_rna(row["sense"]) + "UU"
        c = Candidate(f"b{i}", "benchmark", 0, site)
        c.features["accessibility"] = float("nan")
        cands.append(c)
    y = df["efficacy"].to_numpy(float)
    groups = (
        df["dataset"].astype("category").cat.codes.to_numpy()
        if "dataset" in df.columns
        else np.zeros(len(df), int)
    )
    return cands, y, groups


def e1_ablation(data: Path, profile: dict) -> None:
    print(f"\n{BAR}\nE1  ABLATION -- fused score vs each engine alone\n{BAR}")
    cands, y, groups = _load_bench(data / "benchmark.tsv")
    engines = [e for e in NATIVE_ENGINES if e.name != "accessibility"]
    scores, ran = run_engines(engines, cands)
    ids = [c.candidate_id for c in cands]
    raw = {e: np.array([scores[e].get(i, np.nan) for i in ids]) for e in ran}

    calib = Calibration().fit(raw, y, groups,
                              lambda_blend=float(profile.get("lambda_blend", 0.6)))
    ref = ReferenceDistribution()
    for e in ran:
        ref.add(e, raw[e])

    rows = []
    for e in ran:
        ok = ~np.isnan(raw[e])
        rows.append((e, sps.spearmanr(raw[e][ok], y[ok]).statistic, "single"))

    scorer = UnifiedScorer(profile, calib, ref, {x.name: x.family for x in engines})
    scored = scorer.score(cands, scores)
    by = {s.candidate_id: s.potency for s in scored}
    pred = np.array([by.get(i, np.nan) for i in ids])
    ok = ~np.isnan(pred)
    rows.append(("FUSED (USS potency)", sps.spearmanr(pred[ok], y[ok]).statistic, "fused"))

    # unweighted-mean control: does correlation-aware weighting actually help?
    kd = np.vstack([calib.predict(e, raw[e]) for e in ran])
    naive = np.nanmean(kd, axis=0)
    ok2 = ~np.isnan(naive)
    rows.append(("naive equal-weight mean", sps.spearmanr(naive[ok2], y[ok2]).statistic,
                 "control"))

    rows.sort(key=lambda r: -abs(r[1]))
    print(f"{'engine':<26s} {'Spearman rho':>13s}   kind")
    for name, rho, kind in rows:
        mark = " <--" if kind == "fused" else ""
        print(f"{name:<26s} {rho:>+13.4f}   {kind}{mark}")

    best_single = max(abs(r[1]) for r in rows if r[2] == "single")
    fused = abs([r[1] for r in rows if r[2] == "fused"][0])
    print(f"\nfused vs best single engine: {fused:+.4f} vs {best_single:+.4f} "
          f"({'+' if fused > best_single else ''}{(fused - best_single):.4f})")

    if calib.residual_cov is not None:
        w = min_variance_weights(calib.residual_cov)
        print("\nmin-variance weights (correlated engines share one vote):")
        for e, wi in sorted(zip(calib.engines, w), key=lambda t: -t[1]):
            print(f"  {e:<22s} {wi:.4f}")

    # --- lambda sweep: what does the RRA component cost or buy? ---------
    print("\nlambda sweep (0 = pure rank aggregation, 1 = pure weighted mean):")
    print(f"  {'lambda':>7s} {'rho (all engines)':>18s} {'rho (2 engines dropped)':>24s}")
    dropped = [e for e in ran if e not in ("pwm", "thermo_asymmetry")]
    for lam in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        prof = dict(profile)
        prof["lambda_blend"] = lam
        sc = UnifiedScorer(prof, calib, ref,
                           {x.name: x.family for x in engines}).score(cands, scores)
        p = np.array([{s.candidate_id: s.potency for s in sc}.get(i, np.nan)
                      for i in ids])
        m = ~np.isnan(p)
        rho_all = sps.spearmanr(p[m], y[m]).statistic
        sub = {e: scores[e] for e in dropped}
        sc2 = UnifiedScorer(prof, calib, ref,
                            {x.name: x.family for x in engines}).score(cands, sub)
        p2 = np.array([{s.candidate_id: s.potency for s in sc2}.get(i, np.nan)
                       for i in ids])
        m2 = ~np.isnan(p2)
        rho_sub = sps.spearmanr(p2[m2], y[m2]).statistic
        print(f"  {lam:>7.1f} {rho_all:>+18.4f} {rho_sub:>+24.4f}")
    print("  (the right-hand column is the robustness case the RRA term exists for:")
    print("   engines missing at run time, which is the normal state of this stack)")


def e2_generalisation(data: Path, profile: dict) -> None:
    print(f"\n{BAR}\nE2  GENERALISATION -- strict holdout + conformal coverage\n{BAR}")
    tr_c, tr_y, tr_g = _load_bench(data / "benchmark.tsv")
    te_c, te_y, _ = _load_bench(data / "benchmark_heldout.tsv")
    engines = [e for e in NATIVE_ENGINES if e.name != "accessibility"]

    s_tr, ran = run_engines(engines, tr_c)
    ids_tr = [c.candidate_id for c in tr_c]
    raw_tr = {e: np.array([s_tr[e].get(i, np.nan) for i in ids_tr]) for e in ran}
    calib = Calibration().fit(raw_tr, tr_y, tr_g,
                              lambda_blend=float(profile.get("lambda_blend", 0.6)))
    ref = ReferenceDistribution()
    for e in ran:
        ref.add(e, raw_tr[e])

    s_te, _ = run_engines(engines, te_c)
    scorer = UnifiedScorer(profile, calib, ref, {x.name: x.family for x in engines})
    scored = scorer.score(te_c, s_te)
    by = {s.candidate_id: s for s in scored}
    ids_te = [c.candidate_id for c in te_c]
    pred = np.array([by[i].potency if i in by else np.nan for i in ids_te])
    lo = np.array([by[i].pi_low if i in by else np.nan for i in ids_te])
    hi = np.array([by[i].pi_high if i in by else np.nan for i in ids_te])

    ok = ~np.isnan(pred)
    rho = sps.spearmanr(pred[ok], te_y[ok]).statistic
    cov = float(np.mean((te_y[ok] >= lo[ok]) & (te_y[ok] <= hi[ok])))
    width = float(np.mean(hi[ok] - lo[ok]))
    lab = (te_y >= 0.70).astype(int)
    order = np.argsort(-np.nan_to_num(pred, nan=-1))
    print(f"  n (held out)        {int(ok.sum())}")
    print(f"  Spearman rho        {rho:+.4f}")
    print(f"  precision@10        {lab[order[:10]].mean():.3f}")
    print(f"  conformal coverage  {cov:.3f}   (nominal 0.900)")
    print(f"  mean PI width       {width:.3f}")
    verdict = "VALID" if abs(cov - 0.90) < 0.06 else "MISCALIBRATED"
    print(f"  -> interval calibration: {verdict}")


def e3_specificity_veto(data: Path) -> None:
    """Does the specificity axis actually reorder otherwise-equal candidates?

    Constructing artificial probes confounds potency with specificity, so
    this uses REAL tiled candidates from the demo targets: find pairs whose
    fused potency is nearly identical (within 0.02) but whose seed
    promiscuity differs materially (POTS gap > 0.5 log10), then check how
    often the less promiscuous member ranks higher on USS.

    Under a weighted *arithmetic* mean this reordering is weak; under the
    geometric form it should be near-total.
    """
    print(f"\n{BAR}\nE3  SPECIFICITY VETO -- potency-matched pairs\n{BAR}")
    profile = load_profile("research")
    results = design(
        fasta=data / "targets.fa",
        profile=profile,
        transcriptome=data / "transcriptome.fa",
        seed_table_json=data / "seedtable.json",
        calibration=data / "calibration.json",
        reference=data / "reference.json",
        top_k=4,
    )
    scored = [s for r in results for s in r.scored]
    pot = np.array([s.potency for s in scored])
    pots = np.array([s.candidate.features.get("pots", np.nan) for s in scored])
    acc = np.array([s.desirability.get("accessibility", np.nan) for s in scored])

    def matched(i, j, pots_gap=0.4):
        """Potency AND accessibility matched; only seed promiscuity differs.

        Controlling accessibility matters: in AU-rich 3'UTRs the promiscuous
        hexamers are AU-rich, and AU-rich target sites are also less
        structured, so an uncontrolled comparison measures accessibility
        rather than specificity.
        """
        if abs(pot[i] - pot[j]) > 0.02:
            return False
        if not np.isfinite(acc[i]) or not np.isfinite(acc[j]):
            return False
        if abs(acc[i] - acc[j]) > 0.05:
            return False
        g = pots[i] - pots[j]
        return np.isfinite(g) and abs(g) >= pots_gap

    pairs, correct = 0, 0
    examples = []
    n = len(scored)
    for i in range(n):
        for j in range(i + 1, n):
            if not matched(i, j):
                continue
            pairs += 1
            lo_i = i if pots[i] < pots[j] else j
            hi_i = j if lo_i == i else i
            ok = scored[lo_i].uss > scored[hi_i].uss
            correct += ok
            if len(examples) < 4:
                examples.append((scored[lo_i], scored[hi_i], ok))

    print(f"  potency+accessibility-matched pairs with a POTS gap: {pairs}")
    if pairs:
        print(f"  less-promiscuous member ranks higher: {correct}/{pairs} "
              f"({100 * correct / pairs:.1f}%)")
    print()
    print(f"  {'':>3s} {'candidate':<22s} {'potency':>8s} {'POTS':>6s} "
          f"{'d_spec':>7s} {'USS':>7s}")
    for lo, hi, ok in examples:
        tag = "OK " if ok else "BAD"
        for lbl, s in (("lo", lo), ("hi", hi)):
            print(f"  {lbl:>3s} {s.candidate.candidate_id:<22s} {s.potency:>8.3f} "
                  f"{s.candidate.features.get('pots', float('nan')):>6.2f} "
                  f"{s.desirability.get('specificity', float('nan')):>7.3f} "
                  f"{s.uss:>7.2f}")
        print(f"      -> {tag}")

    # --- exact ceteris-paribus sweep -------------------------------
    # The pairwise statistic above still lets other axes co-vary. Here we
    # take ONE real candidate and vary ONLY its seed-promiscuity feature,
    # which isolates the axis completely.
    probe = max(scored, key=lambda s: s.uss).candidate
    print("\n  ceteris-paribus: one real candidate, only POTS varied")
    print(f"  ({probe.candidate_id}, everything else held fixed)")
    print(f"    {'POTS':>6s} {'d_spec':>8s} {'USS':>8s}")
    scorer = UnifiedScorer(profile, None, None, {})
    prev = None
    monotone = True
    for v in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0):
        import copy
        c = copy.deepcopy(probe)
        c.features["pots"] = v
        c.gate_failures = []
        fake_scores = {e: {c.candidate_id: probe.engine_scores[e]}
                       for e in probe.engine_scores}
        out = scorer.score([c], fake_scores)[0]
        print(f"    {v:>6.1f} {out.desirability['specificity']:>8.3f} "
              f"{out.uss:>8.2f}")
        if prev is not None and out.uss > prev + 1e-9:
            monotone = False
        prev = out.uss
    print(f"    -> USS strictly non-increasing in seed promiscuity: "
          f"{'YES' if monotone else 'NO'}")

    # Control: what would a weighted ARITHMETIC mean have done?
    obj = profile["objectives"]
    w = {k: v["weight"] for k, v in obj.items()}
    arith, ag = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            if not matched(i, j):
                continue
            lo_i = i if pots[i] < pots[j] else j
            hi_i = j if lo_i == i else i
            def am(s):
                tot = sum(w.get(k, 0) for k in s.desirability)
                return sum(w.get(k, 0) * v for k, v in s.desirability.items()) / max(tot, 1e-9)
            ag += 1
            arith += am(scored[lo_i]) > am(scored[hi_i])
    if ag:
        print(f"\n  geometric USS : {100 * correct / pairs:.1f}% correct")
        print(f"  arithmetic mean: {100 * arith / ag:.1f}% correct  (control)")


def e4_portfolio(data: Path) -> None:
    print(f"\n{BAR}\nE4  PORTFOLIO -- picks must not share a failure mode\n{BAR}")
    profile = load_profile("research")
    results = design(
        fasta=data / "targets.fa",
        profile=profile,
        transcriptome=data / "transcriptome.fa",
        seed_table_json=data / "seedtable.json",
        calibration=data / "calibration.json",
        reference=data / "reference.json",
        top_k=4,
    )
    for res in results:
        print(f"\n{res.transcript_id}  "
              f"(tiled {res.n_enumerated}, gated {res.n_gated}, "
              f"scored {len(res.scored)}, W={res.consensus_w:.3f})")
        print(f"  {'candidate':<22s} {'pos':>6s} {'USS':>7s} {'LCB':>7s} "
              f"{'potency':>8s} {'seed':>8s}")
        pos = []
        for s in res.portfolio:
            c = s.candidate
            pos.append(c.start)
            print(f"  {c.candidate_id:<22s} {c.start + 1:>6d} {s.uss:>7.2f} "
                  f"{s.uss_lcb:>7.2f} {s.potency:>8.3f} {c.seed6:>8s}")
        if len(pos) > 1:
            gaps = [abs(a - b) for i, a in enumerate(pos) for b in pos[i + 1:]]
            print(f"  min pairwise spacing: {min(gaps)} nt "
                  f"({'OK' if min(gaps) >= 50 else 'TOO CLOSE'})")
        seeds = [s.candidate.seed6 for s in res.portfolio]
        print(f"  distinct seeds: {len(set(seeds))}/{len(seeds)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="demo_data")
    ap.add_argument("--only", default=None, help="e1|e2|e3|e4")
    args = ap.parse_args()
    data = Path(args.data)
    check_inputs(data)
    profile = load_profile("research")

    run = args.only
    if run in (None, "e1"):
        e1_ablation(data, profile)
    if run in (None, "e2"):
        e2_generalisation(data, profile)
    if run in (None, "e3"):
        e3_specificity_veto(data)
    if run in (None, "e4"):
        e4_portfolio(data)
    print(f"\n{BAR}\ndone\n{BAR}")


if __name__ == "__main__":
    main()
