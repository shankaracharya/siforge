"""siforge command line interface."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .accessibility import HAVE_VIENNA
from .engines import NATIVE_ENGINES, build_external_engines, describe_engines, load_external_specs
from .offtarget import build_seed_table
from .pipeline import design, load_profile, results_to_rows
from .report import write_html_report
from .scoring import Calibration, ReferenceDistribution


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--profile", default="research", help="profile name or YAML path")
    p.add_argument("--quiet", action="store_true")


def cmd_design(args: argparse.Namespace) -> int:
    profile = load_profile(args.profile)
    results = design(
        fasta=args.fasta,
        profile=profile,
        transcriptome=args.transcriptome,
        utr3=args.utr3,
        seed_table_json=args.seed_table,
        calibration=args.calibration,
        reference=args.reference,
        pwm=args.pwm,
        duplex_len=args.duplex_len,
        start_offset=args.start_offset,
        top_k=args.top,
        max_mm=args.max_mismatches,
        force_fallback_accessibility=args.no_vienna,
    )
    rows = results_to_rows(results)
    if not rows:
        print("No candidates survived gating.", file=sys.stderr)
        for r in results:
            print(f"  {r.transcript_id}: {r.gate_reasons}", file=sys.stderr)
        return 2

    df = pd.DataFrame(rows)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "candidates.tsv", sep="\t", index=False)
    try:
        df.to_parquet(outdir / "candidates.parquet", index=False)
    except Exception:
        pass

    port = df[df.in_portfolio]
    port.to_csv(outdir / "order_sheet.tsv", sep="\t", index=False)

    manifest = {
        "siforge_version": __version__,
        "profile": profile.get("name", args.profile),
        "accessibility": results[0].accessibility_source if results else "n/a",
        "viennarna": HAVE_VIENNA,
        "calibrated": bool(results and results[0].calibrated),
        "engines_run": results[0].engines_run if results else [],
        "transcripts": {
            r.transcript_id: {
                "enumerated": r.n_enumerated,
                "gated": r.n_gated,
                "consensus_W": None if np.isnan(r.consensus_w) else round(r.consensus_w, 3),
                "gate_reasons": r.gate_reasons,
            }
            for r in results
        },
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    write_html_report(results, outdir / "report.html", manifest)

    if not args.quiet:
        if not manifest["calibrated"]:
            print(
                "NOTE: no calibration loaded -- potency is a rank percentile, "
                "not an absolute knockdown estimate. Run `siforge calibrate`.",
                file=sys.stderr,
            )
        cols = ["candidate_id", "position", "guide_5to3", "USS", "USS_LCB",
                "potency", "d_specificity", "n_engines"]
        cols = [c for c in cols if c in port.columns]
        print(port[cols].to_string(index=False))
        print(f"\nWrote {outdir}/ (candidates.tsv, order_sheet.tsv, report.html)")
    return 0


def cmd_seedtable(args: argparse.Namespace) -> int:
    table = build_seed_table(args.utr3, k=args.k)
    table.to_json(args.out)
    nz = sum(1 for v in table.counts.values() if v)
    print(
        f"{table.n_utrs} 3'UTRs, {table.total_kmers} {args.k}-mers, "
        f"{nz}/{4**args.k} distinct observed -> {args.out}"
    )
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Fit isotonic calibration + residual covariance + conformal quantiles.

    Input TSV must contain at minimum: ``guide``, ``sense`` (or ``site``),
    ``efficacy`` (fractional knockdown in [0,1]) and optionally ``dataset``
    for grouped cross-fitting.
    """
    from .engines import run_engines
    from .seq import Candidate, to_rna

    df = pd.read_csv(args.benchmark, sep="\t")
    required = {"efficacy"}
    if not required <= set(df.columns):
        print(f"benchmark must contain columns: {required}", file=sys.stderr)
        return 2

    cands = []
    for i, row in df.iterrows():
        if "site" in df.columns and isinstance(row.get("site"), str):
            site = to_rna(row["site"])
        elif "sense" in df.columns and isinstance(row.get("sense"), str):
            site = to_rna(row["sense"]) + "UU"
        else:
            from .seq import revcomp

            site = revcomp(to_rna(row["guide"]))[:19] + "UU"
        cands.append(
            Candidate(
                candidate_id=f"b{i}", transcript_id="benchmark", start=0, site=site
            )
        )
    # accessibility is undefined for isolated duplexes in a benchmark table
    for c in cands:
        c.features.setdefault("accessibility", float("nan"))

    profile = load_profile(args.profile)
    engines = list(NATIVE_ENGINES) + build_external_engines(load_external_specs(profile))
    engines = [e for e in engines if e.name != "accessibility"]
    scores, ran = run_engines(engines, cands)

    ids = [c.candidate_id for c in cands]
    raw = {e: np.array([scores[e].get(i, np.nan) for i in ids]) for e in ran}
    y = df["efficacy"].to_numpy(dtype=float)
    if y.max() > 1.5:  # percentages
        y = y / 100.0
    groups = (
        df["dataset"].astype("category").cat.codes.to_numpy()
        if "dataset" in df.columns
        else None
    )

    calib = Calibration().fit(raw, y, groups, alpha=args.alpha,
                              lambda_blend=float(profile.get('lambda_blend', 0.6)))
    calib.save(args.out)

    ref = ReferenceDistribution()
    for e in ran:
        ref.add(e, raw[e])
    ref.save(args.reference_out)

    print(f"Calibrated {len(ran)} engines on n={len(y)} ({calib.n_calib} OOF)")
    print(f"  conformal q (alpha={args.alpha}): "
          f"{ {k: round(v,4) for k,v in calib.conformal_q.items()} }")
    if calib.residual_cov is not None:
        from .scoring import min_variance_weights

        w = min_variance_weights(calib.residual_cov)
        print("  min-variance weights:")
        for e, wi in sorted(zip(calib.engines, w), key=lambda t: -t[1]):
            print(f"    {e:<20s} {wi:.4f}")
    print(f"  -> {args.out}, {args.reference_out}")
    return 0


def cmd_engines(args: argparse.Namespace) -> int:
    profile = load_profile(args.profile)
    engines = list(NATIVE_ENGINES) + build_external_engines(load_external_specs(profile))
    print(describe_engines(engines))
    print(f"\nViennaRNA available: {HAVE_VIENNA}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Score a labelled benchmark and report ranking metrics."""
    from scipy import stats as sps

    from .engines import run_engines
    from .scoring import UnifiedScorer
    from .seq import Candidate, revcomp, to_rna

    df = pd.read_csv(args.benchmark, sep="\t")
    cands = []
    for i, row in df.iterrows():
        if "site" in df.columns and isinstance(row.get("site"), str):
            site = to_rna(row["site"])
        elif "sense" in df.columns and isinstance(row.get("sense"), str):
            site = to_rna(row["sense"]) + "UU"
        else:
            site = revcomp(to_rna(row["guide"]))[:19] + "UU"
        c = Candidate(f"b{i}", "benchmark", 0, site)
        c.features["accessibility"] = float("nan")
        cands.append(c)

    profile = load_profile(args.profile)
    engines = [e for e in NATIVE_ENGINES if e.name != "accessibility"]
    scores, ran = run_engines(engines, cands)
    calib = Calibration.load(args.calibration) if args.calibration else None
    ref = ReferenceDistribution.load(args.reference) if args.reference else None
    scorer = UnifiedScorer(profile, calib, ref, {e.name: e.family for e in engines})
    scored = scorer.score(cands, scores)

    by_id = {s.candidate_id: s for s in scored}
    y = df["efficacy"].to_numpy(dtype=float)
    if y.max() > 1.5:
        y = y / 100.0
    pred = np.array([by_id[f"b{i}"].potency if f"b{i}" in by_id else np.nan
                     for i in range(len(df))])
    ok = ~np.isnan(pred)
    rho = sps.spearmanr(pred[ok], y[ok]).statistic
    r = sps.pearsonr(pred[ok], y[ok]).statistic

    thr = args.threshold
    lab = (y >= thr).astype(int)
    order = np.argsort(-np.nan_to_num(pred, nan=-1))
    p_at_5 = lab[order[:5]].mean() if len(order) >= 5 else float("nan")
    p_at_10 = lab[order[:10]].mean() if len(order) >= 10 else float("nan")
    ndcg = _ndcg(lab[order], k=10)

    print(f"n = {ok.sum()}   engines = {','.join(ran)}")
    print(f"  Spearman rho      {rho:+.4f}")
    print(f"  Pearson r         {r:+.4f}")
    print(f"  precision@5       {p_at_5:.3f}  (label: efficacy >= {thr})")
    print(f"  precision@10      {p_at_10:.3f}")
    print(f"  NDCG@10           {ndcg:.4f}")
    if calib is not None:
        cov = np.mean(
            [
                by_id[f"b{i}"].pi_low <= y[i] <= by_id[f"b{i}"].pi_high
                for i in range(len(df))
                if f"b{i}" in by_id
            ]
        )
        print(f"  conformal coverage {cov:.3f}  (nominal {1 - args.alpha:.2f})")
    return 0


def _ndcg(rel: np.ndarray, k: int = 10) -> float:
    k = min(k, len(rel))
    gains = rel[:k] / np.log2(np.arange(2, k + 2))
    ideal = np.sort(rel)[::-1][:k] / np.log2(np.arange(2, k + 2))
    return float(gains.sum() / ideal.sum()) if ideal.sum() > 0 else float("nan")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="siforge", description=__doc__)
    ap.add_argument("--version", action="version", version=f"siforge {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("design", help="design siRNAs against a FASTA")
    d.add_argument("fasta")
    d.add_argument("-o", "--outdir", default="siforge_out")
    d.add_argument("--transcriptome", help="FASTA for full-length off-target scanning")
    d.add_argument("--utr3", help="3'UTR FASTA for the POTS seed table")
    d.add_argument("--seed-table", help="prebuilt seed table JSON (faster)")
    d.add_argument("--calibration", help="calibration JSON from `siforge calibrate`")
    d.add_argument("--reference", help="reference distribution JSON")
    d.add_argument("--pwm", help="position-weight TSV (i-Score / DSIR coefficients)")
    d.add_argument("--duplex-len", type=int, default=19)
    d.add_argument("--start-offset", type=int, default=75,
                   help="skip this many nt from the 5' end (default 75)")
    d.add_argument("--top", type=int, default=4, help="portfolio size")
    d.add_argument("--max-mismatches", type=int, default=3)
    d.add_argument("--no-vienna", action="store_true",
                   help="force the coarse accessibility fallback")
    _add_common(d)
    d.set_defaults(func=cmd_design)

    s = sub.add_parser("build-seedtable", help="build a 3'UTR hexamer table")
    s.add_argument("utr3")
    s.add_argument("-o", "--out", default="seedtable.json")
    s.add_argument("-k", type=int, default=6)
    s.set_defaults(func=cmd_seedtable)

    c = sub.add_parser("calibrate", help="fit isotonic calibration + UQ artefacts")
    c.add_argument("benchmark", help="TSV with guide/sense/site + efficacy [+ dataset]")
    c.add_argument("-o", "--out", default="calibration.json")
    c.add_argument("--reference-out", default="reference.json")
    c.add_argument("--alpha", type=float, default=0.10)
    _add_common(c)
    c.set_defaults(func=cmd_calibrate)

    v = sub.add_parser("validate", help="score a labelled benchmark, report metrics")
    v.add_argument("benchmark")
    v.add_argument("--calibration")
    v.add_argument("--reference")
    v.add_argument("--threshold", type=float, default=0.70)
    v.add_argument("--alpha", type=float, default=0.10)
    _add_common(v)
    v.set_defaults(func=cmd_validate)

    e = sub.add_parser("engines", help="list engines and availability")
    _add_common(e)
    e.set_defaults(func=cmd_engines)

    args = ap.parse_args(argv)
    if getattr(args, "quiet", False):
        warnings.simplefilter("ignore")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
