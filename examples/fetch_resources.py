"""Build siforge's real-data resources from public sources.

Three subcommands, matching the three inputs siforge needs beyond your
target sequence:

  utr3       Extract a 3'UTR FASTA from GENCODE pc_transcripts.
  benchmark  Convert published siRNA efficacy datasets to siforge's TSV.
  check      Sanity-check resources you have already built.

Nothing here downloads anything: you fetch the source files with curl or
wget (the commands are printed by `--help-urls`) and point these at them.
Keeping the download step manual means the provenance of every resource is
explicit and reproducible, and it works on clusters with no outbound
network from the compute nodes.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import re
import sys
from pathlib import Path

URLS = """
Transcriptome FASTA (for full-length off-target scanning)
---------------------------------------------------------
  GENCODE human, release 50 (GRCh38):

  curl -O https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_50/gencode.v50.transcripts.fa.gz

  Mouse (release M37):
  curl -O https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M37/gencode.vM37.transcripts.fa.gz

3'UTR source (for the seed-promiscuity / POTS table)
-----------------------------------------------------
  GENCODE does not publish a 3'UTR FASTA. Download the protein-coding
  transcripts file -- its headers carry UTR3 coordinates -- and run
  `fetch_resources.py utr3` on it:

  curl -O https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_50/gencode.v50.pc_transcripts.fa.gz
  python examples/fetch_resources.py utr3 gencode.v50.pc_transcripts.fa.gz -o gencode_v50_3utr.fa

Benchmark datasets (for calibration)
-------------------------------------
  The published siRNA efficacy datasets, already harmonised to 19-nt
  guides with efficacy in [0,1], are redistributed in the OligoFormer
  repository:

  git clone --depth 1 https://github.com/lulab/OligoFormer
  python examples/fetch_resources.py benchmark OligoFormer/data -o benchmark.tsv

  Hu.csv   = Huesken et al. 2005 (2,361 siRNAs)
  Taka.csv = Takayuki/reporter dataset (702)
  Mix.csv  = mixed smaller sets (472)

  Cite the ORIGINAL publications, not OligoFormer alone.
"""


# ---------------------------------------------------------------- utr3
HEADER_UTR3 = re.compile(r"\|UTR3:(\d+)-(\d+)\|")


def _open(path: str | Path):
    p = str(path)
    return gzip.open(p, "rt") if p.endswith(".gz") else open(p, "rt")


def cmd_utr3(args: argparse.Namespace) -> int:
    """Extract 3'UTR sequences using the UTR3 span in each GENCODE header.

    GENCODE's pc_transcripts headers are pipe-delimited and end with the
    region spans, e.g.

      >ENST00000335137.4|ENSG00000186092.6|...|1054|UTR5:1-36|CDS:37-954|UTR3:955-1054|

    The coordinates are 1-based inclusive and relative to the transcript,
    so no genome FASTA or GTF is needed. Transcripts with no UTR3 field
    (incomplete 3' ends, non-stop-decay) are skipped and counted.
    """
    n_in = n_out = n_no_utr3 = n_short = 0
    total_len = 0
    out = Path(args.out)
    with _open(args.pc_transcripts) as fh, open(out, "w") as w:
        name = None
        span = None
        chunks: list[str] = []

        def flush() -> None:
            nonlocal n_out, n_short, total_len
            if name is None or span is None:
                return
            seq = "".join(chunks)
            s, e = span
            sub = seq[s - 1 : e]
            if len(sub) < args.min_len:
                n_short += 1
                return
            w.write(f">{name}\n")
            for i in range(0, len(sub), 60):
                w.write(sub[i : i + 60] + "\n")
            n_out += 1
            total_len += len(sub)

        for line in fh:
            if line.startswith(">"):
                flush()
                chunks = []
                n_in += 1
                header = line[1:].rstrip()
                m = HEADER_UTR3.search(line)
                if not m:
                    name, span = None, None
                    n_no_utr3 += 1
                    continue
                name = header.split("|")[0]
                span = (int(m.group(1)), int(m.group(2)))
            else:
                if span is not None:
                    chunks.append(line.strip())
        flush()

    if n_out == 0:
        print(
            "ERROR: no 3'UTRs extracted. Check that this is the "
            "*pc_transcripts* file -- the plain `transcripts` file has no "
            "UTR3 fields in its headers.",
            file=sys.stderr,
        )
        return 2
    print(f"transcripts read        {n_in}")
    print(f"  no UTR3 in header     {n_no_utr3}")
    print(f"  shorter than {args.min_len:<3d}      {n_short}")
    print(f"3'UTRs written          {n_out}  (mean {total_len / n_out:.0f} nt)")
    print(f"-> {out}")
    print(f"\nNext:  siforge build-seedtable {out} -o seedtable.json")
    return 0


# ----------------------------------------------------------- benchmark
def _rc(s: str) -> str:
    return "".join({"A": "U", "U": "A", "G": "C", "C": "G"}.get(c, "N")
                   for c in reversed(s))


DATASET_FILES = {
    "Hu.csv": "huesken",
    "Taka.csv": "taka",
    "Mix.csv": "mix",
}


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Convert OligoFormer-format CSVs to a siforge benchmark TSV.

    Input columns: siRNA, mRNA, label, y, td
      siRNA  -- the GUIDE (antisense) strand, 19 nt. Verified here rather
                than assumed: a silently flipped strand would corrupt
                every calibration curve downstream without erroring.
      label  -- continuous efficacy in [0,1]  (this is what we want)
      y      -- binary class at a 0.7 threshold (ignored)

    Output columns: guide, sense, efficacy, dataset
    """
    src = Path(args.source)
    files = []
    if src.is_dir():
        for fn, tag in DATASET_FILES.items():
            p = src / fn
            if p.exists():
                files.append((p, tag))
    else:
        files.append((src, src.stem.lower()))
    if not files:
        print(f"ERROR: no known dataset CSVs found in {src}", file=sys.stderr)
        print(f"Expected any of: {', '.join(DATASET_FILES)}", file=sys.stderr)
        return 2

    rows = []
    for path, tag in files:
        n = n_orient_ok = 0
        with open(path) as fh:
            for r in csv.DictReader(fh):
                guide = r["siRNA"].strip().upper().replace("T", "U")
                mrna = r.get("mRNA", "").strip().upper().replace("T", "U")
                try:
                    eff = float(r["label"])
                except (KeyError, ValueError):
                    continue
                if not guide or not (0.0 <= eff <= 1.0):
                    continue
                n += 1
                sense = _rc(guide)
                if mrna and sense in mrna:
                    n_orient_ok += 1
                rows.append({
                    "guide": guide,
                    "sense": sense,
                    "efficacy": f"{eff:.4f}",
                    "dataset": tag,
                })
        frac = n_orient_ok / n if n else 0.0
        flag = "OK" if frac > 0.9 else "SUSPECT"
        print(f"{path.name:<20s} n={n:<6d} guide-orientation verified "
              f"{n_orient_ok}/{n} ({frac:.0%}) {flag}")
        if frac < 0.9 and n:
            print(
                "  WARNING: the siRNA column does not look like the guide "
                "strand in this file. Calibration fitted on a flipped "
                "strand will be silently wrong -- inspect before using.",
                file=sys.stderr,
            )

    out = Path(args.out)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["guide", "sense", "efficacy", "dataset"],
                           delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    if args.holdout:
        ho = [r for r in rows if r["dataset"] == args.holdout]
        tr = [r for r in rows if r["dataset"] != args.holdout]
        if not ho:
            print(f"WARNING: no rows for holdout '{args.holdout}'", file=sys.stderr)
        else:
            for name, subset in ((out, tr), (out.with_name(out.stem + "_heldout.tsv"), ho)):
                with open(name, "w", newline="") as fh:
                    w = csv.DictWriter(
                        fh, fieldnames=["guide", "sense", "efficacy", "dataset"],
                        delimiter="\t")
                    w.writeheader()
                    w.writerows(subset)
            print(f"\nheld out '{args.holdout}': {len(ho)} rows -> "
                  f"{out.with_name(out.stem + '_heldout.tsv')}")
            print(f"training set: {len(tr)} rows -> {out}")

    print(f"\ntotal {len(rows)} siRNAs -> {out}")
    print("\nNext:  siforge calibrate "
          f"{out} -o calibration.json --reference-out reference.json")
    print("\nCite the original publications (Huesken et al. 2005 Nat Biotechnol "
          "23:995, etc.), not the redistribution.")
    return 0


# --------------------------------------------------------------- check
def cmd_check(args: argparse.Namespace) -> int:
    ok = True
    for label, path, kind in (
        ("transcriptome", args.transcriptome, "fasta"),
        ("3'UTR", args.utr3, "fasta"),
        ("benchmark", args.benchmark, "tsv"),
    ):
        if not path:
            continue
        p = Path(path)
        if not p.exists():
            print(f"{label:<14s} MISSING  {p}")
            ok = False
            continue
        if kind == "fasta":
            n = ln = 0
            with _open(p) as fh:
                for line in fh:
                    if line.startswith(">"):
                        n += 1
                    else:
                        ln += len(line.strip())
            print(f"{label:<14s} OK  {n} sequences, {ln / 1e6:.1f} Mb  {p.name}")
            if label == "3'UTR" and n < 1000:
                print("               NOTE: fewer than 1,000 3'UTRs. The POTS "
                      "hexamer table will be noisy.")
        else:
            with open(p) as fh:
                head = fh.readline().rstrip("\n").split("\t")
                n = sum(1 for _ in fh)
            need = {"efficacy"}
            has_seq = {"guide", "sense", "site"} & set(head)
            print(f"{label:<14s} OK  {n} rows, columns: {','.join(head)}  {p.name}")
            if not need <= set(head) or not has_seq:
                print("               ERROR: needs an `efficacy` column and one "
                      "of guide/sense/site.")
                ok = False
            if "dataset" not in head:
                print("               WARNING: no `dataset` column -- grouped "
                      "cross-validation is disabled and performance estimates "
                      "will be optimistic.")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--help-urls", action="store_true",
                    help="print download commands for every source file")
    sub = ap.add_subparsers(dest="cmd")

    u = sub.add_parser("utr3", help="extract 3'UTR FASTA from GENCODE pc_transcripts")
    u.add_argument("pc_transcripts", help="gencode.vNN.pc_transcripts.fa[.gz]")
    u.add_argument("-o", "--out", default="gencode_3utr.fa")
    u.add_argument("--min-len", type=int, default=20)
    u.set_defaults(func=cmd_utr3)

    b = sub.add_parser("benchmark", help="convert published datasets to siforge TSV")
    b.add_argument("source", help="OligoFormer/data directory, or a single CSV")
    b.add_argument("-o", "--out", default="benchmark.tsv")
    b.add_argument("--holdout", help="dataset tag to hold out (e.g. taka)")
    b.set_defaults(func=cmd_benchmark)

    c = sub.add_parser("check", help="sanity-check built resources")
    c.add_argument("--transcriptome")
    c.add_argument("--utr3")
    c.add_argument("--benchmark")
    c.set_defaults(func=cmd_check)

    args = ap.parse_args()
    if args.help_urls:
        print(URLS)
        return 0
    if not getattr(args, "cmd", None):
        ap.print_help()
        print(URLS)
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
