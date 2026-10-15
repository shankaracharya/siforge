"""Generate a self-contained demo dataset.

Why synthetic? siforge containers have no network access to NCBI/Ensembl,
and reciting real transcript sequences from memory would be worse than
useless. These files exercise every code path end to end; swap in real
FASTA (GENCODE transcripts + 3'UTRs) for real work -- the commands are
identical.

The synthetic benchmark has a *known* generative model: efficacy depends
on duplex-end asymmetry, AU content at the sense 3' end, GC fraction and
a few position effects, plus noise. That makes `siforge validate` a real
test -- the engines must recover a signal they were not handed.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

from siforge.seq import gc_fraction, au_count, revcomp, write_fasta
from siforge.thermo import ddg_asymmetry

RNG = random.Random(20261014)
NP = np.random.default_rng(20261014)
BASES = "ACGU"


def rand_seq(n: int, gc: float = 0.48) -> str:
    w = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]  # A C G U
    return "".join(RNG.choices(BASES, weights=w, k=n))


def latent_efficacy(sense: str) -> float:
    """Ground-truth model the engines must rediscover."""
    guide = revcomp(sense)
    z = 0.0
    z += 0.45 * np.tanh(ddg_asymmetry(guide, 4) / 2.0)
    z += 0.30 * (au_count(sense[14:19]) / 5.0 - 0.5) * 2
    gc = gc_fraction(sense)
    z += -1.6 * (gc - 0.45) ** 2 * 4
    z += 0.22 if sense[18] in "AU" else -0.22
    z += 0.12 if sense[0] in "GC" else -0.12
    z += 0.10 if sense[9] == "U" else 0.0
    z += 0.08 if sense[2] == "A" else 0.0
    z += float(NP.normal(0, 0.42))
    return float(np.clip(0.5 + 0.28 * z, 0.0, 1.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--outdir", default="demo_data")
    ap.add_argument("--n-utr", type=int, default=400)
    ap.add_argument("--n-tx", type=int, default=120)
    ap.add_argument("--n-bench", type=int, default=900)
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- targets ----------------------------------------------------
    targets = []
    for name, length, gc in [
        ("DEMO_TARGET_A", 1800, 0.47),
        ("DEMO_TARGET_B", 1200, 0.55),
    ]:
        targets.append((name, rand_seq(length, gc)))
    write_fasta(out / "targets.fa", targets)

    # ---- a "paralog" that shares a 60-nt block with target A --------
    a = dict(targets)["DEMO_TARGET_A"]
    paralog = rand_seq(900, 0.5)
    paralog = paralog[:300] + a[600:660] + paralog[360:]

    # ---- transcriptome (includes the targets + the paralog) ---------
    tx = list(targets) + [("DEMO_PARALOG", paralog)]
    tx += [(f"BG_TX_{i:04d}", rand_seq(RNG.randint(600, 2400), RNG.uniform(0.38, 0.6)))
           for i in range(args.n_tx)]
    write_fasta(out / "transcriptome.fa", tx)

    # ---- 3'UTRome, with deliberately skewed hexamer usage -----------
    # Real 3'UTRs have strongly non-uniform hexamer frequencies; a uniform
    # random set would make every seed look equally safe and the POTS axis
    # meaningless. We over-represent a handful of AU-rich hexamers.
    hot = ["AUUUAU", "UAUUUA", "AUUUUA", "UUAUUU", "AAUAAA", "UGUAAA"]
    utrs = []
    for i in range(args.n_utr):
        s = rand_seq(RNG.randint(300, 1500), RNG.uniform(0.32, 0.46))
        for _ in range(RNG.randint(2, 9)):
            h = RNG.choice(hot)
            p = RNG.randrange(0, max(1, len(s) - 6))
            s = s[:p] + h + s[p + 6:]
        utrs.append((f"UTR3_{i:04d}", s))
    write_fasta(out / "utr3.fa", utrs)

    # ---- labelled benchmark ----------------------------------------
    lines = ["sense\tefficacy\tdataset"]
    n_sets = 3
    for i in range(args.n_bench):
        s = rand_seq(19, RNG.uniform(0.30, 0.62))
        lines.append(f"{s}\t{latent_efficacy(s):.4f}\tsynthset{i % n_sets}")
    (out / "benchmark.tsv").write_text("\n".join(lines) + "\n")

    # held-out set from a different GC regime -> tests generalisation
    lines = ["sense\tefficacy\tdataset"]
    for i in range(300):
        s = rand_seq(19, RNG.uniform(0.34, 0.58))
        lines.append(f"{s}\t{latent_efficacy(s):.4f}\theldout")
    (out / "benchmark_heldout.tsv").write_text("\n".join(lines) + "\n")

    print(f"wrote {out}/: targets.fa transcriptome.fa utr3.fa "
          f"benchmark.tsv benchmark_heldout.tsv")


if __name__ == "__main__":
    main()
