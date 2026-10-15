"""Specificity layer: seed promiscuity (POTS) and full-length off-targets.

Two independent axes, both of which can veto a candidate:

1. **Seed-based off-targeting** -- the dominant mechanism. Guide positions
   2-7 pair with 3'UTR sequences and drive miRNA-like repression. The
   magnitude tracks how often that hexamer's complement occurs across the
   3'UTR transcriptome, so siforge builds a hexamer frequency table from a
   user-supplied 3'UTR FASTA and scores each candidate against *their*
   current annotation rather than a frozen precomputed table.

2. **Full-length near-identity** -- classic cross-hybridisation. Scanned
   with an anchor-and-verify index over the transcriptome, or delegated to
   bowtie when it is on PATH and the transcriptome is large.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .seq import Candidate, read_fasta, revcomp, to_rna

SEED_K = 6


@dataclass(slots=True)
class SeedTable:
    """Hexamer occurrence counts across a 3'UTR transcriptome."""

    k: int
    counts: dict[str, int]
    n_utrs: int
    total_kmers: int
    source: str = ""

    @property
    def mean_count(self) -> float:
        return self.total_kmers / max(1, 4**self.k)

    def count(self, kmer: str) -> int:
        return self.counts.get(kmer, 0)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "k": self.k,
                    "n_utrs": self.n_utrs,
                    "total_kmers": self.total_kmers,
                    "source": self.source,
                    "counts": self.counts,
                }
            )
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "SeedTable":
        d = json.loads(Path(path).read_text())
        return cls(
            k=d["k"],
            counts=d["counts"],
            n_utrs=d["n_utrs"],
            total_kmers=d["total_kmers"],
            source=d.get("source", str(path)),
        )


def build_seed_table(utr3_fasta: str | Path, k: int = SEED_K) -> SeedTable:
    """Count every k-mer across a 3'UTR FASTA.

    The relevant quantity for a guide is how many times its seed
    *complement* appears in 3'UTRs, so lookups complement before counting.
    """
    counts: Counter[str] = Counter()
    n = 0
    total = 0
    for _name, seq in read_fasta(str(utr3_fasta)):
        s = to_rna(seq)
        n += 1
        for i in range(len(s) - k + 1):
            kmer = s[i : i + k]
            if "N" in kmer:
                continue
            counts[kmer] += 1
            total += 1
    return SeedTable(
        k=k, counts=dict(counts), n_utrs=n, total_kmers=total, source=str(utr3_fasta)
    )


def pots_score(candidate: Candidate, table: SeedTable, passenger_weight: float = 0.3) -> float:
    """Potential off-targeting score. **Lower is better.**

    Counts 3'UTR occurrences of the complement of the guide seed hexamer,
    plus a down-weighted contribution from the passenger seed (passenger
    strand loading is reduced but not abolished without chemical
    modification). Reported on a log10 scale so it behaves linearly.
    """
    guide_target = revcomp(candidate.seed6)
    pass_target = revcomp(candidate.passenger_seed6)
    n_guide = table.count(guide_target)
    n_pass = table.count(pass_target)
    candidate.features["pots_guide_hits"] = float(n_guide)
    candidate.features["pots_passenger_hits"] = float(n_pass)
    score = math.log10(1.0 + n_guide + passenger_weight * n_pass)
    candidate.features["pots"] = score
    return score


# --------------------------------------------------------------------------
# Full-length off-target scanning
# --------------------------------------------------------------------------
ANCHOR = 11


class TranscriptomeIndex:
    """Anchor-and-verify index for mismatch-tolerant guide matching.

    Any alignment with <= ``max_mm`` mismatches over an L-nt guide must
    contain an exact match of length >= floor(L / (max_mm + 1)) by the
    pigeonhole principle. We index exact ``ANCHOR``-mers and verify hits,
    which is exact (not heuristic) for max_mm <= L/ANCHOR - 1.
    """

    def __init__(self, anchor: int = ANCHOR) -> None:
        self.anchor = anchor
        self.index: dict[str, list[tuple[int, int]]] = {}
        self.names: list[str] = []
        self.seqs: list[str] = []

    @classmethod
    def from_fasta(cls, path: str | Path, anchor: int = ANCHOR) -> "TranscriptomeIndex":
        idx = cls(anchor)
        for name, seq in read_fasta(str(path)):
            s = to_rna(seq)
            tid = len(idx.seqs)
            idx.names.append(name)
            idx.seqs.append(s)
            for i in range(len(s) - anchor + 1):
                kmer = s[i : i + anchor]
                if "N" in kmer:
                    continue
                idx.index.setdefault(kmer, []).append((tid, i))
        return idx

    def scan(
        self, guide: str, max_mm: int = 3, exclude_prefix: str | None = None
    ) -> list[tuple[str, int, int]]:
        """Return (transcript, position, mismatches) for off-target hits.

        The query is the guide's target site, i.e. ``revcomp(guide)``, since
        we are searching sense-strand transcripts.
        """
        query = revcomp(guide)
        L = len(query)
        seen: set[tuple[int, int]] = set()
        hits: list[tuple[str, int, int]] = []
        n_anchor = L // (max_mm + 1)
        if n_anchor < self.anchor:
            # Pigeonhole guarantee lost; fall back to all anchor offsets.
            offsets = range(0, L - self.anchor + 1)
        else:
            offsets = range(0, L - self.anchor + 1, max(1, self.anchor))
        for off in offsets:
            kmer = query[off : off + self.anchor]
            for tid, pos in self.index.get(kmer, ()):
                start = pos - off
                if start < 0 or start + L > len(self.seqs[tid]):
                    continue
                key = (tid, start)
                if key in seen:
                    continue
                seen.add(key)
                name = self.names[tid]
                if exclude_prefix and name.startswith(exclude_prefix):
                    continue
                sub = self.seqs[tid][start : start + L]
                mm = sum(1 for a, b in zip(sub, query) if a != b)
                if mm <= max_mm:
                    hits.append((name, start, mm))
        return hits


def annotate_offtargets(
    candidates: list[Candidate],
    index: TranscriptomeIndex | None,
    seed_table: SeedTable | None,
    on_target_prefix: str | None = None,
    max_mm: int = 3,
) -> None:
    """Populate POTS and full-length off-target features in place."""
    for c in candidates:
        if seed_table is not None:
            pots_score(c, seed_table)
        else:
            c.features["pots"] = float("nan")
        if index is not None:
            hits = index.scan(c.guide, max_mm=max_mm, exclude_prefix=on_target_prefix)
            c.features["n_offtarget_hits"] = float(len(hits))
            c.features["min_offtarget_mismatches"] = (
                float(min(h[2] for h in hits)) if hits else float(max_mm + 1)
            )
            # Energy-flavoured burden: closer matches weigh exponentially more.
            burden = sum(math.exp(-(h[2])) for h in hits)
            c.features["offtarget_burden"] = math.log10(1.0 + burden)
            c.flags["has_near_identical_offtarget"] = any(h[2] <= 2 for h in hits)
            c.offtarget_hits = hits
        else:
            c.features["n_offtarget_hits"] = float("nan")
            c.features["offtarget_burden"] = float("nan")


def offtarget_jaccard(a: Candidate, b: Candidate) -> float:
    """Overlap of predicted off-target transcript sets (for portfolio spacing)."""
    sa = {h[0] for h in getattr(a, "offtarget_hits", [])}
    sb = {h[0] for h in getattr(b, "offtarget_hits", [])}
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))
