"""Sequence primitives and candidate enumeration.

Conventions used consistently across siforge
--------------------------------------------
* All internal sequences are RNA, uppercase, over {A,C,G,U}.
* ``site`` is the *target mRNA* subsequence (sense orientation, 5'->3').
* ``sense`` (passenger strand) == first 19 nt of ``site``.
* ``guide`` (antisense strand) == reverse complement of ``sense``, 5'->3'.
* Guide position numbering is 1-based from the guide 5' end, so the seed
  region (positions 2-8) is ``guide[1:8]`` and the hexamer seed
  (positions 2-7) is ``guide[1:7]``.
* Sense position numbering is 1-based from the sense 5' end, matching the
  numbering used in Reynolds et al. 2004 and Ui-Tei et al. 2004.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator, Sequence

RNA_ALPHABET = frozenset("ACGU")
_COMPLEMENT = str.maketrans("ACGUT", "UGCAA")
_DNA_TO_RNA = str.maketrans("Tt", "Uu")


def to_rna(seq: str) -> str:
    """Uppercase, DNA->RNA, strip whitespace. Non-ACGU become 'N'."""
    s = re.sub(r"\s+", "", seq).upper().translate(_DNA_TO_RNA)
    return "".join(c if c in RNA_ALPHABET else "N" for c in s)


def revcomp(seq: str) -> str:
    """Reverse complement of an RNA sequence (N maps to N)."""
    return seq.translate(_COMPLEMENT)[::-1].replace("A", "A")


def _rc(seq: str) -> str:
    out = []
    for c in reversed(seq):
        out.append({"A": "U", "U": "A", "G": "C", "C": "G"}.get(c, "N"))
    return "".join(out)


revcomp = _rc  # explicit, N-safe


def gc_fraction(seq: str) -> float:
    if not seq:
        return 0.0
    return (seq.count("G") + seq.count("C")) / len(seq)


def au_count(seq: str) -> int:
    return seq.count("A") + seq.count("U")


def longest_run(seq: str) -> int:
    """Length of the longest run of identical nucleotides."""
    best = run = 0
    prev = ""
    for c in seq:
        run = run + 1 if c == prev else 1
        prev = c
        best = max(best, run)
    return best


def longest_gc_stretch(seq: str) -> int:
    """Length of the longest consecutive G/C stretch (Ui-Tei criterion d)."""
    best = run = 0
    for c in seq:
        run = run + 1 if c in "GC" else 0
        best = max(best, run)
    return best


def has_g_quadruplex(seq: str) -> bool:
    """Crude G-quadruplex motif detector: 4x GG+ separated by 1-7 nt loops."""
    return bool(re.search(r"G{2,}\w{1,7}G{2,}\w{1,7}G{2,}\w{1,7}G{2,}", seq))


# Immunostimulatory motifs associated with TLR7/8 recognition of ssRNA.
# Treated as a conservative screen, not an exhaustive list.
IMMUNOSTIM_MOTIFS = ("UGUGU", "GUCCUUCAA", "UGGC", "GUUGUGU")


def immunostim_hits(seq: str) -> list[str]:
    return [m for m in IMMUNOSTIM_MOTIFS if m in seq]


@dataclass(slots=True)
class Candidate:
    """One siRNA candidate derived from a target site."""

    candidate_id: str
    transcript_id: str
    start: int  # 0-based offset of ``site`` within the transcript
    site: str  # target mRNA subsequence, length ``duplex_len + 2``
    duplex_len: int = 19

    # Populated by engines / pipeline stages
    features: dict[str, float] = field(default_factory=dict)
    flags: dict[str, bool] = field(default_factory=dict)
    engine_scores: dict[str, float] = field(default_factory=dict)
    gate_failures: list[str] = field(default_factory=list)
    offtarget_hits: list = field(default_factory=list)

    @property
    def sense(self) -> str:
        """Passenger strand, 5'->3' (the first ``duplex_len`` nt of the site)."""
        return self.site[: self.duplex_len]

    @property
    def guide(self) -> str:
        """Antisense strand, 5'->3'."""
        return revcomp(self.sense)

    @property
    def seed7(self) -> str:
        """Guide positions 2-8."""
        return self.guide[1:8]

    @property
    def seed6(self) -> str:
        """Guide positions 2-7 (the hexamer used for POTS)."""
        return self.guide[1:7]

    @property
    def passenger_seed6(self) -> str:
        return self.sense[1:7]

    @property
    def sense_oligo(self) -> str:
        """Orderable sense strand with dTdT overhang."""
        return self.sense + "dTdT"

    @property
    def guide_oligo(self) -> str:
        """Orderable guide strand with dTdT overhang."""
        return self.guide + "dTdT"

    @property
    def gated(self) -> bool:
        return bool(self.gate_failures)

    @property
    def end(self) -> int:
        return self.start + len(self.site)


def enumerate_candidates(
    transcript_id: str,
    sequence: str,
    duplex_len: int = 19,
    start_offset: int = 0,
    stride: int = 1,
) -> list[Candidate]:
    """Tile a transcript into overlapping candidate target sites.

    ``start_offset`` lets callers skip the region immediately downstream of
    the start codon, where ribosome occupancy and translation-initiation
    complexes reduce RNAi efficacy in practice.
    """
    seq = to_rna(sequence)
    site_len = duplex_len + 2  # include the 2-nt that templates the overhang
    out: list[Candidate] = []
    for i in range(start_offset, len(seq) - site_len + 1, stride):
        site = seq[i : i + site_len]
        if "N" in site:
            continue
        out.append(
            Candidate(
                candidate_id=f"{transcript_id}:{i + 1}",
                transcript_id=transcript_id,
                start=i,
                site=site,
                duplex_len=duplex_len,
            )
        )
    return out


def read_fasta(path: str) -> Iterator[tuple[str, str]]:
    """Minimal FASTA reader yielding (id, sequence). Handles .gz."""
    opener = open
    if str(path).endswith(".gz"):
        import gzip

        opener = gzip.open  # type: ignore[assignment]
    name: str | None = None
    chunks: list[str] = []
    with opener(path, "rt") as fh:  # type: ignore[operator]
        for line in fh:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks)
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if name is not None:
        yield name, "".join(chunks)


def write_fasta(path: str, records: Sequence[tuple[str, str]], width: int = 70) -> None:
    with open(path, "w") as fh:
        for name, seq in records:
            fh.write(f">{name}\n")
            for i in range(0, len(seq), width):
                fh.write(seq[i : i + width] + "\n")
