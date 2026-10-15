"""Target-site accessibility.

Uses ViennaRNA's local-folding partition function (``RNA.pfl_fold_up``,
the library equivalent of RNAplfold ``-u``) when the bindings are
installed, averaged over several window/span settings because single
parameterisations are not robust. Falls back to a nearest-neighbour
self-structure proxy when ViennaRNA is unavailable, which is clearly
marked in the output so results are never silently degraded.
"""

from __future__ import annotations

import warnings
from functools import lru_cache

from .seq import Candidate
from .thermo import duplex_dg

try:  # pragma: no cover - import guard
    import RNA  # type: ignore

    HAVE_VIENNA = True
except Exception:  # pragma: no cover
    RNA = None  # type: ignore
    HAVE_VIENNA = False

# (window, span) pairs -- averaged to avoid over-fitting one setting
PLFOLD_SETTINGS = ((80, 40), (120, 60), (160, 80))


@lru_cache(maxsize=64)
def _unpaired_profile(seq: str, u: int, window: int, span: int) -> tuple[float, ...]:
    """Probability that each length-``u`` stretch ending at i is unpaired."""
    probs = RNA.pfl_fold_up(seq, u, window, span)
    out = []
    for i in range(len(seq) + 1):
        val = float("nan")
        if i < len(probs):
            row = probs[i]
            if row is not None and len(row) > u:
                v = float(row[u])
                if v == v and v >= 0.0:
                    val = v
        out.append(val)
    return tuple(out)


def vienna_accessibility(transcript: str, candidates: list[Candidate], u: int = 8) -> None:
    """Mean unpaired probability of a ``u``-nt stretch at the 3' end of the site.

    The 3' end of the target site corresponds to the guide's seed-facing
    region, which is where nucleation of RISC binding occurs, so that is
    the window scored.
    """
    profiles = []
    for window, span in PLFOLD_SETTINGS:
        try:
            profiles.append(_unpaired_profile(transcript, u, window, span))
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"RNAplfold failed for window={window}: {exc}")
    for c in candidates:
        idx = c.start + c.duplex_len  # 1-based position of the stretch end
        vals = []
        for prof in profiles:
            if idx < len(prof):
                v = prof[idx]
                if v == v:
                    vals.append(v)
        c.features["accessibility"] = sum(vals) / len(vals) if vals else float("nan")
        c.features["accessibility_source"] = 1.0  # 1 = ViennaRNA


def fallback_accessibility(transcript: str, candidates: list[Candidate]) -> None:
    """Structure proxy used when ViennaRNA is absent.

    Scores the *instability* of the site's own strongest local self-duplex
    relative to a 60-nt neighbourhood. Monotone in the right direction but
    much coarser than a partition function; flagged as source=0.
    """
    for c in candidates:
        lo = max(0, c.start - 20)
        hi = min(len(transcript), c.end + 20)
        local = transcript[lo:hi]
        site_dg = duplex_dg(c.site, include_init=False)
        local_dg = duplex_dg(local, include_init=False) / max(1, len(local) - 1)
        norm = site_dg / max(1, len(c.site) - 1)
        c.features["accessibility"] = max(0.0, min(1.0, 0.5 + (norm - local_dg) * 0.6))
        c.features["accessibility_source"] = 0.0


def annotate_accessibility(
    transcript: str, candidates: list[Candidate], force_fallback: bool = False
) -> None:
    if HAVE_VIENNA and not force_fallback:
        vienna_accessibility(transcript, candidates)
        if all(c.features.get("accessibility") != c.features.get("accessibility")
               for c in candidates):
            fallback_accessibility(transcript, candidates)
    else:
        if not force_fallback:
            warnings.warn(
                "ViennaRNA not installed -- using the coarse accessibility "
                "fallback. Install with `pip install ViennaRNA` for real "
                "RNAplfold accessibility.",
                RuntimeWarning,
            )
        fallback_accessibility(transcript, candidates)
