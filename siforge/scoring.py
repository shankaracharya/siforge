"""The Unified Silencing Score.

Stages, in order:

  0. gates          -- hard elimination (gates.py)
  1. normalize      -- local + global rank percentiles
  2. calibrate      -- isotonic map from raw score to fractional knockdown
  3. fuse           -- correlation-aware weighted mean blended with RRA
  4. desirability   -- Derringer-Suich per objective, geometric mean
  5. uncertainty    -- split-conformal interval, consensus index, LCB
  6. portfolio      -- diversity-constrained greedy selection

Every intermediate is retained on the result object so the report can show
exactly why a candidate ranked where it did.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.isotonic import IsotonicRegression

from .offtarget import offtarget_jaccard
from .seq import Candidate

EPS = 1e-9


# ==========================================================================
# Stage 1 -- normalization
# ==========================================================================
def rank_percentile(values: np.ndarray) -> np.ndarray:
    """(rank - 0.5) / n, NaN-safe. Higher input -> higher percentile."""
    out = np.full(values.shape, np.nan)
    ok = ~np.isnan(values)
    n = int(ok.sum())
    if n == 0:
        return out
    ranks = stats.rankdata(values[ok], method="average")
    out[ok] = (ranks - 0.5) / n
    return out


@dataclass(slots=True)
class ReferenceDistribution:
    """Empirical score distribution per engine, for global percentiles."""

    quantiles: dict[str, list[float]] = field(default_factory=dict)

    def add(self, engine: str, values: np.ndarray, n_q: int = 201) -> None:
        v = values[~np.isnan(values)]
        if v.size == 0:
            return
        qs = np.linspace(0, 1, n_q)
        self.quantiles[engine] = np.quantile(v, qs).tolist()

    def percentile(self, engine: str, values: np.ndarray) -> np.ndarray:
        q = self.quantiles.get(engine)
        if not q:
            return np.full(values.shape, np.nan)
        grid = np.asarray(q)
        out = np.searchsorted(grid, values, side="right") / len(grid)
        return np.clip(out, 0.0, 1.0)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.quantiles))

    @classmethod
    def load(cls, path: str | Path) -> "ReferenceDistribution":
        return cls(quantiles=json.loads(Path(path).read_text()))


# ==========================================================================
# Stage 2 -- calibration
# ==========================================================================
@dataclass
class Calibration:
    """Per-engine isotonic map to fractional knockdown, plus UQ artefacts."""

    engines: list[str] = field(default_factory=list)
    _models: dict[str, IsotonicRegression] = field(default_factory=dict, repr=False)
    residual_cov: np.ndarray | None = None
    conformal_q: dict[str, float] = field(default_factory=dict)
    n_calib: int = 0
    fitted: bool = False
    lambda_blend: float = 0.6

    # ---- fitting -------------------------------------------------------
    def fit(
        self,
        raw: dict[str, np.ndarray],
        y: np.ndarray,
        groups: np.ndarray | None = None,
        alpha: float = 0.10,
        lambda_blend: float = 0.6,
    ) -> "Calibration":
        """Fit isotonic curves, residual covariance and conformal quantiles.

        ``raw[engine]`` is the engine's raw score for each benchmark item,
        ``y`` the observed fractional knockdown in [0, 1]. ``groups``
        (source dataset or gene) drives grouped cross-fitting so the
        calibration is never fitted and evaluated on the same cluster.
        """
        self.engines = sorted(raw)
        n = len(y)
        if groups is None:
            groups = np.zeros(n, dtype=int)
        uniq = np.unique(groups)

        # Out-of-fold calibrated predictions, used for cov + conformal.
        oof = {e: np.full(n, np.nan) for e in self.engines}
        for g in uniq:
            te = groups == g
            tr = ~te
            if tr.sum() < 10 or te.sum() == 0:
                continue
            for e in self.engines:
                x = raw[e]
                m = IsotonicRegression(
                    y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
                )
                ok = tr & ~np.isnan(x)
                if ok.sum() < 10:
                    continue
                m.fit(x[ok], y[ok])
                oof[e][te] = m.predict(np.nan_to_num(x[te], nan=np.nanmedian(x[ok])))

        # Final full-data models for deployment.
        for e in self.engines:
            x = raw[e]
            ok = ~np.isnan(x)
            m = IsotonicRegression(
                y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
            )
            if ok.sum() >= 10:
                m.fit(x[ok], y[ok])
                self._models[e] = m

        # Residual covariance across engines (shrunk).
        R = np.vstack([oof[e] - y for e in self.engines])
        mask = ~np.isnan(R).any(axis=0)
        if mask.sum() >= max(10, len(self.engines) + 2):
            cov = np.cov(R[:, mask])
            cov = np.atleast_2d(cov)
            self.residual_cov = _ledoit_wolf_shrink(cov)
        self.n_calib = int(mask.sum())

        # Split-conformal quantiles, Mondrian by consensus bucket.
        self.lambda_blend = lambda_blend
        self._fit_conformal(oof, y, alpha, lambda_blend)
        self.fitted = True
        return self

    def _fit_conformal(
        self,
        oof: dict[str, np.ndarray],
        y: np.ndarray,
        alpha: float,
        lambda_blend: float = 0.6,
    ) -> None:
        """Calibrate residuals against the *same fused quantity* the scorer
        ranks on. Fitting them against a plain mean while deploying a
        lambda-blended min-variance fusion silently breaks coverage.
        """
        M = np.vstack([oof[e] for e in self.engines])
        mask = ~np.isnan(M).any(axis=0)
        if mask.sum() < 20:
            self.conformal_q = {"global": 0.25}
            return
        Mm = M[:, mask]
        yy = y[mask]
        w = (
            min_variance_weights(self.residual_cov)
            if self.residual_cov is not None
            else np.full(len(self.engines), 1.0 / len(self.engines))
        )
        p_mean = w @ Mm
        ranks = np.vstack([1.0 - rank_percentile(Mm[i]) for i in range(Mm.shape[0])])
        p_rank = rra_score(ranks)
        point = lambda_blend * p_mean + (1 - lambda_blend) * p_rank
        spread = Mm.std(axis=0)
        resid = np.abs(yy - point)
        # Mondrian buckets on inter-engine disagreement.
        edges = np.quantile(spread, [0.33, 0.66])
        buckets = np.digitize(spread, edges)
        for b in (0, 1, 2):
            sel = buckets == b
            if sel.sum() >= 10:
                self.conformal_q[f"spread{b}"] = float(
                    _conformal_quantile(resid[sel], alpha)
                )
        self.conformal_q["global"] = float(_conformal_quantile(resid, alpha))

    # ---- application ---------------------------------------------------
    def predict(self, engine: str, x: np.ndarray) -> np.ndarray:
        m = self._models.get(engine)
        if m is None:
            return np.full(np.shape(x), np.nan)
        out = np.full(np.shape(x), np.nan)
        ok = ~np.isnan(x)
        if ok.any():
            out[ok] = m.predict(x[ok])
        return out

    def q_for(self, spread: float) -> float:
        if not self.conformal_q:
            return 0.25
        if spread <= 0.05 and "spread0" in self.conformal_q:
            return self.conformal_q["spread0"]
        if spread <= 0.12 and "spread1" in self.conformal_q:
            return self.conformal_q["spread1"]
        return self.conformal_q.get("spread2", self.conformal_q["global"])

    # ---- persistence ---------------------------------------------------
    def save(self, path: str | Path) -> None:
        payload = {
            "engines": self.engines,
            "n_calib": self.n_calib,
            "lambda_blend": self.lambda_blend,
            "conformal_q": self.conformal_q,
            "residual_cov": None
            if self.residual_cov is None
            else self.residual_cov.tolist(),
            "models": {
                e: {
                    "x": m.X_thresholds_.tolist(),
                    "y": m.y_thresholds_.tolist(),
                }
                for e, m in self._models.items()
            },
        }
        Path(path).write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        d = json.loads(Path(path).read_text())
        c = cls(engines=d["engines"], n_calib=d.get("n_calib", 0),
                lambda_blend=d.get("lambda_blend", 0.6))
        c.conformal_q = d.get("conformal_q", {})
        rc = d.get("residual_cov")
        c.residual_cov = np.asarray(rc) if rc else None
        for e, m in d.get("models", {}).items():
            iso = IsotonicRegression(
                y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
            )
            x = np.asarray(m["x"])
            y = np.asarray(m["y"])
            iso.fit(x, y)
            c._models[e] = iso
        c.fitted = bool(c._models)
        return c


def _conformal_quantile(resid: np.ndarray, alpha: float) -> float:
    n = len(resid)
    k = math.ceil((n + 1) * (1 - alpha))
    k = min(k, n)
    return float(np.sort(resid)[k - 1])


def _ledoit_wolf_shrink(cov: np.ndarray, intensity: float | None = None) -> np.ndarray:
    p = cov.shape[0]
    mu = np.trace(cov) / p
    target = mu * np.eye(p)
    if intensity is None:
        off = cov - np.diag(np.diag(cov))
        denom = np.sum(cov**2) + EPS
        intensity = float(np.clip(np.sum(off**2) / denom, 0.05, 0.6))
    return (1 - intensity) * cov + intensity * target


# ==========================================================================
# Stage 3 -- fusion
# ==========================================================================
def min_variance_weights(cov: np.ndarray) -> np.ndarray:
    """Minimum-variance weights, projected onto the simplex (w >= 0, sum 1).

    This is the portfolio-theory weight vector, and the analogy is exact:
    engines whose residuals are correlated get collectively down-weighted
    toward the information content of a single engine.
    """
    p = cov.shape[0]
    try:
        inv = np.linalg.pinv(cov + EPS * np.eye(p))
    except np.linalg.LinAlgError:
        return np.full(p, 1.0 / p)
    ones = np.ones(p)
    w = inv @ ones
    denom = ones @ inv @ ones
    if not np.isfinite(denom) or abs(denom) < EPS:
        return np.full(p, 1.0 / p)
    w = w / denom
    w = np.clip(w, 0.0, None)
    s = w.sum()
    return w / s if s > EPS else np.full(p, 1.0 / p)


def family_weights(engines: list[str], families: dict[str, str]) -> np.ndarray:
    """Fallback weighting when no residual covariance is available.

    Each *family* gets an equal share, split among its members. This
    prevents five rule engines from out-voting two independent ML models
    purely by count.
    """
    fams: dict[str, list[int]] = {}
    for i, e in enumerate(engines):
        fams.setdefault(families.get(e, e), []).append(i)
    w = np.zeros(len(engines))
    share = 1.0 / max(1, len(fams))
    for members in fams.values():
        for i in members:
            w[i] = share / len(members)
    return w


def rra_score(ranks: np.ndarray) -> np.ndarray:
    """Robust rank aggregation (Stuart/Kolde beta order-statistic rho).

    ``ranks`` is (n_engines, n_candidates) of normalised ranks in (0,1],
    where smaller = better. Returns 1 - rho, so higher = better.
    """
    n_eng, n_cand = ranks.shape
    out = np.ones(n_cand)
    for j in range(n_cand):
        r = np.sort(ranks[~np.isnan(ranks[:, j]), j])
        k = len(r)
        if k == 0:
            out[j] = 0.0
            continue
        # p-value of the i-th order statistic under Beta(i, k-i+1)
        ps = [stats.beta.cdf(r[i], i + 1, k - i) for i in range(k)]
        rho = min(1.0, min(ps) * k)  # Bonferroni-style correction
        out[j] = 1.0 - rho
    return out


# ==========================================================================
# Stage 4 -- desirability
# ==========================================================================
def derringer(x: float | np.ndarray, low: float, target: float, r: float = 1.0):
    """One-sided larger-is-better Derringer-Suich desirability in [0, 1]."""
    x = np.asarray(x, dtype=float)
    d = np.zeros_like(x)
    span = target - low
    if abs(span) < EPS:
        return np.where(x >= target, 1.0, 0.0)
    mid = (x > low) & (x < target)
    d = np.where(x >= target, 1.0, d)
    d = np.where(mid, np.power(np.clip((x - low) / span, 0, 1), r), d)
    return np.clip(d, 0.0, 1.0)


def geometric_mean(d: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Weighted geometric mean. Any zero objective zeroes the composite."""
    keys = [k for k in d if weights.get(k, 0.0) > 0]
    if not keys:
        raise ValueError("no weighted objectives")
    wsum = sum(weights[k] for k in keys)
    acc = np.zeros_like(np.asarray(d[keys[0]], dtype=float))
    zero = np.zeros_like(acc, dtype=bool)
    for k in keys:
        v = np.asarray(d[k], dtype=float)
        zero |= v <= EPS
        acc += weights[k] * np.log(np.clip(v, EPS, 1.0))
    out = np.exp(acc / wsum)
    return np.where(zero, 0.0, out)


# ==========================================================================
# Result container
# ==========================================================================
@dataclass
class ScoredCandidate:
    candidate: Candidate
    potency: float
    potency_mean: float
    potency_rra: float
    desirability: dict[str, float]
    uss: float
    uss_lcb: float
    pi_low: float
    pi_high: float
    per_engine_kd: dict[str, float]
    weights: dict[str, float]
    n_engines: int
    calibrated: bool

    @property
    def candidate_id(self) -> str:
        return self.candidate.candidate_id


# ==========================================================================
# The orchestrator
# ==========================================================================
class UnifiedScorer:
    def __init__(
        self,
        profile: dict,
        calibration: Calibration | None = None,
        reference: ReferenceDistribution | None = None,
        families: dict[str, str] | None = None,
    ) -> None:
        self.profile = profile
        self.calib = calibration
        self.reference = reference
        self.families = families or {}

    # -- stage 1+2 -------------------------------------------------------
    def _calibrated_matrix(
        self, engines: list[str], raw: np.ndarray
    ) -> tuple[np.ndarray, bool]:
        """Map raw engine scores to expected fractional knockdown.

        Order of preference:
          1. fitted isotonic calibration (true units)
          2. global percentile against the reference distribution
          3. local rank percentile (clearly marked uncalibrated)
        """
        out = np.full(raw.shape, np.nan)
        calibrated = False
        for i, e in enumerate(engines):
            col = raw[i]
            if self.calib is not None and e in self.calib._models:
                out[i] = self.calib.predict(e, col)
                calibrated = True
            elif self.reference is not None and e in self.reference.quantiles:
                out[i] = self.reference.percentile(e, col)
            else:
                out[i] = rank_percentile(col)
        return out, calibrated

    # -- stage 3 ---------------------------------------------------------
    def _fuse(
        self, engines: list[str], kd: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_eng = len(engines)
        # weights
        if (
            self.calib is not None
            and self.calib.residual_cov is not None
            and list(self.calib.engines) == engines
        ):
            w = min_variance_weights(self.calib.residual_cov)
        elif self.calib is not None and self.calib.residual_cov is not None:
            idx = [self.calib.engines.index(e) for e in engines
                   if e in self.calib.engines]
            if len(idx) == n_eng:
                sub = self.calib.residual_cov[np.ix_(idx, idx)]
                w = min_variance_weights(sub)
            else:
                w = family_weights(engines, self.families)
        else:
            w = family_weights(engines, self.families)

        # per-candidate renormalisation over engines that actually reported
        present = ~np.isnan(kd)
        wmat = np.where(present, w[:, None], 0.0)
        wsum = wmat.sum(axis=0)
        wsum[wsum < EPS] = np.nan
        wmat = wmat / wsum
        p_mean = np.nansum(wmat * np.nan_to_num(kd), axis=0)
        p_mean = np.where(np.isnan(wsum), np.nan, p_mean)

        # rank aggregation
        ranks = np.full(kd.shape, np.nan)
        for i in range(n_eng):
            pct = rank_percentile(kd[i])
            ranks[i] = 1.0 - pct  # smaller = better
        p_rank = rra_score(ranks)

        spread = np.nanstd(kd, axis=0)
        n_present = present.sum(axis=0)
        return p_mean, p_rank, spread, n_present

    # -- stage 4+5+6 -----------------------------------------------------
    def score(
        self,
        candidates: list[Candidate],
        engine_scores: dict[str, dict[str, float]],
    ) -> list[ScoredCandidate]:
        passing = [c for c in candidates if not c.gated]
        if not passing:
            return []
        ids = [c.candidate_id for c in passing]
        engines = [e for e in engine_scores if any(i in engine_scores[e] for i in ids)]
        if not engines:
            raise ValueError("no engine produced scores for any passing candidate")

        raw = np.array(
            [[engine_scores[e].get(i, np.nan) for i in ids] for e in engines],
            dtype=float,
        )
        kd, calibrated = self._calibrated_matrix(engines, raw)
        p_mean, p_rank, spread, n_present = self._fuse(engines, kd)

        lam = float(self.profile.get("lambda_blend", 0.6))
        potency = lam * np.nan_to_num(p_mean, nan=0.0) + (1 - lam) * p_rank

        # ---- objectives ------------------------------------------------
        obj = self.profile["objectives"]
        feats = {
            "potency": potency,
            "specificity": self._specificity(passing),
            "accessibility": np.array(
                [c.features.get("accessibility", np.nan) for c in passing]
            ),
            "coverage": np.array(
                [c.features.get("isoform_coverage", 1.0) for c in passing]
            ),
            "conservation": np.array(
                [c.features.get("conservation", 1.0) for c in passing]
            ),
            "manufacturability": self._manufacturability(passing),
        }

        d: dict[str, np.ndarray] = {}
        weights: dict[str, float] = {}
        for name, spec in obj.items():
            v = feats.get(name)
            if v is None:
                continue
            if spec.get("transform") == "percentile":
                # Some quantities are not comparable on an absolute scale --
                # RNAplfold unpaired probabilities depend on the u-length,
                # the folding window and the transcript's overall structure,
                # so a fixed threshold silently zeroes whole transcripts.
                # Rank within the candidate set first, then apply
                # desirability in percentile units.
                v = rank_percentile(v)
            v = np.where(np.isnan(v), spec.get("missing", 0.5), v)
            d[name] = derringer(v, spec["low"], spec["target"], spec.get("r", 1.0))
            weights[name] = float(spec.get("weight", 0.0))

        uss = geometric_mean(d, weights) * 100.0

        # ---- conformal LCB --------------------------------------------
        q = np.array(
            [
                self.calib.q_for(float(s))
                if self.calib is not None and self.calib.conformal_q
                else float(self.profile.get("default_conformal_q", 0.20))
                for s in np.nan_to_num(spread, nan=0.25)
            ]
        )
        # widen when engines are missing
        q = q * (1.0 + 0.15 * np.maximum(0, len(engines) - n_present))
        pot_lo = np.clip(potency - q, 0.0, 1.0)
        d_lo = dict(d)
        spec_pot = obj["potency"]
        d_lo["potency"] = derringer(
            pot_lo, spec_pot["low"], spec_pot["target"], spec_pot.get("r", 1.0)
        )
        uss_lcb = geometric_mean(d_lo, weights) * 100.0

        results = []
        for j, c in enumerate(passing):
            results.append(
                ScoredCandidate(
                    candidate=c,
                    potency=float(potency[j]),
                    potency_mean=float(p_mean[j]) if not np.isnan(p_mean[j]) else float("nan"),
                    potency_rra=float(p_rank[j]),
                    desirability={k: float(v[j]) for k, v in d.items()},
                    uss=float(uss[j]),
                    uss_lcb=float(uss_lcb[j]),
                    pi_low=float(max(0.0, potency[j] - q[j])),
                    pi_high=float(min(1.0, potency[j] + q[j])),
                    per_engine_kd={
                        e: float(kd[i, j])
                        for i, e in enumerate(engines)
                        if not np.isnan(kd[i, j])
                    },
                    weights={},
                    n_engines=int(n_present[j]),
                    calibrated=calibrated,
                )
            )
        results.sort(key=lambda r: r.uss_lcb, reverse=True)
        return results

    # -- objective helpers ----------------------------------------------
    def _specificity(self, cands: list[Candidate]) -> np.ndarray:
        """Sub-composite: POTS, off-target burden, seed-Tm margin.

        Combined geometrically so one bad specificity axis cannot be
        averaged away by two good ones.
        """
        cfg = self.profile.get("specificity", {})
        pots = np.array([c.features.get("pots", np.nan) for c in cands])
        burden = np.array([c.features.get("offtarget_burden", np.nan) for c in cands])
        seed_tm = np.array([c.features.get("seed_tm", np.nan) for c in cands])

        d_pots = derringer(
            -np.nan_to_num(pots, nan=np.nanmedian(pots) if np.isfinite(pots).any() else 2.0),
            -cfg.get("pots_low", 4.0),
            -cfg.get("pots_target", 1.5),
            cfg.get("r", 2.0),
        )
        d_burden = derringer(
            -np.nan_to_num(burden, nan=0.5),
            -cfg.get("burden_low", 2.0),
            -cfg.get("burden_target", 0.3),
            cfg.get("r", 2.0),
        )
        tm_margin = cfg.get("seed_tm_max", 21.5) - np.nan_to_num(seed_tm, nan=21.5)
        d_tm = derringer(tm_margin, cfg.get("tm_margin_low", -5.0),
                         cfg.get("tm_margin_target", 10.0), 1.0)
        comp = geometric_mean(
            {"pots": d_pots, "burden": d_burden, "tm": d_tm},
            {"pots": 0.45, "burden": 0.35, "tm": 0.20},
        )
        for j, c in enumerate(cands):
            c.features["spec_d_pots"] = float(d_pots[j])
            c.features["spec_d_burden"] = float(d_burden[j])
            c.features["spec_d_tm"] = float(d_tm[j])
            c.features["specificity"] = float(comp[j])
        return comp

    def _manufacturability(self, cands: list[Candidate]) -> np.ndarray:
        """GC centrality plus absence of soft synthesis liabilities."""
        out = []
        for c in cands:
            gc = c.features.get("gc_fraction", 0.5)
            centrality = 1.0 - abs(gc - 0.45) / 0.45
            run_pen = 0.0 if c.features.get("longest_run", 0) <= 3 else 0.4
            out.append(max(0.0, min(1.0, centrality - run_pen)))
        return np.array(out)


# ==========================================================================
# Stage 6 -- portfolio selection
# ==========================================================================
def select_portfolio(
    scored: list[ScoredCandidate],
    k: int = 4,
    min_distance_nt: int = 50,
    min_seed_hamming: int = 3,
    max_offtarget_jaccard: float = 0.2,
) -> list[ScoredCandidate]:
    """Greedy diversity-constrained selection ranked by the LCB.

    The point is that the picks should not share a failure mode: if the
    accessibility model is wrong about one region, you lose one pick rather
    than the whole order.
    """
    chosen: list[ScoredCandidate] = []
    for cand in sorted(scored, key=lambda r: r.uss_lcb, reverse=True):
        if len(chosen) >= k:
            break
        ok = True
        for sel in chosen:
            a, b = cand.candidate, sel.candidate
            if a.transcript_id == b.transcript_id and abs(a.start - b.start) < min_distance_nt:
                ok = False
                break
            if _hamming(a.seed7, b.seed7) < min_seed_hamming:
                ok = False
                break
            if offtarget_jaccard(a, b) > max_offtarget_jaccard:
                ok = False
                break
        if ok:
            chosen.append(cand)
    return chosen


def _hamming(a: str, b: str) -> int:
    return sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))


def consensus_index(engine_scores: dict[str, dict[str, float]], ids: list[str],
                    top_n: int = 50) -> float:
    """Kendall's W across engine rankings of the top candidates."""
    mat = []
    for e, d in engine_scores.items():
        v = np.array([d.get(i, np.nan) for i in ids])
        if np.isnan(v).all():
            continue
        mat.append(rank_percentile(v))
    if len(mat) < 2:
        return float("nan")
    M = np.vstack(mat)
    keep = ~np.isnan(M).any(axis=0)
    M = M[:, keep]
    if M.shape[1] < 3:
        return float("nan")
    order = np.argsort(-M.mean(axis=0))[:top_n]
    M = M[:, order]
    m, n = M.shape
    ranks = np.vstack([stats.rankdata(-row) for row in M])
    Rj = ranks.sum(axis=0)
    S = ((Rj - Rj.mean()) ** 2).sum()
    denom = m**2 * (n**3 - n) / 12.0
    return float(S / denom) if denom > 0 else float("nan")
