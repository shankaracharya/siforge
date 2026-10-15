# Findings from building and validating the v1.0 stack

Recorded because each one changed the implementation, and each is the kind
of thing a reviewer will ask about.

## 1. Conformal quantiles must be fitted on the deployed fused quantity

Initially the split-conformal residuals were computed against an unweighted
mean of calibrated engine predictions, while the scorer ranked on the
λ-blended minimum-variance fusion. Coverage came out at **0.832 / 0.800**
against a nominal 0.900 — a silent, systematic undercoverage.

Fitting the residuals against the same fused quantity the scorer deploys
restored coverage to **0.919 / 0.900**.

Generalisable lesson: conformal prediction guarantees coverage *for the
predictor whose residuals you calibrated*. Any mismatch between the
calibration-time and deployment-time predictor voids the guarantee without
raising an error.

## 2. Absolute accessibility thresholds are not portable

RNAplfold unpaired probabilities for u=8 on the demo targets are heavily
right-skewed (median 0.077, 25th percentile 0.029). A fixed desirability
floor of `low: 0.05` therefore zeroed roughly 40% of candidates, and because
the composite is a geometric mean, a zero on one axis sets USS to 0 — so a
large fraction of perfectly reasonable candidates were being silently vetoed
by a threshold chosen for the wrong scale.

Unpaired probabilities are not comparable across transcripts, u-lengths or
folding windows, so the objective now supports `transform: percentile` and
both shipped profiles use it for accessibility. In the potency-matched
pairwise test this moved the specificity-ordering statistic from 2.8% to
53.8%.

Anything whose scale depends on the input should be ranked before it is
thresholded.

## 3. Seed rarity and seed Tm are confounded

Rare 3'UTR hexamers are GC-rich; GC-rich seeds have high seed-target duplex
Tm; high seed Tm is penalised independently (the siDirect 2.0 criterion).
A naive "rare seed vs common seed" comparison therefore measures Tm, not
promiscuity.

The first two attempts at the E3 experiment failed for exactly this reason,
and a third failed because the constructed probes could not pass the Tm gate
at all — i.e. they were candidates a real run would already have eliminated.
Controlled comparisons must match on GC content *and* use probes that
actually survive the gates.

## 4. The geometric form does not simply mean "specificity wins"

The ceteris-paribus sweep is unambiguous: holding one real candidate fixed
and varying only POTS, USS falls monotonically (76.5 → 53.2) and hits 0.00
at the `pots_low` threshold. The veto property works exactly as designed.

But in the observational potency-matched pairwise test, a weighted
*arithmetic* mean tracks the isolated specificity axis more faithfully (98%)
than the geometric mean (54%). This is not a bug, and it is worth
understanding: the geometric form couples all axes multiplicatively, so a
pair that differs in specificity *and* slightly in manufacturability or
coverage can reorder on the latter. Arithmetic means isolate a single axis
better; geometric means express joint quality and enforce vetoes.

We keep the geometric form because the veto property is the design
requirement — a potent, promiscuous guide must not be orderable — and we
report the pairwise number rather than suppressing it.

## 5. The RRA blend costs in-distribution accuracy here

λ sweep on the synthetic benchmark:

| λ | ρ (all engines) | ρ (2 engines dropped) |
|---|---|---|
| 0.0 | +0.696 | +0.598 |
| 0.6 | +0.757 | +0.699 |
| 1.0 | **+0.778** | **+0.718** |

Pure weighted mean wins in both columns. The rank-aggregation term was
included for robustness when engines are missing or badly miscalibrated, and
the synthetic benchmark has neither pathology — all six engines are
deterministic rule models with well-behaved isotonic fits.

λ stays at 0.6 by default only because real deployments do have those
pathologies. **Re-tune λ on your own benchmark**; do not inherit this
default uncritically. The sweep is printed by `examples/run_validation.py`
specifically so this decision is made with evidence.

## 6. Gate attrition is dominated by the seed-Tm criterion

On the demo targets, of 2,810 tiled candidates 2,348 were gated:

| reason | count |
|---|---|
| seed Tm ≥ 21.5 °C | 2,158 |
| GC out of range | 561 |
| homopolymer run | 454 |
| near-identical off-target | 51 |
| G-quadruplex | 14 |

The siDirect 2.0 seed-Tm criterion is by far the most aggressive filter.
This is expected behaviour, not a misconfiguration, but it means loosening
`seed_tm_max` is the first lever to reach for on a target where nothing
survives — and that doing so trades specificity for yield explicitly rather
than silently.

The 51 near-identical off-target rejections correspond exactly to the 60-nt
block deliberately shared between `DEMO_TARGET_A` and `DEMO_PARALOG`, which
is the planted positive control for the full-length scanner.

## 7. Inter-engine consensus is low, and should be reported

Kendall's W across engines was 0.126 and 0.211 on the two demo targets. Rule
engines genuinely disagree about which sites are good. This is surfaced in
the report rather than hidden, because a low W means "the ensemble has little
idea about this target" and should change how much weight a user puts on the
ranking.
