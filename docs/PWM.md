# Supplying real i-Score / DSIR coefficients

The `pwm` engine is a position-weight linear model. siforge ships a
**surrogate** coefficient table (`siforge/data/pwm_surrogate.tsv`) assembled
from widely reported consensus base preferences so the engine is functional
out of the box. It is NOT the published i-Score (Ichihara et al. 2007, NAR
35:e123) or DSIR (Vert et al. 2006, BMC Bioinformatics 7:520) table, and
results from it must never be reported under those names.

## Format

Tab-separated, 1-based positions on the **sense** strand, higher = better:

```
# comments allowed
1	A	-0.35
1	C	0.50
1	G	0.60
1	U	-0.45
2	A	...
```

Scores are mean-centred per position at run time, so any intercept
convention works.

## Using it

```bash
siforge design targets.fa --pwm resources/iscore_coefficients.tsv ...
```

Or replace `siforge/data/pwm_surrogate.tsv` in your deployment.

## Validating a reimplementation

If you reimplement a published model rather than obtaining its coefficients,
validate it against a cached set of outputs from the original tool and report
the concordance. The DSIR server at <http://biodev.cea.fr/DSIR/> was live as
of October 2026 and accepts a FASTA; 200 candidates is enough to establish
agreement.

Keep that oracle run **one-off and cached**, never a runtime dependency:

- respect `robots.txt` — siSPOTR, for one, disallows automated access
- rate-limit to a few requests per minute
- commit the cached responses with their capture date
- assert concordance (Spearman ρ > 0.98, top-20 rank agreement) in CI

That gives you the one thing scraping is genuinely good for — proving your
reimplementation is faithful — without making your pipeline depend on someone
else's uptime.
