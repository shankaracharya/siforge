# siforge

Multi-engine siRNA design with a calibrated **Unified Silencing Score (USS)**.

Tiles a target region, scores every candidate through rule-based,
thermodynamic, structural and (optionally) deep-learning engines, applies a
hard specificity gate, and collapses the survivors into one interpretable
score with a conformal prediction interval. Returns a diversity-constrained
portfolio you can order, not a top-k slice.

Everything in the default stack is **install-only** — no web scraping, no
server dependencies, no network calls at run time.

---

## Install

```bash
git clone <your-repo> siforge && cd siforge
python -m venv .venv && source .venv/bin/activate
pip install -e ".[structure,parquet,dev]"
siforge --version
```

`structure` pulls in ViennaRNA for real RNAplfold accessibility. Without it
siforge still runs, but falls back to a coarse structure proxy and says so in
every report. Verify:

```bash
siforge engines
```

Conda alternative (also gets bowtie2/BLAST for large transcriptomes):

```bash
conda env create -f env/environment.yml && conda activate siforge
```

Docker:

```bash
docker build -t siforge:1.0.0 .
docker run --rm -v "$PWD:/work" siforge:1.0.0 design /work/targets.fa -o /work/out
```

---

## Quick start (runs in ~30 s, no external data)

```bash
# 1. synthetic demo data (targets, transcriptome, 3'UTRs, labelled benchmark)
python examples/make_demo_data.py -o demo_data

# 2. 3'UTR hexamer table for the seed-promiscuity (POTS) axis
siforge build-seedtable demo_data/utr3.fa -o demo_data/seedtable.json

# 3. fit calibration + conformal quantiles on a labelled benchmark
siforge calibrate demo_data/benchmark.tsv \
    -o demo_data/calibration.json \
    --reference-out demo_data/reference.json

# 4. design
siforge design demo_data/targets.fa -o out \
    --transcriptome demo_data/transcriptome.fa \
    --seed-table   demo_data/seedtable.json \
    --calibration  demo_data/calibration.json \
    --reference    demo_data/reference.json \
    --profile research --top 4

# 5. the four validation experiments
python examples/run_validation.py --data demo_data
```

Outputs land in `out/`: `candidates.tsv`, `candidates.parquet`,
`order_sheet.tsv`, `report.html`, `manifest.json`.

### Real data

Identical commands; swap the FASTAs. You need three things from GENCODE or
Ensembl:

```bash
# target transcript(s)
samtools faidx gencode.v4x.transcripts.fa ENST00000237014.8 > targets.fa
# transcriptome for full-length off-target scanning
ln -s gencode.v4x.transcripts.fa transcriptome.fa
# 3'UTRs for the POTS seed table (build once, reuse forever)
siforge build-seedtable gencode_3utr.fa -o resources/seedtable_v4x.json
```

The benchmark for `calibrate` is a TSV with `sense` (or `guide`/`site`),
`efficacy` in [0,1] or percent, and optionally `dataset` for grouped
cross-fitting. Harmonise Huesken / Reynolds / Vickers / Khvorova / Taka into
that shape; **keep `dataset` populated**, it is what prevents leakage.

---

## Getting the real input files

`examples/fetch_resources.py` builds the three non-target inputs from public
sources. It never downloads anything itself — you fetch the source files,
which keeps provenance explicit and works on clusters with no outbound
network. `python examples/fetch_resources.py --help-urls` prints every
command.

### 1. Transcriptome FASTA — direct link

```bash
curl -O https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_50/gencode.v50.transcripts.fa.gz
gunzip gencode.v50.transcripts.fa.gz          # ~1.5 GB uncompressed
```

Mouse: swap in `Gencode_mouse/release_M37/gencode.vM37.transcripts.fa.gz`.

### 2. 3'UTR FASTA — no direct link exists; build it

GENCODE publishes no 3'UTR FASTA. The protein-coding transcripts file
carries `UTR3:start-end` spans in its headers, so the 3'UTRome can be cut
out of it with no genome or GTF:

```bash
curl -O https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_50/gencode.v50.pc_transcripts.fa.gz
python examples/fetch_resources.py utr3 gencode.v50.pc_transcripts.fa.gz -o gencode_v50_3utr.fa
siforge build-seedtable gencode_v50_3utr.fa -o resources/seedtable.json
```

Use `pc_transcripts`, not `transcripts` — only the former has UTR3 fields.

### 3. Benchmark TSV — no harmonised release exists; convert

The published efficacy datasets are scattered across paper supplements.
They are redistributed, already harmonised to 19-nt guides with efficacy
in [0,1], in the OligoFormer repository:

```bash
git clone --depth 1 https://github.com/lulab/OligoFormer
python examples/fetch_resources.py benchmark OligoFormer/data \
    -o benchmark.tsv --holdout taka
```

`Hu.csv` is Huesken et al. 2005 (2,361 siRNAs), `Taka.csv` a reporter set
(702), `Mix.csv` smaller sets (472). The converter **verifies** that the
`siRNA` column is the guide strand by checking its reverse complement
occurs in the supplied mRNA context — a silently flipped strand would
corrupt every calibration curve without raising an error. Cite the
original publications, not the redistribution.

### Verify what you built

```bash
python examples/fetch_resources.py check \
    --transcriptome gencode.v50.transcripts.fa \
    --utr3 gencode_v50_3utr.fa \
    --benchmark benchmark.tsv
```

### Measured performance on this real data

Calibrated on Huesken + Mix (n=2,833), validated on Taka (n=702) held out
entirely:

| Metric | Value |
|---|---|
| Spearman rho (cross-dataset) | +0.511 |
| precision@10 | 1.000 |
| NDCG@10 | 1.000 |
| Best single engine (in-distribution) | +0.522 (pwm) |
| Conformal coverage | 0.977 vs nominal 0.900 |

Two honest readings. Cross-dataset rho around 0.5 is normal for this field —
published tools report 0.6–0.7 within-dataset and less across datasets — and
the top-10 precision is what actually matters when you order a shortlist.
The conformal intervals come out *conservative* across datasets (97.7%
coverage where 90% was requested), because cross-dataset residuals are
larger than the within-distribution residuals they were fitted on. Safe,
but wider than necessary; refit per dataset family if you need tighter
intervals.

---

## How the score is computed

| Stage | What happens | Where |
|---|---|---|
| 0 | **Gates** — GC, homopolymer, G-quadruplex, immunostimulatory motif, seed-target Tm ≥ 21.5 °C, common SNP, near-identical off-target. Failures are *eliminated*, never traded off. | `gates.py` |
| 1 | **Normalize** — local rank percentile within the target, plus global percentile against a reference distribution. | `scoring.py` |
| 2 | **Calibrate** — per-engine isotonic regression to fractional knockdown, fitted with grouped CV. | `Calibration.fit` |
| 3 | **Fuse** — minimum-variance weights from the shrunk residual covariance (correlated engines share one vote), blended λ with robust rank aggregation. | `min_variance_weights`, `rra_score` |
| 4 | **Desirability** — Derringer–Suich per objective, combined as a weighted **geometric** mean so any near-zero axis vetoes. | `derringer`, `geometric_mean` |
| 5 | **Uncertainty** — split-conformal interval, Mondrian by inter-engine disagreement; ranking is by the lower confidence bound. | `Calibration._fit_conformal` |
| 6 | **Portfolio** — greedy selection under transcript spacing, seed Hamming distance and off-target Jaccard constraints. | `select_portfolio` |

```
USS = 100 · ( Π_j d_j^{v_j} )^{1/Σv_j}
```

`USS_LCB` is the same expression with potency replaced by its conformal
lower bound. **Rank on `USS_LCB`**, inspect `USS`.

---

## Profiles

`siforge/profiles/research.yaml` and `therapeutic.yaml`. Copy one and edit —
the objective weights `v_j` are deliberately exposed, documented and *not*
learned, because public benchmarks only measure potency, so fitting the
weights collapses everything back onto that single axis. That boundary is a
values choice and belongs in a config file, not in a loss function.

Therapeutic tightens GC, enables the immunostimulatory and SNP gates, raises
the specificity weight to 0.35 and the coverage weight to 0.15, and widens
portfolio spacing.

---

## Adding the ML engines

Native engines need no setup. Third-party predictors are adapters declared
in the profile:

```yaml
external_engines:
  - name: oligoformer
    family: rnafm_transformer
    enabled: true
    command: ["docker", "run", "--rm", "-v", "/tmp:/tmp",
              "yilanbai/oligoformer", "predict",
              "--in", "{input}", "--out", "{output}"]
```

The adapter contract is deliberately minimal: siforge writes a TSV of
`candidate_id / guide / sense / site`, your command writes back
`candidate_id / score`. Nothing else is assumed, so upstream CLI churn is a
one-line YAML fix rather than a code change.

**Set `family` carefully.** OligoFormer, AttSiOff and DeepSipred all consume
RNA-FM embeddings and overlapping training corpora — give them the same
family or the fusion step will count them as three independent votes. Pin one
RNA-FM version across all of them.

A missing or failing engine is skipped and the weights renormalise; values
are never imputed, and the prediction interval widens to reflect the loss.

---

## Validation experiments

`python examples/run_validation.py` runs four checks. Results on the shipped
synthetic benchmark (n=900 train / 300 held out):

**E1 — ablation.** Fused potency ρ=0.757 vs best single engine ρ=0.669
(+0.089). A λ sweep is printed: on this synthetic data λ=1.0 (pure weighted
mean) is best in both the all-engines and engines-dropped columns, so **re-tune
λ on your real benchmark** rather than trusting the 0.6 default.

**E2 — generalisation.** Held-out ρ=0.724, precision@10 = 0.800, conformal
coverage 0.900 against a nominal 0.900.

**E3 — specificity veto.** A ceteris-paribus sweep varying only POTS on a real
candidate: USS declines monotonically (76.5 → 53.2) and hits 0.00 at the
`pots_low` threshold. The weaker pairwise statistic in the same experiment is
reported honestly and is discussed in `docs/FINDINGS.md`.

**E4 — portfolio.** Picks respect spacing (min 95 nt and 222 nt on the two
demo targets) with 4/4 distinct seeds.

Unit tests: `pytest -q` (29 tests).

---

## Nextflow

```bash
nextflow run nextflow/main.nf \
  --targets targets.fa --transcriptome transcriptome.fa \
  --utr3 gencode_3utr.fa --benchmark benchmark.tsv \
  --profile therapeutic --outdir results -profile conda
```

Parallelises per transcript chunk; `-profile slurm|awsbatch|docker|conda`.

---

## Honest limitations

- The shipped `pwm` coefficients are a **surrogate**, not the published
  i-Score or DSIR tables. Supply the real coefficients via `--pwm` before
  reporting anything as i-Score or DSIR. See `docs/PWM.md`.
- The demo benchmark is synthetic. It exercises every code path and has a
  known generative model, which makes the validation meaningful as a
  *software* test — it is not evidence about real siRNA biology.
- `isoform_coverage` and `conservation` default to 1.0 unless you annotate
  them; the coverage and conservation objectives are inert until you do.
- No wet-lab validation. Treat USS as a ranking aid, not a guarantee.
- Objective weights are expert priors, not fitted values. That is deliberate.

## Citing the underlying methods

siforge reimplements published algorithms; cite the originals, not siforge
alone: Reynolds 2004; Ui-Tei 2004; Amarzguioui & Prydz 2004; Katoh & Suzuki
2007; Naito 2009 (siDirect 2.0); Xia 1998 (nearest-neighbour parameters);
Boudreau 2013 (siSPOTR/POTS); Lorenz 2011 (ViennaRNA); and any third-party
engine you enable.

MIT licensed. Third-party engines carry their own licences — several
academic models are non-commercial; `siforge engines` prints what is loaded.
