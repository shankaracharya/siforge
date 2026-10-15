# Step-by-step install and first run

Tested on Linux (Python 3.13) and should work on macOS. ~5 minutes.

## 1. Get the code and create an environment

```bash
tar xzf siforge-1.0.0.tar.gz && cd siforge
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

## 2. Install

```bash
pip install -e ".[structure,parquet,dev]"
```

`structure` = ViennaRNA (real RNAplfold accessibility). `parquet` = pyarrow.
`dev` = pytest.

## 3. Verify

```bash
siforge --version          # siforge 1.0.0
siforge engines            # 7 native engines available, ViennaRNA: True
pytest -q                  # 29 passed
```

If `ViennaRNA available: False`, run `pip install ViennaRNA`. siforge will
still run without it but every report will say the accessibility axis is
degraded.

## 4. Generate demo data and run everything

```bash
python examples/make_demo_data.py -o demo_data

siforge build-seedtable demo_data/utr3.fa -o demo_data/seedtable.json

siforge calibrate demo_data/benchmark.tsv \
    -o demo_data/calibration.json \
    --reference-out demo_data/reference.json

siforge design demo_data/targets.fa -o out \
    --transcriptome demo_data/transcriptome.fa \
    --seed-table   demo_data/seedtable.json \
    --calibration  demo_data/calibration.json \
    --reference    demo_data/reference.json \
    --profile research --top 4

open out/report.html        # or xdg-open
```

## 5. Run the validation experiments

```bash
python examples/run_validation.py --data demo_data
python examples/run_validation.py --data demo_data --only e2   # one at a time
```

## 6. Switch to real data

```bash
# one-time resources
siforge build-seedtable gencode_3utr.fa -o resources/seedtable.json

# your benchmark: TSV with sense|guide|site, efficacy, dataset
siforge calibrate benchmark_harmonised.tsv \
    -o resources/calibration.json --reference-out resources/reference.json

siforge design my_targets.fa -o out_real \
    --transcriptome gencode_transcripts.fa \
    --seed-table resources/seedtable.json \
    --calibration resources/calibration.json \
    --reference resources/reference.json \
    --profile therapeutic --top 5
```

Building the seed table over the full human 3'UTRome takes a few minutes and
~2 GB of RAM; do it once and reuse the JSON.

## 7. Nextflow (optional)

```bash
nextflow run nextflow/main.nf \
    --targets my_targets.fa \
    --transcriptome gencode_transcripts.fa \
    --utr3 gencode_3utr.fa \
    --benchmark benchmark_harmonised.tsv \
    --profile therapeutic \
    --outdir results \
    -profile conda
```

## 8. Enable a third-party ML engine

Install the tool, write a 10-line adapter that reads the candidate TSV and
writes `candidate_id<TAB>score`, then in your profile set `enabled: true` and
point `command` at it. `siforge engines` confirms it is detected. Give
RNA-FM-based engines a shared `family` so they are not counted as independent.

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `No candidates survived gating` | usually `seed_tm_max`. The stderr dump lists per-reason counts; relax that gate in a copied profile. |
| `ViennaRNA available: False` | `pip install ViennaRNA` |
| POTS column is NaN | pass `--utr3` or `--seed-table` |
| `FileNotFoundError: demo_data/benchmark.tsv` | the demo data hasn't been generated. Run steps 4a-4c (`make_demo_data.py`, `build-seedtable`, `calibrate`) before `run_validation.py`, and run them from the repo root. From v1.0.2 the script lists the exact missing files and commands instead of raising. |
| `no calibration loaded` note | expected until you run `siforge calibrate`; potency is a rank percentile until then |
| parquet not written | `pip install pyarrow` (TSV is written regardless) |
| `ModuleNotFoundError: No module named 'siforge'` | the package isn't installed in the interpreter running pytest. Run `pip install -e ".[structure,parquet,dev]"` **in the same environment**. Check with `which python && which pytest` — if one points at anaconda and the other at `.venv`, that's the mismatch. From v1.0.1 `pyproject.toml` sets `pythonpath = ["."]`, so `pytest` also works from the repo root without installing. |
