#!/usr/bin/env nextflow
/*
 * siforge -- multi-engine siRNA design, Nextflow DSL2 wrapper.
 *
 * The Python package does the work; this exists so the stack composes with
 * your other pipelines, runs per-transcript in parallel on a cluster or
 * AWS Batch, and emits a provenance-complete work directory.
 *
 *   nextflow run nextflow/main.nf \
 *     --targets demo_data/targets.fa \
 *     --transcriptome demo_data/transcriptome.fa \
 *     --utr3 demo_data/utr3.fa \
 *     --benchmark demo_data/benchmark.tsv \
 *     --profile research \
 *     --outdir results
 */

nextflow.enable.dsl = 2

params.targets       = null
params.transcriptome = null
params.utr3          = null
params.seedtable     = null
params.benchmark     = null   // optional: fits calibration before designing
params.calibration   = null   // or supply a pre-fitted one
params.reference     = null
params.pwm           = null
params.profile       = 'research'
params.top           = 4
params.start_offset  = 75
params.duplex_len    = 19
params.outdir        = 'results'
params.chunk_size    = 1      // transcripts per design task

if( !params.targets ) error "Supply --targets <FASTA>"

process BUILD_SEEDTABLE {
    tag 'seedtable'
    publishDir "${params.outdir}/resources", mode: 'copy'
    input:  path utr3
    output: path 'seedtable.json'
    script:
    """
    siforge build-seedtable ${utr3} -o seedtable.json
    """
}

process CALIBRATE {
    tag 'calibrate'
    publishDir "${params.outdir}/calibration", mode: 'copy'
    input:  path bench
    output: tuple path('calibration.json'), path('reference.json')
    script:
    """
    siforge calibrate ${bench} \
        --profile ${params.profile} \
        -o calibration.json \
        --reference-out reference.json
    """
}

process SPLIT_TARGETS {
    tag 'split'
    input:  path fa
    output: path 'chunk_*.fa'
    script:
    """
    python3 - <<'PY'
from siforge.seq import read_fasta, write_fasta
recs = list(read_fasta("${fa}"))
n = int(${params.chunk_size})
for i in range(0, len(recs), n):
    write_fasta(f"chunk_{i//n:04d}.fa", recs[i:i+n])
PY
    """
}

process DESIGN {
    tag { chunk.baseName }
    publishDir "${params.outdir}/design/${chunk.baseName}", mode: 'copy'
    cpus 2
    memory '8 GB'
    input:
        path chunk
        path transcriptome
        path seedtable
        path calibration
        path reference
    output:
        path 'out/*'
        path 'out/candidates.tsv', emit: tsv
    script:
    def tx    = transcriptome.name != 'NO_TX'    ? "--transcriptome ${transcriptome}" : ''
    def st    = seedtable.name    != 'NO_ST'     ? "--seed-table ${seedtable}"        : ''
    def cal   = calibration.name  != 'NO_CAL'    ? "--calibration ${calibration}"     : ''
    def ref   = reference.name    != 'NO_REF'    ? "--reference ${reference}"         : ''
    def pwm   = params.pwm ? "--pwm ${params.pwm}" : ''
    """
    siforge design ${chunk} -o out \
        --profile ${params.profile} \
        --top ${params.top} \
        --start-offset ${params.start_offset} \
        --duplex-len ${params.duplex_len} \
        ${tx} ${st} ${cal} ${ref} ${pwm}
    """
}

process MERGE {
    tag 'merge'
    publishDir "${params.outdir}", mode: 'copy'
    input:  path 'part_*.tsv'
    output: path 'all_candidates.tsv'
    script:
    """
    python3 - <<'PY'
import glob, pandas as pd
fs = sorted(glob.glob("part_*.tsv"))
df = pd.concat([pd.read_csv(f, sep="\\t") for f in fs], ignore_index=True)
df.sort_values(["transcript", "USS_LCB"], ascending=[True, False], inplace=True)
df.to_csv("all_candidates.tsv", sep="\\t", index=False)
print(f"merged {len(fs)} chunks -> {len(df)} rows")
PY
    """
}

workflow {
    targets = Channel.fromPath(params.targets, checkIfExists: true)

    seedtable = params.seedtable
        ? Channel.fromPath(params.seedtable, checkIfExists: true)
        : ( params.utr3
            ? BUILD_SEEDTABLE(Channel.fromPath(params.utr3, checkIfExists: true))
            : Channel.fromPath('NO_ST').ifEmpty(file('NO_ST')) )

    if( params.benchmark ) {
        cal_pair   = CALIBRATE(Channel.fromPath(params.benchmark, checkIfExists: true))
        calibration = cal_pair.map { c, r -> c }
        reference   = cal_pair.map { c, r -> r }
    } else if( params.calibration ) {
        calibration = Channel.fromPath(params.calibration, checkIfExists: true)
        reference   = params.reference
            ? Channel.fromPath(params.reference, checkIfExists: true)
            : Channel.value(file('NO_REF'))
    } else {
        calibration = Channel.value(file('NO_CAL'))
        reference   = Channel.value(file('NO_REF'))
    }

    transcriptome = params.transcriptome
        ? Channel.fromPath(params.transcriptome, checkIfExists: true)
        : Channel.value(file('NO_TX'))

    chunks = SPLIT_TARGETS(targets).flatten()

    DESIGN(
        chunks,
        transcriptome.first(),
        seedtable.first(),
        calibration.first(),
        reference.first()
    )

    MERGE(DESIGN.out.tsv.collect())
}
