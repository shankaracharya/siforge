"""Engine registry.

Every engine declares a ``family``. Families are what the correlation-aware
fusion step uses as its fallback grouping when no residual covariance has
been estimated: engines that share training corpora or share an embedding
backbone must not be counted as independent votes. OligoFormer, AttSiOff
and DeepSipred all consume RNA-FM embeddings and overlapping training data,
so they belong to one family, not three.

External ML engines are *adapters*: they shell out to a containerised or
pip-installed third-party tool. If the tool is not present the engine is
skipped and the score stack renormalises -- it never imputes a value.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .rules import RULE_ENGINES, load_pwm
from .seq import Candidate
from .thermo import ddg_asymmetry, duplex_dg


@dataclass(slots=True)
class Engine:
    name: str
    family: str
    fn: Callable[[list[Candidate]], dict[str, float]]
    higher_is_better: bool = True
    available: bool = True
    version: str = "siforge-native"
    note: str = ""


# --------------------------------------------------------------------------
# Native engines
# --------------------------------------------------------------------------
def _rule_engine(name: str) -> Callable[[list[Candidate]], dict[str, float]]:
    fn = RULE_ENGINES[name]
    pwm = load_pwm() if name == "pwm" else None

    def run(cands: list[Candidate]) -> dict[str, float]:
        out: dict[str, float] = {}
        for c in cands:
            res = fn(c.sense, c.guide, pwm) if name == "pwm" else fn(c.sense, c.guide)
            out[c.candidate_id] = res.score
            c.flags.update(res.flags)
        return out

    return run


def _thermo_asymmetry(cands: list[Candidate]) -> dict[str, float]:
    out = {}
    for c in cands:
        v = ddg_asymmetry(c.guide, n=4)
        c.features["ddg_asymmetry"] = v
        c.features["duplex_dg"] = duplex_dg(c.sense)
        out[c.candidate_id] = v
    return out


def _accessibility_engine(cands: list[Candidate]) -> dict[str, float]:
    return {c.candidate_id: c.features.get("accessibility", float("nan")) for c in cands}


NATIVE_ENGINES: list[Engine] = [
    Engine("reynolds", "rules", _rule_engine("reynolds")),
    Engine("uitei", "rules", _rule_engine("uitei")),
    Engine("amarzguioui", "rules", _rule_engine("amarzguioui")),
    Engine("katoh", "rules", _rule_engine("katoh")),
    Engine("pwm", "linear", _rule_engine("pwm"),
           note="surrogate coefficients; supply --pwm for i-Score/DSIR"),
    Engine("thermo_asymmetry", "thermo", _thermo_asymmetry),
    Engine("accessibility", "structure", _accessibility_engine),
]


# --------------------------------------------------------------------------
# External ML adapters
# --------------------------------------------------------------------------
@dataclass(slots=True)
class ExternalSpec:
    """How to invoke a third-party predictor.

    The adapter writes a two-column TSV (candidate_id, guide, sense, site)
    to a temp dir, runs ``command`` with ``{input}`` and ``{output}``
    substituted, and reads back a TSV with columns candidate_id, score.
    This keeps siforge decoupled from each tool's CLI churn -- you pin the
    glue in one YAML block instead of in Python.
    """

    name: str
    family: str
    command: list[str]
    enabled: bool = True
    version: str = "unknown"
    probe: list[str] = field(default_factory=list)


def _external_runner(spec: ExternalSpec) -> Callable[[list[Candidate]], dict[str, float]]:
    def run(cands: list[Candidate]) -> dict[str, float]:
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "candidates.tsv"
            outp = Path(td) / "scores.tsv"
            with open(inp, "w") as fh:
                fh.write("candidate_id\tguide\tsense\tsite\n")
                for c in cands:
                    fh.write(f"{c.candidate_id}\t{c.guide}\t{c.sense}\t{c.site}\n")
            cmd = [
                part.format(input=str(inp), output=str(outp)) for part in spec.command
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True, timeout=3600)
            except Exception as exc:
                warnings.warn(f"engine {spec.name} failed: {exc}")
                return {}
            scores: dict[str, float] = {}
            if not outp.exists():
                warnings.warn(f"engine {spec.name} produced no output")
                return {}
            with open(outp) as fh:
                header = fh.readline()
                del header
                for line in fh:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 2:
                        continue
                    try:
                        scores[parts[0]] = float(parts[1])
                    except ValueError:
                        continue
            return scores

    return run


def probe_available(spec: ExternalSpec) -> bool:
    if not spec.enabled:
        return False
    if spec.probe:
        exe = spec.probe[0]
        if shutil.which(exe) is None and not Path(exe).exists():
            return False
        try:
            subprocess.run(spec.probe, check=True, capture_output=True, timeout=120)
            return True
        except Exception:
            return False
    exe = spec.command[0]
    return shutil.which(exe) is not None or Path(exe).exists()


def build_external_engines(specs: list[ExternalSpec]) -> list[Engine]:
    engines = []
    for spec in specs:
        ok = probe_available(spec)
        engines.append(
            Engine(
                name=spec.name,
                family=spec.family,
                fn=_external_runner(spec),
                available=ok,
                version=spec.version,
                note="" if ok else "not installed -- skipped",
            )
        )
    return engines


def load_external_specs(cfg: dict | None) -> list[ExternalSpec]:
    specs = []
    for item in (cfg or {}).get("external_engines", []) or []:
        specs.append(
            ExternalSpec(
                name=item["name"],
                family=item.get("family", item["name"]),
                command=item["command"],
                enabled=item.get("enabled", True),
                version=item.get("version", "unknown"),
                probe=item.get("probe", []),
            )
        )
    return specs


def run_engines(
    engines: list[Engine], candidates: list[Candidate]
) -> tuple[dict[str, dict[str, float]], list[str]]:
    """Execute every available engine. Returns (scores_by_engine, engines_run)."""
    scores: dict[str, dict[str, float]] = {}
    ran: list[str] = []
    for eng in engines:
        if not eng.available:
            continue
        try:
            res = eng.fn(candidates)
        except Exception as exc:
            warnings.warn(f"engine {eng.name} raised: {exc}")
            continue
        if not res:
            continue
        scores[eng.name] = res
        ran.append(eng.name)
        for c in candidates:
            if c.candidate_id in res:
                c.engine_scores[eng.name] = res[c.candidate_id]
    return scores, ran


def engine_families(engines: list[Engine]) -> dict[str, str]:
    return {e.name: e.family for e in engines}


def describe_engines(engines: list[Engine]) -> str:
    rows = ["name\tfamily\tavailable\tversion\tnote"]
    for e in engines:
        rows.append(f"{e.name}\t{e.family}\t{e.available}\t{e.version}\t{e.note}")
    return "\n".join(rows)


def dump_engine_manifest(engines: list[Engine], path: str | os.PathLike) -> None:
    Path(path).write_text(
        json.dumps(
            [
                {
                    "name": e.name,
                    "family": e.family,
                    "available": e.available,
                    "version": e.version,
                    "note": e.note,
                }
                for e in engines
            ],
            indent=2,
        )
    )
