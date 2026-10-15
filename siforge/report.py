"""Self-contained HTML report with per-candidate score attribution."""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np

CSS = """
:root{--bg:#fbfaf8;--fg:#1d1b18;--mut:#6b6762;--line:#e2ded7;--ok:#2f7d5c;
--warn:#b4690e;--bad:#a33b2a;--acc:#30506b;}
@media(prefers-color-scheme:dark){:root{--bg:#17161a;--fg:#eceae6;--mut:#9b958d;
--line:#302d34;--ok:#6cc39b;--warn:#e0a152;--bad:#e08270;--acc:#8fb2cf;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;}
.wrap{max-width:1180px;margin:0 auto;padding:32px 16px 72px}
h1{font-size:25px;margin:0 0 4px;letter-spacing:-.01em}
h2{font-size:17px;margin:34px 0 10px;letter-spacing:-.01em}
.sub{color:var(--mut);font-size:13px;margin-bottom:22px}
table{border-collapse:collapse;width:100%;font-size:13px;margin-bottom:8px}
th,td{text-align:left;padding:7px 9px;border-bottom:1px solid var(--line)}
th{font-weight:600;color:var(--mut);font-size:11px;text-transform:uppercase;
letter-spacing:.05em;white-space:nowrap}
td.mono,th.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
tr.port{background:color-mix(in srgb,var(--ok) 9%,transparent)}
.badge{display:inline-block;padding:1px 7px;border-radius:99px;font-size:11px;
border:1px solid var(--line);color:var(--mut)}
.bar{height:7px;border-radius:4px;background:var(--line);overflow:hidden;min-width:70px}
.bar>i{display:block;height:100%;background:var(--acc)}
.card{border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:10px 0}
.k{color:var(--mut);font-size:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.wf{display:flex;align-items:center;gap:8px;margin:3px 0;font-size:12px}
.wf .lbl{width:130px;color:var(--mut)}
.wf .val{width:46px;text-align:right;font-family:ui-monospace,monospace}
.note{border-left:3px solid var(--warn);padding:8px 12px;margin:14px 0;
background:color-mix(in srgb,var(--warn) 8%,transparent);font-size:13px;border-radius:0 6px 6px 0}
code{font-family:ui-monospace,monospace;font-size:12px}
@media(max-width:640px){.wrap{padding:20px 12px 56px}table{font-size:12px}
th,td{padding:6px 5px}.wf .lbl{width:92px}}
"""


def _bar(v: float, lo: float = 0.0, hi: float = 1.0) -> str:
    if v != v:
        return '<span class="k">n/a</span>'
    pct = max(0.0, min(1.0, (v - lo) / (hi - lo))) * 100
    return f'<div class="bar"><i style="width:{pct:.0f}%"></i></div>'


def _waterfall(sc) -> str:
    rows = []
    for k, v in sorted(sc.desirability.items(), key=lambda t: -t[1]):
        rows.append(
            f'<div class="wf"><span class="lbl">{html.escape(k)}</span>'
            f"{_bar(v)}<span class=\"val\">{v:.2f}</span></div>"
        )
    return "".join(rows)


def write_html_report(results, path: str | Path, manifest: dict) -> None:
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>siforge design report</title>",
        f"<style>{CSS}</style></head><body><div class='wrap'>",
        "<h1>siforge design report</h1>",
        f"<div class='sub'>profile <b>{html.escape(str(manifest.get('profile')))}</b> "
        f"&middot; siforge {html.escape(str(manifest.get('siforge_version')))} "
        f"&middot; accessibility: {html.escape(str(manifest.get('accessibility')))} "
        f"&middot; engines: {html.escape(', '.join(manifest.get('engines_run', [])))}</div>",
    ]

    if not manifest.get("calibrated"):
        parts.append(
            "<div class='note'><b>Uncalibrated run.</b> No isotonic calibration was "
            "loaded, so <code>potency</code> is a rank percentile rather than an "
            "estimate of fractional knockdown, and the prediction interval uses the "
            "profile default rather than conformal residuals. Run "
            "<code>siforge calibrate</code> on a labelled benchmark before "
            "interpreting potency in absolute terms.</div>"
        )

    for res in results:
        parts.append(f"<h2>{html.escape(res.transcript_id)}</h2>")
        w = res.consensus_w
        wtxt = "n/a" if (w != w) else f"{w:.3f}"
        parts.append(
            "<div class='grid'>"
            f"<div class='card'><div class='k'>candidates tiled</div><b>{res.n_enumerated}</b></div>"
            f"<div class='card'><div class='k'>gated out</div><b>{res.n_gated}</b></div>"
            f"<div class='card'><div class='k'>scored</div><b>{len(res.scored)}</b></div>"
            f"<div class='card'><div class='k'>consensus W</div><b>{wtxt}</b></div>"
            "</div>"
        )
        if res.gate_reasons:
            chips = " ".join(
                f"<span class='badge'>{html.escape(k)} &times;{v}</span>"
                for k, v in sorted(res.gate_reasons.items(), key=lambda t: -t[1])
            )
            parts.append(f"<div style='margin:6px 0 14px'>{chips}</div>")

        port_ids = {s.candidate_id for s in res.portfolio}
        parts.append(
            "<table><thead><tr><th>#</th><th>candidate</th><th>pos</th>"
            "<th class='mono'>guide 5'&rarr;3'</th><th>USS</th><th>USS<sub>LCB</sub></th>"
            "<th>potency (90% PI)</th><th>spec</th><th>acc</th><th>eng</th></tr></thead><tbody>"
        )
        for i, s in enumerate(res.scored[:25], start=1):
            c = s.candidate
            cls = " class='port'" if c.candidate_id in port_ids else ""
            parts.append(
                f"<tr{cls}><td>{i}</td><td class='mono'>{html.escape(c.candidate_id)}</td>"
                f"<td>{c.start + 1}</td><td class='mono'>{c.guide}</td>"
                f"<td><b>{s.uss:.1f}</b></td><td>{s.uss_lcb:.1f}</td>"
                f"<td class='mono'>{s.potency:.2f} [{s.pi_low:.2f}&ndash;{s.pi_high:.2f}]</td>"
                f"<td>{s.desirability.get('specificity', float('nan')):.2f}</td>"
                f"<td>{s.desirability.get('accessibility', float('nan')):.2f}</td>"
                f"<td>{s.n_engines}</td></tr>"
            )
        parts.append("</tbody></table>")

        parts.append("<h2>Selected portfolio</h2>")
        for s in res.portfolio:
            c = s.candidate
            ot = len(getattr(c, "offtarget_hits", []))
            parts.append(
                "<div class='card'>"
                f"<div><b>{html.escape(c.candidate_id)}</b> "
                f"<span class='badge'>USS {s.uss:.1f}</span> "
                f"<span class='badge'>LCB {s.uss_lcb:.1f}</span> "
                f"<span class='badge'>pos {c.start + 1}</span></div>"
                f"<div class='mono' style='margin:8px 0 4px'>sense&nbsp; 5'-{c.sense_oligo}-3'<br>"
                f"guide&nbsp; 5'-{c.guide_oligo}-3'</div>"
                f"<div class='k' style='margin:6px 0'>GC {c.features.get('gc_fraction', float('nan')):.2f} "
                f"&middot; seed Tm {c.features.get('seed_tm', float('nan')):.1f}&deg;C "
                f"&middot; &Delta;&Delta;G {c.features.get('ddg_asymmetry', float('nan')):+.2f} "
                f"&middot; POTS {c.features.get('pots', float('nan')):.2f} "
                f"&middot; off-targets &le;3mm: {ot}</div>"
                f"{_waterfall(s)}"
                "</div>"
            )

    parts.append(
        "<h2>Run manifest</h2><div class='card'><code>"
        + html.escape(json.dumps(manifest, indent=2))
        + "</code></div>"
    )
    parts.append("</div></body></html>")
    Path(path).write_text("".join(parts))
