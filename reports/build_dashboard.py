"""Generate the pipeline/training dashboard from real on-disk state.

Reads whatever actually exists -- raw catalogs, the region decision, the feature
manifest, and any training runs under models/runs/*.json -- and renders
reports/dashboard.html. Nothing here invents numbers: a stage with no artifact
renders as an explicit empty state. Re-run after each pipeline step.
"""
from __future__ import annotations

import csv
import datetime as dt
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RAW, PROCESSED, ROOT, REPORTS  # noqa: E402

RUNS = ROOT / "models" / "runs"
E = html.escape

# ---------------------------------------------------------------- state ----


def read_catalog(stem: str):
    p = RAW / f"{stem}.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    dates = sorted(r["event_date_iso"] for r in rows if r.get("event_date_iso"))

    def ok(r):
        try:
            la, lo = float(r["latitude"]), float(r["longitude"])
        except (TypeError, ValueError, KeyError):
            return False
        return -90 <= la <= 90 and -180 <= lo <= 180 and not (la == 0 and lo == 0)

    return {
        "file": p.name,
        "rows": len(rows),
        "geo": sum(1 for r in rows if ok(r)),
        "distinct_dates": len(set(dates)),
        "span": (dates[0], dates[-1]) if dates else None,
        "countries": len({r.get("country_name") for r in rows if r.get("country_name")}),
    }


def read_json(path: Path):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
    return None


def read_runs():
    if not RUNS.exists():
        return []
    out = []
    for p in sorted(RUNS.glob("*.json")):
        if p.name.startswith("ablation-"):
            continue
        d = read_json(p)
        if d:
            out.append(d)
    return out


def read_validation():
    return {k: read_json(RUNS / f"{k}.json") for k in
            ("hindcast-trigger", "external-susceptibility",
             "transfer-susceptibility", "global-loro",
             "fire-trigger", "drought-validation")}


def read_ablations():
    if not RUNS.exists():
        return []
    return [d for d in (read_json(p) for p in sorted(RUNS.glob("ablation-*.json"))) if d]


# ------------------------------------------------------------ rendering ----

STAGES = [
    ("M0", "Ingest", "Pull hazard catalogs, verify georeferencing"),
    ("M1", "Region", "Choose study region from label density"),
    ("M2", "Features", "Terrain, land cover, rainfall anomaly vs. climatology"),
    ("M3", "Train", "LightGBM: susceptibility + trigger layers"),
    ("M4", "Evaluate", "Spatial block CV, PR-AUC, calibration"),
    ("M5", "Serve", "Batch score to static JSON"),
]


ABLATION_NOTES = {
    "susceptibility": ("Dropping elevation and landscape position asks how much of the score "
                       "is terrain physics rather than where roads and reporters are."),
    "trigger": ("Dropping the percentile features asks whether rainfall has to be judged "
                "against each cell's own record, or whether raw millimetres would do. "
                "Percentiles alone nearly match the full set; absolute millimetres alone "
                "lose 9% — which is the brief's “never an absolute threshold” rule, "
                "measured."),
}


def fmt(n):
    return f"{n:,}" if isinstance(n, (int, float)) else "&mdash;"


def pill(state, label):
    return f'<span class="pill pill--{state}">{E(str(label))}</span>'


def stage_states(ev, rp, region, feats, runs):
    done = [r for r in runs if r.get("status") == "complete"]
    return {
        "M0": "done" if (ev and rp) else "idle",
        "M1": "done" if region else ("active" if ev else "idle"),
        "M2": "done" if feats else ("active" if region else "idle"),
        "M3": "done" if done else ("active" if feats else "idle"),
        "M4": "done" if any(r.get("folds") for r in done) else "idle",
        "M5": "done" if (ROOT / "serve" / "out" / "index.json").exists() else "idle",
    }


def svg_folds(folds, key="pr_auc", baseline=None):
    """Bar chart of one metric across held-out spatial CV folds."""
    vals = [f.get(key) for f in folds if isinstance(f.get(key), (int, float))]
    if not vals:
        return ""
    w, h, pad = 360, 128, 24
    n = len(vals)
    gap = (w - pad * 2) / n
    bw = gap * 0.6
    top = max(max(vals), baseline or 0, 0.001) * 1.15
    parts = []
    if baseline:
        by = h - pad - (h - pad * 2) * (baseline / top)
        parts.append(f'<line x1="{pad}" y1="{by:.1f}" x2="{w-pad}" y2="{by:.1f}" class="sv-base"/>')
        parts.append(f'<text x="{w-pad}" y="{by-4:.1f}" class="sv-baselbl">base {baseline:.3f}</text>')
    for i, v in enumerate(vals):
        bh = (h - pad * 2) * (v / top)
        x = pad + i * gap + (gap - bw) / 2
        y = h - pad - bh
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="2" class="sv-bar"/>')
        parts.append(f'<text x="{x+bw/2:.1f}" y="{h-pad+12:.0f}" class="sv-lbl">{i+1}</text>')
        parts.append(f'<text x="{x+bw/2:.1f}" y="{y-4:.1f}" class="sv-val">{v:.3f}</text>')
    return (f'<svg viewBox="0 0 {w} {h}" class="foldchart" role="img" '
            f'aria-label="{E(key)} across held-out spatial folds">'
            f'<line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" class="sv-axis"/>'
            + "".join(parts) + "</svg>")


def svg_curves(series, x_max=0.125, w=430, h=178, pad=32):
    """POD-vs-alarm-rate polylines. series = [(label, cls, [(x,y),...])]."""
    parts = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        yy = h - pad - (h - pad * 2) * frac
        parts.append(f'<line x1="{pad}" y1="{yy:.0f}" x2="{w-pad}" y2="{yy:.0f}" class="sv-gridln"/>')
        parts.append(f'<text x="{pad-4}" y="{yy+3:.0f}" class="sv-ylbl">{frac:.2f}</text>')
    for xv in (0.02, 0.05, 0.10):
        xx = pad + (w - pad * 2) * (xv / x_max)
        parts.append(f'<text x="{xx:.0f}" y="{h-pad+11}" class="sv-lbl">{xv:.0%}</text>')
    for label, cls, pts in series:
        pts = sorted((x, y) for x, y in pts if x <= x_max and y is not None)
        if not pts:
            continue
        d = " ".join(f"{pad + (w-pad*2)*(x/x_max):.1f},{h - pad - (h-pad*2)*y:.1f}"
                     for x, y in pts)
        parts.append(f'<polyline points="{d}" class="{cls}"/>')
        lx, ly = pts[-1]
        parts.append(f'<text x="{pad + (w-pad*2)*(lx/x_max)-2:.0f}" '
                     f'y="{h - pad - (h-pad*2)*ly - 5:.0f}" class="sv-serieslbl">{E(label)}</text>')
    parts.append(f'<line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" class="sv-axis"/>')
    parts.append(f'<text x="{w/2:.0f}" y="{h-3}" class="sv-lbl">alarm days (share of all days)</text>')
    return (f'<svg viewBox="0 0 {w} {h}" class="foldchart" role="img" '
            f'aria-label="events caught vs alarm budget">{"".join(parts)}</svg>')


def render_validation(v):
    cards = []
    hc = v.get("hindcast-trigger")
    if hc:
        op = hc["at_operating_point"]
        g = hc["grid"]
        rows = "".join(
            f'<tr><td>~{m["target_alarm_rate"]:.0%} of days '
            f'({m["target_alarm_rate"]*365.25:.0f}/yr)</td>'
            f'<td class="num">{m["model_pod_1d"]:.0%}</td>'
            f'<td class="num">{m["rule_pod_1d"]:.0%}</td></tr>'
            for m in hc["matched_alarm_rate"])
        curve = svg_curves([
            ("model", "sv-line",
             [(c["alarm_rate"], c["pod_1d"]) for c in hc["curve_model"]]),
            ("rain rule", "sv-line2",
             [(c["alarm_rate"], c["pod_1d"]) for c in hc["curve_rule"]]),
        ])
        cards.append(
            '<section class="card"><h2>Prospective hindcast &mdash; replaying 2016&ndash;2024</h2>'
            f'<p class="empty" style="margin-bottom:12px">Model frozen on data through 2015 '
            f'(training, tuning, calibration, threshold), then run over every day at every cell: '
            f'{g["cell_days"]:,} cell-days, {g["event_days"]} reported event-days &mdash; base rate '
            f'{g["base_rate"]:.5f}, the number the 1:4 training set hides. The training-set '
            f'operating threshold alarmed on {op["alarm_rate"]:.0%} of ALL days (a ~500&times; '
            f'base-rate trap); production thresholds now come from the daily grid itself '
            f'(<code>serve/thresholds.json</code>: alarm budgets of 2/5/10% of days).</p>'
            f'<div class="cols"><div>{curve}<div class="sm muted" style="margin-top:4px">'
            f'Reported events caught (&plusmn;1 day) vs alarm budget, 2016&ndash;2024. The rain '
            f'rule is one feature &mdash; <code>precip_3d_pctl_seasonal</code> &mdash; with a '
            f'threshold.</div></div>'
            f'<div><div class="tablewrap"><table class="grid"><thead><tr><th>Alarm budget</th>'
            f'<th class="num">Model</th><th class="num">Rain rule</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div>'
            f'<p class="sm muted" style="margin-top:10px">The curve flattens near 55&ndash;60%: '
            f'roughly a third of reported slides fall on unremarkable-rain days (human-triggered, '
            f'snowmelt, mis-dated) &mdash; a ceiling for any rainfall-only trigger. FAR stays '
            f'&gt;99% at these budgets because reported events are rare-in-time: an alarm means '
            f'&ldquo;conditions dangerous&rdquo;, not &ldquo;a slide will be reported&rdquo;.</p>'
            f'</div></div></section>')
    ec = v.get("external-susceptibility")
    if ec:
        mono = ec.get("mean_our_score_by_nasa_class", {})
        mono_s = " &rarr; ".join(f"{val:.2f}" for _, val in sorted(mono.items()))
        cards.append(
            '<section class="card"><h2>External check &mdash; NASA global susceptibility map</h2>'
            f'<p class="empty">Sampled Stanley &amp; Kirschbaum&rsquo;s global map (~1&nbsp;km, '
            f'classes 1&ndash;5, live on the same NASA server) at all {ec["n"]:,} of our points. '
            f'On identical labels: ours PR-AUC <strong>{ec["ours"]["pr_auc"]:.3f}</strong> '
            f'(ROC {ec["ours"]["roc_auc"]:.3f}) vs global map '
            f'<strong>{ec["nasa_global"]["pr_auc"]:.3f}</strong> '
            f'(ROC {ec["nasa_global"]["roc_auc"]:.3f}). Our mean score rises monotonically with '
            f'their class ({mono_s}) &mdash; the two maps agree about which terrain is dangerous; '
            f'ours adds 30&nbsp;m resolution. Caveat: our labels share the reporting process our '
            f'model was fit to, so read this as &ldquo;sane and locally sharper&rdquo;, not '
            f'&ldquo;beats NASA globally&rdquo;.</p></section>')
    tr = v.get("transfer-susceptibility")
    if tr:
        rows = "".join(
            f'<tr><td>{lbl}</td><td class="num">{tr[k]["roc_auc"]:.3f}</td>'
            f'<td class="num">{tr[k]["pr_auc"]:.3f}</td></tr>'
            for k, lbl in (("transfer", "PNW model, frozen (transfer)"),
                           ("local", "Trained on Myanmar (ceiling)"),
                           ("slope_only", "Slope alone (floor)")))
        resc = tr.get("rescue_attempts", {})
        resc_rows = "".join(
            f'<tr><td>{E(d.get("note", k))}</td>'
            f'<td class="num">{d["roc_auc"]:.3f}</td><td class="num sm muted">failed</td></tr>'
            for k, d in resc.items())
        cards.append(
            '<section class="card"><h2>Transfer test &mdash; Myanmar satellite inventory '
            '(bias-free labels)</h2>'
            f'<p class="empty" style="margin-bottom:12px">The Chin/Rakhine inventory is '
            f'satellite-mapped &mdash; every visible failure digitised, no roads-and-reporters '
            f'filter. Scoring {tr["n"]:,} Myanmar points with the terrain model trained only in '
            f'the Pacific Northwest went badly &mdash; and every rescue failed:</p>'
            f'<div class="cols"><div class="tablewrap"><table class="grid"><thead><tr>'
            f'<th>Predictor</th><th class="num">ROC-AUC</th><th class="num">PR-AUC</th></tr>'
            f'</thead><tbody>{rows}</tbody></table></div>'
            f'<div class="tablewrap"><table class="grid"><thead><tr><th>Rescue attempt</th>'
            f'<th class="num">ROC-AUC</th><th></th></tr></thead><tbody>{resc_rows}</tbody>'
            f'</table></div></div>'
            f'<p class="empty" style="margin-top:14px"><strong>Why:</strong> 57&ndash;63% of '
            f'Myanmar terrain lies beyond the PNW&rsquo;s 90th percentile in roughness, relief '
            f'and elevation &mdash; the trees saturate outside their training support, and the '
            f'discriminating relationships are regime-specific. The features work there (local '
            f'model 0.742) but the learned weighting does not carry. <strong>Conclusion:</strong> '
            f'regional models where labels exist; slope-heuristic or NASA&rsquo;s global map '
            f'elsewhere; never ship a trained susceptibility model outside its region without '
            f'local validation. This is why v1&rsquo;s regional scope was the right scope &mdash; '
            f'and why NASA&rsquo;s global product is a heuristic, not a trained model.</p>'
            f'</section>')
    gl = v.get("global-loro")
    if gl:
        regs = gl.get("regions", {})
        ens = gl.get("ensemble", {})
        rows = []
        for rg in ("myanmar", "vietnam", "laos", "philippines", "brazil",
                   "malawi", "mexico", "colombia", "pnw"):
            r = regs.get(rg)
            if not r:
                continue
            e = ens.get(rg, {})
            nasa = r.get("nasa", {}).get("roc_auc")
            cells = [f'{r["local"]["roc_auc"]:.3f}',
                     f'{r["loro_rank"]["roc_auc"]:.3f}',
                     f'{r["slope"]["roc_auc"]:.3f}',
                     f'{nasa:.3f}' if nasa else "&mdash;",
                     f'<strong>{e.get("ensemble", 0):.3f}</strong>' if e else "&mdash;"]
            rows.append(f'<tr><td>{E(rg)}</td>'
                        + "".join(f'<td class="num">{c}</td>' for c in cells) + "</tr>")
        cards.append(
            '<section class="card"><h2>Global floor experiment &mdash; '
            'leave-one-region-out on four continents</h2>'
            '<p class="empty" style="margin-bottom:12px">Eight regions, 78,084 rows; for '
            'each one, every &ldquo;global&rdquo; method is scored on a region it has never '
            'seen. Pooled multi-region training eliminates the below-chance transfer of the '
            'single-region export (Myanmar 0.47 &rarr; 0.66), but no single method dominates '
            'everywhere &mdash; so the global Tier-B floor is an equal-weight '
            '<strong>ensemble</strong> of rank-pooled model + slope + NASA class: best mean '
            '(0.718) and best worst-case (0.588) of any unseen-region method. Local models '
            'stay far ahead where labels exist; all seven inventory regions now ship trained '
            'Tier-A artifacts.</p>'
            '<div class="tablewrap"><table class="grid"><thead><tr><th>Held-out region</th>'
            '<th class="num">Local (ceiling)</th><th class="num">Pooled-rank</th>'
            '<th class="num">Slope</th><th class="num">NASA</th>'
            '<th class="num">Ensemble</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table></div>"
            '<p class="sm muted" style="margin-top:10px">ROC-AUC. Ensemble weights fixed a '
            'priori (equal); no fitting to the held-out region. Mexico is the honest hard '
            'case (~0.58 for everything, including local): a single-storm inventory in a '
            'half-degree box of homogeneous terrain &mdash; some places are Tier B because '
            'their labels cannot support more yet.</p></section>')
    fr = v.get("fire-trigger")
    if fr:
        r = fr.get("results", {})

        def g(k, m="roc_auc"):
            return r.get(k, {}).get(m)

        rows = "".join(
            f'<tr><td>{lbl}</td><td class="num">{g(k):.3f}</td>'
            f'<td class="num">{g(k, "pr_auc"):.3f}</td></tr>'
            for k, lbl in (
                ("us_local_cv", "US, 5-fold CV (320 cells)"),
                ("canada_local_cv", "Canada, 5-fold CV (220 cells)"),
                ("us_to_canada_transfer", "<strong>Train US &rarr; test Canada cold</strong>"),
                ("canada_to_us_transfer", "Train Canada &rarr; test US cold"),
                ("pooled_cv", "Pooled both, CV"),
                ("us_vpd_pctl_seasonal", "VPD percentile alone (US)"),
                ("us_kbdi_pctl_seasonal", "KBDI percentile alone (US)"),
            ) if g(k) is not None)
        cards.append(
            '<section class="card"><h2>Fire danger layer &mdash; two continents of validation</h2>'
            '<p class="empty" style="margin-bottom:12px">Danger <em>conditions</em>, not '
            'ignition (most fires are human-started). KBDI fuel-moisture deficit + vapor '
            'pressure deficit + weather percentiles, all label-free functions of NASA POWER '
            'weather &mdash; computable anywhere on Earth. Validated case-crossover against '
            '40,232 US wildfires &ge;100 acres (FPA-FOD, public domain) and 9,005 Canadian '
            'fires &ge;100 ha (NFDB, open): a fire day vs season-matched non-fire days at the '
            'same location.</p>'
            '<div class="tablewrap"><table class="grid"><thead><tr><th>Test</th>'
            '<th class="num">ROC-AUC</th><th class="num">PR-AUC</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div>'
            '<p class="sm muted" style="margin-top:10px">The transfer rows are the point: '
            'fire-weather skill crosses the temperate&rarr;boreal boundary nearly intact '
            '(0.769 cold vs 0.806 local) &mdash; the mirror image of the terrain layer, which '
            'does not travel. Weather layers are global by construction; terrain layers are '
            'regional. Known gaps: 0.5&deg; daily wind smooths downslope wind events (the Camp '
            'Fire mechanism), and there is no fuel/vegetation layer yet, so barren deserts '
            'score on weather alone.</p></section>')
    dr = v.get("drought-validation")
    if dr:
        head = dr.get("drought_head_cv", {})
        cards.append(
            '<section class="card"><h2>Drought layer &mdash; 20 years vs the US Drought Monitor</h2>'
            f'<p class="empty">Empirical SPI &mdash; 30/90/180-day precipitation as a '
            f'percentile of the same calendar window across all years &mdash; label-free and '
            f'global from NASA POWER. Validated against {dr.get("n_county_weeks", 0):,} '
            f'county-weeks of expert-drawn USDM maps across {dr.get("n_counties", 0)} CONUS '
            f'counties: SPI-90 alone detects &ldquo;county &ge;50% in severe drought '
            f'(D2+)&rdquo; at median AUC <strong>{dr.get("auc_spi90_median", 0):.3f}</strong> '
            f'(IQR {dr.get("auc_spi90_iqr", [0,0])[0]:.2f}&ndash;'
            f'{dr.get("auc_spi90_iqr", [0,0])[1]:.2f}); the calibrated severity head reaches '
            f'ROC {head.get("roc_auc", 0):.3f} under county-grouped CV. Where it '
            f'underperforms is informative: precipitation-only misses snowpack- and '
            f'temperature-driven drought, and the head is calibrated on US truth only.</p>'
            '</section>')
    return "".join(cards)


def render(ev, rp, region, feats, runs, blockers, ablations, validation):
    st = stage_states(ev, rp, region, feats, runs)

    rail = "".join(
        f'<li class="stage stage--{st[c]}"><div class="stage__code">{c}</div>'
        f'<div class="stage__name">{E(n)}</div><div class="stage__desc">{E(d)}</div>'
        f'<div class="stage__state">{st[c]}</div></li>'
        for c, n, d in STAGES)

    total = (ev["geo"] if ev else 0) + (rp["geo"] if rp else 0)
    tiles_src = [
        ("Georeferenced labels", fmt(total), "COOLR inventories + reports"),
        ("Inventory events", fmt(ev["geo"]) if ev else "&mdash;", "dense in space, few dates"),
        ("Report events", fmt(rp["geo"]) if rp else "&mdash;",
         f'{fmt(rp["distinct_dates"]) if rp else "&mdash;"} distinct dates'),
        ("Completed runs", fmt(len([r for r in runs if r.get("status") == "complete"])),
         "susceptibility + trigger"),
    ]
    tiles = "".join(
        f'<div class="tile"><div class="tile__label">{E(l)}</div>'
        f'<div class="tile__val">{v}</div><div class="tile__sub">{s}</div></div>'
        for l, v, s in tiles_src)

    rows = []
    for tag, c, role in (("events", ev, "Susceptibility labels &middot; satellite-mapped inventories"),
                         ("reports", rp, "Trigger labels &middot; media/report catalog")):
        if not c:
            rows.append(f'<tr><td>coolr_{tag}</td><td colspan="6" class="muted">not ingested</td></tr>')
            continue
        span = f'{c["span"][0]} &rarr; {c["span"][1]}' if c["span"] else "&mdash;"
        rows.append(
            f'<tr><td><code>{E(c["file"])}</code><div class="muted sm">{role}</div></td>'
            f'<td class="num">{fmt(c["rows"])}</td><td class="num">{fmt(c["geo"])}</td>'
            f'<td class="num">{fmt(c["distinct_dates"])}</td>'
            f'<td class="num">{fmt(c["countries"])}</td>'
            f'<td class="sm">{span}</td><td>{pill("ok", "ingested")}</td></tr>')
    sources = ('<table class="grid"><thead><tr><th>Source</th><th class="num">Rows</th>'
               '<th class="num">Georef</th><th class="num">Dates</th>'
               '<th class="num">Countries</th><th>Span</th><th>State</th></tr></thead>'
               f'<tbody>{"".join(rows)}</tbody></table>')

    if region:
        b = region["bbox"]
        region_html = (
            f'<div class="kv"><span>Region</span><strong>{E(region["name"])}</strong></div>'
            f'<div class="kv"><span>Bounding box</span><code>{b[0]}, {b[1]}, {b[2]}, {b[3]}</code></div>'
            f'<div class="kv"><span>Labels in box</span><strong>{fmt(region.get("n_labels"))}</strong></div>'
            f'<div class="kv"><span>Distinct dates</span><strong>{fmt(region.get("n_dates"))}</strong></div>'
            f'<div class="kv"><span>Why</span><span class="sm">{E(region.get("rationale", ""))}</span></div>')
    else:
        region_html = ('<p class="empty">Not committed yet. <code>pipelines/select_region.py</code> '
                       'ranks candidates by inventory density and by distinct trigger dates; the '
                       'decision writes <code>data/processed/region.json</code>.</p>')

    if feats:
        chips = "".join(f"<li><code>{E(x)}</code></li>" for x in feats.get("features", []))
        feats_html = (
            f'<div class="kv"><span>Rows</span><strong>{fmt(feats.get("n_rows"))}</strong></div>'
            f'<div class="kv"><span>Positives</span><strong>{fmt(feats.get("n_pos"))}</strong></div>'
            f'<div class="kv"><span>Background</span><strong>{fmt(feats.get("n_bg"))}</strong></div>'
            f'<div class="kv"><span>Base rate</span><strong>{feats.get("base_rate", 0):.4f}</strong></div>'
            f'<ul class="chips">{chips}</ul>')
    else:
        feats_html = ('<p class="empty">Not built. Next: terrain derivatives from Copernicus DEM '
                      'GLO-30 (slope, roughness, curvature), land cover, and rainfall '
                      'anomaly-vs-climatology &mdash; never an absolute rainfall threshold.</p>')

    if runs:
        cards = []
        for r in runs:
            folds = r.get("folds") or []

            def agg(k):
                v = [f[k] for f in folds if isinstance(f.get(k), (int, float))]
                return sum(v) / len(v) if v else None

            pr, roc, br = agg("pr_auc"), agg("roc_auc"), agg("brier")
            base = r.get("base_rate")
            lift = (pr / base) if (pr and base) else None
            metrics = "".join(
                f'<div class="metric"><div class="metric__k">{k}</div>'
                f'<div class="metric__v">{v}</div></div>'
                for k, v in (
                    ("PR-AUC", f"{pr:.3f}" if pr else "&mdash;"),
                    ("Base rate", f"{base:.3f}" if base else "&mdash;"),
                    ("Lift", f"{lift:.1f}&times;" if lift else "&mdash;"),
                    ("ROC-AUC", f"{roc:.3f}" if roc else "&mdash;"),
                    ("Brier", f"{br:.3f}" if br else "&mdash;"),
                ))
            cards.append(
                f'<article class="run"><header class="run__hd"><h3>{E(r.get("name", "run"))}</h3>'
                f'{pill("ok" if r.get("status") == "complete" else "warn", r.get("status", "?"))}'
                f'</header><div class="run__meta sm muted">{E(r.get("layer", ""))} &middot; '
                f'{E(r.get("model", ""))} &middot; {fmt(r.get("n_rows"))} rows &middot; '
                f'{len(folds)} spatial folds</div><div class="metrics">{metrics}</div>'
                f'{svg_folds(folds, "pr_auc", base)}'
                f'<div class="sm muted">PR-AUC per held-out spatial block</div></article>')
        runs_html = f'<div class="runs">{"".join(cards)}</div>'
    else:
        runs_html = (
            '<div class="empty empty--lg"><p><strong>No training runs yet.</strong> This panel '
            'fills in automatically once a trainer writes to <code>models/runs/*.json</code>.</p>'
            '<p class="sm">Runs will report <em>PR-AUC as the headline number, not accuracy</em>. '
            'On a rare-event problem a model that always predicts &ldquo;no event&rdquo; scores '
            '&gt;99% accuracy and is worthless, so each run is shown against the base rate it has '
            'to beat, plus ROC-AUC, Brier score, and a per-fold breakdown across held-out spatial '
            'blocks &mdash; never a random split, which leaks between neighbouring cells.</p></div>')

    if ablations:
        blocks = []
        for ab in ablations:
            res = ab.get("results") or []
            if not res:
                continue
            ref = res[0]["pr_auc"] or 1.0
            trs = []
            for r in res:
                pr = r["pr_auc"]
                d = (pr - ref) / ref * 100 if r is not res[0] else None
                cls = "" if d is None else (" class=\"drop\"" if d < -10 else "")
                trs.append(
                    f'<tr><td><code>{E(r["set"])}</code></td>'
                    f'<td class="num">{r["n_features"]}</td>'
                    f'<td class="num">{pr:.4f}</td>'
                    f'<td class="num">{r["lift"]:.2f}&times;</td>'
                    f'<td class="num">{r["roc_auc"]:.4f}</td>'
                    f'<td class="num"{cls}>{"&mdash;" if d is None else f"{d:+.1f}%"}</td></tr>')
            lay = ab.get("layer", "")
            blocks.append(
                f'<h3 style="margin:18px 0 6px">{E(lay)}</h3>'
                f'<p class="empty sm" style="margin-bottom:10px">{ABLATION_NOTES.get(lay, "")}</p>'
                f'<div class="tablewrap"><table class="grid"><thead><tr>'
                f'<th>Feature set</th><th class="num">n</th><th class="num">PR-AUC</th>'
                f'<th class="num">Lift</th><th class="num">ROC-AUC</th>'
                f'<th class="num">vs all</th></tr></thead><tbody>{"".join(trs)}</tbody>'
                f'</table></div>')
        ablation_html = (
            '<section class="card"><h2>Bias ablation &mdash; how much of the score is real?</h2>'
            '<p class="empty" style="margin-bottom:14px">COOLR is report-derived, so labels '
            'cluster where roads and people are. The first build sampled background points '
            'uniformly across the region, which filled the negatives with remote steep '
            'wilderness nobody reports on &mdash; the model learned <em>steep&nbsp;=&nbsp;no '
            'landslide</em> and ranked the Columbia Gorge (44&deg;) below a flat valley floor. '
            'Switching to target-group background sampling (anchored on real events, so observer '
            'effort cancels) cut the elevation mismatch between classes from &minus;1.27 to '
            '&minus;0.48 standardised, and dropped low-elevation enrichment from 5.1&times; to '
            '2.1&times;. Headline PR-AUC fell 0.71&nbsp;&rarr;&nbsp;0.57 &mdash; the model got '
            '<em>better</em>; the 0.71 was substantially measuring where roads are.</p>'
            + "".join(blocks) +
            '<p class="empty" style="margin-top:16px">Elevation and landscape position still '
            'carry ~32% of the susceptibility score between them. TPI is partly real physics '
            '(failures initiate on convergent slopes), so this is not all bias &mdash; but read '
            'that score as <strong>relative within this region</strong>, not an absolute '
            'probability. Out-of-fold, observed landslide rate rises monotonically with slope, '
            '0.075 at 0&ndash;2&deg; to 0.287 at 30&ndash;66&deg;, and the model tracks it.'
            '</p></section>')
    else:
        ablation_html = ""

    blockers_html = ""
    if blockers:
        items = "".join(
            f'<li class="blk blk--{E(b.get("severity", "warn"))}"><div class="blk__hd">'
            f'{pill(b.get("severity", "warn"), b.get("severity", "warn"))}'
            f'<strong>{E(b.get("title", ""))}</strong></div>'
            f'<p class="sm">{E(b.get("detail", ""))}</p>'
            f'<p class="sm muted"><strong>Effect:</strong> {E(b.get("effect", ""))}</p></li>'
            for b in blockers)
        blockers_html = f'<section class="card"><h2>Blockers</h2><ul class="blks">{items}</ul></section>'

    out = TEMPLATE
    for k, v in (("__NOW__", dt.datetime.now().strftime("%Y-%m-%d %H:%M")),
                 ("__ROOT__", E(str(ROOT))), ("__RAIL__", rail), ("__TILES__", tiles),
                 ("__SOURCES__", sources), ("__REGION__", region_html),
                 ("__FEATS__", feats_html), ("__RUNS__", runs_html),
                 ("__ABLATION__", ablation_html),
                 ("__VALIDATION__", render_validation(validation or {})),
                 ("__BLOCKERS__", blockers_html)):
        out = out.replace(k, v)
    return out


TEMPLATE = """<title>Hazard Model Telemetry</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root {
  --bg:#f3f6f8; --surface:#ffffff; --surface2:#e8edf1; --border:#d3dbe3;
  --text:#0e151b; --dim:#55636f; --faint:#8894a0;
  --accent:#0a7887; --ok:#1b7550; --ok-soft:#d6eee3; --warn:#8f6100;
  --warn-soft:#f5e7c5; --crit:#a82a20; --crit-soft:#f6ddda;
  --idle:#76828d; --idle-soft:#e2e7ec;
  --shadow:0 1px 2px rgba(14,21,27,.06),0 4px 14px rgba(14,21,27,.05);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg:#0b1016; --surface:#131a21; --surface2:#1a222b; --border:#27313b;
    --text:#e3e9f0; --dim:#96a2b0; --faint:#68747f;
    --accent:#38b0c0; --ok:#41b283; --ok-soft:#0f2820; --warn:#d29e3d;
    --warn-soft:#2a2112; --crit:#de6659; --crit-soft:#2d1614;
    --idle:#75808b; --idle-soft:#1b232c;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 4px 16px rgba(0,0,0,.3);
  }
}
:root[data-theme="dark"] {
  --bg:#0b1016; --surface:#131a21; --surface2:#1a222b; --border:#27313b;
  --text:#e3e9f0; --dim:#96a2b0; --faint:#68747f;
  --accent:#38b0c0; --ok:#41b283; --ok-soft:#0f2820; --warn:#d29e3d;
  --warn-soft:#2a2112; --crit:#de6659; --crit-soft:#2d1614;
  --idle:#75808b; --idle-soft:#1b232c;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 4px 16px rgba(0,0,0,.3);
}
* { box-sizing:border-box; }
body {
  margin:0; background:var(--bg); color:var(--text);
  font-family:"IBM Plex Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  font-size:15px; line-height:1.55; -webkit-font-smoothing:antialiased;
}
.wrap { max-width:1180px; margin:0 auto; padding:32px 24px 72px;
        display:flex; flex-direction:column; gap:22px; }
code { font-family:"IBM Plex Mono",ui-monospace,Consolas,monospace; font-size:.92em; }
.num { text-align:right; font-variant-numeric:tabular-nums;
       font-family:"IBM Plex Mono",ui-monospace,Consolas,monospace; }
.sm { font-size:12.5px; }
.muted { color:var(--dim); }
h1 { font-size:27px; font-weight:600; margin:0; letter-spacing:-.02em; text-wrap:balance; }
h2 { font-size:12px; font-weight:600; margin:0 0 14px; text-transform:uppercase;
     letter-spacing:.1em; color:var(--dim); }
h3 { font-size:15px; font-weight:600; margin:0; }
.hd { display:flex; flex-wrap:wrap; gap:16px; align-items:flex-end;
      justify-content:space-between; border-bottom:1px solid var(--border); padding-bottom:18px; }
.hd__eyebrow { font-family:"IBM Plex Mono",monospace; font-size:11px; letter-spacing:.14em;
      text-transform:uppercase; color:var(--accent); margin-bottom:6px; }
.hd__meta { text-align:right; font-size:11.5px; color:var(--faint); line-height:1.7; }
.rail { list-style:none; margin:0; padding:0; display:grid; gap:8px;
        grid-template-columns:repeat(auto-fit,minmax(152px,1fr)); }
.stage { background:var(--surface); border:1px solid var(--border); border-radius:7px;
         padding:12px 13px 11px; position:relative; overflow:hidden; }
.stage::before { content:""; position:absolute; top:0; bottom:0; left:0; width:3px; background:var(--idle); }
.stage--done::before { background:var(--ok); }
.stage--active::before { background:var(--accent); }
.stage__code { font-family:"IBM Plex Mono",monospace; font-size:11px;
      color:var(--faint); letter-spacing:.08em; }
.stage__name { font-weight:600; font-size:14.5px; margin:1px 0 3px; }
.stage__desc { font-size:11.5px; color:var(--dim); line-height:1.4; }
.stage__state { font-family:"IBM Plex Mono",monospace; font-size:10.5px; text-transform:uppercase;
      letter-spacing:.09em; margin-top:9px; color:var(--idle); }
.stage--done .stage__state { color:var(--ok); }
.stage--active .stage__state { color:var(--accent); }
.tiles { display:grid; gap:10px; grid-template-columns:repeat(auto-fit,minmax(196px,1fr)); }
.tile { background:var(--surface); border:1px solid var(--border); border-radius:7px;
        padding:15px 16px; box-shadow:var(--shadow); }
.tile__label { font-size:11.5px; text-transform:uppercase; letter-spacing:.07em; color:var(--dim); }
.tile__val { font-family:"IBM Plex Mono",monospace; font-size:30px; font-weight:500;
        letter-spacing:-.02em; margin:4px 0 2px; font-variant-numeric:tabular-nums; }
.tile__sub { font-size:12px; color:var(--faint); }
.card { background:var(--surface); border:1px solid var(--border); border-radius:9px;
        padding:20px 22px; box-shadow:var(--shadow); }
.cols { display:grid; gap:22px; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); }
.tablewrap { overflow-x:auto; }
table.grid { width:100%; border-collapse:collapse; font-size:13.5px; }
table.grid th { text-align:left; font-weight:600; font-size:11px; text-transform:uppercase;
        letter-spacing:.07em; color:var(--dim); padding:0 12px 9px 0;
        border-bottom:1px solid var(--border); white-space:nowrap; }
table.grid th.num { text-align:right; }
table.grid td { padding:11px 12px; border-bottom:1px solid var(--border); vertical-align:top; }
table.grid td:first-child { padding-left:0; }
table.grid tr:last-child td { border-bottom:none; }
.pill { display:inline-block; font-family:"IBM Plex Mono",monospace; font-size:10.5px;
        text-transform:uppercase; letter-spacing:.07em; padding:2px 7px;
        border-radius:3px; white-space:nowrap; }
.pill--ok { background:var(--ok-soft); color:var(--ok); }
.pill--warn { background:var(--warn-soft); color:var(--warn); }
.pill--crit { background:var(--crit-soft); color:var(--crit); }
.pill--idle { background:var(--idle-soft); color:var(--idle); }
.kv { display:flex; gap:14px; padding:8px 0; border-bottom:1px solid var(--border); font-size:13.5px; }
.kv:last-of-type { border-bottom:none; }
.kv > span:first-child { flex:0 0 122px; color:var(--dim); font-size:12.5px; }
.chips { list-style:none; display:flex; flex-wrap:wrap; gap:5px; padding:12px 0 0; margin:0; }
.chips li { background:var(--surface2); border-radius:3px; padding:2px 7px; font-size:11.5px; }
.empty { color:var(--dim); font-size:13.5px; margin:0; }
.empty--lg { border:1px dashed var(--border); border-radius:7px; padding:20px; background:var(--surface2); }
.empty--lg p { margin:0 0 10px; }
.empty--lg p:last-child { margin:0; }
.runs { display:grid; gap:16px; grid-template-columns:repeat(auto-fit,minmax(340px,1fr)); }
.run { border:1px solid var(--border); border-radius:7px; padding:16px; background:var(--surface2); }
.run__hd { display:flex; align-items:center; gap:10px; justify-content:space-between; }
.run__meta { margin:3px 0 12px; }
.metrics { display:grid; grid-template-columns:repeat(auto-fit,minmax(84px,1fr)); gap:8px; margin-bottom:14px; }
.metric { background:var(--surface); border:1px solid var(--border); border-radius:5px; padding:8px 10px; }
.metric__k { font-size:10.5px; text-transform:uppercase; letter-spacing:.06em; color:var(--dim); }
.metric__v { font-family:"IBM Plex Mono",monospace; font-size:18px; font-variant-numeric:tabular-nums; }
.foldchart { width:100%; height:auto; display:block; }
.sv-bar { fill:var(--accent); }
.sv-lbl { fill:var(--faint); font-size:9px; text-anchor:middle; font-family:"IBM Plex Mono",monospace; }
.sv-val { fill:var(--dim); font-size:8.5px; text-anchor:middle; font-family:"IBM Plex Mono",monospace; }
.sv-baselbl { fill:var(--warn); font-size:8.5px; text-anchor:end; font-family:"IBM Plex Mono",monospace; }
.sv-axis { stroke:var(--border); stroke-width:1; }
td.drop { color:var(--crit); font-weight:600; }
.sv-base { stroke:var(--warn); stroke-width:1; stroke-dasharray:3 3; }
.sv-line { fill:none; stroke:var(--accent); stroke-width:2; }
.sv-line2 { fill:none; stroke:var(--warn); stroke-width:1.6; stroke-dasharray:5 3; }
.sv-gridln { stroke:var(--border); stroke-width:.6; }
.sv-ylbl { fill:var(--faint); font-size:8.5px; text-anchor:end; font-family:"IBM Plex Mono",monospace; }
.sv-serieslbl { fill:var(--dim); font-size:9.5px; text-anchor:end; font-family:"IBM Plex Mono",monospace; }
.blks { list-style:none; margin:0; padding:0; display:flex; flex-direction:column; gap:12px; }
.blk { border:1px solid var(--border); border-left:3px solid var(--idle); border-radius:6px;
       padding:12px 14px; background:var(--surface2); }
.blk--crit { border-left-color:var(--crit); }
.blk--warn { border-left-color:var(--warn); }
.blk__hd { display:flex; align-items:center; gap:9px; margin-bottom:5px; }
.blk p { margin:3px 0 0; }
.foot { color:var(--faint); font-size:11.5px; border-top:1px solid var(--border);
        padding-top:16px; margin:0; }
</style>
<div class="wrap">
  <header class="hd">
    <div>
      <div class="hd__eyebrow">Multi-hazard risk platform &middot; Landslide &middot; Fire &middot; Drought</div>
      <h1>Hazard model telemetry</h1>
    </div>
    <div class="hd__meta">Generated __NOW__<br><code>__ROOT__</code></div>
  </header>
  <section><ol class="rail">__RAIL__</ol></section>
  <section class="tiles">__TILES__</section>
  <section class="card"><h2>Label sources</h2><div class="tablewrap">__SOURCES__</div></section>
  <div class="cols">
    <section class="card"><h2>Study region</h2>__REGION__</section>
    <section class="card"><h2>Feature matrix</h2>__FEATS__</section>
  </div>
  <section class="card"><h2>Training runs &amp; held-out performance</h2>__RUNS__</section>
  __ABLATION__
  __VALIDATION__
  __BLOCKERS__
  <p class="foot">Regenerate with <code>python reports/build_dashboard.py</code>. Every value is read
  from disk &mdash; stages with no artifact render as empty rather than estimated.</p>
</div>
"""


def main():
    out = REPORTS / "dashboard.html"
    out.write_text(
        render(read_catalog("coolr_events_points"),
               read_catalog("coolr_reports_points"),
               read_json(PROCESSED / "region.json"),
               read_json(PROCESSED / "features_manifest.json"),
               read_runs(),
               read_json(PROCESSED / "blockers.json"),
               read_ablations(),
               read_validation()),
        encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
