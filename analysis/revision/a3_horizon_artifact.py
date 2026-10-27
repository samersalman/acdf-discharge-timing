#!/usr/bin/env python3
"""a3_horizon_artifact.py - A3, the follow-up horizon.

Question. Residual risk R(t) is the cumulative incidence over days t to 30 given event free and
alive at the START of day t. Because follow-up stops at day 30, the window shrinks as t grows.
Two quantities address this: (a) the discrete-time hazard, which does not depend on the horizon,
and (b) the size of the shrinking-horizon effect. This script computes both for the primary
composite (WINDOW_COMP, POD_WINDOW), death competing, with the v1 estimator imported through
r1_common.

(a) Discrete-time hazard by band. h(t) = events on day t / patients at risk at the START of day t.
    Bands: day 0, 1, 2, 3, days 4 to 6, 7 to 13, 14 to 30. Rate = band events / sum of the band's
    daily at-risk counts x 100,000 (each patient at risk at the start of a day contributes one
    patient-day for that day), with exact Poisson (Garwood) 95% intervals. Overall and for every
    stratum of the four locked stratifiers; a band-by-stratum cell with fewer than 10 events is
    withheld ("Suppressed") with its counts shown. Last-band rate as a share of the day 0 rate.

(b) Horizon-matched window. H16(t) = (CIF at the start of day t+16 minus CIF at the start of day t)
    / (event free and alive at the start of day t), t = 0..14, the same fixed 16-day horizon
    (days t to t+15) at every t. Compared with the v1 definition R(t) (days t to 30). Threshold
    days at 1%, 0.5%, 0.25%, residual risk at days 3, 7, 14, and the gap R(t) - H16(t), which is
    the shrinking-horizon effect. Both definitions are computed on the SAME 1,000 bootstrap
    replicates (np.random.seed(VD.SEED); strata in W.build_strata order; one multinomial draw block
    per non-suppressed stratum, made inside R1.fit_overall_and_strata and nowhere else).
    One added quantity, labelled as such: the share of the decline in residual risk since day 0 that
    the gap accounts for, ( gap(0) - gap(t) ) / ( R(0) - R(t) ), the measure of option (b).

The v1-definition numbers are compared with v1's own Table 5 and Table 5B CSVs cell by cell.

Run:  PYTHONDONTWRITEBYTECODE=1 python3 a3_horizon_artifact.py
Writes (under the results directory): fragments/results_a3.json (prefix a3.),
        tables-csv/tableS7_hazard_bands.csv, tables-csv/tableS7_horizon_matched.csv,
        figures/figS3_hazard_and_horizon.png and .pdf (skipped when ACDF_FIGURES=0).
        Aggregates only; no patient-level data.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy import stats

import r1_common as R1

VD, W = R1.VD, R1.W
np.random.seed(VD.SEED)

DAYS = R1.DAYS                      # 31
MIN_EVENTS = VD.MIN_STRATUM_EVENTS  # 10
SUPPRESSED = R1.suppressed_text     # "Suppressed"
NOT_APPLICABLE = "Not applicable"
HORIZON = 16                        # fixed forward window, days t to t+15
H_DAYS = DAYS - HORIZON             # starting days t = 0..14 (15 values)
H16_NOT_REACHED = "Not reached by day 14"
V1_NOT_REACHED = "Not reached within 30 days"
THRESHOLDS = VD.RISK_THRESHOLDS     # (0.01, 0.005, 0.0025)
REPORT_DAYS = (3, 7, 14)            # days reported for the gap and for the decline since day 0
RISK_DAYS = (0,) + REPORT_DAYS      # residual-risk rows also carry day 0 (30-day against 16-day cumulative incidence)

# (first day, last day, table label, key segment)
BANDS: Tuple[Tuple[int, int, str, str], ...] = (
    (0, 0, "Day 0", "day0"),
    (1, 1, "Day 1", "day1"),
    (2, 2, "Day 2", "day2"),
    (3, 3, "Day 3", "day3"),
    (4, 6, "Days 4 to 6", "days4-6"),
    (7, 13, "Days 7 to 13", "days7-13"),
    (14, 30, "Days 14 to 30", "days14-30"),
)
LAST_BAND = BANDS[-1]


# ---------------------------------------------------------------------------
# small statistics and formatting helpers (also imported by a8_selection_riskset.py)
# ---------------------------------------------------------------------------

def poisson_ci(k: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Exact (Garwood, chi-square) two-sided interval for a Poisson count k, on the count scale."""
    k = int(k)
    lo = 0.0 if k == 0 else 0.5 * float(stats.chi2.ppf(alpha / 2.0, 2 * k))
    hi = 0.5 * float(stats.chi2.ppf(1.0 - alpha / 2.0, 2 * k + 2))
    return lo, hi


def rate_per_100k(events: int, patient_days: int) -> Tuple[float, float, float]:
    """(rate, lo, hi) in events per 100,000 patient-days with the exact Poisson interval."""
    lo, hi = poisson_ci(events)
    scale = 100_000.0 / float(patient_days)
    return events * scale, lo * scale, hi * scale


def fmt_rate(rate: float, lo: float, hi: float) -> str:
    return f"{rate:.1f} ({lo:.1f} to {hi:.1f})"


def fmt_est_ci(est: float, lo: float, hi: float, decimals: int = 3) -> str:
    return f"{est:.{decimals}f} ({lo:.{decimals}f} to {hi:.{decimals}f})"


def _no_negative_zero(x: float, decimals: int) -> str:
    """Render x with `decimals` places; a negative value that rounds to zero prints without the minus sign."""
    text = f"{x:.{decimals}f}"
    if text.startswith("-") and float(text) == 0.0:
        text = text[1:]
    return text


def fmt_est_ci_nz(est: float, lo: float, hi: float, decimals: int = 3) -> str:
    """fmt_est_ci for a table cell: no minus sign on a value that rounds to zero (CSV text only)."""
    return (f"{_no_negative_zero(est, decimals)} ({_no_negative_zero(lo, decimals)} to "
            f"{_no_negative_zero(hi, decimals)})")


def r6(x: float) -> float:
    """Round a float for a JSON value so the file is tidy and identical across runs."""
    return round(float(x), 6)


def pci(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The v1 two-sided 95% percentile interval (nan-aware), along axis 0."""
    return W.percentile_ci(values, axis=0)


# ---------------------------------------------------------------------------
# counts by stratum, straight from the masks (suppressed strata included)
# ---------------------------------------------------------------------------

def stratum_counts(mask: np.ndarray, ev: "R1.Events") -> Tuple[np.ndarray, np.ndarray]:
    """(at risk at the START of day d, composite events ON day d), each (31,) int64.

    The conditioning set at day d is {i in the stratum : exit_day[i] >= d}; this is the same set
    W.riskset_composition counts and the Aalen-Johansen at_risk array carries.
    """
    exit_in = ev.exit_day[mask]
    at = np.array([int((exit_in >= d).sum()) for d in range(DAYS)], dtype=np.int64)
    sel = mask & ev.is_event
    e1 = np.bincount(ev.pod_event[sel].astype(int), minlength=DAYS).astype(np.int64)
    if len(e1) != DAYS:
        raise R1.R1Error("a composite event day outside 0 to 30")
    return at, e1


def band_cells(at: np.ndarray, e1: np.ndarray) -> List[Dict[str, Any]]:
    """One dict per band: events, patient-days, suppression flag and (if shown) rate and interval."""
    cells = []
    for first, last, label, seg in BANDS:
        k = int(e1[first:last + 1].sum())
        pdays = int(at[first:last + 1].sum())
        cell: Dict[str, Any] = dict(label=label, seg=seg, events=k, patient_days=pdays,
                                    suppressed=k < MIN_EVENTS)
        if not cell["suppressed"]:
            cell["rate"], cell["lo"], cell["hi"] = rate_per_100k(k, pdays)
        cells.append(cell)
    return cells


# ---------------------------------------------------------------------------
# horizon-matched window
# ---------------------------------------------------------------------------

def h16_point(f: "R1.StratumFit") -> np.ndarray:
    """H16(t) for t = 0..14 as proportions: (CIF at start of day t+16 minus at start of day t) / S(t)."""
    return (f.cif[HORIZON:DAYS] - f.cif[0:H_DAYS]) / f.surv[0:H_DAYS]


def h16_boot(f: "R1.StratumFit") -> np.ndarray:
    """(B, 15) the same quantity on the same bootstrap replicates as f.boot.curve."""
    b = f.boot
    return (b.cif[:, HORIZON:DAYS] - b.cif[:, 0:H_DAYS]) / b.surv[:, 0:H_DAYS]


def day_text_h16(day: int) -> str:
    return H16_NOT_REACHED if day > H_DAYS - 1 else str(int(day))


def day_ci_text_h16(lo: float, hi: float) -> str:
    lo_i, hi_i = int(math.floor(lo)), int(math.ceil(hi))
    if lo_i > H_DAYS - 1:
        return H16_NOT_REACHED
    if hi_i > H_DAYS - 1:
        return f"{lo_i} to {H16_NOT_REACHED[0].lower() + H16_NOT_REACHED[1:]}"
    return f"{lo_i} to {hi_i}"


def stays_below_h16(h: np.ndarray, threshold: float, day: int) -> bool:
    """True if the 16-day curve (15 values) is below the threshold on every start day from `day` to 14."""
    if day > H_DAYS - 1:
        return False
    return bool(np.all(h[day:] < threshold))


def horizon_block(f: "R1.StratumFit") -> Dict[str, Any]:
    """Everything the horizon-matched comparison needs for one non-suppressed stratum."""
    h = h16_point(f)
    hb = h16_boot(f)
    r = f.curve[0:H_DAYS]
    rb = f.boot.curve[:, 0:H_DAYS]
    if not (np.all(np.isfinite(h)) and np.all(np.isfinite(hb)) and np.all(np.isfinite(rb))):
        raise R1.R1Error(f"non-finite residual risk in stratum {f.slug}")
    gap = r - h
    gapb = rb - hb
    out: Dict[str, Any] = dict(h=h, hb=hb, r=r, rb=rb, gap=gap, gapb=gapb)
    if float(gap.min()) < -1e-12 or float(gapb.min()) < -1e-12:
        raise R1.R1Error(f"R(t) < H16(t) in stratum {f.slug}: a longer window cannot carry less cumulative risk")
    # thresholds
    thr: Dict[float, Dict[str, Any]] = {}
    for t in THRESHOLDS:
        d_v1 = f.threshold_day(t)
        d_h = R1.threshold_day(h, t)
        bd_v1 = f.boot_threshold_days(t)
        bd_h = R1.threshold_day(hb, t)
        v_lo, v_hi = pci(bd_v1.astype(float))
        h_lo, h_hi = pci(bd_h.astype(float))
        thr[t] = dict(
            v1_day=d_v1, v1_lo=int(math.floor(v_lo)), v1_hi=int(math.ceil(v_hi)),
            v1_stays=R1.stays_below_from(f.curve, t, d_v1),
            h_day=d_h, h_lo=int(math.floor(h_lo)), h_hi=int(math.ceil(h_hi)),
            h_stays=stays_below_h16(h, t, d_h) if d_h <= H_DAYS - 1 else False,
            v1_ci_text=W.day_ci_text(v_lo, v_hi), h_ci_text=day_ci_text_h16(h_lo, h_hi))
    out["thr"] = thr
    # residual risk at the report days, both definitions, the gap, and the decline since day 0
    days: Dict[int, Dict[str, Any]] = {}
    for t in RISK_DAYS:
        rec: Dict[str, Any] = {}
        for tag, point, boot in (("v1", r, rb), ("h16", h, hb)):
            lo, hi = pci(100.0 * boot[:, t])
            rec[tag] = (100.0 * float(point[t]), float(lo), float(hi))
        g_lo, g_hi = pci(100.0 * gapb[:, t])
        rec["gap_pp"] = (100.0 * float(gap[t]), float(g_lo), float(g_hi))
        with np.errstate(divide="ignore", invalid="ignore"):
            share_b = np.where(rb[:, t] > 0, 100.0 * gapb[:, t] / rb[:, t], np.nan)
        s_lo, s_hi = pci(share_b)
        rec["gap_share_pct"] = (100.0 * float(gap[t] / r[t]), float(s_lo), float(s_hi))
        rec["gap_share_undefined_n"] = int(np.isnan(share_b).sum())
        days[t] = rec
        if t not in REPORT_DAYS:
            continue
        # decline since day 0
        d_r = 100.0 * (r[0] - r[t])
        d_h = 100.0 * (h[0] - h[t])
        d_rb = 100.0 * (rb[:, 0] - rb[:, t])
        d_hb = 100.0 * (hb[:, 0] - hb[:, t])
        rlo, rhi = pci(d_rb)
        hlo, hhi = pci(d_hb)
        part_b = d_rb - d_hb
        plo, phi = pci(part_b)
        with np.errstate(divide="ignore", invalid="ignore"):
            dshare_b = np.where(d_rb > 0, 100.0 * part_b / d_rb, np.nan)
        slo, shi = pci(dshare_b)
        rec["decline_v1_pp"] = (d_r, float(rlo), float(rhi))
        rec["decline_h16_pp"] = (d_h, float(hlo), float(hhi))
        rec["decline_gap_pp"] = (d_r - d_h, float(plo), float(phi))
        rec["decline_share_pct"] = (100.0 * (d_r - d_h) / d_r, float(slo), float(shi))
        rec["decline_share_undefined_n"] = int(np.isnan(dshare_b).sum())
    out["days"] = days
    return out


# ---------------------------------------------------------------------------
# comparison with v1's own Table 5 and Table 5B
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> Tuple[List[str], List[List[str]]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    return rows[0], rows[1:]


def check_against_v1_tables(fits: Sequence["R1.StratumFit"]) -> Dict[str, int]:
    """Every v1-definition number reported here must equal v1's Table 5 and Table 5B, cell for cell.

    Table 5 (table5_safe_discharge_day.csv): threshold day, day interval, residual risk at that day,
    its interval, patients at risk, threshold reached, stays below. Table 5B
    (table5b_conditional_risk.csv): at-risk count, conditional risk, lower, upper, cumulative
    incidence at the start of day, for every day 0..30. Strings are compared exactly.
    """
    tables = R1.ORIG_TABLES
    head5, rows5 = _read_csv(tables / "table5_safe_discharge_day.csv")
    head5b, rows5b = _read_csv(tables / "table5b_conditional_risk.csv")
    i5 = {n: head5.index(n) for n in head5}
    i5b = {n: head5b.index(n) for n in head5b}
    col_strat, col_level = head5[0], head5[1]
    by5 = {(r[0], r[1], r[i5["Risk threshold"]]): r for r in rows5}
    by5b = {(r[i5b["Stratifier"]], r[i5b["Stratum"]], int(r[i5b["Postoperative day"]])): r for r in rows5b}
    n5 = n5b = 0
    for f in fits:
        s = f.stratum
        if f.suppressed:
            row = by5[(s.stratifier, s.level, SUPPRESSED)]       # v1 writes ONE row for a suppressed stratum
            assert row[i5["Residual-risk threshold day"]] == SUPPRESSED, f"Table 5 suppressed row {f.slug}"
            assert row[i5["Composite events"]] == VD.fmt_count(f.events), f"Table 5 suppressed events {f.slug}"
            n5 += 2
            continue
        for t in THRESHOLDS:
            row = by5[(s.stratifier, s.level, VD.threshold_label(t))]
            day = f.threshold_day(t)
            d_lo, d_hi = pci(f.boot_threshold_days(t).astype(float))
            reached = day <= VD.FOLLOWUP_DAYS
            want = {
                "Residual-risk threshold day": W.day_text(day),
                "Residual-risk threshold day, 95% confidence interval": W.day_ci_text(d_lo, d_hi),
                "Residual risk at that day (%)": W.pct(float(f.curve[day])) if reached else W.NOT_REACHED,
                "Threshold reached within 30 days": "Yes" if reached else "No",
                "Residual risk remains below the threshold through day 30":
                    ("Yes" if W.stays_below(f.curve, t, day) else "No") if reached else W.NOT_APPLICABLE,
            }
            for name, text in want.items():
                got = row[i5[name]]
                assert got == text, f"Table 5 mismatch {f.slug} {t} {name!r}: v1 {got!r} vs mine {text!r}"
                n5 += 1
            if reached:
                ci_name = "Residual risk, 95% confidence interval (%)"
                ci_text = W.pct_ci(float(f.boot.lo[day]), float(f.boot.hi[day]))
                assert row[i5[ci_name]] == ci_text, f"Table 5 risk CI mismatch {f.slug} {t}"
                n5 += 1
        for d in range(DAYS):
            row = by5b[(s.stratifier, s.level, d)]
            checks = {
                "Patients event free and alive at the start of the day": str(int(round(float(f.at_risk[d])))),
                "Conditional risk of a subsequent composite event (%)": W.num(100.0 * float(f.curve[d])),
                "95% confidence interval, lower (%)": W.num(100.0 * float(f.boot.lo[d])),
                "95% confidence interval, upper (%)": W.num(100.0 * float(f.boot.hi[d])),
                "Cumulative incidence of the composite to the start of the day (%)": W.num(100.0 * float(f.cif[d])),
            }
            for name, text in checks.items():
                assert row[i5b[name]] == text, f"Table 5B mismatch {f.slug} day {d} {name!r}"
                n5b += 1
    return {"table5_cells_n": n5, "table5b_cells_n": n5b}


# ---------------------------------------------------------------------------
# figure S3
# ---------------------------------------------------------------------------

def _spread_labels(ys: Sequence[float], min_ratio: float) -> List[float]:
    """Nudge label heights apart on a log axis so direct labels do not collide (order preserved)."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    pos = [None] * len(ys)
    prev = None
    for i in order:
        y = ys[i]
        if prev is not None and y < prev * min_ratio:
            y = prev * min_ratio
        pos[i] = y
        prev = y
    return pos


def draw_figure(path: Path, by_slug: Dict[str, Dict[str, Any]], hblocks: Dict[str, Dict[str, Any]],
                fits_by: Dict[str, "R1.StratumFit"]) -> List[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    R1.style_figure(base_pt=8.0)
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.6, 3.55))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.955, bottom=0.15, wspace=0.42)

    # ---- panel A: band hazards, log y --------------------------------------------------------
    xs = np.arange(len(BANDS))
    tick_labels = ["0", "1", "2", "3", "4-6", "7-13", "14-30"]
    age_slugs = ["age.18-49", "age.50-64", "age.65-74", "age.ge75"]
    right_labels: List[Tuple[float, str, str]] = []
    # overall with exact Poisson intervals
    ov = by_slug["overall"]["cells"]
    ov_rate = np.array([c["rate"] for c in ov])
    ov_lo = np.array([c["lo"] for c in ov])
    ov_hi = np.array([c["hi"] for c in ov])
    axa.errorbar(xs, ov_rate, yerr=[ov_rate - ov_lo, ov_hi - ov_rate], fmt="none", ecolor=R1.INK_MUTED,
                 elinewidth=0.9, capsize=2.2, capthick=0.9, zorder=3)
    axa.plot(xs, ov_rate, color=R1.INK_MUTED, lw=1.0, linestyle=R1.OVERALL_LINESTYLE, marker="o", ms=3.4,
             markerfacecolor="white", markeredgecolor=R1.INK_MUTED, zorder=4)
    right_labels.append((float(ov_rate[-1]), "All patients", R1.INK_MUTED))
    for i, slug in enumerate(age_slugs):
        cells = by_slug[slug]["cells"]
        y = np.array([c["rate"] if not c["suppressed"] else np.nan for c in cells])
        style, marker = R1.SERIES_STYLE[i]
        col = R1.BLUE_STEP[i]
        axa.plot(xs, y, color=col, lw=1.4, linestyle=style, marker=marker, ms=3.4, zorder=5)
        last_ok = [j for j in range(len(cells)) if not cells[j]["suppressed"]]
        if last_ok and last_ok[-1] == len(cells) - 1:
            right_labels.append((float(y[-1]), fits_by[slug].stratum.level, col))
    axa.set_yscale("log")
    axa.set_ylim(1.2, 700.0)
    axa.set_yticks([2, 5, 10, 20, 50, 100, 200, 500])
    axa.set_yticklabels(["2", "5", "10", "20", "50", "100", "200", "500"])
    axa.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axa.set_xlim(-0.35, 8.55)
    axa.set_xticks(xs)
    axa.set_xticklabels(tick_labels)
    axa.set_xlabel("Postoperative day (band)")
    axa.set_ylabel("Events per 100,000 patient-days (log scale)")
    axa.grid(axis="y", which="major", alpha=0.45)
    ys = [v for v, _, _ in right_labels]
    placed = _spread_labels(ys, 1.30)
    # the "All patients" and 50-64 labels sit one spacing step apart and nearly touch: move the 50-64 label down
    label_50_64 = fits_by["age.50-64"].stratum.level
    placed = [y / 1.20 if text == label_50_64 else y for (_v, text, _c), y in zip(right_labels, placed)]
    for (v, text, col), y_lab in zip(right_labels, placed):
        axa.annotate(text, xy=(xs[-1], v), xytext=(xs[-1] + 0.28, y_lab), textcoords="data",
                     fontsize=7.0, color=col if col != R1.BLUE_STEP[0] else "#3c78c4", va="center", ha="left",
                     arrowprops=dict(arrowstyle="-", color=col, lw=0.5, shrinkA=0, shrinkB=1.5))
    axa.text(-0.12, 1.02, "A", transform=axa.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right")

    # ---- panel B: residual risk, v1 definition against the 16-day window --------------------
    days = np.arange(H_DAYS)
    groups = [("overall", R1.INK_MUTED, "All patients"),
              ("age.ge75", R1.BLUE_STEP[3], None),
              ("age.18-49", R1.BLUE_STEP[1], None)]
    label_pts: List[Tuple[float, str, str]] = []
    for slug, col, name in groups:
        blk = hblocks[slug]
        label = name or fits_by[slug].stratum.level
        r = 100.0 * blk["r"]
        h = 100.0 * blk["h"]
        axb.step(days, r, where="post", color=col, lw=1.5, linestyle="-", zorder=4)
        axb.step(days, h, where="post", color=col, lw=1.2, linestyle=(0, (2.2, 1.6)), zorder=4)
        axb.plot(days[::5], h[::5], lw=0, marker="o", ms=3.4, markerfacecolor="white", markeredgecolor=col,
                 zorder=5)
        axb.plot(days[::5], r[::5], lw=0, marker="o", ms=3.4, color=col, zorder=5)
        label_pts.append((float(r[-1]), label, col))
    for t in (0.01, 0.005, 0.0025):
        axb.hlines(100.0 * t, -0.4, days[-1] + 0.45, color=R1.INK_MUTED, lw=0.7,
                   linestyle=R1.THRESHOLD_LINESTYLE, zorder=2)
    axb.set_yscale("log")
    axb.set_xlim(-0.4, 20.4)
    axb.set_ylim(0.017, 3.6)
    axb.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
    axb.set_xlabel("Postoperative day t")
    axb.set_ylabel("Residual risk (%, log scale)")
    axb.set_yticks([0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5])
    axb.set_yticklabels(["0.02", "0.05", "0.1", "0.25", "0.5", "1", "2.5"])
    axb.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axb.grid(False)
    placed_b = _spread_labels([v for v, _, _ in label_pts], 1.32)
    for (v, text, col), y_lab in zip(label_pts, placed_b):
        axb.text(days[-1] + 1.1, y_lab, text, fontsize=7.0,
                 color=col if col != R1.BLUE_STEP[1] else "#3c78c4", va="center", ha="left")
    handles = [Line2D([0], [0], color=R1.INK_MUTED, lw=1.5, linestyle="-", marker="o", ms=3.4,
                      label="Through day 30 (Figure 6 definition)"),
               Line2D([0], [0], color=R1.INK_MUTED, lw=1.2, linestyle=(0, (2.2, 1.6)), marker="o", ms=3.4,
                      markerfacecolor="white", label="Fixed 16-day window")]
    axb.legend(handles=handles, loc="lower left", fontsize=6.8, handlelength=2.6, borderaxespad=0.2)
    axb.text(-0.25, 1.0 * 1.07, "1%", fontsize=6.5, color=R1.INK_MUTED, ha="left", va="bottom")
    axb.text(-0.25, 0.5 * 1.07, "0.5%", fontsize=6.5, color=R1.INK_MUTED, ha="left", va="bottom")
    axb.text(-0.25, 0.25 * 0.93, "0.25%", fontsize=6.5, color=R1.INK_MUTED, ha="left", va="top")
    axb.text(-0.12, 1.02, "B", transform=axb.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right")
    return R1.save_png(fig, path)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="A3: discrete-time hazard and the horizon-matched window.")
    ap.parse_args(argv)
    out_dir, fig_dir, json_path = R1.TABLES_OUT, R1.FIGURES_OUT, R1.FRAGMENTS_OUT / "results_a3.json"

    df = R1.load_cohort()
    ev = R1.make_events(df, "WINDOW_COMP", "POD_WINDOW")
    assert ev.n_events == 1015 and ev.n_untimed_events == 0, "composite is not 1,015 timed events"
    fits = R1.fit_overall_and_strata(df, ev)            # reseeds with VD.SEED; every draw happens here
    assert [f.slug for f in fits] == ["overall", "age.18-49", "age.50-64", "age.65-74", "age.ge75",
                                      "asa.1-2", "asa.ge3", "levels.1", "levels.2", "levels.ge3",
                                      "sex.male", "sex.female", "sex.not-recorded"]
    fits_by = R1.index_by_slug(fits)

    # 0. v1-definition numbers equal v1's Table 5 and Table 5B
    tcheck = check_against_v1_tables(fits)

    # 1. band hazards, every stratum (suppressed strata included, counts only)
    by_slug: Dict[str, Dict[str, Any]] = {}
    for f in fits:
        at, e1 = stratum_counts(f.mask, ev)
        assert at[0] == f.n and int(e1.sum()) == f.events, f"counts {f.slug}"
        if not f.suppressed:
            assert np.array_equal(at, np.rint(f.at_risk).astype(np.int64)), f"at-risk {f.slug}"
            assert np.array_equal(e1, f.e1.astype(np.int64)), f"events by day {f.slug}"
        cells = band_cells(at, e1)
        assert sum(c["events"] for c in cells) == f.events, f"band events do not sum to events {f.slug}"
        assert sum(c["patient_days"] for c in cells) == int(at.sum()), f"band patient-days {f.slug}"
        first, last, _, _ = BANDS[0]
        d0, dl = cells[0], cells[-1]
        share = None
        if not d0["suppressed"] and not dl["suppressed"]:
            share = 100.0 * dl["rate"] / d0["rate"]
        by_slug[f.slug] = dict(at=at, e1=e1, cells=cells, share_last_of_day0_pct=share)
    assert by_slug["overall"]["e1"].sum() == 1015

    # 2. horizon-matched comparison, every non-suppressed stratum
    hblocks = {f.slug: horizon_block(f) for f in fits if not f.suppressed}

    # ---------------- results_a3.json -----------------------------------------------------------
    res: Dict[str, Any] = {}
    res["a3.cohort.n"] = len(df)
    res["a3.composite.events_n"] = ev.n_events
    res["a3.horizon.window_days_n"] = HORIZON
    res["a3.horizon.start_days_n"] = H_DAYS
    res["a3.hazard.min_cell_events_n"] = MIN_EVENTS
    res["a3.check.table5_cells_n"] = tcheck["table5_cells_n"]
    res["a3.check.table5b_cells_n"] = tcheck["table5b_cells_n"]
    res["a3.check.v1_definition_equals_tables"] = True
    for f in fits:
        s = f.slug
        info = by_slug[s]
        res[f"a3.hazard.{s}.patients_n"] = f.n
        res[f"a3.hazard.{s}.events_n"] = f.events
        res[f"a3.hazard.{s}.suppressed"] = bool(f.suppressed)
        for c in info["cells"]:
            p = f"a3.hazard.{s}.{c['seg']}"
            res[f"{p}.events_n"] = c["events"]
            res[f"{p}.patient_days_n"] = c["patient_days"]
            res[f"{p}.suppressed"] = bool(c["suppressed"])
            if not c["suppressed"]:
                res[f"{p}.rate_per100k"] = r6(c["rate"])
                res[f"{p}.rate_per100k_lo"] = r6(c["lo"])
                res[f"{p}.rate_per100k_hi"] = r6(c["hi"])
        if info["share_last_of_day0_pct"] is not None:
            res[f"a3.hazard.{s}.last_band_share_of_day0_pct"] = r6(info["share_last_of_day0_pct"])
    for f in fits:
        if f.suppressed:
            res[f"a3.horizon.{f.slug}.suppressed"] = True
            continue
        s = f.slug
        blk = hblocks[s]
        res[f"a3.horizon.{s}.suppressed"] = False
        for t in THRESHOLDS:
            tt = blk["thr"][t]
            p = f"a3.horizon.{s}.{R1.threshold_slug(t)}"
            res[f"{p}.v1_day"] = R1.day_or_none(tt["v1_day"])
            res[f"{p}.v1_day_lo"] = R1.day_or_none(tt["v1_lo"])
            res[f"{p}.v1_day_hi"] = R1.day_or_none(tt["v1_hi"])
            res[f"{p}.v1_stays_below"] = bool(tt["v1_stays"])
            res[f"{p}.h16_day"] = None if tt["h_day"] > H_DAYS - 1 else int(tt["h_day"])
            res[f"{p}.h16_day_lo"] = None if tt["h_lo"] > H_DAYS - 1 else int(tt["h_lo"])
            res[f"{p}.h16_day_hi"] = None if tt["h_hi"] > H_DAYS - 1 else int(tt["h_hi"])
            res[f"{p}.h16_stays_below"] = bool(tt["h_stays"])
            if tt["v1_day"] <= VD.FOLLOWUP_DAYS and tt["h_day"] <= H_DAYS - 1:
                res[f"{p}.day_difference_n"] = int(tt["v1_day"] - tt["h_day"])
        for t in RISK_DAYS:
            rec = blk["days"][t]
            p = f"a3.horizon.{s}.day{t}"
            for tag, key in (("v1", "v1_risk"), ("h16", "h16_risk")):
                est, lo, hi = rec[tag]
                res[f"{p}.{key}_pct"] = r6(est)
                res[f"{p}.{key}_lo_pct"] = r6(lo)
                res[f"{p}.{key}_hi_pct"] = r6(hi)
            parts = [("gap_pp", "gap", "_pp"), ("gap_share_pct", "gap_share", "_pct")]
            if t in REPORT_DAYS:
                parts += [("decline_v1_pp", "decline_v1", "_pp"), ("decline_h16_pp", "decline_h16", "_pp"),
                          ("decline_gap_pp", "decline_gap", "_pp"), ("decline_share_pct", "decline_share", "_pct")]
            for src, base, suffix in parts:
                est, lo, hi = rec[src]
                res[f"{p}.{base}{suffix}"] = r6(est)
                res[f"{p}.{base}_lo{suffix}"] = r6(lo)
                res[f"{p}.{base}_hi{suffix}"] = r6(hi)
            res[f"{p}.gap_share_bootstrap_undefined_n"] = rec["gap_share_undefined_n"]
            if t in REPORT_DAYS:
                res[f"{p}.decline_share_bootstrap_undefined_n"] = rec["decline_share_undefined_n"]
    json_n = R1.write_fragment(json_path, res, ("a3.",))

    # ---------------- Table S7A: band hazards ----------------------------------------------------
    head_a = ["Stratifier", "Level", "Postoperative days", "Events, n", "Patient-days at risk, n",
              "Events per 100,000 patient-days (95% CI)",
              "Rate in days 14 to 30 as a percentage of the day 0 rate"]
    rows_a: List[List[Any]] = []
    for f in fits:
        info = by_slug[f.slug]
        for c in info["cells"]:
            is_last = c["seg"] == LAST_BAND[3]
            if c["suppressed"]:
                rate_txt = SUPPRESSED
            else:
                rate_txt = fmt_rate(c["rate"], c["lo"], c["hi"])
            if is_last:
                share_txt = (f"{info['share_last_of_day0_pct']:.1f}"
                             if info["share_last_of_day0_pct"] is not None else SUPPRESSED)
            else:
                share_txt = ""
            rows_a.append([f.stratum.stratifier, f.stratum.level, c["label"], VD.fmt_count(c["events"]),
                           VD.fmt_count(c["patient_days"]), rate_txt, share_txt])
    R1.write_csv_table(out_dir / "tableS7_hazard_bands.csv", head_a, rows_a)

    # ---------------- Table S7B: horizon-matched window ------------------------------------------
    head_b = ["Stratifier", "Level", "Measure", "Patients event free and alive at the start of the day",
              "Through day 30 (Figure 6 definition)",
              "Fixed 16-day window", "Difference, through day 30 minus 16-day window",
              "Difference as a percentage of the through day 30 value",
              "Stays below the threshold through day 30", "Stays below the threshold through day 14"]
    rows_b: List[List[Any]] = []
    for f in fits:
        s = f.stratum
        if f.suppressed:
            rows_b.append([s.stratifier, s.level,
                           f"All measures ({VD.fmt_count(f.n)} patients, {VD.fmt_count(f.events)} composite events)",
                           VD.fmt_count(int(by_slug[f.slug]["at"][0])),
                           SUPPRESSED, SUPPRESSED, SUPPRESSED, SUPPRESSED, SUPPRESSED, SUPPRESSED])
            continue
        blk = hblocks[f.slug]
        at_f = by_slug[f.slug]["at"]
        for t in THRESHOLDS:
            tt = blk["thr"][t]
            v1_txt = (f"{W.day_text(tt['v1_day'])} ({tt['v1_ci_text']})")
            h_txt = (f"{day_text_h16(tt['h_day'])} ({tt['h_ci_text']})")
            if tt["v1_day"] <= VD.FOLLOWUP_DAYS and tt["h_day"] <= H_DAYS - 1:
                diff_txt = str(int(tt["v1_day"] - tt["h_day"]))
            else:
                diff_txt = NOT_APPLICABLE
            sb1 = ("Yes" if tt["v1_stays"] else "No") if tt["v1_day"] <= VD.FOLLOWUP_DAYS else NOT_APPLICABLE
            sbh = ("Yes" if tt["h_stays"] else "No") if tt["h_day"] <= H_DAYS - 1 else NOT_APPLICABLE
            rows_b.append([s.stratifier, s.level, f"Threshold day, {R1.threshold_text(t)}", NOT_APPLICABLE,
                           v1_txt, h_txt, diff_txt, NOT_APPLICABLE, sb1, sbh])
        for t in RISK_DAYS:
            rec = blk["days"][t]
            rows_b.append([s.stratifier, s.level, f"Residual risk at day {t}, %", VD.fmt_count(int(at_f[t])),
                           fmt_est_ci_nz(*rec["v1"]), fmt_est_ci_nz(*rec["h16"]),
                           fmt_est_ci_nz(*rec["gap_pp"]), fmt_est_ci_nz(*rec["gap_share_pct"], decimals=1),
                           NOT_APPLICABLE, NOT_APPLICABLE])
        for t in REPORT_DAYS:
            rec = blk["days"][t]
            rows_b.append([s.stratifier, s.level, f"Decline in residual risk from day 0 to day {t}, percentage points",
                           VD.fmt_count(int(at_f[0])),
                           fmt_est_ci_nz(*rec["decline_v1_pp"]), fmt_est_ci_nz(*rec["decline_h16_pp"]),
                           fmt_est_ci_nz(*rec["decline_gap_pp"]), fmt_est_ci_nz(*rec["decline_share_pct"], decimals=1),
                           NOT_APPLICABLE, NOT_APPLICABLE])
    R1.write_csv_table(out_dir / "tableS7_horizon_matched.csv", head_b, rows_b)

    # ---------------- Figure S3 -------------------------------------------------------------------
    fig_paths = (draw_figure(fig_dir / "figS3_hazard_and_horizon.png", by_slug, hblocks, fits_by)
                 if R1.FIGURES else [])

    print(f"[a3] wrote {json_n} keys, 2 tables, {len(fig_paths)} figure files; "
          f"Table 5 cells {tcheck['table5_cells_n']}, "
          f"Table 5B cells {tcheck['table5b_cells_n']} equal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
