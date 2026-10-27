#!/usr/bin/env python3
"""a2_component_residual.py - A2, component-specific residual risk.

Residual risk is estimated separately for hematoma and for airway morbidity, so that the
composite can be interpreted by its parts. Three outcomes are run through the SAME v1
Aalen-Johansen estimator (death competing), overall and in the four locked stratifiers (13 strata):

  1. hematoma requiring reoperation     HEMATOMA      / HEMATOMA_POD
  2. composite airway morbidity         AIRWAY_COMP   / POD_AIRWAY
  3. the primary composite              WINDOW_COMP   / POD_WINDOW   (carried for direct comparison)

No model is fitted. Thresholds are 0.5%, 0.25% and 0.10% (the 1% level is crossed at day 0 by every
outcome here and carries no information); 0.10% is defined locally because v1's
VD.threshold_label() rejects it. Overall hematoma is ALREADY below 0.5% at day 0 (its 30-day
incidence is 0.479%); that is a fact about the outcome, not an error, and the table says so.

DESIGN (documented because the bootstrap draw order is part of the result)
  Each outcome is fitted by ONE call of r1_common.fit_overall_and_strata(), which reseeds the global
  NumPy stream with VD.SEED = 20260731 and then draws exactly one W.bootstrap_cells() block (1,000
  multinomial replicates) per non-suppressed stratum, strata in W.build_strata order. The outcomes
  are run in the order hematoma, airway, composite, and because every call reseeds, no outcome's
  draws depend on another's. The composite's draws are therefore v1's own, bit for bit, and the
  script ASSERTS that its point estimates, intervals and bootstrap threshold-day intervals equal
  v1 Table 5 and Table 5B cell for cell, through the same formatting this script uses for Table S6A.

UNTIMED HEMATOMA. Exactly one patient has HEMATOMA = 1 and no HEMATOMA_POD (the raw RETORPODAYS is
  missing). v1's 03_timing.py convention (has_event = flag and finite day) is used as the primary
  handling: the event is not counted and the patient stays in the denominator as event free
  (r1_common untimed="drop"). That removes one event of 541 (at most a 0.18% relative understatement
  of residual risk), so it is NOT conservative for a safety claim; the sensitivity analyses place the
  event on day 0, day 30 (and day 1, the day the patient's airway event was recorded) and rerun all
  13 strata.

SCOPE OF THE MAKEUP TABLE. The composite makeup table reports counts only (events by component and
  the hematoma-only, airway-only and both counts that v1 06_window.py already publishes). It reports
  no rate of any outcome among hematoma patients, no order of events, no percentage conditional on
  hematoma status and no length-of-stay comparison.

Needs outputs/results_03.json, outputs/results_06.json and the Table 5 and Table 5B CSVs of the
analysis folder (written by steps 03 and 06).

Writes (under the results directory): fragments/results_a2.json (namespace a2.),
  tables-csv/tableS6_component_residual_risk.csv, tables-csv/tableS6_composite_makeup.csv,
  figures/figS2_component_residual_risk.png and .pdf (skipped when ACDF_FIGURES=0).
Writes no patient-level data.

    PYTHONDONTWRITEBYTECODE=1 python3 a2_component_residual.py
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import r1_common as R1  # noqa: E402

VD, W = R1.VD, R1.W

THRESHOLDS: Tuple[float, ...] = (0.005, 0.0025, 0.001)     # 0.5%, 0.25%, 0.10% (defined locally)
REPORT_DAYS: Tuple[int, ...] = (0, 1, 3, 7, 14)
PREFIX = ("a2.",)

# Display name of the primary composite in every table and figure of this script. The v1 label layer calls it
# "Composite hematoma or airway outcome"; the paper says "primary composite outcome".
COMPOSITE_LABEL = "Primary composite outcome"

# (fragment key, event flag, day column, treatment of an event without a day, display label)
OUTCOMES: Tuple[Tuple[str, str, str, str], ...] = (
    ("hematoma", "HEMATOMA", "HEMATOMA_POD", "drop"),
    ("airway", "AIRWAY_COMP", "POD_AIRWAY", "raise"),
    ("composite", "WINDOW_COMP", "POD_WINDOW", "raise"),
)
AGE_SLUGS = ("age.18-49", "age.50-64", "age.65-74", "age.ge75")

SUPPRESSED = R1.suppressed_text
NOT_REACHED = W.NOT_REACHED

S6A_HEADER: List[str] = (
    ["Outcome", "Stratifier", "Stratum", "Patients", "Events", "denominator",
     "30-day cumulative incidence (%)", "30-day cumulative incidence, 95% confidence interval (%)"]
    + [h for t in THRESHOLDS for h in (
        f"Residual-risk threshold day at {R1.threshold_text(t)}",
        f"Residual-risk threshold day at {R1.threshold_text(t)}, 95% confidence interval",
        f"Residual risk remains below {R1.threshold_text(t)} through day 30")]
    + [h for d in REPORT_DAYS for h in (
        f"Residual risk at day {d} (%)",
        f"Residual risk at day {d}, 95% confidence interval (%)")]
    + ["Suppressed", "Note"]
)

S6B_HEADER: List[str] = ["Section", "Category", "Patients", "denominator",
                         "Share of all patients (%)", "Note"]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    print(f"[a2] {msg}", flush=True)


def need(cond: bool, msg: str) -> None:
    """A hard assertion that survives python -O and names the failure."""
    if not cond:
        raise SystemExit(f"[a2] ASSERTION FAILED: {msg}")


def rnd(x: float, places: int = 6) -> float:
    return round(float(x), places)


def day_bounds(lo: float, hi: float) -> Tuple[Optional[int], Optional[int]]:
    """Floor of the lower and ceiling of the upper percentile (the v1 W.day_ci_text rule);
    None when that bound lies past day 30 (not reached within 30 days)."""
    lo_i, hi_i = int(math.floor(lo)), int(math.ceil(hi))
    return (None if lo_i > VD.FOLLOWUP_DAYS else lo_i,
            None if hi_i > VD.FOLLOWUP_DAYS else hi_i)


def threshold_info(fit: R1.StratumFit, t: float) -> Dict[str, Any]:
    """Threshold day (first crossing), its bootstrap interval and the stays-below flag, one stratum."""
    day = fit.threshold_day(t)
    lo, hi = W.percentile_ci(fit.boot_threshold_days(t).astype(float))
    reached = day <= VD.FOLLOWUP_DAYS
    info: Dict[str, Any] = {
        "day": day, "reached": reached, "lo": float(lo), "hi": float(hi),
        "day_text": W.day_text(day), "ci_text": W.day_ci_text(lo, hi),
        "stays": R1.stays_below_from(fit.curve, t, day) if reached else None,
    }
    return info


# ---------------------------------------------------------------------------
# step 1: load, build events, fit
# ---------------------------------------------------------------------------


def build_events(df) -> Dict[str, R1.Events]:
    ev: Dict[str, R1.Events] = {}
    for key, flag, pod, untimed in OUTCOMES:
        ev[key] = R1.make_events(df, flag, pod, untimed=untimed)
    return ev


def check_partition(key: str, fits: Sequence[R1.StratumFit], n_total: int) -> None:
    """The four stratifiers each partition the cohort, the events and the competing deaths."""
    head = fits[0]
    need(head.slug == "overall" and head.n == n_total, f"{key}: first stratum is not the overall row")
    for column in VD.STRATIFIER_COLUMNS:
        members = [f for f in fits if f.stratum.column == column]
        need(sum(f.n for f in members) == n_total, f"{key}/{column}: patients do not partition the cohort")
        need(sum(f.events for f in members) == head.events,
             f"{key}/{column}: events {sum(f.events for f in members)} do not sum to {head.events}")
        need(sum(f.competing for f in members) == head.competing,
             f"{key}/{column}: competing deaths do not sum to {head.competing}")


# ---------------------------------------------------------------------------
# step 2: the composite must equal v1 Table 5 and Table 5B exactly
# ---------------------------------------------------------------------------


def table5_cells(fit: R1.StratumFit, t: float) -> List[str]:
    """One v1 Table 5 row (06_window.py table5_rows, lines 836 to 870), built from a StratumFit."""
    s = fit.stratum
    info = threshold_info(fit, t)
    if info["reached"]:
        day = info["day"]
        risk = W.pct(float(fit.curve[day]))
        ci = W.pct_ci(float(fit.boot.lo[day]), float(fit.boot.hi[day]))
        at_risk = VD.fmt_count(round(float(fit.at_risk[day])))
        stays = "Yes" if info["stays"] else "No"
    else:
        risk, ci, at_risk, stays = NOT_REACHED, NOT_REACHED, W.NOT_APPLICABLE, W.NOT_APPLICABLE
    return [s.stratifier, s.level, VD.fmt_count(fit.n), VD.fmt_count(fit.events), VD.fmt_count(fit.n),
            W.pct(fit.cif30), VD.threshold_label(t), info["day_text"], info["ci_text"],
            risk, ci, at_risk, "Yes" if info["reached"] else "No", stays]


def read_csv_rows(path: Path) -> List[List[str]]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.reader(fh))


def assert_composite_equals_v1(fits_c: Sequence[R1.StratumFit]) -> Dict[str, int]:
    """Composite point estimates, intervals and bootstrap threshold-day intervals equal v1 Table 5
    (every cell, thresholds 1%, 0.5%, 0.25%) and Table 5B (conditional risk, band and cumulative
    incidence to 6 decimals, patients at risk), read from the analysis folder's tables-csv.
    Returns the cell counts."""
    rows5 = read_csv_rows(W.TABLE5_CSV)
    need(rows5[0] == W.TABLE5_HEADER, "Table 5 CSV header differs from W.TABLE5_HEADER")
    disk5 = {(r[0], r[1], r[6]): r for r in rows5[1:]}
    cells5 = 0
    rows_seen = 0
    for fit in fits_c:
        s = fit.stratum
        if fit.suppressed:
            r = disk5.get((s.stratifier, s.level, SUPPRESSED))
            need(r is not None, f"Table 5 has no suppressed row for {s.level}")
            need(r[2] == VD.fmt_count(fit.n) and r[3] == VD.fmt_count(fit.events),
                 f"Table 5 suppressed counts differ for {s.level}")
            cells5 += 2
            rows_seen += 1
            continue
        for t in VD.RISK_THRESHOLDS:                 # v1's own three thresholds
            mine = table5_cells(fit, t)
            theirs = disk5.get((s.stratifier, s.level, VD.threshold_label(t)))
            need(theirs is not None, f"Table 5 has no row for {s.level} at {VD.threshold_label(t)}")
            for col, (a, b) in enumerate(zip(mine, theirs)):
                need(a == b, f"Table 5 cell differs: {s.level}, {VD.threshold_label(t)}, column "
                             f"{W.TABLE5_HEADER[col]!r}: mine {a!r}, v1 {b!r}")
            cells5 += len(mine)
            rows_seen += 1
    need(rows_seen == len(rows5) - 1, f"Table 5 rows compared {rows_seen} of {len(rows5) - 1}")

    rows5b = read_csv_rows(W.TABLE5B_CSV)
    need(rows5b[0] == W.TABLE5B_HEADER, "Table 5B CSV header differs from W.TABLE5B_HEADER")
    disk5b = {(r[1], r[2], int(r[3])): r for r in rows5b[1:]}
    cells5b = 0
    for fit in fits_c:
        s = fit.stratum
        for d in range(R1.DAYS):
            r = disk5b[(s.stratifier, s.level, d)]
            if fit.suppressed:
                need(r[8] == "" and r[12] == "Yes", f"Table 5B suppressed row differs: {s.level} day {d}")
                continue
            mine = [str(int(round(float(fit.at_risk[d])))), W.num(100.0 * float(fit.curve[d])),
                    W.num(100.0 * float(fit.boot.lo[d])), W.num(100.0 * float(fit.boot.hi[d])),
                    W.num(100.0 * float(fit.cif[d])), "No"]
            theirs = [r[7], r[8], r[9], r[10], r[11], r[12]]
            need(mine == theirs, f"Table 5B differs: {s.level} day {d}: mine {mine}, v1 {theirs}")
            cells5b += len(mine)
    return {"table5_cells": cells5, "table5b_cells": cells5b}


# ---------------------------------------------------------------------------
# step 3: composite makeup (counts only, companion-paper boundary)
# ---------------------------------------------------------------------------


def makeup_counts(df) -> Dict[str, int]:
    h = (df["HEMATOMA"] == 1).to_numpy()
    a = (df["AIRWAY_COMP"] == 1).to_numpy()
    c = (df["WINDOW_COMP"] == 1).to_numpy()
    ri = (df["REINTUBATION"] == 1).to_numpy()
    v4 = (df["VENT48"] == 1).to_numpy()
    tr = (df["REOP_TRACH"] == 1).to_numpy()
    need(bool(np.array_equal(c, h | a)), "WINDOW_COMP is not exactly HEMATOMA or AIRWAY_COMP row by row")
    need(bool(np.array_equal(a, ri | v4 | tr)),
         "AIRWAY_COMP is not exactly reintubation or ventilator or tracheostomy row by row")
    out = {
        "patients_n": int(len(df)),
        "hematoma_n": int(h.sum()), "reintubation_n": int(ri.sum()), "vent48_n": int(v4.sum()),
        "reop_trach_n": int(tr.sum()), "airway_n": int(a.sum()), "composite_n": int(c.sum()),
        "hematoma_and_airway_n": int((h & a).sum()),
        "hematoma_only_n": int((h & ~a).sum()), "airway_only_n": int((a & ~h).sum()),
    }
    need(out["hematoma_only_n"] + out["airway_only_n"] + out["hematoma_and_airway_n"] == out["composite_n"],
         "hematoma only + airway only + both does not equal the composite")
    return out


def check_makeup_against_v1(mk: Dict[str, int]) -> None:
    """The counts equal what v1 06_window.py published (results_06.json, read only)."""
    v1 = json.loads(W.RESULTS_JSON.read_text(encoding="utf-8"))
    pairs = {"hematoma_n": "hematoma", "reintubation_n": "reintub", "vent48_n": "vent48",
             "reop_trach_n": "trach", "airway_n": "airway", "hematoma_and_airway_n": "hematoma_and_airway"}
    for mine, theirs in pairs.items():
        need(v1[f"window.component.{theirs}.n"] == mk[mine],
             f"makeup {mine} {mk[mine]} differs from v1 results_06 window.component.{theirs}.n")


# ---------------------------------------------------------------------------
# step 4: the untimed hematoma and its sensitivity
# ---------------------------------------------------------------------------


def untimed_block(df, fits_primary: Sequence[R1.StratumFit], ev_h: R1.Events
                  ) -> Tuple[Dict[str, Any], List[Tuple[str, R1.StratumFit]]]:
    """Fragment keys for the one untimed hematoma and the sensitivity fits (day 0, 1, 30)."""
    frag: Dict[str, Any] = {}
    flagged = (df["HEMATOMA"] == 1).to_numpy()
    untimed_mask = flagged & ~np.isfinite(df["HEMATOMA_POD"].to_numpy(dtype=float, na_value=np.nan))
    need(int(untimed_mask.sum()) == ev_h.n_untimed_events == 1,
         f"expected exactly one untimed hematoma, found {int(untimed_mask.sum())}")
    base = "a2.hematoma"
    frag[f"{base}.untimed_n"] = ev_h.n_untimed_events
    frag[f"{base}.flagged_n"] = ev_h.n_event_flagged
    frag[f"{base}.timed_events_n"] = ev_h.n_events
    frag[f"{base}.untimed_handling"] = (
        "Not counted as an event; the patient stays in the denominator as event free, as in the "
        "main timing analysis. Sensitivity: the event is placed on day 0, "
        "day 1 or day 30.")
    frag[f"{base}.untimed_max_understatement_pct"] = rnd(100.0 * ev_h.n_untimed_events / len(df), 4)
    frag[f"{base}.crude_incidence_pct"] = rnd(100.0 * ev_h.n_event_flagged / len(df), 4)

    primary = {f.slug: f for f in fits_primary}
    sens_fits: List[Tuple[str, R1.StratumFit]] = []
    changed_any_overall = False
    for label, day in (("day0", 0), ("day1", 1), ("day30", 30)):
        ev = R1.make_events(df, "HEMATOMA", "HEMATOMA_POD", untimed="assign", untimed_day=day)
        need(ev.n_events == 541 and ev.n_untimed_events == 1, f"sensitivity {label}: events {ev.n_events}")
        fits = R1.fit_overall_and_strata(df, ev, bootstrap=(label != "day1"))
        need(fits[0].events == 541, f"sensitivity {label}: overall events {fits[0].events}")
        if label != "day1":
            sens_fits.append((label, fits[0]))
        sb = f"{base}.untimed_sens.{label}"
        frag[f"{sb}.events_n"] = fits[0].events
        frag[f"{sb}.incidence30_pct"] = rnd(100.0 * fits[0].cif30)
        frag[f"{sb}.incidence30_change_pct"] = rnd(100.0 * (fits[0].cif30 - primary["overall"].cif30), 6)
        overall_same = True
        for t in THRESHOLDS:
            d_p, d_s = primary["overall"].threshold_day(t), fits[0].threshold_day(t)
            frag[f"{sb}.{R1.threshold_slug(t)}.day"] = R1.day_or_none(d_s)
            if d_p != d_s:
                overall_same = False
        frag[f"{sb}.overall_threshold_days_unchanged"] = overall_same
        changed_any_overall = changed_any_overall or not overall_same
        n_changed = 0
        n_compared = 0
        detail: List[str] = []
        for f in fits:
            if f.suppressed or primary[f.slug].suppressed:
                continue
            for t in THRESHOLDS:
                n_compared += 1
                d_p, d_s = primary[f.slug].threshold_day(t), f.threshold_day(t)
                if d_s != d_p:
                    n_changed += 1
                    detail.append(f"{f.stratum.stratifier}, {f.stratum.level}, {R1.threshold_text(t)} "
                                  f"threshold: day {W.day_text(d_p)} to day {W.day_text(d_s)}")
        frag[f"{sb}.strata_threshold_days_changed_detail"] = "; ".join(detail) if detail else "None"
        frag[f"{sb}.strata_threshold_days_compared_n"] = n_compared
        frag[f"{sb}.strata_threshold_days_changed_n"] = n_changed
    frag[f"{base}.untimed_sens.overall_threshold_days_changed_any"] = changed_any_overall
    return frag, sens_fits


# ---------------------------------------------------------------------------
# step 5: fragment keys and Table S6A rows
# ---------------------------------------------------------------------------


def stratum_keys(key: str, fit: R1.StratumFit, frag: Dict[str, Any]) -> None:
    base = f"a2.{key}.{fit.slug}"
    frag[f"{base}.patients_n"] = fit.n
    frag[f"{base}.events_n"] = fit.events
    frag[f"{base}.competing_n"] = fit.competing
    frag[f"{base}.suppressed"] = bool(fit.suppressed)
    if fit.suppressed:
        return                                      # withheld: no estimate keys at all
    frag[f"{base}.incidence30_pct"] = rnd(100.0 * fit.cif30)
    frag[f"{base}.incidence30_lo_pct"] = rnd(100.0 * fit.boot.cif30_lo)
    frag[f"{base}.incidence30_hi_pct"] = rnd(100.0 * fit.boot.cif30_hi)
    for t in THRESHOLDS:
        tb = f"{base}.{R1.threshold_slug(t)}"
        info = threshold_info(fit, t)
        lo_i, hi_i = day_bounds(info["lo"], info["hi"])
        frag[f"{tb}.day"] = R1.day_or_none(info["day"])
        frag[f"{tb}.day_lo"] = lo_i
        frag[f"{tb}.day_hi"] = hi_i
        frag[f"{tb}.reached"] = bool(info["reached"])
        frag[f"{tb}.stays_below"] = info["stays"]
    for d in REPORT_DAYS:
        frag[f"{base}.resid_day{d}_pct"] = rnd(100.0 * fit.curve[d])
        frag[f"{base}.resid_day{d}_lo_pct"] = rnd(100.0 * fit.boot.lo[d])
        frag[f"{base}.resid_day{d}_hi_pct"] = rnd(100.0 * fit.boot.hi[d])


def note_for(fit: R1.StratumFit, infos: Optional[Dict[float, Dict[str, Any]]], extra: str = "") -> str:
    parts: List[str] = []
    if fit.suppressed:
        parts.append(f"Fewer than {VD.MIN_STRATUM_EVENTS} events of this outcome in the stratum "
                     f"({fit.events} observed), so every estimate is withheld")
    else:
        at_zero = [R1.threshold_text(t) for t in THRESHOLDS if infos[t]["day"] == 0]
        if at_zero:
            joined = " and ".join([", ".join(at_zero[:-1]), at_zero[-1]] if len(at_zero) > 2 else at_zero)
            parts.append(f"The 30-day cumulative incidence is already below {joined}, "
                         f"so the threshold day is 0")
        missed = [R1.threshold_text(t) for t in THRESHOLDS if not infos[t]["reached"]]
        if missed:
            parts.append("Not reached within 30 days: " + ", ".join(missed))
        bumpy = [R1.threshold_text(t) for t in THRESHOLDS
                 if infos[t]["reached"] and infos[t]["stays"] is False]
        if bumpy:
            parts.append("Residual risk rises above the threshold again after the threshold day: "
                         + ", ".join(bumpy))
    if extra:
        parts.append(extra)
    return "; ".join(parts)


def s6a_row(outcome_label: str, fit: R1.StratumFit, extra_note: str = "") -> List[Any]:
    s = fit.stratum
    head = [outcome_label, s.stratifier, s.level, VD.fmt_count(fit.n), VD.fmt_count(fit.events),
            VD.fmt_count(fit.n)]
    if fit.suppressed:
        return head + [SUPPRESSED] * (len(S6A_HEADER) - len(head) - 2) + ["Yes", note_for(fit, None, extra_note)]
    infos = {t: threshold_info(fit, t) for t in THRESHOLDS}
    cells: List[Any] = [W.pct(fit.cif30), W.pct_ci(fit.boot.cif30_lo, fit.boot.cif30_hi)]
    for t in THRESHOLDS:
        i = infos[t]
        cells += [i["day_text"], i["ci_text"],
                  (("Yes" if i["stays"] else "No") if i["reached"] else W.NOT_APPLICABLE)]
    for d in REPORT_DAYS:
        cells += [W.pct(float(fit.curve[d])), W.pct_ci(float(fit.boot.lo[d]), float(fit.boot.hi[d]))]
    row = head + cells + ["No", note_for(fit, infos, extra_note)]
    need(len(row) == len(S6A_HEADER), "Table S6A row width")
    return row


# ---------------------------------------------------------------------------
# step 6: Table S6B
# ---------------------------------------------------------------------------


def s6b_rows(mk: Dict[str, int]) -> List[List[Any]]:
    n = mk["patients_n"]

    def share(x: int) -> str:
        return W.pct(x / n)

    sec1 = "Components of the primary composite (a patient can have more than one)"
    sec2 = "Composition of the primary composite (each patient counted once)"
    return [
        [sec1, VD.pretty("HEMATOMA"), VD.fmt_count(mk["hematoma_n"]), VD.fmt_count(n), share(mk["hematoma_n"]),
         "Reoperation slot 1; crude share of all patients, whereas Table S6A gives the cumulative "
         "incidence with death as a competing event"],
        [sec1, VD.pretty("REINTUBATION"), VD.fmt_count(mk["reintubation_n"]), VD.fmt_count(n),
         share(mk["reintubation_n"]), "Airway component"],
        [sec1, VD.pretty("VENT48"), VD.fmt_count(mk["vent48_n"]), VD.fmt_count(n), share(mk["vent48_n"]),
         "Airway component"],
        [sec1, VD.pretty("REOP_TRACH"), VD.fmt_count(mk["reop_trach_n"]), VD.fmt_count(n),
         share(mk["reop_trach_n"]), "Airway component; reoperation slot 1 only"],
        [sec1, VD.pretty("AIRWAY_COMP"), VD.fmt_count(mk["airway_n"]), VD.fmt_count(n), share(mk["airway_n"]),
         "Reintubation, ventilator dependence beyond 48 hours or reoperative tracheostomy; the three "
         "components overlap, so they do not sum to this row"],
        [sec1, COMPOSITE_LABEL, VD.fmt_count(mk["composite_n"]), VD.fmt_count(n),
         share(mk["composite_n"]), "Hematoma requiring reoperation or composite airway morbidity"],
        [sec2, "Hematoma requiring reoperation only", VD.fmt_count(mk["hematoma_only_n"]), VD.fmt_count(n),
         share(mk["hematoma_only_n"]), ""],
        [sec2, "Composite airway morbidity only", VD.fmt_count(mk["airway_only_n"]), VD.fmt_count(n),
         share(mk["airway_only_n"]), ""],
        [sec2, "Both hematoma requiring reoperation and composite airway morbidity",
         VD.fmt_count(mk["hematoma_and_airway_n"]), VD.fmt_count(n), share(mk["hematoma_and_airway_n"]),
         "Overlap count"],
        [sec2, "Total, equal to the primary composite", VD.fmt_count(mk["composite_n"]), VD.fmt_count(n),
         share(mk["composite_n"]),
         f"{VD.fmt_count(mk['hematoma_only_n'])} + {VD.fmt_count(mk['airway_only_n'])} + "
         f"{VD.fmt_count(mk['hematoma_and_airway_n'])} = {VD.fmt_count(mk['composite_n'])}"],
    ]


# ---------------------------------------------------------------------------
# step 7: Figure S2
# ---------------------------------------------------------------------------

FIG_FLOOR_PCT = 0.005       # exact zeros cannot sit on a log axis; they are drawn at this floor


def make_figure(by_outcome: Dict[str, Dict[str, R1.StratumFit]], path: Path) -> List[Path]:
    R1.style_figure(base_pt=8.0)
    import matplotlib.pyplot as plt

    days = np.arange(R1.DAYS)

    def series(fit: R1.StratumFit) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        floor = FIG_FLOOR_PCT
        return (np.maximum(100.0 * fit.curve, floor), np.maximum(100.0 * fit.boot.lo, floor),
                np.maximum(100.0 * fit.boot.hi, floor))

    fig, axes = plt.subplots(1, 3, figsize=(7.9, 3.55), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.965, bottom=0.215, wspace=0.18)

    # Panel A: overall, three outcomes
    ax = axes[0]
    handles = []
    spec = (("hematoma", R1.ORANGE, 0, VD.pretty("HEMATOMA")),
            ("airway", R1.BLUE, 1, VD.pretty("AIRWAY_COMP")),
            ("composite", R1.INK_MUTED, 2, COMPOSITE_LABEL))
    for key, colour, idx, label in spec:
        est, lo, hi = series(by_outcome[key]["overall"])
        handles.append(R1.plot_step_curve(ax, days, est, lo, hi, level_index=idx, color=colour, label=label))
    ax.legend(handles=handles, loc="upper right", fontsize=6.6, handlelength=2.6, borderaxespad=0.2,
              labelspacing=0.35)
    ax.set_ylabel("Residual risk, %\n(log scale)")

    # Panels B and C: hematoma and airway morbidity by age group
    for ax, key, legend_title in ((axes[1], "hematoma", "Hematoma by age group"),
                                  (axes[2], "airway", "Airway morbidity by age group")):
        handles = []
        for i, slug in enumerate(AGE_SLUGS):
            fit = by_outcome[key][slug]
            est, lo, hi = series(fit)
            handles.append(R1.plot_step_curve(ax, days, est, lo, hi, level_index=i, label=fit.stratum.level))
        ax.legend(handles=handles, loc="upper right", fontsize=6.4, handlelength=2.6, borderaxespad=0.2,
                  labelspacing=0.3, title=legend_title, title_fontsize=6.6)

    for letter, ax in zip("ABC", axes):
        ax.set_yscale("log")
        ax.set_ylim(FIG_FLOOR_PCT, 4.0)
        ax.set_xlim(0, 30)
        ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
        ax.set_yticks([0.01, 0.1, 1.0])
        ax.set_yticklabels(["0.01", "0.1", "1"])
        ax.minorticks_off()
        ax.grid(axis="y", which="major", alpha=0.35)
        for t in THRESHOLDS:
            ax.axhline(100.0 * t, color=R1.INK_MUTED, lw=0.7, linestyle=R1.THRESHOLD_LINESTYLE, zorder=2)
            ax.text(29.8, 100.0 * t * 1.07, R1.threshold_text(t), ha="right", va="bottom", fontsize=6.4,
                    color=R1.INK_MUTED, zorder=6)
        ax.set_xlabel("Postoperative day, patient event free\nand alive at the start of the day")
        ax.text(-0.03, 1.0, letter, transform=ax.transAxes, fontsize=10.5, fontweight="bold",
                ha="right", va="bottom", color=R1.INK)
    return R1.save_png(fig, path, also_pdf=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="A2 component-specific residual risk.")
    ap.parse_args(argv)

    R1.seed_all()
    df = R1.load_cohort()
    n_total = len(df)

    # ---- events and fits (order: hematoma, airway, composite; each call reseeds with VD.SEED) ----
    events = build_events(df)
    ev_h, ev_a, ev_c = events["hematoma"], events["airway"], events["composite"]
    need(ev_h.n_event_flagged == 541 and ev_h.n_events == 540 and ev_h.n_untimed_events == 1,
         f"hematoma events {ev_h.n_event_flagged}/{ev_h.n_events}/{ev_h.n_untimed_events}, expected 541/540/1")
    need(ev_a.n_events == 608 and ev_a.n_untimed_events == 0, f"airway events {ev_a.n_events}")
    need(ev_c.n_events == 1015 and ev_c.n_untimed_events == 0, f"composite events {ev_c.n_events}")
    for key, ev in events.items():
        need(ev.n_death_strictly_first == 0, f"{key}: a patient dies strictly before their event day")

    fits: Dict[str, List[R1.StratumFit]] = {}
    for key, flag, pod, _untimed in OUTCOMES:
        fits[key] = R1.fit_overall_and_strata(df, events[key])
        check_partition(key, fits[key], n_total)
        log(f"{key}: {fits[key][0].events} events, 30-day incidence {100 * fits[key][0].cif30:.4f}%, "
            f"{sum(1 for f in fits[key] if f.suppressed)} suppressed stratum")
    by_outcome = {k: R1.index_by_slug(v) for k, v in fits.items()}

    # ---- the composite equals v1 Table 5 and 5B ----
    counts = assert_composite_equals_v1(fits["composite"])
    log(f"composite equals v1 Table 5 ({counts['table5_cells']} cells) and Table 5B "
        f"({counts['table5b_cells']} cells) exactly")
    res03 = json.loads((W.OUTPUTS / "results_03.json").read_text(encoding="utf-8"))
    for key, tag in (("hematoma", "hematoma"), ("airway", "airway")):
        need(abs(fits[key][0].cif30 - float(res03[f"cif.{tag}.day30.est"])) < 5e-9,
             f"{key} 30-day incidence {fits[key][0].cif30} differs from v1 results_03 {res03[f'cif.{tag}.day30.est']}")
    need(res03["cif.hematoma.day_unobserved_events_n"] == 1 and res03["cif.hematoma.day_observed_events_n"] == 540,
         "v1 results_03 does not report 540 timed and 1 untimed hematoma events")
    log("hematoma and airway 30-day incidence equal v1 results_03 (timing step)")

    # ---- composite makeup and the untimed hematoma ----
    mk = makeup_counts(df)
    check_makeup_against_v1(mk)
    log(f"makeup: hematoma {mk['hematoma_n']}, airway {mk['airway_n']}, both {mk['hematoma_and_airway_n']}, "
        f"hematoma only {mk['hematoma_only_n']}, airway only {mk['airway_only_n']}, "
        f"composite {mk['composite_n']} (ties)")
    untimed_frag, sens_fits = untimed_block(df, fits["hematoma"], ev_h)

    # ---- fragment ----
    frag: Dict[str, Any] = {}
    for key, _flag, _pod, _untimed in OUTCOMES:
        for fit in fits[key]:
            stratum_keys(key, fit, frag)
        reported = [f for f in fits[key] if f.slug != "overall" and not f.suppressed]
        low = min(reported, key=lambda f: (f.events, f.stratum.order))
        frag[f"a2.{key}.min_stratum_events_n"] = low.events
        frag[f"a2.{key}.min_stratum_events_label"] = low.stratum.level
        frag[f"a2.{key}.min_stratum_events_stratifier"] = low.stratum.stratifier
        frag[f"a2.{key}.suppressed_strata_n"] = sum(1 for f in fits[key] if f.suppressed)
        frag[f"a2.{key}.reported_strata_excluding_overall_n"] = len(reported)
        frag[f"a2.{key}.min_events_including_suppressed_n"] = min(f.events for f in fits[key] if f.slug != "overall")
    frag.update(untimed_frag)
    for name, value in mk.items():
        frag[f"a2.makeup.{name}"] = value
    frag["a2.makeup.parts_sum_to_composite"] = True
    frag["a2.check.composite_equals_table5"] = True
    frag["a2.check.composite_table5_cells_n"] = counts["table5_cells"]
    frag["a2.check.composite_table5b_cells_n"] = counts["table5b_cells"]
    frag["a2.check.strata_partition_cohort"] = True
    frag["a2.design.thresholds_pct"] = [rnd(100.0 * t, 4) for t in THRESHOLDS]
    frag["a2.design.bootstrap_replicates_n"] = W.BOOTSTRAP_REPLICATES
    frag["a2.design.seed"] = VD.SEED
    frag["a2.design.patients_n"] = n_total

    # ---- Table S6A ----
    rows_a: List[List[Any]] = []
    for key, _flag, _pod, _untimed in OUTCOMES:
        label = {"hematoma": VD.pretty("HEMATOMA"), "airway": VD.pretty("AIRWAY_COMP"),
                 "composite": COMPOSITE_LABEL}[key]
        for fit in fits[key]:
            extra = ""
            if key == "hematoma" and fit.slug == "overall":
                extra = (f"{VD.fmt_count(ev_h.n_event_flagged)} hematomas flagged; "
                         f"{ev_h.n_untimed_events} has no recorded postoperative day and is not counted as an "
                         f"event (the patient stays in the denominator)")
            rows_a.append(s6a_row(label, fit, extra))
    for label, fit in sens_fits:
        day = 0 if label == "day0" else 30
        rows_a.append(s6a_row(
            f"{VD.pretty('HEMATOMA')}, sensitivity: the event without a recorded day placed on day {day}", fit,
            "Sensitivity row for the one hematoma without a recorded postoperative day; not one of the "
            "13 strata of the partition"))
    need(len(rows_a) == 3 * 13 + 2, f"Table S6A has {len(rows_a)} rows")

    rows_b = s6b_rows(mk)

    # ---- write (each file once) ----
    path_json = R1.FRAGMENTS_OUT / "results_a2.json"
    path_a = R1.TABLES_OUT / "tableS6_component_residual_risk.csv"
    path_b = R1.TABLES_OUT / "tableS6_composite_makeup.csv"
    path_fig = R1.FIGURES_OUT / "figS2_component_residual_risk.png"
    nkeys = R1.write_fragment(path_json, frag, PREFIX)
    R1.write_csv_table(path_a, S6A_HEADER, rows_a)
    R1.write_csv_table(path_b, S6B_HEADER, rows_b)
    figs = make_figure(by_outcome, path_fig) if R1.FIGURES else []
    log(f"wrote {nkeys} keys to {path_json}")
    for p in (path_a, path_b, *figs):
        log(f"wrote {p}")

    # ---- self-check: re-read the outputs ----
    back = json.loads(path_json.read_text(encoding="utf-8"))
    need(back["a2.composite.overall.events_n"] == 1015 and back["a2.hematoma.overall.events_n"] == 540
         and back["a2.airway.overall.events_n"] == 608, "re-read fragment event counts")
    with open(path_a, encoding="utf-8", newline="") as fh:
        reread = list(csv.reader(fh))
    need(len(reread) == len(rows_a) + 1 and reread[0] == S6A_HEADER, "re-read Table S6A")

    # ---- console digest ----
    def line(key: str, slug: str) -> str:
        f = by_outcome[key][slug]
        if f.suppressed:
            return f"{key:9s} {f.stratum.level:18s} n={f.n:>7,} events={f.events:>4} suppressed"
        days = []
        for t in THRESHOLDS:
            i = threshold_info(f, t)
            days.append(f"{R1.threshold_text(t)}:{i['day_text']} ({i['ci_text']}; stays={i['stays']})")
        return (f"{key:9s} {f.stratum.level:18s} n={f.n:>7,} events={f.events:>4} "
                f"inc30={100 * f.cif30:.3f}% ({100 * f.boot.cif30_lo:.3f}-{100 * f.boot.cif30_hi:.3f})  "
                + "  ".join(days))

    print()
    for slug in ("overall",) + AGE_SLUGS:
        for key, *_ in OUTCOMES:
            print(line(key, slug))
        print()
    for key, *_ in OUTCOMES:
        print(f"{key}: minimum events among reported strata = {frag[f'a2.{key}.min_stratum_events_n']} "
              f"({frag[f'a2.{key}.min_stratum_events_label']}); minimum among all 12 non-overall strata = "
              f"{frag[f'a2.{key}.min_events_including_suppressed_n']}")
    print()
    for k in sorted(untimed_frag):
        if k.endswith((".day", "unchanged", "changed_n", "compared_n", "change_pct", "incidence30_pct",
                       "untimed_n", "any", "detail")):
            print(f"  {k} = {untimed_frag[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
