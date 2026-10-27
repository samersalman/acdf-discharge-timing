#!/usr/bin/env python3
"""a9_convention_sensitivity.py - A9, sensitivity analyses of two implementation conventions.

Two implementation conventions of the main analysis were not prespecified: the conditioning set
included all patients alive and event-free regardless of hospitalization, and a primary composite
event tied with death counted first. This script runs a sensitivity analysis of each and reports
them as they come out.
Primary composite (WINDOW_COMP, POD_WINDOW), death competing, integer days 0 to 30, v1
Aalen-Johansen estimator (through r1_common).

PANEL A (Table S4F, panel A). The 24 patients whose composite event and death fall on the SAME day
are the composite in v1 (primary). Here they enter as the competing death instead. Everything else
follows v1: overall row plus the four locked stratifiers, suppression on the SENSITIVITY event
count, 1,000 within-stratum multinomial bootstrap replicates through R1.fit_overall_and_strata
(reseeded with VD.SEED, so the draw order is v1's). Because a tied patient leaves the risk set on
the same day either way, the conditioning sets are identical, the sensitivity curve is the primary
curve minus (tied events on that day or later) / (patients at risk), and this is asserted.

PANEL B (Table S4F, panel B). Primary conditioning set at the start of day D (v1): event free and
alive. Sensitivity: event free, alive AND still in hospital, LOS_POSTOP >= D (v1's definition in
W.riskset_composition). The 83 patients with no recorded stay cannot be placed and are excluded
from this panel (counted, with their events). The risk at day D is the share of that set whose
first event is the primary composite on day D or later; with complete follow-up and no censoring
the Aalen-Johansen estimator reduces to this count, and that is verified against W.aalen_johansen
for every stratum and day. Bootstrap: patients resampled within stratum, 1,000 replicates, ONE
separate np.random.RandomState(VD.SEED + 9) (NOT the global stream) visited in W.build_strata
order; a stratum with fewer than 10 events in its day-0 set is suppressed and makes no draw.
Percentile 95% intervals. Any estimate computed on fewer than 10 events in its numerator is
withheld (counts shown): a risk at a day with fewer than 10 events in the set, and a threshold day
whose crossing day has fewer than 10 events in the set. Non-monotone curves are reported as they
are; nothing is smoothed.

Run:  PYTHONDONTWRITEBYTECODE=1 python3 a9_convention_sensitivity.py   (run a8 first: it reads the a8 fragment)
Writes (under the results directory): fragments/results_a9.json (prefix a9.),
tables-csv/tableS4F_tie_convention.csv, tables-csv/tableS4F_conditioning_set.csv.
Needs outputs/results_06.json and tables-csv/table5_safe_discharge_day.csv of the analysis folder
(written by step 06). Aggregates only; no patient-level output.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import csv
import dataclasses
import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import r1_common as R1

VD, W = R1.VD, R1.W
np.random.seed(VD.SEED)

DAYS = R1.DAYS                               # 31
LAST_DAY = VD.FOLLOWUP_DAYS                  # 30
MIN_EVENTS = VD.MIN_STRATUM_EVENTS           # 10
SUPPRESSED = R1.suppressed_text
NOT_APPLICABLE = "Not applicable"
NOT_REACHED = W.NOT_REACHED
THRESHOLDS = tuple(VD.RISK_THRESHOLDS)       # (0.01, 0.005, 0.0025)
REPORT_DAYS = (0, 1, 3, 7, 14)
COND_SEED = VD.SEED + 9
REPLICATES = W.BOOTSTRAP_REPLICATES
EXPECTED_TIE_N = 24
EXPECTED_CELLS_N = 36
EXPECTED_NO_STAY_N = 83


def r6(x: float) -> float:
    return round(float(x), 6)


class Keys(dict):
    """Result fragment collector that refuses a duplicate key."""

    def put(self, key: str, value: Any) -> None:
        if key in self:
            raise R1.R1Error(f"duplicate result key {key}")
        self[key] = value


# ===========================================================================
# PANEL A helpers
# ===========================================================================


def award_tie_to_death(ev: "R1.Events") -> Tuple["R1.Events", np.ndarray]:
    """Return (Events with the same-day composite-and-death ties scored as the competing death, tie mask).

    The tied patient stays in the risk set until the shared day either way, so exit_day is unchanged
    (asserted). Only is_event, pod_event, is_competing and the two counts change.
    """
    tie = ev.is_event & ev.is_death & (ev.pod_death == ev.pod_event)
    if int(tie.sum()) != ev.n_tie_same_day:
        raise R1.R1Error("tie mask disagrees with Events.n_tie_same_day")
    if bool(np.any(ev.is_competing & tie)):
        raise R1.R1Error("a tied patient is already the competing event in the primary rule")
    is_event = ev.is_event & ~tie
    is_comp = ev.is_competing | tie
    new = dataclasses.replace(
        ev, is_event=is_event, pod_event=np.where(is_event, ev.pod_event, np.nan),
        is_competing=is_comp, n_events=int(is_event.sum()), n_competing=int(is_comp.sum()))
    exit_day = np.full(len(is_event), np.inf, dtype=float)
    exit_day[new.is_event] = new.pod_event[new.is_event]
    exit_day[new.is_competing] = new.pod_death[new.is_competing]
    if not np.array_equal(exit_day, ev.exit_day):
        raise R1.R1Error("awarding the tie to death changed a leaving day")
    return dataclasses.replace(new, exit_day=exit_day), tie


def threshold_cells(fit: "R1.StratumFit", t: float) -> Dict[str, Any]:
    """One stratum-threshold cell in v1's Table 5 form (text) plus integer days for the fragment."""
    day = R1.threshold_day(fit.curve, t)
    boot = fit.boot_threshold_days(t)
    d_lo, d_hi = W.percentile_ci(boot.astype(float))
    reached = day <= LAST_DAY
    lo_i, hi_i = int(math.floor(d_lo)), int(math.ceil(d_hi))
    cell: Dict[str, Any] = dict(day=int(day), reached=bool(reached), day_text=W.day_text(day),
                                ci_text=W.day_ci_text(d_lo, d_hi),
                                lo=(None if lo_i > LAST_DAY else lo_i),
                                hi=(None if hi_i > LAST_DAY else hi_i))
    if reached:
        cell["risk_text"] = W.pct(float(fit.curve[day]))
        cell["risk_ci_text"] = W.pct_ci(float(fit.boot.lo[day]), float(fit.boot.hi[day]))
        cell["at_risk_text"] = VD.fmt_count(round(float(fit.at_risk[day])))
        cell["stays"] = bool(W.stays_below(fit.curve, t, int(day)))
        cell["stays_text"] = "Yes" if cell["stays"] else "No"
    else:
        cell["risk_text"] = cell["risk_ci_text"] = NOT_REACHED
        cell["at_risk_text"] = cell["stays_text"] = NOT_APPLICABLE
        cell["stays"] = None
    return cell


def load_v1_table5() -> Tuple[List[str], Dict[Tuple[str, str, str], List[str]]]:
    path = R1.ORIG_TABLES / "table5_safe_discharge_day.csv"
    with open(path, newline="", encoding="utf-8") as fh:                  # read only
        rows = list(csv.reader(fh))
    head = rows[0]
    by_key: Dict[Tuple[str, str, str], List[str]] = {}
    for r in rows[1:]:
        by_key[(r[0], r[1], r[6])] = r
    return head, by_key


def assert_primary_equals_table5(fits: Sequence["R1.StratumFit"]) -> int:
    """Every primary cell this script prints or uses equals v1 Table 5 (string for string). Returns cells compared."""
    head, by_key = load_v1_table5()
    col = {name: head.index(name) for name in head}
    compared = 0
    n_rows = 0
    for fit in fits:
        s = fit.stratum
        if fit.suppressed:
            row = by_key[(s.stratifier, s.level, SUPPRESSED)]       # the suppressed row carries the word in the threshold column
            if row[col["Patients"]] != VD.fmt_count(fit.n) or row[col["Composite events"]] != VD.fmt_count(fit.events):
                raise R1.R1Error(f"primary counts differ from Table 5 on suppressed {s.stratifier} / {s.level}")
            compared += 2
            n_rows += 1
            continue
        for t in THRESHOLDS:
            row = by_key[(s.stratifier, s.level, VD.threshold_label(t))]
            c = threshold_cells(fit, t)
            expect = {
                "Patients": VD.fmt_count(fit.n),
                "Composite events": VD.fmt_count(fit.events),
                "Cumulative incidence of the composite to day 30 (%)": W.pct(fit.cif30),
                "Residual-risk threshold day": c["day_text"],
                "Residual-risk threshold day, 95% confidence interval": c["ci_text"],
                "Residual risk at that day (%)": c["risk_text"],
                "Residual risk, 95% confidence interval (%)": c["risk_ci_text"],
                "Patients event free and alive at the start of that day": c["at_risk_text"],
                "Threshold reached within 30 days": "Yes" if c["reached"] else "No",
                "Residual risk remains below the threshold through day 30": c["stays_text"],
            }
            for name, want in expect.items():
                if row[col[name]] != want:
                    raise R1.R1Error(f"primary differs from v1 Table 5 at {s.stratifier} / {s.level} / "
                                     f"{VD.threshold_label(t)} / {name}: {row[col[name]]!r} vs {want!r}")
                compared += 1
            n_rows += 1
    if n_rows != len(by_key):
        raise R1.R1Error(f"Table 5 has {len(by_key)} rows, compared {n_rows}")
    return compared


def day_diff_text(primary_day: int, sens_day: int) -> str:
    return W.day_difference(int(primary_day), int(sens_day))


# ===========================================================================
# PANEL B helpers
# ===========================================================================


def reverse_cumsum(counts: np.ndarray) -> np.ndarray:
    return np.cumsum(counts[::-1], axis=-1)[::-1] if counts.ndim == 1 else np.cumsum(counts[:, ::-1], axis=1)[:, ::-1]


def set_counts(last_day: np.ndarray, is_event: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(31,) patients in the day-D still-in-hospital set and the composite events among them, D = 0..30.

    last_day[i] is the last day D for which patient i is in the set: min(leaving day, stay, 30).
    """
    n_d = reverse_cumsum(np.bincount(last_day, minlength=DAYS))
    k_d = reverse_cumsum(np.bincount(last_day[is_event], minlength=DAYS))
    return n_d.astype(np.int64), k_d.astype(np.int64)


def risk_curve(n_d: np.ndarray, k_d: np.ndarray) -> np.ndarray:
    return np.where(n_d > 0, k_d / np.maximum(n_d, 1), np.nan)


def boot_set_curves(last_day: np.ndarray, is_event: np.ndarray, rs: np.random.RandomState,
                    replicates: int) -> np.ndarray:
    """(B, 31) bootstrap risk curves: patients resampled with replacement within the stratum.

    One rs.randint draw of n patient indices per replicate, replicates in order. NaN where the
    resampled set is empty.
    """
    n = len(last_day)
    n_b = np.empty((replicates, DAYS), dtype=np.int64)
    k_b = np.empty((replicates, DAYS), dtype=np.int64)
    for b in range(replicates):
        idx = rs.randint(0, n, size=n)
        ld = last_day[idx]
        ev = is_event[idx]
        n_b[b] = reverse_cumsum(np.bincount(ld, minlength=DAYS))
        k_b[b] = reverse_cumsum(np.bincount(ld[ev], minlength=DAYS))
    return np.where(n_b > 0, k_b / np.maximum(n_b, 1), np.nan)


def aj_terminal_risk(ev: "R1.Events", in_set: np.ndarray, day: int) -> float:
    """The Aalen-Johansen day-30 cumulative incidence of a cohort that STARTS at `day` (time re-zeroed).

    The patients in `in_set` are event free and alive at the start of `day`; their composite days and
    competing-death days are shifted by `day`. Complete follow-up, no censoring: this must equal
    events / patients exactly, and the test below asserts it.
    """
    sel_e = in_set & ev.is_event
    sel_c = in_set & ev.is_competing
    de = (ev.pod_event[sel_e] - day).astype(int)
    dc = (ev.pod_death[sel_c] - day).astype(int)
    if (de.size and de.min() < 0) or (dc.size and dc.min() < 0):
        raise R1.R1Error("an event or death precedes the start of the day in the conditioning set")
    length = DAYS - day
    e1 = np.bincount(de, minlength=length).astype(float)
    e2 = np.bincount(dc, minlength=length).astype(float)
    if len(e1) != length or len(e2) != length:
        raise R1.R1Error("shifted days overflow the window")
    cif, _surv, _ = W.aalen_johansen(e1[None, :], e2[None, :], np.array([float(in_set.sum())]))
    return float(cif[0, -1])


def day_estimate_text(pct_text: str, ci_text: str) -> str:
    return f"{pct_text} ({ci_text})"


# ===========================================================================
# MAIN
# ===========================================================================


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="A9: sensitivity analyses of the two implementation conventions.")
    ap.parse_args(argv)
    out_dir, json_path = R1.TABLES_OUT, R1.FRAGMENTS_OUT / "results_a9.json"

    R1.seed_all()
    df = R1.load_cohort()
    n_all = len(df)
    ev = R1.make_events(df, "WINDOW_COMP", "POD_WINDOW")
    if not (ev.n_events == 1015 and ev.n_untimed_events == 0 and ev.n_competing == 119
            and ev.n_death == 213 and ev.n_death_strictly_first == 0):
        raise R1.R1Error("the primary composite does not reproduce v1 (1,015 events, 119 competing)")

    res = Keys()

    # =======================================================================
    # PANEL A: the same-day composite-and-death tie awarded to death
    # =======================================================================
    ev_s, tie = award_tie_to_death(ev)
    n_tie = int(tie.sum())
    v1_json = json.loads((R1.ORIG_OUTPUTS / "results_06.json").read_text(encoding="utf-8"))
    assert v1_json["window.competing.tie_same_day.n"] == n_tie == EXPECTED_TIE_N, "tie count differs from v1 and the expected 24"
    assert ev_s.n_events == 1015 - n_tie and ev_s.n_competing == 119 + n_tie
    assert not bool(np.any(ev_s.is_event & ev_s.is_competing))

    fits_p = R1.fit_overall_and_strata(df, ev)                  # primary, reseeded with VD.SEED: v1's draws
    fits_s = R1.fit_overall_and_strata(df, ev_s)                # sensitivity, same seed and order
    assert len(fits_p) == len(fits_s) == 13
    table5_cells = assert_primary_equals_table5(fits_p)         # primary columns equal v1 Table 5, string for string

    res.put("a9.tie.events_moved_n", n_tie)
    res.put("a9.tie.events_primary_n", ev.n_events)
    res.put("a9.tie.events_sens_n", ev_s.n_events)
    res.put("a9.tie.competing_primary_n", ev.n_competing)
    res.put("a9.tie.competing_sens_n", ev_s.n_competing)
    res.put("a9.tie.table5_cells_compared_n", table5_cells)
    res.put("a9.tie.thresholds_per_stratum_n", len(THRESHOLDS))

    rows_a: List[List[Any]] = []
    changed_cells: List[str] = []
    changed_days: List[int] = []
    n_cells = 0
    n_earlier = 0
    n_later = 0
    max_pp_all, max_pp_all_where = -1.0, ""
    moved_by_stratifier: Dict[str, int] = {}
    for fp, fs in zip(fits_p, fits_s):
        assert fp.slug == fs.slug and fp.n == fs.n
        s = fp.stratum
        slug = fp.slug
        mask = fp.mask
        moved = int((mask & tie).sum())
        assert fp.events - fs.events == moved, f"events moved disagree in {slug}"
        assert fs.competing - fp.competing == moved
        moved_by_stratifier[s.column] = moved_by_stratifier.get(s.column, 0) + moved
        p = f"a9.tie.{slug}"
        res.put(f"{p}.patients_n", fp.n)
        res.put(f"{p}.events_primary_n", fp.events)
        res.put(f"{p}.events_sens_n", fs.events)
        res.put(f"{p}.events_moved_n", moved)
        res.put(f"{p}.competing_primary_n", fp.competing)
        res.put(f"{p}.competing_sens_n", fs.competing)
        res.put(f"{p}.suppressed_primary", bool(fp.suppressed))
        res.put(f"{p}.suppressed_sens", bool(fs.suppressed))
        head = [s.stratifier, s.level, VD.fmt_count(fp.n), VD.fmt_count(fp.events),
                VD.fmt_count(fs.events), VD.fmt_count(moved)]
        if fp.suppressed or fs.suppressed:
            rows_a.append(head + [SUPPRESSED] * 9)
            continue

        # closed-form identity: same conditioning sets, curve drops by later tied events / patients at risk
        assert np.array_equal(fp.at_risk, fs.at_risk), f"conditioning sets differ in {slug}"
        tie_by_day = np.bincount(ev.pod_event[mask & tie].astype(int), minlength=DAYS).astype(float)
        later_ties = np.cumsum(tie_by_day[::-1])[::-1]
        expected_drop = later_ties / fp.at_risk
        drop = fp.curve - fs.curve
        assert np.allclose(drop, expected_drop, rtol=0.0, atol=1e-12), f"closed-form identity fails in {slug}"
        assert float(drop.min()) >= -1e-15, f"sensitivity curve above primary in {slug}"
        pp = 100.0 * drop
        k = int(np.argmax(pp))
        max_pp, max_day = float(pp[k]), k
        if max_pp > max_pp_all:
            max_pp_all, max_pp_all_where = max_pp, f"{s.stratifier}: {s.level}, day {max_day}"
            res_max_day = max_day
        res.put(f"{p}.cif30_primary_pct", r6(100.0 * fp.cif30))
        res.put(f"{p}.cif30_primary_lo_pct", r6(100.0 * fp.boot.cif30_lo))
        res.put(f"{p}.cif30_primary_hi_pct", r6(100.0 * fp.boot.cif30_hi))
        res.put(f"{p}.cif30_sens_pct", r6(100.0 * fs.cif30))
        res.put(f"{p}.cif30_sens_lo_pct", r6(100.0 * fs.boot.cif30_lo))
        res.put(f"{p}.cif30_sens_hi_pct", r6(100.0 * fs.boot.cif30_hi))
        res.put(f"{p}.cif30_diff_pp", r6(100.0 * (fs.cif30 - fp.cif30)))
        res.put(f"{p}.max_change_pp", r6(max_pp))
        res.put(f"{p}.max_change_day", int(max_day))
        res.put(f"{p}.max_change_primary_pct", r6(100.0 * fp.curve[max_day]))
        res.put(f"{p}.max_change_sens_pct", r6(100.0 * fs.curve[max_day]))
        res.put(f"{p}.day0_change_pp", r6(pp[0]))

        for t in THRESHOLDS:
            cp, cs = threshold_cells(fp, t), threshold_cells(fs, t)
            tseg = R1.threshold_slug(t)
            q = f"{p}.{tseg}"
            res.put(f"{q}.day_primary", R1.day_or_none(cp["day"]))
            res.put(f"{q}.day_primary_lo", cp["lo"])
            res.put(f"{q}.day_primary_hi", cp["hi"])
            res.put(f"{q}.day_sens", R1.day_or_none(cs["day"]))
            res.put(f"{q}.day_sens_lo", cs["lo"])
            res.put(f"{q}.day_sens_hi", cs["hi"])
            changed = cp["day"] != cs["day"]
            res.put(f"{q}.changed", bool(changed))
            if cp["reached"] and cs["reached"]:
                res.put(f"{q}.day_diff", int(cs["day"] - cp["day"]))
            if cp["stays"] is not None:
                res.put(f"{q}.stays_below_primary", cp["stays"])
            if cs["stays"] is not None:
                res.put(f"{q}.stays_below_sens", cs["stays"])
            if cp["reached"]:
                res.put(f"{q}.risk_at_day_primary_pct", r6(100.0 * fp.curve[cp["day"]]))
                res.put(f"{q}.risk_at_day_sens_same_day_pct", r6(100.0 * fs.curve[cp["day"]]))
            n_cells += 1
            if changed:
                changed_cells.append(f"{s.stratifier}: {s.level}, {R1.threshold_text(t)}")
                changed_days.append(int(cs["day"] - cp["day"]) if (cp["reached"] and cs["reached"]) else 99)
                if cs["day"] < cp["day"]:
                    n_earlier += 1
                else:
                    n_later += 1
            p_txt = NOT_REACHED if not cp["reached"] else f"{cp['day_text']} ({cp['ci_text']})"
            s_txt = NOT_REACHED if not cs["reached"] else f"{cs['day_text']} ({cs['ci_text']})"
            rows_a.append(head + [
                W.pct(fp.cif30), W.pct(fs.cif30), VD.threshold_label(t),
                p_txt, s_txt, day_diff_text(cp["day"], cs["day"]),
                cp["stays_text"], cs["stays_text"], f"{max_pp:.4f} (day {max_day})"])

    assert n_cells == EXPECTED_CELLS_N, f"{n_cells} stratum-threshold cells, expected {EXPECTED_CELLS_N}"
    for column, total in moved_by_stratifier.items():
        assert total == n_tie, f"events moved do not partition across {column}"
    res.put("a9.tie.cells_total_n", n_cells)
    res.put("a9.tie.cells_changed_n", len(changed_cells))
    res.put("a9.tie.cells_unchanged_n", n_cells - len(changed_cells))
    res.put("a9.tie.cells_earlier_n", n_earlier)
    res.put("a9.tie.cells_later_n", n_later)
    res.put("a9.tie.cells_changed_list", "; ".join(changed_cells) if changed_cells else "None")
    if changed_days:
        assert 99 not in changed_days
        res.put("a9.tie.max_abs_day_change_days", int(max(abs(d) for d in changed_days)))
    else:
        res.put("a9.tie.max_abs_day_change_days", 0)
    res.put("a9.tie.max_change_pp_any_stratum", r6(max_pp_all))
    res.put("a9.tie.max_change_pp_any_stratum_where", max_pp_all_where)
    res.put("a9.tie.max_change_pp_any_stratum_day", int(res_max_day))
    ov_p, ov_s = fits_p[0], fits_s[0]
    # the overall row in the headline form
    res.put("a9.tie.overall.events_moved_share_pct", r6(100.0 * n_tie / ov_p.events))

    head_a = [
        "Stratifier", "Stratum", "Patients",
        "Composite events, tie counted as the composite (primary)",
        "Composite events, tie counted as death (sensitivity)",
        "Events moved to death",
        "Cumulative incidence of the composite to day 30, primary (%)",
        "Cumulative incidence of the composite to day 30, sensitivity (%)",
        "Risk threshold",
        "Residual-risk threshold day, primary (95% confidence interval)",
        "Residual-risk threshold day, sensitivity (95% confidence interval)",
        "Difference in threshold day",
        "Residual risk remains below the threshold through day 30, primary",
        "Residual risk remains below the threshold through day 30, sensitivity",
        "Largest decrease in conditional risk across days 0 to 30 (percentage points, day)"]
    R1.write_csv_table(out_dir / "tableS4F_tie_convention.csv", head_a, rows_a)

    # =======================================================================
    # PANEL B: condition only on patients still in hospital at the start of the day
    # =======================================================================
    los = ev.los
    recorded = np.isfinite(los)
    assert int((~recorded).sum()) == EXPECTED_NO_STAY_N, "patients with no recorded stay differ from 83"
    assert bool(np.all(los[recorded] >= 0)) and bool(np.all(los[recorded] % 1 == 0)), "stay is not a non-negative integer"
    leave = np.minimum(np.where(np.isinf(ev.exit_day), float(LAST_DAY), ev.exit_day), float(LAST_DAY))
    last_day_all = np.floor(np.minimum(leave, np.where(recorded, los, float(LAST_DAY)))).astype(np.int64)

    # day-0 relation, stated and asserted below per stratum
    nostay_n = int((~recorded).sum())
    nostay_events = int((ev.is_event & ~recorded).sum())
    assert int((ev.is_competing & ~recorded).sum()) == 0
    res.put("a9.cond.nostay.patients_n", nostay_n)
    res.put("a9.cond.nostay.events_n", nostay_events)
    res.put("a9.cond.nostay.event_rate_pct", r6(100.0 * nostay_events / nostay_n))
    res.put("a9.cond.nostay.competing_deaths_n", 0)

    # v1's own numbers for the share of the primary set still in hospital (read only)
    a8 = json.loads((R1.FRAGMENTS_OUT / "results_a8.json").read_text(encoding="utf-8"))
    rs = np.random.RandomState(COND_SEED)                      # separate stream; the global one is not used here
    strata = W.build_strata(df)
    assert [s.slug for s, _m in strata] == [f.slug for f in fits_p]
    rows_b: List[List[Any]] = []
    max_aj_gap = 0.0
    aj_checks = 0
    n_v1_set_checks = 0
    n_day0_checks = 0
    point_never = 0
    n_bootstrapped = 0
    cond_summary: Dict[str, Dict[str, Any]] = {}
    for (stratum, mask), fp in zip(strata, fits_p):
        slug = stratum.slug
        p = f"a9.cond.{slug}"
        rec_mask = mask & recorded
        n_str = int(mask.sum())
        n_rec = int(rec_mask.sum())
        n_nostay = n_str - n_rec
        ld = last_day_all[rec_mask]
        evs = ev.is_event[rec_mask]
        n_d, k_d = set_counts(ld, evs)

        # direct definition (v1): event free and alive at the start of the day AND stay >= day
        at_all, in_h, dis, miss = W.riskset_composition(mask, ev.exit_day, ev.los)       # asserts the partition
        for d in range(DAYS):
            with np.errstate(invalid="ignore"):
                in_set = rec_mask & (ev.exit_day >= d) & (los >= d)
            assert int(in_set.sum()) == int(n_d[d]) == int(in_h[d]), f"set size differs {slug} day {d}"
            assert int((in_set & ev.is_event).sum()) == int(k_d[d]), f"set events differ {slug} day {d}"
            if n_d[d] > 0:                                                              # AJ equals counting
                aj = aj_terminal_risk(ev, in_set, d)
                gap = abs(aj - k_d[d] / n_d[d])
                max_aj_gap = max(max_aj_gap, gap)
                aj_checks += 1
                assert gap < 1e-12, f"Aalen-Johansen disagrees with counting {slug} day {d}: {gap}"
        key_v1 = f"condrisk.{slug}.in_hospital_n"
        if key_v1 in v1_json:
            assert [int(x) for x in v1_json[key_v1]] == [int(x) for x in n_d], f"v1 in-hospital counts differ {slug}"
            n_v1_set_checks += 1
        # day-0 relation: the set is the stratum minus the patients with no recorded stay
        assert int(n_d[0]) == n_rec and int(k_d[0]) == int(fp.events) - int((mask & ev.is_event & ~recorded).sum())
        e_ns = int((mask & ev.is_event & ~recorded).sum())
        if fp.events >= MIN_EVENTS and n_rec > 0:
            sens0 = k_d[0] / n_d[0]
            prim0 = fp.events / fp.n
            gap0 = sens0 - prim0
            formula = (fp.events * n_nostay - e_ns * fp.n) / (fp.n * n_rec)       # = k/(n-m) - K/n
            assert abs(gap0 - formula) < 1e-12, f"day-0 relation fails {slug}"
            if n_nostay == 0:
                assert gap0 == 0.0
            n_day0_checks += 1
            if slug == "overall":
                res.put("a9.cond.overall.day0_gap_pp", r6(100.0 * gap0))
                res.put("a9.cond.overall.day0_gap_formula_pp", r6(100.0 * formula))
        res.put(f"{p}.patients_n", n_str)
        res.put(f"{p}.nostay_n", n_nostay)
        res.put(f"{p}.nostay_events_n", e_ns)

        suppressed = int(k_d[0]) < MIN_EVENTS
        res.put(f"{p}.suppressed", bool(suppressed))
        risk = risk_curve(n_d, k_d)
        boot_curve = None
        lo = hi = None
        if not suppressed:
            boot_curve = boot_set_curves(ld, evs, rs, REPLICATES)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                lo, hi = W.percentile_ci(boot_curve, axis=0)
            n_bootstrapped += 1

        share = 100.0 * in_h / at_all
        if slug == "overall":
            # recompute and assert equality with v1 and A8 for the share of the primary set still in hospital
            v1_pct = v1_json["condrisk.overall.in_hospital_pct"]
            for d in (1, 3, 7, 14):
                assert abs(float(share[d]) - float(v1_pct[d])) < 1e-9, f"share in hospital differs from v1 at day {d}"
                if d in (1, 3, 7):
                    assert r6(share[d]) == a8[f"a8.composition.day{d}.inhosp.share_of_set_pct"], f"A8 share differs at day {d}"
                    assert int(in_h[d]) == a8[f"a8.composition.day{d}.inhosp.n"]
                    assert int(at_all[d]) == a8[f"a8.composition.day{d}.all.n"]

        # per-day facts
        for d in REPORT_DAYS:
            q = f"{p}.day{d}"
            res.put(f"{q}.primary_set_n", int(at_all[d]))
            res.put(f"{q}.inhosp_set_n", int(n_d[d]))
            res.put(f"{q}.inhosp_share_of_primary_set_pct", r6(share[d]))
            res.put(f"{q}.events_n", int(k_d[d]))
            res.put(f"{q}.suppressed", bool(suppressed or k_d[d] < MIN_EVENTS or n_d[d] == 0))
            if not suppressed and k_d[d] >= MIN_EVENTS and n_d[d] > 0:
                res.put(f"{q}.risk_pct", r6(100.0 * risk[d]))
                res.put(f"{q}.risk_lo_pct", r6(100.0 * lo[d]))
                res.put(f"{q}.risk_hi_pct", r6(100.0 * hi[d]))
            if not fp.suppressed:
                res.put(f"{q}.primary_risk_pct", r6(100.0 * fp.curve[d]))
                res.put(f"{q}.primary_risk_lo_pct", r6(100.0 * fp.boot.lo[d]))
                res.put(f"{q}.primary_risk_hi_pct", r6(100.0 * fp.boot.hi[d]))
            if not suppressed and k_d[d] >= MIN_EVENTS and n_d[d] > 0 and not fp.suppressed:
                res.put(f"{q}.risk_over_primary_ratio", r6(risk[d] / fp.curve[d]))

        # evaluable days: those whose numerator has at least 10 events
        evaluable = [d for d in range(DAYS) if (not suppressed) and n_d[d] > 0 and k_d[d] >= MIN_EVENTS]
        res.put(f"{p}.evaluable_days_n", len(evaluable))
        low_day = high_day = None
        if evaluable:
            res.put(f"{p}.last_evaluable_day", int(max(evaluable)))
            low_day = min(evaluable, key=lambda d: (risk[d], d))
            high_day = max(evaluable, key=lambda d: (risk[d], -d))
            res.put(f"{p}.lowest_risk_pct", r6(100.0 * risk[low_day]))
            res.put(f"{p}.lowest_risk_day", int(low_day))
            res.put(f"{p}.highest_risk_pct", r6(100.0 * risk[high_day]))
            res.put(f"{p}.highest_risk_day", int(high_day))
            rises = sum(1 for d in evaluable if (d - 1) in evaluable and risk[d] > risk[d - 1])
            falls = sum(1 for d in evaluable if (d - 1) in evaluable and risk[d] < risk[d - 1])
            res.put(f"{p}.rises_n", int(rises))
            res.put(f"{p}.falls_n", int(falls))

        # thresholds
        thr_info: Dict[float, Dict[str, Any]] = {}
        for t in THRESHOLDS:
            tseg = R1.threshold_slug(t)
            q = f"{p}.{tseg}"
            info: Dict[str, Any] = {}
            if not fp.suppressed:
                cp = threshold_cells(fp, t)
                info["primary"] = cp
                res.put(f"{q}.primary_day", R1.day_or_none(cp["day"]))
                res.put(f"{q}.primary_day_lo", cp["lo"])
                res.put(f"{q}.primary_day_hi", cp["hi"])
            if suppressed:
                thr_info[t] = info
                continue
            day = R1.threshold_day(risk, t)
            reached = day <= LAST_DAY
            bdays = W.first_day_below(boot_curve, t)
            never = int(np.sum(bdays > LAST_DAY))
            d_lo, d_hi = W.percentile_ci(bdays.astype(float))
            lo_i, hi_i = int(math.floor(d_lo)), int(math.ceil(d_hi))
            res.put(f"{q}.reached", bool(reached))
            res.put(f"{q}.never_crossed_reps_n", never)
            res.put(f"{q}.replicates_n", REPLICATES)
            info.update(day=int(day), reached=bool(reached), never=never, ci_text=W.day_ci_text(d_lo, d_hi),
                        lo=(None if lo_i > LAST_DAY else lo_i), hi=(None if hi_i > LAST_DAY else hi_i))
            if reached:
                numerator, setn = int(k_d[day]), int(n_d[day])
                withheld = numerator < MIN_EVENTS
                info.update(events_at_day=numerator, set_at_day=setn, withheld=withheld)
                res.put(f"{q}.events_at_day_n", numerator)
                res.put(f"{q}.set_at_day_n", setn)
                res.put(f"{q}.withheld", bool(withheld))
                if not withheld:
                    stays = bool(R1.stays_below_from(risk, t, int(day)))
                    info["stays"] = stays
                    res.put(f"{q}.day", int(day))
                    res.put(f"{q}.day_lo", info["lo"])
                    res.put(f"{q}.day_hi", info["hi"])
                    res.put(f"{q}.stays_below", stays)
                    res.put(f"{q}.risk_at_day_pct", r6(100.0 * risk[day]))
            else:
                point_never += 1
                res.put(f"{q}.day", None)
            thr_info[t] = info
        cond_summary[slug] = dict(stratum=stratum, fit=fp, n_d=n_d, k_d=k_d, risk=risk, lo=lo, hi=hi,
                                  suppressed=suppressed, share=share, at_all=at_all, thr=thr_info,
                                  low_day=low_day, n_str=n_str, n_nostay=n_nostay)

        # ---- rows for the table: 5 days, 3 thresholds, 1 lowest-risk row
        head_b = [stratum.stratifier, stratum.level, VD.fmt_count(n_str), VD.fmt_count(n_nostay)]
        for d in REPORT_DAYS:
            if fp.suppressed:
                prim_txt = SUPPRESSED
            else:
                prim_txt = day_estimate_text(W.pct(float(fp.curve[d])),
                                             W.pct_ci(float(fp.boot.lo[d]), float(fp.boot.hi[d])))
            if suppressed or k_d[d] < MIN_EVENTS or n_d[d] == 0:
                sens_txt = SUPPRESSED
            else:
                sens_txt = day_estimate_text(W.pct(float(risk[d])), W.pct_ci(float(lo[d]), float(hi[d])))
            rows_b.append(head_b + [f"Residual risk at the start of day {d}",
                                    VD.fmt_count(at_all[d]), prim_txt,
                                    VD.fmt_count(n_d[d]), VD.fmt_pct(share[d], 1), VD.fmt_count(k_d[d]), sens_txt,
                                    NOT_APPLICABLE, NOT_APPLICABLE])
        for t in THRESHOLDS:
            info = thr_info[t]
            if fp.suppressed:
                prim_txt = SUPPRESSED
            else:
                cp = info["primary"]
                prim_txt = NOT_REACHED if not cp["reached"] else f"{cp['day_text']} ({cp['ci_text']})"
            label = f"First day below {R1.threshold_text(t)}"
            if suppressed:
                rows_b.append(head_b + [label, NOT_APPLICABLE, prim_txt, NOT_APPLICABLE, NOT_APPLICABLE,
                                        NOT_APPLICABLE, SUPPRESSED, NOT_APPLICABLE, NOT_APPLICABLE])
                continue
            if not info["reached"]:
                sens_txt, setn_txt, ev_txt, stays_txt = NOT_REACHED, NOT_APPLICABLE, NOT_APPLICABLE, NOT_APPLICABLE
                share_txt = NOT_APPLICABLE
            else:
                setn_txt, ev_txt = VD.fmt_count(info["set_at_day"]), VD.fmt_count(info["events_at_day"])
                share_txt = VD.fmt_pct(float(share[info["day"]]), 1)
                if info["withheld"]:
                    sens_txt, stays_txt = SUPPRESSED, NOT_APPLICABLE
                else:
                    sens_txt = f"{info['day']} ({info['ci_text']})"
                    stays_txt = "Yes" if info["stays"] else "No"
            rows_b.append(head_b + [label, NOT_APPLICABLE, prim_txt, setn_txt, share_txt, ev_txt, sens_txt,
                                    stays_txt, VD.fmt_count(info["never"])])
        if suppressed or low_day is None:
            rows_b.append(head_b + ["Lowest residual risk on days with 10 or more events", NOT_APPLICABLE,
                                    NOT_APPLICABLE, NOT_APPLICABLE, NOT_APPLICABLE, NOT_APPLICABLE, SUPPRESSED,
                                    NOT_APPLICABLE, NOT_APPLICABLE])
        else:
            rows_b.append(head_b + [f"Lowest residual risk on days with 10 or more events (day {low_day})",
                                    NOT_APPLICABLE, NOT_APPLICABLE, VD.fmt_count(n_d[low_day]),
                                    VD.fmt_pct(float(share[low_day]), 1), VD.fmt_count(k_d[low_day]),
                                    W.pct(float(risk[low_day])), NOT_APPLICABLE, NOT_APPLICABLE])

    # strata partition checks for panel B
    for column in VD.STRATIFIER_COLUMNS:
        tot = sum(c["n_str"] for c in cond_summary.values() if c["stratum"].column == column)
        assert tot == n_all
        tot_ns = sum(c["n_nostay"] for c in cond_summary.values() if c["stratum"].column == column)
        assert tot_ns == nostay_n, f"patients with no recorded stay do not partition across {column}"

    # overall facts, upper bound that places the 83 in hospital (point estimates only)
    ov = cond_summary["overall"]
    res.put("a9.cond.bootstrap_seed", COND_SEED)
    res.put("a9.cond.bootstrap_replicates_n", REPLICATES)
    res.put("a9.cond.strata_bootstrapped_n", n_bootstrapped)
    res.put("a9.cond.aj_equals_counting_checks_n", aj_checks)
    res.put("a9.cond.aj_equals_counting_max_abs_gap", float(f"{max_aj_gap:.2e}"))
    res.put("a9.cond.v1_inhospital_count_checks_n", n_v1_set_checks)
    res.put("a9.cond.day0_relation_checks_n", n_day0_checks)
    res.put("a9.cond.point_threshold_never_reached_cells_n", point_never)
    for d in REPORT_DAYS:
        in_up = (ev.exit_day >= d) & (~recorded | (los >= d))
        n_up = int(in_up.sum())
        k_up = int((in_up & ev.is_event).sum())
        res.put(f"a9.cond.overall.upper.day{d}.set_n", n_up)
        res.put(f"a9.cond.overall.upper.day{d}.events_n", k_up)
        res.put(f"a9.cond.overall.upper.day{d}.risk_pct", r6(100.0 * k_up / n_up))

    # cells (stratum x threshold) of the panel: how many days are shown, withheld, never reached
    shown = withheld_n = not_reached_n = suppressed_n = 0
    for slug, c in cond_summary.items():
        for t in THRESHOLDS:
            info = c["thr"][t]
            if c["suppressed"]:
                suppressed_n += 1
            elif not info["reached"]:
                not_reached_n += 1
            elif info["withheld"]:
                withheld_n += 1
            else:
                shown += 1
    res.put("a9.cond.cells_total_n", 3 * len(cond_summary))
    res.put("a9.cond.cells_day_shown_n", shown)
    res.put("a9.cond.cells_day_withheld_n", withheld_n)
    res.put("a9.cond.cells_not_reached_n", not_reached_n)
    res.put("a9.cond.cells_stratum_suppressed_n", suppressed_n)
    assert shown + withheld_n + not_reached_n + suppressed_n == 3 * len(cond_summary)

    head_b = [
        "Stratifier", "Stratum", "Patients", "Patients with no recorded stay (excluded from the sensitivity set)",
        "Measure",
        "Primary set: patients event free and alive at the start of the day",
        "Primary set: residual risk (95% confidence interval) or threshold day (95% confidence interval)",
        "Sensitivity set: patients event free, alive and still in hospital at the start of the day",
        "Sensitivity set as a share of the primary set (%)",
        "Sensitivity set: composite events on that day or later",
        "Sensitivity set: residual risk (95% confidence interval) or threshold day (95% confidence interval)",
        "Residual risk remains below the threshold through day 30 (sensitivity set)",
        "Bootstrap replicates in which the threshold was never crossed (of 1,000)"]
    R1.write_csv_table(out_dir / "tableS4F_conditioning_set.csv", head_b, rows_b)

    # =======================================================================
    # note sentences (numbers tied to keys above)
    # =======================================================================
    g = res
    changed_n = g["a9.tie.cells_changed_n"]
    where = "all by 1 day earlier" if (changed_n and n_later == 0 and g["a9.tie.max_abs_day_change_days"] == 1) \
        else ("none" if not changed_n else f"{n_earlier} earlier and {n_later} later")
    res.put("a9.tie.note",
            f"Scoring the {n_tie} patients whose composite event and death fell on the same day as deaths rather "
            f"than as composite events lowered the overall 30-day cumulative incidence from "
            f"{g['a9.tie.overall.cif30_primary_pct']:.3f}% to {g['a9.tie.overall.cif30_sens_pct']:.3f}% and changed "
            f"the threshold day in {changed_n} of {n_cells} stratum-threshold cells ({where}).")
    o = "a9.cond.overall"
    res.put("a9.cond.note",
            f"Among the {n_all - nostay_n:,} patients with a recorded stay, the residual risk in the set still in "
            f"hospital at the start of the day was {g[o + '.day0.risk_pct']:.3f}% at day 0, "
            f"{g[o + '.day1.risk_pct']:.3f}% at day 1, {g[o + '.day3.risk_pct']:.3f}% at day 3 and "
            f"{g[o + '.day7.risk_pct']:.3f}% at day 7, against {g[o + '.day0.primary_risk_pct']:.3f}%, "
            f"{g[o + '.day1.primary_risk_pct']:.3f}%, {g[o + '.day3.primary_risk_pct']:.3f}% and "
            f"{g[o + '.day7.primary_risk_pct']:.3f}% in the primary set.")

    json_n = R1.write_fragment(json_path, res, ("a9.",))

    print(f"[a9] wrote {json_n} keys and 2 tables; primary equals v1 Table 5 ({table5_cells} cells); "
          f"AJ equals counting in {aj_checks} checks (max gap {max_aj_gap:.2e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
