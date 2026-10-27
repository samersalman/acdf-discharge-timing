#!/usr/bin/env python3
"""a8_selection_riskset.py - A8, selection among patients who are still admitted.

Descriptive selection among patients who are still admitted. No model, no odds ratio, no causal claim.
Primary composite (WINDOW_COMP, POD_WINDOW), death competing; the conditioning set at day d is
patients event free and alive at the START of day d (the v1 set).

1. Hospital status at the start of day d, as in W.riskset_composition: still in hospital when the
   recorded stay is at least d (LOS_POSTOP >= d), already discharged when it is shorter (< d), stay not
   recorded as its own group (83 patients; never placed silently). Composite events on day d are split
   by the status their patient had at the start of that day (a patient discharged ON day d was in
   hospital at its start, v1's rule). Each patient at risk at the start of a day contributes one
   patient-day to their status group.
2. Table S11A: band hazards (same bands as A3) by status with exact Poisson 95% intervals, and the ratio
   of the two rates (in hospital over discharged) with an exact conditional (binomial) 95% interval.
   Any cell under 10 events is withheld, and so is a ratio that would reveal it. The ratio is descriptive,
   subject to confounding by indication, and is NOT an estimate of the effect of staying.
3. Table S11B: at the start of days 1, 3 and 7, the composition of the conditioning set: the share still
   in hospital and, within each status, the share aged 75 or older, ASA class 3 or 4 and 3 or more fusion
   levels (the locked strata only).
4. Counts measured from the parquet: the INOUT outpatient label and LOS_POSTOP == 0, compared with the
   counts quoted in the paper.

Run:  PYTHONDONTWRITEBYTECODE=1 python3 a8_selection_riskset.py
Writes (under the results directory): fragments/results_a8.json (prefix a8.),
tables-csv/tableS11_selection_still_admitted.csv (panel A and panel B in one file, told apart by the
Panel column). Nothing here is random. Aggregates only.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy import stats

import r1_common as R1
import a3_horizon_artifact as A3

VD, W = R1.VD, R1.W
np.random.seed(VD.SEED)

DAYS = R1.DAYS
MIN_EVENTS = VD.MIN_STRATUM_EVENTS
SUPPRESSED = R1.suppressed_text
NOT_APPLICABLE = "Not applicable"
BANDS = A3.BANDS

# status index -> (key segment, table label)
STATUS = (("inhosp", "Still in hospital"),
          ("discharged", "Already discharged"),
          ("notrecorded", "Stay not recorded"))
COMPOSITION_DAYS = (1, 3, 7)


def rate_ratio_exact(k1: int, pd1: int, k2: int, pd2: int, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Rate ratio (k1/pd1)/(k2/pd2) with the exact conditional (Clopper-Pearson on the split) interval.

    Given the total k = k1 + k2, k1 is binomial(k, p) with p = rho*pd1 / (rho*pd1 + pd2), rho the rate
    ratio. The Clopper-Pearson interval for p is mapped back to rho = p/(1-p) * pd2/pd1.
    """
    k = k1 + k2
    if k1 < 1 or k2 < 1:
        raise R1.R1Error("rate_ratio_exact needs at least one event in each group")
    p_lo = float(stats.beta.ppf(alpha / 2.0, k1, k2 + 1))
    p_hi = float(stats.beta.ppf(1.0 - alpha / 2.0, k1 + 1, k2))
    scale = pd2 / pd1
    est = (k1 / pd1) / (k2 / pd2)
    return est, p_lo / (1.0 - p_lo) * scale, p_hi / (1.0 - p_hi) * scale


def events_by_status_and_day(ev: "R1.Events") -> np.ndarray:
    """(3, 31) composite events ON each day, split by the status of the patient at the START of that day.

    Row 0 still in hospital (LOS_POSTOP >= day), row 1 already discharged (LOS_POSTOP < day), row 2 stay
    not recorded. The three rows sum to the overall events by day.
    """
    out = np.zeros((3, DAYS), dtype=np.int64)
    sel = np.flatnonzero(ev.is_event)
    days = ev.pod_event[sel].astype(int)
    los = ev.los[sel]
    recorded = np.isfinite(los)
    with np.errstate(invalid="ignore"):
        in_h = recorded & (los >= days)
        dis = recorded & (los < days)
    for d in range(DAYS):
        out[0, d] = int(np.sum(in_h & (days == d)))
        out[1, d] = int(np.sum(dis & (days == d)))
        out[2, d] = int(np.sum(~recorded & (days == d)))
    return out


def status_band_cells(at: np.ndarray, e: np.ndarray) -> List[Dict[str, Any]]:
    """Band cells for one status group (at: patients at risk by day, e: events by day), A3's rules."""
    cells = []
    for first, last, label, seg in BANDS:
        k = int(e[first:last + 1].sum())
        pdays = int(at[first:last + 1].sum())
        cell: Dict[str, Any] = dict(label=label, seg=seg, events=k, patient_days=pdays,
                                    suppressed=(k < MIN_EVENTS) and pdays > 0, empty=(pdays == 0))
        if pdays > 0 and k >= MIN_EVENTS:
            cell["rate"], cell["lo"], cell["hi"] = A3.rate_per_100k(k, pdays)
        cells.append(cell)
    return cells


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="A8: selection among those still admitted (descriptive).")
    ap.parse_args(argv)
    out_dir, json_path = R1.TABLES_OUT, R1.FRAGMENTS_OUT / "results_a8.json"

    df = R1.load_cohort()
    n_all = len(df)
    ev = R1.make_events(df, "WINDOW_COMP", "POD_WINDOW")
    assert ev.n_events == 1015 and ev.n_untimed_events == 0

    # ---- the conditioning set and its status split, every day -----------------------------------
    everyone = np.ones(n_all, dtype=bool)
    at_all, in_h, dis, miss = W.riskset_composition(everyone, ev.exit_day, ev.los)   # asserts the partition
    assert np.array_equal(in_h + dis + miss, at_all)
    fit0 = R1.fit_overall_and_strata(df, ev, bootstrap=False)[0]       # no random draw; checks the denominator
    assert fit0.slug == "overall" and np.array_equal(np.rint(fit0.at_risk).astype(np.int64), at_all)
    assert int(miss[0]) == 83 and int(at_all[0]) == n_all
    assert int(dis[0]) == 0, "nobody can be already discharged at the start of day 0"
    at_status = np.vstack([in_h, dis, miss])                          # (3, 31) patient-days by status
    ev_status = events_by_status_and_day(ev)
    e1_all = np.bincount(ev.pod_event[ev.is_event].astype(int), minlength=DAYS)
    assert np.array_equal(ev_status.sum(axis=0), e1_all) and int(e1_all.sum()) == 1015, \
        "events by status do not partition the composite events"

    # ---- S11A: band cells by status, rate ratios ------------------------------------------------
    cells = {seg: status_band_cells(at_status[i], ev_status[i]) for i, (seg, _) in enumerate(STATUS)}
    for i, (seg, _) in enumerate(STATUS):
        assert sum(c["events"] for c in cells[seg]) == int(ev_status[i].sum())
    for b, (first, last, label, bseg) in enumerate(BANDS):                 # partition within each band
        assert sum(cells[seg][b]["patient_days"] for seg, _ in STATUS) == int(at_all[first:last + 1].sum())
        assert sum(cells[seg][b]["events"] for seg, _ in STATUS) == int(e1_all[first:last + 1].sum())
    ratios: List[Dict[str, Any]] = []
    for b, (first, last, label, bseg) in enumerate(BANDS):
        ih, dc = cells["inhosp"][b], cells["discharged"][b]
        rec: Dict[str, Any] = dict(label=label, seg=bseg)
        if dc["empty"] or ih["empty"]:
            rec["status"] = "not_applicable"
        elif ih["events"] < MIN_EVENTS or dc["events"] < MIN_EVENTS:
            rec["status"] = "suppressed"
        else:
            rec["status"] = "shown"
            rec["rr"], rec["lo"], rec["hi"] = rate_ratio_exact(ih["events"], ih["patient_days"],
                                                               dc["events"], dc["patient_days"])
        ratios.append(rec)

    # ---- S11B: composition of the conditioning set at days 1, 3, 7 ------------------------------
    def level_mask(column: str, raw: str) -> np.ndarray:
        return (df[column].astype("string") == raw).fillna(False).to_numpy()
    age75 = level_mask("AGE_CAT", ">=75")
    asa34 = level_mask("ASA_CAT", ">=3")
    lev3 = level_mask("LEVELS_CAT", ">=3")
    for col in ("AGE_CAT", "ASA_CAT", "LEVELS_CAT"):
        assert df[col].notna().all(), f"{col} has missing values; the shares would need a denominator rule"
    comp: Dict[int, Dict[str, Dict[str, Any]]] = {}
    los_all = ev.los
    recorded_all = np.isfinite(los_all)
    for d in COMPOSITION_DAYS:
        alive = ev.exit_day >= d
        with np.errstate(invalid="ignore"):
            groups = {"all": alive,
                      "inhosp": alive & recorded_all & (los_all >= d),
                      "discharged": alive & recorded_all & (los_all < d),
                      "notrecorded": alive & ~recorded_all}
        assert int(groups["all"].sum()) == int(at_all[d])
        assert int(groups["inhosp"].sum()) == int(in_h[d]) and int(groups["discharged"].sum()) == int(dis[d]) \
            and int(groups["notrecorded"].sum()) == int(miss[d])
        assert int(groups["inhosp"].sum() + groups["discharged"].sum() + groups["notrecorded"].sum()) \
            == int(groups["all"].sum()), "status groups do not partition the set"
        comp[d] = {}
        total = int(groups["all"].sum())
        for seg, m in groups.items():
            n = int(m.sum())
            comp[d][seg] = dict(n=n, share_of_set_pct=100.0 * n / total,
                                age75_n=int((m & age75).sum()), asa34_n=int((m & asa34).sum()),
                                lev3_n=int((m & lev3).sum()))
        # the locked strata sizes tie to the full cohort at day 0 by construction; here tie the "all" group
        assert comp[d]["all"]["n"] == total

    # ---- INOUT and same-day counts ---------------------------------------------------------------
    inout = df["INOUT"].astype("string")
    outpatient = (inout == VD.INOUT_OUTPATIENT_LEVEL).fillna(False).to_numpy()
    inpatient = (inout == "Inpatient").fillna(False).to_numpy()
    assert int((outpatient | inpatient).sum()) == n_all and not bool((outpatient & inpatient).any()), \
        "INOUT is not exactly Outpatient or Inpatient"
    assert int(outpatient.sum()) == int((df["INOUT_OUTPT"] == 1).sum()), "INOUT_OUTPT disagrees with INOUT"
    los_series = df["LOS_POSTOP"].to_numpy(dtype=float, na_value=np.nan)
    rec_stay = np.isfinite(los_series)
    with np.errstate(invalid="ignore"):
        same_day = rec_stay & (los_series == 0)
        overnight = rec_stay & (los_series >= 1)
    io: Dict[str, int] = dict(
        total_n=n_all, outpatient_n=int(outpatient.sum()), inpatient_n=int(inpatient.sum()),
        recorded_stay_n=int(rec_stay.sum()), outpatient_recorded_stay_n=int((outpatient & rec_stay).sum()),
        outpatient_stay_not_recorded_n=int((outpatient & ~rec_stay).sum()),
        same_day_n=int(same_day.sum()), same_day_of_outpatient_n=int((same_day & outpatient).sum()),
        same_day_of_inpatient_n=int((same_day & inpatient).sum()),
        outpatient_overnight_n=int((outpatient & overnight).sum()),
        inpatient_recorded_stay_n=int((inpatient & rec_stay).sum()))
    assert io["same_day_of_outpatient_n"] + io["same_day_of_inpatient_n"] == io["same_day_n"]
    assert io["outpatient_n"] + io["inpatient_n"] == n_all
    assert io["outpatient_overnight_n"] + io["same_day_of_outpatient_n"] == io["outpatient_recorded_stay_n"]
    quoted_outpatient, quoted_same_day = 46042, 14715      # the counts quoted in the paper, compared here, never relied on

    # ---------------- results_a8.json ------------------------------------------------------------
    res: Dict[str, Any] = {}
    res["a8.cohort.n"] = n_all
    res["a8.composite.events_n"] = ev.n_events
    res["a8.stay_not_recorded.patients_n"] = int(miss[0])
    res["a8.band.min_cell_events_n"] = MIN_EVENTS
    for i, (seg, _) in enumerate(STATUS):
        for c in cells[seg]:
            p = f"a8.band.{c['seg']}.{seg}"
            res[f"{p}.events_n"] = c["events"]
            res[f"{p}.patient_days_n"] = c["patient_days"]
            res[f"{p}.suppressed"] = bool(c["suppressed"])
            if "rate" in c:
                res[f"{p}.rate_per100k"] = A3.r6(c["rate"])
                res[f"{p}.rate_per100k_lo"] = A3.r6(c["lo"])
                res[f"{p}.rate_per100k_hi"] = A3.r6(c["hi"])
    for r in ratios:
        p = f"a8.ratio.{r['seg']}"
        res[f"{p}.available"] = r["status"] == "shown"
        if r["status"] == "shown":
            res[f"{p}.rate_ratio"] = A3.r6(r["rr"])
            res[f"{p}.rate_ratio_lo"] = A3.r6(r["lo"])
            res[f"{p}.rate_ratio_hi"] = A3.r6(r["hi"])
    for d in COMPOSITION_DAYS:
        for seg, rec in comp[d].items():
            p = f"a8.composition.day{d}.{seg}"
            res[f"{p}.n"] = rec["n"]
            res[f"{p}.share_of_set_pct"] = A3.r6(rec["share_of_set_pct"])
            for key, num in (("age_ge75", "age75_n"), ("asa_ge3", "asa34_n"), ("levels_ge3", "lev3_n")):
                res[f"{p}.{key}_n"] = rec[num]
                res[f"{p}.{key}_pct"] = A3.r6(100.0 * rec[num] / rec["n"]) if rec["n"] else None
    for key, val in io.items():
        res[f"a8.inout.{key}"] = val
    res["a8.inout.outpatient_same_day_pct_of_outpatient_recorded"] = A3.r6(
        100.0 * io["same_day_of_outpatient_n"] / io["outpatient_recorded_stay_n"])
    res["a8.inout.outpatient_label_pct_of_same_day"] = A3.r6(
        100.0 * io["same_day_of_outpatient_n"] / io["same_day_n"])
    res["a8.inout.outpatient_pct_of_cohort"] = A3.r6(100.0 * io["outpatient_n"] / n_all)
    res["a8.inout.same_day_pct_of_recorded"] = A3.r6(100.0 * io["same_day_n"] / io["recorded_stay_n"])
    res["a8.inout.quoted_outpatient_n"] = quoted_outpatient
    res["a8.inout.quoted_same_day_n"] = quoted_same_day
    res["a8.inout.quoted_outpatient_equals_all_labelled"] = io["outpatient_n"] == quoted_outpatient
    res["a8.inout.quoted_outpatient_equals_with_recorded_stay"] = io["outpatient_recorded_stay_n"] == quoted_outpatient
    res["a8.inout.quoted_same_day_equals_measured"] = io["same_day_n"] == quoted_same_day
    json_n = R1.write_fragment(json_path, res, ("a8.",))

    # ---------------- Table S11 (panel A and panel B in one file) ----------------------------------
    head = ["Panel", "Postoperative days or start of day", "Hospital status at the start of the day",
            "Events, n", "Patient-days at risk, n", "Events per 100,000 patient-days (95% CI)",
            "Rate ratio, in hospital over discharged (95% CI)",
            "Patients event free and alive, n", "Share of the event-free and alive set, %",
            f"{VD.pretty('AGE_CAT')}: {VD.pretty_level('AGE_CAT', '>=75')}, n (%)",
            f"{VD.pretty('ASA_CAT')}: {VD.pretty_level('ASA_CAT', '>=3')}, n (%)",
            f"{VD.pretty('LEVELS_CAT')}: {VD.pretty_level('LEVELS_CAT', '>=3')}, n (%)"]
    rows: List[List[Any]] = []
    for b, (first, last, label, bseg) in enumerate(BANDS):
        for i, (seg, status_label) in enumerate(STATUS):
            c = cells[seg][b]
            if c["empty"]:
                rate_txt = NOT_APPLICABLE
            elif c["suppressed"]:
                rate_txt = SUPPRESSED
            else:
                rate_txt = A3.fmt_rate(c["rate"], c["lo"], c["hi"])
            ratio_txt = ""
            if seg == "inhosp":
                r = ratios[b]
                if r["status"] == "shown":
                    ratio_txt = f"{r['rr']:.2f} ({r['lo']:.2f} to {r['hi']:.2f})"
                elif r["status"] == "suppressed":
                    ratio_txt = SUPPRESSED
                else:
                    ratio_txt = NOT_APPLICABLE
            rows.append(["A", label, status_label, VD.fmt_count(c["events"]), VD.fmt_count(c["patient_days"]),
                         rate_txt, ratio_txt, "", "", "", "", ""])
    for d in COMPOSITION_DAYS:
        for seg, glabel in (("all", "All patients event free and alive"),) + STATUS:
            rec = comp[d][seg]
            n = rec["n"]

            def cell(k: int) -> str:
                return f"{VD.fmt_count(k)} ({100.0 * k / n:.1f})" if n else NOT_APPLICABLE
            rows.append(["B", f"Start of day {d}", glabel, "", "", "", "", VD.fmt_count(n),
                         f"{rec['share_of_set_pct']:.1f}", cell(rec["age75_n"]), cell(rec["asa34_n"]),
                         cell(rec["lev3_n"])])
    R1.write_csv_table(out_dir / "tableS11_selection_still_admitted.csv", head, rows)

    print(f"[a8] wrote {json_n} keys and 1 table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
