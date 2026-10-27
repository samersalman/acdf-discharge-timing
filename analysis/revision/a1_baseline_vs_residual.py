#!/usr/bin/env python3
"""a1_baseline_vs_residual.py - A1 (baseline versus residual risk) and A6 (paired bootstrap on the
spread of threshold days).

Run:  PYTHONDONTWRITEBYTECODE=1 python3 a1_baseline_vs_residual.py

What it does (composite WINDOW_COMP, POD_WINDOW, death competing, 13 strata in v1 order):
  1. Fits the 13 strata with r1_common.fit_overall_and_strata, which reseeds with VD.SEED and makes
     exactly the v1 bootstrap draws, so the bootstrap intervals of the absolute 0.5% and 0.25%
     threshold days must equal Table 5 of v1. The script ASSERTS that equality (and several more
     Table 5 columns) before it does anything else.
  2. A1 relative residual risk rho_s(t) = R_s(t) / R_s(0) with 95% bootstrap percentile bands, and
     the first day below 50%, 25% and 10% of the stratum's own baseline, with bootstrap intervals.
  3. A1 decomposition: the baseline-only predicted curve R_s(0) * rho_overall(t), its predicted
     threshold day at 1%, 0.5% and 0.25%, and the observed minus predicted difference.
  4. A1 decision metric (fixed in advance): at 0.25% the two eligible strata with the
     largest and smallest observed day, f = baseline-only spread / observed spread, with a bootstrap
     interval, and the mechanical category from the POINT estimate (reframe_baseline if f >= 2/3,
     keep_different_rates if f < 1/3, otherwise both_contributions).
  5. A6 paired bootstrap on the difference and ratio of threshold days (and on the baseline
     30-day cumulative incidence) for the age 75 or older versus 18 to 49 pair and every adjacent
     pair within a stratifier. Replicate-index pairing is valid because the strata of one
     stratifier are disjoint patient sets and v1 draws them independently.
  6. Sensitivity (not part of the decision): a JOINT patient-level bootstrap within one stratifier,
     which keeps the overall curve and the stratum curves correlated (a separate RandomState, so the
     v1 stream is untouched), for the f interval.

Needs tables-csv/table5_safe_discharge_day.csv of the analysis folder (written by step 06).

Writes (only these, under the results directory): tables-csv/tableS5_baseline_vs_residual.csv,
tables-csv/tableS5_spread_intervals.csv, figures/figS1_relative_residual_risk.png and .pdf (skipped
when ACDF_FIGURES=0), fragments/results_a1.json (namespaces a1. and a6.).
Replicate arrays stay in memory and are never written. No patient-level data is written.
"""

import sys; sys.dont_write_bytecode = True
from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))

import csv
import math
from fractions import Fraction

import numpy as np

import r1_common as R1

VD, W = R1.VD, R1.W

ABS_THR = tuple(VD.RISK_THRESHOLDS)            # 0.01, 0.005, 0.0025 (v1)
REL_THR = (0.5, 0.25, 0.10)                    # fractions of the stratum's own baseline
REL_SLUG = {0.5: "rel_50pct", 0.25: "rel_25pct", 0.10: "rel_10pct"}
REL_TEXT = {0.5: "50%", 0.25: "25%", 0.10: "10%"}
THR_05, THR_025 = 0.005, 0.0025
REPS = W.BOOTSTRAP_REPLICATES                  # 1,000
NOT_REACHED, NOT_APPLICABLE, SUPPRESSED = W.NOT_REACHED, W.NOT_APPLICABLE, W.SUPPRESSED_TEXT
EN = chr(0x2013)

# Direction of every A6 pair: the FIRST stratum minus the SECOND. The second stratum is the one
# expected to carry the lower risk (younger, lower ASA class, fewer levels, female).
PAIRS = (
    ("age-ge75-vs-18-49", "age.ge75", "age.18-49"),     # the "sevenfold" pair (14 days against 2)
    ("age-50-64-vs-18-49", "age.50-64", "age.18-49"),
    ("age-65-74-vs-50-64", "age.65-74", "age.50-64"),
    ("age-ge75-vs-65-74", "age.ge75", "age.65-74"),
    ("asa-ge3-vs-1-2", "asa.ge3", "asa.1-2"),
    ("levels-2-vs-1", "levels.2", "levels.1"),
    ("levels-ge3-vs-2", "levels.ge3", "levels.2"),
    ("sex-male-vs-female", "sex.male", "sex.female"),
)
HEADLINE_PAIR = "age-ge75-vs-18-49"


# ---------------------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------------------

def ci_ints(lo, hi):
    """Table 5 convention (W.day_ci_text): floor the lower and ceil the upper percentile.
    Returns (lo_int, hi_int); None means 'not reached within 30 days' (beyond day 30)."""
    lo_i, hi_i = int(math.floor(lo)), int(math.ceil(hi))
    return (None if lo_i > VD.FOLLOWUP_DAYS else lo_i), (None if hi_i > VD.FOLLOWUP_DAYS else hi_i)


def day_cell(day, boot_days):
    """'8 (6 to 12)' in the style of Table 5; the not-reached text when the point day is beyond 30."""
    if day > VD.FOLLOWUP_DAYS:
        return NOT_REACHED
    lo, hi = W.percentile_ci(np.asarray(boot_days, dtype=float))
    return f"{day} ({W.day_ci_text(lo, hi)})"


def yes_no(flag):
    return "Yes" if flag else "No"


def signed(n):
    return str(int(n))


def f_text(x):
    return f"{x:.2f}"


def threshold_key(t):
    return R1.threshold_slug(t)


def put(res, key, value):
    if key in res:
        raise R1.R1Error(f"duplicate key {key}")
    res[key] = value


# ---------------------------------------------------------------------------------------------
# Table 5 equality (the free consistency test against v1)
# ---------------------------------------------------------------------------------------------

def assert_table5_equal(fits):
    """Every estimated stratum, every v1 threshold: day, day interval, 30-day incidence, residual risk
    and its interval, risk-set size and the stays-below flag must equal the printed v1 Table 5."""
    path = R1.ORIG_TABLES / "table5_safe_discharge_day.csv"
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    index = {(r["Stratifier"], r["Stratum"], r["Risk threshold"]): r for r in rows}
    compared = 0
    for f in fits:
        if f.suppressed:
            continue
        s = f.stratum
        for t in ABS_THR:
            row = index[(s.stratifier, s.level, VD.threshold_label(t))]
            day = f.threshold_day(t)
            lo, hi = W.percentile_ci(f.boot_threshold_days(t).astype(float))
            want = {
                "Cumulative incidence of the composite to day 30 (%)": W.pct(f.cif30),
                "Residual-risk threshold day": W.day_text(day),
                "Residual-risk threshold day, 95% confidence interval": W.day_ci_text(lo, hi),
                "Residual risk at that day (%)": W.pct(float(f.curve[day])),
                "Residual risk, 95% confidence interval (%)": W.pct_ci(float(f.boot.lo[day]), float(f.boot.hi[day])),
                "Patients event free and alive at the start of that day": VD.fmt_count(round(float(f.at_risk[day]))),
                "Residual risk remains below the threshold through day 30":
                    yes_no(R1.stays_below_from(f.curve, t, day)),
            }
            for col, text in want.items():
                if row[col] != text:
                    raise R1.R1Error(f"Table 5 mismatch for {s.stratifier} / {s.level} at "
                                     f"{VD.threshold_label(t)}, column {col!r}: v1 prints "
                                     f"{row[col]!r}, this script gets {text!r}")
                compared += 1
    return compared


# ---------------------------------------------------------------------------------------------
# the joint patient-level bootstrap (sensitivity only)
# ---------------------------------------------------------------------------------------------

def joint_curves(fits, events, column, seed, n_total):
    """Resample ALL patients once per replicate over the cells of one stratifier, so that the overall
    curve and the stratum curves of that stratifier come from the same resampled cohort.
    Returns {slug: (B, 31) conditional-risk curves} for the estimated strata and 'overall'."""
    members = [f for f in fits if f.stratum.column == column]
    blocks = []
    for f in members:
        e1, e2 = W.counts_by_day(events.pod_event[f.mask & events.is_event],
                                 events.pod_death[f.mask & events.is_competing])
        free = f.n - int(e1.sum()) - int(e2.sum())
        blocks.append(np.concatenate([e1, e2, [float(free)]]))
    cells = np.concatenate(blocks)
    if int(cells.sum()) != n_total:
        raise R1.R1Error("joint bootstrap cells do not add up to the cohort")
    probs = cells / cells.sum()
    probs = probs / probs.sum()
    rng = np.random.RandomState(seed)               # NOT the global stream
    draws = rng.multinomial(n_total, probs, size=REPS).astype(float)
    width = 2 * W.DAYS + 1
    tot1 = np.zeros((REPS, W.DAYS))
    tot2 = np.zeros((REPS, W.DAYS))
    out = {}
    for k, f in enumerate(members):
        blk = draws[:, k * width:(k + 1) * width]
        b1, b2 = blk[:, :W.DAYS], blk[:, W.DAYS:2 * W.DAYS]
        tot1 += b1
        tot2 += b2
        if f.suppressed:
            continue
        cif, surv, _ = W.aalen_johansen(b1, b2, blk.sum(axis=1))
        out[f.slug] = W.conditional_risk_aj(cif, surv)
    cif, surv, _ = W.aalen_johansen(tot1, tot2, np.full(REPS, float(n_total)))
    out["overall"] = W.conditional_risk_aj(cif, surv)
    return out


# ---------------------------------------------------------------------------------------------
# the decision metric
# ---------------------------------------------------------------------------------------------

def pick_extremes(members, obs_day, threshold):
    """(hi, lo, tie_hi_slugs, tie_lo_slugs) among `members` (StratumFit list) at one threshold.
    Largest observed day = hi, smallest = lo; ties go to the LOWER display order."""
    days = {f.slug: obs_day[f.slug][threshold] for f in members}
    if any(d > VD.FOLLOWUP_DAYS for d in days.values()):
        raise R1.R1Error("a stratum never reached the threshold; the observed spread is undefined")
    top, bottom = max(days.values()), min(days.values())
    ties_hi = [f for f in members if days[f.slug] == top]
    ties_lo = [f for f in members if days[f.slug] == bottom]
    hi = min(ties_hi, key=lambda f: f.stratum.order)
    lo = min(ties_lo, key=lambda f: f.stratum.order)
    if hi.slug == lo.slug:
        raise R1.R1Error("the largest and smallest observed days belong to the same stratum (no spread)")
    return hi, lo, [f.slug for f in ties_hi], [f.slug for f in ties_lo]


def f_replicates(obs_hi, obs_lo, pred_hi, pred_lo):
    """Per-replicate f = (pred_hi - pred_lo) / (obs_hi - obs_lo) with exclusions.
    Excluded: any of the four days not reached; then a denominator below 1."""
    unreached = ((obs_hi > VD.FOLLOWUP_DAYS) | (obs_lo > VD.FOLLOWUP_DAYS)
                 | (pred_hi > VD.FOLLOWUP_DAYS) | (pred_lo > VD.FOLLOWUP_DAYS))
    den = (obs_hi - obs_lo).astype(float)
    bad = (~unreached) & (den < 1)
    ok = (~unreached) & (~bad)
    f = (pred_hi[ok] - pred_lo[ok]).astype(float) / den[ok]
    return f, int(unreached.sum()), int(bad.sum())


def f_summary(f):
    lo, hi = W.percentile_ci(f)
    return float(lo), float(np.median(f)), float(hi)


def f_cut_shares(f):
    """Share of the replicates (the ones used for the interval) with f at least 2/3, below 1/3, and above 1.
    The float cut points are the correctly rounded 2/3 and 1/3, the same as the Fraction cuts of the category."""
    f = np.asarray(f, dtype=float)
    return (float(np.mean(f >= float(Fraction(2, 3)))), float(np.mean(f < float(Fraction(1, 3)))),
            float(np.mean(f > 1.0)))


# ---------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------

def main():
    R1.seed_all()
    df = R1.load_cohort()
    comp = R1.make_events(df, "WINDOW_COMP", "POD_WINDOW")
    if comp.n_events != 1015 or comp.n_untimed_events != 0:
        raise R1.R1Error("the composite is not the 1,015 timed events of v1")
    fits = R1.fit_overall_and_strata(df, comp)          # reseeds with VD.SEED: v1's draws
    idx = R1.index_by_slug(fits)
    n_cells = assert_table5_equal(fits)
    print(f"[a1] Table 5 equality: {n_cells} printed cells equal v1 (days, intervals, incidence, risk, risk set, stays-below)")

    ov = idx["overall"]
    est = [f for f in fits if not f.suppressed]
    res = {}

    put(res, "a1.boot.replicates_n", REPS)
    put(res, "a1.boot.seed", int(VD.SEED))
    put(res, "a1.check.table5_cells_compared_n", n_cells)
    put(res, "a1.check.table5_cells_equal_v1", True)
    put(res, "a1.cohort.n", int(ov.n))
    put(res, "a1.cohort.events_n", int(ov.events))

    # ---- curves, observed and predicted -----------------------------------------------------
    rho_ov = ov.curve / ov.curve[0]
    rho_ov_b = ov.boot.curve / ov.boot.curve[:, [0]]
    obs_day, obs_b = {}, {}          # [slug][thr] -> int / (B,) array (absolute thresholds)
    prd_day, prd_b, prd_curve = {}, {}, {}
    rel_day, rel_b = {}, {}
    rho, rho_b, rho_lo, rho_hi = {}, {}, {}, {}
    for f in est:
        if not np.all(f.boot.curve[:, 0] > 0):
            raise R1.R1Error(f"{f.slug}: a bootstrap replicate has zero baseline risk")
        rho[f.slug] = f.curve / f.curve[0]
        rho_b[f.slug] = f.boot.curve / f.boot.curve[:, [0]]
        rho_lo[f.slug], rho_hi[f.slug] = W.percentile_ci(rho_b[f.slug], axis=0)
        obs_day[f.slug] = {t: f.threshold_day(t) for t in ABS_THR}
        obs_b[f.slug] = {t: f.boot_threshold_days(t) for t in ABS_THR}
        if f.slug == "overall":
            prd_curve[f.slug] = f.curve                         # the overall curve defines the prediction
            pred_b_curve = f.boot.curve
        else:
            prd_curve[f.slug] = f.curve[0] * rho_ov
            pred_b_curve = f.boot.curve[:, [0]] * rho_ov_b
        prd_day[f.slug] = {t: R1.threshold_day(prd_curve[f.slug], t) for t in ABS_THR}
        prd_b[f.slug] = {t: R1.threshold_day(pred_b_curve, t) for t in ABS_THR}
        rel_day[f.slug] = {t: R1.threshold_day(rho[f.slug], t) for t in REL_THR}
        rel_b[f.slug] = {t: R1.threshold_day(rho_b[f.slug], t) for t in REL_THR}

    # ---- per-stratum keys -------------------------------------------------------------------
    for f in fits:
        p = f"a1.{f.slug}."
        put(res, p + "patients_n", int(f.n))
        put(res, p + "events_n", int(f.events))
        put(res, p + "suppressed", bool(f.suppressed))
        if f.suppressed:
            continue
        put(res, p + "cif30_pct", 100.0 * f.cif30)
        put(res, p + "cif30_lo_pct", 100.0 * f.boot.cif30_lo)
        put(res, p + "cif30_hi_pct", 100.0 * f.boot.cif30_hi)
        put(res, p + "rho_est_prop", rho[f.slug])
        put(res, p + "rho_lo_prop", rho_lo[f.slug])
        put(res, p + "rho_hi_prop", rho_hi[f.slug])
        rises = [d for d in range(1, W.DAYS) if rho[f.slug][d] > rho[f.slug][d - 1] + 1e-15]
        put(res, p + "rho_rise_n", len(rises))
        if rises:
            put(res, p + "rho_rise_days", rises)
        for t in ABS_THR:
            q = p + threshold_key(t) + "."
            day = obs_day[f.slug][t]
            lo, hi = W.percentile_ci(obs_b[f.slug][t].astype(float))
            lo_i, hi_i = ci_ints(lo, hi)
            put(res, q + "day", R1.day_or_none(day))
            put(res, q + "ci_lo_day", lo_i)
            put(res, q + "ci_hi_day", hi_i)
            put(res, q + "stays_below", R1.stays_below_from(f.curve, t, day))
            pday = prd_day[f.slug][t]
            plo, phi = W.percentile_ci(prd_b[f.slug][t].astype(float))
            plo_i, phi_i = ci_ints(plo, phi)
            put(res, q + "pred_day", R1.day_or_none(pday))
            put(res, q + "pred_ci_lo_day", plo_i)
            put(res, q + "pred_ci_hi_day", phi_i)
            put(res, q + "pred_stays_below", R1.stays_below_from(prd_curve[f.slug], t, pday))
            if day <= VD.FOLLOWUP_DAYS and pday <= VD.FOLLOWUP_DAYS:
                put(res, q + "obs_minus_pred_days", int(day - pday))
                both = ((obs_b[f.slug][t] <= VD.FOLLOWUP_DAYS) & (prd_b[f.slug][t] <= VD.FOLLOWUP_DAYS))
                dd = (obs_b[f.slug][t] - prd_b[f.slug][t])[both].astype(float)
                if f.slug != "overall":
                    dlo, dhi = W.percentile_ci(dd)
                    put(res, q + "obs_minus_pred_ci_lo_days", int(math.floor(dlo)))
                    put(res, q + "obs_minus_pred_ci_hi_days", int(math.ceil(dhi)))
                    put(res, q + "obs_minus_pred_reps_n", int(both.sum()))
        for rt in REL_THR:
            q = p + REL_SLUG[rt] + "."
            day = rel_day[f.slug][rt]
            lo, hi = W.percentile_ci(rel_b[f.slug][rt].astype(float))
            lo_i, hi_i = ci_ints(lo, hi)
            put(res, q + "day", R1.day_or_none(day))
            put(res, q + "ci_lo_day", lo_i)
            put(res, q + "ci_hi_day", hi_i)
            put(res, q + "stays_below", R1.stays_below_from(rho[f.slug], rt, day))

    # ---- the decision metric ----------------------------------------------------------------
    eligible = [f for f in est if f.slug != "overall" and not f.stratum.is_residual_level]
    stratifier_slugs = []
    for f in eligible:
        sl = f.slug.split(".")[0]
        if sl not in stratifier_slugs:
            stratifier_slugs.append(sl)
    n_eligible = len(eligible)
    put(res, "a1.f.eligible_strata_n", n_eligible)

    f_specs = []            # (key, thr, hi, lo)
    decision = None
    for t in (THR_025, THR_05):
        groups = [("all", eligible)] + [(sl, [f for f in eligible if f.slug.split(".")[0] == sl])
                                       for sl in stratifier_slugs]
        for gname, members in groups:
            hi, lo, ties_hi, ties_lo = pick_extremes(members, obs_day, t)
            d_hi, d_lo = obs_day[hi.slug][t], obs_day[lo.slug][t]
            p_hi, p_lo = prd_day[hi.slug][t], prd_day[lo.slug][t]
            if p_hi > VD.FOLLOWUP_DAYS or p_lo > VD.FOLLOWUP_DAYS:
                raise R1.R1Error(f"predicted threshold day beyond 30 for the pair {hi.slug}, {lo.slug}")
            f_pt = Fraction(p_hi - p_lo, d_hi - d_lo)
            fr, n_un, n_bad = f_replicates(obs_b[hi.slug][t], obs_b[lo.slug][t],
                                           prd_b[hi.slug][t], prd_b[lo.slug][t])
            q = f"a1.f.{gname}.{threshold_key(t)}."
            f_lo, f_med, f_hi = f_summary(fr)
            put(res, q + "f_prop", float(f_pt))
            put(res, q + "f_lo_prop", f_lo)
            put(res, q + "f_median_prop", f_med)
            put(res, q + "f_hi_prop", f_hi)
            put(res, q + "reps_used_n", int(len(fr)))
            put(res, q + "reps_excluded_unreached_n", n_un)
            put(res, q + "reps_excluded_denominator_n", n_bad)
            put(res, q + "hi_stratum_slug", hi.slug)
            put(res, q + "lo_stratum_slug", lo.slug)
            put(res, q + "hi_stratum_label", f"{hi.stratum.stratifier}: {hi.stratum.level}")
            put(res, q + "lo_stratum_label", f"{lo.stratum.stratifier}: {lo.stratum.level}")
            put(res, q + "obs_hi_day", int(d_hi))
            put(res, q + "obs_lo_day", int(d_lo))
            put(res, q + "pred_hi_day", int(p_hi))
            put(res, q + "pred_lo_day", int(p_lo))
            put(res, q + "obs_spread_days", int(d_hi - d_lo))
            put(res, q + "pred_spread_days", int(p_hi - p_lo))
            put(res, q + "tie_hi_n", len(ties_hi))
            put(res, q + "tie_lo_n", len(ties_lo))
            put(res, q + "tie_hi_slugs", ties_hi)
            put(res, q + "tie_lo_slugs", ties_lo)
            if gname == "all" and t == THR_05:
                p_ge, p_lt, p_gt = f_cut_shares(fr)
                put(res, q + "p_f_ge_two_thirds_prop", p_ge)
                put(res, q + "p_f_lt_one_third_prop", p_lt)
                put(res, q + "p_f_gt_one_prop", p_gt)
            f_specs.append((q, t, hi, lo, fr))
            if gname == "all" and t == THR_025:
                decision = dict(f=f_pt, hi=hi, lo=lo, d_hi=d_hi, d_lo=d_lo, p_hi=p_hi, p_lo=p_lo,
                                lo_ci=f_lo, med=f_med, hi_ci=f_hi, used=int(len(fr)),
                                ties_hi=ties_hi, ties_lo=ties_lo, n_un=n_un, n_bad=n_bad, fr=fr)

    two_thirds, one_third = Fraction(2, 3), Fraction(1, 3)
    f_head = decision["f"]
    if f_head >= two_thirds:
        category = "reframe_baseline"
    elif f_head < one_third:
        category = "keep_different_rates"
    else:
        category = "both_contributions"
    put(res, "a1.decision.category", category)
    put(res, "a1.decision.f_headline_prop", float(f_head))
    put(res, "a1.decision.f_headline_num_days", int(decision["p_hi"] - decision["p_lo"]))
    put(res, "a1.decision.f_headline_den_days", int(decision["d_hi"] - decision["d_lo"]))
    put(res, "a1.decision.f_headline_lo_prop", decision["lo_ci"])
    put(res, "a1.decision.f_headline_median_prop", decision["med"])
    put(res, "a1.decision.f_headline_hi_prop", decision["hi_ci"])
    put(res, "a1.decision.f_headline_reps_used_n", decision["used"])
    put(res, "a1.decision.cut_reframe_prop", float(two_thirds))
    put(res, "a1.decision.cut_keep_prop", float(one_third))
    put(res, "a1.decision.hi_stratum_slug", decision["hi"].slug)
    put(res, "a1.decision.lo_stratum_slug", decision["lo"].slug)
    put(res, "a1.decision.threshold_pct", 0.25)
    put(res, "a1.decision.hi_lo_pair_is_headline_age_pair",
        bool(decision["hi"].slug == "age.ge75" and decision["lo"].slug == "age.18-49"))
    put(res, "a1.decision.interval_crosses_cut_reframe",
        bool(decision["lo_ci"] < float(two_thirds) <= decision["hi_ci"]))
    put(res, "a1.decision.interval_crosses_cut_keep",
        bool(decision["lo_ci"] < float(one_third) <= decision["hi_ci"]))
    d_ge, d_lt, d_gt = f_cut_shares(decision["fr"])
    put(res, "a1.decision.p_f_ge_two_thirds_prop", d_ge)
    put(res, "a1.decision.p_f_lt_one_third_prop", d_lt)
    put(res, "a1.decision.p_f_gt_one_prop", d_gt)
    # the pair is picked from the OBSERVED extremes and then held fixed; the bootstrap interval ignores that selection
    put(res, "a1.decision.pair_chosen_from_observed_extremes", True)

    # ---- sensitivity: joint patient-level bootstrap within the stratifier of each pair -----------
    column_of = {"age": "AGE_CAT", "asa": "ASA_CAT", "levels": "LEVELS_CAT", "sex": "MALE"}
    assert tuple(column_of.values()) == tuple(VD.STRATIFIER_COLUMNS) or set(column_of.values()) == set(VD.STRATIFIER_COLUMNS)
    joint = {}
    for sl, col in column_of.items():
        joint[sl] = joint_curves(fits, comp, col, VD.SEED + 1 + list(column_of).index(sl), len(df))
    joint_headline = None
    for (q, t, hi, lo, fr_ind) in f_specs:
        s_hi, s_lo = hi.slug.split(".")[0], lo.slug.split(".")[0]
        jq = q.replace("a1.f.", "a1.sens_joint.", 1)
        if s_hi != s_lo:
            put(res, jq + "computed", False)
            continue
        jc = joint[s_hi]
        rho_j = jc["overall"] / jc["overall"][:, [0]]
        od = {}
        pd_ = {}
        for f in (hi, lo):
            od[f.slug] = R1.threshold_day(jc[f.slug], t)
            pd_[f.slug] = R1.threshold_day(jc[f.slug][:, [0]] * rho_j, t)
        fr, n_un, n_bad = f_replicates(od[hi.slug], od[lo.slug], pd_[hi.slug], pd_[lo.slug])
        lo_j, med_j, hi_j = f_summary(fr)
        put(res, jq + "computed", True)
        put(res, jq + "f_lo_prop", lo_j)
        put(res, jq + "f_median_prop", med_j)
        put(res, jq + "f_hi_prop", hi_j)
        put(res, jq + "reps_used_n", int(len(fr)))
        put(res, jq + "reps_excluded_unreached_n", n_un)
        put(res, jq + "reps_excluded_denominator_n", n_bad)
        if q == "a1.f.all." + threshold_key(THR_025) + ".":
            joint_headline = (lo_j, med_j, hi_j, int(len(fr)))
    put(res, "a1.sens_joint.seed_base", int(VD.SEED + 1))

    # ---- A6: paired bootstrap on the spread -------------------------------------------------------
    rows_b = []
    zero_pairs = {THR_05: [], THR_025: []}
    for pid, sa, sb in PAIRS:
        a, b = idx[sa], idx[sb]
        label_a, label_b = a.stratum.level, b.stratum.level
        comparison = f"{label_a} minus {label_b}"
        pq = f"a6.{pid}."
        put(res, pq + "first_stratum_slug", sa)
        put(res, pq + "second_stratum_slug", sb)
        put(res, pq + "first_label", f"{a.stratum.stratifier}: {label_a}")
        put(res, pq + "second_label", f"{b.stratum.stratifier}: {label_b}")
        for t in (THR_05, THR_025):
            da, db = obs_day[sa][t], obs_day[sb][t]
            ba, bb = obs_b[sa][t], obs_b[sb][t]
            valid = (ba <= VD.FOLLOWUP_DAYS) & (bb <= VD.FOLLOWUP_DAYS)
            diffs = (ba - bb)[valid].astype(float)
            valid_r = valid & (bb > 0)
            ratios = ba[valid_r].astype(float) / bb[valid_r].astype(float)
            ex_d, ex_r = int(REPS - valid.sum()), int(REPS - valid_r.sum())
            d_lo, d_hi = W.percentile_ci(diffs)
            d_lo_i, d_hi_i = int(math.floor(d_lo)), int(math.ceil(d_hi))
            r_lo, r_hi = W.percentile_ci(ratios)
            rev = float(np.mean(diffs <= 0))
            incl0 = bool(d_lo <= 0 <= d_hi)
            incl0_printed = bool(d_lo_i <= 0 <= d_hi_i)
            lo_a, hi_a = W.percentile_ci(ba.astype(float))
            lo_b, hi_b = W.percentile_ci(bb.astype(float))
            la, ha = math.floor(lo_a), math.ceil(hi_a)
            lb, hb = math.floor(lo_b), math.ceil(hi_b)
            overlap = bool(la <= hb and lb <= ha)
            if incl0:
                zero_pairs[t].append(pid)
            q = pq + threshold_key(t) + "."
            put(res, q + "first_day", int(da))
            put(res, q + "second_day", int(db))
            put(res, q + "diff_days", int(da - db))
            put(res, q + "diff_ci_lo_days", d_lo_i)
            put(res, q + "diff_ci_hi_days", d_hi_i)
            put(res, q + "diff_ci_lo_exact_days", float(d_lo))
            put(res, q + "diff_ci_hi_exact_days", float(d_hi))
            put(res, q + "diff_median_days", float(np.median(diffs)))
            put(res, q + "diff_includes_zero", incl0)
            put(res, q + "diff_includes_zero_printed", incl0_printed)
            put(res, q + "reps_excluded_difference_n", ex_d)
            put(res, q + "reps_excluded_ratio_n", ex_r)
            put(res, q + "reps_included_difference_n", int(valid.sum()))
            put(res, q + "reps_included_ratio_n", int(valid_r.sum()))
            put(res, q + "reversed_or_tied_prop", rev)
            put(res, q + "table5_intervals_overlap", overlap)
            put(res, q + "ratio_defined", bool(db > 0))
            if db > 0:
                put(res, q + "ratio", da / db)
                put(res, q + "ratio_ci_lo", float(r_lo))
                put(res, q + "ratio_ci_hi", float(r_hi))
                ratio_pt = f_text(da / db)
                ratio_ci = f"{f_text(r_lo)} to {f_text(r_hi)}"
            else:
                # the point ratio is undefined (second day is 0); an interval from the replicates
                # that happen to have a positive second day would be conditional and biased, so it is withheld
                ratio_pt = "Undefined (second day is 0)"
                ratio_ci = "Not estimable"
            rows_b.append([
                a.stratum.stratifier, comparison,
                f"Day residual risk falls below {R1.threshold_text(t)}",
                VD.fmt_count(a.n), VD.fmt_count(b.n),
                str(da), str(db), signed(da - db), f"{d_lo_i} to {d_hi_i}",
                ratio_pt, ratio_ci, str(ex_d), str(ex_r),
                f"{100.0 * rev:.1f}", yes_no(incl0), yes_no(overlap), str(REPS)])
        # baseline 30-day cumulative incidence (percentage points)
        pa, pb = 100.0 * a.cif30, 100.0 * b.cif30
        ca, cb = 100.0 * a.boot.cif[:, -1], 100.0 * b.boot.cif[:, -1]
        pd_all = ca - cb
        c_lo, c_hi = W.percentile_ci(pd_all)
        cr = ca / cb
        cr_lo, cr_hi = W.percentile_ci(cr)
        rev = float(np.mean(pd_all <= 0))
        q = pq + "cif30."
        put(res, q + "first_pct", pa)
        put(res, q + "second_pct", pb)
        put(res, q + "diff_pp", pa - pb)
        put(res, q + "diff_ci_lo_pp", float(c_lo))
        put(res, q + "diff_ci_hi_pp", float(c_hi))
        put(res, q + "ratio", pa / pb)
        put(res, q + "ratio_ci_lo", float(cr_lo))
        put(res, q + "ratio_ci_hi", float(cr_hi))
        put(res, q + "reversed_or_tied_prop", rev)
        put(res, q + "diff_includes_zero", bool(c_lo <= 0 <= c_hi))
        rows_b.append([
            a.stratum.stratifier, comparison, "30-day cumulative incidence, % (difference in percentage points)",
            VD.fmt_count(a.n), VD.fmt_count(b.n),
            f"{pa:.3f}", f"{pb:.3f}", f"{pa - pb:.3f}", f"{c_lo:.3f}{EN}{c_hi:.3f}",
            f_text(pa / pb), f"{f_text(cr_lo)} to {f_text(cr_hi)}", "0", "0",
            f"{100.0 * rev:.1f}", yes_no(c_lo <= 0 <= c_hi), NOT_APPLICABLE, str(REPS)])
    for t in (THR_05, THR_025):
        adjacent = [pid for pid in zero_pairs[t] if pid != HEADLINE_PAIR]
        put(res, f"a6.adjacent_including_zero.{threshold_key(t)}.n", len(adjacent))
        put(res, f"a6.adjacent_including_zero.{threshold_key(t)}.pairs", ", ".join(adjacent))
        put(res, f"a6.headline_includes_zero.{threshold_key(t)}", HEADLINE_PAIR in zero_pairs[t])
    put(res, "a6.headline_pair_slug", HEADLINE_PAIR)
    put(res, "a6.pairs_n", len(PAIRS))
    put(res, "a6.replicates_n", REPS)

    # ---- Table S5A -------------------------------------------------------------------------------
    header_a = [
        "Stratifier", "Stratum", "Patients", "Composite events", "denominator",
        "Baseline risk: 30-day cumulative incidence, % (95% CI)",
        "Observed 0.5% threshold day (95% CI)", "Remains below 0.5% through day 30",
        "Observed 0.25% threshold day (95% CI)", "Remains below 0.25% through day 30",
        "Day residual risk falls below 50% of the stratum's own baseline (95% CI)",
        "Remains below 50% of baseline through day 30",
        "Day residual risk falls below 25% of the stratum's own baseline (95% CI)",
        "Remains below 25% of baseline through day 30",
        "Day residual risk falls below 10% of the stratum's own baseline (95% CI)",
        "Remains below 10% of baseline through day 30",
        "Baseline-only predicted 0.5% day", "Observed minus predicted 0.5% day (days)",
        "Baseline-only predicted 0.25% day", "Observed minus predicted 0.25% day (days)",
        "Observed minus predicted 0.25% day, 95% CI (days)",
    ]
    rows_a = []
    for f in fits:
        s = f.stratum
        head = [s.stratifier, s.level, VD.fmt_count(f.n), VD.fmt_count(f.events), VD.fmt_count(f.n)]
        if f.suppressed:
            rows_a.append(head + [SUPPRESSED] * (len(header_a) - len(head)))
            continue
        cells = [f"{100.0 * f.cif30:.3f} ({100.0 * f.boot.cif30_lo:.3f}{EN}{100.0 * f.boot.cif30_hi:.3f})"]
        for t in (THR_05, THR_025):
            day = obs_day[f.slug][t]
            cells.append(day_cell(day, obs_b[f.slug][t]))
            cells.append(yes_no(R1.stays_below_from(f.curve, t, day)) if day <= VD.FOLLOWUP_DAYS else NOT_APPLICABLE)
        for rt in REL_THR:
            day = rel_day[f.slug][rt]
            cells.append(day_cell(day, rel_b[f.slug][rt]))
            cells.append(yes_no(R1.stays_below_from(rho[f.slug], rt, day)) if day <= VD.FOLLOWUP_DAYS else NOT_APPLICABLE)
        for t in (THR_05, THR_025):
            pday, day = prd_day[f.slug][t], obs_day[f.slug][t]
            cells.append(str(pday) if pday <= VD.FOLLOWUP_DAYS else NOT_REACHED)
            cells.append(signed(day - pday) if (pday <= VD.FOLLOWUP_DAYS and day <= VD.FOLLOWUP_DAYS) else NOT_APPLICABLE)
        if f.slug == "overall":
            cells.append("Reference")
        else:
            both = ((obs_b[f.slug][THR_025] <= VD.FOLLOWUP_DAYS) & (prd_b[f.slug][THR_025] <= VD.FOLLOWUP_DAYS))
            dd = (obs_b[f.slug][THR_025] - prd_b[f.slug][THR_025])[both].astype(float)
            dlo, dhi = W.percentile_ci(dd)
            cells.append(f"{int(math.floor(dlo))} to {int(math.ceil(dhi))}")
        rows_a.append(head + cells)
    R1.write_csv_table(R1.TABLES_OUT / "tableS5_baseline_vs_residual.csv", header_a, rows_a)

    header_b = [
        "Stratifier", "Comparison (first stratum minus second stratum)", "Measure",
        "Patients, first stratum", "Patients, second stratum",
        "First stratum, observed value", "Second stratum, observed value",
        "Difference (first minus second)", "Difference, 95% bootstrap CI",
        "Ratio (first divided by second)", "Ratio, 95% bootstrap CI",
        "Replicates excluded from the difference (n)", "Replicates excluded from the ratio (n)",
        "Included replicates with the difference zero or negative (%)",
        "Difference interval includes zero", "Single-stratum intervals of Table 5 overlap",
        "Bootstrap replicates (n)",
    ]
    R1.write_csv_table(R1.TABLES_OUT / "tableS5_spread_intervals.csv", header_b, rows_b)

    # ---- Figure S1 -------------------------------------------------------------------------------
    if R1.FIGURES:
        make_figure(fits, idx, rho, rho_lo, rho_hi)

    # ---- fragment --------------------------------------------------------------------------------
    n_keys = R1.write_fragment(R1.FRAGMENTS_OUT / "results_a1.json", res, ("a1.", "a6."))

    # ---- console digest ------------------------------------------------------------------------
    print(f"[a1] wrote {n_keys} keys")
    for t in (THR_025, THR_05):
        print(f"[a1] observed and baseline-only predicted {R1.threshold_text(t)} day:")
        for f in fits:
            if f.suppressed:
                print(f"       {f.stratum.stratifier} / {f.stratum.level}: suppressed ({f.n} patients, {f.events} events)")
                continue
            print(f"       {f.stratum.stratifier} / {f.stratum.level}: observed {obs_day[f.slug][t]} "
                  f"(95% CI {W.day_ci_text(*W.percentile_ci(obs_b[f.slug][t].astype(float)))}), "
                  f"predicted {prd_day[f.slug][t]}")
    d = decision
    print(f"[a1] HEADLINE f at 0.25% = {d['p_hi'] - d['p_lo']}/{d['d_hi'] - d['d_lo']} = {float(d['f']):.4f} "
          f"(95% CI {d['lo_ci']:.3f} to {d['hi_ci']:.3f}, median {d['med']:.3f}, {d['used']} of {REPS} replicates; "
          f"excluded unreached {d['n_un']}, denominator {d['n_bad']}); pair {d['hi'].slug} vs {d['lo'].slug}; "
          f"ties hi {d['ties_hi']} lo {d['ties_lo']}; category {category}")
    if joint_headline:
        print(f"[a1] sensitivity joint bootstrap f at 0.25%: {joint_headline[1]:.3f} median, "
              f"95% interval {joint_headline[0]:.3f} to {joint_headline[2]:.3f}, {joint_headline[3]} replicates")
    for (q, t, hi, lo, fr) in f_specs:
        print(f"[a1] {q[:-1]}: f = {res[q + 'f_prop']:.3f} ({res[q + 'pred_hi_day']}-{res[q + 'pred_lo_day']})/"
              f"({res[q + 'obs_hi_day']}-{res[q + 'obs_lo_day']}), interval {res[q + 'f_lo_prop']:.3f} to "
              f"{res[q + 'f_hi_prop']:.3f}, {res[q + 'hi_stratum_slug']} vs {res[q + 'lo_stratum_slug']}, "
              f"ties hi {res[q + 'tie_hi_slugs']} lo {res[q + 'tie_lo_slugs']}")
    for pid, sa, sb in PAIRS:
        for t in (THR_05, THR_025):
            q = f"a6.{pid}.{threshold_key(t)}."
            print(f"[a6] {pid} {R1.threshold_text(t)}: {res[q + 'first_day']} - {res[q + 'second_day']} = "
                  f"{res[q + 'diff_days']} (95% CI {res[q + 'diff_ci_lo_days']} to {res[q + 'diff_ci_hi_days']}; "
                  f"exact {res[q + 'diff_ci_lo_exact_days']:.2f} to {res[q + 'diff_ci_hi_exact_days']:.2f}), "
                  f"includes 0: {res[q + 'diff_includes_zero']} (printed {res[q + 'diff_includes_zero_printed']}); "
                  f"ratio {res.get(q + 'ratio', 'undefined')} ({res.get(q + 'ratio_ci_lo', float('nan')):.2f} to {res.get(q + 'ratio_ci_hi', float('nan')):.2f}); "
                  f"excluded diff {res[q + 'reps_excluded_difference_n']}, ratio {res[q + 'reps_excluded_ratio_n']}; "
                  f"reversed or tied {100 * res[q + 'reversed_or_tied_prop']:.1f}%; Table 5 overlap {res[q + 'table5_intervals_overlap']}")
    for t in (THR_05, THR_025):
        k = threshold_key(t)
        print(f"[a6] adjacent pairs whose difference interval includes 0 at {R1.threshold_text(t)}: "
              f"{res[f'a6.adjacent_including_zero.{k}.pairs'] or 'none'}; headline includes 0: "
              f"{res[f'a6.headline_includes_zero.{k}']}")
    return 0


# ---------------------------------------------------------------------------------------------
# Figure S1
# ---------------------------------------------------------------------------------------------

def make_figure(fits, idx, rho, rho_lo, rho_hi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    R1.style_figure(base_pt=8.0)
    days = np.arange(W.DAYS)
    stratifiers = []
    for f in fits:
        if f.slug != "overall" and f.stratum.stratifier not in stratifiers:
            stratifiers.append(f.stratum.stratifier)
    if len(stratifiers) != 4:
        raise R1.R1Error("expected four stratifiers")

    fig, axes = plt.subplots(2, 2, figsize=R1.FIG_SIZE_FOUR_PANEL, sharey=True)
    fig.subplots_adjust(**R1.FOUR_PANEL_ADJUST)
    ov = idx["overall"]
    for k, (ax, name) in enumerate(zip(axes.ravel(), stratifiers)):
        levels = [f for f in fits if f.stratum.stratifier == name]
        for t in REL_THR:
            ax.axhline(t, color=R1.INK_MUTED, lw=0.7, linestyle=R1.THRESHOLD_LINESTYLE, zorder=2)
            ax.text(W.DAYS - 1, t + 0.012, REL_TEXT[t], fontsize=7.0, color=R1.INK_MUTED, va="bottom", ha="right")
        ax.step(days, rho["overall"], where="post", color=R1.INK_MUTED, lw=1.0,
                linestyle=R1.OVERALL_LINESTYLE, zorder=3)
        handles, notes = [], []
        for i, f in enumerate(levels):
            if f.suppressed:
                notes.append(f"{f.stratum.level} suppressed: {VD.fmt_count(f.n)} patients, "
                             f"{VD.fmt_count(f.events)} primary composite events (minimum {VD.MIN_STRATUM_EVENTS}).")
                continue
            label = f"{f.stratum.level} (n = {VD.fmt_count(f.n)}, {VD.fmt_count(f.events)} events)"
            handles.append(R1.plot_step_curve(ax, days, rho[f.slug], rho_lo[f.slug], rho_hi[f.slug],
                                              level_index=i, label=label))
        ax.set_ylim(0, 1.30)
        ax.set_xlim(-0.4, W.DAYS - 1 + 0.4)
        ax.set_xticks(list(range(0, W.DAYS, 5)))
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{100 * v:.0f}"))
        ax.yaxis.grid(True, linestyle="-", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(labelbottom=True, labelleft=True)
        ax.text(-0.135, 1.02, "ABCD"[k], transform=ax.transAxes, fontsize=10.5, fontweight="bold",
                color=R1.INK, va="bottom", ha="left")
        ax.text(0.035, 0.985, name, transform=ax.transAxes, fontsize=8.0, fontweight="bold",
                color=R1.INK, va="top", ha="left")
        legend = ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 0.915), fontsize=7.0,
                           handlelength=2.0, labelspacing=0.40, borderaxespad=0.35)
        for text in legend.get_texts():
            text.set_color(R1.INK)
        if notes:
            ax.text(0.40, 0.915 - 0.085 * len(handles) - 0.02, "\n".join(_wrap(" ".join(notes), 30)),
                    transform=ax.transAxes, fontsize=6.6, color=R1.INK_MUTED, va="top", ha="left",
                    linespacing=1.4, style="italic")
        if k % 2 == 0:
            ax.set_ylabel("Residual risk, % of the stratum's own\n30-day cumulative incidence", linespacing=1.35)
        if k >= 2:
            ax.set_xlabel("Postoperative day (patients event free and alive\nat the start of that day)", linespacing=1.35)
    top_legend = fig.legend(handles=[
        Line2D([0], [0], color=R1.INK_MUTED, lw=1.0, linestyle=R1.OVERALL_LINESTYLE,
               label=f"All patients (n = {VD.fmt_count(ov.n)}, {VD.fmt_count(ov.events)} events)"),
        Line2D([0], [0], color=R1.INK_MUTED, lw=0.7, linestyle=R1.THRESHOLD_LINESTYLE,
               label="Thresholds at 50%, 25% and 10% of each stratum's baseline"),
    ], loc="lower left", bbox_to_anchor=(0.084, 0.945), bbox_transform=fig.transFigure, ncol=2,
        fontsize=7.0, handlelength=2.0, columnspacing=1.6, borderaxespad=0.0)
    for text in top_legend.get_texts():
        text.set_color(R1.INK)
    R1.save_png(fig, R1.FIGURES_OUT / "figS1_relative_residual_risk.png", also_pdf=True)


def _wrap(text, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
